"""Train a box-conditioned refiner with guarded fitting-only background crops."""
import csv,hashlib,json,math,time
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from pycocotools import mask as coco
from .seed_refiner import feature_crop,predict_maps,restore_masks
from .v6_refiner_audit import load_components,references,evaluate,postprocess,REFINER_SHA,SOURCE_SHA
from .v6_resolution_audit import select_pilot_dates,original_detector,detect
from .v6_cached_search import micro,paired_ci
from .v5_replay import load_predictions,aggregate
from .v6_submission import resolve_test_directory,write_rle_submission


def prepare(annotation,rows,stems,helpers,negative_paths,out,extra_negatives=None):
    extra_negatives=extra_negatives or []
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data=json.loads(Path(annotation).read_text())
    images={str(r['id']):r for r in data['images']};stems=set(stems)
    annotations=[a for a in data['annotations'] if Path(images[str(a['image_id'])]['file_name']).stem in stems]
    paths={r['observation_id']:Path(annotation).parent/'train_images'/r['file_name'] for r in rows}
    capacity=len(annotations)+len(negative_paths)+len(extra_negatives)
    x=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.uint8,shape=(capacity,3,256,256))
    y=np.lib.format.open_memmap(out/'targets.npy',mode='w+',dtype=np.uint8,shape=(capacity,1,256,256))
    @lru_cache(maxsize=8)
    def image_for(stem):
        im=cv2.imread(str(paths[stem]),cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape!=(2048,2048):raise ValueError('Bad native fitting image')
        return im
    count=0
    for a in annotations:
        stem=Path(images[str(a['image_id'])]['file_name']).stem
        rle=coco.merge(coco.frPyObjects(a['segmentation'],2048,2048))
        if coco.area(rle)==0:continue
        feature,bounds=feature_crop(image_for(stem),rle,helpers);x0,y0,x1,y1=bounds
        x[count]=feature;y[count,0]=cv2.resize(coco.decode(rle)[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_NEAREST);count+=1
    positives=count;rng=np.random.default_rng(2033)
    for path in negative_paths:
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('Missing guarded negative crop')
        gray=cv2.resize(gray,(256,256),interpolation=cv2.INTER_LINEAR)
        seed=x[int(rng.integers(positives)),2].copy()
        x[count]=np.stack((gray,cv2.createCLAHE(2.,(8,8)).apply(gray),seed));y[count]=0;count+=1
    for path in extra_negatives:
        feature=np.load(path,allow_pickle=False)
        if feature.shape!=(3,256,256) or feature.dtype!=np.uint8:raise ValueError("Bad mined negative feature")
        x[count]=feature;y[count]=0;count+=1
    x.flush();y.flush();image_for.cache_clear()
    info={'positive_crops':positives,'negative_crops':count-positives,'stored_crops':count,'physical_observations':len(stems)}
    (out/'summary.json').write_text(json.dumps(info,indent=2));return out,info


class Crops(Dataset):
    def __init__(self,folder,info):
        self.x=np.load(Path(folder)/'features.npy',mmap_mode='r');self.y=np.load(Path(folder)/'targets.npy',mmap_mode='r')
        n=info['positive_crops'];self.indices=list(range(n))
        if info['negative_crops']:
            self.indices+=np.random.default_rng(2033).choice(np.arange(n,info['stored_crops']),max(1,n//4),replace=True).tolist()
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index];x=self.x[i].copy();y=self.y[i].copy()
        yy,xx=np.nonzero(x[2])
        if len(xx):
            cx=(xx.min()+xx.max())/2;cy=(yy.min()+yy.max())/2;sx,sy=np.random.uniform(.9,1.1,2)
            dx=np.random.uniform(-.1,.1)*(xx.max()-xx.min()+1);dy=np.random.uniform(-.1,.1)*(yy.max()-yy.min()+1)
            matrix=np.array([[sx,0,cx*(1-sx)+dx],[0,sy,cy*(1-sy)+dy]],np.float32)
            x[2]=cv2.warpAffine(x[2],matrix,(256,256),flags=cv2.INTER_NEAREST,borderValue=0)
        k=np.random.randint(4);x=np.rot90(x,k,axes=(1,2));y=np.rot90(y,k,axes=(1,2))
        if np.random.rand()<.5:x=np.flip(x,2);y=np.flip(y,2)
        if np.random.rand()<.5:x=np.flip(x,1);y=np.flip(y,1)
        return torch.from_numpy(np.ascontiguousarray(x)).float()/255,torch.from_numpy(np.ascontiguousarray(y)).float()


def run(annotation,manifest,detector_cache,plain_cache,best_refiner,final_detector,source,yolo_selected,yolo_final,yolo_report,negative_paths,out,initial_refiner_sha=REFINER_SHA,epochs=8,learning_rate=1e-4,hard_proposals=False):
    from ultralytics import YOLO,settings
    for key in ('hub','wandb','comet','mlflow','clearml','neptune','dvc','dvclive','raytune'):
        if key in settings:settings.update({key:False})
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    r={'status':'STARTED','training':True,'test_inference':False,'official_submission':False,'input_contract':'gray,CLAHE,proposal-box seed',
       'empty_mask_policy':'reject; no original-mask fallback','selection':'full-pipeline tuning reference micro PQ',
       'limitations':['Historical validation reuse; official Public PQ remains decisive.','Fitting-only negative crops exclude all known annotations; incomplete annotations remain possible.']}
    def save():
        p=out/'seed_result.pending';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(out/'seed_result.json')
    save()
    try:
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=SOURCE_SHA or hashlib.sha256(Path(best_refiner).read_bytes()).hexdigest()!=initial_refiner_sha:raise ValueError('Original source/refiner mismatch')
        prior=json.loads(Path(yolo_report).read_text())
        for path,key in ((yolo_selected,'selected_checkpoint_sha256'),(yolo_final,'final_checkpoint_sha256')):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=prior[key]:raise ValueError('YOLO checkpoint lineage mismatch')
        if hashlib.sha256(Path(final_detector).read_bytes()).hexdigest()!='f5e1b6acd0bdaf17f048f9ec3af7999b73ef966b28f469c6cb56bf8977493229':raise ValueError('Original final detector mismatch')
        torch.manual_seed(2033);np.random.seed(2033);torch.set_num_threads(2);cv2.setNumThreads(0)
        rows=list(csv.DictReader(Path(manifest).open()));fitting=[v['observation_id'] for v in rows if v['split']=='train'];val={v['observation_id']:v for v in rows if v['split']=='val'}
        tuning,confirmation=select_pilot_dates(fitting,val,None)
        from .v5_finetune import validate_partitions,safe_scaled_step
        validate_partitions(fitting,tuning,confirmation)
        truth=references(annotation,set(tuning+confirmation));cached=load_predictions(Path(detector_cache));plain=load_predictions(Path(plain_cache))
        baseline=evaluate(truth,plain,confirmation)
        helpers=load_components(source)
        if hard_proposals:
            from .proposal_negatives import mine
            miner=YOLO(str(yolo_selected));extra_negatives=mine(annotation,rows,fitting,helpers,miner,out/'actual-false-proposals');del miner;torch.cuda.empty_cache()
        else:extra_negatives=[]
        folder,info=prepare(annotation,rows,fitting,helpers,negative_paths,out/'fitting-crops',extra_negatives)
        r.update(actual_false_proposal_negatives=len(extra_negatives),initial_refiner_sha256=initial_refiner_sha,epochs_requested=epochs,learning_rate=learning_rate)
        r.update(dataset=info,tuning_observations=len(tuning),confirmation_observations=len(confirmation));save()
        model=helpers['CropUNet'](pretrained=False);model.load_state_dict(torch.load(best_refiner,map_location='cpu',weights_only=True)['model'],strict=True);model=model.cuda()
        yolo=YOLO(str(yolo_selected))
        from .yolo_solar_experiment import proposals
        coarse={}
        def get_proposals(image,family):
            if family=='multi':
                from .scale_refiner_experiment import scaled_proposals
                parts=[scaled_proposals(yolo,image,scale) for scale in (1024,1536)]
                return [r for part in parts for r in part[0]],[v for part in parts for v in part[1]]
            return proposals(yolo,image)
        def image_for(stem):
            im=cv2.imread(str(Path(annotation).parent/'train_images'/val[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape!=(2048,2048):raise ValueError('Bad native validation image')
            return im
        configs=[('maskrcnn',1.,p,t) for p in ('union','nms') for t in (.5,.65,.85)]
        configs += [('yolo',c,p,t) for c in (.2,.35,.5) for p in ('union','nms') for t in (.5,.65,.85)]
        if hard_proposals:configs=[(family,.35,policy,t) for family in ('yolo','multi') for policy in ('union','nms') for t in (.5,.65,.85)]
        def key(config):return '|'.join(str(x) for x in config)
        def variants(stem,configs):
            image=image_for(stem);result={}
            for family in {c[0] for c in configs}:
                if family=='maskrcnn':rles=cached[stem];scores=[1.]*len(rles)
                else:
                    if (stem,family) not in coarse:coarse[(stem,family)]=get_proposals(image,family)
                    rles,scores=coarse[(stem,family)]
                    score_map={id(x):s for x,s in zip(rles,scores)};rles=postprocess(rles,scores,'nms');scores=[score_map[id(x)] for x in rles]
                bounds,probs=predict_maps(model,helpers,image,rles)
                for config in [c for c in configs if c[0]==family]:
                    _,confidence,policy,threshold=config;keep=[i for i,s in enumerate(scores) if s>=confidence]
                    result[key(config)]=restore_masks([bounds[i] for i in keep],[probs[i] for i in keep],image.shape,threshold,policy)
            return result
        loader=DataLoader(Crops(folder,info),batch_size=32,shuffle=True,num_workers=2,pin_memory=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.);best=-1.;selected=None;selected_path=None;history=[]
        def fit_epoch(loader,optimizer):
            model.train();losses=[]
            for x,y in loader:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(x);bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
                    pos=y.sum((1,2,3))>0
                    if pos.any():
                        p=logits[pos].float().sigmoid();target=y[pos];dice=1-((2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)).mean()
                    else:dice=logits.sum()*0
                    loss=bce+dice
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite seeded-refiner loss')
                scaler.scale(loss).backward();updated=safe_scaled_step(scaler,optimizer,model.parameters())
                r['optimizer_updates']=r.get('optimizer_updates',0)+int(updated)
                r['overflow_skips']=r.get('overflow_skips',0)+int(not updated)
                losses.append(float(loss.detach()))
            return sum(losses)/len(losses)
        for epoch in range(1,epochs+1):
            loss=fit_epoch(loader,optimizer);scheduler.step();row={'epoch':epoch,'loss':loss}
            if epoch%2==0:
                model.eval();predictions={key(c):{} for c in configs}
                for stem in tuning:
                    for name,value in variants(stem,configs).items():predictions[name][stem]=value
                scores={name:micro(evaluate(truth,pred,tuning)) for name,pred in predictions.items()};winner=max(scores,key=scores.get)
                row['best_tuning_pq']=scores[winner]
                if scores[winner]>best:
                    best=scores[winner];selected=next(c for c in configs if key(c)==winner);selected_path=out/f'seed-epoch-{epoch:02d}.pt'
                    torch.save({'model':model.state_dict(),'epoch':epoch,'input_contract':r['input_contract']},selected_path)
                    r.update(selected_epoch=epoch,selected_tuning_pq=best,selected_config=list(selected),selected_checkpoint_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest())
                del predictions
            history.append(row);r['history']=history;save();print('SEED_REFINER_EPOCH',json.dumps(row),flush=True)
        model.load_state_dict(torch.load(selected_path,map_location='cuda',weights_only=True)['model'],strict=True);model.eval()
        frozen={stem:variants(stem,[selected])[key(selected)] for stem in confirmation};candidate=evaluate(truth,frozen,confirmation)
        allowed=best>max(prior['tuning_grid'].values()) and micro(candidate)>prior['confirmation_candidate']['reference_micro_pq']
        r.update(confirmation_candidate=aggregate(candidate),paired_delta_against_cached_plain=paired_ci(baseline,candidate),
                 prior_yolo_confirmation_pq=prior['confirmation_candidate']['reference_micro_pq'],official_assessment_allowed=allowed,validation_seconds=time.perf_counter()-started)
        save()
        # Official evaluation is required by the owner even when local selection does not improve.
        r['local_improves_previous_yolo']=allowed
        r['official_assessment_allowed']=True
        save()
        del loader;full_folder,full_info=prepare(annotation,rows,[v['observation_id'] for v in rows],helpers,negative_paths,out/'all-crops',extra_negatives)
        full_loader=DataLoader(Crops(full_folder,full_info),batch_size=32,shuffle=True,num_workers=2,pin_memory=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);r['full_fit_loss']=fit_epoch(full_loader,optimizer);model.eval()
        checkpoint=out/'final_seed_refiner.pt';torch.save({'model':model.state_dict(),'input_contract':r['input_contract'],'selected_config':selected},checkpoint)
        family,confidence,policy,threshold=selected
        if family!='maskrcnn':yolo=YOLO(str(yolo_final));detector=None
        else:
            del yolo;torch.cuda.empty_cache();detector=original_detector(source);detector.load_state_dict(torch.load(final_detector,map_location='cpu',weights_only=True)['model'],strict=True);detector=detector.cuda().eval()
        test_dir=resolve_test_directory(annotation);files=sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
        if len(files)!=180 or len({p.stem for p in files})!=180:raise ValueError('Incomplete test image inventory')
        predictions={}
        for index,path in enumerate(files,1):
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if family!='maskrcnn':
                rles,scores=get_proposals(image,family);pairs=[(x,s) for x,s in zip(rles,scores) if s>=confidence]
                rles=postprocess([x for x,s in pairs],[s for x,s in pairs],'nms')
            else:rles,scores=detect(detector,image,1024);rles=postprocess(rles,scores,'union')
            bounds,probs=predict_maps(model,helpers,image,rles);predictions[path.stem]=restore_masks(bounds,probs,image.shape,threshold,policy)
            if index%30==0:print('SEED_TEST_IMAGES',index,'/180',flush=True)
        inventory=write_rle_submission(predictions,[p.stem for p in files],out/'submission.csv')
        r.update(status='COMPLETE',test_inference=True,final_refiner_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),inventory=inventory,submission_sha256=inventory['sha256'],elapsed_seconds=time.perf_counter()-started)
        save();print('SEED_FINAL_CANDIDATE_READY',json.dumps(r),flush=True);return r
    except Exception as error:
        r.update(status='FAILED',error_type=type(error).__name__,diagnosis=str(error)[:700],elapsed_seconds=time.perf_counter()-started);save();raise
