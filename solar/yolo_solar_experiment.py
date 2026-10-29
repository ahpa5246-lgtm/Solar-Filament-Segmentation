"""COCO-pretrained anchor-free instance segmentation with PQ checkpoint selection.

Uses the unmodified Ultralytics 8.3.221 package (AGPL-3.0), not copied source.
Competition labels/images stay inside the private Kaggle execution.
"""
import csv,hashlib,json,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from pycocotools import mask as coco
from .yolo_solar_data import build_dataset
from .disjoint_delivery import make_disjoint
from .v6_resolution_audit import select_pilot_dates,refine
from .v6_refiner_audit import load_components,postprocess,references,evaluate,SOURCE_SHA,REFINER_SHA
from .v6_cached_search import micro,paired_ci
from .v5_replay import load_predictions,aggregate
from .v6_submission import resolve_test_directory,write_rle_submission


@torch.inference_mode()
def proposals(detector,image):
    result=detector.predict(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),imgsz=1024,conf=.1,iou=.9,
                            max_det=100,device=0,retina_masks=True,verbose=False)[0]
    rles=[];scores=[]
    if result.masks is None:return rles,scores
    masks=result.masks.data.cpu().numpy();confidence=result.boxes.conf.cpu().numpy()
    for mask,score in zip(masks,confidence):
        if mask.shape!=(2048,2048):mask=cv2.resize(mask,(2048,2048),interpolation=cv2.INTER_NEAREST)
        mask=(mask>.5).astype(np.uint8)
        if mask.any():rles.append(coco.encode(np.asfortranarray(mask)));scores.append(float(score))
    return rles,scores


def finish_masks(raw,confidence,image,refiner,helpers,threshold,mode):
    keep=[i for i,score in enumerate(confidence) if score>=threshold]
    rles=[raw[i] for i in keep];scores=[confidence[i] for i in keep]
    rles=postprocess(rles,scores,'nms')
    return make_disjoint(refine(refiner,helpers,image,rles,overlap_policy='nms') if mode=='refined' else rles)


def run(annotation,manifest,plain_cache,best_refiner,final_refiner,source,out):
    from ultralytics import YOLO,settings,__version__
    for key in ('hub','wandb','comet','mlflow','clearml','neptune','dvc','dvclive','raytune'):
        if key in settings:settings.update({key:False})
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    report={'status':'STARTED','training':True,'test_inference':False,'official_submission':False,
            'architecture':'YOLO11m-seg COCO transfer; optional preserved CropUNet',
            'ultralytics_version':__version__,'dependency_license':'AGPL-3.0',
            'overlap_mask':False,'checkpoint_selection':'complete-pipeline tuning reference micro PQ, never mAP',
            'epochs_requested':40,'training_time_budget_hours':.6,
            'limitations':['Historical validation; official Public PQ is decisive.','Negative crops exclude all known annotator masks but cannot guarantee completeness of annotations.']}
    def save():
        p=out/'yolo_result.pending';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(out/'yolo_result.json')
    save()
    try:
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=SOURCE_SHA or hashlib.sha256(Path(best_refiner).read_bytes()).hexdigest()!=REFINER_SHA:
            raise ValueError('Preserved source/refiner mismatch')
        if hashlib.sha256(Path(final_refiner).read_bytes()).hexdigest()!='a08df74538b5752c0df1047fea81b1b1aea4d81d3f749b340c993b4d5c8da924':
            raise ValueError('Final full-training refiner mismatch')
        rows=list(csv.DictReader(Path(manifest).open()));fitting=[r['observation_id'] for r in rows if r['split']=='train']
        validation={r['observation_id']:r for r in rows if r['split']=='val'}
        tuning,confirmation=select_pilot_dates(fitting,validation,None)
        from .v5_finetune import validate_partitions
        validate_partitions(fitting,tuning,confirmation)
        truth=references(annotation,set(tuning+confirmation));cached=load_predictions(Path(plain_cache))
        cached_tuning=evaluate(truth,cached,tuning);cached_confirmation=evaluate(truth,cached,confirmation)
        select_yaml,final_yaml,data_report=build_dataset(annotation,rows,tuning,out/'data')
        report.update(dataset=data_report,tuning_observations=len(tuning),confirmation_observations=len(confirmation),
                      cached_plain_baseline_tuning=micro(cached_tuning),cached_plain_baseline_confirmation=aggregate(cached_confirmation),
                      known_flip4_baseline_tuning=.40503051142227725,known_flip4_baseline_confirmation=.40736969428848335)
        helpers=load_components(source);refiner=helpers['CropUNet'](pretrained=False)
        refiner.load_state_dict(torch.load(best_refiner,map_location='cpu',weights_only=True)['model'],strict=True);refiner=refiner.cuda().eval()
        for p in refiner.parameters():p.requires_grad_(False)
        torch.set_num_threads(2);cv2.setNumThreads(0)
        best=-1.;selected_path=None;history=[]
        def image_for(stem):
            im=cv2.imread(str(Path(annotation).parent/'train_images'/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape!=(2048,2048):raise ValueError('Bad native validation image')
            return im
        def assess_checkpoint(trainer):
            nonlocal best,selected_path
            epoch=trainer.epoch+1
            if epoch%5 and not trainer.stop and epoch!=trainer.epochs:return
            detector=YOLO(str(trainer.last));predictions={}
            for stem in tuning:
                image=image_for(stem);rles,scores=proposals(detector,image)
                predictions[stem]=finish_masks(rles,scores,image,refiner,helpers,.35,'refined')
            score=micro(evaluate(truth,predictions,tuning));row={'epoch':epoch,'tuning_pipeline_pq':score}
            history.append(row)
            if score>best:
                best=score;selected_path=out/'candidates'/f'epoch-{epoch:03d}.pt';selected_path.parent.mkdir(exist_ok=True)
                shutil.copyfile(trainer.last,selected_path)
                report.update(selected_epoch=epoch,selected_tuning_pq=best,selected_checkpoint_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest())
            report['history']=history;save();print('YOLO_PIPELINE_PQ_EPOCH',json.dumps(row),flush=True)
            del detector,predictions;torch.cuda.empty_cache()
        detector=YOLO('yolo11m-seg.pt')
        pretrained=Path(detector.ckpt_path)
        if pretrained.exists():report['pretrained_sha256']=hashlib.sha256(pretrained.read_bytes()).hexdigest()
        detector.add_callback('on_fit_epoch_end',assess_checkpoint)
        args=dict(imgsz=1024,batch=4,nbs=16,device=0,workers=2,optimizer='SGD',lr0=.003,lrf=.05,
                  momentum=.937,weight_decay=.0005,cos_lr=True,warmup_epochs=1.,freeze=5,
                  overlap_mask=False,mask_ratio=1,mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,
                  flipud=.5,fliplr=.5,hsv_h=0.,hsv_s=0.,hsv_v=.1,amp=True,seed=2032,deterministic=True,
                  patience=1000,plots=False,save=True,val=False,cache=False,verbose=False)
        report.update(status='TRAINING',training_configuration=args);save()
        detector.train(data=str(select_yaml),epochs=40,time=.6,project=str(out/'training'),name='selection',exist_ok=False,**args)
        report['selection_epochs_completed']=int(detector.trainer.epoch+1)
        if selected_path is None:raise RuntimeError('No PQ-selected checkpoint produced')
        del detector;torch.cuda.empty_cache();detector=YOLO(str(selected_path))
        # Freeze checkpoint first, tune confidence/refinement on tuning only.
        tuning_predictions={f'{mode}|{threshold}':{} for mode in ('raw','refined') for threshold in (.2,.35,.5)}
        for stem in tuning:
            image=image_for(stem);rles,scores=proposals(detector,image)
            for mode in ('raw','refined'):
                for threshold in (.2,.35,.5):
                    tuning_predictions[f'{mode}|{threshold}'][stem]=finish_masks(rles,scores,image,refiner,helpers,threshold,mode)
        trials={key:micro(evaluate(truth,pred,tuning)) for key,pred in tuning_predictions.items()}
        selected=max(trials,key=trials.get);mode,value=selected.split('|');threshold=float(value)
        frozen_predictions={}
        for stem in confirmation:
            image=image_for(stem);rles,scores=proposals(detector,image)
            frozen_predictions[stem]=finish_masks(rles,scores,image,refiner,helpers,threshold,mode)
        candidate=evaluate(truth,frozen_predictions,confirmation)
        allowed=trials[selected]>max(micro(cached_tuning),.40503051142227725) and micro(candidate)>max(micro(cached_confirmation),.40736969428848335)
        report.update(status='VALIDATION_COMPLETE',selected_pipeline=selected,tuning_grid=trials,
                      confirmation_candidate=aggregate(candidate),paired_delta_against_cached_plain=paired_ci(cached_confirmation,candidate),
                      official_assessment_allowed=allowed,selection_elapsed_seconds=time.perf_counter()-started)
        save()
        if not allowed:
            report.update(status='LOCAL_REJECTED',next_action='Retain official best; diagnose new-architecture error profiles before another fit');save();return report
        del tuning_predictions,frozen_predictions;torch.cuda.empty_cache()
        # One declared all-training fit; selection is already frozen. Ignore mAP best.pt.
        detector=YOLO(str(selected_path));final_args=dict(args);final_args.update(lr0=.0003,warmup_epochs=0.)
        detector.train(data=str(final_yaml),epochs=1,time=None,project=str(out/'training'),name='final',exist_ok=False,**final_args)
        final_path=out/'final_yolo.pt';shutil.copyfile(detector.trainer.last,final_path)
        del detector;torch.cuda.empty_cache();detector=YOLO(str(final_path))
        refiner.load_state_dict(torch.load(final_refiner,map_location='cuda',weights_only=True)['model'],strict=True);refiner.eval()
        test_dir=resolve_test_directory(annotation);files=sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
        if len(files)!=180 or len({p.stem for p in files})!=180:raise ValueError('Expected complete 180-image test inventory')
        predictions={}
        for index,path in enumerate(files,1):
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(2048,2048):raise ValueError('Bad native test image')
            rles,scores=proposals(detector,image);predictions[path.stem]=finish_masks(rles,scores,image,refiner,helpers,threshold,mode)
            if index%30==0:print('YOLO_TEST_IMAGES',index,'/180',flush=True)
        inventory=write_rle_submission(predictions,[p.stem for p in files],out/'submission.csv')
        report.update(status='COMPLETE',test_inference=True,final_checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
                      final_refiner_sha256=hashlib.sha256(Path(final_refiner).read_bytes()).hexdigest(),
                      submission_sha256=inventory['sha256'],inventory=inventory,elapsed_seconds=time.perf_counter()-started)
        save();print('YOLO_FINAL_CANDIDATE_READY',json.dumps(report),flush=True);return report
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,diagnosis=str(error)[:700],elapsed_seconds=time.perf_counter()-started);save();raise
