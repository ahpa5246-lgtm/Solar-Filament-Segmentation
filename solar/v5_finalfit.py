"""One declared full-training epoch and validated test inference for a selected V5 continuation."""
import ast, csv, hashlib, json, random, time
from pathlib import Path


def overlap_promotion_supported(tuning_pq,baseline_pq,candidate_pq,delta):
    """A tuning gain alone cannot justify replacing the baseline policy."""
    return (tuning_pq['preserve']>tuning_pq['union'] and
            candidate_pq>baseline_pq and delta['low']>0)


def run(annotation,manifest,trained,refiner_checkpoint,source,out,expected_sha,seconds=6600,stable_finetune=False,final_refiner_checkpoint=None,final_refiner_expected_sha=None):
    import cv2, numpy as np, pandas as pd, torch
    from pycocotools.coco import COCO
    from .v5_finetune import safe_scaled_step
    from .v6_resolution_audit import original_detector,detect,refine
    from .v6_refiner_audit import load_components,SOURCE_SHA,REFINER_SHA
    from .v6_submission import write_rle_submission,resolve_test_directory
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(trained)!=expected_sha or sha(source)!=SOURCE_SHA or sha(refiner_checkpoint)!=REFINER_SHA:
        raise ValueError('Frozen input fingerprint mismatch')
    if final_refiner_checkpoint is not None:
        if not isinstance(final_refiner_expected_sha,str) or len(final_refiner_expected_sha)!=64 or sha(final_refiner_checkpoint)!=final_refiner_expected_sha:
            raise ValueError('Frozen final refiner fingerprint mismatch')
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    random.seed(2028);np.random.seed(2028);torch.manual_seed(2028);torch.cuda.manual_seed_all(2028)
    torch.backends.cudnn.benchmark=False;torch.set_num_threads(2);cv2.setNumThreads(0)
    rows=list(csv.DictReader(Path(manifest).open()))
    image_dir=Path(annotation).parent/'train_images'
    archived=json.loads(Path(source).read_text())
    code=next(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in archived['cells']
              if c['cell_type']=='code' and (''.join(c['source']) if isinstance(c['source'],list) else c['source']).startswith('%%writefile'))
    definitions=[n for n in ast.parse(code.split('\n',1)[1]).body
                 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'masks_to_boxes','FilamentDataset'}]
    if len(definitions)!=2:raise RuntimeError('Archived dataset definitions changed')
    namespace=dict(np=np,cv2=cv2,torch=torch,COCO=COCO,Dataset=torch.utils.data.Dataset,
                   ANN_PATH=Path(annotation),IMAGE_DIR=image_dir,IMAGE_SIZE=1024)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'archived_dataset','exec'),namespace)
    dataset=namespace['FilamentDataset'](pd.DataFrame(rows),augment=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,
        collate_fn=lambda batch:tuple(zip(*batch)),generator=torch.Generator().manual_seed(2028))
    model=original_detector(source)
    model.load_state_dict(torch.load(trained,map_location='cpu',weights_only=True)['model'],strict=True)
    model=model.cuda().eval()
    helpers=load_components(source);refiner=helpers['CropUNet'](pretrained=False)
    refiner.load_state_dict(torch.load(refiner_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    refiner=refiner.cuda().eval()
    from .v6_resolution_audit import select_pilot_dates
    from .v6_refiner_audit import postprocess,references,evaluate
    from .v6_cached_search import micro,paired_ci
    from .v5_replay import aggregate
    from .v5_finetune import validate_partitions
    fitting=[r['observation_id'] for r in rows if r['split']=='train']
    validation={r['observation_id']:r for r in rows if r['split']=='val'}
    tuning,confirmation=select_pilot_dates(fitting,validation,None)
    validate_partitions(fitting,tuning,confirmation)
    truth=references(annotation,set(tuning+confirmation))
    predictions_by_policy={policy:{} for policy in ('union','preserve')}
    for index,stem in enumerate(tuning+confirmation,1):
        if time.perf_counter()-start>1200:raise TimeoutError('Overlap comparison budget exhausted')
        image=cv2.imread(str(image_dir/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape!=(2048,2048):raise ValueError('Invalid validation image')
        masks,scores=detect(model,image,1024)
        for policy in predictions_by_policy:
            proposals=postprocess(masks,scores,'union' if policy=='union' else 'nms')
            predictions_by_policy[policy][stem]=refine(refiner,helpers,image,proposals,
                overlap_policy='union' if policy=='union' else 'none')
        if index%20==0:print('V5_COMPLETE_OVERLAP_COMPARISON',index,'/',len(tuning+confirmation),flush=True)
    tuning_metrics={policy:evaluate(truth,predictions,tuning) for policy,predictions in predictions_by_policy.items()}
    tuning_pq={policy:micro(metrics) for policy,metrics in tuning_metrics.items()}
    selected_policy='preserve' if tuning_pq['preserve']>tuning_pq['union'] else 'union'
    # Selection is frozen before confirmation metrics are calculated.
    confirmation_metrics={policy:evaluate(truth,predictions,confirmation) for policy,predictions in predictions_by_policy.items()}
    proposed_policy=selected_policy
    delta=paired_ci(confirmation_metrics['union'],confirmation_metrics['preserve'])
    supported=overlap_promotion_supported(tuning_pq,micro(confirmation_metrics['union']),
                                         micro(confirmation_metrics['preserve']),delta)
    selected_policy=proposed_policy if supported else 'union'
    comparison=dict(selected_policy=selected_policy,proposed_policy=proposed_policy,
        confirmation_gain_supported=supported,tuning_pq=tuning_pq,
        tuning_observations=len(tuning),confirmation_observations=len(confirmation),
        confirmation={policy:aggregate(metrics) for policy,metrics in confirmation_metrics.items()},
        paired_macro_delta_ci=delta,
        interpretation='Historical validation reuse; exploratory joint-policy comparison, not an official score')
    (out/'overlap_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    del predictions_by_policy,tuning_metrics,confirmation_metrics
    model.train()
    if stable_finetune:
        from .stable_detector_training import restore_training_contract
        training_contract=restore_training_contract(model)
    else:training_contract=None
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.);optimizer.zero_grad(set_to_none=True)
    updates=0;skipped=0;consecutive=0
    for step,(batch,targets,_) in enumerate(loader,1):
        if time.perf_counter()-start>seconds-1200:raise TimeoutError('Final fit incomplete; no submission generated')
        batch=[im.cuda() for im in batch];targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
        with torch.autocast('cuda',dtype=torch.float16):loss=sum(model(batch,targets).values())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite final-fit loss')
        divisor=min(4,len(loader)-((step-1)//4)*4)
        scaler.scale(loss/divisor).backward()
        if step%4==0 or step==len(loader):
            finite=safe_scaled_step(scaler,optimizer,model.parameters())
            updates+=int(finite);skipped+=int(not finite);consecutive=0 if finite else consecutive+1
            if consecutive>=20:raise RuntimeError('Persistent final-fit gradient overflow')
        if step%100==0:print('V5_FINAL_FIT_STEP',step,'/',len(loader),flush=True)
    if not updates:raise RuntimeError('Final fit performed no optimizer updates')
    checkpoint=out/'final_detector.pt'
    torch.save({'model':model.state_dict(),'epoch':1,'source_sha256':expected_sha},checkpoint)
    model.eval()
    if final_refiner_checkpoint is not None:
        refiner.load_state_dict(torch.load(final_refiner_checkpoint,map_location='cuda',weights_only=True)['model'],strict=True)
        refiner.eval()
    from .v6_refiner_audit import postprocess
    files=sorted(p for p in resolve_test_directory(annotation).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
    if len(files)!=180 or len({p.stem for p in files})!=180:raise ValueError('Expected 180 unique competition test images')
    predictions={}
    for i,path in enumerate(files,1):
        if time.perf_counter()-start>seconds:raise TimeoutError('Test inference incomplete; no submission generated')
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape!=(2048,2048):raise ValueError('Invalid test image')
        masks,scores=detect(model,image,1024)
        predictions[path.stem]=refine(refiner,helpers,image,
            postprocess(masks,scores,'union' if selected_policy=='union' else 'nms'),
            overlap_policy='union' if selected_policy=='union' else 'none')
        if i%20==0:print('V5_TEST_IMAGES',i,'/',len(files),flush=True)
    result=write_rle_submission(predictions,[p.stem for p in files],out/'submission.csv')
    report=dict(status='COMPLETE',training=True,test_inference=True,official_submission=False,
        full_fit_epochs=1,full_fit_observations=len(dataset),learning_rate=1e-6,
        optimizer_updates=updates,overflow_skipped_updates=skipped,
        continuation_checkpoint_sha256=expected_sha,final_detector_sha256=sha(checkpoint),
        refiner_sha256=sha(final_refiner_checkpoint if final_refiner_checkpoint is not None else refiner_checkpoint),
        validation_refiner_sha256=sha(refiner_checkpoint),stable_finetune=stable_finetune,training_contract=training_contract,source_sha256=sha(source),
        submission_sha256=sha(out/'submission.csv'),inventory=result,
        torch=str(torch.__version__),numpy=np.__version__,seed=2028,
        architecture='unchanged V5',overlap_policy=selected_policy,overlap_comparison=comparison,
        elapsed_seconds=time.perf_counter()-start,
        limitations=['A new official assessment, not proof of 0.50 or completed research fixes.'])
    (out/'finalfit_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('V5_FINALFIT_COMPLETE',json.dumps(report),flush=True)
    return report
