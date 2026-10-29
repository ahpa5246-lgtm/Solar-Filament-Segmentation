"""Measured detector-level multi-view ablation, then frozen final-weight inference."""
import csv,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from .detector_consensus import detect_views,fuse_views
from .v6_resolution_audit import original_detector,refine,select_pilot_dates,DETECTOR_SHA
from .v6_refiner_audit import load_components,postprocess,references,evaluate,REFINER_SHA,SOURCE_SHA
from .v6_cached_search import micro,paired_ci
from .v5_replay import load_predictions,aggregate,V3_ARTIFACT_SHA256,verify_artifacts
from .v6_audit import measure_rles


def run(annotation,manifest,detector_cache,plain_cache,detector_checkpoint,
        refiner_checkpoint,source,final_cache,final_hashes,out,seconds=3300):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    report={'status':'STARTED','training':False,'test_inference':False,'official_submission':False,
            'experiment':'Four-view DETECTOR consensus; unchanged flip4 refiner .85',
            'limitations':['Historical validation reused; official scoring is required.',
                           'No test labels, external annotation sets, or training on test observations.']}
    def save():
        temp=out/'consensus_result.pending';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'consensus_result.json')
    save()
    try:
        verify_artifacts({'training_manifest.csv':Path(manifest),'best_detector_predictions_1024.pkl':Path(detector_cache),
                          'refined_validation_predictions_2048.pkl':Path(plain_cache)},V3_ARTIFACT_SHA256)
        for path,expected in ((source,SOURCE_SHA),(detector_checkpoint,DETECTOR_SHA),(refiner_checkpoint,REFINER_SHA)):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Validation input fingerprint differs')
        torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2);cv2.setNumThreads(0)
        model=original_detector(source);model.load_state_dict(torch.load(detector_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        model=model.cuda().eval();helpers=load_components(source);refiner=helpers['CropUNet'](pretrained=False)
        refiner.load_state_dict(torch.load(refiner_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True);refiner=refiner.cuda().eval()
        rows=list(csv.DictReader(Path(manifest).open()));training=[r['observation_id'] for r in rows if r['split']=='train']
        validation={r['observation_id']:r for r in rows if r['split']=='val'}
        tuning,confirmation=select_pilot_dates(training,validation,None)
        from .v5_finetune import validate_partitions
        validate_partitions(training,tuning,confirmation)
        truth=references(annotation,set(tuning+confirmation));cached=load_predictions(Path(detector_cache))
        predictions={name:{} for name in ('baseline','augment','consensus')};parity=[]
        report.update(tuning_observations=len(tuning),confirmation_observations=len(confirmation),
                      detector_view_count=4,match_iou=.3,min_distinct_views=2,refiner_threshold=.85,gpu=torch.cuda.get_device_name(0))
        for index,stem in enumerate(tuning+confirmation,1):
            if time.perf_counter()-started>seconds-1200:raise TimeoutError('Validation budget exhausted; no partial export')
            image=cv2.imread(str(Path(annotation).parent/'train_images'/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(2048,2048):raise ValueError('Bad validation image')
            views=detect_views(model,image);masks,scores=views[0];base=postprocess(masks,scores,'union')
            parity.append(measure_rles(cached[stem],base,shape=(1024,1024))['pq_local'])
            if parity[-1]<.99:raise RuntimeError('Identity detector-view cache parity failed')
            predictions['baseline'][stem]=refine(refiner,helpers,image,base)
            for name in ('augment','consensus'):
                masks,scores=fuse_views(views,keep_base=name=='augment')
                predictions[name][stem]=refine(refiner,helpers,image,postprocess(masks,scores,'union'))
            report.update(processed_validation=index,baseline_detector_parity_min=min(parity));save()
            if index%20==0:print('DETECTOR_CONSENSUS_VALIDATION',index,'/',len(tuning+confirmation),flush=True)
        tuning_pq={name:micro(evaluate(truth,pred,tuning)) for name,pred in predictions.items()}
        selected=max(('augment','consensus'),key=lambda name:tuning_pq[name])
        baseline_metrics=evaluate(truth,predictions['baseline'],confirmation)
        candidate_metrics=evaluate(truth,predictions[selected],confirmation)
        delta=paired_ci(baseline_metrics,candidate_metrics)
        assessment_allowed=tuning_pq[selected]>tuning_pq['baseline'] and micro(candidate_metrics)>micro(baseline_metrics)
        report.update(selected=selected,tuning_pq=tuning_pq,confirmation_baseline=aggregate(baseline_metrics),
                      confirmation_candidate=aggregate(candidate_metrics),paired_macro_delta_ci=delta,
                      local_gain_supported=assessment_allowed and delta['low']>0,
                      official_assessment_allowed=assessment_allowed,validation_seconds=time.perf_counter()-started)
        import pickle
        with (out/'validation_predictions.pkl').open('wb') as handle:pickle.dump(predictions,handle)
        (out/'validation_observations.json').write_text(json.dumps({'tuning':tuning,'confirmation':confirmation}))
        del predictions,model,refiner;torch.cuda.empty_cache();save()
        if not assessment_allowed:
            report.update(status='LOCAL_REJECTED',next_action='Inspect actual error profiles and independent code review before changed experiment');save();return report
        from .v6_submission import run as infer,resolve_test_directory
        final=infer(resolve_test_directory(annotation),final_cache,final_hashes,source,
                    out/'test',Path('/kaggle/working/submission.csv'),seconds=1200,
                    candidate_policy='union',refiner_threshold=.85,detector_strategy=selected)
        report.update(status='COMPLETE',test_inference=True,final_inference=final,
                      elapsed_seconds=time.perf_counter()-started)
        save();print('DETECTOR_CONSENSUS_READY',json.dumps(report),flush=True);return report
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,diagnosis=str(error)[:600],elapsed_seconds=time.perf_counter()-started);save();raise
