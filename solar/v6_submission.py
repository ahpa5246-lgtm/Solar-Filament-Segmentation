"""Final all-training checkpoint inference and strict submission delivery.

Reuses exact archived Apache-attributed V5 components. Never reads test labels.
"""
import csv
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as coco
from .v6_resolution_audit import original_detector,detect,refine
from .v6_refiner_audit import load_components,postprocess
from .v5_replay import load_predictions
from .v6_audit import measure_rles


def write_rle_submission(predictions,stems,path):
    stems=list(stems);path=Path(path)
    if len(stems)!=len(set(stems)) or set(stems)!=set(predictions):raise ValueError('Incomplete or duplicate test inventory')
    if path.exists():raise ValueError('Refusing to overwrite an existing submission')
    path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_suffix('.pending')
    expected={};counts={}
    try:
        with pending.open('x',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['filament_id','segmentation_rle']);writer.writeheader()
            for stem in stems:
                seen=set();counts[stem]=len(predictions[stem])
                for number,rle in enumerate(predictions[stem],1):
                    if list(rle['size'])!=[2048,2048]:raise ValueError('Mask dimensions must be native 2048x2048')
                    decoded=coco.decode(rle)
                    if decoded.shape!=(2048,2048) or not decoded.any():raise ValueError('Empty or malformed mask')
                    encoded=coco.encode(np.asfortranarray(decoded.astype(np.uint8)))
                    fingerprint=hashlib.sha256(encoded['counts']).hexdigest()
                    if fingerprint in seen:raise ValueError('Duplicate instance mask')
                    seen.add(fingerprint);identifier=f'{stem}_{number}'
                    if identifier in expected:raise ValueError('Duplicate filament ID')
                    expected[identifier]=hashlib.sha256(decoded.tobytes()).hexdigest()
                    writer.writerow({'filament_id':identifier,'segmentation_rle':encoded['counts'].decode('ascii')})
        if not expected:raise ValueError('Empty competition submission')
        with pending.open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Incorrect CSV columns')
            seen=set()
            for row in reader:
                identifier=row['filament_id']
                if identifier in seen or identifier not in expected:raise ValueError('CSV ID inventory differs')
                seen.add(identifier)
                decoded=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                if hashlib.sha256(decoded.tobytes()).hexdigest()!=expected[identifier]:raise ValueError('CSV mask round trip failed')
            if seen!=set(expected):raise ValueError('CSV instance inventory differs')
        pending.rename(path)
    finally:
        pending.unlink(missing_ok=True)
    return {'images':len(stems),'instances':len(expected),'empty_images':sum(n==0 for n in counts.values()),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
            'image_inventory':counts,'roundtrip':'all rows decoded and matched native binary-mask hashes'}


def verify_original_csv(path,predictions):
    expected={f'{stem}_{number}':rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
              for stem,rles in predictions.items() for number,rle in enumerate(rles,1)}
    with Path(path).open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Original CSV schema differs')
        rows=list(reader)
    actual={r['filament_id']:r['segmentation_rle'] for r in rows}
    if len(actual)!=len(rows) or actual!=expected:raise ValueError('Original CSV does not match saved predictions')


@torch.inference_mode()
def run(test_dir,cache,expected_hashes,source,out,submission_path,seconds=1000,candidate_policy='nms',refiner_threshold=.85):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','training':False,'test_inference':True,'official_submission':False,
            'policy':f'detector {candidate_policy}; 1024; score0.6/mask0.4; flip4 refiner{refiner_threshold}; final union',
            'candidate_policy':candidate_policy,'refiner_threshold':refiner_threshold}
    try:
        cache=Path(cache)
        for name,expected in expected_hashes.items():
            if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=expected:raise ValueError('Final artifact fingerprint differs: '+name)
        required={'final_detector.pt','final_refiner.pt','final_configuration.json','test_predictions_2048.pkl','submission.csv'}
        if not required<=set(expected_hashes):raise ValueError('Final provenance inventory incomplete')
        configuration=json.loads((cache/'final_configuration.json').read_text())
        required_config={'score_threshold':.6,'mask_threshold':.4,'refiner_threshold':.85,'use_refiner':True,'refiner_flip4':True}
        if any(configuration.get(k)!=v for k,v in required_config.items()):raise ValueError('Final saved configuration differs from validated pipeline')
        files=sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        stems=[p.stem for p in files]
        if len(files)!=180 or len(set(stems))!=180:raise ValueError('Expected exactly 180 unique official test observations')
        original=load_predictions(cache/'test_predictions_2048.pkl')
        if set(original)!=set(stems):raise ValueError('Original test prediction inventory differs')
        verify_original_csv(cache/'submission.csv',original)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for bounded full test inference')
        torch.manual_seed(42);np.random.seed(42)
        detector=original_detector(source)
        checkpoint=torch.load(cache/'final_detector.pt',map_location='cpu',weights_only=True)
        if int(checkpoint['epoch'])!=int(configuration['detector_epochs']):raise ValueError('Final detector epoch differs')
        detector.load_state_dict(checkpoint['model'],strict=True);detector=detector.cuda().eval()
        helpers=load_components(source);refiner=helpers['CropUNet'](pretrained=False)
        checkpoint=torch.load(cache/'final_refiner.pt',map_location='cpu',weights_only=True)
        if int(checkpoint['epoch'])!=int(configuration['refiner_epochs']):raise ValueError('Final refiner epoch differs')
        refiner.load_state_dict(checkpoint['model'],strict=True);refiner=refiner.cuda().eval()
        report.update(gpu=torch.cuda.get_device_name(0),used_gpus=1,torch=str(torch.__version__),
                      final_artifact_sha256=expected_hashes,baseline_source='preserved original V5/v3 final predictions; not ground truth')
        predictions={};parity=[];changed=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for index,path in enumerate(files):
            if time.perf_counter()-started>seconds:raise RuntimeError('Test inference budget exhausted; no partial CSV accepted')
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(2048,2048):raise ValueError('Missing or incorrect native test image')
            masks,scores=detect(detector,image,1024)
            baseline=refine(refiner,helpers,image,postprocess(masks,scores,'union'))
            match=measure_rles(original[path.stem],baseline)['pq_local'];parity.append(match)
            report['baseline_parity_min']=min(parity)
            if match<.99:raise RuntimeError('Final original-pipeline parity failed; refusing candidate submission')
            candidate=refine(refiner,helpers,image,postprocess(masks,scores,candidate_policy),threshold=refiner_threshold)
            candidate.sort(key=lambda r:(float(coco.toBbox(r)[1]),float(coco.toBbox(r)[0])))
            predictions[path.stem]=candidate
            if measure_rles(baseline,candidate)['pq_local']<1-1e-8:changed+=1
            report['processed_observations']=index+1
            print(f'Final checkpoint test inference {index+1}/180',flush=True)
        torch.cuda.synchronize()
        report.update(inference_and_reconstruction_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      changed_observations_vs_baseline=changed)
        if not changed:raise RuntimeError('Candidate is identical to original baseline; avoid redundant official submission')
        integrity=write_rle_submission(predictions,stems,submission_path)
        (out/'submission-inventory.json').write_text(json.dumps(integrity['image_inventory'],indent=2)+'\n')
        report.update(status='completed',submission={k:v for k,v in integrity.items() if k!='image_inventory'},
                      original_official_public_score_owner_confirmed=.32,new_official_score=None,
                      limitations=['Local confirmation is exploratory; original validation was reused in historical model selection.',
                                   'Association of recovered source session to owner-scored V5 submission is not independently established.'])
    except Exception as error:
        report['status']='failed';report['error_type']=type(error).__name__
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)[:500]
        (out/'test_inference.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (out/'test_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
