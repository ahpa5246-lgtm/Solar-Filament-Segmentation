"""Inference-only replay of preserved Apache-attributed V5 refiner components.

No training or pretrained downloads. Only an exact archived source fingerprint
is accepted; only named pure definitions are loaded, never notebook cells.
"""
import ast
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from pycocotools import mask as coco
from .v6_audit import measure_rles,overlay
from .v6_cached_search import micro,paired_ci

SOURCE_SHA='2781c89d49a31a08e4f3b96d39b439e029bc7535fc596bee3d8e074baf9a2bea'
REFINER_SHA='0bf638645601d436f048ef238c9cd1336b4ff73184ed4553f20b11b44db09e57'
THRESHOLDS=(.65,.75,.85,.9)


def load_components(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Unexpected archived V5 training source')
    names={'ConvBlock','CropUNet','square_bounds','predict_refiner_probabilities'}
    found={}
    for cell in json.loads(raw)['cells']:
        if cell['cell_type']!='code':continue
        code=''.join(cell['source'])
        if code.startswith('%%writefile'):code=code.split('\n',1)[1]
        if code.lstrip().startswith('!'):continue
        try:tree=ast.parse(code)
        except SyntaxError:continue
        for node in tree.body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:found.setdefault(node.name,node)
    if set(found)!=names:raise ValueError('Missing archived model definitions')
    namespace={'torch':torch,'nn':nn,'np':np,'resnet18':resnet18,'ResNet18_Weights':ResNet18_Weights}
    selected=[found[name] for name in ['ConvBlock','CropUNet','square_bounds','predict_refiner_probabilities']]
    exec(compile(ast.Module(body=selected,type_ignores=[]),'archived_apache_v5_helpers','exec'),namespace)
    return namespace


def postprocess(rles,scores,policy):
    if policy not in {'union','nms','none'}:raise ValueError('Unknown overlap policy')
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(rles),) or not np.isfinite(scores).all():raise ValueError('One finite confidence per mask is required')
    if not rles:return []
    if policy=='none':return list(rles)
    iou=coco.iou(rles,rles,[0]*len(rles))
    if policy=='nms':
        keep=[]
        for index in np.argsort(-np.asarray(scores),kind='stable'):
            if all(iou[index,previous]<=.5 for previous in keep):keep.append(int(index))
        return [rles[index] for index in keep]
    if policy!='union':raise ValueError('Unknown overlap policy')
    seen=set();result=[]
    for start in range(len(rles)):
        if start in seen:continue
        stack=[start];group=[]
        while stack:
            index=stack.pop()
            if index in seen:continue
            seen.add(index);group.append(index)
            stack.extend(np.flatnonzero(iou[index]>0).tolist())
        result.append(coco.merge([rles[index] for index in group]))
    return result


def key(mode,threshold,policy):return f'{mode}|threshold={threshold:.2f}|{policy}'


def references(path,stems):
    payload=json.loads(Path(path).read_text());annotations=defaultdict(list);truth=defaultdict(list)
    for ann in payload['annotations']:annotations[str(ann['image_id'])].append(ann)
    for image in payload['images']:
        stem=Path(image['file_name']).stem
        if stem not in stems:continue
        masks=[]
        for ann in annotations[str(image['id'])]:
            parts=coco.frPyObjects(ann['segmentation'],2048,2048)
            if parts:
                rle=coco.merge(parts)
                if coco.area(rle)>0:masks.append(rle)
        truth[stem].append(masks)
    if set(truth)!=set(stems):raise ValueError('Ground-truth observation inventory differs')
    return dict(truth)


def evaluate(truth,predictions,stems):
    return {stem:[measure_rles(gt,predictions[stem]) for gt in truth[stem]] for stem in stems}


@torch.inference_mode()
def run(annotation,manifest,detector_path,refined_path,checkpoint,source,out,seconds=1000):
    from .v5_replay import load_predictions,verify_artifacts,V3_ARTIFACT_SHA256
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','training':False,'test_inference':False,'official_submission':False,
            'source_sha256':SOURCE_SHA,'refiner_sha256':REFINER_SHA,'threshold_grid':list(THRESHOLDS)}
    torch.manual_seed(42);np.random.seed(42)
    try:
        verify_artifacts({'training_manifest.csv':manifest,'best_detector_predictions_1024.pkl':detector_path,
                          'refined_validation_predictions_2048.pkl':refined_path},V3_ARTIFACT_SHA256)
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=REFINER_SHA:raise ValueError('Refiner checkpoint fingerprint differs')
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this bounded replay')
        helpers=load_components(source)
        model=helpers['CropUNet'](pretrained=False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True);model=model.cuda().eval()
        report.update(gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),used_gpus=1,
                      torch=str(torch.__version__),checkpoint_epoch=int(saved['epoch']))
        rows=list(csv.DictReader(Path(manifest).open()))
        training={r['observation_id'] for r in rows if r['split']=='train'}
        validation={r['observation_id']:r for r in rows if r['split']=='val'}
        detector=load_predictions(Path(detector_path));cached_plain=load_predictions(Path(refined_path))
        if set(detector)!=set(validation) or set(cached_plain)!=set(validation):raise ValueError('Original validation inventory differs')
        truth=references(annotation,set(validation))
        strategies={key(mode,threshold,policy):{} for mode in ['plain','flip4'] for threshold in THRESHOLDS for policy in ['union','nms','none']}
        images=Path(annotation).parent/'train_images'
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats();completed=0
        for stem,row in sorted(validation.items()):
            if time.perf_counter()-started>seconds:raise RuntimeError('Inference time budget exhausted; no partial-result promotion')
            path=images/row['file_name'];image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(2048,2048):raise ValueError('Missing native validation image')
            tasks=[]
            for rle in detector[stem]:
                native=cv2.resize(coco.decode(rle),(2048,2048),interpolation=cv2.INTER_NEAREST)
                fallback=coco.encode(np.asfortranarray(native))
                bounds=helpers['square_bounds'](native)
                x0,y0,x1,y1=bounds
                crop=cv2.resize(image[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_LINEAR)
                tasks.append((bounds,crop,fallback))
            raw={(mode,threshold):([],[]) for mode in ['plain','flip4'] for threshold in THRESHOLDS}
            for start in range(0,len(tasks),32):
                batch_tasks=tasks[start:start+32]
                batch=torch.from_numpy(np.stack([task[1] for task in batch_tasks])).float().unsqueeze(1).repeat(1,3,1,1).cuda()/255
                for mode in ['plain','flip4']:
                    probs=helpers['predict_refiner_probabilities'](model,batch,use_tta=(mode=='flip4')).cpu().numpy()
                    for (bounds,_,fallback),probability in zip(batch_tasks,probs):
                        x0,y0,x1,y1=bounds
                        full_probability=cv2.resize(probability,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
                        for threshold in THRESHOLDS:
                            mask=full_probability>=threshold
                            if mask.any():
                                native=np.zeros((2048,2048),np.uint8);native[y0:y1,x0:x1]=mask
                                rle=coco.encode(np.asfortranarray(native))
                                confidence=float(full_probability[mask].mean())
                            else:rle=fallback;confidence=0.
                            raw[(mode,threshold)][0].append(rle);raw[(mode,threshold)][1].append(confidence)
            for (mode,threshold),(masks,scores) in raw.items():
                for policy in ['union','nms','none']:strategies[key(mode,threshold,policy)][stem]=postprocess(masks,scores,policy)
            completed+=1
            report['processed_observations']=completed
            print(f'Inference replay {completed}/{len(validation)}',flush=True)
        torch.cuda.synchronize()
        report['inference_and_reconstruction_seconds']=time.perf_counter()-started
        report['end_to_end_seconds_per_observation']=report['inference_and_reconstruction_seconds']/completed
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        plain_groups=evaluate(truth,strategies[key('plain',.85,'union')],validation)
        cached_groups=evaluate(truth,cached_plain,validation)
        report['plain_parity']={'cached_micro_pq':micro(cached_groups),'replayed_micro_pq':micro(plain_groups)}
        if abs(micro(plain_groups)-micro(cached_groups))>.005:raise RuntimeError('Original plain-refiner parity check failed')
        train_dates={stem[:8] for stem in training}
        eligible=sorted(stem for stem in validation if stem[:8] not in train_dates)
        dates=sorted({stem[:8] for stem in eligible});np.random.default_rng(42).shuffle(dates)
        tuning_dates=set(dates[:len(dates)//2]);confirmation_dates=set(dates[len(dates)//2:])
        tuning=sorted(stem for stem in eligible if stem[:8] in tuning_dates)
        confirmation=sorted(stem for stem in eligible if stem[:8] in confirmation_dates)
        if len(tuning)<1 or len(confirmation)<1:raise ValueError('Insufficient date-isolated holdout')
        baseline=key('flip4',.85,'union')
        trials=[{'strategy':name,'tuning_reference_micro_pq':micro(evaluate(truth,predictions,tuning))} for name,predictions in strategies.items()]
        selected=max(trials,key=lambda row:(row['tuning_reference_micro_pq'],row['strategy']==baseline))['strategy']
        baseline_groups=evaluate(truth,strategies[baseline],confirmation)
        selected_groups=evaluate(truth,strategies[selected],confirmation)
        delta=paired_ci(baseline_groups,selected_groups)
        from .v5_replay import aggregate
        baseline_summary=aggregate(baseline_groups);candidate_summary=aggregate(selected_groups)
        gain=candidate_summary['reference_micro_pq']>baseline_summary['reference_micro_pq'] and delta['low']>0
        report.update(status='completed',baseline=baseline,selected=selected,tuning_grid=trials,
                      tuning_observations=len(tuning),confirmation_observations=len(confirmation),
                      excluded_training_date_observations=len(validation)-len(eligible),
                      confirmation_baseline=baseline_summary,confirmation_candidate=candidate_summary,
                      paired_macro_delta_ci=delta,confirmation_gain_supported=gain,
                      score_status='LOCAL exploratory validation, not an official Kaggle score',
                      limitations=['Original V5 selection already used all original validation data.',
                                   'Cached detector proposals were already merged; detector recall cannot be recovered here.',
                                   'Final full-data model/test predictions were not run or submitted.'])
        # Private Kaggle outputs only: preserve chosen predictions and diagnostic panels.
        serial={name:{stem:[{'size':r['size'],'counts':r['counts'].decode() if isinstance(r['counts'],bytes) else r['counts']} for r in strategies[name][stem]] for stem in validation} for name in {baseline,selected}}
        (out/'validation_predictions.json').write_text(json.dumps(serial))
        hardest=sorted(confirmation,key=lambda stem:sum(r['fp']+r['fn'] for r in baseline_groups[stem]),reverse=True)[:6]
        for number,stem in enumerate(hardest):
            image=cv2.imread(str(images/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            gt=np.stack([coco.decode(r) for r in truth[stem][0]]) if truth[stem][0] else np.zeros((0,2048,2048),bool)
            pred=strategies[selected][stem]
            masks=np.stack([coco.decode(r) for r in pred]) if pred else np.zeros((0,2048,2048),bool)
            overlay(image,gt,masks,out/f'figures/confirmation-{number:02d}.png')
    except Exception as error:
        report['status']='failed';report['error_type']=type(error).__name__
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)[:500]
        (out/'refiner_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    (out/'refiner_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
