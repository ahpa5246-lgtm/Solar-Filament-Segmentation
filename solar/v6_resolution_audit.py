"""Bounded native-resolution pilot using the exact Apache-attributed V5 detector.

No training/test inference. Pilot samples are fixed before examining errors.
"""
import ast
import csv
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as coco
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from .v6_refiner_audit import SOURCE_SHA,REFINER_SHA,load_components,postprocess,references,evaluate
from .v5_replay import aggregate,load_predictions,verify_artifacts,V3_ARTIFACT_SHA256
from .v6_cached_search import micro,paired_ci
from .v6_audit import measure_rles,overlay

DETECTOR_SHA='60a8f48340cc1076710027c1b159c747035217613a61e186b8705aa6fc310a36'


def original_detector(source):
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Archived detector source fingerprint differs')
    for cell in json.loads(raw)['cells']:
        code=''.join(cell['source'])
        if cell['cell_type']!='code' or not code.startswith('%%writefile'):continue
        tree=ast.parse(code.split('\n',1)[1])
        definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='build_model']
        if len(definitions)==1:
            namespace={name:globals()[name] for name in ['maskrcnn_resnet50_fpn_v2','MaskRCNN_ResNet50_FPN_V2_Weights','FastRCNNPredictor','MaskRCNNPredictor','AnchorGenerator','MultiScaleRoIAlign']}
            namespace['IMAGE_SIZE']=1024
            exec(compile(ast.Module(body=definitions,type_ignores=[]),'archived_apache_v5_detector','exec'),namespace)
            return namespace['build_model'](pretrained=False)
    raise ValueError('Original detector builder missing')


def select_pilot_dates(training,validation,per_subset=16):
    train_dates={s[:8] for s in training}
    eligible=sorted(s for s in validation if s[:8] not in train_dates)
    dates=sorted({s[:8] for s in eligible});np.random.default_rng(42).shuffle(dates)
    tune_dates=set(dates[:len(dates)//2]);confirm_dates=set(dates[len(dates)//2:])
    rng=np.random.default_rng(43);subsets=[]
    for allowed in [tune_dates,confirm_dates]:
        stems=sorted(s for s in eligible if s[:8] in allowed)
        if per_subset is None:subsets.append(stems);continue
        if len(stems)<per_subset:raise ValueError('Insufficient date-isolated pilot observations')
        subsets.append(sorted(rng.choice(stems,per_subset,replace=False).tolist()))
    return subsets


def confirmation_dates(training,validation):
    tuning,confirmation=select_pilot_dates(training,validation,None)
    _,pilot_confirmation=select_pilot_dates(training,validation,16)
    pilot_dates={stem[:8] for stem in pilot_confirmation}
    confirmation=sorted(stem for stem in confirmation if stem[:8] not in pilot_dates)
    if not confirmation:raise ValueError('No unused confirmation observations remain')
    return tuning,confirmation


@torch.inference_mode()
def detect(model,image,resolution):
    model.transform.min_size=(resolution,);model.transform.max_size=resolution
    resized=cv2.resize(image,(resolution,resolution),interpolation=cv2.INTER_AREA)
    tensor=torch.from_numpy(resized.astype(np.float32)/255).unsqueeze(0).repeat(3,1,1).cuda()
    output=model([tensor])[0];keep=output['scores']>=.6
    scores=output['scores'][keep].cpu().numpy()
    masks=(output['masks'][keep,0]>=.4).cpu().numpy().astype(np.uint8)
    rles=[];confidence=[]
    for mask,score in zip(masks,scores):
        if mask.any():rles.append(coco.encode(np.asfortranarray(mask)));confidence.append(float(score))
    return rles,confidence


@torch.inference_mode()
def refine(model,helpers,image,rles):
    tasks=[];results=[]
    for rle in rles:
        mask=coco.decode(rle)
        if mask.shape!=(2048,2048):mask=cv2.resize(mask,(2048,2048),interpolation=cv2.INTER_NEAREST)
        fallback=coco.encode(np.asfortranarray(mask.astype(np.uint8)))
        bounds=helpers['square_bounds'](mask);x0,y0,x1,y1=bounds
        crop=cv2.resize(image[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_LINEAR)
        tasks.append((bounds,crop,fallback))
    for start in range(0,len(tasks),32):
        chunk=tasks[start:start+32]
        batch=torch.from_numpy(np.stack([r[1] for r in chunk])).float().unsqueeze(1).repeat(1,3,1,1).cuda()/255
        probs=helpers['predict_refiner_probabilities'](model,batch,use_tta=True).cpu().numpy()
        for (bounds,_,fallback),probability in zip(chunk,probs):
            x0,y0,x1,y1=bounds
            mask=cv2.resize(probability,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)>=.85
            if not mask.any():results.append(fallback);continue
            native=np.zeros((2048,2048),np.uint8);native[y0:y1,x0:x1]=mask
            results.append(coco.encode(np.asfortranarray(native)))
    return postprocess(results,[1.]*len(results),'union')


@torch.inference_mode()
def run(annotation,manifest,detector_predictions,plain_predictions,detector_checkpoint,refiner_checkpoint,source,out,seconds=1000,full_confirmation=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','training':False,'test_inference':False,'official_submission':False,
            'pilot_observations_per_subset':None if full_confirmation else 16,'refiner':'flip4 threshold 0.85, final overlap union',
            'score_threshold':.6,'mask_threshold':.4,'source_sha256':SOURCE_SHA}
    try:
        verify_artifacts({'training_manifest.csv':manifest,'best_detector_predictions_1024.pkl':detector_predictions,
                          'refined_validation_predictions_2048.pkl':plain_predictions},V3_ARTIFACT_SHA256)
        for checkpoint,expected in [(detector_checkpoint,DETECTOR_SHA),(refiner_checkpoint,REFINER_SHA)]:
            if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint fingerprint differs')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.manual_seed(42);np.random.seed(42)
        detector=original_detector(source)
        saved=torch.load(detector_checkpoint,map_location='cpu',weights_only=True)
        detector.load_state_dict(saved['model'],strict=True);detector=detector.cuda().eval()
        helpers=load_components(source);refiner=helpers['CropUNet'](pretrained=False)
        refiner.load_state_dict(torch.load(refiner_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        refiner=refiner.cuda().eval()
        rows=list(csv.DictReader(Path(manifest).open()));training={r['observation_id'] for r in rows if r['split']=='train'}
        validation={r['observation_id']:r for r in rows if r['split']=='val'}
        tuning,confirmation=confirmation_dates(training,validation) if full_confirmation else select_pilot_dates(training,validation)
        stems=sorted(tuning+confirmation);truth=references(annotation,set(stems))
        cached=load_predictions(Path(detector_predictions));images=Path(annotation).parent/'train_images'
        resolutions=[1024] if full_confirmation else [1024,2048]
        strategies={f'{resolution}|{policy}':{} for resolution in resolutions for policy in ['union','nms']}
        parity=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        report.update(gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),used_gpus=1,
                      torch=str(torch.__version__),detector_epoch=int(saved['epoch']))
        report.update(fixed_nms_confirmation=full_confirmation,tuning_observations=len(tuning),confirmation_observations=len(confirmation),pilot_confirmation_excluded=len(select_pilot_dates(training,validation,None)[1])-len(confirmation) if full_confirmation else 0)
        for number,stem in enumerate(stems):
            if time.perf_counter()-started>seconds:raise RuntimeError('Pilot budget exhausted; refusing partial-result selection')
            image=cv2.imread(str(images/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(2048,2048):raise ValueError('Native training observation missing')
            for resolution in resolutions:
                masks,scores=detect(detector,image,resolution)
                for policy in ['union','nms']:
                    proposals=postprocess(masks,scores,policy)
                    if resolution==1024 and policy=='union':
                        parity.append(measure_rles(cached[stem],proposals,shape=(1024,1024))['pq_local'])
                        if parity[-1]<.99:raise RuntimeError('Archived detector mask parity failed before resolution comparison')
                    strategies[f'{resolution}|{policy}'][stem]=refine(refiner,helpers,image,proposals)
            report['processed_observations']=number+1
            print(f'Native resolution pilot {number+1}/{len(stems)}',flush=True)
        torch.cuda.synchronize();report['inference_and_reconstruction_seconds']=time.perf_counter()-started
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated();report['detector_cache_parity_min']=min(parity)
        trials=[{'strategy':name,'tuning_micro_pq':micro(evaluate(truth,pred,tuning))} for name,pred in strategies.items()]
        baseline='1024|union';selected='1024|nms' if full_confirmation else max(trials,key=lambda row:(row['tuning_micro_pq'],row['strategy']==baseline))['strategy']
        base=evaluate(truth,strategies[baseline],confirmation);candidate=evaluate(truth,strategies[selected],confirmation)
        delta=paired_ci(base,candidate)
        report.update(status='completed',tuning_grid=trials,baseline=baseline,selected=selected,
                      confirmation_baseline=aggregate(base),confirmation_candidate=aggregate(candidate),
                      paired_macro_delta_ci=delta,confirmation_gain_supported=(micro(candidate)>micro(base) and delta['low']>0),
                      score_status='LOCAL small exploratory pilot, not official PQ',
                      limitations=['Fixed NMS and pilot-excluded confirmation; historical validation reuse remains.' if full_confirmation else '16 observations per subset give limited statistical power.',
                                   'Original model selection reused the old validation set.',
                                   'No training, alternate architecture, test inference or official submission.'])
        # Keep observation IDs, masks and qualitative panels only in private Kaggle output.
        (out/'pilot_observations.json').write_text(json.dumps({'tuning':tuning,'confirmation':confirmation}))
        for index,stem in enumerate(confirmation[:4]):
            image=cv2.imread(str(images/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
            gt=np.stack([coco.decode(r) for r in truth[stem][0]]) if truth[stem][0] else np.zeros((0,2048,2048),bool)
            pred=strategies[selected][stem];masks=np.stack([coco.decode(r) for r in pred]) if pred else np.zeros((0,2048,2048),bool)
            overlay(image,gt,masks,out/f'figures/native-{index:02d}.png')
    except Exception as error:
        report['status']='failed';report['error_type']=type(error).__name__
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)[:500]
        (out/'resolution_audit.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (out/'resolution_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
