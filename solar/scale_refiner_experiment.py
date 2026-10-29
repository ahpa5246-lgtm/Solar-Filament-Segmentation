"""Compare native proposal recall across resolutions with frozen trained models."""
import csv,json,hashlib,time
from pathlib import Path
import cv2,numpy as np,torch
from pycocotools import mask as coco
from .v6_refiner_audit import references,evaluate,load_components,postprocess
from .v6_resolution_audit import select_pilot_dates
from .v6_cached_search import micro
from .v5_replay import aggregate
from .seed_refiner import predict_maps,restore_masks
from .v6_submission import resolve_test_directory,write_rle_submission

@torch.inference_mode()
def scaled_proposals(detector,image,scale):
 result=detector.predict(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),imgsz=scale,conf=.2,iou=.9,max_det=100,device=0,retina_masks=True,verbose=False)[0]
 rles=[];scores=[]
 if result.masks is not None:
  for mask,score in zip(result.masks.data.cpu().numpy(),result.boxes.conf.cpu().numpy()):
   if mask.shape!=image.shape:mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
   mask=(mask>.5).astype(np.uint8)
   if mask.any():rles.append(coco.encode(np.asfortranarray(mask)));scores.append(float(score))
 return rles,scores

def run(annotation,manifest,source,yolo_selected,yolo_final,seed_selected,seed_final,out):
 from ultralytics import YOLO
 out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.set_num_threads(2);cv2.setNumThreads(0)
 rows=list(csv.DictReader(Path(manifest).open()));fit=[r['observation_id'] for r in rows if r['split']=='train'];val={r['observation_id']:r for r in rows if r['split']=='val'}
 tuning,confirmation=select_pilot_dates(fit,val,None);truth=references(annotation,set(tuning+confirmation))
 helpers=load_components(source);model=helpers['CropUNet'](pretrained=False).cuda();model.load_state_dict(torch.load(seed_selected,map_location='cuda',weights_only=True)['model']);model.eval();detector=YOLO(str(yolo_selected))
 # Baseline is included; multi-resolution alternatives seek additional small-object proposals.
 configs=[((1024,),.35),((1536,),.25),((1536,),.35),((2048,),.25),((2048,),.35),((1024,1536),.25),((1024,1536),.35)]
 def key(c):return '+'.join(map(str,c[0]))+'|'+str(c[1])
 def image_for(stem):
  im=cv2.imread(str(Path(annotation).parent/'train_images'/val[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
  if im is None or im.shape!=(2048,2048):raise ValueError('Native validation image missing')
  return im
 def predict(image,settings):
  coarse={scale:scaled_proposals(detector,image,scale) for scale in {s for scales,c in settings for s in scales}}
  result={}
  for scales in {c[0] for c in settings}:
   rles=[r for s in scales for r in coarse[s][0]];scores=[v for s in scales for v in coarse[s][1]]
   score_map={id(r):s for r,s in zip(rles,scores)};rles=postprocess(rles,scores,'nms');scores=[score_map[id(r)] for r in rles]
   bounds,probabilities=predict_maps(model,helpers,image,rles)
   for cfg in [c for c in settings if c[0]==scales]:
    keep=[i for i,s in enumerate(scores) if s>=cfg[1]]
    result[key(cfg)]=restore_masks([bounds[i] for i in keep],[probabilities[i] for i in keep],image.shape,.65,'union')
  return result
 predictions={key(c):{} for c in configs}
 for index,stem in enumerate(tuning,1):
  for k,v in predict(image_for(stem),configs).items():predictions[k][stem]=v
  if index%10==0:print('SCALE_TUNING',index,flush=True)
 grid={k:micro(evaluate(truth,v,tuning)) for k,v in predictions.items()};winner=max((k for k in grid if k!=key(((1024,),.35))),key=grid.get);selected=next(c for c in configs if key(c)==winner);del predictions
 frozen={stem:predict(image_for(stem),[selected])[winner] for stem in confirmation}
 r={'status':'VALIDATED','training':False,'checkpoint_policy':'Frozen seed v2 and YOLO v17 checkpoints','selection_note':'Best nonbaseline configuration on tuning; baseline included as comparison and already officially scored 0.37.','tuning_grid':grid,'selected_config':[list(selected[0]),selected[1]],'confirmation':aggregate(evaluate(truth,frozen,confirmation)),'official_public_pq':None,'official_assessment_allowed':True}
 (out/'scale_result.json').write_text(json.dumps(r,indent=2));print('SCALE_SELECTED',json.dumps(r),flush=True)
 detector=YOLO(str(yolo_final));model.load_state_dict(torch.load(seed_final,map_location='cuda',weights_only=True)['model']);model.eval()
 files=sorted(p for p in resolve_test_directory(annotation).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
 if len(files)!=180:raise ValueError('Expected all 180 test images')
 test={}
 for i,path in enumerate(files,1):
  image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);test[path.stem]=predict(image,[selected])[winner]
  if i%30==0:print('SCALE_TEST',i,flush=True)
 inventory=write_rle_submission(test,[p.stem for p in files],out/'submission.csv')
 r.update(status='COMPLETE',inventory=inventory,submission_sha256=inventory['sha256'],elapsed_seconds=time.perf_counter()-start)
 (out/'scale_result.json').write_text(json.dumps(r,indent=2));print('SCALE_FINAL_READY',json.dumps(r),flush=True);return r
