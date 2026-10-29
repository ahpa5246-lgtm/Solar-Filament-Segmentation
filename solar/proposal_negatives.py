"""Mine training-only false proposals with annotation-free protected context."""
import json
from pathlib import Path
import cv2,numpy as np
from pycocotools import mask as coco
from .seed_refiner import feature_crop
from .v6_refiner_audit import references,postprocess

def safe_negative_feature(image,rle,protected,helpers):
 if image.shape!=protected.shape:raise ValueError('Protection mask dimensions differ')
 feature,bounds=feature_crop(image,rle,helpers);x0,y0,x1,y1=bounds
 if protected[y0:y1,x0:x1].any():return None
 return feature

def mine(annotation,rows,stems,helpers,detector,out):
 from .scale_refiner_experiment import scaled_proposals
 out=Path(out);out.mkdir(parents=True,exist_ok=True);stems=set(stems)
 fit={r['observation_id'] for r in rows if r['split']=='train'}
 if not stems<=fit:raise ValueError('Proposal mining must use fitting observations only')
 truth=references(annotation,stems);lookup={r['observation_id']:r for r in rows};paths=[];provenance=[]
 for index,stem in enumerate(sorted(stems),1):
  image=cv2.imread(str(Path(annotation).parent/'train_images'/lookup[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
  if image is None or image.shape!=(2048,2048):raise ValueError('Native fitting image missing')
  known=[r for variant in truth[stem] for r in variant];union=coco.decode(coco.merge(known)) if known else np.zeros_like(image)
  protected=cv2.dilate(union.astype(np.uint8),np.ones((97,97),np.uint8))
  pairs=[scaled_proposals(detector,image,s) for s in (1024,1536)]
  rles=[r for p in pairs for r in p[0]];scores=[v for p in pairs for v in p[1]];rles=postprocess(rles,scores,'nms')
  saved=0
  for rle in rles:
   feature=safe_negative_feature(image,rle,protected,helpers)
   if feature is None:continue
   path=out/f'{stem}-false-{saved:02d}.npy';np.save(path,feature);paths.append(path);provenance.append({'observation_id':stem,'file':path.name,'context_free_of_all_annotations_with_48px_margin':True});saved+=1
   if saved>=5:break
  if index%50==0:print('PROPOSAL_NEGATIVES',index,'/565','mined',len(paths),flush=True)
 if len(paths)<20:raise RuntimeError('Insufficient verified actual false-proposal negatives')
 (out/'provenance.json').write_text(json.dumps({'fitting_only':True,'negative_examples':provenance},indent=2))
 return paths
