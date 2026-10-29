"""Resume one existing seeded-refiner Kaggle version; assess its unique output officially."""
import hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/SEED_REFINER_EXECUTION_20261029.json'
def save(s):
 p=STATE.with_suffix('.pending');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)
def main():
 a=KaggleApi();a.authenticate()
 if not STATE.exists():
  from build_seed_refiner import build
  folder=Path('/workspace/scratch/solar-seed-refiner');folder.mkdir(parents=True,exist_ok=True)
  nb=folder/'seed.ipynb';meta=build(nb)
  s={'status':'PREPARED','kernel':meta['id'],'notebook_sha256':hashlib.sha256(nb.read_bytes()).hexdigest(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'previous_official_best':.36,'minimum_milestone':.4,'ultimate_target':.6,'push_attempted':False}
  save(s);s['push_attempted']=True;save(s)
  r=a.kernels_push(str(folder));s.update(version=int(r.version_number),reference=meta['id']+'/v'+str(r.version_number),status='KAGGLE_RUNNING');save(s)
 s=json.loads(STATE.read_text())
 if not s.get('reference'):raise RuntimeError('Uncertain push state; inspect Kaggle history before relaunch')
 out=Path('/workspace/scratch/solar-seed-refiner/output-v'+str(s['version']))
 while not s.get('official_submission_ref'):
  status=str(a.kernels_status(s['reference']).status).split('.')[-1].upper();s['kaggle_status']=status;save(s)
  if status in {'ERROR','FAILED','CANCELED','CANCELLED'}:raise RuntimeError('Kaggle failed; inspect exact version logs')
  if status!='COMPLETE':time.sleep(30);continue
  token=''
  for page in range(1000):
   _,token=a.kernels_output(s['reference'],str(out),file_pattern=r'(seed_result\.json|submission\.csv)$',page_token=token,page_size=100)
   reports=list(out.rglob('seed_result.json'));files=list(out.rglob('submission.csv'))
   if len(reports)==len(files)==1:break
   if not token:break
  if len(reports)!=1 or len(files)!=1:raise RuntimeError('Kaggle completed without final report and predictions')
  r=json.loads(reports[0].read_text());csv=files[0];digest=hashlib.sha256(csv.read_bytes()).hexdigest()
  if r.get('status')!='COMPLETE' or not r.get('official_assessment_allowed') or digest!=r['submission_sha256'] or digest!=r['inventory']['sha256']:raise ValueError('Final candidate not verified')
  # Reconstruct and independently invoke the production writer: rejects overlap, bad masks and schema/inventory errors.
  import sys
  sys.path.insert(0,str(ROOT))
  import csv as csvlib
  from solar.v6_submission import write_rle_submission
  pred={stem:[] for stem in r['inventory']['image_inventory']};seen=set()
  with csv.open(newline='') as h:
   reader=csvlib.DictReader(h)
   if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Bad CSV schema')
   for row in reader:
    ident=row['filament_id'];stem,num=ident.rsplit('_',1)
    if ident in seen or stem not in pred or int(num)<1:raise ValueError('Duplicate or invalid prediction')
    seen.add(ident);pred[stem].append({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
  import tempfile
  with tempfile.TemporaryDirectory(prefix='solar-seed-export-check-') as tmp:
   check=write_rle_submission(pred,sorted(pred),Path(tmp)/'submission.csv')
  if check['images']!=180 or check['instances']!=r['inventory']['instances'] or {k:len(v) for k,v in pred.items()}!=r['inventory']['image_inventory']:raise ValueError('Prediction inventory differs')
  description='Box conditioned refiner official assessment sha256:'+digest
  history=list(a.competition_submissions('filament-segmentation-2026'))
  matches=[x for x in history if x.description==description]
  if any(digest in (x.description or '') for x in history) and not matches:raise RuntimeError('Identical predictions already assessed')
  s.update(result=r,preserved_csv=str(csv),submission_sha256=digest,submission_description=description,inventory_verified=True,pixel_disjoint_verified=True);save(s)
  if not matches:
   if s.get('submission_attempted'):raise RuntimeError('Uncertain submission; inspect history before retry')
   if a.competition_get_submission_limits('filament-segmentation-2026').num_allowed_now<1:raise RuntimeError('Submission quota exhausted; preserve candidate and resume later')
   s.update(submission_attempted=True,status='SUBMISSION_REQUESTED');save(s);a.competition_submit(str(csv),description,'filament-segmentation-2026')
  matches=[x for x in a.competition_submissions('filament-segmentation-2026') if x.description==description]
  if matches:s['official_submission_ref']=int(matches[0].ref);save(s)
  else:time.sleep(30)
 while True:
  matches=[x for x in a.competition_submissions('filament-segmentation-2026') if int(x.ref)==s['official_submission_ref']]
  if matches:
   r=matches[0];status=str(r.status).split('.')[-1].upper();s['official_status']=status;save(s)
   if status=='COMPLETE' and str(r.public_score) not in {'','None'}:
    score=float(r.public_score);s.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,improves_previous_best=score>.36,milestone_04_reached=score>=.4,target_06_reached=score>=.6,completed_at_utc=datetime.now(timezone.utc).isoformat());save(s);print('SEED_OFFICIAL_RESULT',s['official_submission_ref'],score,flush=True);return
   if status in {'ERROR','FAILED'}:raise RuntimeError('Official evaluator rejected file; inspect error')
  time.sleep(30)
if __name__=='__main__':main()
