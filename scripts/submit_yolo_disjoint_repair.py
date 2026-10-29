"""Score the same completed YOLO model after competition-valid disjoint delivery."""
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from pycocotools import mask as coco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/YOLO_DISJOINT_REPAIR_20261029.json'
OUT=Path('/workspace/scratch/solar-yolo-disjoint/output-v18')


def save(s):
    p=STATE.with_suffix('.pending');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)


def main():
    api=KaggleApi();api.authenticate();s=json.loads(STATE.read_text())
    if not s.get('requested_assessment') or s.get('version')!=18:raise ValueError('Expected current-model repair v18')
    if not s.get('official_submission_ref'):
        for i in range(35):
            status=str(api.kernels_status(s['reference']).status).split('.')[-1].upper();s['kaggle_status']=status;save(s)
            if status=='COMPLETE':break
            if status in {'ERROR','FAILED','CANCELED','CANCELLED'}:raise RuntimeError('Kaggle correction run failed; inspect existing version')
            time.sleep(20)
        else:raise TimeoutError('Resume same CPU validation; do not relaunch')
        token=''
        for page in range(1000):
            _,token=api.kernels_output(s['reference'],str(OUT),file_pattern=r'(disjoint_repair_result\.json|submission\.csv)$',page_token=token,page_size=100)
            reports=list(OUT.rglob('disjoint_repair_result.json'));files=list(OUT.rglob('submission.csv'))
            if len(reports)==len(files)==1:break
            if not token:break
        if len(reports)!=1 or len(files)!=1:raise RuntimeError('Corrected CSV/report required')
        r=json.loads(reports[0].read_text());target=files[0];inventory=r['inventory'];expected=inventory['image_inventory']
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if r.get('status')!='COMPLETE' or not r.get('pixel_disjoint') or digest!=r['submission_sha256'] or digest!=inventory['sha256']:
            raise ValueError('Correction report or file hash differs')
        if r['original_submission_sha256']!='ca7e216361cfa288a6fb6b5a663c6d1b7fd9136ff9eb31076377b1a5c88119c6' or digest==r['original_submission_sha256']:
            raise ValueError('Incorrect source lineage or unchanged invalid CSV')
        if inventory['images']!=180 or len(expected)!=180:raise ValueError('Incomplete test-image inventory')
        counts={stem:0 for stem in expected};predictions={stem:[] for stem in expected};seen=set()
        with target.open(newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Bad CSV schema')
            for row in reader:
                identifier=row['filament_id'];stem,number=identifier.rsplit('_',1)
                if identifier in seen or stem not in counts or int(number)<1:raise ValueError('Invalid or duplicate ID')
                mask={'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')}
                decoded=coco.decode(mask)
                if decoded.shape!=(2048,2048) or not decoded.any():raise ValueError('Bad native mask')
                seen.add(identifier);counts[stem]+=1;predictions[stem].append(mask)
        if counts!=expected or len(seen)!=inventory['instances']:raise ValueError('Row inventory mismatch')
        for masks in predictions.values():
            if masks:
                overlaps=coco.iou(masks,masks,[0]*len(masks));np.fill_diagonal(overlaps,0)
                if (overlaps>0).any():raise ValueError('Overlapping masks may not be submitted')
        description='YOLO11m corrected disjoint masks official assessment sha256:'+digest
        s.update(result=r,preserved_csv=str(target),submission_sha256=digest,inventory_verified=True,pixel_disjoint_verified=True,
                 submission_description=description,minimum_milestone=.4,ultimate_target=.6)
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==description]
        if not matches:
            if s.get('submission_attempted'):raise RuntimeError('Uncertain request; inspect history before retry')
            if api.competition_get_submission_limits('filament-segmentation-2026').num_allowed_now<1:raise RuntimeError('Submission quota exhausted')
            s.update(status='SUBMISSION_REQUESTED',submission_attempted=True);save(s)
            api.competition_submit(str(target),description,'filament-segmentation-2026')
            print('DISJOINT_OFFICIAL_SUBMISSION_SENT',digest,flush=True)
        save(s)
    for i in range(60):
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==s['submission_description']]
        if matches:
            r=matches[0];status=str(r.status).split('.')[-1].upper();s.update(official_submission=True,official_submission_ref=int(r.ref),official_status=status)
            if status=='COMPLETE' and str(r.public_score) not in {'','None'}:
                score=float(r.public_score);s.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,milestone_04_reached=score>=.4,target_06_reached=score>=.6,target_reached=score>=.6,improves_previous_best=score>.32,completed_at_utc=datetime.now(timezone.utc).isoformat());save(s)
                print('DISJOINT_OFFICIAL_RESULT',json.dumps({'ref':int(r.ref),'public_pq':score,'milestone_04_reached':score>=.4,'target_06_reached':score>=.6}),flush=True);return
            save(s)
            if status in {'ERROR','FAILED'}:raise RuntimeError('Official evaluator rejected corrected file; inspect exact error')
        time.sleep(20)
    raise TimeoutError('Submitted; official score pending. Resume lookup, never resubmit blindly')


if __name__=='__main__':main()
