"""Validate the immutable YOLO v17 output and submit it exactly once."""
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from pycocotools import mask as coco
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/YOLO_SOLAR_EXECUTION_20261028.json'
CSV=Path('/workspace/scratch/solar-yolo-training/final-v17/yolo-solar/submission.csv')
REPORT=CSV.parent/'yolo_result.json'


def save(s):
    p=STATE.with_suffix('.pending');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)


def main():
    api=KaggleApi();api.authenticate();s=json.loads(STATE.read_text())
    if not s.get('requested_assessment') or s.get('version')!=17:raise ValueError('Expected the authorized immutable v17')
    if not s.get('official_submission_ref'):
        if not str(api.kernels_status(s['reference']).status).endswith('COMPLETE'):raise ValueError('Kaggle execution not complete')
        r=json.loads(REPORT.read_text());inventory=r['inventory'];counts=inventory['image_inventory']
        if r.get('status')!='COMPLETE' or not r.get('official_assessment_allowed') or r.get('ultralytics_version')!='8.3.221' or r.get('overlap_mask') is not False:
            raise ValueError('Wrong or unvalidated model result')
        digest=hashlib.sha256(CSV.read_bytes()).hexdigest()
        if digest!=r['submission_sha256'] or digest!=inventory['sha256'] or inventory['images']!=180 or len(counts)!=180:
            raise ValueError('Final artifact hash/inventory mismatch')
        if r.get('final_refiner_sha256')!='a08df74538b5752c0df1047fea81b1b1aea4d81d3f749b340c993b4d5c8da924':raise ValueError('Wrong final refiner')
        seen=set();observed={stem:0 for stem in counts}
        with CSV.open(newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Wrong columns')
            for row in reader:
                identifier=row['filament_id'];stem,num=identifier.rsplit('_',1)
                if identifier in seen or stem not in observed or int(num)<1:raise ValueError('Bad/duplicate ID')
                seen.add(identifier);observed[stem]+=1
                mask=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                if mask.shape!=(2048,2048) or not mask.any():raise ValueError('Malformed native mask')
        if observed!=counts or len(seen)!=inventory['instances']:raise ValueError('Row/image inventory mismatch')
        description='YOLO11m independent instances + refiner official assessment sha256:'+digest
        s.update(result=r,submission_sha256=digest,preserved_csv=str(CSV),inventory_verified=True,submission_description=description)
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==description]
        if not matches:
            if s.get('submission_attempted'):raise RuntimeError('Uncertain prior submission; inspect instead of retrying')
            limits=api.competition_get_submission_limits('filament-segmentation-2026')
            if limits.num_allowed_now<1:raise RuntimeError('Kaggle submission quota currently exhausted')
            s.update(status='SUBMISSION_REQUESTED',submission_attempted=True);save(s)
            api.competition_submit(str(CSV),description,'filament-segmentation-2026')
            print('YOLO_OFFICIAL_SUBMISSION_SENT',digest,flush=True)
        save(s)
    for attempt in range(60):
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==s['submission_description']]
        if matches:
            r=matches[0];status=str(r.status).split('.')[-1].upper();s.update(official_submission=True,official_submission_ref=int(r.ref),official_status=status)
            if status=='COMPLETE' and str(r.public_score) not in {'','None'}:
                score=float(r.public_score);s.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,target_reached=score>=.5,improves_original_best=score>.32,completed_at_utc=datetime.now(timezone.utc).isoformat());save(s)
                print('YOLO_OFFICIAL_RESULT',json.dumps({'ref':int(r.ref),'public_pq':score,'target_reached':score>=.5}),flush=True);return
            save(s)
            if status in {'ERROR','FAILED'}:raise RuntimeError('Official evaluator rejected the exact submission')
        time.sleep(20)
    raise TimeoutError('Submitted; score pending. Resume score lookup without resubmitting')


if __name__=='__main__':main()
