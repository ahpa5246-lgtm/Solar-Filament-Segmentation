"""Validate and submit the already-authorized BN-restored all-training candidate once."""
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from pycocotools import mask as coco
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/BN_RECOVERY_FINAL_ASSESSMENT_20261028.json'
OUT=Path('/workspace/scratch/solar-bn-final/output-v2')


def save(s):
    p=STATE.with_suffix('.pending');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)


def main():
    api=KaggleApi();api.authenticate();s=json.loads(STATE.read_text())
    if not s.get('requested_assessment') or s.get('version')!=2:raise ValueError('Expected authorized immutable BN assessment v2')
    if not s.get('official_submission_ref'):
        for step in range(90):
            status=str(api.kernels_status(s['reference']).status).split('.')[-1].upper();s['kaggle_status']=status;save(s)
            if status=='COMPLETE':break
            if status in {'ERROR','FAILED','CANCELED','CANCELLED'}:raise RuntimeError('BN final fit/inference failed; do not submit')
            time.sleep(40)
        else:raise TimeoutError('Resume same run; do not duplicate')
        token=''
        for page in range(60):
            _,token=api.kernels_output(s['reference'],str(OUT),file_pattern=r'(finalfit_result\.json|submission\.csv)$',page_token=token,page_size=100,force=True)
            reports=list(OUT.rglob('finalfit_result.json'));files=list(OUT.rglob('submission.csv'))
            if len(reports)==len(files)==1:break
            if not token:break
        if len(reports)!=1 or len(files)!=1:raise RuntimeError('Final report and CSV required')
        r=json.loads(reports[0].read_text());inventory=r['inventory'];counts=inventory['image_inventory'];target=files[0]
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if r.get('status')!='COMPLETE' or not r.get('stable_finetune') or r.get('training_contract',{}).get('trainable_backbone_stages')!=3:
            raise ValueError('Wrong stable training artifact')
        if r['continuation_checkpoint_sha256']!=s['selected_checkpoint_sha256']:raise ValueError('Wrong selected checkpoint')
        if r['refiner_sha256']!='a08df74538b5752c0df1047fea81b1b1aea4d81d3f749b340c993b4d5c8da924':raise ValueError('Wrong final refiner')
        if digest!=r['submission_sha256'] or digest!=inventory['sha256'] or inventory['images']!=180 or len(counts)!=180:
            raise ValueError('Submission hash/image inventory mismatch')
        observed={stem:0 for stem in counts};seen=set()
        with target.open(newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Wrong columns')
            for row in reader:
                identifier=row['filament_id'];stem,number=identifier.rsplit('_',1)
                if identifier in seen or stem not in observed or int(number)<1:raise ValueError('Invalid or duplicate identifier')
                seen.add(identifier);observed[stem]+=1
                mask=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                if mask.shape!=(2048,2048) or not mask.any():raise ValueError('Malformed native RLE')
        if observed!=counts or len(seen)!=inventory['instances']:raise ValueError('CSV observation/instance mismatch')
        if digest in {'94b985ff19e0848d655d0a524265ca67d150fa203531ca18780d3a009cd586d6','b9d6dbba8d8c9a82e44a5d3dcd9eb361aa4068529273715d8b8e4c554640174e'}:
            raise ValueError('Previously submitted predictions must not be submitted again')
        s.update(finalfit_result=r,submission_sha256=digest,preserved_csv=str(target),inventory_verified=True,
                 submission_description='BN-restored V5 stable full fit assessment sha256:'+digest)
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==s['submission_description']]
        if not matches:
            if s.get('submission_attempted'):raise RuntimeError('Ambiguous earlier request; inspect before resubmission')
            s.update(status='SUBMISSION_REQUESTED',submission_attempted=True);save(s)
            api.competition_submit(str(target),s['submission_description'],'filament-segmentation-2026')
            print('BN_OFFICIAL_SUBMISSION_SENT',digest,flush=True)
        save(s)
    for attempt in range(60):
        matches=[r for r in api.competition_submissions('filament-segmentation-2026') if r.description==s['submission_description']]
        if matches:
            r=matches[0];status=str(r.status).split('.')[-1].upper();s.update(official_submission=True,official_submission_ref=int(r.ref),official_status=status)
            if status=='COMPLETE' and str(r.public_score) not in {'','None'}:
                score=float(r.public_score);s.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,target_reached=score>=.5,improves_original_best=score>.32,completed_at_utc=datetime.now(timezone.utc).isoformat());save(s)
                print('BN_OFFICIAL_RESULT',json.dumps({'ref':int(r.ref),'public_pq':score,'target_reached':score>=.5}),flush=True);return
            save(s)
            if status in {'ERROR','FAILED'}:raise RuntimeError('Official evaluator rejected exact submission; inspect it')
        time.sleep(20)
    raise TimeoutError('Submitted; score pending. Resume same ledger without resubmission')


if __name__=='__main__':main()
