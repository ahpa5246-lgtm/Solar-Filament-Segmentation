"""Retrieve the owner's explicit assessment, validate every mask, submit once."""
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from pycocotools import mask as coco

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/DETECTOR_CONSENSUS_OFFICIAL_ASSESSMENT_20261028.json'
OUT=Path('/workspace/scratch/solar-consensus-final/output-v9')


def save(state):
    temp=STATE.with_suffix('.pending');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def main():
    api=KaggleApi();api.authenticate();state=json.loads(STATE.read_text())
    if not state.get('user_requested_official_assessment') or state.get('version')!=9:
        raise ValueError('Expected the explicitly requested immutable assessment v9')
    if not state.get('official_submission_ref'):
        for step in range(65):
            status=str(api.kernels_status(state['reference']).status).split('.')[-1].upper()
            state.update(kaggle_status=status,checked_at_utc=datetime.now(timezone.utc).isoformat());save(state)
            if status=='COMPLETE':break
            if status in {'ERROR','FAILED','CANCELED','CANCELLED'}:raise RuntimeError('Inference failed; do not submit')
            time.sleep(45)
        else:raise TimeoutError('Resume this same assessment; do not duplicate')
        token=''
        for page in range(60):
            _,token=api.kernels_output(state['reference'],str(OUT),
                file_pattern=r'(test_inference\.json|submission-inventory\.json|submission\.csv)$',
                page_token=token,page_size=100,force=True)
            reports=list(OUT.rglob('test_inference.json'));files=list(OUT.rglob('submission.csv'))
            inventories=list(OUT.rglob('submission-inventory.json'))
            if len(reports)==len(files)==len(inventories)==1:break
            if not token:break
        if len(reports)!=1 or len(files)!=1 or len(inventories)!=1:raise RuntimeError('Final report, CSV and inventory required')
        report=json.loads(reports[0].read_text());inventory=json.loads(inventories[0].read_text());target=files[0]
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if report.get('status')!='completed' or report.get('candidate_policy')!='union' or report.get('refiner_threshold')!=.85 or report.get('detector_strategy')!='consensus':
            raise ValueError('Wrong inference artifact or policy')
        if report.get('baseline_parity_min',0)<.99:raise ValueError('Original scored prediction reconstruction failed')
        if digest!=report['submission']['sha256'] or report['submission']['images']!=180 or len(inventory)!=180:
            raise ValueError('Submission hash/inventory mismatch')
        if digest=='94b985ff19e0848d655d0a524265ca67d150fa203531ca18780d3a009cd586d6':
            raise ValueError('Do not resubmit unchanged baseline')
        seen=set();counts={stem:0 for stem in inventory}
        with target.open(newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Wrong columns')
            for row in reader:
                identifier=row['filament_id'];stem,number=identifier.rsplit('_',1)
                if identifier in seen or stem not in counts or int(number)<1:raise ValueError('Bad or duplicate instance identifier')
                seen.add(identifier);counts[stem]+=1
                mask=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                if mask.shape!=(2048,2048) or not mask.any():raise ValueError('Malformed RLE')
        if counts!=inventory or len(seen)!=report['submission']['instances']:raise ValueError('CSV inventory differs')
        state.update(final_inference=report,submission_sha256=digest,preserved_csv=str(target),
                     inventory_verified=True,submission_description='Detector flip4 consensus official assessment sha256:'+digest)
        rows=api.competition_submissions('filament-segmentation-2026') or []
        matching=[r for r in rows if r.description==state['submission_description']]
        if not matching:
            if state.get('submission_attempted'):raise RuntimeError('Ambiguous earlier submission; inspect history before retry')
            state.update(submission_attempted=True,status='SUBMISSION_REQUESTED');save(state)
            api.competition_submit(str(target),state['submission_description'],'filament-segmentation-2026')
            print('OFFICIAL_SUBMISSION_SENT',digest,flush=True)
        save(state)
    for attempt in range(45):
        rows=api.competition_submissions('filament-segmentation-2026') or []
        matching=[r for r in rows if r.description==state['submission_description']]
        if matching:
            r=matching[0];status=str(r.status).split('.')[-1].upper()
            state.update(official_submission=True,official_submission_ref=int(r.ref),official_status=status)
            if status=='COMPLETE' and str(r.public_score) not in {'','None'}:
                score=float(r.public_score)
                state.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,target_reached=score>=.5,
                             improves_preserved_best=score>.32,completed_at_utc=datetime.now(timezone.utc).isoformat())
                save(state);print('OFFICIAL_ASSESSMENT_RESULT',json.dumps({'ref':int(r.ref),'public_pq':score,'target_reached':score>=.5}),flush=True);return
            save(state)
            if status in {'ERROR','FAILED'}:raise RuntimeError('Official evaluator rejected this submission; inspect exact reference')
        time.sleep(20)
    raise TimeoutError('Submitted; score pending. Resume lookup without resubmitting')


if __name__=='__main__':main()
