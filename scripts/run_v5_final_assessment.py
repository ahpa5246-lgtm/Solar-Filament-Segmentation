"""Continue a completed trained candidate through full fitting, CSV validation and official scoring."""
import csv,hashlib,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from build_v5_finalfit import build,KERNEL
STATE=ROOT/'reports/V5_FINAL_ASSESSMENT.json'


def save(state):
    pending=STATE.with_suffix('.pending');pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(STATE)


def require_candidate(training):
    result=training.get('result',{})
    if training.get('status')!='COMPLETE' or result.get('status')!='COMPLETE':
        raise ValueError('Wait for a complete repaired continuation; no duplicate or fallback GPU run')
    if result.get('best_epoch',0)<1 or not result.get('best_tuning_pq',0)>result.get('baseline_tuning_pq',1):
        raise ValueError('No trained PQ improvement selected; keep baseline and diagnose before official assessment')
    return result


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    training=json.loads((ROOT/'reports/V5_FINETUNE_STATE.json').read_text());candidate=require_candidate(training)
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get('official_submission_ref'):
        print('Known official assessment retained; refreshing its score only',flush=True)
    if not state:
        quota=api.quota_view().gpu_quota
        remaining=(quota.total_time_allowed-quota.time_used).total_seconds()/3600
        if remaining<4.25:raise RuntimeError('Insufficient quota for full fit and inference')
        state=dict(status='LAUNCH_REQUESTED',push_attempted=True,training_reference=training['reference'],
                   selected_checkpoint_sha256=candidate['best_checkpoint_sha256'],official_submission=False,
                   local_confirmation_supported=candidate.get('local_gain_supported',False))
        with tempfile.TemporaryDirectory() as folder:
            build(Path(folder)/'final.ipynb',int(training['version']),candidate['best_checkpoint_sha256'])
            save(state)
            response=api.kernels_push(folder,timeout='7200',acc='NvidiaTeslaT4',no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError('Final assessment inputs rejected; inspect launch intent, never duplicate')
            version=int(response.versionNumber)
            if version<1:raise RuntimeError('Final assessment returned no version')
            state.update(status='RUNNING',reference=KERNEL+'/v'+str(version));save(state)
    if not state.get('reference'):raise RuntimeError('Ambiguous final launch; inspect Kaggle before any retry')
    deadline=time.monotonic()+7500
    while time.monotonic()<deadline:
        status=str(api.kernels_status(state['reference']).status).split('.')[-1].upper()
        state['kaggle_status']=status;save(state);print('V5_FINAL_ASSESSMENT',state['reference'],status,flush=True)
        if status=='COMPLETE':break
        if status in {'ERROR','FAILED','CANCELED','CANCELLED'}:raise RuntimeError('Final fit/inference failed; no CSV submitted')
        time.sleep(60)
    if state.get('kaggle_status')!='COMPLETE':raise TimeoutError('Final inference not complete; no CSV submitted')
    with tempfile.TemporaryDirectory() as folder:
        token=''
        for page in range(60):
            for attempt,delay in enumerate((0,30,60,120)):
                if delay:time.sleep(delay)
                try:
                    _,following=api.kernels_output(state['reference'],folder,
                        file_pattern=r'(finalfit_result\.json|submission\.csv)$',page_token=token,page_size=100)
                    break
                except Exception as error:
                    if getattr(getattr(error,'response',None),'status_code',None)!=429 or attempt==3:raise
            reports=list(Path(folder).rglob('finalfit_result.json'));files=list(Path(folder).rglob('submission.csv'))
            if len(reports)==len(files)==1:break
            if not following:break
            token=following
        if len(reports)!=1 or len(files)!=1:raise RuntimeError('Complete final CSV and report not retrieved; do not rerun GPU')
        report=json.loads(reports[0].read_text());target=files[0]
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if report.get('status')!='COMPLETE' or digest!=report.get('submission_sha256') or report['inventory']['images']!=180:
            raise ValueError('Final CSV artifact validation failed')
        if report['continuation_checkpoint_sha256']!=state['selected_checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        from pycocotools import mask as coco
        seen=set()
        with target.open(newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Invalid official CSV header')
            for row in reader:
                if row['filament_id'] in seen:raise ValueError('Duplicate official filament identifier')
                seen.add(row['filament_id'])
                mask=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode('ascii')})
                if mask.shape!=(2048,2048) or not mask.any():raise ValueError('Invalid official RLE')
        if len(seen)!=report['inventory']['instances']:raise ValueError('Official row count mismatch')
        if digest=='94b985ff19e0848d655d0a524265ca67d150fa203531ca18780d3a009cd586d6':
            raise ValueError('Unchanged baseline CSV must not be submitted again')
        description='V5 continuation official assessment sha256:'+digest
        rows=api.competition_submissions('filament-segmentation-2026') or []
        matching=[r for r in rows if r.description==description]
        state.update(finalfit_result=report,submission_sha256=digest,submission_description=description)
        if not matching and not state.get('submission_attempted'):
            state.update(submission_attempted=True,status='SUBMISSION_REQUESTED');save(state)
            response=api.competition_submit(str(target),description,'filament-segmentation-2026')
            print('KAGGLE_SUBMISSION_RESPONSE',str(response),flush=True)
        elif not matching and state.get('submission_attempted'):
            raise RuntimeError('Uncertain submission request; inspect competition history, never blindly resubmit')
        save(state)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        rows=api.competition_submissions('filament-segmentation-2026') or []
        matching=[r for r in rows if r.description==state['submission_description']]
        if matching:
            row=matching[0];status=str(row.status).split('.')[-1].upper()
            state.update(official_submission=True,official_submission_ref=int(row.ref),official_status=status)
            if status=='COMPLETE' and str(row.public_score) not in {'','None'}:
                score=float(row.public_score)
                state.update(status='OFFICIAL_SCORE_VERIFIED',official_public_pq=score,
                             target_reached=score>=.5,improves_original_v5=score>.32)
                save(state);print('V5_OFFICIAL_RESULT',json.dumps(state),flush=True);return
            save(state)
            if status in {'ERROR','FAILED'}:raise RuntimeError('Kaggle rejected submission; inspect this exact reference')
        time.sleep(30)
    raise TimeoutError('Official score pending; resume lookup without re-submitting')


if __name__=='__main__':main()
