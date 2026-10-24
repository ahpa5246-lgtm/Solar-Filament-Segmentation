"""One gated final-model inference run and one idempotent official submission."""
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COMPETITION='filament-segmentation-2026'
REPORT=Path('reports/V6_OFFICIAL_SUBMISSION.json')


def gate(result):
    r=result.get('result',{})
    return (result.get('status')=='COMPLETE' and r.get('fixed_nms_confirmation') is True
            and r.get('confirmation_observations',0)>=40 and r.get('selected')=='1024|nms'
            and r.get('confirmation_gain_supported') is True and r['paired_macro_delta_ci']['low']>0
            and r['confirmation_candidate']['reference_micro_pq']>r['confirmation_baseline']['reference_micro_pq'])


def existing_submission(api,message,previous,csv_sha):
    attempts=[previous,*previous.get('previous_attempts',[])]
    matching=[p for p in attempts if p.get('csv_sha256')==csv_sha]
    references={int(p['submission_ref']) for p in matching if p.get('submission_ref')}
    if len(references)>1:raise RuntimeError('Duplicate stored submission references')
    if references:return references.pop()
    for page in range(1,11):
        rows=api.competition_submissions(COMPETITION,page_number=page,page_size=100) or []
        matches=[r for r in rows if getattr(r,'description','')==message]
        if len(matches)>1:raise RuntimeError('Duplicate experiment submissions found')
        if matches:return int(matches[0].ref)
        if len(rows)<100:break
    else:raise RuntimeError('Submission-history budget exceeded; refusing duplicate risk')
    if any(p.get('submission_attempted') for p in matching):
        raise RuntimeError('Previous upload outcome is uncertain; inspect account before retrying')
    return None


def save_report(report):
    pending=REPORT.with_suffix('.pending')
    pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(REPORT)


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from run_v6_cpu_audit import error_context
    api=KaggleApi();api.authenticate()
    report={'training':False,'test_inference':True,'competition':COMPETITION,'official_submission':False,
            'notebook':'https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-submit',
            'download_page':'https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-submit/output?select=submission.csv'}
    previous=json.loads(REPORT.read_text()) if REPORT.exists() else {}
    history=previous.get('previous_attempts',[])[:]
    if previous.get('csv_sha256'):
        record={k:previous[k] for k in ['csv_sha256','submission_ref','submission_attempted'] if k in previous}
        if record not in history:history.append(record)
    report['previous_attempts']=history
    reference=None
    try:
        confirmation=json.loads(Path('reports/V6_KAGGLE_CONFIRMATION_EXECUTION.json').read_text())
        if not gate(confirmation):raise RuntimeError('Unused-date confirmation did not support promotion; no test execution or submission')
        final=json.loads(Path('reports/V5_FINAL_CACHE.json').read_text())
        if final.get('private_verified') is not True or final.get('status')!='ready':raise RuntimeError('Private final cache not ready')
        quota=api.quota_view().gpu_quota
        report['gpu_hours_remaining_before']=(quota.total_time_allowed-quota.time_used).total_seconds()/3600
        if report['gpu_hours_remaining_before']<.75:raise RuntimeError('Insufficient bounded inference quota')
        with tempfile.TemporaryDirectory(prefix='v6-submit-') as tmp:
            folder=Path(tmp)
            subprocess.run([sys.executable,'scripts/build_v6_submission_notebook.py','--output',str(folder/'v6_final.ipynb')],check=True)
            response=api.kernels_push(str(folder),timeout='1200',acc='NvidiaTeslaT4',no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                from audit_v5_source import _redact
                report['launch_rejection']=_redact(str(response.error))[:1000]
                raise RuntimeError('Final test launch rejected')
            version=int(response.versionNumber)
            if version<1:raise RuntimeError('Missing immutable final notebook version')
            reference='minahussain111/solar-filament-research-v6-submit/v'+str(version);report['notebook_version']=version
            deadline=time.monotonic()+1260;state='UNKNOWN'
            while time.monotonic()<deadline:
                state=str(api.kernels_status(reference).status).split('.')[-1].upper()
                if state in {'COMPLETE','ERROR','FAILED','CANCELLED','CANCELED'}:break
                time.sleep(15)
            report['execution_status']=state
            api.kernels_output(reference,str(folder/'output'),file_pattern=r'(submission\.csv|test_inference\.json)$')
            results=list((folder/'output').rglob('test_inference.json'))
            if len(results)==1:report['inference']=json.loads(results[0].read_text())
            if state!='COMPLETE' or report.get('inference',{}).get('status')!='completed':raise RuntimeError('Final test inference did not complete')
            csvs=list((folder/'output').rglob('submission.csv'));expected=report['inference']['submission']['sha256']
            if not csvs or any(hashlib.sha256(p.read_bytes()).hexdigest()!=expected for p in csvs):raise RuntimeError('Downloaded submission fingerprint differs')
            report['csv_sha256']=expected
            message='V6 fixed detector mask NMS; sha256:'+expected[:24]
            # Look up accepted uploads before sending: a rerun must not duplicate one.
            submission_ref=existing_submission(api,message,previous,expected)
            if submission_ref is not None:
                report['reused_existing_submission']=True
            else:
                limits=api.competition_get_submission_limits(COMPETITION)
                if limits.num_allowed_now<1:raise RuntimeError('No official submissions remaining today; CSV remains available')
                report['submission_attempted']=True
                # Persist before the external write; do not blindly retry uncertain outcomes.
                save_report(report)
                response=api.competition_submit(str(csvs[0]),message,COMPETITION,quiet=True)
                submission_ref=getattr(response,'ref',None)
                if not submission_ref:raise RuntimeError('Official submission returned no reference; inspect account before retrying')
                submission_ref=int(submission_ref)
            report.update(official_submission=True,submission_ref=submission_ref)
            save_report(report)
            deadline=time.monotonic()+480
            while time.monotonic()<deadline:
                try:submission=api.get_submission(submission_ref)
                except Exception as error:
                    if getattr(getattr(error,'response',None),'status_code',None)==404:
                        time.sleep(15);continue
                    raise
                report['submission_status']=getattr(submission.status,'name',str(submission.status).split('.')[-1])
                score=getattr(submission,'public_score',None)
                if report['submission_status']=='COMPLETE' and score not in (None,''):
                    value=float(score)
                    if not math.isfinite(value) or not 0<=value<=1:raise RuntimeError('Unexpected official PQ value')
                    report['official_public_score']=value;report['beats_v5_0_32']=value>.32;break
                if 'ERROR' in report['submission_status'].upper():break
                time.sleep(15)
    except Exception as error:
        report['error_type']=type(error).__name__;report['http_status']=getattr(getattr(error,'response',None),'status_code',None)
        if type(error) in (RuntimeError,ValueError):report['diagnosis']=str(error)[:500]
        if reference and report.get('execution_status')!='COMPLETE':
            try:report['execution_errors']=error_context(api,reference)
            except Exception:report['execution_errors_unavailable']=True
    save_report(report)
    if not report.get('official_submission') or 'error_type' in report or 'ERROR' in report.get('submission_status','').upper():raise SystemExit(1)


if __name__=='__main__':main()
