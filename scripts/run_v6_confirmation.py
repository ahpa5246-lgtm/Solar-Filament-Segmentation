"""Launch at most one bounded inference-only Kaggle experiment after CPU readiness."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def read_calibration_result(api, reference, output):
    """Retry rate-limited reads of one immutable version; never execute a kernel."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    token = ''  # Explicit token disables SDK automatic pagination/restarts.
    for page in range(100):
        for attempt, delay in enumerate((0, 30, 60, 120)):
            if delay:
                time.sleep(delay)
            try:
                _, next_token = api.kernels_output(reference, str(output),
                    file_pattern=r'score_calibration\.json$', page_token=token, page_size=100)
                break
            except Exception as error:
                if getattr(getattr(error, 'response', None), 'status_code', None) != 429 or attempt == 3:
                    raise
        print('Read-only output page', page + 1, flush=True)
        if list(output.rglob('score_calibration.json')) or not next_token:
            break
        token = next_token
    files = list(output.rglob('score_calibration.json'))
    if len(files) != 1:
        raise RuntimeError('Expected exactly one immutable calibration result')
    return json.loads(files[0].read_text())


def resume_calibration(api):
    path = Path('reports/V5_SCORE_CALIBRATION_EXECUTION.json')
    report = json.loads(path.read_text())
    if report.get('status') != 'COMPLETE' or type(report.get('version')) is not int:
        raise RuntimeError('Only a known completed immutable calibration run can be recovered')
    reference = 'minahussain111/solar-filament-research-v6-confirmation/v' + str(report['version'])
    with tempfile.TemporaryDirectory(prefix='v5-calibration-read-') as tmp:
        report['result'] = read_calibration_result(api, reference, tmp)
    if report['result'].get('status') != 'completed':
        raise RuntimeError('Recovered comparison did not complete successfully')
    for key in ('error_type', 'http_status', 'execution_error_detail'):
        report.pop(key, None)
    report['recovery'] = 'read-only immutable output retrieval; no GPU launch'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Recovered V5 comparison:', json.dumps(report['result'], sort_keys=True), flush=True)


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from run_v6_cpu_audit import error_context
    api=KaggleApi();api.authenticate()
    if '--resume-score-calibration' in sys.argv:
        resume_calibration(api)
        return
    calibration = '--score-calibration' in sys.argv
    result_name = 'score_calibration.json' if calibration else 'resolution_audit.json'
    report_path = Path('reports/V5_SCORE_CALIBRATION_EXECUTION.json' if calibration else 'reports/V6_KAGGLE_CONFIRMATION_EXECUTION.json')
    report={'training':False,'test_inference':False,'official_submission':False,
            'requested_accelerator':'NvidiaTeslaT4','server_timeout_seconds':1200,
            'notebook':'https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-confirmation'}
    reference=None
    try:
        cache=json.loads(Path('reports/V5_PRIVATE_CACHE.json').read_text())
        cpu=json.loads(Path('reports/V6_KAGGLE_CPU_EXECUTION.json').read_text())
        if cache.get('status')!='ready' or cache.get('private_verified') is not True:raise RuntimeError('Verified private cache is not ready')
        if 'result' not in cpu or not cpu['result'].get('artifact_sha256'):raise RuntimeError('Fingerprint-verified Kaggle CPU replay is not complete')
        previous=json.loads(Path('reports/V6_KAGGLE_REFINER_EXECUTION.json').read_text())
        if previous.get('status')!='COMPLETE':raise RuntimeError('Original refiner replay must complete first')
        quota=api.quota_view().gpu_quota
        remaining=(quota.total_time_allowed-quota.time_used).total_seconds()/3600
        report['gpu_hours_remaining_before']=remaining
        if remaining<.75:raise RuntimeError('Insufficient quota for bounded inference budget')
        with tempfile.TemporaryDirectory(prefix='v6-resolution-') as tmp:
            folder=Path(tmp)
            builder = 'scripts/build_v5_score_calibration.py' if calibration else 'scripts/build_v6_confirmation_notebook.py'
            subprocess.run([sys.executable,builder,'--output',str(folder/'v6_resolution_audit.ipynb')],check=True)
            metadata=json.loads((folder/'kernel-metadata.json').read_text())
            if not metadata['is_private'] or not metadata['enable_gpu']:raise RuntimeError('Private GPU metadata guard failed')
            response=api.kernels_push(str(folder),timeout='1200',acc='NvidiaTeslaT4',no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                from audit_v5_source import _redact
                report['launch_rejection']={'error':_redact(str(response.error))[:1000],'invalid_dataset_sources':response.invalidDatasetSources,'invalid_competition_sources':response.invalidCompetitionSources}
                raise RuntimeError('Kaggle rejected required inference inputs or launch; sanitized details recorded')
            version=int(response.versionNumber)
            if version<1:raise RuntimeError('Missing immutable execution version')
            reference=metadata['id']+'/v'+str(version);report['version']=version
            print(f'::notice title=Kaggle inference accepted::Private resolution experiment {reference}; no training; server timeout 1200 seconds')
            deadline=time.monotonic()+1260;state='UNKNOWN'
            while time.monotonic()<deadline:
                state=str(api.kernels_status(reference).status).split('.')[-1].upper()
                if state in {'COMPLETE','ERROR','FAILED','CANCELLED','CANCELED'}:break
                time.sleep(15)
            report['status']=state
            import re
            api.kernels_output(reference,str(folder/'output'),file_pattern=re.escape(result_name)+'$')
            files=list((folder/'output').rglob(result_name))
            if len(files)==1:report['result']=json.loads(files[0].read_text())
            if state!='COMPLETE' or report.get('result',{}).get('status')!='completed':raise RuntimeError('Inference experiment did not complete successfully')
    except Exception as error:
        report['error_type']=type(error).__name__
        report['http_status']=getattr(getattr(error,'response',None),'status_code',None)
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)[:500]
        if reference:
            try:report['execution_errors']=error_context(api,reference)
            except Exception:report['execution_error_detail']='Log diagnostics unavailable'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if report.get('status')!='COMPLETE' or report.get('result',{}).get('status')!='completed' or 'error_type' in report:raise SystemExit(1)


if __name__=='__main__':main()
