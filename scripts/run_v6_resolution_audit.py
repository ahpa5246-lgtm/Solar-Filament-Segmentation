"""Launch at most one bounded inference-only Kaggle experiment after CPU readiness."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from run_v6_cpu_audit import error_context
    api=KaggleApi();api.authenticate()
    report={'training':False,'test_inference':False,'official_submission':False,
            'requested_accelerator':'NvidiaTeslaT4','server_timeout_seconds':1200,
            'notebook':'https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-resolution'}
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
            subprocess.run([sys.executable,'scripts/build_v6_resolution_notebook.py','--output',str(folder/'v6_resolution_audit.ipynb')],check=True)
            metadata=json.loads((folder/'kernel-metadata.json').read_text())
            if not metadata['is_private'] or not metadata['enable_gpu']:raise RuntimeError('Private GPU metadata guard failed')
            response=api.kernels_push(str(folder),timeout='1200',acc='NvidiaTeslaT4',no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:raise RuntimeError('Kaggle rejected required inference inputs or launch')
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
            api.kernels_output(reference,str(folder/'output'),file_pattern=r'resolution_audit\.json$')
            files=list((folder/'output').rglob('resolution_audit.json'))
            if len(files)==1:report['result']=json.loads(files[0].read_text())
            if state!='COMPLETE' or report.get('result',{}).get('status')!='completed':raise RuntimeError('Inference experiment did not complete successfully')
    except Exception as error:
        report['error_type']=type(error).__name__
        report['http_status']=getattr(getattr(error,'response',None),'status_code',None)
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)[:500]
        if reference:
            try:report['execution_errors']=error_context(api,reference)
            except Exception:report['execution_error_detail']='Log diagnostics unavailable'
    Path('reports/V6_KAGGLE_RESOLUTION_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n')
    if report.get('status')!='COMPLETE' or report.get('result',{}).get('status')!='completed' or 'error_type' in report:raise SystemExit(1)


if __name__=='__main__':main()
