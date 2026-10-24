"""Execute one bounded CPU Kaggle replay; never request an accelerator."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def error_context(api,reference):
    import re
    from audit_v5_source import _redact
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner,slug,*version=reference.split('/')
    request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
    if version:request.version_label=version[0]
    with api.build_kaggle_client() as client:
        raw=client.kernels.kernels_api_client.list_kernel_session_output(request).log
    events=json.loads(raw or '[]')
    text='\n'.join(event.get('data','') for event in events)
    return [line[:500] for line in _redact(text).splitlines() if re.search(r'(RuntimeError:|ModuleNotFoundError:|ImportError:|AssertionError:|FileNotFoundError:|KeyError:)',line)][-8:]


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    report={'requested_gpu':False,'requested_private':True,'timeout_seconds':600,
            'notebook':'https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-audit'}
    inspect_only='--inspect-only' in sys.argv
    report_path=Path('reports/V6_KAGGLE_CPU_STATUS.json' if inspect_only else 'reports/V6_KAGGLE_CPU_EXECUTION.json')
    report['mode']='inspect_existing_only' if inspect_only else 'execute_one_cpu_notebook'
    reference=None
    try:
        if inspect_only:
            reference='minahussain111/solar-filament-research-v6-audit'
            report['status']=str(api.kernels_status(reference).status)
            report['execution_errors']=error_context(api,reference)
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            print('Inspected existing CPU run; no new notebook executed.')
            return
        with tempfile.TemporaryDirectory(prefix='v6-cpu-') as tmp:
            folder=Path(tmp)
            subprocess.run([sys.executable,'scripts/build_v6_replay_notebook.py','--output',str(folder/'v6_cpu_audit.ipynb')],check=True)
            metadata=json.loads((folder/'kernel-metadata.json').read_text())
            if metadata['enable_gpu'] or not metadata['is_private']:raise ValueError('CPU/private guard failed')
            response=api.kernels_push(str(folder),timeout='600',no_run=False)
            if response.error:raise RuntimeError('Kaggle rejected notebook push')
            if response.invalidKernelSources or response.invalidCompetitionSources:raise RuntimeError('Kaggle rejected required notebook inputs')
            version=int(response.versionNumber)
            if version<1:raise RuntimeError('Missing immutable execution version')
            report['version']=version;reference=metadata['id']+'/v'+str(version)
            deadline=time.monotonic()+660
            state='UNKNOWN'
            while time.monotonic()<deadline:
                state=str(api.kernels_status(reference).status).split('.')[-1].upper()
                if state in {'COMPLETE','ERROR','FAILED','CANCELLED','CANCELED'}:break
                time.sleep(15)
            report['status']=state
            if state!='COMPLETE':raise RuntimeError('CPU audit did not complete successfully')
            api.kernels_output(reference,str(folder/'output'),file_pattern=r'v6_cpu_replay\.json$')
            files=list((folder/'output').rglob('v6_cpu_replay.json'))
            if len(files)!=1:raise RuntimeError('Missing fresh replay result')
            report['result']=json.loads(files[0].read_text())
    except Exception as error:
        report['error_type']=type(error).__name__
        report['http_status']=getattr(getattr(error,'response',None),'status_code',None)
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)
        if reference:
            try:
                report['execution_errors']=error_context(api,reference)
            except Exception:
                report['execution_error_detail']='Log diagnostics unavailable'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if 'result' not in report:raise SystemExit(1)
    print('Kaggle CPU replay completed; no GPU quota consumed by this job.')


if __name__=='__main__':main()
