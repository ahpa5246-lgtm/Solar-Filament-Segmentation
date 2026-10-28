"""Launch once, publish immutable identity, then monitor the same real training run."""
import argparse, json, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solar.v5_finetune import training_allowed
from build_v5_finetune import build,KERNEL

STATE=ROOT/'reports/V5_FINETUNE_STATE.json'


def save(value):
    STATE.parent.mkdir(parents=True,exist_ok=True)
    pending=STATE.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(STATE)


def retrieve(api,ref,folder):
    token=''
    for page in range(60):
        for attempt,delay in enumerate((0,30,60,120)):
            if delay:time.sleep(delay)
            try:
                _,following=api.kernels_output(ref,str(folder),file_pattern=r'finetune_result\.json$',
                                               page_token=token,page_size=100)
                break
            except Exception as error:
                if getattr(getattr(error,'response',None),'status_code',None)!=429 or attempt==3:raise
        files=list(Path(folder).rglob('finetune_result.json'))
        if files:return json.loads(files[0].read_text())
        if not following:break
        token=following
    raise RuntimeError('Completed training result not available; never relaunch on read failure')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--monitor',action='store_true')
    args=parser.parse_args()
    if args.launch==args.monitor:parser.error('Select exactly one phase')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.launch:
        if not training_allowed(state):
            print('Existing or ambiguous V5 launch retained; no duplicate:',state.get('version'),flush=True);return
        if state.get('retry_of'):
            old_status=str(api.kernels_status(state['retry_of']).status).split('.')[-1].upper()
            if old_status not in {'ERROR','FAILED','CANCELED','CANCELLED'}:
                raise RuntimeError('Previous attempt is not terminal; retry forbidden')
        quota=api.quota_view().gpu_quota
        remaining=(quota.total_time_allowed-quota.time_used).total_seconds()/3600
        if remaining<4.25:raise RuntimeError('Insufficient GPU quota for bounded continuation')
        state=dict(status='PREPARING',kernel=KERNEL,training=True,official_submission=False,
                   target_official_pq=.50,gpu_hours_remaining_before=remaining)
        with tempfile.TemporaryDirectory(prefix='v5-training-build-') as temp:
            meta=build(Path(temp)/'v5_finetune.ipynb')
            assert meta['is_private'] and meta['enable_gpu'] and meta['id']==KERNEL
            state.update(push_attempted=True,status='LAUNCH_REQUESTED');save(state)
            response=api.kernels_push(temp,timeout='7200',acc='NvidiaTeslaT4',no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError('Kaggle rejected V5 training inputs; intent preserved for investigation')
            version=int(response.versionNumber)
            if version<1:raise RuntimeError('No immutable V5 training version returned')
            state.update(version=version,status='ACCEPTED',reference=KERNEL+'/v'+str(version));save(state)
            print('V5_TRAINING_ACCEPTED',json.dumps(state),flush=True)
        return
    if not state.get('version'):raise RuntimeError('No known training version; monitor cannot launch')
    ref=state['reference']
    deadline=time.monotonic()+7500
    while time.monotonic()<deadline:
        response=api.kernels_status(ref)
        status=str(response.status).split('.')[-1].upper()
        state['kaggle_status']=status;save(state)
        print('V5_TRAINING_STATUS',ref,status,flush=True)
        if status in {'COMPLETE','ERROR','FAILED','CANCELED','CANCELLED'}:break
        time.sleep(60)
    if state.get('kaggle_status')!='COMPLETE':
        state['status']='NEEDS_DIAGNOSIS';save(state)
        try:
            raw=api.kernels_logs(ref)
            events=json.loads(raw) if isinstance(raw,str) else raw
            text='\n'.join(str(e.get('data','')) for e in events if isinstance(e,dict))
            # Only controlled exception summaries; never signed URLs, credentials or environment dumps.
            lines=[line[:500] for line in text.splitlines() if line.startswith(('RuntimeError:','ValueError:','ImportError:','ModuleNotFoundError:','KeyError:','AssertionError:'))]
            state['execution_errors']=lines[-8:];save(state)
        except Exception:pass
        raise RuntimeError('V5 training not complete; inspect existing version, do not duplicate')
    with tempfile.TemporaryDirectory(prefix='v5-training-read-') as temp:
        state['result']=retrieve(api,ref,temp)
    if state['result'].get('status')!='COMPLETE':
        state['status']='INCOMPLETE';save(state)
        raise RuntimeError('Training result did not pass completion checks')
    state['status']='COMPLETE';save(state)
    print('V5_TRAINING_COMPLETE',json.dumps(state),flush=True)


if __name__=='__main__':main()
