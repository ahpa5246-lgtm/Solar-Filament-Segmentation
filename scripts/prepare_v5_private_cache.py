"""Create one private, immutable V5/v3 input cache for reproducible Kaggle runs."""
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

CACHE_ID='minahussain111/solar-filament-v5-v3-cache-356bd788'
FILES={'training_manifest.csv','best_detector_predictions_1024.pkl','refined_validation_predictions_2048.pkl',
       'best_detector.pt','best_refiner.pt','detector_thresholds.json','final_configuration.json'}
MARKER='V5_SOURCE_SESSION=v3; MANIFEST_SHA256=356bd788d677819a81922f132b0d65b6a709e773ac0f4f8f2a28a2369e014f21;'


def metadata():
    return {'id':CACHE_ID,'title':'Solar Filament V5 v3 private audit cache','isPrivate':True,
            'licenses':[{'name':'CC-BY-NC-4.0'}],
            'description':MARKER+' Private cached predictions and own trained checkpoints from official competition TRAIN only. No source images, raw labels, test predictions or credentials. MAGFiLO/GONG data attribution applies. Original model notebook attribution: Nomannic, Apache-2.0; PyTorch/torchvision BSD.'}


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from replay_v5_cached import V3_ARTIFACT_SHA256,verify_artifacts
    api=KaggleApi();api.authenticate()
    report={'dataset':CACHE_ID,'requested_private':True,'source_session':'minahussain111/solar-filament-research-v5/v3',
            'gpu_execution':False,'weights_budget_bytes':600*1024*1024}
    try:
        with tempfile.TemporaryDirectory(prefix='v5-private-cache-') as tmp:
            root=Path(tmp)
            exists=False
            report['stage']='discover_owned_cache'
            matches=api.dataset_list(mine=True,search=metadata()['title']) or []
            refs=[getattr(item,'ref','') for item in matches if getattr(item,'title','')==metadata()['title']]
            report['matching_owned_refs']=refs
            if refs and CACHE_ID not in refs:raise RuntimeError('Owned cache has a different reference; inspect returned refs before reuse')
            exists=CACHE_ID in refs
            report['stage']='check_target'
            try:api.dataset_status(CACHE_ID);exists=True
            except Exception as error:
                status=getattr(getattr(error,'response',None),'status_code',None)
                report['initial_status_http']=status
                # Kaggle can return 403 for a nonexistent/private dataset. This
                # does not prove creation is denied. The official create API
                # checks existence itself and cannot overwrite a dataset.
                if status not in (403,404):raise
            if not exists:
                report['stage']='download_whitelisted_artifacts'
                pattern=r'(^|/)('+ '|'.join(re.escape(name) for name in sorted(FILES))+r')$'
                api.kernels_output('minahussain111/solar-filament-research-v5/v3',str(root/'download'),file_pattern=pattern,page_size=100)
                stage=root/'stage';stage.mkdir()
                for name in sorted(FILES):
                    found=list((root/'download').rglob(name))
                    if len(found)!=1:raise ValueError('Required artifact missing or ambiguous: '+name)
                    shutil.copyfile(found[0],stage/name)
                total=sum(path.stat().st_size for path in stage.iterdir())
                if total>report['weights_budget_bytes']:raise ValueError('Artifact cache exceeds 600 MiB budget')
                verify_artifacts({name:stage/name for name in V3_ARTIFACT_SHA256},V3_ARTIFACT_SHA256)
                fingerprints={name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in sorted(FILES)}
                report['artifact_sha256']=fingerprints;report['bytes']=total
                (stage/'provenance.json').write_text(json.dumps({'source_session':report['source_session'],'sha256':fingerprints},indent=2))
                (stage/'dataset-metadata.json').write_text(json.dumps(metadata(),indent=2))
                report['stage']='create_private_dataset'
                response=api.dataset_create_new(str(stage),public=False,quiet=True,convert_to_csv=False)
                report['creation_status']=str(response.status)
                from urllib.parse import urlsplit
                url=urlsplit(str(response.url or ''))
                if url.hostname in {'www.kaggle.com','kaggle.com'}:report['creation_url']=url.scheme+'://'+url.hostname+url.path
                if response.error or str(response.status).lower()!='ok':raise RuntimeError('Private dataset creation was rejected')
                report['created']=True
            else:report['created']=False
            report['stage']='wait_for_cache'
            deadline=time.monotonic()+300
            state='unknown'
            while time.monotonic()<deadline:
                try:
                    state=api.dataset_status(CACHE_ID)
                except Exception as error:
                    status=getattr(getattr(error,'response',None),'status_code',None)
                    if status not in (403,404):raise
                    report['last_processing_status_http']=status
                    time.sleep(10)
                    continue
                if str(state).lower()=='ready':break
                if str(state).lower() in {'error','failed'}:raise RuntimeError('Private cache processing failed')
                time.sleep(10)
            if str(state).lower()!='ready':raise RuntimeError('Private cache did not become ready within budget')
            report['stage']='verify_privacy_and_provenance'
            meta_path=api.dataset_metadata(CACHE_ID,str(root/'remote-metadata'))
            remote=json.loads(Path(meta_path).read_text())
            private=remote.get('isPrivate',remote.get('is_private'))
            if private not in [True,'true','True']:raise RuntimeError('Private status could not be verified; refusing use')
            description=remote.get('description','')
            if MARKER not in description:raise RuntimeError('Existing dataset provenance differs; refusing reuse')
            report['private_verified']=True;report['status']='ready'
    except Exception as error:
        report['status']='blocked';report['error_type']=type(error).__name__
        report['http_status']=getattr(getattr(error,'response',None),'status_code',None)
        if type(error) in (ValueError,RuntimeError):report['diagnosis']=str(error)
        try:
            import os
            message=getattr(error,'response',None).json().get('message','')
            token=os.environ.get('KAGGLE_API_TOKEN','')
            if token:message=message.replace(token,'[REDACTED]')
            message=re.sub(r'KGAT_[A-Za-z0-9_-]+','[REDACTED]',message)
            message=re.sub(r'https?://\S+','[URL omitted]',message)
            if message:report['api_message']=message[:500]
        except Exception:pass
    Path('reports/V5_PRIVATE_CACHE.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='ready':raise SystemExit(1)
    print('Private V5 cache is ready; privacy and source fingerprints verified.')


if __name__=='__main__':main()
