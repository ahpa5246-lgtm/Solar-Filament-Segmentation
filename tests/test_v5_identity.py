from types import SimpleNamespace
from scripts.inspect_v5_identity import public_metadata, source_summary


def test_identity_metadata_never_serializes_source_or_credentials():
    record=SimpleNamespace(id=123,ref='minahussain111/solar-filament-research-v5',version_number=5,
                           source='private source',api_token='never-serialize',title='solar-filament-research-v5')
    result=public_metadata(record)
    assert result['id']==123 and result['version_number']==5
    assert 'source' not in result and 'api_token' not in result


def test_source_summary_records_execution_presence_without_outputs():
    import json
    source=json.dumps({'cells':[{'cell_type':'code','execution_count':3,
                                'outputs':[{'text':['private validation output']}]}]})
    result=source_summary(source)
    assert result['executed_code_cells']==1 and result['cells_with_outputs']==1
    assert 'private validation output' not in str(result)


def test_cpu_audit_notebook_cannot_request_gpu(tmp_path):
    import ast,json
    from scripts.build_v6_replay_notebook import build
    build(tmp_path/'audit.ipynb')
    metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is False
    assert metadata['kernel_sources']==[]
    assert metadata['dataset_sources']==['minahussain111/solar-filament-v5-v3-cache-356bd788']
    notebook=json.loads((tmp_path/'audit.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))


def test_artifact_fingerprint_rejects_changed_cached_inputs(tmp_path):
    import hashlib
    import pytest
    from scripts.replay_v5_cached import verify_artifacts
    path=tmp_path/'cache.pkl';path.write_bytes(b'original')
    expected={'cache.pkl':hashlib.sha256(b'original').hexdigest()}
    assert verify_artifacts({'cache.pkl':path},expected)==expected
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):verify_artifacts({'cache.pkl':path},expected)


def test_private_cache_excludes_test_predictions_and_raw_labels():
    from scripts.prepare_v5_private_cache import metadata,FILES
    assert metadata()['isPrivate'] is True
    assert 'test_predictions_2048.pkl' not in FILES
    assert not any('Annotations' in name or 'crop_masks' in name for name in FILES)


def test_refiner_notebook_is_private_and_has_explicit_inference_budget(tmp_path):
    import ast,json
    from scripts.build_v6_refiner_notebook import build
    build(tmp_path/'refiner.ipynb')
    meta=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['enable_gpu'] is True
    nb=json.loads((tmp_path/'refiner.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    ast.parse(source)
    assert 'seconds=1000' in source


def test_resolution_notebook_mounts_both_exact_checkpoints(tmp_path):
    import ast,json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'pilot.ipynb'
    subprocess.run([sys.executable,str(root/'scripts/build_v6_resolution_notebook.py'),'--output',str(out)],check=True)
    notebook=json.loads(out.read_text());metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    code=''.join(notebook['cells'][1]['source']);ast.parse(code)
    assert metadata['id'].endswith('-resolution') and metadata['is_private'] and metadata['enable_gpu']
    assert "one('best_detector.pt'),one('best_refiner.pt')" in code
    assert 'from solar.v6_resolution_audit import run' in code
    workflow=(root/'.github/workflows/resolution-audit.yml').read_text()
    assert 'V6_KAGGLE_RESOLUTION_EXECUTION.json' in workflow
    assert 'V6_KAGGLE_REFINER_EXECUTION.json' not in workflow


def test_status_inspection_preserves_completed_cpu_execution_evidence(tmp_path,monkeypatch):
    import importlib.util,json,sys,types
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('cpu_runner_under_test',root/'scripts/run_v6_cpu_audit.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    class API:
        def authenticate(self):pass
        def kernels_status(self,reference):return types.SimpleNamespace(status='KernelWorkerStatus.COMPLETE')
    fake=types.ModuleType('kaggle.api.kaggle_api_extended');fake.KaggleApi=API
    for name in ['kaggle','kaggle.api']:monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    monkeypatch.setitem(sys.modules,'kaggle.api.kaggle_api_extended',fake)
    monkeypatch.setattr(runner,'error_context',lambda api,reference:[])
    monkeypatch.setattr(sys,'argv',['run_v6_cpu_audit.py','--inspect-only'])
    monkeypatch.chdir(tmp_path);(tmp_path/'reports').mkdir()
    result=tmp_path/'reports/V6_KAGGLE_CPU_EXECUTION.json'
    original=json.dumps({'version':2,'status':'COMPLETE','result':{'artifact_sha256':{'manifest':'fingerprint'}}})
    result.write_text(original)
    runner.main()
    assert result.read_text()==original
    assert json.loads((tmp_path/'reports/V6_KAGGLE_CPU_STATUS.json').read_text())['mode']=='inspect_existing_only'
