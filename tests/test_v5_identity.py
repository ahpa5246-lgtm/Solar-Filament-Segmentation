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
    assert metadata['kernel_sources']==['minahussain111/solar-filament-research-v5']
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
