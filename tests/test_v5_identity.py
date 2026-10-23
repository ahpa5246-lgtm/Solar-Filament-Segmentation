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
