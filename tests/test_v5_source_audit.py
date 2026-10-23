import importlib.util
from pathlib import Path


def test_source_audit_reports_public_metric_without_executing_notebook():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('source_audit',root/'scripts/audit_v5_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    v5={'cells':[{'cell_type':'code','source':['best_strategy = comparison["all_dice"].idxmax()\n'],
                  'outputs':[{'output_type':'stream','text':['credential KGAT_do_not_display\n','PQ=0.99\n']}]}]}
    official={'cells':[{'cell_type':'code','source':['def panoptic_quality(gt,pred):\n    raise RuntimeError("do not execute")\n']} ]}
    summary=module.summarize(v5,official)
    assert 'all_dice' in summary and 'def panoptic_quality' in summary
    assert 'KGAT_do_not_display' not in summary and 'PQ=0.99' not in summary
    assert 'No execution outputs' in summary
