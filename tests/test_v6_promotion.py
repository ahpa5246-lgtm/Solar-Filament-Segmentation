"""Evidence-based promotion and no-submission regression tests."""
import copy
import json
from pathlib import Path

import pytest

from solar.v6_promotion import promotion_decision
from scripts.run_v6_submission import gate
from scripts.build_v6_submission_notebook import build

REPORT = Path(__file__).resolve().parents[1] / 'reports/V6_KAGGLE_CONFIRMATION_EXECUTION.json'


def evidence():
    return json.loads(REPORT.read_text())


def test_actual_kaggle_confirmation_fails_closed():
    report = evidence()
    verdict = promotion_decision(report)
    assert report['status'] == 'COMPLETE'
    assert report['result']['confirmation_observations'] == 51
    assert report['result']['paired_macro_delta_ci']['low'] < 0
    assert verdict['promote'] is False
    assert any('confidence interval' in item for item in verdict['reasons'])
    assert not gate(report)


def test_notebook_builder_refuses_failed_promotion(tmp_path):
    with pytest.raises(RuntimeError, match='confirmation'):
        build(tmp_path / 'v6_submission.ipynb')
    assert not (tmp_path / 'v6_submission.ipynb').exists()


def supported_synthetic_evidence():
    """Positive fixture tests programming logic, NOT model performance."""
    result = evidence()
    result['result']['paired_macro_delta_ci'] = {
        'low': 0.001, 'estimate': 0.006, 'high': 0.012, 'replicates': 2000
    }
    result['result']['confirmation_gain_supported'] = True
    return result


def test_synthetic_supported_record_can_pass():
    report = supported_synthetic_evidence()
    assert promotion_decision(report)['promote']
    assert gate(report)


@pytest.mark.parametrize('change', [
    lambda r: r.update(status='RUNNING'),
    lambda r: r['result'].update(fixed_nms_confirmation=False),
    lambda r: r['result'].update(pilot_observations_per_subset=16),
    lambda r: r['result'].update(confirmation_observations=16),
    lambda r: r['result'].update(pilot_confirmation_excluded=0),
    lambda r: r['result'].update(processed_observations=80),
    lambda r: r['result'].update(detector_cache_parity_min=0.8),
    lambda r: r['result'].update(baseline='2048|union'),
    lambda r: r['result']['confirmation_candidate'].update(reference_micro_pq=0.3),
    lambda r: r['result']['confirmation_candidate'].update(physical_observations=50),
    lambda r: r['result']['paired_macro_delta_ci'].update(low=0.0),
    lambda r: r['result']['paired_macro_delta_ci'].update(estimate=float('nan')),
    lambda r: r['result']['paired_macro_delta_ci'].update(replicates=20),
    lambda r: r.update(training=True),
])
def test_synthetic_corruptions_block_promotion(change):
    report = copy.deepcopy(supported_synthetic_evidence())
    change(report)
    assert not promotion_decision(report)['promote']
    assert not gate(report)


def test_missing_report_never_promotes():
    for payload in (None, {}, {'status': 'COMPLETE', 'result': {}}):
        assert not promotion_decision(payload)['promote']
