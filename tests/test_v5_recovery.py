"""Pure selection and paused-training regression tests; no GPU required."""
import json
import ast
import hashlib
import sys
from pathlib import Path

from solar.v5_score_calibration import choose_threshold
from scripts import run_auto14_daily as runner


def test_threshold_selected_only_from_tuning_with_baseline_tie_break():
    assert choose_threshold({0.3: 0.4, 0.45: 0.4, 0.6: 0.4}) == 0.6
    assert choose_threshold({0.3: 0.42, 0.45: 0.41, 0.6: 0.40}) == 0.3


def test_nonfinite_or_incomplete_tuning_cannot_select():
    for value in ({}, {0.3: 0.5}, {0.3: float('nan'), 0.45: 0.4, 0.6: 0.4}):
        try:
            choose_threshold(value)
        except ValueError:
            continue
        raise AssertionError('Malformed tuning evidence accepted')


def test_paused_research_never_authenticates_or_launches(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    payload = {'status': 'paused_v5_recovery', 'results': {}}
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, 'day_index', lambda _: 4)
    def forbidden(*args):
        raise AssertionError('GPU execution must not be reached')
    monkeypatch.setattr(runner, 'execute', forbidden)
    runner.main(['--state', str(path)])
    assert json.loads(path.read_text()) == payload


def test_paused_arm_cannot_reach_final_candidate_selection():
    from scripts.run_auto14_final import candidate
    try:
        candidate({'status': 'paused_v5_recovery', 'results': {}})
    except RuntimeError as error:
        assert 'paused' in str(error)
    else:
        raise AssertionError('Paused arm reached final selection')


def test_read_only_recovery_retries_429_without_gpu(tmp_path, monkeypatch):
    from scripts import run_v6_confirmation as recovery
    from types import SimpleNamespace
    calls=[]
    class Limited(Exception):
        response=SimpleNamespace(status_code=429)
    class API:
        def kernels_output(self, reference, folder, file_pattern, page_token, page_size):
            calls.append(reference)
            if len(calls)==1: raise Limited()
            Path(folder, 'score_calibration.json').write_text(json.dumps({'status':'completed'}))
            return [], ''
    monkeypatch.setattr(recovery.time, 'sleep', lambda _: None)
    result=recovery.read_calibration_result(API(), 'owner/kernel/v2', tmp_path)
    assert result['status']=='completed'
    assert calls==['owner/kernel/v2', 'owner/kernel/v2']


def test_output_recovery_keeps_pagination_progress(tmp_path, monkeypatch):
    from scripts import run_v6_confirmation as recovery
    tokens=[]
    class API:
        def kernels_output(self, reference, folder, file_pattern, page_token, page_size):
            tokens.append(page_token)
            assert page_size==100
            if page_token=='': return [], 'page-two'
            Path(folder,'score_calibration.json').write_text(json.dumps({'status':'completed'}))
            return [], 'many-unused-package-pages'
    assert recovery.read_calibration_result(API(),'owner/kernel/v2',tmp_path)['status']=='completed'
    assert tokens==['','page-two']


def test_bootstrap_uses_py313_wheels_without_changing_frozen_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from build_v5_score_calibration import build
    target = tmp_path / 'audit.ipynb'
    build(target)
    code = ''.join(json.loads(target.read_text())['cells'][1]['source'])
    tree = ast.parse(code)
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {'SOURCE', 'EXPECTED_SHA', 'PACKAGES'}}
    assert 'numpy==2.2.6' in values['PACKAGES']
    assert 'scipy==1.15.3' in values['PACKAGES']
    assert hashlib.sha256(values['SOURCE'].encode()).hexdigest() == values['EXPECTED_SHA']
