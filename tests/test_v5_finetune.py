import math
import pytest
import ast
import hashlib
import json
from pathlib import Path
from solar.v5_finetune import training_allowed, better_checkpoint, validate_partitions


def test_running_or_ambiguous_training_is_never_duplicated():
    assert training_allowed({})
    for state in ({'push_attempted': True}, {'version': 1}, {'status': 'COMPLETE'}):
        assert not training_allowed(state)


def test_checkpoint_selection_requires_finite_strict_pq_improvement():
    assert better_checkpoint(.42, .40)
    assert not better_checkpoint(.40, .40)
    for value in (float('nan'), float('inf'), -1, 2):
        with pytest.raises(ValueError): better_checkpoint(value, .40)


def test_old_fitting_split_and_validation_dates_stay_separate():
    validate_partitions(['20200101a'], ['20200201a'], ['20200301a'])
    for parts in ((['20200101a'], ['20200101b'], ['20200301a']),
                  (['20200101a'], [], ['20200301a'])):
        with pytest.raises(ValueError): validate_partitions(*parts)


def test_notebook_keeps_frozen_source_and_offline_v5_checkpoints(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from build_v5_finetune import build
    out=tmp_path/'train.ipynb';meta=build(out)
    code=''.join(json.loads(out.read_text())['cells'][1]['source'])
    assignments={n.targets[0].id:n.value for n in ast.parse(code).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    source=ast.literal_eval(assignments['SOURCE'])
    assert hashlib.sha256(source.encode()).hexdigest()==ast.literal_eval(assignments['EXPECTED_SHA'])
    assert 'best_detector.pt' in source and 'best_refiner.pt' in source
    assert 'pandas==2.2.3' in ast.literal_eval(assignments['PACKAGES'])
    assert meta['is_private'] and meta['enable_gpu'] and not meta['kernel_sources']
    assert 'competition_submit' not in source
    assert '/tmp/solar_v5_finetune_packages' in code
