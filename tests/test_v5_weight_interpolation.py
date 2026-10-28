import pytest
import torch
from solar.v5_weight_interpolation import ALPHAS, interpolate_states


def test_interpolation_endpoints_and_midpoint():
    a = {"weight": torch.tensor([2.0, 4.0]), "count": torch.tensor(5)}
    b = {"weight": torch.tensor([6.0, 8.0]), "count": torch.tensor(5)}
    assert torch.equal(interpolate_states(a, b, 0)["weight"], a["weight"])
    assert torch.equal(interpolate_states(a, b, 1)["weight"], b["weight"])
    assert torch.equal(interpolate_states(a, b, 0.5)["weight"], torch.tensor([4.0, 6.0]))
    assert torch.equal(interpolate_states(a, b, 0.5)["count"], a["count"])


def test_rejects_architecture_drift_and_bad_values():
    a = {"w": torch.tensor([1.0]), "counter": torch.tensor(2)}
    for b in ({"other": torch.tensor([1.0])},
              {"w": torch.tensor([1.0, 2.0]), "counter": torch.tensor(2)},
              {"w": torch.tensor([1.0]), "counter": torch.tensor(3)},
              {"w": torch.tensor([float("nan")]), "counter": torch.tensor(2)}):
        with pytest.raises((ValueError, TypeError)):
            interpolate_states(a, b, 0.2)
    for alpha in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            interpolate_states(a, a, alpha)


def test_grid_is_predeclared_and_baseline_included():
    assert ALPHAS[0] == 0.0
    assert ALPHAS[-1] == 1.0
    assert len(ALPHAS) == len(set(ALPHAS))


def test_batchnorm_tracking_counters_may_drift_but_are_not_interpolated():
    base = {"backbone.body.bn1.num_batches_tracked": torch.tensor(125, dtype=torch.int64),
            "backbone.body.bn1.running_mean": torch.tensor([1.0])}
    trained = {"backbone.body.bn1.num_batches_tracked": torch.tensor(240, dtype=torch.int64),
               "backbone.body.bn1.running_mean": torch.tensor([3.0])}
    candidate = interpolate_states(base, trained, 0.25)
    assert candidate["backbone.body.bn1.num_batches_tracked"].item() == 125
    assert torch.allclose(candidate["backbone.body.bn1.running_mean"], torch.tensor([1.5]))
