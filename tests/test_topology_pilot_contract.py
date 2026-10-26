"""CPU-only safety and provenance tests for the independent topology pilot."""
import copy
import hashlib
from unittest.mock import Mock, patch

import pytest

from solar.auto14_plan import UPSTREAM_SHA, choose_experiment
from scripts import launch_topology_pilot as pilot


def complete_record():
    config = choose_experiment(pilot.DAY)
    return {
        "status": "COMPLETE",
        "day": pilot.DAY,
        "config": config,
        "official_submission": False,
        "test_inference": False,
        "confirmation_evaluated": False,
        "split_months_disjoint": True,
        "boundary_weight": config["boundary_weight"],
        "topology_weight": config["topology_weight"],
        "source_patches_sha256": "a" * 64,
        "best_checkpoint_sha256": "b" * 64,
        "source_sha": UPSTREAM_SHA,
        "local_tuning_pq": 0.42,
        "split_counts": {"train": 489, "val": 111, "confirmation": 107},
    }


def test_complete_pilot_is_local_evidence_only():
    record = complete_record()
    evidence = pilot.validate_result(record, record["config"])
    assert evidence["local_tuning_pq"] == pytest.approx(0.42)
    assert evidence["official_public_pq"] is None
    assert evidence["official_submission"] is False
    assert evidence["better_than_day2_local_pq"] is True


@pytest.mark.parametrize("change", [
    {"status": "RUNNING"},
    {"day": 6},
    {"official_submission": True},
    {"test_inference": True},
    {"confirmation_evaluated": True},
    {"split_months_disjoint": False},
    {"boundary_weight": 0},
    {"topology_weight": 0},
    {"source_patches_sha256": "missing"},
    {"best_checkpoint_sha256": "missing"},
    {"source_sha": "wrong"},
    {"local_tuning_pq": float("nan")},
    {"local_tuning_pq": 1.5},
    {"split_counts": {"train": 1, "val": 111, "confirmation": 107}},
])
def test_rejects_unproven_or_leaky_training(change):
    record = complete_record()
    record.update(change)
    with pytest.raises(ValueError):
        pilot.validate_result(record, complete_record()["config"])


def test_refuse_second_upload_when_push_was_uncertain(tmp_path):
    report = {"push_attempted": True}
    with patch.object(pilot, "REPORT", tmp_path / "pilot.json"), \
         patch.object(pilot, "recover_existing", return_value=None), \
         patch.object(pilot, "gpu_hours") as quota:
        api = Mock()
        with pytest.raises(RuntimeError, match="refuse duplicate launch"):
            pilot.launch_once(api, report, choose_experiment(pilot.DAY))
        api.kernels_push.assert_not_called()
        quota.assert_not_called()


def test_existing_private_kernel_is_recovered_not_uploaded(tmp_path):
    report = {}
    with patch.object(pilot, "REPORT", tmp_path / "pilot.json"), \
         patch.object(pilot, "recover_existing", return_value="RUNNING"), \
         patch.object(pilot, "gpu_hours") as quota:
        api = Mock()
        ref = pilot.launch_once(api, report, choose_experiment(pilot.DAY))
        assert ref == pilot.SLUG
        assert report["resumed_existing"] is True
        api.kernels_push.assert_not_called()
        quota.assert_not_called()


def test_quota_floor_prevents_training(tmp_path):
    report = {}
    with patch.object(pilot, "REPORT", tmp_path / "pilot.json"), \
         patch.object(pilot, "recover_existing", return_value=None), \
         patch.object(pilot, "gpu_hours", return_value=1.0):
        api = Mock()
        assert pilot.launch_once(api, report, choose_experiment(pilot.DAY)) == ""
        assert report["status"] == "SKIPPED_QUOTA"
        api.kernels_push.assert_not_called()
