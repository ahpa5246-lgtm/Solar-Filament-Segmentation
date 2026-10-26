"""No Kaggle access: verify frozen day-two identity and recovery safety gates."""
import ast
import json
from pathlib import Path

import pytest

from scripts.build_day2_verification_notebook import build_day2, CHECKPOINT
from scripts.build_auto14_final_notebook import DAY2_VERIFICATION_KERNEL
from scripts.run_day2_independent import preflight
from solar.auto14_plan import UPSTREAM_SHA


def source_state():
    return {"results": {"2": {
        "day": 2, "status": "COMPLETE", "execution_status": "COMPLETE",
        "checkpoint_sha256": CHECKPOINT, "source_sha": UPSTREAM_SHA,
        "local_tuning_pq": 0.40438, "confirmation_evaluated": False,
        "official_submission": False, "split_counts": {
            "train": 489, "val": 111, "confirmation": 107
        },
    }}}


def previous():
    return {"kaggle_execution_status": "ERROR", "official_submission": False}


def diagnosis():
    return {"execution_status": "ERROR", "result_manifest_count": 1,
            "result_manifest": {"status": "FAILED", "error_type": "ModuleNotFoundError",
                                "confirmation_local_pq": None,
                                "confirmation_observations": None,
                                "official_submission": False},
            "logs": {"missing_modules": ["numpy.rec"]}}


def test_reuses_verified_day_two_checkpoint_without_promotion():
    assert preflight(source_state(), previous(), diagnosis()) == CHECKPOINT
    bad = source_state()
    bad["results"]["2"]["checkpoint_sha256"] = "a" * 64
    with pytest.raises(RuntimeError): preflight(bad, previous(), diagnosis())
    bad = source_state()
    bad["results"]["2"]["confirmation_evaluated"] = True
    with pytest.raises(RuntimeError): preflight(bad, previous(), diagnosis())


@pytest.mark.parametrize("changed", [
    {"kaggle_execution_status": "COMPLETE"}, {"official_submission": True},
    {"official_submission_attempted": True}, {"submission_ref": 57571678},
])
def test_previously_unknown_or_completed_upload_is_unsafe(changed):
    with pytest.raises(RuntimeError):
        preflight(source_state(), {**previous(), **changed}, diagnosis())


@pytest.mark.parametrize("changed", [
    {"status": "COMPLETE"}, {"error_type": "OtherError"},
    {"confirmation_local_pq": 0.39}, {"confirmation_observations": 19},
    {"official_submission": True},
])
def test_partially_revealed_confirmation_refuses_second_evaluation(changed):
    diag = diagnosis()
    diag["result_manifest"] = {**diag["result_manifest"], **changed}
    with pytest.raises(RuntimeError):
        preflight(source_state(), previous(), diag)


def test_wrong_numpy_error_refuses_the_new_gpu_job():
    d = diagnosis()
    d["logs"]["missing_modules"] = ["torch"]
    with pytest.raises(RuntimeError):
        preflight(source_state(), previous(), d)


def test_isolated_kaggle_notebook_compiles_and_preserves_identity(tmp_path):
    dest = tmp_path / "day2.ipynb"
    meta = build_day2(2, CHECKPOINT, dest)
    assert meta["id"] == DAY2_VERIFICATION_KERNEL
    assert meta["kernel_sources"] == ["minahussain111/solar-autoresearch-d02"]
    assert meta["is_private"] and meta["enable_gpu"]
    assert meta["dataset_sources"] == []
    nb = json.loads(dest.read_text())
    code = "".join(nb["cells"][1]["source"])
    ast.parse(code)
    compile(code, "day-two isolated inference", "exec")
    assert "numpy.rec" in code
    assert "solar_day2_fixed_inference.py" in code
    assert "confirmation_local_pq" in code
    assert CHECKPOINT in code
    assert "competition_submit(" not in code
    with pytest.raises(ValueError): build_day2(3, CHECKPOINT, tmp_path / "wrongday.ipynb")
    with pytest.raises(ValueError): build_day2(2, "0" * 64, tmp_path / "wrong.pt")
