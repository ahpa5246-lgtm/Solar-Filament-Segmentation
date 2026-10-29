import ast
import hashlib
import json
from pathlib import Path

import pytest

from solar.v5_hardnegative import sampling_weights
from solar.v5_finetune import validate_partitions


def test_train_only_sampler_inventory_and_bounded_boost():
    stems = ["20200101a", "20200102a", "20200103a"]
    assert sampling_weights(stems, dict(zip(stems, [0, 2, 999]))) == [1., 2.5, 4.]
    assert sampling_weights(stems, dict(zip(stems, [0, 2, 999]))) == [1., 2.5, 4.]
    with pytest.raises(ValueError):
        sampling_weights(stems, {**dict(zip(stems, [0, 2, 3])), "20200401_validation": 10})
    with pytest.raises(ValueError):
        sampling_weights(stems, {stems[0]: 1})
    with pytest.raises(ValueError):
        sampling_weights(["duplicate", "duplicate"], {"duplicate": 1})
    with pytest.raises(ValueError):
        sampling_weights(stems, dict(zip(stems, [0, -1, 1])))


def test_training_dates_remain_separate():
    validate_partitions(["20200101a"], ["20200201a"], ["20200301a"])
    with pytest.raises(ValueError):
        validate_partitions(["20200101a"], ["20200101b"], ["20200301a"])


def test_hardnegative_notebook_is_private_isolated_and_has_no_submission(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_v5_hardnegative import build, KERNEL
    out = tmp_path / "hardnegative.ipynb"
    meta = build(out)
    notebook = json.loads(out.read_text())
    code = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(code)
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    driver = ast.literal_eval(assignments["SOURCE"])
    assert hashlib.sha256(driver.encode()).hexdigest() == ast.literal_eval(assignments["EXPECTED_SHA"])
    assert "hard_negative=True" in driver
    assert "solar/v5_hardnegative.py" in driver
    assert "best_detector.pt" in driver and "best_refiner.pt" in driver
    assert "competition_submit" not in driver
    assert meta["id"] == KERNEL and meta["is_private"] and meta["enable_gpu"]
    assert meta["kernel_sources"] == []
    assert meta["dataset_sources"]


def test_new_experiment_requires_two_completed_no_gain_precursors():
    from scripts.run_v5_hardnegative import prerequisites
    training = {"status": "COMPLETE", "version": 2,
                "result": {"status": "COMPLETE", "epochs_completed": 16,
                           "best_epoch": 0, "local_gain_supported": False}}
    interpolation = {"status": "COMPLETE", "version": 3,
                     "result": {"status": "COMPLETE", "selected_alpha": 0,
                                "local_gain_supported": False}}
    prerequisites(training, interpolation)
    interpolation["result"]["selected_alpha"] = 0.05
    with pytest.raises(ValueError):
        prerequisites(training, interpolation)


def test_hardnegative_launch_fails_closed_on_existing_or_uncertain_kernel():
    from types import SimpleNamespace
    from scripts.run_v5_hardnegative import refuse_untracked_kernel
    known = ("minahussain111/solar-v5-finetune-pq50-20261028/v2",
             "minahussain111/solar-v5-weight-interpolation-pq50/v3")
    base = [ref.rsplit("/v", 1)[0] for ref in known]
    target = "minahussain111/solar-v5-hardnegative-pq50"

    class FakeApi:
        def __init__(self, status, pages=None):
            self.status = status
            self.pages = pages if pages is not None else [base]
            self.calls = 0
        def kernels_status(self, slug):
            assert slug == target
            if self.status == 200:
                return object()
            if self.status == "wrapped403":
                raise ValueError("Cannot access kernel (Permission 'kernels.get' was denied)")
            error = RuntimeError("API response")
            error.status = self.status
            raise error
        def kernels_list_with_response(self, *, mine, page_size, page_token):
            assert mine is True and page_size == 100
            assert page_token == (str(self.calls) if self.calls else None)
            page = self.pages[self.calls]
            self.calls += 1
            return SimpleNamespace(
                kernels=[SimpleNamespace(ref=ref) for ref in page],
                next_page_token=str(self.calls) if self.calls < len(self.pages) else "")

    # Both a literal 404 and the CLI's wrapped 403 require full owner inventory.
    for status in (404, 403, "wrapped403"):
        api = FakeApi(status, [base[:1], base[1:]])
        refuse_untracked_kernel(api, known)
        assert api.calls == 2
    for status in (200, 429, 500):
        with pytest.raises(RuntimeError):
            refuse_untracked_kernel(FakeApi(status), known)
    with pytest.raises(RuntimeError, match="Untracked Kaggle"):
        refuse_untracked_kernel(FakeApi("wrapped403", [base + [target]]), known)
    with pytest.raises(RuntimeError, match="Cannot verify Kaggle account"):
        refuse_untracked_kernel(FakeApi(403, [base[:1]]), known)
    with pytest.raises(RuntimeError, match="Missing or invalid"):
        refuse_untracked_kernel(FakeApi(404), known[:1])
    class BrokenInventory(FakeApi):
        def kernels_list_with_response(self, **kwargs):
            raise RuntimeError("profile API 429")
    with pytest.raises(RuntimeError, match="profile API 429"):
        refuse_untracked_kernel(BrokenInventory(403), known)
