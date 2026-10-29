"""CPU-only guards for the controlled V5 ROI-head-only experiment."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from solar.v5_finetune import freeze_to_roi_heads


class FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return 1


class FakeModel:
    def __init__(self, names):
        self.params = {name: FakeParameter() for name in names}

    def named_parameters(self):
        return list(self.params.items())


def test_only_roi_heads_receive_gradients():
    model = FakeModel([
        "backbone.body.layer1.weight", "rpn.head.cls_logits.weight",
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.weight"])
    assert freeze_to_roi_heads(model) == 2
    assert not model.params["backbone.body.layer1.weight"].requires_grad
    assert not model.params["rpn.head.cls_logits.weight"].requires_grad
    assert model.params["roi_heads.box_predictor.cls_score.weight"].requires_grad
    assert model.params["roi_heads.mask_predictor.mask_fcn_logits.weight"].requires_grad
    with pytest.raises(ValueError, match="ROI heads"):
        freeze_to_roi_heads(FakeModel(["backbone.weight", "rpn.weight"]))


def test_private_notebook_uses_original_weights_and_headonly_driver(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from scripts.build_v5_headonly import build, KERNEL
    output = tmp_path / "headonly.ipynb"
    meta = build(output)
    notebook = json.loads(output.read_text())
    code = "".join(notebook["cells"][1]["source"])
    assignments = {node.targets[0].id: node.value for node in ast.parse(code).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)}
    source = ast.literal_eval(assignments["SOURCE"])
    assert hashlib.sha256(source.encode()).hexdigest() == ast.literal_eval(assignments["EXPECTED_SHA"])
    assert "head_only=True" in source and "hard_negative=True" not in source
    assert "best_detector.pt" in source and "best_refiner.pt" in source
    assert "competition_submit" not in source
    assert meta["id"] == KERNEL and meta["is_private"] and meta["enable_gpu"]
    assert meta["kernel_sources"] == []


def test_headonly_requires_completed_failed_hardnegative(tmp_path, monkeypatch):
    from scripts import run_v5_headonly as runner
    training = {"status": "COMPLETE", "version": 2,
                "result": {"status": "COMPLETE", "epochs_completed": 16,
                           "best_epoch": 0, "local_gain_supported": False,
                           "best_checkpoint_sha256": "baseline-checkpoint"}}
    interpolation = {"status": "COMPLETE", "version": 3,
                     "result": {"status": "COMPLETE", "selected_alpha": 0,
                                "local_gain_supported": False}}
    hard = {"status": "COMPLETE", "version": 1,
            "reference": "minahussain111/solar-v5-hardnegative-pq50/v1",
            "kaggle_status": "COMPLETE",
            "result": {"status": "COMPLETE", "epochs_completed": 4,
                       "best_epoch": 0, "local_gain_supported": False,
                       "best_checkpoint_sha256": "baseline-checkpoint"}}
    state = tmp_path / "hard.json"
    monkeypatch.setattr(runner, "HARD_STATE", state)
    state.write_text(json.dumps(hard))
    assert runner.prerequisites(training, interpolation) == hard["reference"]
    hard["result"]["best_epoch"] = 1
    state.write_text(json.dumps(hard))
    with pytest.raises(ValueError, match="no-gain"):
        runner.prerequisites(training, interpolation)
    hard["result"]["best_epoch"] = 0
    hard["result"]["best_checkpoint_sha256"] = "wrong"
    state.write_text(json.dumps(hard))
    with pytest.raises(ValueError, match="no-gain"):
        runner.prerequisites(training, interpolation)


def test_headonly_direct_cli_import():
    """Match the actual GitHub Actions command, not pytest's import path."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, 'scripts/run_v5_headonly.py', '--help'],
        cwd=root, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert '--phase' in result.stdout
