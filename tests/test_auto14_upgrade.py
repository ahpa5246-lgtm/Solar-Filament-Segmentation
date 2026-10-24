"""Real CPU numerical contracts for the pinned-source boundary-learning upgrade."""
from __future__ import annotations

import ast
import os
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from solar.auto14_plan import choose_experiment
from solar.auto14_upgrade import patch_auto14_sources
from scripts.build_auto14_notebook import build


TRAIN = '''import json
def train(stems, args):
    if args.all_data:
        train_stems, val_stems = stems, []
    else:
        train_stems, val_stems = split_stems(stems, args.val_frac, args.seed)
    return json.dumps({"train": train_stems, "val": val_stems}, indent=1)
'''

LOSSES = '''import torch
import torch.nn.functional as F
from torch import nn

def soft_dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    return 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return bce + self.dice_weight * soft_dice_loss(logits, target)
'''

TUNE = '''def pick(val_stems, args):
    if args.max_images:
        val_stems = val_stems[: args.max_images]
    return val_stems
'''


def patch():
    result=patch_auto14_sources(TRAIN,LOSSES,TUNE)
    for filename,source in result.items():
        ast.parse(source,filename=filename)
    return result


def test_loss_zero_weight_is_exact_upstream_and_backward_is_finite(monkeypatch):
    source=patch()["solarseg/losses.py"]
    namespace={}
    exec(compile(source,"patched loss","exec"),namespace)
    loss=namespace["BCEDiceLoss"](pos_weight=3.0)
    rng=torch.Generator().manual_seed(122)
    logits=torch.randn((2,1,16,16),generator=rng,requires_grad=True)
    target=torch.zeros((2,1,16,16))
    target[:,:,4:11,5:12]=1
    expected=F.binary_cross_entropy_with_logits(logits,target,pos_weight=loss.pos_weight)
    expected=expected+namespace["soft_dice_loss"](logits.float(),target)
    measured=loss(logits,target)
    assert torch.equal(expected,measured)
    measured.backward()
    assert torch.isfinite(logits.grad).all()


def test_boundary_loss_upweights_wrong_edge_logits_and_has_gradient(monkeypatch):
    monkeypatch.setenv("SOLAR_BOUNDARY_WEIGHT","6")
    namespace={}
    exec(compile(patch()["solarseg/losses.py"],"loss","exec"),namespace)
    loss=namespace["BCEDiceLoss"](pos_weight=1.0,dice_weight=0)
    y=torch.zeros((1,1,20,20))
    y[:,:,6:14,6:14]=1
    logits=torch.full_like(y,-5.0,requires_grad=True)
    with torch.no_grad():
        logits[:,:,6:14,6:14]=5
        logits[:,:,6,6:14]=-5  # incorrect filament boundary prediction
    baseline=F.binary_cross_entropy_with_logits(logits,y)
    weighted=loss(logits,y)
    assert weighted.item()>baseline.item()
    weighted.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    monkeypatch.setenv("SOLAR_BOUNDARY_WEIGHT","9")
    with pytest.raises(ValueError,match="between 0 and 8"):
        namespace["BCEDiceLoss"]()


def test_deterministic_month_round_robin_samples_every_month():
    namespace={}
    exec(compile(patch()["scripts/tune.py"],"balanced sampler","exec"),namespace)
    stems=[f"{month}{i:02}M" for month in ("202501","202502","202503","202504") for i in range(9)]
    first=namespace["pick"](stems,SimpleNamespace(max_images=8))
    assert first==namespace["pick"](stems,SimpleNamespace(max_images=8))
    assert len(first)==len(set(first))==8
    assert Counter(s[:6] for s in first)=={
        "202501":2,"202502":2,"202503":2,"202504":2
    }


def test_source_drift_is_rejected():
    for inputs in (
        (TRAIN.replace("train_stems, val_stems = split_stems","a, b = split_stems"),LOSSES,TUNE),
        (TRAIN,LOSSES.replace("import torch\n","import something\n"),TUNE),
        (TRAIN,LOSSES,TUNE.replace("val_stems[: args.max_images]","val_stems[-args.max_images:]")),
    ):
        with pytest.raises(ValueError,match="source has drifted"):
            patch_auto14_sources(*inputs)


def test_immutable_first_day_plan_and_bounded_research_arms(tmp_path):
    original=choose_experiment(1)
    assert "boundary_weight" not in original
    assert "boundary_weight" not in choose_experiment(3)
    for day in range(4,15):
        p=choose_experiment(day)
        assert 0<p["boundary_weight"]<=8
        out=tmp_path/str(day)/"auto14.ipynb"
        build(day,out)
        import json
        code="".join(json.loads(out.read_text())["cells"][1]["source"])
        compile(code,"Kaggle boundary-aware arm","exec")
        assert "patch_auto14_sources(*originals)" in code
        assert '"SOLAR_BOUNDARY_WEIGHT"' in code
        assert "source_patches.json" in code
        assert "competitions_submit" not in code
