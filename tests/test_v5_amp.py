"""Real GradScaler regression: overflow must not update model weights."""
import pytest
import torch
from solar.v5_finetune import safe_scaled_step


def test_overflow_skips_update_reduces_scale_and_next_finite_step_recovers():
    p=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.SGD([p],lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    scaler.scale(p.sum()).backward()
    p.grad.fill_(float('inf'))
    assert safe_scaled_step(scaler,optimizer,[p]) is False
    assert p.item()==1. and p.grad is None
    assert scaler.get_scale()==4.
    scaler.scale(p.sum()).backward()
    assert safe_scaled_step(scaler,optimizer,[p]) is True
    assert p.item()==pytest.approx(.9)
    assert torch.isfinite(p).all()


def test_finite_gradients_are_clipped_before_update():
    p=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.SGD([p],lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    scaler.scale(p.sum()*100).backward()
    assert safe_scaled_step(scaler,optimizer,[p],max_norm=5.) is True
    assert p.item()==pytest.approx(.5)
