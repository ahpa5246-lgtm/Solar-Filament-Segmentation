"""Compact 2-channel multi-task U-Net trained from scratch (no external weights)."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                                    nn.GroupNorm(min(8, c_out), c_out), nn.SiLU(inplace=True),
                                    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                                    nn.GroupNorm(min(8, c_out), c_out), nn.SiLU(inplace=True))
    def forward(self, x): return self.layers(x)


class Net(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.base = base
        self.e1 = Block(2, base)
        self.e2 = Block(base, base*2)
        self.e3 = Block(base*2, base*4)
        self.e4 = Block(base*4, base*8)
        self.mid = Block(base*8, base*12)
        self.d4 = Block(base*20, base*8)
        self.d3 = Block(base*12, base*4)
        self.d2 = Block(base*6, base*2)
        self.d1 = Block(base*3, base)
        self.head = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.e4(F.avg_pool2d(c, 2))
        m = self.mid(F.avg_pool2d(d, 2))
        def up(v, skip): return torch.cat([F.interpolate(v, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], dim=1)
        x = self.d4(up(m, d))
        x = self.d3(up(x, c))
        x = self.d2(up(x, b))
        x = self.d1(up(x, a))
        return self.head(x)


def loss_fn(logits, target):
    fglog, bdlog = logits[:, 0:1], logits[:, 1:2]
    fg, bd = target[:, 0:1], target[:, 1:2]
    bce = F.binary_cross_entropy_with_logits(fglog, fg, pos_weight=torch.tensor(5.0, device=fg.device))
    probabilities = fglog.sigmoid()
    dice_loss = 1 - (2*(probabilities*fg).sum((2,3)) + 1).div(probabilities.sum((2,3)) + fg.sum((2,3)) + 1).mean()
    bd_loss = F.binary_cross_entropy_with_logits(bdlog, bd, pos_weight=torch.tensor(3.0, device=fg.device))
    return bce + dice_loss + .25 * bd_loss
