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


class EdgeGuidedAttention(nn.Module):
    """Edge-conditioned bottleneck attention, inspired by EdgeAttNet.

    This is an original lightweight ablation, NOT an exact reproduction of
    EdgeAttNet. Edge priors are learned from the same H-alpha input only.
    """
    def __init__(self, channels):
        super().__init__()
        if channels % 4:
            raise ValueError('Edge attention channels must be divisible by four.')
        self.edge = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 4, channels)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, features, input_image):
        edge_input = F.interpolate(input_image, size=features.shape[-2:],
                                   mode='bilinear', align_corners=False)
        edge = torch.sigmoid(self.edge(edge_input)).flatten(2).transpose(1, 2)
        tokens = features.flatten(2).transpose(1, 2)
        # Query/key conditioning changes attention affinity near learned edges.
        conditioned = tokens * (1.0 + edge)
        attended, _ = self.attention(conditioned, conditioned, tokens,
                                     need_weights=False)
        attended = attended.transpose(1, 2).reshape_as(features)
        return self.norm(features + self.gain.tanh() * attended)


def _soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_skeleton(x, iterations=5):
    # Differentiable morphological skeleton, as used in soft-clDice.
    skeleton = F.relu(x - F.max_pool2d(_soft_erode(x), 3, stride=1, padding=1))
    for _ in range(iterations):
        x = _soft_erode(x)
        delta = F.relu(x - F.max_pool2d(_soft_erode(x), 3, stride=1, padding=1))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(pred, target, iterations=5):
    """Topology-sensitive loss for thin structures; optional training ablation."""
    pred_skel = _soft_skeleton(pred, iterations)
    true_skel = _soft_skeleton(target, iterations)
    dims = (1, 2, 3)
    tprec = (pred_skel * target).sum(dims).add(1).div(pred_skel.sum(dims).add(1))
    tsens = (true_skel * pred).sum(dims).add(1).div(true_skel.sum(dims).add(1))
    return (1 - (2 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6)).mean()


class Net(nn.Module):
    def __init__(self, base=24, architecture='unet'):
        super().__init__()
        if architecture not in {'unet', 'edge_attention'}:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.base = base
        self.architecture = architecture
        self.e1 = Block(2, base)
        self.e2 = Block(base, base*2)
        self.e3 = Block(base*2, base*4)
        self.e4 = Block(base*4, base*8)
        self.mid = Block(base*8, base*12)
        self.edge_attention = EdgeGuidedAttention(base*12) if architecture == 'edge_attention' else None
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
        if self.edge_attention is not None:
            m = self.edge_attention(m, x)
        def up(v, skip): return torch.cat([F.interpolate(v, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], dim=1)
        x = self.d4(up(m, d))
        x = self.d3(up(x, c))
        x = self.d2(up(x, b))
        x = self.d1(up(x, a))
        return self.head(x)


def loss_fn(logits, target, cldice_weight=0.0):
    fglog, bdlog = logits[:, 0:1], logits[:, 1:2]
    fg, bd = target[:, 0:1], target[:, 1:2]
    bce = F.binary_cross_entropy_with_logits(fglog, fg, pos_weight=torch.tensor(5.0, device=fg.device))
    probabilities = fglog.sigmoid()
    dice_loss = 1 - (2*(probabilities*fg).sum((2,3)) + 1).div(probabilities.sum((2,3)) + fg.sum((2,3)) + 1).mean()
    bd_loss = F.binary_cross_entropy_with_logits(bdlog, bd, pos_weight=torch.tensor(3.0, device=fg.device))
    result = bce + dice_loss + .25 * bd_loss
    if cldice_weight > 0:
        result = result + cldice_weight * soft_cldice_loss(probabilities, fg)
    return result
