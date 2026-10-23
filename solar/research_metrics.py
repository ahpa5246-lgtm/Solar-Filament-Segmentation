"""Diagnostics for MAGFiLO instance segmentation.

The instance Dice score here is an explicitly labelled LOCAL PROXY, not a
reverse-engineered claim about Kaggle's private implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class InstanceDiagnostics:
    matched_dice_proxy: float
    gt_count: int
    pred_count: int
    likely_splits: int
    likely_merges: int


def diagnose_instances(gt: np.ndarray, pred: np.ndarray, overlap_fraction: float = .10) -> InstanceDiagnostics:
    """Report a matched Dice proxy and likely split/merge relationships.

    A split means one reference object overlaps >=2 predicted objects;
    a merge means one predicted object overlaps >=2 reference objects.
    Overlap is measured relative to the reference/predicted object area,
    respectively, and these counts are diagnostic, not official scores.
    """
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError('gt and pred must be same-shape 2D instance maps')
    if overlap_fraction <= 0 or overlap_fraction >= 1:
        raise ValueError('overlap_fraction must be between zero and one')
    gl, gi = np.unique(gt, return_inverse=True)
    pl, pi = np.unique(pred, return_inverse=True)
    gidx = np.flatnonzero(gl > 0)
    pidx = np.flatnonzero(pl > 0)
    ng, npred = len(gidx), len(pidx)
    if not ng and not npred:
        return InstanceDiagnostics(1., 0, 0, 0, 0)
    if not ng or not npred:
        return InstanceDiagnostics(0., ng, npred, 0, 0)
    gi, pi = gi.ravel(), pi.ravel()
    intersections = np.bincount(gi * len(pl) + pi, minlength=len(gl) * len(pl)).reshape(len(gl), len(pl))
    ga = np.bincount(gi, minlength=len(gl))
    pa = np.bincount(pi, minlength=len(pl))
    shared = intersections[np.ix_(gidx, pidx)].astype(np.float64)
    garea = ga[gidx].astype(np.float64)
    parea = pa[pidx].astype(np.float64)
    dice = 2 * shared / np.maximum(garea[:, None] + parea[None, :], 1)
    rows, cols = linear_sum_assignment(-dice)
    # Missing or extra instances penalized by the larger cardinality.
    matched = float(dice[rows, cols].sum()) / max(ng, npred)
    # Count objects overlapping two or more partners (not just exactly two).
    splits = int(np.count_nonzero(((shared / np.maximum(garea[:, None], 1)) >= overlap_fraction).sum(axis=1) >= 2))
    merges = int(np.count_nonzero(((shared / np.maximum(parea[None, :], 1)) >= overlap_fraction).sum(axis=0) >= 2))
    return InstanceDiagnostics(matched, ng, npred, splits, merges)
