"""Instance-level PQ with IoU > .5 matching; group-aware aggregation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Counts:
    sum_iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def pq(self):
        denom = self.tp + .5*self.fp + .5*self.fn
        return self.sum_iou / denom if denom else 1.0

    def __iadd__(self, other):
        self.sum_iou += other.sum_iou
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self


def compare(gt: np.ndarray, pred: np.ndarray, match_threshold=.5) -> Counts:
    """Compute instance IoU on a 2D integer label map. Background label = 0."""
    if gt.shape != pred.shape:
        raise ValueError('Ground truth and prediction dimensions differ.')
    g = np.unique(gt[gt > 0])
    p = np.unique(pred[pred > 0])
    if len(g) == 0 or len(p) == 0:
        return Counts(tp=0, fp=len(p), fn=len(g))
    # Labels are compact in both datasets, so sparse overlap accumulation is cheap.
    gf, pg = np.unique(gt, return_inverse=True)
    pg = pg.ravel()
    pf, pp = np.unique(pred, return_inverse=True)
    pp = pp.ravel()
    valid = (gt.ravel() > 0) & (pred.ravel() > 0)
    ng, npred = len(gf), len(pf)
    intersections = np.bincount(pg[valid] * npred + pp[valid], minlength=ng*npred).reshape(ng,npred)
    garea = np.bincount(pg, minlength=ng).astype(np.float64)
    parea = np.bincount(pp, minlength=npred).astype(np.float64)
    iou = intersections / np.maximum(garea[:, None]+parea[None, :]-intersections, 1)
    iou[gf == 0, :] = 0
    iou[:, pf == 0] = 0
    rows, cols = linear_sum_assignment(-iou)
    matched = [(r,c) for r,c in zip(rows,cols) if iou[r,c] > match_threshold]
    total = float(sum(iou[r,c] for r,c in matched))
    return Counts(sum_iou=total, tp=len(matched), fp=len(p)-len(matched), fn=len(g)-len(matched))
