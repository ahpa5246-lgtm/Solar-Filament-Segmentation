"""Conservative directional reconnection of broken filament instances.

An instance can consist of several disconnected islands in COCO RLE. Grouping
visually aligned islands without *painting* a bridge avoids artificial positive
pixels and addresses the fragmentation penalty of Panoptic Quality.

All thresholds are full-resolution pixels. This module is purely geometric:
it never sees annotation targets, hidden test labels, or Kaggle leaderboard.
"""
from __future__ import annotations

import math

import numpy as np


def _tips(labels: np.ndarray, ident: int, *, min_area: int, min_elongation: float):
    ys, xs = np.nonzero(labels == ident)
    if len(ys) < min_area:
        return []
    # Deterministic subsampling bounds PCA work for very long filament bodies.
    if len(xs) > 20000:
        indices = np.linspace(0, len(xs) - 1, num=20000, dtype=np.intp)
        xs, ys = xs[indices], ys[indices]
    pixels = np.stack((xs, ys), axis=1).astype(np.float64)
    center = pixels.mean(axis=0)
    centered = pixels - center
    covariance = (centered.T @ centered) / max(len(pixels), 1)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0 or math.sqrt((values[-1] + 1e-6)/(values[0] + 1e-6)) < min_elongation:
        return []
    axis = vectors[:, -1]
    projections = centered @ axis
    # Robustly estimate the physical tip using the end 2% of filament pixels.
    low = np.quantile(projections, 0.02)
    high = np.quantile(projections, 0.98)
    if high - low < 4:
        return []
    return [(ident, pixels[projections <= low].mean(axis=0), -axis),
            (ident, pixels[projections >= high].mean(axis=0), axis)]


def link_aligned_fragments(
    labels: np.ndarray,
    *,
    max_gap: int,
    min_area: int = 80,
    min_elongation: float = 2.5,
    min_alignment: float = 0.90,
) -> np.ndarray:
    """Link near-collinear filament tips, without changing any foreground pixel.

    Greedy sorted endpoint matching allows one partner per tip. Components with
    round/noisy shapes are excluded by minimum elongation and area. A zero gap
    produces the exact original instance array. The newly assigned labels are
    deterministic and compact; both source objects remain present geometrically.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise TypeError("Expected a 2D integer instance label map")
    if np.any(labels < 0):
        raise ValueError("Instance ids must be nonnegative")
    if type(max_gap) is not int or not 0 <= max_gap <= 80:
        raise ValueError("max_gap must be an integer between 0 and 80")
    if type(min_area) is not int or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    if not (1 < min_elongation <= 20 and 0 < min_alignment <= 1):
        raise ValueError("Invalid direction or elongation limits")
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if max_gap == 0 or len(ids) <= 1:
        return labels.copy()

    endpoints = []
    for ident in ids:
        endpoints.extend(_tips(labels, int(ident), min_area=min_area,
                               min_elongation=min_elongation))
    candidates = []
    for a, (ia, pa, ua) in enumerate(endpoints):
        for b in range(a + 1, len(endpoints)):
            ib, pb, ub = endpoints[b]
            if ia == ib:
                continue
            delta = pb - pa
            distance = float(np.linalg.norm(delta))
            if not 0 < distance <= max_gap:
                continue
            direction = delta / distance
            first = float(np.dot(ua, direction))
            second = float(np.dot(ub, -direction))
            if first < min_alignment or second < min_alignment:
                continue
            # Proximity wins, then larger alignment, then stable component ids.
            cost = distance + max_gap * ((1-first) + (1-second))
            candidates.append((cost, int(ia), int(ib), a, b))
    candidates.sort()
    parents = {int(x): int(x) for x in ids}

    def root(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    used = set()
    linked = 0
    for _, ia, ib, a, b in candidates:
        if a in used or b in used:
            continue
        ra, rb = root(ia), root(ib)
        if ra == rb:
            continue
        parents[max(ra, rb)] = min(ra, rb)
        used.update((a, b))
        linked += 1
    if not linked:
        return labels.copy()

    roots = {int(ident): root(int(ident)) for ident in ids}
    canonical = {k: i + 1 for i, k in enumerate(sorted(set(roots.values())))}
    replacement = {ident: canonical[rt] for ident, rt in roots.items()}
    result = np.zeros_like(labels, dtype=np.int32)
    for ident, new_id in replacement.items():
        result[labels == ident] = new_id
    return result
