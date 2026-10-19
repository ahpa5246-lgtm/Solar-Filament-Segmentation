"""Morphology-aware instance extraction; parameters tuned against VALIDATION PQ."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


@dataclass(frozen=True)
class Params:
    threshold: float = .42
    boundary: float = .62
    min_area: int = 40
    closing: int = 3
    min_seed: int = 12


def to_instances(fg_probability: np.ndarray, boundary_probability: np.ndarray, params: Params=Params()) -> np.ndarray:
    if fg_probability.shape != boundary_probability.shape:
        raise ValueError('Foreground and boundary must have the same shape.')
    fg = (fg_probability >= params.threshold).astype(np.uint8)
    if params.closing > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.closing, params.closing))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    # Barrier only splits where an actual foreground region exists.
    seed_fg = (fg > 0) & (boundary_probability < params.boundary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_fg.astype(np.uint8), 8)
    markers = np.zeros_like(labels, dtype=np.int32)
    marker_i = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= params.min_seed:
            marker_i += 1
            markers[labels == i] = marker_i
    if marker_i == 0:
        return np.zeros_like(labels, dtype=np.int32)
    # Watershed propagates seed labels through the complete filament area,
    # preserving thin barbs instead of throwing away boundary pixels.
    distance = distance_transform_edt(fg)
    mask = watershed(-distance, markers, mask=(fg > 0)).astype(np.int32)
    output = np.zeros_like(mask)
    idx = 0
    for label in range(1, marker_i + 1):
        region = mask == label
        if int(region.sum()) >= params.min_area:
            idx += 1
            output[region] = idx
    return output
