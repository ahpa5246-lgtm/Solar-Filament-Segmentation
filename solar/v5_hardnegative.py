"""Train-only false-positive mining for a bounded V5 detector continuation.

Never accepts tuning/confirmation/test images. This is a sampling intervention,
not evidence of a score improvement or an independent validation holdout.
"""
import math


def sampling_weights(stems, false_positives, boost=0.75, cap=4):
    """Return one positive weight per fitting row; no validation-derived input."""
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("Fitting inventory must be nonempty and unique")
    if set(stems) != set(false_positives):
        raise ValueError("False-positive inventory must equal fitting inventory")
    if not math.isfinite(boost) or boost < 0 or not isinstance(cap, int) or cap < 1:
        raise ValueError("Invalid bounded hard-negative sampling parameters")
    weights = []
    for stem in stems:
        count = false_positives[stem]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("False-positive counts must be nonnegative integers")
        weights.append(1.0 + boost * min(count, cap))
    return weights


def count_unmatched_predictions(predicted, ground_truth_variants, native_size=2048, size=1024):
    """Count proposal false positives against *all* fitting annotation variants."""
    import cv2
    import numpy as np
    from pycocotools import mask as coco
    if not ground_truth_variants:
        raise ValueError("Missing fitting annotation variants")
    if any(list(mask["size"]) != [size, size] for mask in predicted):
        raise ValueError("Expected detector-resolution proposal RLE")
    gt = []
    for variant in ground_truth_variants:
        for rle in variant:
            if list(rle["size"]) != [native_size, native_size]:
                raise ValueError("Unexpected native fitting annotation shape")
            mask = coco.decode(rle)
            resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            if resized.any():
                gt.append(coco.encode(np.asfortranarray(resized.astype(np.uint8))))
    if not predicted:
        return 0
    if not gt:
        return len(predicted)
    overlap = coco.iou(predicted, gt, [0] * len(gt))
    return int(np.sum(np.max(overlap, axis=1) < 0.5))
