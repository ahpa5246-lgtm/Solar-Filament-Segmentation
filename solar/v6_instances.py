"""Optional V6 postprocessing ablations; no threshold is claimed validated.

Keep binary masks separate through evaluation. Resolving overlaps is optional,
changes their area and must be compared on validation before use.
"""
import numpy as np
from .v6_audit import _masks


def _scores(masks,scores):
    masks = _masks(masks)
    scores = np.asarray(scores,dtype=float)
    if scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError('Need one finite score per instance')
    return masks,scores


def mask_nms(masks,scores,iou_threshold=.5):
    """Stable mask IoU NMS; never union neighboring instances."""
    masks,scores = _scores(masks,scores)
    if not 0 < iou_threshold < 1:
        raise ValueError('NMS IoU threshold must lie in (0,1)')
    keep=[]
    for idx in np.argsort(-scores,kind='stable'):
        duplicate=False
        for previous in keep:
            shared = np.count_nonzero(masks[idx]&masks[previous])
            union = np.count_nonzero(masks[idx]|masks[previous])
            if shared/union > iou_threshold:
                duplicate=True; break
        if not duplicate: keep.append(int(idx))
    return keep


def resolve_overlap(masks,scores):
    """Assign shared pixels to the highest-confidence instance, no dilation."""
    masks,scores = _scores(masks,scores)
    occupied=np.zeros(masks.shape[1:],bool)
    result=np.zeros_like(masks)
    for idx in np.argsort(-scores,kind='stable'):
        result[idx] = masks[idx]&~occupied
        occupied |= masks[idx]
    return result


def restore_tile(masks,origin,shape):
    """Place N native-resolution tile masks into full-image coordinates."""
    masks = _masks(masks)
    y,x = origin; h,w = masks.shape[1:]
    if y < 0 or x < 0 or y+h > shape[0] or x+w > shape[1]:
        raise ValueError('Tile exceeds image bounds')
    output=np.zeros((len(masks),*shape),bool)
    output[:,y:y+h,x:x+w] = masks
    return output
