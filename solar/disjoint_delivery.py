"""Competition-valid instance masks with stable confidence-priority ownership."""
import numpy as np
from pycocotools import mask as coco


def make_disjoint(rles,scores=None):
    if not rles:return []
    shape=tuple(rles[0]['size'])
    if any(tuple(r['size'])!=shape for r in rles):raise ValueError('Mixed mask dimensions')
    if scores is None:order=range(len(rles))
    else:
        scores=np.asarray(scores,dtype=float)
        if scores.shape!=(len(rles),) or not np.isfinite(scores).all():raise ValueError('One finite confidence per mask is required')
        order=np.argsort(-scores,kind='stable')
    occupied=np.zeros(shape,dtype=bool);result=[]
    for index in order:
        mask=coco.decode(rles[index]).astype(bool)
        mask &= ~occupied
        if not mask.any():continue
        occupied |= mask
        result.append(coco.encode(np.asfortranarray(mask.astype(np.uint8))))
    return result
