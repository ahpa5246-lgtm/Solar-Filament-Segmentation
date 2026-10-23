"""Overlapping-instance forensic tools. All reported PQ values are LOCAL proxies.

Independent binary masks deliberately avoid label-map overwrite of overlapping
annotations. This implementation is not asserted equivalent to Kaggle's scorer.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as coco


def _masks(value):
    value = np.asarray(value)
    if value.ndim != 3 or not np.isin(value, [0,1]).all():
        raise ValueError('Expected binary masks with shape [N,H,W]')
    if len(value) and (value.sum(axis=(1,2)) == 0).any():
        raise ValueError('Empty instances are invalid; use [0,H,W] for no detections')
    return value.astype(bool, copy=False)


def pairwise(gt, pred):
    gt, pred = _masks(gt), _masks(pred)
    if gt.shape[1:] != pred.shape[1:]:
        raise ValueError('Mask dimensions differ')
    ga = gt.sum(axis=(1,2)).astype(float)
    pa = pred.sum(axis=(1,2)).astype(float)
    # Avoid N*M*H*W temporaries and uint8 matrix multiplication overflow.
    intersection = np.zeros((len(gt),len(pred)),float)
    for i,g in enumerate(gt):
        for j,p in enumerate(pred):
            intersection[i,j] = np.count_nonzero(g & p)
    iou = intersection / np.maximum(ga[:,None]+pa[None,:]-intersection,1)
    return intersection, ga, pa, iou


def measure(gt, pred):
    intersection,ga,pa,iou = pairwise(gt,pred)
    # Prioritize valid matches before IoU sum for overlapping annotation masks.
    valid = iou > .5
    rows,cols = linear_sum_assignment(-(valid*(min(iou.shape,default=0)+1)+iou))
    matches = [(int(r),int(c)) for r,c in zip(rows,cols) if valid[r,c]]
    values = [float(iou[r,c]) for r,c in matches]
    tp,fp,fn = len(values), len(pa)-len(values), len(ga)-len(values)
    denominator = tp+.5*(fp+fn)
    total = sum(values)
    return {
        'pq_local':total/denominator if denominator else 1.,
        'sq':total/tp if tp else (1. if not denominator else 0.),
        'rq':tp/denominator if denominator else 1.,
        'tp':tp,'fp':fp,'fn':fn,'sum_iou':total,
        'precision':tp/(tp+fp) if tp+fp else (1. if not fn else 0.),
        'recall':tp/(tp+fn) if tp+fn else 1.,
        'matched_iou':values,
        'matched_dice':[float(2*intersection[r,c]/(ga[r]+pa[c])) for r,c in matches],
        'likely_splits':int(np.count_nonzero((intersection/np.maximum(ga[:,None],1) >= .1).sum(1)>=2)),
        'likely_merges':int(np.count_nonzero((intersection/np.maximum(pa[None,:],1) >= .1).sum(0)>=2)),
        'unmatched_gt':[i for i in range(len(ga)) if i not in {r for r,_ in matches}],
        'unmatched_pred':[i for i in range(len(pa)) if i not in {c for _,c in matches}],
    }


def partition(records, seed=42):
    """60/20/20 date partition for NEW runs; never reuse for already trained V5.

    Dates must be eight leading YYYYMMDD characters. V5 audit requires its
    original training manifest, rather than a newly invented holdout.
    """
    import datetime
    days = sorted({r['stem'][:8] for r in records})
    for day in days:
        datetime.datetime.strptime(day,'%Y%m%d')
    if len(days) < 5:
        raise ValueError('Need at least five observation days for three-way evaluation')
    np.random.default_rng(seed).shuffle(days)
    n = max(1,round(.2*len(days)))
    tuning,confirmation = set(days[:n]),set(days[n:2*n])
    return tuple([r for r in records if (r['stem'][:8] in group)]
                 for group in (set(days)-tuning-confirmation,tuning,confirmation))


def bootstrap(observations, seed=42, n=2000):
    """Macro-observation PQ; average annotators first, bootstrap observations."""
    if not observations or n < 1 or any(not rows for rows in observations.values()):
        raise ValueError('Need nonempty observation groups and positive replicates')
    scores = np.array([np.mean([r['pq_local'] for r in rows]) for rows in observations.values()])
    rng = np.random.default_rng(seed)
    samples = [float(rng.choice(scores,len(scores),replace=True).mean()) for _ in range(n)]
    low,high = np.quantile(samples,[.025,.975])
    return {'estimate':float(scores.mean()),'low':float(low),'high':float(high),
            'unit':'observation','aggregation':'macro observation; annotators averaged',
            'observations':len(scores),'replicates':n,'seed':seed}


def export_submission(predictions, image_stems, dest):
    """Validate the full inventory, round-trip compressed COCO RLE, then export."""
    stems = list(image_stems)
    if len(set(stems)) != len(stems) or set(predictions) != set(stems):
        raise ValueError('Duplicate image stems or incomplete/extra image predictions')
    rows=[]
    for stem in stems:
        masks = _masks(predictions[stem])
        if masks.shape[1:] != (2048,2048):
            raise ValueError('Competition masks must be 2048 by 2048')
        for i,mask in enumerate(masks,1):
            encoded = coco.encode(np.asfortranarray(mask.astype(np.uint8)))
            if not np.array_equal(coco.decode(encoded),mask):
                raise ValueError('COCO RLE round trip failed')
            rows.append({'filament_id':f'{stem}_{i}','segmentation_rle':encoded['counts'].decode('ascii')})
    if not rows or len({r['filament_id'] for r in rows}) != len(rows):
        raise ValueError('Empty submission or duplicate instance IDs')
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['filament_id','segmentation_rle'])
        writer.writeheader(); writer.writerows(rows)
    dest.with_suffix('.inventory.json').write_text(json.dumps({k:len(v) for k,v in predictions.items()},indent=2))


def overlay(image,gt,pred,path):
    """Two panels with individual contours: green GT; red predictions."""
    gt,pred = _masks(gt),_masks(pred)
    if image.shape != gt.shape[1:] or image.shape != pred.shape[1:]:
        raise ValueError('Image-mask shape mismatch')
    panels=[]
    for masks,color in ((gt,(0,255,0)),(pred,(0,0,255))):
        panel=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        for i,mask in enumerate(masks,1):
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel,contours,-1,color,1)
            y,x=np.argwhere(mask)[0]; cv2.putText(panel,str(i),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        panels.append(panel)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),np.concatenate(panels,axis=1)):
        raise OSError(f'Cannot save {path}')
