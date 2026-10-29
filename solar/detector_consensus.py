"""Four-view detector consensus, inspired by ttach and weighted mask fusion.

Original code: mask IoU matching counts distinct views, not duplicate proposals.
No labels or model-specific preprocessing are used by the fusion step.
"""
import numpy as np
from pycocotools import mask as coco

FLIPS=((),(1,),(0,),(0,1))


def inverse_flip(mask,axes):
    return np.ascontiguousarray(np.flip(mask,axis=axes) if axes else mask)


def fuse_views(views,keep_base=True,match_iou=.3,min_views=2):
    if not 0<match_iou<1 or not 1<=min_views<=len(views):raise ValueError('Invalid consensus settings')
    proposals=[]
    for view,(rles,scores) in enumerate(views):
        if len(rles)!=len(scores):raise ValueError('Mask/confidence inventory mismatch')
        for mask,score in zip(rles,scores):
            if not np.isfinite(score) or not 0<score<=1:raise ValueError('Invalid confidence')
            proposals.append((float(score),view,mask))
    proposals.sort(key=lambda item:-item[0])
    groups=[]
    for score,view,mask in proposals:
        eligible=[(index,group) for index,group in enumerate(groups) if view not in group['views']]
        best=-1;overlap=match_iou
        if eligible:
            values=coco.iou([mask],[group['masks'][0] for _,group in eligible],[0]*len(eligible))[0]
            for (index,_),value in zip(eligible,values):
                if value>overlap:best=index;overlap=value
        if best<0:
            groups.append({'views':{view},'masks':[mask],'scores':[score]})
        else:
            groups[best]['views'].add(view);groups[best]['masks'].append(mask);groups[best]['scores'].append(score)
    result=[];confidence=[]
    for group in groups:
        if len(group['views'])<min_views and not (keep_base and 0 in group['views']):continue
        weights=np.asarray(group['scores'],np.float32)
        masks=coco.decode(group['masks']).astype(np.float32)
        fused=np.tensordot(masks,weights,axes=([2],[0]))>=float(weights.sum())*.5
        if fused.any():
            result.append(coco.encode(np.asfortranarray(fused.astype(np.uint8))))
            confidence.append(float(weights.mean()))
    return result,confidence


def detect_views(model,image,resolution=1024):
    import cv2,torch
    if model.training:raise ValueError('Detector consensus requires evaluation mode')
    model.transform.min_size=(resolution,);model.transform.max_size=resolution
    resized=cv2.resize(image,(resolution,resolution),interpolation=cv2.INTER_AREA)
    views=[]
    with torch.inference_mode():
        for axes in FLIPS:
            pixels=inverse_flip(resized,axes)
            tensor=torch.from_numpy(pixels.astype(np.float32)/255).unsqueeze(0).repeat(3,1,1).cuda()
            output=model([tensor])[0];keep=output['scores']>=.6
            scores=output['scores'][keep].cpu().numpy()
            masks=(output['masks'][keep,0]>=.4).cpu().numpy().astype(np.uint8)
            rles=[];confidence=[]
            for mask,score in zip(masks,scores):
                mask=inverse_flip(mask,axes)
                if mask.any():rles.append(coco.encode(np.asfortranarray(mask)));confidence.append(float(score))
            views.append((rles,confidence))
    return views
