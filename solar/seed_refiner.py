"""Condition crop refinement on image, local contrast and proposal-box identity.

The input seed contains a bounding rectangle, not ground-truth mask pixels.
Empty outputs reject proposals instead of restoring a false-positive fallback.
"""
import cv2,numpy as np,torch
from pycocotools import mask as coco
from .v6_refiner_audit import postprocess
from .disjoint_delivery import make_disjoint


def feature_crop(image,rle,helpers):
    mask=coco.decode(rle)
    if mask.shape!=image.shape:mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    native=coco.encode(np.asfortranarray(mask.astype(np.uint8)))
    bounds=helpers['square_bounds'](mask);x0,y0,x1,y1=bounds
    crop=cv2.resize(image[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_LINEAR)
    contrast=cv2.createCLAHE(2.,(8,8)).apply(crop)
    x,y,w,h=coco.toBbox(native).astype(int)
    seed=np.zeros_like(mask,dtype=np.uint8);seed[y:y+h,x:x+w]=255
    seed=cv2.resize(seed[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_NEAREST)
    return np.stack((crop,contrast,seed)),bounds


def restore_masks(bounds,probabilities,shape,threshold,policy):
    if not 0<threshold<1:raise ValueError('Invalid mask threshold')
    rles=[];scores=[]
    for (x0,y0,x1,y1),prob in zip(bounds,probabilities):
        resized=cv2.resize(prob,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
        mask=resized>=threshold
        if not mask.any():continue
        native=np.zeros(shape,np.uint8);native[y0:y1,x0:x1]=mask
        rles.append(coco.encode(np.asfortranarray(native)));scores.append(float(resized[mask].mean()))
    return make_disjoint(postprocess(rles,scores,policy))


@torch.inference_mode()
def predict_variants(model,helpers,image,rles,settings=None):
    settings=settings or [(p,t) for p in ('union','nms') for t in (.5,.65,.85)]
    features=[];bounds=[];probabilities=[]
    for rle in rles:
        crop,bound=feature_crop(image,rle,helpers);features.append(crop);bounds.append(bound)
    for start in range(0,len(features),32):
        batch=torch.from_numpy(np.stack(features[start:start+32])).float().cuda()/255
        probs=helpers['predict_refiner_probabilities'](model,batch,use_tta=True).cpu().numpy()
        probabilities.extend(probs)
    return {f'{policy}|{threshold}':restore_masks(bounds,probabilities,image.shape,threshold,policy) for policy,threshold in settings}


@torch.inference_mode()
def predict_maps(model,helpers,image,rles):
    features=[];bounds=[];probabilities=[]
    for rle in rles:
        crop,bound=feature_crop(image,rle,helpers);features.append(crop);bounds.append(bound)
    for start in range(0,len(features),32):
        batch=torch.from_numpy(np.stack(features[start:start+32])).float().cuda()/255
        probabilities.extend(helpers['predict_refiner_probabilities'](model,batch,use_tta=True).cpu().numpy())
    return bounds,probabilities
