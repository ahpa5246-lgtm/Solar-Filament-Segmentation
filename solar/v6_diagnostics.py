"""Descriptive instance errors; bounding-box shape is not skeleton morphology."""
import numpy as np
from pycocotools import mask as coco


def instance_profile(truth,predictions):
    iou=coco.iou(truth,predictions,[0]*len(predictions)) if truth and predictions else np.zeros((len(truth),len(predictions)))
    matched=iou>.5
    result={}
    for name,rles,flags in [('truth',truth,matched.any(axis=1)),('prediction',predictions,matched.any(axis=0))]:
        counts={'all':{'total':len(rles),'matched':int(flags.sum())}}
        for index,rle in enumerate(rles):
            area=int(coco.area(rle));box=coco.toBbox(rle)
            side_min=min(box[2:]);side_max=max(box[2:])
            if area<=0 or side_min<=0:raise ValueError('Profiles require nonempty masks')
            area_bin=int(np.searchsorted([1024,4096,16384,65536],area,side='right'))
            tags=[f'area_bin_{area_bin}']
            if side_max/side_min>=5:tags.append('elongation_ge_5')
            if side_max/side_min>=10:tags.append('elongation_ge_10')
            for tag in tags:
                row=counts.setdefault(tag,{'total':0,'matched':0})
                row['total']+=1;row['matched']+=int(flags[index])
        result[name]=counts
    best=iou.max(axis=1) if predictions else np.zeros(len(truth))
    result['unmatched_truth_best_iou']=best[~matched.any(axis=1)].tolist()
    return result


def aggregate_profiles(profiles):
    result={'truth':{},'prediction':{}}
    for profile in profiles:
        for kind in ['truth','prediction']:
            for tag,counts in profile[kind].items():
                row=result[kind].setdefault(tag,{'total':0,'matched':0})
                for name in row:row[name]+=counts[name]
    for kind in ['truth','prediction']:
        for row in result[kind].values():row['matched_fraction']=row['matched']/row['total'] if row['total'] else None
    best=[value for profile in profiles for value in profile['unmatched_truth_best_iou']]
    result['unmatched_truth_best_iou_quantiles']=np.quantile(best,[0,.25,.5,.75,1]).tolist() if best else []
    result['area_bin_edges_native_pixels']=[0,1024,4096,16384,65536,'infinity']
    result['interpretation']='Descriptive strict-IoU>0.5 presence; annotator variants counted separately; shape uses bounding boxes, not skeletons.'
    return result
