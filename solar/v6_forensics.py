"""Offline V5 diagnostic runner, independent of private Kaggle authentication.

Prediction JSON: {strategy: {physical_image_stem: [{size:[H,W], counts:str}]}}.
Training manifest: JSON list of every physical stem used for fitting the model
OR its refiner. A new random split cannot establish that V5 never saw an image.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as coco
from .v6_audit import measure, bootstrap, overlay


def run_audit(ground_truth,predictions,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if not ground_truth or not predictions:
        raise ValueError('Need ground truth and prediction strategies')
    report={}
    for strategy,by_image in predictions.items():
        if set(by_image) != set(ground_truth):
            raise ValueError(f'{strategy}: missing/extra physical observation predictions')
        observations={}
        for stem,variants in ground_truth.items():
            if not variants: raise ValueError('Observation has no annotation variants')
            shape=variants[0].shape[1:]
            masks=[]
            for rle in by_image[stem]:
                if tuple(rle['size']) != shape: raise ValueError('Prediction dimensions differ')
                masks.append(coco.decode({'size':rle['size'],'counts':rle['counts'].encode('ascii')}))
            pred=np.stack(masks).astype(bool) if masks else np.zeros((0,*shape),bool)
            observations[stem]=[measure(gt,pred) for gt in variants]
        rows=[row for variants in observations.values() for row in variants]
        tp,fp,fn=(sum(r[k] for r in rows) for k in ('tp','fp','fn'))
        total=sum(r['sum_iou'] for r in rows); denominator=tp+.5*(fp+fn)
        report[strategy]={
            'micro':{'pq_local':total/denominator if denominator else 1.,
                     'sq':total/tp if tp else (1. if not denominator else 0.),
                     'rq':tp/denominator if denominator else 1.,'tp':tp,'fp':fp,'fn':fn},
            'macro_observation_ci':bootstrap(observations),
            'observations':observations,
            'metric_status':'local proxy; official evaluator equivalence unverified',
        }
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def load_holdout(data,train_stems,validation_stems):
    from .data import load_records,read_gray
    records,_,_=load_records(data)
    fitting=set(train_stems); selected=set(validation_stems)
    if not fitting or not selected or fitting&selected:
        raise ValueError('Need nonempty, disjoint original fitting and validation manifests')
    if {s[:8] for s in fitting}&{s[:8] for s in selected}:
        raise ValueError('Training and holdout dates overlap')
    known={r['stem'] for r in records}
    if not (fitting|selected)<=known: raise ValueError('Unknown stems in split manifest')
    truth={s:[] for s in sorted(selected)}; paths={}
    for record in records:
        stem=record['stem']
        if stem not in selected: continue
        image=read_gray(record['path']); h,w=image.shape
        masks=[]
        for annotation in record['ann']:
            seg=annotation.get('segmentation')
            if not isinstance(seg,list): raise ValueError('Expected official train polygons')
            rles=coco.frPyObjects(seg,h,w)
            if rles:
                mask=coco.decode(coco.merge(rles)).astype(bool)
                if mask.any(): masks.append(mask)
        truth[stem].append(np.stack(masks) if masks else np.zeros((0,h,w),bool))
        paths[stem]=record['path']
    return truth,paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--fitting-stems',type=Path,required=True)
    parser.add_argument('--validation-stems',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    truth,paths=load_holdout(args.data,json.loads(args.fitting_stems.read_text()),json.loads(args.validation_stems.read_text()))
    predictions=json.loads(args.predictions.read_text())
    report=run_audit(truth,predictions,args.out)
    from .data import read_gray
    for strategy,by_image in predictions.items():
        # Numeric filenames keep strategy/observation names out of output paths.
        folder=args.out/f'strategy-{list(predictions).index(strategy)}'
        index={}
        for number,(stem,variants) in enumerate(truth.items()):
            shape=variants[0].shape[1:]
            pred=[coco.decode({'size':r['size'],'counts':r['counts'].encode()}) for r in by_image[stem]]
            pred=np.stack(pred) if pred else np.zeros((0,*shape),bool)
            for annotator,gt in enumerate(variants):
                name=f'{number:04d}-{annotator:02d}.png'
                overlay(read_gray(paths[stem]),gt,pred,folder/name)
                index[name]={'observation':stem,'annotation_variant':annotator}
        (folder/'index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps({k:v['micro'] for k,v in report.items()},indent=2))


if __name__=='__main__': main()
