"""Bounded V6 detector experiment. Not a validated replacement for V5.

Starts only with explicit --run-training and an archived official evaluator.
Current local metric remains a proxy until equivalence is independently tested.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as coco
from .data import load_records,read_gray
from .v6_audit import partition,measure,bootstrap
from .v6_detector import build_detector,infer,save_checkpoint,load_checkpoint


class DetectionDataset:
    def __init__(self,records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        record=self.records[index]
        image=read_gray(record['path']); h,w=image.shape
        masks=[]; boxes=[]
        for annotation in record['ann']:
            polygons=annotation['segmentation']
            if not isinstance(polygons,list): raise ValueError('Expected train polygons')
            rles=coco.frPyObjects(polygons,h,w)
            if not rles: continue
            mask=coco.decode(coco.merge(rles))
            yy,xx=np.nonzero(mask)
            if not len(xx): continue
            masks.append(mask); boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        tensor=torch.from_numpy(image.copy()).float().div(255).unsqueeze(0).expand(3,-1,-1)
        target={'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                'labels':torch.ones(len(masks),dtype=torch.int64),
                'masks':torch.tensor(np.stack(masks) if masks else np.zeros((0,h,w)),dtype=torch.uint8),
                'image_id':torch.tensor(index)}
        return tensor,target


def evaluate(model,records,score_threshold=.5):
    groups={}; started=time.perf_counter()
    grouped={}
    for record in records:
        grouped.setdefault(record['stem'],[]).append(record)
    # Release full-resolution predictions after each observation. Never cache
    # the whole validation split (hundreds of MB per observation).
    for stem,variants in grouped.items():
        pred=infer(model,read_gray(variants[0]['path']),score_threshold)[0]
        dataset=DetectionDataset(variants)
        groups[stem]=[measure(dataset[i][1]['masks'].numpy(),pred) for i in range(len(dataset))]
    result=bootstrap(groups,metric="pq_reference")
    rows=[row for variants in groups.values() for row in variants]
    tp,fp,fn=(sum(row[k] for row in rows) for k in ('reference_tp','reference_fp','reference_fn'))
    denominator=tp+.5*(fp+fn)
    result['reference_micro_pq']=sum(row['reference_sum_iou'] for row in rows)/denominator if denominator else 0.
    result['evaluation_seconds_per_observation']=(time.perf_counter()-started)/len(groups)
    result['by_observation']=groups
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--official-evaluation',type=Path,required=True)
    p.add_argument('--run-training',action='store_true')
    p.add_argument('--epochs',type=int,default=2); p.add_argument('--max-steps',type=int,default=100)
    p.add_argument('--resolution',type=int,default=1024); p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not args.run_training: p.error('Training requires explicit --run-training')
    if not args.official_evaluation.is_file(): p.error('Archive and inspect official evaluator first')
    if args.epochs<1 or args.max_steps<1: p.error('Epoch and per-epoch step budget must be positive')
    if not torch.cuda.is_available(): p.error('GPU required for this bounded research run')
    if args.out.exists() and any(args.out.iterdir()): p.error('Use a new empty output directory; preserve previous runs')
    args.out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    records,_,_=load_records(args.data)
    train,tune,confirm=partition(records,args.seed)
    config={'seed':args.seed,'min_size':args.resolution,'max_size':args.resolution,
            'epochs':args.epochs,'max_steps_per_epoch':args.max_steps,'learning_rate':.0001,
            'metric_status':'micro-aggregated public-reference PQ for selection, macro observation bootstrap CI; not an official score',
            'official_evaluation_sha256':hashlib.sha256(args.official_evaluation.read_bytes()).hexdigest(),
            'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
            'device_count_visible':torch.cuda.device_count(),'devices_used':1}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    (args.out/'split.json').write_text(json.dumps({name:[{'stem':r['stem'],'image_id':r['image_id']} for r in part]
                                                 for name,part in [('train',train),('tune',tune),('confirmation',confirm)]},indent=2))
    model=build_detector(True,args.resolution,args.resolution).cuda()
    loader=torch.utils.data.DataLoader(DetectionDataset(train),batch_size=1,shuffle=True,
                                        collate_fn=lambda batch:tuple(zip(*batch)),num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda'); best=-1.; history=[]; start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for step,(images,targets) in enumerate(loader):
            if step>=args.max_steps: break
            images=[im.cuda() for im in images]; targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=sum(model(images,targets).values())
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        validation=evaluate(model,tune)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'tuning_pq_reference':validation['reference_micro_pq']}
        history.append(row); print(json.dumps(row),flush=True)
        if validation['reference_micro_pq']>best:
            best=validation['reference_micro_pq']; save_checkpoint(model,args.out/'best.pt',config)
        (args.out/'history.json').write_text(json.dumps(history,indent=2))
    model,_=load_checkpoint(args.out/'best.pt','cuda')
    confirmation=evaluate(model,confirm)
    confirmation['training_seconds']=time.perf_counter()-start
    confirmation['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    (args.out/'confirmation.json').write_text(json.dumps(confirmation,indent=2))
    print('Completed bounded experiment, NOT a final full-data model or official submission.')


if __name__=='__main__': main()
