"""Honest date-heldout Panoptic Quality hyperparameter calibration."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
import torch
from .data import load_records, split_records, read_gray, polygons_to_instances
from .metrics import compare, Counts
from .pipeline import load_model, predict_probabilities
from .postprocess import Params,to_instances


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='/kaggle/input')
    ap.add_argument('--checkpoint',default='/kaggle/working/solar_artifacts/best.pt')
    ap.add_argument('--out',default='/kaggle/working/solar_artifacts')
    ap.add_argument('--max-images',type=int,default=16)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tile',type=int,default=512)
    ap.add_argument('--stride',type=int,default=384)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.max_images < 1:
        ap.error('Must validate at least one observation')
    records,_,_=load_records(args.data)
    _,validation=split_records(records,seed=args.seed)
    groups=defaultdict(list)
    for record in validation:
        groups[record['stem']].append(record)
    # Deterministic subset, chosen without regard to image content or score.
    rng=np.random.default_rng(args.seed)
    observed=sorted(rng.choice(sorted(groups),size=min(args.max_images,len(groups)),replace=False).tolist())
    device=args.device
    model,_=load_model(args.checkpoint,device)
    grid=[Params(threshold=t,boundary=b,min_area=a,closing=3,min_seed=12)
          for t,b,a in product([.30,.40,.50],[.50,.65],[20,50,100])]
    summaries=[Counts() for _ in grid]
    for n,stem in enumerate(observed,1):
        records_for_image=groups[stem]
        image=read_gray(records_for_image[0]['path'])
        probabilities=predict_probabilities(model,image,args.tile,args.stride,args.batch,device)
        truth=[polygons_to_instances(r['ann'],image.shape) for r in records_for_image]
        for i,params in enumerate(grid):
            pred=to_instances(probabilities[0],probabilities[1],params)
            for gt in truth:
                summaries[i]+=compare(gt,pred)
        print(f'calibrated {n}/{len(observed)}: {stem}',flush=True)
    ranking=sorted(range(len(grid)),key=lambda i:summaries[i].pq,reverse=True)
    results=[{'params':asdict(grid[i]),'pq':summaries[i].pq,'tp':summaries[i].tp,
              'fp':summaries[i].fp,'fn':summaries[i].fn} for i in ranking]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'tuning_results.json').write_text(json.dumps(results,indent=2))
    (out/'best_params.json').write_text(json.dumps(results[0]['params'],indent=2))
    print('Best date-heldout PQ:',results[0],flush=True)
    print('Note: this is a validation SUBSET score, not a Kaggle score.',flush=True)

if __name__=='__main__': main()
