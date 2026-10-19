"""GPU training on Kaggle / local Nvidia CUDA. Never uses test masks or metadata."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_records, split_records, TileDataset
from .model import Net, loss_fn


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='/kaggle/input')
    ap.add_argument('--out',default='/kaggle/working/solar_artifacts')
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--steps-per-epoch',type=int,default=280,help='gradient steps each epoch')
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--tile',type=int,default=512)
    ap.add_argument('--base',type=int,default=24)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    if args.device == 'cpu':
        print('WARNING: CPU training is very slow. Kaggle GPU recommended.',flush=True)
    if args.tile % 16:
        ap.error('--tile must be divisible by 16')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records,_,_=load_records(args.data)
    training, validation=split_records(records,seed=args.seed)
    (out/'split_manifest.json').write_text(json.dumps({
        'seed':args.seed,'train_image_ids':sorted(r['image_id'] for r in training),
        'val_image_ids':sorted(r['image_id'] for r in validation),
        'val_observations':sorted({r['stem'] for r in validation})},indent=2))
    ds=TileDataset(training,tile=args.tile,samples_per_epoch=args.steps_per_epoch*args.batch,seed=args.seed)
    vl=TileDataset(validation,tile=args.tile,samples_per_epoch=32,seed=args.seed+1)
    loader=DataLoader(ds,batch_size=args.batch,num_workers=args.workers,pin_memory=args.device=='cuda')
    vloader=DataLoader(vl,batch_size=args.batch,num_workers=0)
    model=Net(base=args.base).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs*len(loader),eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
    best=float('inf');history=[]
    for epoch in range(1,args.epochs+1):
        model.train();running=0.
        for feature,target in loader:
            feature,target=feature.to(args.device,non_blocking=True),target.to(args.device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=args.device=='cuda'):
                loss=loss_fn(model(feature),target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();scheduler.step()
            running+=loss.item()
        model.eval();val_loss=0.
        with torch.inference_mode():
            for feature,target in vloader:
                feature,target=feature.to(args.device),target.to(args.device)
                with torch.autocast(device_type='cuda',enabled=args.device=='cuda'):
                    val_loss+=loss_fn(model(feature),target).item()
        val_loss/=len(vloader)
        train_loss=running/len(loader)
        history.append({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss})
        print(f'epoch {epoch}/{args.epochs} training {train_loss:.4f} validation {val_loss:.4f}',flush=True)
        if val_loss < best:
            best=val_loss
            torch.save({'state_dict':model.state_dict(),'base':args.base,'epoch':epoch,'seed':args.seed,
                        'val_loss':val_loss,'tile':args.tile},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
    print('Saved:',out/'best.pt')

if __name__=='__main__':main()
