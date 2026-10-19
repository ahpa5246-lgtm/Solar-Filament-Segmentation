"""Generate Kaggle submission.csv from competition test JPEGs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .data import discover
from .pipeline import load_model,predict_folder
from .postprocess import Params


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='/kaggle/input')
    ap.add_argument('--checkpoint',default='/kaggle/working/solar_artifacts/best.pt')
    ap.add_argument('--params',default='/kaggle/working/solar_artifacts/best_params.json')
    ap.add_argument('--out',default='/kaggle/working/submission.csv')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tile',type=int,default=512)
    ap.add_argument('--stride',type=int,default=384)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--tta',action='store_true')
    ap.add_argument('--max-images',type=int,default=0,help='DEBUG ONLY: competition requires all images')
    args=ap.parse_args()
    _,_,test_dir=discover(args.data)
    model,_=load_model(args.checkpoint,args.device)
    params=Params(**json.loads(Path(args.params).read_text())) if Path(args.params).exists() else Params()
    predict_folder(model,test_dir,args.out,params,args.tile,args.stride,args.batch,args.device,args.tta,args.max_images)

if __name__=='__main__':main()
