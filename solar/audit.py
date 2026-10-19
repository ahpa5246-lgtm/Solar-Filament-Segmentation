"""Inspect MAGFiLO source before training; no annotations are altered."""
from __future__ import annotations
import argparse
from collections import Counter
import numpy as np
from .data import load_records, split_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='/kaggle/input')
    args=p.parse_args()
    records,train,test=load_records(args.data)
    a,b=split_records(records)
    grouped=Counter(r['stem'] for r in records)
    filaments=[len(r['ann']) for r in records]
    test_files=list(test.glob('*.jpeg'))+list(test.glob('*.jpg'))
    print('Annotated records:',len(records))
    print('Unique H-alpha observations:',len(grouped))
    print('Observations with multiple independent annotations:',sum(n>1 for n in grouped.values()))
    print('Training/validation records:',len(a),len(b))
    print('Distinct train/validation days:',len({r['stem'][:8] for r in a}),len({r['stem'][:8] for r in b}))
    print('Filament annotations per record: median',float(np.median(filaments)), 'max',max(filaments))
    print('Test images:',len(test_files))
    print('Train folder:',train,'Test folder:',test)

if __name__=='__main__':main()
