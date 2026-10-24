#!/usr/bin/env python
"""Tune conservative filament-tip linking on a small held-in month-balanced slice.

Runs only AFTER the original threshold/area/dilation grid has been frozen.
The baseline and all three geometry settings see the exact same observations.
Confirmation months, test images, and official Kaggle scores are never accessed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random

import numpy as np

from solarseg.data import group_by_stem, load_records, record_instance_rles, resolve_paths
from solarseg.infer import get_device, load_checkpoint, predict_image
from solarseg.metrics import PQStats, match
from solarseg.postprocess import PostConfig, instances_from_prob
from solarseg.auto14_geometry import link_aligned_fragments
from solarseg.rle import counts_to_rle, labels_to_counts


def balanced_sample(stems, maximum=12):
    buckets=defaultdict(list)
    for stem in stems:
        buckets[stem[:6]].append(stem)
    rng=Random(2027)
    for month in sorted(buckets):
        rng.shuffle(buckets[month])
    chosen=[]
    while len(chosen)<min(maximum,len(stems)):
        progressed=False
        for month in sorted(buckets):
            if len(chosen)>=maximum:break
            if buckets[month]:
                chosen.append(buckets[month].pop())
                progressed=True
        if not progressed:break
    return sorted(chosen)


def evaluate_one(labels, truth):
    h,w=labels.shape
    predicted=[counts_to_rle(c,h,w) for c in labels_to_counts(labels)]
    stats=PQStats()
    for gt in truth:
        stats+=match(predicted,gt)
    return stats


def run(run_dir: Path, max_images=12, gaps=(0,16,32,48)):
    config=PostConfig.load(run_dir/"postprocess.json")
    if config.link_gap!=0:
        raise ValueError("Geometry tuner expects an untouched baseline postprocess config")
    split=json.loads((run_dir/"split.json").read_text())
    train,validation,confirmation=(set(split[k]) for k in ("train","val","confirmation"))
    if train & validation or train & confirmation or validation & confirmation:
        raise RuntimeError("Data leakage among split partitions")
    month_sets=({stem[:6] for stem in partition} for partition in (train,validation,confirmation))
    m_train,m_val,m_confirm=month_sets
    if m_train&m_val or m_train&m_confirm or m_val&m_confirm:
        raise RuntimeError("Month leakage among split partitions")
    selected=balanced_sample(sorted(validation),max_images)
    if not selected:
        raise RuntimeError("No development-only validation images to score")
    paths=resolve_paths()
    records,_=load_records(paths.annotations,paths.train_images)
    grouped=group_by_stem(records)
    if not set(selected).issubset(grouped):
        raise RuntimeError("Tuner validation samples missing from loaded labels")
    device=get_device("cuda")
    model,cfg=load_checkpoint(run_dir/"best.pt",device)
    totals={int(g):PQStats() for g in gaps}
    months={int(g):defaultdict(PQStats) for g in gaps}
    for stem in selected:
        anns=grouped[stem]
        prob,disk=predict_image(
            model,anns[0].image_path,device,cfg.get("scale",0.5),
            tile=cfg.get("tile",512),overlap=cfg.get("overlap",128),tta=False
        )
        original=instances_from_prob(prob,disk,config)
        references=[record_instance_rles(ann) for ann in anns]
        for gap in gaps:
            labels=original if not gap else link_aligned_fragments(original,max_gap=int(gap))
            stats=evaluate_one(labels,references)
            totals[int(gap)]+=stats
            months[int(gap)][stem[:6]]+=stats

    baseline=totals[0].pq
    # Minimum practical gain and plurality of monthly improvements protect
    # against adopting a split/merge tweak from just one cherry-picked image.
    needed=(len(m_val)+1)//2
    candidates=[]
    for gap in gaps:
        if not gap:continue
        improved=sum(months[int(gap)][m].pq>months[0][m].pq+1e-9 for m in
                     sorted({s[:6] for s in selected}))
        candidate=totals[int(gap)].pq
        candidates.append((int(gap),candidate,improved))
    allowed=[row for row in candidates if row[1]>=baseline+0.015
             and row[2]>=min(needed,len({s[:6] for s in selected}))]
    selected_gap,best_pq,better_months=(max(allowed,key=lambda x:(x[1],-x[0]))
                                         if allowed else (0,baseline,0))
    # Keep the detailed result in the same postprocess.json that the final
    # notebook reads. All gains are *local tuning* and unconfirmed.
    info=json.loads((run_dir/"postprocess.json").read_text()).get("validation",{})
    info["geometry_trial"]={
        "baseline_local_pq":float(baseline),
        "selected_local_pq":float(best_pq),
        "link_gap":int(selected_gap),
        "improved_validation_months":int(better_months),
        "sample_images":len(selected),
        "candidates":[{"gap":g,"pq":float(pq),"improved_months":n}
                      for g,pq,n in candidates],
        "metric":"LOCAL matching proxy; same development-only labels for selection",
    }
    if selected_gap:
        config.link_gap=int(selected_gap)
        info["pq"]=float(best_pq)
    config.save(run_dir/"postprocess.json",extra=info)
    return info["geometry_trial"]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--max-images",type=int,default=12)
    args=parser.parse_args()
    print(json.dumps(run(args.run,args.max_images),sort_keys=True,indent=2))
