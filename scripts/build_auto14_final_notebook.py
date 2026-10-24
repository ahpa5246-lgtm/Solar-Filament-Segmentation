"""Build one blind-confirmation, then optionally full-test Kaggle inference notebook.

Only the fixed day selected from GitHub local-tuning records is evaluated on the
untouched date-grouped confirmation set. Its thresholds and weights are frozen.
The official competition API is NEVER invoked from the notebook.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solar.auto14_plan import UPSTREAM,UPSTREAM_SHA,notebook_slug

KERNEL_ID="minahussain111/solar-autoresearch-final-2026"
MIN_CONFIRMATION_PQ=0.36
MIN_CONFIRMATION_OBSERVATIONS=20

RUN_CODE=r'''
import hashlib,json,math,subprocess,sys,time
from pathlib import Path

DAY=__DAY__
EXPECTED_SHA=__SHA__
PINNED_SOURCE=__SOURCE__
UPSTREAM=__UPSTREAM__
RESULT=Path("/kaggle/working/auto14_final.json")
FINAL_CSV=Path("/kaggle/working/submission.csv")
report={"status":"STARTED","day":DAY,"training":False,"official_submission":False,
        "test_inference":False,"source_sha":PINNED_SOURCE,
        "selection_metric":"local tuning PQ (selection only, not independent)",
        "confirmation_metric":"held-out local one-to-one PQ (not official metric)",
        "confirmation_threshold":__MIN_PQ__}
started=time.perf_counter()
try:
    import torch
    assert torch.cuda.is_available(),"CUDA required for final assessment"
    import cv2
    import numpy as np
    from pycocotools import mask as coco
    input_root=Path("/kaggle/input")
    matches=[]
    for path in input_root.rglob("best.pt"):
        if not path.is_file():continue
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for part in iter(lambda:handle.read(1<<20),b""):digest.update(part)
        if digest.hexdigest()==EXPECTED_SHA:matches.append(path)
    if len(matches)!=1:raise RuntimeError("Expected exactly one fingerprint-matching selected checkpoint")
    checkpoint=matches[0]
    training=checkpoint.parent
    for name in ("split.json","postprocess.json"):
        if not (training/name).is_file():raise RuntimeError("Selected notebook checkpoint is missing "+name)
    split=json.loads((training/"split.json").read_text())
    sets={k:set(split[k]) for k in ("train","val","confirmation")}
    months={k:{stem[:6] for stem in value} for k,value in sets.items()}
    if min(map(len,sets.values()))<__MIN_N__:raise RuntimeError("Insufficient partition inventory")
    for a,b in (("train","val"),("train","confirmation"),("val","confirmation")):
        if sets[a]&sets[b] or months[a]&months[b]:
            raise RuntimeError("Train/tuning/confirmation date leakage")
    report["split_counts"]={k:len(sets[k]) for k in sets}
    source=Path("/kaggle/working/final-mit-reference")
    subprocess.run(["git","clone","-q",UPSTREAM,str(source)],check=True)
    subprocess.run(["git","-C",str(source),"checkout","-q","--detach",PINNED_SOURCE],check=True)
    assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()==PINNED_SOURCE
    license_text=(source/"LICENSE").read_text()
    assert "MIT License" in license_text and "Farnoosh Memari" in license_text
    subprocess.run([sys.executable,"-m","pip","install","-q",
                    "numpy==1.26.4","scipy==1.13.1",
                    "opencv-python-headless==4.10.0.84","pycocotools==2.0.8"],check=True)
    sys.path.insert(0,str(source))
    from solarseg.data import resolve_paths,load_records,group_by_stem,record_instance_rles
    from solarseg.infer import get_device,load_checkpoint,predict_image
    from solarseg.postprocess import PostConfig,instances_from_prob
    from solarseg.rle import labels_to_counts,counts_to_rle
    from solarseg.metrics import PQStats,match
    paths=resolve_paths()
    records,_=load_records(paths.annotations,paths.train_images)
    grouped=group_by_stem(records)
    if not sets["confirmation"].issubset(grouped):
        raise RuntimeError("Missing confirmation observations")
    device=get_device("cuda")
    model,config=load_checkpoint(checkpoint,device)
    post=PostConfig.load(training/"postprocess.json")
    reference=PQStats()
    observations=0
    started_confirmation=time.perf_counter()
    for stem in sorted(sets["confirmation"]):
        group=grouped[stem]
        prob,disk=predict_image(model,group[0].image_path,device,
                                config.get("scale",0.5),tile=config.get("tile",512),
                                overlap=config.get("overlap",128),tta=False)
        labels=instances_from_prob(prob,disk,post)
        predicted=[counts_to_rle(value) for value in labels_to_counts(labels)]
        for annotation in group:
            reference+=match(predicted,record_instance_rles(annotation))
        observations+=1
    pq=float(reference.pq)
    if not math.isfinite(pq) or not 0<=pq<=1:raise RuntimeError("Invalid confirmation PQ")
    report.update(confirmation_observations=observations,
                  confirmation_annotation_pairs=sum(len(grouped[stem]) for stem in sets["confirmation"]),
                  confirmation_local_pq=pq,confirmation_local_sq=float(reference.sq),
                  confirmation_local_rq=float(reference.rq),
                  confirmation_tp=int(reference.tp),
                  confirmation_fp=int(reference.fp),
                  confirmation_fn=int(reference.fn),
                  confirmation_seconds=round(time.perf_counter()-started_confirmation,1),
                  training_checkpoint_sha256=EXPECTED_SHA)
    if pq<__MIN_PQ__:
        report.update(status="REJECTED",reason="Unseen local confirmation PQ below fixed 0.36 threshold; V5 retained")
    else:
        if paths.test_images is None:raise RuntimeError("Official test-images directory absent")
        files=sorted(p for p in paths.test_images.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
        if len(files)!=180 or len({p.stem for p in files})!=180:
            raise RuntimeError("Official test inventory differs from 180 unique images")
        result=subprocess.run([sys.executable,str(source/"scripts/predict.py"),
                      "--run",str(training),"--images",str(paths.test_images),
                      "--out",str(FINAL_CSV),"--device","cuda"],
                      cwd=source,check=True)
        subprocess.run([sys.executable,str(source/"scripts/check_submission.py"),
                        "--csv",str(FINAL_CSV),"--images",str(paths.test_images)],
                        cwd=source,check=True)
        summary=json.loads(FINAL_CSV.with_suffix(".summary.json").read_text())
        if summary.get("images")!=180 or summary.get("rows",0)<1:
            raise RuntimeError("Incomplete final CSV")
        digest=hashlib.sha256()
        with FINAL_CSV.open("rb") as handle:
            for part in iter(lambda:handle.read(1<<20),b""):digest.update(part)
        report.update(status="CSV_READY",test_inference=True,
                      final_test_images=180,submission_rows=int(summary["rows"]),
                      csv_sha256=digest.hexdigest(),
                      note="New trained U-Net challenger, independently locally confirmed; OFFICIAL score unknown")
except Exception as error:
    report.update(status="FAILED",error_type=type(error).__name__,
                  reason="Review private Kaggle notebook error logs; no promotion on failure")
    raise
finally:
    report["elapsed_seconds"]=round(time.perf_counter()-started,1)
    RESULT.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
'''


def build(day,checkpoint_sha,out):
    if type(day) is not int or day not in range(1,15):raise ValueError("Invalid chosen day")
    if not isinstance(checkpoint_sha,str) or len(checkpoint_sha)!=64 or any(c not in "0123456789abcdef" for c in checkpoint_sha):
        raise ValueError("Invalid selected checkpoint sha256")
    code=(RUN_CODE.replace("__DAY__",str(day))
          .replace("__SHA__",repr(checkpoint_sha))
          .replace("__SOURCE__",repr(UPSTREAM_SHA))
          .replace("__UPSTREAM__",repr(UPSTREAM))
          .replace("__MIN_PQ__",repr(MIN_CONFIRMATION_PQ))
          .replace("__MIN_N__",repr(MIN_CONFIRMATION_OBSERVATIONS)))
    ast.parse(code)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    nb={"nbformat":4,"nbformat_minor":5,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "cells":[{"cell_type":"markdown","metadata":{},"source":[
            "# Solar AutoResearch: one untouched confirmation and final test inference\n",
            "Model trained on our Kaggle account from Farnoosh Memari MIT-licensed source.\n",
            "One immutable best prior day selected by LOCAL tuning only; no retuning after confirmation.\n",
            "The output submission.csv is NOT an official score. See original upstream LICENSE.\n"]},
            {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
             "source":code.splitlines(keepends=True)}]}
    out.write_text(json.dumps(nb,indent=1)+"\n")
    metadata={"id":KERNEL_ID,"title":"solar-autoresearch-final-2026",
              "code_file":out.name,"language":"python","kernel_type":"notebook",
              "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
              "enable_internet":True,"dataset_sources":[],
              "competition_sources":["filament-segmentation-2026"],
              "kernel_sources":[notebook_slug(day)],"model_sources":[]}
    (out.parent/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--day",type=int,required=True)
    p.add_argument("--checkpoint-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    build(args.day,args.checkpoint_sha,args.output)
