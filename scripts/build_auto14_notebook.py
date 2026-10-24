"""Build a reproducible private Kaggle GPU experiment from a pinned MIT source."""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solar.auto14_plan import UPSTREAM, UPSTREAM_SHA, notebook_slug, choose_experiment

RUN_CODE = r'''
import hashlib, json, subprocess, sys, time
from pathlib import Path
import torch
PLAN = __PLAN__
REPO = __REPO__
REVISION = __REVISION__
DAY = PLAN["day"]
ROOT = Path("/kaggle/working/solar_auto14")
TARGET = ROOT / ("day_%02d" % DAY)
RESULT = Path("/kaggle/working/auto14_result.json")
result = {"status":"STARTED","day":DAY,"training":True,"official_submission":False,
          "test_inference":False,"source_sha":REVISION,"config":PLAN,"research_only":True}
started = time.perf_counter()
try:
    assert torch.cuda.is_available(), "CUDA required for scheduled GPU training"
    assert not TARGET.exists(), "Refuse to overwrite prior experiment artifacts"
    TARGET.mkdir(parents=True, exist_ok=False)
    src = ROOT / "farnoosh_mit_reference"
    subprocess.run(["git","clone","-q",REPO,str(src)],check=True)
    subprocess.run(["git","-C",str(src),"checkout","-q","--detach",REVISION],check=True)
    actual = subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
    assert actual == REVISION, "Pinned MIT source revision mismatch"
    licence = (src/"LICENSE").read_text()
    assert "MIT License" in licence and "Farnoosh Memari" in licence, "Third-party attribution missing"
    original = (src/"scripts/train.py").read_text()
    anchor = "train_stems, val_stems = split_stems(stems, args.val_frac, args.seed)"
    patch = ("development_stems, confirmation_stems = split_stems(stems, 0.15, 2026)\n"
             "        train_stems, val_stems = split_stems(development_stems, 0.18, 2027)")
    split_original = 'json.dumps({"train": train_stems, "val": val_stems}, indent=1)'
    split_patch = 'json.dumps({"train": train_stems, "val": val_stems, "confirmation": confirmation_stems}, indent=1)'
    assert original.count(anchor) == 1 and original.count(split_original) == 1, "Upstream training source drift detected"
    revised = original.replace(anchor, patch).replace(split_original, split_patch)
    import ast
    ast.parse(revised)
    (src/"scripts/train.py").write_text(revised)
    (TARGET/"source_patch_sha256.txt").write_text(hashlib.sha256(revised.encode()).hexdigest()+"\n")
    subprocess.run([sys.executable,"-m","pip","install","-q",
                    "numpy==1.26.4","scipy==1.13.1",
                    "opencv-python-headless==4.10.0.84","pycocotools==2.0.8"],check=True)
    train_dir = TARGET / "training"
    cmd = [sys.executable, "scripts/train.py", "--out",str(train_dir),
           "--scale",str(PLAN["scale"]),"--base",str(PLAN["base"]),
           "--epochs",str(PLAN["epochs"]), "--steps-per-epoch",str(PLAN["steps"]),
           "--batch-size",str(PLAN["batch"]), "--workers","2",
           "--pos-weight",str(PLAN["pos_weight"]),
           "--pos-frac",str(PLAN["pos_frac"]), "--seed",str(PLAN["seed"])]
    subprocess.run(cmd,cwd=src,check=True)
    split = json.loads((train_dir/"split.json").read_text())
    months = [{s[:6] for s in split[k]} for k in ("train","val","confirmation")]
    assert all(months) and not months[0]&months[1] and not months[0]&months[2] and not months[1]&months[2], "Month leakage or empty partitions"
    result["split_counts"] = {k:len(split[k]) for k in ("train","val","confirmation")}
    result["split_months_disjoint"] = True
    tune = [sys.executable,"scripts/tune.py","--run",str(train_dir),
            "--thresholds","0.35,0.45,0.55,0.65",
            "--min-areas","100,250,400,650", "--merge-dists","0,5,10"]
    subprocess.run(tune,cwd=src,check=True)
    post = json.loads((train_dir/"postprocess.json").read_text())
    local = post["validation"]
    pq = float(local["pq"])
    assert 0 <= pq <= 1 and pq == pq, "Local tuning PQ invalid"
    weight_path = train_dir / "best.pt"
    assert weight_path.exists() and weight_path.stat().st_size > 1024, "No checkpoint"
    sha = hashlib.sha256()
    with weight_path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024*1024),b""): sha.update(piece)
    result.update(status="COMPLETE",local_tuning_pq=pq,
                  tuning_pq_status="Exploratory: tuning and model selection reuse same validation",
                  best_checkpoint_sha256=sha.hexdigest(),
                  tuning_metric={k:local.get(k) for k in ("pq","sq","rq","images")},
                  checkpoint_output=str(weight_path),
                  confirmation_evaluated=False,
                  next_step="Preserve checkpoint for one independent final assessment")
except Exception as error:
    result.update(status="FAILED",error_type=type(error).__name__,
                  stage_error="Review private Kaggle logs; do not promote this candidate")
    raise
finally:
    result["elapsed_seconds"]=round(time.perf_counter()-started,1)
    RESULT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
'''


def build(day, out, history=None):
    plan=choose_experiment(day,history)
    code=RUN_CODE.replace("__PLAN__",repr(plan)).replace("__REPO__",repr(UPSTREAM)).replace("__REVISION__",repr(UPSTREAM_SHA))
    ast.parse(code)
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    notebook={"nbformat":4,"nbformat_minor":5,
              "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
              "cells":[
                  {"cell_type":"markdown","metadata":{},"source":[
                      "# Solar AutoResearch — day %02d\n" % day,
                      "MIT upstream credit: Farnoosh Memari, pinned immutable revision.\n",
                      "TRAIN / tuning / untouched-confirmation months are separated.\n",
                      "GPU training; NO competition test inference or official submission.\n"]},
                  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}
              ]}
    out.write_text(json.dumps(notebook,indent=1)+"\n")
    meta={"id":notebook_slug(day),"title":"solar-autoresearch-d%02d" % day,
          "code_file":out.name,"language":"python","kernel_type":"notebook",
          "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
          "enable_internet":True,"dataset_sources":[],
          "competition_sources":["filament-segmentation-2026"],
          "kernel_sources":[],"model_sources":[]}
    (out.parent/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--day",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--history",type=Path)
    args=p.parse_args()
    history=json.loads(args.history.read_text()).get("results",{}) if args.history else {}
    build(args.day,args.output,history)
