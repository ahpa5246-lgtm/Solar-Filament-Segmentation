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
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import torch
PLAN = __PLAN__
REPO = __REPO__
REVISION = __REVISION__
GEOMETRY_SOURCE = __GEOMETRY_SOURCE__
GEOMETRY_TUNER_SOURCE = __GEOMETRY_TUNER_SOURCE__
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
    original_paths = ("scripts/train.py", "solarseg/losses.py", "scripts/tune.py", "solarseg/postprocess.py")
    originals = [(src/path).read_text() for path in original_paths]
    upgrades = patch_auto14_sources(*originals)
    provenance = {}
    for name in original_paths:
        revised = upgrades[name]
        (src/name).write_text(revised)
        provenance[name] = hashlib.sha256(revised.encode()).hexdigest()
    # Private third-party-compatible runtime extension; no original repository
    # source is copied into our repository. Geometric links NEVER paint pixels.
    geometry_path = src/"solarseg/auto14_geometry.py"
    geometry_path.write_text(GEOMETRY_SOURCE)
    tuner_path = src/"scripts/tune_geometry.py"
    tuner_path.write_text(GEOMETRY_TUNER_SOURCE)
    provenance["solarseg/auto14_geometry.py"] = hashlib.sha256(GEOMETRY_SOURCE.encode()).hexdigest()
    provenance["scripts/tune_geometry.py"] = hashlib.sha256(GEOMETRY_TUNER_SOURCE.encode()).hexdigest()
    (TARGET/"source_patches.json").write_text(json.dumps(provenance, sort_keys=True, indent=2)+"\n")
    result["source_patches_sha256"] = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    result["boundary_weight"] = float(PLAN.get("boundary_weight", 0.0))
    subprocess.run([sys.executable,"-m","pip","install","-q",
                    "numpy==1.26.4","scipy==1.13.1",
                    "opencv-python-headless==4.10.0.84","pycocotools==2.0.11"],check=True)
    # Verify the actual compiled COCO extension in a NEW process, not merely
    # that import/installation succeeded. The Kaggle day-1 traceback ended in
    # pycocotools.mask.decode with missing NumPy macro NPY_OWNDATA.
    # The modern 2.0.11 wheel supports the current Python/NumPy C ABI.
    smoke = r"""
import numpy as np
from pycocotools import mask as coco
a=np.zeros((64,64),dtype=np.uint8)
a[12:17,30:35]=1
rle=coco.encode(np.asfortranarray(a))
assert int(coco.area(rle)) == 25
assert np.array_equal(coco.decode(rle),a)
assert np.array_equal(coco.decode(coco.merge([rle])),a)
"""
    subprocess.run([sys.executable,"-c",smoke],check=True)
    train_dir = TARGET / "training"
    cmd = [sys.executable, "scripts/train.py", "--out",str(train_dir),
           "--scale",str(PLAN["scale"]),"--base",str(PLAN["base"]),
           "--epochs",str(PLAN["epochs"]), "--steps-per-epoch",str(PLAN["steps"]),
           "--batch-size",str(PLAN["batch"]), "--workers","2",
           "--pos-weight",str(PLAN["pos_weight"]),
           "--pos-frac",str(PLAN["pos_frac"]), "--seed",str(PLAN["seed"]),
           "--val-every","5"]
    subprocess.run(cmd,cwd=src,check=True,
                   env={**os.environ,"SOLAR_BOUNDARY_WEIGHT":str(PLAN.get("boundary_weight",0.0))})
    split = json.loads((train_dir/"split.json").read_text())
    months = [{s[:6] for s in split[k]} for k in ("train","val","confirmation")]
    assert all(months) and not months[0]&months[1] and not months[0]&months[2] and not months[1]&months[2], "Month leakage or empty partitions"
    result["split_counts"] = {k:len(split[k]) for k in ("train","val","confirmation")}
    result["split_months_disjoint"] = True
    tune = [sys.executable,"scripts/tune.py","--run",str(train_dir),
            "--max-images","24",
            "--thresholds","0.35,0.45,0.55,0.65",
            "--min-areas","100,250,400,650", "--merge-dists","0,5,10"]
    subprocess.run(tune,cwd=src,check=True)
    # Evaluate the experimental orientation-constrained linker on the SAME
    # development-only validation months. If it fails, mark the entire run
    # incomplete rather than using a falsely high local model-selection PQ.
    if DAY in (5,7,9,11,13) and time.perf_counter()-started < 5400:
        geo = [sys.executable,"scripts/tune_geometry.py",
               "--run",str(train_dir),"--max-images","24"]
        subprocess.run(geo,cwd=src,check=True)
        result["geometry_tuning_attempted"] = True
    else:
        result["geometry_tuning_attempted"] = False
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
    patch_source=(ROOT/"solar/auto14_upgrade.py").read_text()
    geometry_source=(ROOT/"solar/auto14_geometry.py").read_text()
    geometry_tuner_source=(ROOT/"scripts/auto14_geometry_tune.py").read_text()
    code=(patch_source+"\n"+RUN_CODE.replace("__PLAN__",repr(plan))
          .replace("__REPO__",repr(UPSTREAM)).replace("__REVISION__",repr(UPSTREAM_SHA))
          .replace("__GEOMETRY_SOURCE__",repr(geometry_source))
          .replace("__GEOMETRY_TUNER_SOURCE__",repr(geometry_tuner_source)))
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
                      "GPU training with optional boundary-aware BCE and balanced tuning-month sampling.\n",
                      "NO competition test inference or official submission.\n"]},
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
