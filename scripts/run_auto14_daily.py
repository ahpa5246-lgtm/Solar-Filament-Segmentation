"""Run one time-bounded Kaggle GPU research job per UTC day, for fourteen days.

No competition submission is made, no secret/weight/label is committed to Git.
Kernel names are deterministic so uncertain network writes are not retried.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solar.auto14_plan import (
    MAX_GPU_SECONDS, MIN_REMAINING_GPU_HOURS, best_result,
    choose_experiment, day_index, notebook_slug,
)
from scripts.build_auto14_notebook import build

STATE=ROOT/"reports/AUTO14_STATE.json"
STOP={"COMPLETE","ERROR","FAILED","CANCELED","CANCELLED"}


def load_state(path=STATE):
    if not path.exists():
        return {"program":"Solar 14-day independent-challenger training", "status":"active",
                "best_day":None, "results":{}}
    state=json.loads(path.read_text())
    if not isinstance(state.get("results"),dict):
        raise ValueError("Invalid research ledger; refusing to erase previous experiments")
    return state


def save_state(state,path=STATE):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(".pending")
    pending.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    pending.replace(path)


def _status(value):
    return str(value).split(".")[-1].upper()


def recover_existing(api,slug):
    """Fail closed on private/API errors; only HTTP 404 proves absent."""
    try:
        response=api.kernels_status(slug)
    except Exception as exc:
        code=getattr(getattr(exc,"response",None),"status_code",None)
        if code==404:
            return None
        raise
    return _status(response.status)


def gpu_hours(api):
    q=api.quota_view().gpu_quota
    return (q.total_time_allowed-q.time_used).total_seconds()/3600


def execute(api,day,history,report):
    slug=notebook_slug(day)
    report["kaggle_notebook"]="https://www.kaggle.com/code/"+slug
    report["config"]=choose_experiment(day,history)
    report["source_sha"]=report["config"].get("source_sha","ef37d2e93d4cc9e248595e791ee39fe520d392d0")
    available=gpu_hours(api)
    report["gpu_hours_remaining_before"]=round(available,3)
    if available<MIN_REMAINING_GPU_HOURS:
        report.update(status="SKIPPED_QUOTA",reason="Less than 2.25 GPU hours remaining")
        return
    status=recover_existing(api,slug)
    reference=slug
    if status is None:
        with tempfile.TemporaryDirectory(prefix="solar-auto14-") as temp:
            path=Path(temp)/"auto14.ipynb"
            metadata=build(day,path,history)
            assert metadata["is_private"] and metadata["enable_gpu"]
            assert metadata["kernel_sources"]==[] and metadata["competition_sources"]==["filament-segmentation-2026"]
            response=api.kernels_push(str(path.parent),timeout=str(MAX_GPU_SECONDS),
                                      acc="NvidiaTeslaT4",no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError("Kaggle rejected notebook execution or competition input; private logs required")
            version=int(response.versionNumber)
            if version<1:raise RuntimeError("Kaggle did not return an immutable version")
            reference=slug+"/v"+str(version)
            report["version"]=version
            report["launch"]="new"
    else:
        # Pre-existing job could be RUNNING or COMPLETE; never resubmit blindly.
        report["launch"]="resume_existing"
    limit=time.monotonic()+MAX_GPU_SECONDS+240
    while time.monotonic()<limit:
        status=recover_existing(api,reference)
        if status in STOP:
            break
        time.sleep(25)
    report["execution_status"]=status
    if status!="COMPLETE":
        raise RuntimeError("Kaggle execution not complete; do not retry automatically")
    with tempfile.TemporaryDirectory(prefix="solar-auto14-output-") as temp:
        target=Path(temp)
        api.kernels_output(reference,str(target),file_pattern=r"auto14_result\.json$")
        found=list(target.rglob("auto14_result.json"))
        if len(found)!=1:
            raise RuntimeError("Completed notebook did not emit exactly one result")
        record=json.loads(found[0].read_text())
    if record.get("day")!=day or record.get("status")!="COMPLETE" or not record.get("split_months_disjoint"):
        raise RuntimeError("Malformed or failed training output; no local PQ accepted")
    if record.get("official_submission") is not False or record.get("test_inference") is not False:
        raise RuntimeError("Unexpected submission/test flag")
    expected=report["config"]
    if record.get("config")!=expected:
        raise RuntimeError("Kaggle experiment configuration drifted")
    report.update(status="COMPLETE",local_tuning_pq=record["local_tuning_pq"],
                  checkpoint_sha256=record["best_checkpoint_sha256"],
                  split_counts=record["split_counts"],
                  validation_note=record["tuning_pq_status"],
                  confirmation_evaluated=False,
                  elapsed_seconds=record["elapsed_seconds"])
    report["source_sha"]=record["source_sha"]


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--day",type=int,default=None)
    parser.add_argument("--state",type=Path,default=STATE)
    args=parser.parse_args(argv)
    active=day_index(dt.datetime.now(dt.timezone.utc))
    if active is None:
        print("Outside Oct 24–Nov 6, 2026: no GPU work authorized")
        return
    day=args.day or active
    if day!=active:
        raise SystemExit("Manual day must equal today's UTC slot; no backfill/duplicate GPU runs")
    state=load_state(args.state)
    history=state["results"]
    key=str(day)
    if key in history and history[key].get("status") in {"COMPLETE","SKIPPED_QUOTA","FAILED"}:
        print("Day already recorded; no duplicate launch:",day)
        return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN secret missing; refusing to launch")
    report={"day":day,"utc_day":dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "status":"STARTED","training":True,"test_inference":False,"official_submission":False}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi()
        api.authenticate()
        execute(api,day,history,report)
    except Exception as exc:
        report["status"]="FAILED"
        report["error_type"]=type(exc).__name__
        report["reason"]="Check private Actions/Kaggle execution. Never auto-retry an uncertain Kaggle launch."
    history[key]=report
    state["best_day"]=best_result(history)
    state["latest_day"]=day
    state["status"]="complete" if day==14 else "active"
    save_state(state,args.state)
    print("Research experiment state:",report["status"],"day",day,
          "local tuning PQ",report.get("local_tuning_pq"),"best",state["best_day"])
    if report["status"]=="FAILED":
        raise SystemExit(1)


if __name__=="__main__":main()
