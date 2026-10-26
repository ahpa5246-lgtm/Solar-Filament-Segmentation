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
    """Check exact launched versions directly, discover only unversioned slugs.

    After a successful push, Kaggle returns a version reference owner/slug/vN.
    Searching the owned list for slug/vN cannot match a kernel ref (which is
    just owner/slug), so treating it as absent stalls polling until timeout.
    """
    parts=slug.split("/")
    if len(parts)==3 and parts[2].startswith("v") and parts[2][1:].isdigit():
        response=api.kernels_status(slug)
        return _status(response.status)
    if len(parts)!=2:
        raise ValueError("Expected owner/notebook or owner/notebook/vN")
    kernel_slug=parts[1]
    owned=api.kernels_list(mine=True,search=kernel_slug,page_size=100)
    if owned is None:
        raise RuntimeError("Owned kernel inventory unavailable; refusing an ambiguous launch")
    matches=[k for k in owned if k is not None and
             str(getattr(k,"ref","")).lower()==slug.lower()]
    if len(matches)>1:
        raise RuntimeError("Ambiguous existing kernel inventory")
    if not matches:
        return None
    response=api.kernels_status(slug)
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
    prior_version=report.get("known_version")
    if prior_version is not None:
        # A launched Kaggle version is an immutable recovery target. If it
        # disappeared, fail closed rather than silently training twice.
        if type(prior_version) is not int or prior_version < 1:
            raise RuntimeError("Invalid recovery version")
        reference=slug+"/v"+str(prior_version)
        status=recover_existing(api,reference)
        report["version"]=prior_version
        report["launch"]="resume_existing"
    else:
        status=recover_existing(api,slug)
        reference=slug
    if status is None and prior_version is None:
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
    # Historical day one has an immutable, already-launched version whose
    # original config has no boundary arm. Do not mistake its evidence for a
    # newer training variant; all upgraded arm outputs must prove the patch.
    if "boundary_weight" in expected:
        if record.get("boundary_weight") != expected["boundary_weight"]:
            raise RuntimeError("Kaggle boundary research configuration mismatch")
        digest = record.get("source_patches_sha256")
        if (not isinstance(digest,str) or len(digest)!=64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise RuntimeError("No verified source patch fingerprint from Kaggle")
        report["boundary_weight"] = record["boundary_weight"]
        report["source_patches_sha256"] = digest
    if "topology_weight" in expected:
        if record.get("topology_weight") != expected["topology_weight"]:
            raise RuntimeError("Kaggle topology research configuration mismatch")
        if not record.get("source_patches_sha256"):
            raise RuntimeError("Topology source patch not attested")
        report["topology_weight"] = record["topology_weight"]
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
    parser.add_argument("--retry-failed",action="store_true",help="At most one safe retry after an early failure")
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
    prior=history.get(key,{})
    if prior.get("status") in {"COMPLETE","SKIPPED_QUOTA"}:
        print("Day already recorded; no duplicate launch:",day)
        return
    if prior.get("status")=="FAILED":
        safe_retry=(args.retry_failed and prior.get("attempts",1)<=2 and (
            (not prior.get("version") and not prior.get("launch") and
             prior.get("attempts",1)==1) or
            (type(prior.get("version")) is int and prior["version"]>=1 and
             prior.get("launch")=="new")
        ))
        if not safe_retry:
            print("Previous failure is not safe to retry without investigation:",day)
            return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN secret missing; refusing to launch")
    report={"day":day,"utc_day":dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "status":"STARTED","training":True,"test_inference":False,"official_submission":False,
            "attempts":prior.get("attempts",0)+1}
    if type(prior.get("version")) is int and prior.get("launch")=="new":
        report["known_version"]=prior["version"]
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi()
        api.authenticate()
        execute(api,day,history,report)
    except Exception as exc:
        report["status"]="FAILED"
        report["error_type"]=type(exc).__name__
        # Never publish raw API messages: they can contain private URLs or tokens.
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
