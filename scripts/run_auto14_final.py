"""One Nov 7 finalization: genuine held-out validation and guarded Kaggle upload.

No local-tuning-to-official score equivalence is asserted. No submission or
training is allowed until one immutable candidate passes independent local PQ.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solar.auto14_plan import best_result,UPSTREAM_SHA
from scripts.build_auto14_final_notebook import build,KERNEL_ID,MIN_CONFIRMATION_PQ,DAY2_VERIFICATION_KERNEL
from scripts.run_auto14_daily import recover_existing,gpu_hours

DATE=dt.date(2026,11,7)
STATE=ROOT/"reports/AUTO14_STATE.json"
REPORT=ROOT/"reports/AUTO14_FINAL.json"
COMPETITION="filament-segmentation-2026"
STOP={"COMPLETE","ERROR","FAILED","CANCELED","CANCELLED"}


def save(report,path=REPORT):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(".pending")
    pending.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    pending.replace(path)


def candidate(state,minimum_completed=3,required_day=None):
    """Select only immutable completed checkpoints using local scores or frozen day."""
    if type(minimum_completed) is not int or not 1<=minimum_completed<=14:
        raise ValueError("Invalid candidate minimum")
    if required_day is not None and (type(required_day) is not int or not 1<=required_day<=14):
        raise ValueError("Invalid requested frozen training day")
    results=state.get("results",{})
    key=best_result(results) if required_day is None else str(required_day)
    if key is None:raise RuntimeError("No successfully trained locally scored candidates")
    if sum(record.get("status")=="COMPLETE" for record in results.values() if isinstance(record,dict))<minimum_completed:
        raise RuntimeError(
            "Fewer than three completed research days; no final selection"
            if minimum_completed==3 else
            "Fewer than required completed research days; no final selection"
        )
    if key not in results or results[key].get("status")!="COMPLETE":
        raise RuntimeError("Required day has no completed checkpoint")
    record=results[key]
    sha=record.get("checkpoint_sha256")
    if (not isinstance(sha,str) or len(sha)!=64 or
            any(c not in "0123456789abcdef" for c in sha)):
        raise RuntimeError("Chosen checkpoint has no verified SHA256")
    if record.get("source_sha")!=UPSTREAM_SHA:
        raise RuntimeError("Chosen checkpoint source differs from frozen MIT commit")
    if type(record.get("day")) is not int or int(key)!=record["day"]:
        raise RuntimeError("Selected day provenance mismatch")
    return record["day"],sha


def official_history(api,message):
    """Find prior matching official upload; ambiguous submissions fail closed."""
    for page in range(1,11):
        rows=api.competition_submissions(COMPETITION,page_number=page,page_size=100) or []
        matching=[r for r in rows if getattr(r,"description","")==message]
        if len(matching)>1:raise RuntimeError("More than one prior matching official submission")
        if matching:return int(matching[0].ref)
        if len(rows)<100:return None
    raise RuntimeError("Submission history exceeds safe paging budget")


def final_check(record,chosen_sha):
    if record.get("training") is not False or record.get("official_submission") is not False:
        raise RuntimeError("Final notebook is not inference-only")
    if record.get("source_sha")!=UPSTREAM_SHA or record.get("training_checkpoint_sha256")!=chosen_sha:
        raise RuntimeError("Final notebook provenance differs")
    pq=record.get("confirmation_local_pq")
    if type(pq) not in (float,int) or not math.isfinite(pq) or pq<MIN_CONFIRMATION_PQ or pq>1:
        raise RuntimeError("Candidate did not pass local independent confirmation")
    if record.get("status")!="CSV_READY" or record.get("final_test_images")!=180:
        raise RuntimeError("Final 180-image CSV was not completed")
    if type(record.get("submission_rows")) is not int or record["submission_rows"]<1:
        raise RuntimeError("Invalid final submission row inventory")
    expected=record.get("csv_sha256")
    if not isinstance(expected,str) or len(expected)!=64:
        raise RuntimeError("Final CSV checksum not present")
    return expected


def execute(api,history,report,*,minimum_completed=3,required_day=None,
            kernel_id=KERNEL_ID,report_path=REPORT):
    if kernel_id not in (KERNEL_ID,"minahussain111/solar-autoresearch-early-final-20261025",DAY2_VERIFICATION_KERNEL):
        raise ValueError("Unapproved final Kaggle notebook identity")
    day,sha=candidate(history,minimum_completed=minimum_completed,required_day=required_day)
    report.update(selected_day=day,checkpoint_sha256=sha,
                  candidate_selected_by=("Frozen research day "+str(day) if required_day is not None
                                         else "best prior local tuning PQ; no leaderboard optimisation"),
                  notebook_url="https://www.kaggle.com/code/"+kernel_id)
    remaining=gpu_hours(api)
    report["gpu_hours_remaining_before"]=round(remaining,3)
    if remaining<1.5:
        report.update(status="SKIPPED_QUOTA",reason="Insufficient final inference GPU quota")
        return
    with tempfile.TemporaryDirectory(prefix="auto14-final-") as tmp:
        folder=Path(tmp)
        status=recover_existing(api,kernel_id)
        reference=kernel_id
        if status is None:
            if kernel_id==DAY2_VERIFICATION_KERNEL:
                from scripts.build_day2_verification_notebook import build_day2
                metadata=build_day2(day,sha,folder/"auto14_final.ipynb")
            else:
                metadata=build(day,sha,folder/"auto14_final.ipynb",kernel_id=kernel_id)
            if not metadata.get("is_private") or not metadata.get("enable_gpu"):
                raise RuntimeError("Final Kaggle notebook metadata is unsafe")
            response=api.kernels_push(str(folder),timeout="6000",
                                      acc="NvidiaTeslaT4",no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError("Kaggle did not accept private final experiment")
            version=int(response.versionNumber)
            if version<1:raise RuntimeError("Missing immutable Kaggle execution version")
            reference=kernel_id+"/v"+str(version)
            report["version"]=version
        else:
            report["resumed_existing_notebook"]=True
        deadline=time.monotonic()+6240
        while time.monotonic()<deadline:
            status=recover_existing(api,reference) if reference==kernel_id else (
                str(api.kernels_status(reference).status).split(".")[-1].upper()
            )
            if status in STOP:break
            time.sleep(30)
        report["kaggle_execution_status"]=status
        if status!="COMPLETE":raise RuntimeError("Final inference did not complete")
        output=folder/"output"
        api.kernels_output(reference,str(output),file_pattern=r"(auto14_final\.json|submission\.csv)$")
        records=list(output.rglob("auto14_final.json"))
        if len(records)!=1:raise RuntimeError("No unique final validation evidence in Kaggle outputs")
        record=json.loads(records[0].read_text())
        report.update(candidate_validation_status=record.get("status"),
                      confirmation_local_pq=record.get("confirmation_local_pq"),
                      confirmation_observations=record.get("confirmation_observations"),
                      confirmation_annotation_pairs=record.get("confirmation_annotation_pairs"),
                      final_inference_seconds=record.get("elapsed_seconds"),
                      confirmation_threshold=MIN_CONFIRMATION_PQ)
        if record.get("status")=="REJECTED":
            report.update(status="REJECTED",reason="Independent local PQ gate failed; V5 preserved; no test submission")
            return
        expected=final_check(record,sha)
        csvs=list(output.rglob("submission.csv"))
        if len(csvs)!=1 or hashlib.sha256(csvs[0].read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Final CSV download missing or checksum differs")
        # Never call an old V5 CSV a new trained challenger. The frozen
        # V5 fingerprint comes from real Kaggle v3 artifact inventory.
        if expected=="94b985ff19e0848d655d0a524265ca67d150fa203531ca18780d3a009cd586d6":
            raise RuntimeError("Challenger CSV duplicates old V5 submission")
        report.update(status="CSV_VERIFIED",csv_sha256=expected,final_test_images=180,
                      submission_rows=record["submission_rows"],csv_available_in_private_kaggle=True)
        # Upload is explicitly one-shot, identified by model checkpoint and CSV hash.
        message="Solar Auto14 independent U-Net; sha256:"+expected[:24]
        report["submission_description"]=message
        old=official_history(api,message)
        if old is not None:
            report["submission_ref"]=old
            report["reused_existing_submission"]=True
        else:
            limits=api.competition_get_submission_limits(COMPETITION)
            if int(limits.num_allowed_now)<1:
                report.update(status="CSV_VERIFIED_LIMIT",reason="Kaggle daily official submission limit reached")
                return
            report["official_submission_attempted"]=True
            # This persists within the runner before upload; GitHub step commits
            # it after return. A failed/uncertain upload is NEVER blindly retried.
            save(report,report_path)
            response=api.competition_submit(str(csvs[0]),message,COMPETITION,quiet=True)
            ref=getattr(response,"ref",None)
            if ref is None:raise RuntimeError("Official submission outcome is uncertain; inspect Kaggle account")
            report["submission_ref"]=int(ref)
        report.update(status="SUBMITTED",official_submission=True)
        save(report,report_path)
        until=time.monotonic()+360
        while time.monotonic()<until:
            try:
                submission=api.get_submission(report["submission_ref"])
            except Exception as error:
                if getattr(getattr(error,"response",None),"status_code",None)==404:
                    time.sleep(20);continue
                raise
            status=getattr(submission.status,"name",str(submission.status).split(".")[-1]).upper()
            report["scoring_status"]=status
            score=getattr(submission,"public_score",None)
            if status=="COMPLETE" and score not in (None,""):
                value=float(score)
                if not math.isfinite(value) or not 0<=value<=1:raise RuntimeError("Unexpected official score")
                report.update(status="SCORED",official_public_score=value,
                              official_score_kind="Kaggle public evaluation metric (not local PQ)",
                              official_target_0_70_reached=value>=0.70,
                              official_target_0_60_reached=value>=0.60)
                break
            if "ERROR" in status:break
            time.sleep(20)


def main():
    if dt.datetime.now(dt.timezone.utc).date()!=DATE:
        print("The only authorized finalization date is 2026-11-07 UTC. No work done.")
        return
    if not STATE.exists():
        raise SystemExit("No research ledger available")
    if REPORT.exists():
        prior=json.loads(REPORT.read_text())
        if prior.get("official_submission_attempted") or prior.get("status") in (
                "REJECTED","SKIPPED_QUOTA","SCORED","SUBMITTED","CSV_VERIFIED_LIMIT"):
            print("Existing final result or uncertain official write; refusing repeated upload.")
            return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Missing authenticated Kaggle secret")
    report={"status":"STARTED","date_utc":str(DATE),"training":False,
            "test_inference":False,"official_submission":False}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        execute(api,json.loads(STATE.read_text()),report)
    except Exception as error:
        report["error_type"]=type(error).__name__
        report["error_note"]="Review private Kaggle/Actions logs; never blindly repeat uncertain submission"
        if not report.get("official_submission"):
            report["status"]="FAILED"
    finally:
        save(report)
        print("Finalization:",report["status"],"local confirmation:",report.get("confirmation_local_pq"),
              "official score:",report.get("official_public_score"))
    if report.get("status")=="FAILED":
        raise SystemExit(1)


if __name__=="__main__":main()
