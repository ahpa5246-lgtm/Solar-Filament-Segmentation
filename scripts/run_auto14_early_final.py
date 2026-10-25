"""One sealed early official Kaggle assessment of the genuinely trained day-two U-Net.

The user explicitly authorized a current leaderboard score. Reuse frozen day-two
checkpoint, evaluate never-seen confirmation months, and submit exactly one
validated 180-image CSV only if independent local PQ >=0.36. An unchanged V5
submission, an untrained model, an uncertain API write, or an invalid COCO
file cannot pass the gate.

All dataset inputs and trained weights remain PRIVATE in Kaggle.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.run_auto14_final import execute
from scripts.run_auto14_daily import load_state

TZ=dt.timezone(dt.timedelta(hours=3))
DATE=dt.date(2026,10,25)
REPORT=ROOT/"reports/AUTO14_EARLY_FINAL.json"
KERNEL_ID="minahussain111/solar-autoresearch-early-final-20261025"
STATE=ROOT/"reports/AUTO14_STATE.json"

def date_allowed(now=None):
    now=now or dt.datetime.now(dt.timezone.utc)
    return now.astimezone(TZ).date()==DATE

def preflight(state):
    row=state.get("results",{}).get("2",{})
    if row.get("status")!="COMPLETE" or row.get("execution_status")!="COMPLETE":
        raise RuntimeError("Actual Kaggle training for day two is not complete")
    if row.get("official_submission") is not False or row.get("confirmation_evaluated") is not False:
        raise RuntimeError("Training provenance does not preserve unseen confirmation")
    if row.get("split_counts",{}).get("confirmation",0)<20:
        raise RuntimeError("Fewer than twenty reserved confirmation observations")
    digest=row.get("checkpoint_sha256")
    if not isinstance(digest,str) or len(digest)!=64 or any(x not in "0123456789abcdef" for x in digest):
        raise RuntimeError("No immutable SHA256 trained checkpoint")
    if row.get("local_tuning_pq") is None:
        raise RuntimeError("No local developmental tuning PQ for selected checkpoint")
    return row

def main():
    if not date_allowed():
        print("Early confirmation slot outside Baghdad 2026-10-25; no Kaggle work")
        return
    if REPORT.exists():
        previous=json.loads(REPORT.read_text())
        if previous.get("official_submission_attempted") or previous.get("status") in {
            "SCORED","SUBMITTED","REJECTED","CSV_VERIFIED_LIMIT","SKIPPED_QUOTA",
            "CSV_VERIFIED","STARTED"}:
            print("Earlier early-final assessment or uncertain official write; refusing a duplicate")
            return
        if previous.get("status")=="FAILED" and previous.get("kaggle_execution_status") not in (
            None,"ERROR","FAILED"):
            print("Prior failed run had uncertain Kaggle state; refusing resubmission")
            return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Missing KAGGLE secret, no Kaggle call attempted")
    state=load_state(STATE)
    row=preflight(state)
    report={"date_local":str(DATE),"status":"STARTED","training":False,
            "test_inference":False,"official_submission":False,
            "target_pq":0.60,"checkpoint_sha256":row["checkpoint_sha256"],
            "frozen_training_day":2,
            "validation_rule":"untouched date-grouped confirmation >=0.36",
            "user_requested_immediate_official_assessment":True}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        execute(api,state,report,minimum_completed=1,required_day=2,
                kernel_id=KERNEL_ID,report_path=REPORT)
    except Exception as err:
        report["status"]="FAILED"
        report["error_type"]=type(err).__name__
        report["reason"]="Check private Kaggle/Actions logs; never infer score on error"
    finally:
        REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("Early Kaggle confirmation:",report["status"],
              "heldout local PQ",report.get("confirmation_local_pq"),
              "official PQ",report.get("official_public_score"),
              "actual upload",report.get("official_submission",False))
    if report["status"]=="FAILED":
        raise SystemExit(1)

if __name__=="__main__":
    main()
