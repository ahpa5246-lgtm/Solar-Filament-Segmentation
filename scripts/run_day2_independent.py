"""One guarded, authenticated day-two Kaggle confirmation and official score.

This is a *single* new notebook identity, distinct from the previous ERROR v1,
and reuses the exact day-two model fingerprint. The earlier failure was
ModuleNotFoundError: numpy.rec, before any held-out PQ was reported.

No duplicate official submissions, no test labels, no retuning on confirmation,
and never re-launch a remote Kaggle job after an ambiguous network response.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_day2_verification_notebook import CHECKPOINT
from scripts.build_auto14_final_notebook import DAY2_VERIFICATION_KERNEL
from scripts.run_auto14_daily import load_state, recover_existing
from scripts.run_auto14_final import execute, candidate, save, official_history, COMPETITION

REPORT = ROOT / "reports/DAY2_INDEPENDENT_20261026.json"
DIAG = ROOT / "reports/DAY2_EARLY_FAILURE_DIAGNOSIS.json"
OLD = ROOT / "reports/AUTO14_EARLY_FINAL.json"
STATE = ROOT / "reports/AUTO14_STATE.json"
SLOT = dt.date(2026, 10, 26)


def preflight(state: dict, original: dict, diagnosis: dict) -> str:
    day, sha = candidate(state, minimum_completed=1, required_day=2)
    record = state["results"]["2"]
    if day != 2 or sha != CHECKPOINT:
        raise RuntimeError("Day-two checkpoint SHA256 mismatch")
    if record.get("execution_status") != "COMPLETE" or record.get("confirmation_evaluated") is not False:
        raise RuntimeError("No complete unconfirmed day-two training")
    if record.get("official_submission") is not False:
        raise RuntimeError("Day-two training already submitted")
    if (original.get("kaggle_execution_status") != "ERROR"
            or original.get("official_submission") is not False
            or original.get("official_submission_attempted")
            or original.get("submission_ref")):
        raise RuntimeError("Prior early-final official submission state not proven safe")
    manifest = diagnosis.get("result_manifest", {})
    if (diagnosis.get("execution_status") != "ERROR"
            or diagnosis.get("result_manifest_count") != 1
            or manifest.get("status") != "FAILED"
            or manifest.get("error_type") != "ModuleNotFoundError"
            or manifest.get("confirmation_local_pq") is not None
            or manifest.get("confirmation_observations") is not None
            or manifest.get("official_submission") is not False):
        raise RuntimeError("Old early-final failure may have evaluated holdout or submitted; fail closed")
    if "numpy.rec" not in diagnosis.get("logs", {}).get("missing_modules", []):
        raise RuntimeError("Error was not the diagnosed NumPy import-state problem")
    return sha


def main():
    # Hard date gate prevents this once-off verification from launching weeks later.
    if dt.datetime.now(dt.timezone.utc).date() != SLOT:
        print("Day-two isolated verification is restricted to UTC 2026-10-26")
        return
    if REPORT.exists():
        print("Day-two verification has already started; no second remote launch or official upload")
        return
    if not all(p.exists() for p in (STATE, OLD, DIAG)):
        raise SystemExit("Missing immutable earlier Kaggle evidence")
    state = load_state(STATE)
    original = json.loads(OLD.read_text())
    diagnosis = json.loads(DIAG.read_text())
    sha = preflight(state, original, diagnosis)
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Authenticated Kaggle credential absent; no remote write")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    # Fail closed if the distinct fixed kernel was somehow previously created.
    existing = recover_existing(api, DAY2_VERIFICATION_KERNEL)
    if existing is not None:
        raise RuntimeError("Day-two verification kernel already exists: never overwrite")
    # API inventory must be readable BEFORE we ask for even an inference run.
    records = api.competition_submissions(COMPETITION, page_number=1, page_size=100)
    if records is None:
        raise RuntimeError("Kaggle submission history unavailable")
    # The run_auto14_final.execute later checks an exact digest-based message
    # across all pages, skips any matching prior upload and checks daily quota.
    report = {
        "status": "STARTED", "date_utc": str(SLOT),
        "frozen_training_day": 2, "checkpoint_sha256": sha,
        "previous_failure": "numpy.rec import, no confirmation score",
        "fix": "isolated Python after pinned NumPy/compiled-extension preflight",
        "notebook_id": DAY2_VERIFICATION_KERNEL,
        "training": False, "official_submission": False,
        "official_submission_attempted": False, "test_inference": False,
        "push_attempted": True, "sealed_confirmation_repeated_for_tuning": False,
        "preexisting_submission_count_page1": len(records),
    }
    # Durable *intent to write* precedes network push, preventing duplicates
    # even after Actions restarts or Kaggle returns an uncertain response.
    save(report, REPORT)
    try:
        execute(api, state, report, minimum_completed=1, required_day=2,
                kernel_id=DAY2_VERIFICATION_KERNEL, report_path=REPORT)
    except Exception as error:
        report["status"] = "FAILED"
        report["error_type"] = type(error).__name__
        report["reason"] = "Inspect immutable Kaggle and Actions logs, no auto retry"
    finally:
        save(report, REPORT)
        print("Actual frozen day-two verification:", report["status"])
        print("Local independent PQ:", report.get("confirmation_local_pq"))
        print("Official Kaggle Public PQ:", report.get("official_public_score"))
        print("Submission reference:", report.get("submission_ref"))
    if report["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
