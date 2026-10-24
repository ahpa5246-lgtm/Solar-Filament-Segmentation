"""One user-requested EARLY day-two Kaggle training attempt on Oct 25 Baghdad.

Preserves the deterministic research-day identity and the immutable notebook
slug; the ordinary Oct 25 UTC daily runner will notice the completed or failed
ledger record and MUST NOT upload a duplicate. This one-shot is not a submission.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from solar.auto14_plan import best_result, choose_experiment
from scripts.build_auto14_notebook import build
from scripts.run_auto14_daily import execute, load_state, save_state

DAY = 2
LOCAL_DATE = dt.date(2026, 10, 25)
BAGHDAD = dt.timezone(dt.timedelta(hours=3))
STATE = Path("reports/AUTO14_STATE.json")
STOP_STATES = {"COMPLETE", "FAILED", "SKIPPED_QUOTA", "STARTED"}


def date_allowed(when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    return (when.astimezone(BAGHDAD).date() == LOCAL_DATE and
            when.astimezone(dt.timezone.utc).date() == dt.date(2026, 10, 24))


def validated_day_two(state):
    results = state["results"]
    prior_day = results.get("1", {})
    if prior_day.get("status") != "FAILED" or prior_day.get("execution_status") != "ERROR":
        raise RuntimeError("No confirmed Kaggle day-one failure; an early rescue is not permitted")
    prior = results.get(str(DAY), {})
    if prior:
        # Never retrain a known day-two version regardless of its state.
        raise RuntimeError("Day two already has a ledger entry; refusing an early duplicate")
    if not (STATE.parent / "V5_FINAL_CACHE.json").is_file():
        raise RuntimeError("Baseline model provenance file missing")
    return choose_experiment(DAY, results)


def main():
    if not date_allowed():
        print("Outside one authorized Baghdad Oct 25 early-start slot. No GPU used.")
        return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Missing Kaggle secret")
    state = load_state(STATE)
    config = validated_day_two(state)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    # Recheck at launch time. No GPU is used until the existing deterministic
    # slug has been verified absent using the ordinary fail-closed execute.
    report = {"day": DAY, "utc_day": "2026-10-24",
              "research_slot_date_utc": "2026-10-25",
              "early_baghdad_request": True,
              "status": "STARTED", "training": True, "official_submission": False,
              "test_inference": False, "attempts": 1}
    try:
        execute(api, DAY, state["results"], report)
        if report.get("config") != config:
            raise RuntimeError("Launch config changed while preparing")
    except Exception as err:
        report["status"] = "FAILED"
        report["error_type"] = type(err).__name__
        report["reason"] = "Inspect private Kaggle notebook and official Actions log; no unverified score"
    state["results"][str(DAY)] = report
    state["latest_day"] = DAY
    state["best_day"] = best_result(state["results"])
    save_state(state, STATE)
    print("Early Kaggle result:", report["status"], "day", DAY,
          "notebook", report.get("kaggle_notebook"),
          "official_submission", report["official_submission"])
    if report["status"] == "FAILED":
        raise SystemExit(1)


if __name__=="__main__":
    main()
