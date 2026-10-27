"""One-shot immediate Kaggle confirmation of the frozen day-two checkpoint."""
import json
import os
from pathlib import Path
from scripts.run_auto14_final import execute, STATE
from scripts.build_auto14_final_notebook import IMMEDIATE_KERNEL

REPORT=Path("reports/AUTO14_IMMEDIATE_20261027.json")

def main():
    if REPORT.exists():
        print("An immediate attempt is already recorded; do not retry ambiguous remote writes.")
        return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE secret unavailable")
    state=json.loads(STATE.read_text())
    record=state["results"]["2"]
    if record.get("status")!="COMPLETE" or record.get("confirmation_evaluated") is not False:
        raise RuntimeError("Day-two candidate not eligible for independent confirmation")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    report={"status":"STARTED","training":False,"official_submission":False,
            "official_submission_attempted":False,"notebook_id":IMMEDIATE_KERNEL}
    REPORT.write_text(json.dumps(report,indent=2)+"\n")
    try:
        execute(api,state,report,minimum_completed=1,required_day=2,
                kernel_id=IMMEDIATE_KERNEL,report_path=REPORT)
    except Exception as exc:
        report["status"]="FAILED"
        report["error_type"]=type(exc).__name__
        report["reason"]="Inspect private Kaggle logs before any manual retry"
        raise
    finally:
        REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
