"""Inspect only the immutable failed day-two recovery notebook, no GPU writes.

Sanitize error class, module name, basename frames and manifest. Never publish
raw Kaggle logs, labels, weights or private predictions.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import sys

from scripts.diagnose_day2_early_final import extract, KEYS

REF = "minahussain111/solar-day2-independent-20261026/v1"
OUT = Path("reports/DAY2_RECOVERY_FAILURE_DIAGNOSIS.json")


def main():
    global REF, OUT
    if sys.argv[1:] == ["--immediate"]:
        # Read-only inspection of the actual failed 2026-10-27 Kaggle kernel.
        # This does not relaunch evaluation or inspect sealed labels.
        REF = "minahussain111/solar-day2-confirmation-20261027/v1"
        OUT = Path("reports/AUTO14_IMMEDIATE_FAILURE_DIAGNOSIS.json")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: python -m scripts.diagnose_day2_recovery [--immediate]")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Missing authenticated Kaggle token")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    report = {
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notebook": "https://www.kaggle.com/code/" + REF.split("/v", 1)[0],
        "immutable_version": 1,
        "read_only": True,
        "gpu_jobs_launched": 0,
        "submission_attempted": False,
    }
    try:
        response = api.kernels_status(REF)
        report["execution_status"] = str(response.status).split(".")[-1].upper()
    except Exception as exc:
        report["status_error_type"] = type(exc).__name__
    try:
        raw = api.kernels_logs(REF)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except (ValueError, TypeError):
                decoded = raw
        else:
            decoded = raw
        full = ("\n".join(str(item.get("data", "")) for item in decoded
                          if isinstance(item, dict))
                if isinstance(decoded, list) else str(decoded))
        report["logs"] = extract(full)
        # Categorize failures from remote logs without exposing notebook source,
        # private paths, model weights, data, or authentication tokens.
        import re
        patterns = {
            "failed_package_install": r"(?i)(?:pip install|wheel|requirement|No matching distribution)",
            "failed_venv": r"(?i)(?:venv|ensurepip|virtual environment)",
            "failed_git_clone": r"(?i)(?:git clone|fatal: repository|checkout.*failed)",
            "torch_cuda_error": r"(?i)(?:cuda.*(?:unavailable|failed|error)|libcudart)",
            "numpy_abi_error": r"(?i)(?:numpy\\.rec|NPY_OWNDATA|dtype size changed|multiarray)",
            "checkpoint_error": r"(?i)(?:best\\.pt|checkpoint.*(?:missing|failed|error))",
            "process_error": r"(?i)(?:CalledProcessError|returned non-zero exit status)",
            "notebook_syntax_error": r"(?i)(?:SyntaxError|IndentationError)",
            "permission_error": r"(?i)(?:Permission denied|Errno 13)",
            "disk_space_error": r"(?i)(?:No space left|Errno 28)",
        }
        report["logs"]["failure_indicators"] = {
            label: bool(re.search(pattern, full))
            for label, pattern in patterns.items()
        }
        report["logs"]["process_exit_codes"] = sorted(set(re.findall(
            r"returned non-zero exit status (\\d{1,3})", full
        )))[:5]
        # Only classified metadata is recorded; raw Kaggle logs stay private.
    except Exception as exc:
        report["logs_error_type"] = type(exc).__name__
    try:
        with tempfile.TemporaryDirectory(prefix="day2-recovery-audit-") as folder:
            api.kernels_output(REF, folder, file_pattern=r"auto14_final\.json$")
            matches = list(Path(folder).rglob("auto14_final.json"))
            report["result_manifest_count"] = len(matches)
            if len(matches) == 1:
                manifest = json.loads(matches[0].read_text())
                report["result_manifest"] = {k: manifest.get(k) for k in KEYS}
    except Exception as exc:
        report["manifest_error_type"] = type(exc).__name__
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("Read-only recovery diagnosis:", report.get("execution_status"))
    print("Missing modules:", report.get("logs", {}).get("missing_modules"))
    print("Failure indicators:", report.get("logs", {}).get("failure_indicators"))
    print("Exit codes:", report.get("logs", {}).get("process_exit_codes"))
    print("Manifest:", report.get("result_manifest", {}).get("status"))
    print("No GPU, Kaggle notebook push or official submission.")


if __name__ == "__main__":
    main()
