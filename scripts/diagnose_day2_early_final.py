"""Read-only, sanitized Kaggle day-two early-final failure diagnosis.

Uses Kaggle-owned private notebook v1, never creates notebooks, reads test
data or submits competition CSV. Public report contains code error class and
relative script basename only; no raw private logs, images, or predictions.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

REF = "minahussain111/solar-autoresearch-early-final-20261025/v1"
OUT = Path("reports/DAY2_EARLY_FAILURE_DIAGNOSIS.json")
ALLOWED = ("AssertionError", "ValueError", "RuntimeError", "KeyError",
           "TypeError", "ModuleNotFoundError", "ImportError", "FileNotFoundError",
           "OSError", "AttributeError", "PermissionError", "MemoryError",
           "SystemExit", "CalledProcessError", "CUDAOutOfMemoryError",
           "NameError", "UnboundLocalError", "IndexError", "NotImplementedError")
KEYS = ("status", "error_type", "training", "official_submission",
        "confirmation_local_pq", "confirmation_observations",
        "final_test_images", "source_sha", "day", "test_inference",
        "elapsed_seconds", "training_checkpoint_sha256")

def extract(text: str) -> dict:
    # Do not emit log text or hidden competition assets.
    return {
        "log_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "log_bytes": len(text.encode("utf-8")),
        "exception_types": sorted(set(re.findall(r"\b(?:"+"|".join(ALLOWED)+r")\b", text))),
        "python_frames": [
            {"file": name, "line": int(number)}
            for name, number in re.findall(
                r"""File ["'](?:[^"'\n\r]*/)?([A-Za-z0-9_]+\.py)["'], line (\d{1,6})""",
                text,
            )[-25:]
        ],
        "missing_modules": sorted(set(re.findall(
            r"""No module named [\u0027\u0022]([A-Za-z_][A-Za-z_0-9.]{0,100})[\u0027\u0022]""", text
        )))[:20],
        "missing_attributes": sorted(set(re.findall(
            r"""has no attribute ['"]([A-Za-z_][A-Za-z_0-9]{0,80})['"]""", text
        )))[:20],
        "traceback_count": text.count("Traceback (most recent call last)"),
    }

def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("No Kaggle read-only authentication")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    report = {"observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "notebook": "https://www.kaggle.com/code/minahussain111/solar-autoresearch-early-final-20261025",
              "requested_immutable_version": 1,
              "read_only": True, "gpu_jobs_launched": 0, "submission_attempted": False}
    try:
        state = api.kernels_status(REF)
        report["execution_status"] = str(state.status).split(".")[-1].upper()
    except Exception as error:
        report["status_error_type"] = type(error).__name__
    try:
        raw = api.kernels_logs(REF)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except (ValueError, TypeError):
                decoded = raw
        else:
            decoded = raw
        if isinstance(decoded, list):
            full = "\n".join(str(item.get("data", "")) for item in decoded if isinstance(item, dict))
        else:
            full = str(decoded)
        report["logs"] = extract(full)
    except Exception as error:
        report["logs_error_type"] = type(error).__name__
    try:
        with tempfile.TemporaryDirectory(prefix="day2-diagnose-") as tmp:
            api.kernels_output(REF, tmp, file_pattern=r"auto14_final\.json$")
            paths = list(Path(tmp).rglob("auto14_final.json"))
            report["result_manifest_count"] = len(paths)
            if len(paths) == 1:
                result = json.loads(paths[0].read_text())
                report["result_manifest"] = {key: result.get(key) for key in KEYS}
    except Exception as error:
        report["manifest_error_type"] = type(error).__name__
    OUT.parent.mkdir(exist_ok=True, parents=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("Immutable Kaggle early-final:",report.get("execution_status"))
    print("Exception types:",report.get("logs",{}).get("exception_types"))
    print("Sanitized manifest:", report.get("result_manifest",{}).get("status"))
    print("No GPU requested, no submissions, no sealed confirmation evaluation.")

if __name__=="__main__":
    main()
