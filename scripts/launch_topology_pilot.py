#!/usr/bin/env python3
"""One uniquely named, quota-gated real Kaggle GPU experiment.

Never reuses the daily research kernel identity and NEVER submits test
predictions to the competition. Immutable prior versions are only observed;
network-ambiguous pushes are never repeated. This is a research experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solar.auto14_plan import MAX_GPU_SECONDS, MIN_REMAINING_GPU_HOURS, UPSTREAM_SHA, choose_experiment
from scripts.build_auto14_notebook import build
from scripts.run_auto14_daily import gpu_hours, recover_existing

SLUG = "minahussain111/solar-topology-pilot-20261026"
REPORT = ROOT / "reports/TOPOLOGY_PILOT_20261026.json"
DAY = 7
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELED", "CANCELLED"}


def save(state: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(".pending")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(REPORT)


def validate_result(record: dict, config: dict) -> dict:
    """Only promote numerical *local* evidence from an attested real run."""
    if not isinstance(record, dict) or record.get("status") != "COMPLETE":
        raise ValueError("Notebook has no completed training result")
    if record.get("day") != DAY or record.get("config") != config:
        raise ValueError("Kaggle returned a different experiment plan")
    if (record.get("official_submission") is not False
            or record.get("test_inference") is not False
            or record.get("confirmation_evaluated") is not False
            or record.get("split_months_disjoint") is not True):
        raise ValueError("Data leakage or prohibited evaluation flags")
    if record.get("boundary_weight") != config["boundary_weight"]:
        raise ValueError("Wrong boundary setting")
    if record.get("topology_weight") != config["topology_weight"]:
        raise ValueError("Wrong topology setting")
    for key in ("source_patches_sha256", "best_checkpoint_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing attested SHA256: " + key)
    if record.get("source_sha") != UPSTREAM_SHA:
        raise ValueError("Incorrect pinned upstream revision")
    pq = record.get("local_tuning_pq")
    if type(pq) not in (float, int) or not math.isfinite(pq) or not 0 <= pq <= 1:
        raise ValueError("Invalid local tuning PQ")
    counts = record.get("split_counts")
    if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 20
                                              for k in ("train", "val", "confirmation")):
        raise ValueError("Invalid disjoint partition counts")
    return {
        "local_tuning_pq": float(pq),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "checkpoint_sha256": record["best_checkpoint_sha256"],
        "source_patches_sha256": record["source_patches_sha256"],
        "split_counts": counts,
        "better_than_day2_local_pq": float(pq) > 0.40438,
        "official_public_pq": None,
        "official_submission": False,
    }


def launch_once(api, report: dict, plan: dict) -> str:
    """Return immutable known version or slug; refuse ambiguous reuploads."""
    report["kaggle_notebook"] = "https://www.kaggle.com/code/" + SLUG
    report["config"] = plan
    known = report.get("version")
    if known is not None:
        if type(known) is not int or known < 1:
            raise RuntimeError("Bad immutable version in prior ledger")
        return SLUG + "/v" + str(known)

    status = recover_existing(api, SLUG)
    if status is not None:
        report["resumed_existing"] = True
        return SLUG

    if report.get("push_attempted"):
        raise RuntimeError("Prior uncertain push: no kernel found; refuse duplicate launch")

    remaining = gpu_hours(api)
    report["gpu_hours_remaining_before"] = round(remaining, 3)
    if remaining < MIN_REMAINING_GPU_HOURS:
        report.update(status="SKIPPED_QUOTA",
                      reason="Insufficient free GPU quota for bounded training")
        save(report)
        return ""

    with tempfile.TemporaryDirectory(prefix="solar-topology-pilot-") as temp:
        folder = Path(temp)
        nb = folder / "auto14.ipynb"
        meta = build(DAY, nb)
        if not meta["is_private"] or not meta["enable_gpu"] or meta["kernel_sources"]:
            raise RuntimeError("Notebook GPU/private-source identity invalid")
        meta["id"] = SLUG
        meta["title"] = "solar-topology-pilot-20261026"
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        # Record write intent BEFORE the remote network call. Never speculate
        # that a timed-out push failed and launch an accidental second GPU job.
        report["push_attempted"] = True
        report["status"] = "LAUNCH_REQUESTED"
        save(report)
        response = api.kernels_push(str(folder), timeout=str(MAX_GPU_SECONDS),
                                    acc="NvidiaTeslaT4", no_run=False)
        if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
            raise RuntimeError("Kaggle rejected private training launch")
        version = int(response.versionNumber)
        if version < 1:
            raise RuntimeError("No immutable Kaggle execution reference returned")
        report["version"] = version
        save(report)
    return SLUG + "/v" + str(version)


def main() -> None:
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    if report.get("status") in {"COMPLETE", "SKIPPED_QUOTA"}:
        print("No duplicate training:", report["status"])
        return
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN secret missing. No GPU requested.")
    plan = choose_experiment(DAY)
    assert plan["topology_weight"] > 0 and plan["boundary_weight"] > 0
    report.setdefault("experiment", "Independent topology-aware U-Net private GPU training")
    report.setdefault("official_submission", False)
    report.setdefault("official_public_pq", None)
    report.setdefault("status", "STARTED")
    save(report)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        ref = launch_once(api, report, plan)
        if not ref:
            return
        until = time.monotonic() + MAX_GPU_SECONDS + 180
        state = None
        while time.monotonic() < until:
            state = recover_existing(api, ref)
            if state in TERMINAL:
                break
            time.sleep(25)
        report["kaggle_execution_status"] = state
        if state != "COMPLETE":
            raise RuntimeError("GPU training not COMPLETE; inspect existing Kaggle version")
        with tempfile.TemporaryDirectory(prefix="solar-topology-result-") as temp:
            api.kernels_output(ref, temp, file_pattern=r"auto14_result\.json$")
            items = list(Path(temp).rglob("auto14_result.json"))
            if len(items) != 1:
                raise RuntimeError("Exactly one model result artifact required")
            data = json.loads(items[0].read_text())
        report.update(validate_result(data, plan))
        report["status"] = "COMPLETE"
        save(report)
        print("TRUE GPU model training COMPLETE; LOCAL PQ:", report["local_tuning_pq"])
        print("Independent final and official competition scoring: NOT PERFORMED")
    except Exception as error:
        report["status"] = "ERROR_NEEDS_AUDIT"
        report["error_type"] = type(error).__name__
        report["failure_note"] = "Check Kaggle notebook and Actions logs, never duplicate uncertain upload"
        save(report)
        raise


if __name__ == "__main__":
    main()
