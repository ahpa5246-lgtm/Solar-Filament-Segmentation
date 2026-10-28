"""Idempotent one-version launcher/monitor for a bounded no-training V5 ablation."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_v5_interpolation import build, KERNEL

STATE = ROOT / "reports/V5_INTERPOLATION_STATE.json"
FINETUNE = ROOT / "reports/V5_FINETUNE_STATE.json"


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    pending = STATE.with_suffix(".pending")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(STATE)


def retrieve(api, reference, folder):
    token = ""
    for _ in range(60):
        for delay in (0, 30, 60, 120):
            if delay:
                time.sleep(delay)
            try:
                _, following = api.kernels_output(reference, str(folder),
                    file_pattern=r"interpolation_result\.json$", page_token=token, page_size=100)
                break
            except Exception as error:
                if getattr(getattr(error, "response", None), "status_code", None) != 429 or delay == 120:
                    raise
        files = list(Path(folder).rglob("interpolation_result.json"))
        if len(files) == 1:
            return json.loads(files[0].read_text())
        if not following:
            break
        token = following
    raise RuntimeError("Completed immutable output unavailable; do not launch again")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    training = json.loads(FINETUNE.read_text())
    result = training.get("result", {})
    if (training.get("status") != "COMPLETE" or training.get("version") != 2
            or result.get("status") != "COMPLETE" or result.get("best_epoch") != 0
            or result.get("epochs_completed") != 16 or result.get("local_gain_supported") is not False):
        raise RuntimeError("Only the verified completed no-gain v2 is eligible")
    expected_sha = result["best_checkpoint_sha256"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not state:
        quota = api.quota_view().gpu_quota
        remaining = (quota.total_time_allowed - quota.time_used).total_seconds() / 3600
        if remaining < 1.5:
            raise RuntimeError("GPU quota guard: need 1.5 hours")
        state = dict(status="LAUNCH_REQUESTED", push_attempted=True, training=False,
                     official_submission=False, source_reference=training["reference"],
                     expected_best_checkpoint_sha256=expected_sha,
                     gpu_hours_remaining_before=remaining)
        with tempfile.TemporaryDirectory(prefix="v5-interpolation-build-") as temp:
            build(Path(temp) / "interpolation.ipynb", expected_sha)
            save(state)  # Commit this intent before the API write; no automatic retry.
            response = api.kernels_push(temp, timeout="3900", acc="NvidiaTeslaT4", no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError("Kaggle rejected interpolation inputs; intent retained")
            version = int(response.versionNumber)
            if version < 1:
                raise RuntimeError("Ambiguous push version; inspect before retry")
            state.update(version=version, reference=KERNEL + "/v" + str(version), status="ACCEPTED")
            save(state)
            print("V5_INTERPOLATION_ACCEPTED", state["reference"], flush=True)
        return
    if not state.get("reference"):
        raise RuntimeError("Ambiguous launch intent; inspect Kaggle manually, never repush")
    if state.get("status") == "COMPLETE" and state.get("result", {}).get("status") == "COMPLETE":
        print("Existing immutable result retained", flush=True)
        return
    deadline = time.monotonic() + 4400
    while time.monotonic() < deadline:
        status = str(api.kernels_status(state["reference"]).status).split(".")[-1].upper()
        state["kaggle_status"] = status
        save(state)
        print("V5_INTERPOLATION_STATUS", state["reference"], status, flush=True)
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELED", "CANCELLED"}:
            break
        time.sleep(45)
    if state.get("kaggle_status") != "COMPLETE":
        state["status"] = "NEEDS_DIAGNOSIS"
        # Read the already-accepted immutable version. Never push or rerun it.
        try:
            raw = api.kernels_logs(state["reference"])
            events = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(events, dict):
                events = events.get("logs", events.get("log", [events]))
            if not isinstance(events, list):
                events = [events]
            lines = []
            for event in events:
                entry = event.get("data", "") if isinstance(event, dict) else str(event)
                for line in str(entry).splitlines():
                    line = line.strip()
                    if any(marker in line for marker in (
                        "Traceback", "Error:", "Exception:", "ModuleNotFoundError",
                        "ImportError", "KeyError", "AssertionError", "CUDA out of memory",
                        "No such file", "Killed", "not found", "failed")):
                        # Record only diagnostic lines, never full logs or signed URLs.
                        if "http" not in line and "token" not in line.lower():
                            lines.append(line[:500])
            state["execution_errors"] = lines[-16:]
        except Exception as error:
            state["log_read_error_type"] = type(error).__name__
        save(state)
        raise RuntimeError("Kaggle interpolation incomplete; do not duplicate GPU")
    with tempfile.TemporaryDirectory(prefix="v5-interpolation-read-") as temp:
        state["result"] = retrieve(api, state["reference"], temp)
    state["status"] = "COMPLETE" if state["result"].get("status") == "COMPLETE" else "NEEDS_DIAGNOSIS"
    save(state)
    if state["status"] != "COMPLETE":
        raise RuntimeError("Notebook completed without valid experiment result")
    print("V5_INTERPOLATION_RESULT", json.dumps(state["result"]), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["launch", "monitor"])
    phase = parser.parse_args().phase
    if phase == "launch" and STATE.exists():
        prior = json.loads(STATE.read_text())
        if prior.get("push_attempted"):
            print("Existing launch retained; no duplicate", flush=True)
            sys.exit(0)
    if phase == "monitor" and not STATE.exists():
        raise SystemExit("Cannot monitor before accepted launch")
    main()
