"""One immutable V5 hard-negative Kaggle run; no ambiguous retries or submissions."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_v5_hardnegative import build, KERNEL

STATE = ROOT / "reports/V5_HARDNEGATIVE_STATE.json"
FINETUNE = ROOT / "reports/V5_FINETUNE_STATE.json"
INTERPOLATION = ROOT / "reports/V5_INTERPOLATION_STATE.json"


def prerequisites(training, interpolation):
    tr = training.get("result", {})
    ir = interpolation.get("result", {})
    if not (training.get("status") == "COMPLETE" and training.get("version") == 2
            and tr.get("status") == "COMPLETE" and tr.get("epochs_completed") == 16
            and tr.get("best_epoch") == 0 and tr.get("local_gain_supported") is False):
        raise ValueError("Verified completed no-gain V5 v2 required")
    if not (interpolation.get("status") == "COMPLETE" and interpolation.get("version") == 3
            and ir.get("status") == "COMPLETE" and ir.get("selected_alpha") == 0
            and ir.get("local_gain_supported") is False):
        raise ValueError("Verified completed no-gain interpolation v3 required")


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    pending = STATE.with_suffix(".pending")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(STATE)


def retrieve(api, ref, folder):
    token = ""
    for _ in range(60):
        for delay in (0, 30, 60, 120):
            if delay:
                time.sleep(delay)
            try:
                _, following = api.kernels_output(
                    ref, str(folder), file_pattern=r"finetune_result\.json$",
                    page_token=token, page_size=100)
                break
            except Exception as error:
                if getattr(getattr(error, "response", None), "status_code", None) != 429 or delay == 120:
                    raise
        matches = list(Path(folder).rglob("finetune_result.json"))
        if len(matches) == 1:
            return json.loads(matches[0].read_text())
        if len(matches) > 1:
            raise ValueError("Ambiguous Kaggle result inventory")
        if not following:
            break
        token = following
    raise RuntimeError("Completed immutable output unavailable; never repush")


def main(phase):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    training = json.loads(FINETUNE.read_text())
    interpolation = json.loads(INTERPOLATION.read_text())
    prerequisites(training, interpolation)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if phase == "launch":
        if state:
            print("Existing hard-negative launch intent retained; never duplicate", flush=True)
            return
        for reference in (training["reference"], interpolation["reference"]):
            observed = str(api.kernels_status(reference).status).split(".")[-1].upper()
            if observed != "COMPLETE":
                raise RuntimeError("Prerequisite Kaggle version not terminal COMPLETE")
        quota = api.quota_view().gpu_quota
        remaining = (quota.total_time_allowed - quota.time_used).total_seconds() / 3600
        if remaining < 4.25:
            raise RuntimeError("GPU quota guard: at least 4.25 hours required")
        with tempfile.TemporaryDirectory(prefix="v5-hardnegative-build-") as folder:
            meta = build(Path(folder) / "hardnegative.ipynb")
            if not (meta["is_private"] and meta["enable_gpu"] and meta["id"] == KERNEL):
                raise ValueError("Private Kaggle notebook configuration invalid")
            state = dict(status="LAUNCH_REQUESTED", push_attempted=True, training=True,
                         official_submission=False, kernel=KERNEL,
                         source_checkpoint_sha256=training["result"]["best_checkpoint_sha256"],
                         gpu_hours_remaining_before=remaining,
                         hypothesis="Fitting-only false-positive upweighting; 4 epochs at 2e-6")
            save(state)  # Persist write intent before any API call.
            response = api.kernels_push(folder, timeout="7200", acc="NvidiaTeslaT4", no_run=False)
            if response.error or response.invalidDatasetSources or response.invalidCompetitionSources:
                raise RuntimeError("Kaggle rejected hard-negative inputs; inspect intent")
            version = int(response.versionNumber)
            if version < 1:
                raise RuntimeError("Ambiguous Kaggle version; inspect before any retry")
            state.update(version=version, reference=KERNEL + "/v" + str(version), status="ACCEPTED")
            save(state)
            print("V5_HARDNEGATIVE_ACCEPTED", state["reference"], flush=True)
        return
    if not state.get("reference"):
        raise RuntimeError("No known immutable Kaggle reference; no relaunch")
    if state.get("status") == "COMPLETE" and state.get("result", {}).get("status") == "COMPLETE":
        print("Existing complete result retained", flush=True)
        return
    deadline = time.monotonic() + 7400
    while time.monotonic() < deadline:
        status = str(api.kernels_status(state["reference"]).status).split(".")[-1].upper()
        state["kaggle_status"] = status
        save(state)
        print("V5_HARDNEGATIVE_STATUS", state["reference"], status, flush=True)
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELED", "CANCELLED"}:
            break
        time.sleep(60)
    if state.get("kaggle_status") != "COMPLETE":
        state["status"] = "NEEDS_DIAGNOSIS"
        save(state)
        raise RuntimeError("Kaggle hard-negative job not COMPLETE; never duplicate")
    with tempfile.TemporaryDirectory(prefix="v5-hardnegative-read-") as folder:
        result = retrieve(api, state["reference"], folder)
    state["result"] = result
    state["status"] = "COMPLETE" if result.get("status") == "COMPLETE" else "NEEDS_DIAGNOSIS"
    save(state)
    if state["status"] != "COMPLETE":
        raise RuntimeError("Notebook completed without valid full training result")
    print("V5_HARDNEGATIVE_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("launch", "monitor"))
    main(parser.parse_args().phase)
