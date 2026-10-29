"""Launch or monitor one immutable head-only V5 Kaggle experiment.

Reuse the audited Kaggle idempotency, pagination, quota and read-only monitoring
path; keep the previously completed hard-negative version untouched.
"""
import argparse
import json
import sys
from pathlib import Path

# Direct `python scripts/run_v5_headonly.py` runs with scripts/ as sys.path[0].
# Include the repository root before importing the shared package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v5_hardnegative as engine
from scripts.build_v5_headonly import build, KERNEL

ROOT = Path(__file__).resolve().parents[1]
HARD_STATE = ROOT / "reports/V5_HARDNEGATIVE_STATE.json"
HEAD_STATE = ROOT / "reports/V5_HEADONLY_STATE.json"
_base_prerequisites = engine.prerequisites
_base_refuse = engine.refuse_untracked_kernel


def prerequisites(training, interpolation):
    _base_prerequisites(training, interpolation)
    hard = json.loads(HARD_STATE.read_text())
    result = hard.get("result", {})
    if not (hard.get("status") == "COMPLETE"
            and hard.get("version") == 1
            and hard.get("reference") == "minahussain111/solar-v5-hardnegative-pq50/v1"
            and hard.get("kaggle_status") == "COMPLETE"
            and result.get("status") == "COMPLETE"
            and result.get("epochs_completed") == 4
            and result.get("best_epoch") == 0
            and result.get("local_gain_supported") is False
            and result.get("best_checkpoint_sha256")
                == training["result"]["best_checkpoint_sha256"]):
        raise ValueError("Completed no-gain V5 hard-negative v1 required")
    return hard["reference"]


def refuse_untracked_kernel(api, known_references):
    _base_refuse(api, known_references)
    hard_ref = prerequisites(
        json.loads(engine.FINETUNE.read_text()),
        json.loads(engine.INTERPOLATION.read_text()))
    observed = str(api.kernels_status(hard_ref).status).split(".")[-1].upper()
    if observed != "COMPLETE":
        raise RuntimeError("Hard-negative prerequisite not COMPLETE on Kaggle")


def main(phase):
    engine.STATE = HEAD_STATE
    engine.KERNEL = KERNEL
    engine.build = build
    engine.HYPOTHESIS = (
        "Freeze V5 backbone and RPN; tune ROI heads only, 4 epochs LR 1e-6")
    engine.prerequisites = prerequisites
    engine.refuse_untracked_kernel = refuse_untracked_kernel
    engine.main(phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("launch", "monitor"), required=True)
    main(parser.parse_args().phase)
