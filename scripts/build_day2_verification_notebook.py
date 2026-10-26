"""Build the single day-two checkpoint verification notebook.

Fixes the diagnosed Kaggle NumPy import-state error by setting pinned package
versions in the parent notebook, checking numpy.rec in a clean subprocess, then
running the already-reviewed immutable inference source in a new Python
interpreter. The child performs all sealed confirmation and guarded inference.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from scripts.build_auto14_final_notebook import build, DAY2_VERIFICATION_KERNEL

DAY = 2
CHECKPOINT = "d213d3d8bb616a6b9d44fef1219f48727eed148fe68a79e7c0800f534af97a3f"
PACKAGES = (
    "numpy==1.26.4", "scipy==1.13.1",
    "opencv-python-headless==4.10.0.84", "pycocotools==2.0.11",
)

def build_day2(day: int, checkpoint_sha: str, out: Path) -> dict:
    if day != DAY or checkpoint_sha != CHECKPOINT:
        raise ValueError("Day-2-only genuine trained checkpoint fingerprint required")
    out = Path(out)
    meta = build(day, checkpoint_sha, out, kernel_id=DAY2_VERIFICATION_KERNEL)
    if (not meta["is_private"] or not meta["enable_gpu"]
            or meta["kernel_sources"] != ["minahussain111/solar-autoresearch-d02"]):
        raise RuntimeError("Private checkpoint source binding is wrong")
    notebook = json.loads(out.read_text())
    child = "".join(notebook["cells"][1]["source"])
    ast.parse(child)
    if CHECKPOINT not in child or "confirmation_local_pq" not in child:
        raise RuntimeError("Lost immutable checkpoint or independent confirmation")
    if "competition_submit" in child or "competition_submissions" in child:
        raise RuntimeError("The private notebook must never perform official submission")
    # The parent imports only stdlib, not NumPy or Torch. Kaggle/IPython can
    # pre-import binary extensions; all inference happens in a child process.
    # This also fails before any held-out read if numpy.rec cannot import.
    preamble = '''import hashlib, json, subprocess, sys
from pathlib import Path
SOURCE = __SOURCE__
PACKAGES = __PACKAGES__
EXPECTED_SOURCE_HASH = __HASH__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
preflight = r"""
import numpy as np
import numpy.rec
import scipy
import cv2
from pycocotools import mask as coco
import torch
print("clean numpy.rec & compiled PyTorch dependencies PASS", np.__version__)
"""
subprocess.run([sys.executable, "-c", preflight], check=True)
actual = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
assert actual == EXPECTED_SOURCE_HASH, "Notebook inference source altered"
driver = Path("/kaggle/working/solar_day2_fixed_inference.py")
assert not driver.exists(), "Refuse to overwrite prior inference runner"
driver.write_text(SOURCE, encoding="utf-8")
subprocess.run([sys.executable, str(driver)], check=True)
'''
    preamble = (preamble.replace("__SOURCE__", repr(child))
                 .replace("__PACKAGES__", repr(PACKAGES))
                 .replace("__HASH__", repr(hashlib.sha256(child.encode("utf-8")).hexdigest())))
    ast.parse(preamble)
    notebook["cells"][0]["source"].extend([
        "Environment repair: pin packages before an isolated Python child process.\n",
        "No repeat of failed 2026-10-25 immutable Kaggle kernel; selected trained day-two weights only.\n",
        "Official upload, if permitted, is exclusively in GitHub Actions after CSV verification.\n",
    ])
    notebook["cells"][1]["source"] = preamble.splitlines(keepends=True)
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    return meta
