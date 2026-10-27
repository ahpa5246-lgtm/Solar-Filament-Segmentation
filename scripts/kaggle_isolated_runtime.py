"""Create an isolated Kaggle inference environment without mutating kernel NumPy.

The 2026-10-26 immutable day-two confirmation notebook failed at numpy.rec,
even after pip pinning and a child process. Install compiled dependencies into
a separate venv with system Torch visible, then run *all* confirmation/test
inference in that environment. No Kaggle calls or data reads occur in this
module. Used only by a new, not-yet-launched final notebook.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PACKAGES = (
    "numpy==1.26.4", "scipy==1.13.1",
    "opencv-python-headless==4.10.0.84", "pycocotools==2.0.11",
)


def wrap_final_inference(source: str) -> str:
    """Generate a notebook cell that fails before sealed validation on ABI errors."""
    if not isinstance(source, str) or not source:
        raise ValueError("Missing inference source")
    ast.parse(source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    wrapper = '''"""Kaggle final inference: isolated pinned NumPy environment."""
import hashlib
import subprocess
import sys
from pathlib import Path

SOURCE = __SOURCE__
EXPECTED_SHA = __SHA__
PACKAGES = __PACKAGES__
root = Path("/kaggle/working")
environment = root / "solar_final_isolated_venv"
driver = root / "solar_final_isolated_inference.py"
if environment.exists() or driver.exists():
    raise RuntimeError("Refusing to reuse existing final inference environment")
if hashlib.sha256(SOURCE.encode("utf-8")).hexdigest() != EXPECTED_SHA:
    raise RuntimeError("Frozen inference source SHA256 mismatch")
# --system-site-packages retains Kaggle's CUDA Torch; the venv site-packages
# take precedence for NumPy, OpenCV, SciPy and COCO RLE.
subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                str(environment)], check=True)
python = environment / "bin" / "python"
subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir",
                "--no-deps", "--ignore-installed", *PACKAGES], check=True)
preflight = r"""
import numpy as np
# NumPy 1.26 exposes np.rec but does not provide the numpy.rec import path.
# A checkpoint serialized in a NumPy 2 session may reference that path.
import sys
if not hasattr(np, "rec") or not hasattr(np.rec, "recarray"):
    raise RuntimeError("NumPy record-array API missing")
sys.modules.setdefault("numpy.rec", np.rec)
import numpy.rec
import scipy
import cv2
from pycocotools import mask as coco
import torch
from pathlib import Path
import sys
# Fail before opening sealed confirmation data if GPU or checkpoint
# deserialization compatibility prerequisites are unavailable.
assert torch.cuda.is_available(), "CUDA unavailable in isolated inference process"
import pickle
probe = np.rec.array([(1,)], dtype=[("value", "i4")])
assert np.array_equal(pickle.loads(pickle.dumps(probe)), probe), "NumPy record-array pickle compatibility failed"
assert np.__version__ == "1.26.4", "NumPy version mismatch"
assert Path(np.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "NumPy not isolated"
m = np.zeros((16, 16), dtype=np.uint8)
m[2:5, 4:7] = 1
r = coco.encode(np.asfortranarray(m))
assert int(coco.area(r)) == 9
assert np.array_equal(coco.decode(r), m)
print("Isolated NumPy, numpy.rec, SciPy, OpenCV, pycocotools and Torch import OK")
"""
# This check occurs before any confirmation image is opened.
subprocess.run([str(python), "-c", preflight], check=True)
driver.write_text(SOURCE, encoding="utf-8")
subprocess.run([str(python), str(driver)], check=True)
'''
    wrapper = (wrapper.replace("__SOURCE__", repr(source))
               .replace("__SHA__", repr(digest))
               .replace("__PACKAGES__", repr(PACKAGES)))
    ast.parse(wrapper)
    return wrapper
