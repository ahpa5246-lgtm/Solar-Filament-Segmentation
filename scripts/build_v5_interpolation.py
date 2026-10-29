"""Build a read-only Kaggle experiment from the immutable completed V5 v2 output."""
import ast
import hashlib
import json
from pathlib import Path
from build_v5_finetune import build as training_build, ROOT

KERNEL = "minahussain111/solar-v5-weight-interpolation-pq50"


def build(output, expected_best_sha):
    if len(expected_best_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_best_sha):
        raise ValueError("Need the verified v2 best checkpoint SHA256")
    output = Path(output)
    meta = training_build(output)
    notebook = json.loads(output.read_text())
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    values = {n.targets[0].id: n for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    old = ast.literal_eval(values["SOURCE"].value)
    driver = ast.parse(old)
    sources = ast.literal_eval(driver.body[0].value)
    sources["solar/v5_weight_interpolation.py"] = (ROOT / "solar/v5_weight_interpolation.py").read_text()
    driver.body[0].value = ast.parse(repr(sources), mode="eval").body
    prefix = ast.unparse(ast.fix_missing_locations(driver)).split("from solar.v5_finetune import run", 1)[0]
    new_driver = prefix + """
from hashlib import sha256
from solar.v5_weight_interpolation import run
from solar.v6_resolution_audit import DETECTOR_SHA
expected_best_sha = """ + repr(expected_best_sha) + """
cache = [p for p in Path('/kaggle/input').rglob('best_detector.pt')
         if 'solar-filament-v5-v3-cache' in str(p)]
if len(cache) != 1: raise RuntimeError('Expected exactly one original cache detector')
if sha256(cache[0].read_bytes()).hexdigest() != DETECTOR_SHA:
    raise RuntimeError('Original detector fingerprint mismatch')
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    one('training_manifest.csv'), one('best_detector_predictions_1024.pkl'),
    one('refined_validation_predictions_2048.pkl'), cache[0], one('best_refiner.pt'),
    source_path, one('last_detector.pt'), one('finetune_result.json'),
    expected_best_sha, WORK, seconds=3300,test_bn_restore=True)
"""
    ast.parse(new_driver)
    values["SOURCE"].value = ast.Constant(new_driver)
    values["EXPECTED_SHA"].value = ast.Constant(hashlib.sha256(new_driver.encode()).hexdigest())
    notebook["cells"][1]["source"] = ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# V5 detector interpolation — hypothesis test, no retraining\\n",
        "Compare frozen original checkpoint with completed v2 epoch-16 weights.\\n",
        "Choose alpha only on tuning; read confirmation once; never access test or submit.\\n",
    ]
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    meta.update(id=KERNEL, title=KERNEL.split("/")[1],
                kernel_sources=["minahussain111/solar-v5-finetune-pq50-20261028"])
    (output.parent / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
