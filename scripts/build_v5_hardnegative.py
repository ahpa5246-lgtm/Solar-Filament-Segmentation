"""Build a separate private Kaggle V5 hard-negative experiment without modifying V5 v2."""
import ast
import hashlib
import json
from pathlib import Path
from build_v5_finetune import build as build_original, ROOT

KERNEL = "minahussain111/solar-v5-hardnegative-pq50"


def build(output):
    output = Path(output)
    meta = build_original(output)
    notebook = json.loads(output.read_text())
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    assigns = {node.targets[0].id: node for node in tree.body
               if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name)}
    driver = ast.parse(ast.literal_eval(assigns["SOURCE"].value))
    source_node = next(node for node in driver.body
                       if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == "sources")
    sources = ast.literal_eval(source_node.value)
    sources["solar/v5_hardnegative.py"] = (ROOT / "solar/v5_hardnegative.py").read_text()
    source_node.value = ast.parse(repr(sources), mode="eval").body
    text = ast.unparse(ast.fix_missing_locations(driver))
    original = "source_path, WORK, seconds=6600)"
    if text.count(original) != 1:
        raise RuntimeError("Original V5 notebook driver call changed")
    text = text.replace(original, "source_path, WORK, seconds=6600, hard_negative=True)")
    ast.parse(text)
    assigns["SOURCE"].value = ast.Constant(text)
    assigns["EXPECTED_SHA"].value = ast.Constant(hashlib.sha256(text.encode()).hexdigest())
    notebook["cells"][1]["source"] = ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# V5 hard-negative detector continuation — train-only hypothesis\n",
        "Original V5 Mask R-CNN + fixed Crop U-Net; archived weights preserved read-only.\n",
        "Mine false-positive proposals on fitting images only, weight fitting samples, train four epochs at LR 2e-6.\n",
        "Select by end-to-end tuning PQ, read confirmation after selection; no test or submission.\n",
        "Historical validation is not pristine; results are local only.\n"]
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    meta.update(id=KERNEL, title=KERNEL.split("/")[1])
    (output.parent / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
