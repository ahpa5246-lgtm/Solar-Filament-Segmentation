"""Build a private frozen-backbone/RPN ROI-head V5 experiment from archived weights."""
import ast
import hashlib
import json
from pathlib import Path
from build_v5_finetune import build as build_original, ROOT

KERNEL = "minahussain111/solar-v5-headonly-pq50"


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
    source_node.value = ast.parse(repr(sources), mode="eval").body
    text = ast.unparse(ast.fix_missing_locations(driver))
    original = "source_path, WORK, seconds=6600)"
    if text.count(original) != 1:
        raise RuntimeError("Original V5 notebook driver call changed")
    text = text.replace(original, "source_path, WORK, seconds=6600, head_only=True)")
    ast.parse(text)
    assigns["SOURCE"].value = ast.Constant(text)
    assigns["EXPECTED_SHA"].value = ast.Constant(hashlib.sha256(text.encode()).hexdigest())
    notebook["cells"][1]["source"] = ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# V5 ROI-head-only continuation — controlled detector adaptation\n",
        "Original V5 Mask R-CNN + frozen Crop U-Net; archived weights read-only.\n",
        "Freeze backbone and RPN; update only ROI classification, box and mask heads for four epochs at LR 1e-6.\n",
        "Use fitting images only; select by full-pipeline tuning PQ and read confirmation after selection.\n",
        "Historical validation is not pristine; no test inference or official submission.\n"]
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    meta.update(id=KERNEL, title=KERNEL.split("/")[1])
    (output.parent / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
