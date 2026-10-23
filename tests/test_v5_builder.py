"""Pure-CPU tests for the reproducible Apache-licensed V5 notebook builder."""
import ast
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "build_v5_from_apache_reference.py"
spec = importlib.util.spec_from_file_location("v5_builder", SOURCE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fake_source():
    cells = [{"cell_type": "code", "source": ["pass\n"]} for _ in range(34)]
    data = {
        20: "from pycocotools.coco import COCO\n",
        21: "class FilamentDataset:\n    pass\n",
        26: "class CropUNet:\n    pass\n",
        27: """from collections import defaultdict
refined_unmerged = defaultdict(list)
refiner_threshold = 0.85
with torch.inference_mode():
    for start in []:
        batch_tasks = []
        images = None
        probabilities = refiner(images).sigmoid().cpu().numpy()[:, 0]

        for task, probability in zip(batch_tasks, probabilities):
            pass

refined_predictions_2048 = {observation_id: merge_overlapping_rles(refined_unmerged.get(observation_id, [])) for observation_id in best_detector_predictions}

comparison = []
for name, predictions in [("Detector", baseline_predictions_2048), ("Detector + refiner", refined_predictions_2048)]:
    pass
""",
        30: """use_refiner = bool(comparison.loc["Detector + refiner", "all_dice"] >= comparison.loc["Detector", "all_dice"])
final_configuration = {"use_refiner": use_refiner, "refiner_epochs": 8}
""",
        32: """final_detector = detector.build_model(pretrained=False)
probabilities = final_refiner(crop_tensor).sigmoid().cpu().numpy()[:, 0]
""",
        33: "submission_rows = []\n",
    }
    for index, content in data.items():
        cells[index]["source"] = content.splitlines(keepends=True)
    return {"cells": cells}


def test_reject_missing_upstream_cells():
    with pytest.raises(ValueError, match="too short"):
        builder.build_notebook({"cells": []})


def test_reject_upstream_signature_drift():
    source = fake_source()
    source["cells"][27]["source"] = ["changed\n"]
    with pytest.raises(ValueError, match="Upstream changed"):
        builder.build_notebook(source)


def test_generated_notebook_keeps_credit_and_valid_python():
    result = builder.build_notebook(fake_source())
    assert result["nbformat"] == 4
    assert len(result["cells"]) == 18
    combined = "\n".join(builder.src(c) for c in result["cells"])
    assert "Nomannic" in combined and "Apache" in combined
    assert "Detector + refiner + flip4" in combined
    assert "use_tta=final_configuration.get('refiner_flip4',False)" in combined
    assert "len(test_files)==180" in combined
    for cell in result["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            code = builder.src(cell)
            if code.startswith(("%%writefile", "!torchrun")):
                continue
            ast.parse(code)


def test_refiner_tta_inverts_flips_exactly():
    import torch
    namespace = {"torch": torch}
    exec(builder.TTA_HELPER, namespace)
    class Equivariant(torch.nn.Module):
        def forward(self, image):
            return image[:, :1]
    images = torch.rand(2, 3, 16, 16)
    plain = namespace["predict_refiner_probabilities"](Equivariant(), images, False)
    flip4 = namespace["predict_refiner_probabilities"](Equivariant(), images, True)
    assert torch.allclose(plain, flip4, atol=1e-6)
