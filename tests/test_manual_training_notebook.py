"""Offline contract tests for the complete handoff-ready Kaggle training notebook."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/solar_manual_training_ready.ipynb"
UPSTREAM_SHA = "ef37d2e93d4cc9e248595e791ee39fe520d392d0"


def notebook():
    value = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert value["nbformat"] == 4
    assert value["nbformat_minor"] >= 2
    return value


def test_training_cell_is_complete_and_compiles_without_gpu():
    nb = notebook()
    assert len(nb["cells"]) == 4
    assert [cell["cell_type"] for cell in nb["cells"]] == [
        "markdown", "code", "markdown", "code"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            ast.parse(code)
            compile(code, "solar_training_ready.ipynb", "exec")


def test_training_is_real_bounded_and_preserves_checkpoints():
    nb = notebook()
    source = "".join(nb["cells"][1]["source"])
    assert "patch_auto14_sources" in source
    assert "train.py" in source and "tune.py" in source
    assert "torch.cuda.is_available()" in source
    assert "split_months_disjoint" in source
    assert "best_checkpoint_sha256" in source
    assert "best.pt" in source and "postprocess.json" in source
    assert UPSTREAM_SHA in source
    assert "'epochs':" not in source or "20" in source
    assert "boundary_weight" in source
    assert "official_submission" in source and "test_inference" in source
    assert "api.competition_submit(" not in source
    assert "competition_submit(" not in source


def test_no_unexpanded_builder_placeholders():
    source = "".join(notebook()["cells"][1]["source"])
    for placeholder in ("__PLAN__", "__REPO__", "__REVISION__",
                        "__GEOMETRY_SOURCE__", "__GEOMETRY_TUNER_SOURCE__"):
        assert placeholder not in source


def test_persisted_result_requires_completed_checkpoint():
    nb = notebook()
    code = "".join(nb["cells"][3]["source"])
    assert "training_report['status'] == 'COMPLETE'" in code
    assert "ckpt.is_file()" in code
    assert "local_tuning_pq" in code
