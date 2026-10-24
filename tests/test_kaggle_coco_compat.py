"""Contract tests for the actual Kaggle pycocotools NumPy ABI regression."""
import json
from pathlib import Path

from scripts.build_auto14_notebook import build as daily
from scripts.build_auto14_final_notebook import build as final


def _code(path):
    notebook=json.loads(path.read_text())
    return "".join(notebook["cells"][1]["source"])


def test_daily_and_final_pin_modern_coco_extension_with_real_binary_smoke(tmp_path):
    a=tmp_path/"d01"/"daily.ipynb"
    b=tmp_path/"final"/"final.ipynb"
    daily(2,a)
    final(2,"e"*64,b)
    for path in (a,b):
        s=_code(path)
        compile(s,str(path),"exec")
        assert "pycocotools==2.0.11" in s
        assert "pycocotools==2.0.8" not in s
        assert "coco.encode(np.asfortranarray(a))" in s
        assert "coco.area(rle)" in s
        assert "np.array_equal(coco.decode(rle),a)" in s
        assert "np.array_equal(coco.decode(coco.merge([rle])),a)" in s
        assert "subprocess.run([sys.executable,\"-c\",smoke],check=True)" in s


def test_original_leaderboard_score_does_not_depend_on_dependency_version():
    from solar.auto14_plan import choose_experiment
    # An ABI fix is a prerequisite for actual GPU training, not proof of
    # improved model accuracy or a license to resubmit V5's existing CSV.
    plan=choose_experiment(2)
    assert "official_public_score" not in plan
    assert "submission" not in plan["model"].lower()
