"""CPU contract tests for one-shot independent confirmation and official delivery."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.auto14_plan import UPSTREAM_SHA
from scripts.build_auto14_final_notebook import build, KERNEL_ID
from scripts.run_auto14_final import candidate,final_check,official_history,MIN_CONFIRMATION_PQ


def record(day,score):
    return {"day":day,"status":"COMPLETE","source_sha":UPSTREAM_SHA,
            "local_tuning_pq":score,"checkpoint_sha256":format(day,"064x")}


def test_final_selection_uses_local_only_and_requires_three_successes():
    results={"1":record(1,.34),"2":record(2,.42),"3":record(3,.41)}
    results["1"]["official_public_score"]=1.0
    assert candidate({"results":results})==(2,format(2,"064x"))
    with pytest.raises(RuntimeError,match="three completed"):
        candidate({"results":{"1":record(1,.45),"2":record(2,.46)}})
    results["2"]["source_sha"]="unreviewed"
    # An unverified source is discarded from selection, rather than throwing
    # away other legitimate completed models. Day 3 is the next-best.
    assert candidate({"results":results})==(3,format(3,"064x"))


def test_kaggle_notebook_is_private_and_frozen_before_test(tmp_path):
    sha="1"*64
    meta=build(2,sha,tmp_path/"final.ipynb")
    nb=json.loads((tmp_path/"final.ipynb").read_text())
    assert meta["id"]==KERNEL_ID
    assert meta["is_private"] and meta["enable_gpu"]
    assert meta["competition_sources"]==["filament-segmentation-2026"]
    assert meta["kernel_sources"]==["minahussain111/solar-autoresearch-d02"]
    assert not meta["dataset_sources"]
    code="".join(nb["cells"][1]["source"])
    compile(code,"final Kaggle cell","exec")
    assert "confirmation_stems" not in code or "confirmation" in code
    assert "confirmation_local_pq" in code and "train" in code
    assert "competitions_submit" not in code
    assert "api.competition_submit" not in code
    assert "180" in code and sha in code and UPSTREAM_SHA in code
    assert "0.36" in code


def test_builder_rejects_malformed_selected_checkpoint(tmp_path):
    for day,sha in ((0,"0"*64),(15,"0"*64),(1,"wrong"),(1,"Z"*64)):
        with pytest.raises(ValueError):
            build(day,sha,tmp_path/"bad.ipynb")


def evidence(pq=.42):
    return {"training":False,"official_submission":False,"source_sha":UPSTREAM_SHA,
            "training_checkpoint_sha256":"a"*64,
            "confirmation_local_pq":pq,"status":"CSV_READY",
            "final_test_images":180,"submission_rows":1234,"csv_sha256":"b"*64}


def test_final_confirmation_gate_rejects_below_threshold_and_bad_inventories():
    row=evidence()
    assert final_check(row,"a"*64)=="b"*64
    for changes in (
            {"confirmation_local_pq":MIN_CONFIRMATION_PQ-.0001},
            {"status":"FAILED"},
            {"training":True},
            {"official_submission":True},
            {"final_test_images":20},
            {"submission_rows":0},
            {"source_sha":"unknown"},
            {"csv_sha256":"bad"}
    ):
        invalid={**row,**changes}
        with pytest.raises(RuntimeError):final_check(invalid,"a"*64)


def test_existing_official_submission_is_reused_not_uploaded():
    class API:
        def competition_submissions(self,_,page_number,page_size):
            return [SimpleNamespace(description="Auto14 verified",ref=123)]
    assert official_history(API(),"Auto14 verified")==123


def test_ambiguous_official_history_blocks_second_upload():
    class API:
        def competition_submissions(self,_,page_number,page_size):
            return [SimpleNamespace(description="Auto14 verified",ref=123),
                    SimpleNamespace(description="Auto14 verified",ref=124)]
    with pytest.raises(RuntimeError,match="More than one"):
        official_history(API(),"Auto14 verified")


def test_future_final_notebook_isolates_compiled_numpy_before_confirmation(tmp_path):
    from scripts.kaggle_isolated_runtime import wrap_final_inference, PACKAGES
    meta = build(2, "d" * 64, tmp_path / "final.ipynb")
    assert meta["id"] == KERNEL_ID and meta["is_private"]
    notebook = json.loads((tmp_path / "final.ipynb").read_text())
    source = "".join(notebook["cells"][1]["source"])
    compile(source, "isolated final cell", "exec")
    assert "solar_final_isolated_venv" in source
    assert "venv" in source and "--system-site-packages" in source
    assert "--ignore-installed" in source and "--no-deps" in source
    assert "numpy.rec" in source
    assert "np.__version__" in source
    assert "confirmation_local_pq" in source
    assert "training_checkpoint_sha256" in source
    assert "competition_submit(" not in source
    assert "numpy==1.26.4" in PACKAGES
    with pytest.raises((SyntaxError, ValueError)):
        wrap_final_inference("this is not python ?!")

def test_immediate_day_two_notebook_reuses_private_isolated_final_builder(tmp_path):
    from scripts.build_auto14_final_notebook import IMMEDIATE_KERNEL
    sha = "d213d3d8bb616a6b9d44fef1219f48727eed148fe68a79e7c0800f534af97a3f"
    meta = build(2, sha, tmp_path / "immediate.ipynb", kernel_id=IMMEDIATE_KERNEL)
    nb = json.loads((tmp_path / "immediate.ipynb").read_text())
    code = "".join(nb["cells"][1]["source"])
    compile(code, "immediate Kaggle notebook", "exec")
    assert meta["id"] == IMMEDIATE_KERNEL
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["kernel_sources"] == ["minahussain111/solar-autoresearch-d02"]
    assert "solar_final_isolated_venv" in code
    assert "numpy.rec" in code and "submission.csv" in code
    assert sha in code
    assert "competition_submit(" not in code


def test_immediate_guard_does_not_relaunch_existing_remote_attempt(tmp_path, monkeypatch):
    import scripts.run_auto14_final as runner
    intent = tmp_path / "already-started.json"
    intent.write_text('{"status":"STARTED"}')
    monkeypatch.setattr(runner, "IMMEDIATE_REPORT", intent)
    assert runner.immediate_main() is None
    assert json.loads(intent.read_text()) == {"status":"STARTED"}
