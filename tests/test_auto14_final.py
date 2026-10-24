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
    with pytest.raises(RuntimeError,match="source"):
        candidate({"results":results})


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
