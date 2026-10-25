"""Sealed immediate Kaggle validation/official upload has hard provenance gates."""
import datetime as dt
import json

import pytest

from scripts import run_auto14_early_final as immediate
from scripts.build_auto14_final_notebook import build, KERNEL_ID as NOV7_KERNEL
from scripts.run_auto14_final import candidate
from solar.auto14_plan import UPSTREAM_SHA


def authentic_completed_record():
    return {
        "status":"COMPLETE","execution_status":"COMPLETE","day":2,
        "source_sha":UPSTREAM_SHA,"local_tuning_pq":.40438,
        "checkpoint_sha256":"d213d3d8bb616a6b9d44fef1219f48727eed148fe68a79e7c0800f534af97a3f",
        "confirmation_evaluated":False,"official_submission":False,
        "split_counts":{"train":489,"val":111,"confirmation":107},
    }


def test_only_oct25_baghdad_is_a_valid_single_shot():
    t=dt.timezone.utc
    assert immediate.date_allowed(dt.datetime(2026,10,25,9,tzinfo=t))
    assert not immediate.date_allowed(dt.datetime(2026,10,24,20,59,tzinfo=t))
    assert not immediate.date_allowed(dt.datetime(2026,10,25,21,0,tzinfo=t))


def test_require_authentic_trained_checkpoint_and_unseen_confirmation():
    state={"results":{"2":authentic_completed_record()}}
    assert immediate.preflight(state)["split_counts"]["confirmation"]==107
    assert candidate(state,minimum_completed=1,required_day=2)[0]==2
    with pytest.raises(RuntimeError,match="three|Fewer"):
        candidate(state)
    for changed in (
        {"status":"FAILED"},
        {"execution_status":"ERROR"},
        {"confirmation_evaluated":True},
        {"official_submission":True},
        {"checkpoint_sha256":"wrong"},
        {"local_tuning_pq":None},
        {"split_counts":{"confirmation":2}},
    ):
        bad={"results":{"2":{**authentic_completed_record(),**changed}}}
        with pytest.raises(RuntimeError):
            immediate.preflight(bad)


def test_immediate_and_original_final_use_distinct_kernel_ids(tmp_path):
    sha=authentic_completed_record()["checkpoint_sha256"]
    for slug in (immediate.KERNEL_ID,NOV7_KERNEL):
        outfile=tmp_path/slug.rsplit("/",1)[1]/"final.ipynb"
        meta=build(2,sha,outfile,kernel_id=slug)
        assert meta["id"]==slug and meta["is_private"]
        assert meta["kernel_sources"]==["minahussain111/solar-autoresearch-d02"]
        code="".join(json.loads(outfile.read_text())["cells"][1]["source"])
        compile(code,"isolated final Kaggle notebook","exec")
        assert code.index('from pycocotools import mask as coco') > code.index('pycocotools==2.0.11')
        assert sha in code and "confirmation_local_pq" in code
    with pytest.raises(ValueError):
        build(2,sha,tmp_path/"wrong.ipynb",kernel_id="outsider/unsafe")


def test_early_runner_never_contacts_kaggle_outside_slot(monkeypatch):
    monkeypatch.setattr(immediate,"date_allowed",lambda:False)
    monkeypatch.setattr(immediate,"load_state",lambda _:pytest.fail("No Kaggle outside slot"))
    assert immediate.main() is None
