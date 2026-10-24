"""CPU-only contract tests; genuine Kaggle GPU training requires scheduled Actions."""
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.auto14_plan import (
    DAYS, START, UPSTREAM_SHA, best_result, choose_experiment, day_index, notebook_slug
)
from scripts.build_auto14_notebook import build, RUN_CODE
from scripts import run_auto14_daily as runner


def test_exactly_fourteen_utc_day_slots():
    assert day_index(START-dt.timedelta(days=1)) is None
    for d in range(14):
        assert day_index(START+dt.timedelta(days=d))==d+1
    assert day_index(START+dt.timedelta(days=14)) is None
    assert DAYS==14


def test_plans_and_private_notebooks_have_bounded_distinct_identity(tmp_path):
    for day in range(1,15):
        plan=choose_experiment(day)
        assert plan["day"]==day and 0.5 <= plan["scale"] <= 0.75
        assert plan["epochs"] <= 36 and plan["steps"] <= 185
        assert plan["batch"] <= 8
        assert "score" not in plan or plan["comparison"].startswith("tuning")
        out=tmp_path/str(day)/"auto14.ipynb"
        meta=build(day,out)
        nb=json.loads(out.read_text())
        assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
        assert meta["competition_sources"]==["filament-segmentation-2026"]
        assert not meta["dataset_sources"] and not meta["kernel_sources"]
        assert meta["id"]==notebook_slug(day)
        assert nb["cells"][1]["cell_type"]=="code"
        code="".join(nb["cells"][1]["source"])
        compile(code,"Kaggle auto14","exec")
        assert UPSTREAM_SHA in code
        assert "confirmation_stems" in code and '2026' in code
        assert '"--max-images","24"' in code and '"--val-every","5"' in code
        assert "competitions_submit" not in code and "competition_submit" not in code
        assert "__PLAN__" not in code


def test_adaptation_uses_only_successful_prior_tuning_not_leaderboard():
    history={
        "1":{"day":1,"status":"COMPLETE","source_sha":UPSTREAM_SHA,
             "local_tuning_pq":0.4,"config":choose_experiment(1),"official_public_score":0.99},
        "2":{"day":2,"status":"COMPLETE","source_sha":UPSTREAM_SHA,
             "local_tuning_pq":0.45,"config":choose_experiment(2)},
        "3":{"day":3,"status":"FAILED","source_sha":UPSTREAM_SHA,
             "local_tuning_pq":0.9,"config":choose_experiment(3)}
    }
    assert best_result(history)=="2"
    proposal=choose_experiment(10,history)
    assert proposal["base"]==history["2"]["config"]["base"]
    assert proposal["pos_weight"]==history["2"]["config"]["pos_weight"]
    assert proposal["seed"] != history["2"]["config"]["seed"]
    assert proposal["epochs"]>history["2"]["config"]["epochs"]
    for d in (0,15,"7"):
        with pytest.raises(ValueError):choose_experiment(d)


def test_skip_duplicates_and_reject_missing_credentials(monkeypatch,tmp_path):
    path=tmp_path/"state.json"
    state=runner.load_state(path)
    assert state["results"]=={}
    state["results"]["1"]={"status":"COMPLETE"}
    runner.save_state(state,path)
    monkeypatch.setattr(runner,"day_index",lambda _:1)
    assert runner.main(["--day","1","--state",str(path)]) is None
    state["results"]={}
    runner.save_state(state,path)
    monkeypatch.delenv("KAGGLE_API_TOKEN",raising=False)
    with pytest.raises(SystemExit,match="secret missing"):
        runner.main(["--day","1","--state",str(path)])


def test_fake_kaggle_recovers_completed_notebook_without_launch(monkeypatch):
    config=choose_experiment(2)
    class Api:
        def quota_view(self):
            return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=dt.timedelta(hours=30),
                                               time_used=dt.timedelta(hours=2)))
        def kernels_list(self,**kwargs):
            return [SimpleNamespace(ref=notebook_slug(2))]
        def kernels_status(self,slug):
            return SimpleNamespace(status="COMPLETE")
        def kernels_output(self,slug,path,file_pattern=None):
            (Path(path)/"auto14_result.json").write_text(json.dumps({
                "day":2,"status":"COMPLETE","split_months_disjoint":True,
                "official_submission":False,"test_inference":False,
                "config":config,"source_sha":UPSTREAM_SHA,
                "local_tuning_pq":0.419,"best_checkpoint_sha256":"a"*64,
                "split_counts":{"train":500,"val":120,"confirmation":87},
                "tuning_pq_status":"exploratory","elapsed_seconds":1234
            }))
        def kernels_push(self,*args,**kwargs):
            raise AssertionError("Existing notebook must never be relaunched")
    report={}
    runner.execute(Api(),2,{},report)
    assert report["launch"]=="resume_existing"
    assert report["status"]=="COMPLETE"
    assert report["local_tuning_pq"]==pytest.approx(0.419)
    assert report["official_submission"] if "official_submission" in report else True
    assert report["confirmation_evaluated"] is False


def test_low_quota_fails_closed_without_launch():
    class Api:
        def quota_view(self):
            return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=dt.timedelta(hours=30),
                                            time_used=dt.timedelta(hours=29)))
        def kernels_status(self,*args,**kwargs):
            raise AssertionError("Quota check must precede any access")
    report={}
    runner.execute(Api(),1,{},report)
    assert report["status"]=="SKIPPED_QUOTA"


def test_non_404_kernel_status_errors_are_not_silent():
    class E(Exception):
        response=SimpleNamespace(status_code=403)
    class Api:
        def kernels_list(self,**kwargs):
            return [SimpleNamespace(ref="owner/slug")]
        def kernels_status(self,slug):
            raise E("private access denied")
    with pytest.raises(E):runner.recover_existing(Api(),"owner/slug")


def test_missing_owned_notebook_must_not_call_status_or_launch():
    class Api:
        def kernels_list(self,**kwargs):
            return [SimpleNamespace(ref="owner/unrelated")]
        def kernels_status(self,*args):
            raise AssertionError("A missing notebook cannot be status-polled")
    assert runner.recover_existing(Api(),"owner/new-notebook") is None


def test_absent_owned_inventory_is_an_error_not_permission_to_reupload():
    class Api:
        def kernels_list(self,**kwargs):
            return None
    with pytest.raises(RuntimeError,match="inventory unavailable"):
        runner.recover_existing(Api(),"owner/new-notebook")


def test_version_status_bypasses_list_search_and_reports_completion():
    """An immutable Kaggle v1 reference is NOT a separate kernel-list slug."""
    class Api:
        def kernels_list(self, **kwargs):
            raise AssertionError("Version polling must never list a versioned slug")
        def kernels_status(self, slug):
            assert slug == "minahussain111/solar-autoresearch-d01/v1"
            return SimpleNamespace(status="COMPLETE")
    assert runner.recover_existing(Api(),"minahussain111/solar-autoresearch-d01/v1")=="COMPLETE"


def test_unversioned_absence_still_requires_authenticated_inventory():
    class Api:
        def kernels_list(self, **kwargs):
            return []
        def kernels_status(self,slug):
            raise AssertionError("Must not status-check unknown private kernels")
    assert runner.recover_existing(Api(),"minahussain111/solar-autoresearch-d05") is None


def test_invalid_version_reference_is_rejected_before_network():
    class Api:
        def kernels_list(self, **kwargs):
            raise AssertionError("Unexpected listing call")
    with pytest.raises(ValueError):
        runner.recover_existing(Api(),"minahussain111/solar-autoresearch-d01/invalid")


def test_known_immutable_version_reuses_same_run_and_never_pushes():
    """Old runner timeout cannot cause a duplicate Kaggle GPU training upload."""
    config=choose_experiment(1)
    class Api:
        def quota_view(self):
            return SimpleNamespace(gpu_quota=SimpleNamespace(
                total_time_allowed=dt.timedelta(hours=29),
                time_used=dt.timedelta(hours=6)))
        def kernels_list(self,**kwargs):
            raise AssertionError("Known version should never list or re-create kernel")
        def kernels_status(self,slug):
            assert slug=="minahussain111/solar-autoresearch-d01/v1"
            return SimpleNamespace(status="COMPLETE")
        def kernels_push(self,*args,**kwargs):
            raise AssertionError("Do not reupload a known immutable version")
        def kernels_output(self,slug,path,file_pattern=None):
            assert slug.endswith("/v1")
            (Path(path)/"auto14_result.json").write_text(json.dumps({
                "day":1,"status":"COMPLETE","split_months_disjoint":True,
                "official_submission":False,"test_inference":False,
                "config":config,"source_sha":UPSTREAM_SHA,
                "local_tuning_pq":0.35,"best_checkpoint_sha256":"c"*64,
                "split_counts":{"train":500,"val":100,"confirmation":90},
                "tuning_pq_status":"exploratory","elapsed_seconds":4300
            }))
    record={"known_version":1}
    runner.execute(Api(),1,{},record)
    assert record["launch"]=="resume_existing" and record["version"]==1
    assert record["status"]=="COMPLETE" and record["checkpoint_sha256"]=="c"*64
