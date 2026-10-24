"""Deterministic early Kaggle training: zero duplicate uploads, known time slot."""
import datetime as dt
from pathlib import Path
import pytest

from scripts import early_kaggle_challenger_once as r


def test_early_authorization_is_a_finite_baghdad_time_window():
    utc=dt.timezone.utc
    assert not r.date_allowed(dt.datetime(2026,10,24,20,59,tzinfo=utc))
    assert r.date_allowed(dt.datetime(2026,10,24,21,0,tzinfo=utc))
    assert r.date_allowed(dt.datetime(2026,10,24,23,59,tzinfo=utc))
    assert not r.date_allowed(dt.datetime(2026,10,25,0,0,tzinfo=utc))


def test_early_challenger_requires_a_real_previous_failed_kernel(tmp_path,monkeypatch):
    (tmp_path/"V5_FINAL_CACHE.json").write_text("{}")
    monkeypatch.setattr(r,"STATE",tmp_path/"AUTO14_STATE.json")
    state={"results":{"1":{"status":"FAILED","execution_status":"ERROR"}}}
    plan=r.validated_day_two(state)
    assert plan["day"]==2 and plan["scale"]==0.5
    state["results"]["2"]={"status":"STARTED"}
    with pytest.raises(RuntimeError,match="already has"):
        r.validated_day_two(state)
    state["results"]["2"]={}
    with pytest.raises(RuntimeError,match="already has"):
        r.validated_day_two(state)
    del state["results"]["2"]
    state["results"]["1"]["execution_status"]="UNKNOWN"
    with pytest.raises(RuntimeError,match="confirmed"):
        r.validated_day_two(state)


def test_early_script_exits_without_gpu_outside_time_window(monkeypatch):
    monkeypatch.setattr(r,"date_allowed",lambda *args:False)
    def fail(*a,**k):raise AssertionError("Should not contact Kaggle")
    monkeypatch.setattr(r,"load_state",fail)
    r.main()
