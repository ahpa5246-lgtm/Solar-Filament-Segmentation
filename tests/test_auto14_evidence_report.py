"""Evidence report must never invent a KPI or silently relabel Dice as PQ."""
import pytest
from scripts.auto14_evidence_report import report


def test_no_run_is_not_an_official_score():
    rendered=report({}, {})
    assert "No new official Kaggle score verified" in rendered
    assert "0.70 score is an aspiration" in rendered
    assert "not interchangeable with binary Dice" in rendered
    assert "No recorded trials" in rendered
    assert "Farnoosh Memari" in rendered


def test_failure_records_only_exploratory_not_official():
    state={"results":{
        "1":{"day":1,"status":"FAILED"},
        "2":{"day":2,"status":"COMPLETE","local_tuning_pq":.4171,
             "checkpoint_sha256":"a"*64}}}
    output=report(state,{"status":"REJECTED","confirmation_local_pq":.35})
    assert "0.41710" in output and "0.35000" in output
    assert "No new official Kaggle score verified" in output
    assert "REJECTED" in output and "0.32" in output


def test_only_explicitly_scored_official_submission_reports_new_score():
    state={"results":{"1":{"day":1,"status":"COMPLETE","local_tuning_pq":.45}}}
    unscored=report(state,{"status":"SUBMITTED","official_submission":True,"official_public_score":.99})
    assert "0.99000" not in unscored
    scored=report(state,{"status":"SCORED","official_submission":True,
                         "official_public_score":.3789,"selected_day":1})
    assert "0.37890" in scored
    assert "Chosen candidate (from LOCAL tuning only): **1**" in scored


@pytest.mark.parametrize("score",[float("nan"),float("inf"),-1,1.2,"0.7",None])
def test_invalid_official_score_rejected(score):
    with pytest.raises(ValueError):
        report({},{"status":"SCORED","official_submission":True,"official_public_score":score})


def test_paused_challenger_does_not_claim_scheduled_gpu_or_submission():
    state={"status":"paused_v5_recovery","results":{
        "2":{"status":"COMPLETE","execution_status":"COMPLETE",
             "training":True,"local_tuning_pq":0.40438,
             "checkpoint_sha256":"a"*64},
        "3":{"status":"COMPLETE","execution_status":"ERROR",
             "training":True,"local_tuning_pq":0.9,
             "checkpoint_sha256":"b"*64}}}
    output=report(state,{})
    assert "Successful training notebooks with verified COMPLETE evidence: **1**" in output
    assert "Independent-challenger training is paused" in output
    assert "blocked from candidate selection and submission" in output
    assert "No new official Kaggle score verified" in output
    assert "0.40438" in output
    assert "0.90000" not in output  # Incomplete execution cannot publish a local score.\n    assert "| 3 | UNVERIFIED_COMPLETE | n/a | n/a |" in output


def test_complete_without_checkpoint_is_not_successful_training():
    state={"results":{"1":{"status":"COMPLETE","execution_status":"COMPLETE",
                              "training":True,"local_tuning_pq":0.5}}}
    output=report(state,{})
    assert "Successful training notebooks with verified COMPLETE evidence: **0**" in output
