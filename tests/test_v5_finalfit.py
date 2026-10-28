import ast,hashlib,json
from pathlib import Path
import pytest


def test_final_assessment_freezes_version_hash_and_test_inventory(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from build_v5_finalfit import build
    with pytest.raises(ValueError):build(tmp_path/'bad.ipynb',1,'a'*64)
    out=tmp_path/'final.ipynb';meta=build(out,2,'a'*64)
    assert meta['is_private'] and meta['kernel_sources']==['minahussain111/solar-v5-finetune-pq50-20261028']
    tree=ast.parse(''.join(json.loads(out.read_text())['cells'][1]['source']))
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'SOURCE','EXPECTED_SHA'}}
    assert hashlib.sha256(values['SOURCE'].encode()).hexdigest()==values['EXPECTED_SHA']
    assert "expected='" not in values['SOURCE'] or 'a'*64 in values['SOURCE']
    assert 'solar/v5_finalfit.py' in values['SOURCE']
    assert 'selected_training_version=2' in values['SOURCE']
    assert 'len(files)!=180' in values['SOURCE']
    assert 'competition_submit' not in values['SOURCE']


def test_incomplete_or_unimproved_training_cannot_launch_final_gpu(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_v5_final_assessment import require_candidate
    for state in ({}, {'status':'ACCEPTED'}, {'status':'COMPLETE','result':{'status':'FAILED'}},
                  {'status':'COMPLETE','result':{'status':'COMPLETE','best_epoch':0,'best_tuning_pq':.5,'baseline_tuning_pq':.4}},
                  {'status':'COMPLETE','result':{'status':'COMPLETE','best_epoch':2,'best_tuning_pq':.4,'baseline_tuning_pq':.4}}):
        with pytest.raises(ValueError):require_candidate(state)
    result={'status':'COMPLETE','best_epoch':2,'best_tuning_pq':.42,'baseline_tuning_pq':.4}
    assert require_candidate({'status':'COMPLETE','result':result})==result


def test_tuning_gain_cannot_promote_overlap_policy_with_confirmation_regression():
    from solar.v5_finalfit import overlap_promotion_supported
    assert not overlap_promotion_supported({'union':.405,'preserve':.413},.407,.399,{'low':-.022})
    assert not overlap_promotion_supported({'union':.405,'preserve':.413},.407,.411,{'low':-.01})
    assert overlap_promotion_supported({'union':.405,'preserve':.43},.407,.44,{'low':.01})
