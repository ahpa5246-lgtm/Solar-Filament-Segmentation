"""CPU geometry invariants and both-stage inference-consistency contracts."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from solar.auto14_geometry import link_aligned_fragments
from solar.auto14_upgrade import patch_auto14_postprocess_source
from scripts.build_auto14_notebook import build as build_daily
from scripts.build_auto14_final_notebook import build as build_final


def _broken_filament():
    a=np.zeros((90,120),dtype=np.int32)
    a[30:33,10:47]=1
    a[30:33,54:95]=2
    return a


def test_directional_link_rejoins_aligned_filament_tips_without_painting():
    a=_broken_filament()
    result=link_aligned_fragments(a,max_gap=16,min_area=30)
    assert len(np.unique(result[result>0]))==1
    assert np.array_equal(a>0,result>0)
    assert result.dtype==np.int32
    assert np.array_equal(a,link_aligned_fragments(a,max_gap=0))


def test_parallel_or_orthogonal_close_neighbours_remain_separate():
    parallel=_broken_filament()
    parallel[30:33,54:95]=0
    parallel[45:48,54:95]=2
    assert len(np.unique(link_aligned_fragments(parallel,max_gap=30,min_area=30)[parallel>0]))==2
    perpendicular=np.zeros((90,120),dtype=np.int32)
    perpendicular[30:33,10:47]=1
    perpendicular[35:80,51:54]=2
    assert len(np.unique(link_aligned_fragments(perpendicular,max_gap=30,min_area=30)[perpendicular>0]))==2


def test_round_noise_is_not_falsely_linked():
    a=np.zeros((80,80),dtype=np.int32)
    a[20:32,20:32]=1
    a[20:32,35:47]=2
    assert len(np.unique(link_aligned_fragments(a,max_gap=20)[a>0]))==2


def test_directional_link_is_deterministic_and_handles_three_fragments():
    a=_broken_filament()
    a[30:33,54:95]=0
    a[30:33,53:69]=2
    a[30:33,75:110]=3
    out1=link_aligned_fragments(a,max_gap=16,min_area=30)
    out2=link_aligned_fragments(a,max_gap=16,min_area=30)
    assert np.array_equal(out1,out2)
    assert np.array_equal(out1>0,a>0)
    assert len(np.unique(out1[a>0]))==1


def test_incorrect_instance_maps_and_geometry_inputs_fail_closed():
    with pytest.raises(TypeError):
        link_aligned_fragments(np.zeros((2,2),dtype=np.float32),max_gap=10)
    with pytest.raises(ValueError):
        link_aligned_fragments(-np.ones((2,2),dtype=np.int32),max_gap=10)
    with pytest.raises(ValueError):
        link_aligned_fragments(np.zeros((2,2),dtype=np.int32),max_gap=81)
    with pytest.raises(ValueError):
        link_aligned_fragments(np.zeros((2,2),dtype=np.int32),max_gap=2,min_elongation=0.1)


def test_postprocess_patching_defaults_to_upstream_identity():
    fragment='''import numpy as np
from dataclasses import dataclass
@dataclass
class PostConfig:
    merge_dist: int = 0
    min_area: int = 20
    max_instances: int = 60

def instances_from_prob(raw, cfg):
    return filter_components(raw, cfg.min_area, cfg.max_instances)
'''
    patched=patch_auto14_postprocess_source(fragment)
    ast.parse(patched)
    assert 'link_gap: int = 0' in patched
    assert 'if cfg.link_gap:' in patched
    assert 'link_aligned_fragments' in patched
    with pytest.raises(ValueError,match="source has drifted"):
        patch_auto14_postprocess_source(fragment.replace("merge_dist: int = 0","merge_dist: int = 3"))


def test_daily_and_final_notebooks_embed_identical_geometry(tmp_path):
    from solar.auto14_geometry import __file__ as geometry_path
    code=Path(geometry_path).read_text()
    daily=tmp_path/"daily"/"auto14.ipynb"
    final=tmp_path/"final"/"final.ipynb"
    build_daily(7,daily)
    build_final(7,"c"*64,final)
    for path in (daily,final):
        generated="".join(json.loads(path.read_text())["cells"][1]["source"])
        compile(generated,str(path),"exec")
        assert "link_aligned_fragments" in generated
        assert "from .auto14_geometry import link_aligned_fragments" in generated
        assert "do_not_use_test_labels" not in generated or "test" in generated
        assert "competitions_submit" not in generated
    assert "def link_aligned_fragments" in code
    daily_source="".join(json.loads(daily.read_text())["cells"][1]["source"])
    assert "tune_geometry.py" in daily_source
    assert "source_patches_sha256" in daily_source


def test_geometry_tuner_is_valid_python_and_enforces_heldout_partition():
    src=(Path(__file__).parents[1]/"scripts/auto14_geometry_tune.py").read_text()
    ast.parse(src)
    assert '("train","val","confirmation")' in src
    assert "Month leakage" in src
    assert "selected_local_pq" in src
    assert "min_alignment" not in src or "gap" in src
