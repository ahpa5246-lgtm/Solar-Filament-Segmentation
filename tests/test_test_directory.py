from pathlib import Path
import pytest


def test_finds_official_sibling_train_test_layout(tmp_path):
    from solar.v6_submission import resolve_test_directory
    annotation=tmp_path/'train'/'annotations.json';annotation.parent.mkdir();annotation.touch()
    target=tmp_path/'test'/'test_images';target.mkdir(parents=True)
    assert resolve_test_directory(annotation)==target


def test_accepts_colocated_test_images_and_rejects_ambiguity(tmp_path):
    from solar.v6_submission import resolve_test_directory
    annotation=tmp_path/'annotations.json';annotation.touch()
    target=tmp_path/'test_images';target.mkdir()
    assert resolve_test_directory(annotation)==target
    (tmp_path/'other'/'test_images').mkdir(parents=True)
    with pytest.raises(ValueError):resolve_test_directory(annotation)
