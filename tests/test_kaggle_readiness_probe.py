"""CPU unit tests for Kaggle submission-readiness inspection."""
import csv
from types import SimpleNamespace
import numpy as np
import pytest
from pycocotools import mask as coco

from scripts.kaggle_readiness_probe import audit,csv_integrity

def make_file(path,entries):
    with path.open("w",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=["filament_id","segmentation_rle"])
        w.writeheader()
        for label,image in entries:
            compressed=coco.encode(np.asfortranarray(image))
            w.writerow({"filament_id":label,"segmentation_rle":compressed["counts"].decode("ascii")})

def test_every_submitted_mask_is_checked_as_real_rle(tmp_path):
    mask=np.zeros((2048,2048),dtype=np.uint8)
    mask[100:120,150:190]=1
    file=tmp_path/"submission.csv"
    make_file(file,[("20150125172714Mh_1",mask)])
    result=csv_integrity(file)
    assert result["rows"]==1 and result["unique_stems"]==1
    assert result["min_area"]==800 and result["max_area"]==800
    assert len(result["sha256"])==64
    assert result["rle_all_rows_area_checked"] is True
    assert result["input_inventory_matched"] is False

def test_duplicate_or_invalid_id_and_empty_mask_are_rejected(tmp_path):
    mask=np.zeros((2048,2048),dtype=np.uint8);mask[:4,:4]=1
    file=tmp_path/"dup.csv"
    make_file(file,[("20150125172714Mh_1",mask),("20150125172714Mh_1",mask)])
    with pytest.raises(ValueError,match="repeated"):csv_integrity(file)
    make_file(file,[("unknown_1",mask)])
    with pytest.raises(ValueError,match="Invalid"):csv_integrity(file)
    make_file(file,[("20150125172714Mh_1",np.zeros_like(mask))])
    with pytest.raises(ValueError,match="Empty"):csv_integrity(file)

def test_inventory_never_submits_and_failure_is_explicit():
    class Fake:
        def kernels_status(self,ref):return SimpleNamespace(status="COMPLETE")
        def competition_submissions(self,*args,**kw):
            return [SimpleNamespace(ref=1234,score=.32,description="existing V5",status="complete")]
        def kernels_list(self,*args,**kwargs):return []
        def kernels_output(self,*args,**kwargs):pass
        def competition_submit(self,*args,**kwargs):
            raise AssertionError("Read-only inventory must never submit")
    a=audit(Fake())
    assert a["submissions"]["items"][0]["public_score"]==pytest.approx(.32)
    assert a["assessment"]=="NO_NEW_SCORED_CHALLENGER"
    assert a["v5_csv"]["status"]=="MISSING_OR_NONUNIQUE"
    assert a["submission_attempted"] is False
