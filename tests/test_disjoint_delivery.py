import numpy as np
import pytest
from pycocotools import mask as coco


def crossing():
    a=np.zeros((2048,2048),np.uint8);a[100:110,80:180]=1
    b=np.zeros_like(a);b[60:160,120:130]=1
    return [coco.encode(np.asfortranarray(x)) for x in (a,b)],(a,b)


def test_export_rejects_overlapping_masks_before_any_csv_is_published(tmp_path):
    from solar.v6_submission import write_rle_submission
    masks,_=crossing();target=tmp_path/'submission.csv'
    with pytest.raises(ValueError,match='overlap'):write_rle_submission({'image':masks},['image'],target)
    assert not target.exists()


def test_confidence_priority_retains_two_crossing_instances_without_overlap():
    from solar.disjoint_delivery import make_disjoint
    masks,(a,b)=crossing();rles=make_disjoint(masks)
    assert len(rles)==2
    np.testing.assert_array_equal(coco.decode(rles[0]),a)
    np.testing.assert_array_equal(coco.decode(rles[1]),b & (1-a))
    assert coco.iou(rles,rles,[0,0])[0,1]==0
