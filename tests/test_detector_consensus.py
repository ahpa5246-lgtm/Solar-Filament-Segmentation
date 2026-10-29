import numpy as np
from pycocotools import mask as coco


def encode(mask):return coco.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_inverse_flips_restore_rectangular_coordinates():
    from solar.detector_consensus import inverse_flip
    mask=np.zeros((11,17),np.uint8);mask[1:4,3:8]=1
    for flip in ((),(0,),(1,),(0,1)):
        np.testing.assert_array_equal(inverse_flip(np.flip(mask,axis=flip) if flip else mask,flip),mask)


def test_consensus_counts_distinct_views_and_rejects_single_view_extra():
    from solar.detector_consensus import fuse_views
    a=np.zeros((32,32),np.uint8);a[2:8,2:12]=1
    b=np.zeros_like(a);b[20:26,18:29]=1
    c=np.zeros_like(a);c[2:6,23:27]=1
    views=[([encode(a)],[.9]),([encode(a),encode(b),encode(c)],[.8,.9,.99]),([encode(b)],[.85]),([],[])]
    rles,_=fuse_views(views,keep_base=True)
    assert len(rles)==2
    assert any(np.array_equal(coco.decode(r),a) for r in rles)
    assert any(np.array_equal(coco.decode(r),b) for r in rles)
    duplicates=[([],[]),([encode(c),encode(c)],[.9,.8]),([],[]),([],[])]
    assert fuse_views(duplicates,keep_base=True)[0]==[]


def test_consensus_preserves_crossings_and_identity_replay():
    from solar.detector_consensus import fuse_views
    a=np.zeros((48,48),np.uint8);a[20:24,3:44]=1
    b=np.zeros_like(a);b[3:44,20:24]=1
    rles=[encode(a),encode(b)]
    result,_=fuse_views([(rles,[.9,.8])]*4,keep_base=False)
    assert len(result)==2
    for actual,expected in zip(result,(a,b)):np.testing.assert_array_equal(coco.decode(actual),expected)
