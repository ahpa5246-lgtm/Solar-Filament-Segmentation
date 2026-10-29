import numpy as np
from pycocotools import mask as coco


def test_seed_features_use_box_identity_without_ground_truth_shape():
    from solar.seed_refiner import feature_crop
    image=np.arange(32*32,dtype=np.uint8).reshape(32,32)
    a=np.zeros((32,32),np.uint8);a[5:20,6:23]=1
    b=np.zeros_like(a);b[5,6:23]=1;b[5:20,6]=1
    helpers={'square_bounds':lambda mask:(0,0,32,32)}
    fa,bounds=feature_crop(image,coco.encode(np.asfortranarray(a)),helpers)
    fb,_=feature_crop(image,coco.encode(np.asfortranarray(b)),helpers)
    assert fa.shape==(3,256,256)
    np.testing.assert_array_equal(fa,fb)
    assert set(np.unique(fa[2]))=={0,255}


def test_empty_refinement_rejects_proposal_instead_of_restoring_false_positive():
    from solar.seed_refiner import restore_masks
    probabilities=[np.zeros((256,256),np.float32)]
    assert restore_masks([(3,3,20,20)],probabilities,(32,32),.65,'union')==[]
    probabilities[0][80:160,80:160]=.9
    result=restore_masks([(3,3,20,20)],probabilities,(32,32),.65,'union')
    assert len(result)==1 and coco.area(result[0])>0
