import numpy as np
import cv2
import pytest
import torch
from solar.data import split_records,polygons_to_instances,derive_targets, feature_channels
from solar.metrics import compare
from solar.model import Net,loss_fn
from solar.pipeline import starts,predict_probabilities, write_submission
from solar.postprocess import Params,to_instances


def test_no_duplicate_or_date_leakage():
    records=[{'stem':'20150101120000Bh','image_id':'a'},
             {'stem':'20150101120000Bh','image_id':'b'},
             {'stem':'20150102120000Bh','image_id':'c'},
             {'stem':'20150103120000Bh','image_id':'d'},
             {'stem':'20150104120000Bh','image_id':'e'}]
    train,val=split_records(records,seed=2)
    assert {r['stem'][:8] for r in train}.isdisjoint({r['stem'][:8] for r in val})


def test_render_and_targets():
    label=polygons_to_instances([{'segmentation':[[10,10,30,10,30,30,10,30]]}],(64,64))
    assert label[20,20] == 1 and label[0,0] == 0
    f,b=derive_targets(label)
    assert f.sum()>100 and b.sum()>0


def test_pq_perfect_split_and_empty():
    gt=np.zeros((32,32),dtype=np.int32);gt[2:14,2:14]=1;gt[18:30,18:30]=2
    assert compare(gt,gt).pq == 1
    assert compare(gt,np.zeros_like(gt)).fn == 2
    assert compare(np.zeros_like(gt),gt).fp == 2
    # Two GT instances merged into one prediction => zero matches if IoU <= .5.
    merged=np.zeros_like(gt);merged[gt>0]=1
    assert compare(gt,merged).pq == 0


def test_instance_extraction_connected_components():
    prob=np.zeros((100,100),np.float32)
    prob[10:30,10:30]=.9
    prob[60:85,60:85]=.9
    inst=to_instances(prob,np.zeros_like(prob),Params())
    assert set(np.unique(inst)) == {0,1,2}
    assert inst[15,15] != inst[70,70]


def test_tile_coverage():
    pos=starts(2048,512,384)
    assert pos[0]==0 and pos[-1]==1536
    coverage=np.zeros(2048,bool)
    for p in pos: coverage[p:p+512] = True
    assert coverage.all()
    with pytest.raises(ValueError): starts(10,8,9)


def test_forward_and_loss():
    net=Net(base=8)
    a=torch.rand((1,2,32,32))
    out=net(a)
    assert out.shape==(1,2,32,32)
    loss=loss_fn(out,torch.zeros_like(out));loss.backward()
    assert np.isfinite(loss.item())


def test_full_tiled_predict_small_synthetic():
    img=np.full((64,64),100,np.uint8)
    net=Net(base=8)
    probs=predict_probabilities(net,img,tile=32,stride=24,batch_size=2)
    assert probs.shape==(2,64,64)
    assert np.isfinite(probs).all()


def test_invalid_submission_rejected(tmp_path):
    with pytest.raises(ValueError):write_submission([],tmp_path/'submission.csv')
    with pytest.raises(ValueError):write_submission([{'filament_id':'x_1','segmentation_rle':'a'},
                                                  {'filament_id':'x_1','segmentation_rle':'b'}],tmp_path/'submission.csv')


def test_features_compatible_with_two_channels():
    arr=feature_channels(np.full((32,32),255,dtype=np.uint8))
    assert arr.shape==(32,32,2) and np.isfinite(arr).all()
