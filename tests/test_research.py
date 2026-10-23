"""CPU-only regression tests for the 2026 research upgrades.

No Kaggle competition data, GPU, or private test labels required.
"""
import cv2
import numpy as np
import pytest
import torch

from solar.data import TileDataset
from solar.model import Net, loss_fn, soft_cldice_loss
from solar.metrics import compare
from solar.pipeline import load_model, predict_probabilities
from solar.postprocess import Params
from solar.research_metrics import diagnose_instances
from solar.tune import parameter_grid


def test_validation_tiles_repeat_exactly(tmp_path):
    path = tmp_path / '20150101120000Bh.jpeg'
    img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    assert cv2.imwrite(str(path), img)
    rec = {'stem':path.stem, 'path':str(path), 'image_id':'one',
           'ann':[{'segmentation':[[10, 10, 40, 10, 40, 40, 10, 40]]}]}
    dataset = TileDataset([rec], tile=32, samples_per_epoch=4,
                          seed=42, training=False)
    first = dataset[2]
    dataset.set_epoch(200)
    second = dataset[2]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_edge_guided_model_forward_backward():
    torch.manual_seed(123)
    model = Net(base=8, architecture='edge_attention')
    batch = torch.rand(1, 2, 32, 32)
    labels = torch.zeros(1, 2, 32, 32)
    labels[:, 0, 4:24, 8:12] = 1
    output = model(batch)
    assert output.shape == labels.shape
    loss = loss_fn(output, labels, cldice_weight=.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.edge_attention.edge[0].weight.grad is not None
    assert torch.isfinite(model.edge_attention.edge[0].weight.grad).all()


def test_soft_cldice_is_finite_on_empty_and_thin_masks():
    for target in (torch.zeros(1,1,24,24), torch.ones(1,1,24,24)):
        pred = torch.full_like(target, .5, requires_grad=True)
        loss = soft_cldice_loss(pred, target, iterations=3)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(pred.grad).all()


def test_checkpoint_backward_compatibility(tmp_path):
    model = Net(base=8)
    checkpoint = tmp_path / 'best.pt'
    torch.save({'state_dict':model.state_dict(), 'base':8}, checkpoint)
    restored, info = load_model(checkpoint)
    assert restored.architecture == 'unet'
    assert info['base'] == 8


class EquivariantModel(torch.nn.Module):
    def forward(self, x):
        return x


def test_four_view_tta_inverse_transforms():
    im = np.arange(64*64, dtype=np.uint16).reshape(64,64).astype(np.uint8)
    model = EquivariantModel()
    original = predict_probabilities(model, im, tile=32, stride=24, batch_size=2)
    augmented = predict_probabilities(model, im, tile=32, stride=24, batch_size=2, tta='d4')
    np.testing.assert_allclose(original, augmented, rtol=1e-5, atol=1e-6)


def test_split_and_merge_diagnostics():
    gt = np.zeros((48,48), np.int32)
    gt[2:20,2:20] = 1
    gt[25:43,25:43] = 2
    perfect = diagnose_instances(gt, gt)
    assert perfect.matched_dice_proxy == pytest.approx(1.)
    assert perfect.likely_merges == perfect.likely_splits == 0

    split = gt.copy()
    split[2:20,11:20] = 3
    assert diagnose_instances(gt, split).likely_splits == 1

    merged = gt.copy()
    merged[merged == 2] = 1
    assert diagnose_instances(gt, merged).likely_merges == 1
    assert diagnose_instances(gt, merged).matched_dice_proxy < 1.
    assert compare(gt, merged).pq < 1.


def test_empty_diagnostics_and_shape_errors():
    empty = np.zeros((8,8), np.int32)
    assert diagnose_instances(empty, empty).matched_dice_proxy == 1.
    one = empty.copy()
    one[1:4,1:4] = 1
    assert diagnose_instances(empty, one).matched_dice_proxy == 0.
    with pytest.raises(ValueError):
        diagnose_instances(one, one[:, :4])


def test_precision_search_explores_high_thresholds():
    baseline = Params(threshold=.5, boundary=.5, min_area=100, closing=3, min_seed=12)
    assert baseline in parameter_grid('baseline')
    grid = parameter_grid('precision')
    assert len(grid) == 24
    assert any(p.threshold >= .7 and p.min_area >= 400 for p in grid)
    with pytest.raises(ValueError):
        parameter_grid('not-a-profile')


def test_architecture_rejects_invalid_name():
    with pytest.raises(ValueError):
        Net(base=8, architecture='magic_model')
