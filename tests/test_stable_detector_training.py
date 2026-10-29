import torch
from torch import nn


def test_restores_pretrained_backbone_freezing_and_stops_batch_one_bn_drift():
    from solar.stable_detector_training import restore_training_contract
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=nn.Module();self.backbone.body=nn.ModuleDict({
                'conv1':nn.Conv2d(3,3,1),'bn1':nn.BatchNorm2d(3),
                'layer1':nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3)),
                'layer2':nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3)),
                'layer3':nn.Conv2d(3,3,1),'layer4':nn.Conv2d(3,3,1)})
            self.head=nn.Conv2d(3,1,1)
    model=Model().train();before=model.backbone.body['layer2'][1].running_mean.clone()
    result=restore_training_contract(model)
    assert not any(p.requires_grad for p in model.backbone.body['conv1'].parameters())
    assert not any(p.requires_grad for p in model.backbone.body['layer1'].parameters())
    assert all(p.requires_grad for p in model.backbone.body['layer2'].parameters())
    assert result['batchnorm_modules_frozen']==3
    x=torch.randn(1,3,8,8)+4
    y=model.head(model.backbone.body['layer2'](x));y.mean().backward()
    assert model.head.weight.grad.abs().sum()>0
    assert torch.equal(model.backbone.body['layer2'][1].running_mean,before)
    assert model.backbone.body['layer2'][1].num_batches_tracked.item()==0
