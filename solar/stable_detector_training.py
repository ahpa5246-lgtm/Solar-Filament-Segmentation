"""Restore pretrained ResNet-FPN trainability after loading a state dictionary.

TorchVision defaults to five trainable backbone stages when weights=None,
whereas the original COCO-pretrained v2 constructor defaults to three.
requires_grad and BatchNorm training flags are not stored in state_dict.
"""
def restore_training_contract(model):
    from torch.nn.modules.batchnorm import _BatchNorm
    body=model.backbone.body
    trainable_prefixes=('layer2.','layer3.','layer4.')
    for name,parameter in body.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    frozen=0
    for module in model.modules():
        if isinstance(module,_BatchNorm):
            module.eval();frozen+=1
    return {'trainable_backbone_stages':3,'batchnorm_modules_frozen':frozen,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
