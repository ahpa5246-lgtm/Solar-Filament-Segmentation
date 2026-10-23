"""V6 torchvision detector adapter, separate from and compatible with no V5 files.

New checkpoints use this module's explicit format. Never guess the structure of
V5 checkpoints or load untrusted pickles. No pretrained downloads in CPU tests.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_detector(pretrained=True,min_size=1024,max_size=1024):
    if min_size < 32 or max_size < min_size:
        raise ValueError('Require 32 <= min_size <= max_size')
    model=maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,min_size=min_size,max_size=max_size)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.roi_heads.mask_predictor=MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels,256,2)
    return model


def save_checkpoint(model,path,config):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format':'solar-v6-detector-1','state_dict':model.state_dict(),'config':json.loads(json.dumps(config))},path)


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format') != 'solar-v6-detector-1':
        raise ValueError('Not a V6 detector checkpoint; V5 files require their original loader')
    config=checkpoint['config']
    model=build_detector(False,config['min_size'],config['max_size'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval(),config


@torch.inference_mode()
def infer(model,image,score_threshold=.5,mask_threshold=.5):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 grayscale image')
    if not 0 <= score_threshold <= 1 or not 0 < mask_threshold < 1:
        raise ValueError('Invalid confidence/mask threshold')
    device=next(model.parameters()).device
    tensor=torch.from_numpy(image.copy()).float().div(255).unsqueeze(0).expand(3,-1,-1).to(device)
    model.eval()
    prediction=model([tensor])[0]
    keep=prediction['scores']>=score_threshold
    masks=(prediction['masks'][keep,0]>=mask_threshold).cpu().numpy()
    scores=prediction['scores'][keep].cpu().numpy()
    nonempty=masks.any(axis=(1,2))
    return masks[nonempty],scores[nonempty]
