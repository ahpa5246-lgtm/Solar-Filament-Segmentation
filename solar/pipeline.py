"""Whole-resolution tiled inference, checkpoint IO, and COCO submission export."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from .data import feature_channels, read_gray
from .model import Net
from .postprocess import Params, to_instances


def load_model(path: str | Path, device: str | torch.device = 'cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError('Invalid model checkpoint, expected {state_dict, base, ...}')
    model = Net(base=ckpt.get('base',24)).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.eval(), ckpt


def starts(length: int, tile: int, stride: int) -> list[int]:
    if tile > length or tile <= 0 or stride <= 0 or stride > tile:
        raise ValueError('Require 0 < stride <= tile <= image size')
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


@torch.inference_mode()
def predict_probabilities(model, grayscale: np.ndarray, tile=512, stride=384, batch_size=4, device='cpu', tta=False) -> np.ndarray:
    """Return [2,H,W] probabilities with overlapping feathered tile blending."""
    feature = feature_channels(grayscale)
    h,w,_ = feature.shape
    y_starts, x_starts = starts(h,tile,stride), starts(w,tile,stride)
    coords = [(y,x) for y in y_starts for x in x_starts]
    pred = np.zeros((2,h,w),np.float32)
    weights = np.zeros((h,w),np.float32)
    window = np.outer(np.hanning(tile),np.hanning(tile)).astype(np.float32)
    window = np.maximum(window, .03)
    autocast = torch.autocast(device_type='cuda') if str(device).startswith('cuda') else nullcontext()
    model.eval()
    for start in range(0,len(coords),batch_size):
        group = coords[start:start+batch_size]
        patches = np.stack([feature[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group])
        batch = torch.from_numpy(np.ascontiguousarray(patches)).to(device)
        with autocast:
            logits = model(batch)
            probabilities = logits.float().sigmoid()
            if tta:
                flip = model(torch.flip(batch,(-1,))).float().sigmoid()
                probabilities = .5*(probabilities+torch.flip(flip,(-1,)))
        probs = probabilities.cpu().numpy()
        for (y,x),prob in zip(group,probs):
            pred[:,y:y+tile,x:x+tile] += prob*window
            weights[y:y+tile,x:x+tile] += window
    if not np.all(weights > 0):
        raise RuntimeError('Incomplete tile coverage: check stride and image size.')
    pred /= weights[None]
    return pred


def instances_to_rows(stem: str, instances: np.ndarray) -> list[dict]:
    """COCO compressed counts only, one row per instance (NOT polygon/RLE sizes)."""
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError('Install requirements.txt to export official COCO RLE counts.') from exc
    rows=[]
    for value in np.unique(instances):
        if value == 0:
            continue
        bitmap = np.asfortranarray((instances==value).astype(np.uint8))
        encoded = coco_mask.encode(bitmap)
        counts = encoded['counts'].decode('ascii')
        rows.append({'filament_id':f'{stem}_{len(rows)+1}', 'segmentation_rle':counts})
    return rows


def write_submission(rows: list[dict], dest: str | Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('No detected instances. Check training and thresholds; do not upload an empty CSV.')
    seen = set()
    for row in rows:
        if set(row) != {'filament_id','segmentation_rle'}:
            raise ValueError('CSV columns must be filament_id and segmentation_rle')
        if not row['filament_id'] or not row['segmentation_rle'] or row['filament_id'] in seen:
            raise ValueError('Missing/duplicate filament_id or RLE count')
        seen.add(row['filament_id'])
    with dest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['filament_id','segmentation_rle'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} predicted filaments to {dest}')


def predict_folder(model, directory, dest, params=Params(), tile=512, stride=384, batch_size=4, device='cpu', tta=False, max_images=0):
    paths=sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if max_images:
        paths=paths[:max_images]
    if not paths:
        raise FileNotFoundError(f'No JPEG images under {directory}')
    rows=[]
    for n,p in enumerate(paths,1):
        grayscale=read_gray(p)
        if grayscale.shape != (2048,2048):
            raise ValueError(f'Invalid Kaggle image shape: {p} = {grayscale.shape}')
        probabilities=predict_probabilities(model,grayscale,tile,stride,batch_size,device,tta)
        instances=to_instances(probabilities[0],probabilities[1],params)
        rows.extend(instances_to_rows(p.stem,instances))
        print(f'[{n}/{len(paths)}] {p.stem}: {instances.max()} instances',flush=True)
    write_submission(rows,dest)
    return rows
