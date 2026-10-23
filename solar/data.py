"""MAGFiLO reader. Splits are grouped by acquisition DAY, not annotation ID.

Multiple annotators see an identical image. Never place that observation
in both training and validation, even via separate annotation IDs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

SIZE = 2048
EXT = {'.jpg', '.jpeg', '.png'}


def discover(root: str | Path) -> tuple[Path, Path, Path]:
    """Find images/annotations under an unzipped data directory or /kaggle/input."""
    root = Path(root)
    found = list(root.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one training JSON under {root}; found {len(found)}. Attach competition data first.')
    ann = found[0]
    train = ann.parent / 'train_images'
    base = ann.parent.parent
    test = base / 'test' / 'test_images'
    if not train.is_dir() or not test.is_dir():
        raise FileNotFoundError(f'Expected {train} and {test}.')
    return ann, train, test


def load_records(root: str | Path) -> tuple[list[dict], Path, Path]:
    ann, train_dir, test_dir = discover(root)
    with ann.open(encoding='utf-8') as f:
        payload = json.load(f)
    by_image = defaultdict(list)
    for a in payload['annotations']:
        by_image[str(a['image_id'])].append(a)
    indexed = {p.stem: p for p in train_dir.rglob('*') if p.suffix.lower() in EXT}
    records = []
    for img in payload['images']:
        stem = Path(img['file_name']).stem
        if stem not in indexed:
            raise FileNotFoundError(f'Missing training image {stem} in {train_dir}')
        records.append({'image_id': str(img['id']), 'stem': stem,
                        'path': str(indexed[stem]), 'ann': by_image[str(img['id'])]})
    if not records:
        raise ValueError('Empty MAGFiLO training annotations.')
    return records, train_dir, test_dir


def split_records(records: list[dict], val_frac: float = .16, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Hold out whole observation DATES; all re-annotations share a split."""
    if not 0 < val_frac < .5:
        raise ValueError('val_frac must be between 0 and 0.5')
    days = sorted({r['stem'][:8] for r in records})
    if len(days) < 2:
        raise ValueError('Need at least two observation days for honest validation.')
    rng = np.random.default_rng(seed)
    rng.shuffle(days)
    count = min(len(days) - 1, max(1, round(len(days) * val_frac)))
    val_days = set(days[:count])
    train = [r for r in records if r['stem'][:8] not in val_days]
    val = [r for r in records if r['stem'][:8] in val_days]
    assert {x['stem'] for x in train}.isdisjoint({x['stem'] for x in val})
    return train, val


def read_gray(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f'Cannot read image: {path}')
    return arr


def polygons_to_instances(annotations: list[dict], shape=(SIZE, SIZE)) -> np.ndarray:
    """Render COCO polygon targets; use one annotator's labels per sample.

    All source annotations use a single polygon per filament, but we also
    handle multiple polygons defensively. Label 0 is the background.
    """
    canvas = np.zeros(shape, dtype=np.int32)
    for idx, ann in enumerate(annotations, start=1):
        seg = ann.get('segmentation') or []
        if isinstance(seg, dict):
            raise ValueError('RLE training labels not expected; polygon required.')
        for polygon in seg:
            pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts = np.rint(pts).astype(np.int32)
            cv2.fillPoly(canvas, [pts], idx)
    return canvas


def feature_channels(image: np.ndarray) -> np.ndarray:
    """Two channels, fixed image-wide preprocessing; input shape H,W,2."""
    image = image.astype(np.float32) / 255.0
    # Contrast against a broad smooth background; dark filaments go positive.
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=23.0)
    dark = np.clip((background - image) * 5.0 + .5, 0, 1)
    return np.stack([image, dark], axis=-1).astype(np.float32)


def derive_targets(instance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fg = (instance > 0).astype(np.float32)
    # Boundary along adjacent instance borders (where separation matters).
    boundary = np.zeros_like(fg)
    for dy, dx in [(0, 1), (1, 0)]:
        a = instance[:instance.shape[0]-dy or None, :instance.shape[1]-dx or None]
        b = instance[dy:, dx:]
        mismatch = (a != b) & (a > 0) & (b > 0)
        boundary[:boundary.shape[0]-dy or None, :boundary.shape[1]-dx or None] = np.maximum(
            boundary[:boundary.shape[0]-dy or None, :boundary.shape[1]-dx or None], mismatch)
    # Thin exterior contours also sharpen localization for small filaments.
    border = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return fg, np.maximum(boundary, border * .5).astype(np.float32)


class TileDataset:
    """On-the-fly full-resolution training patches, positive-heavy sampling.

    No pre-cropping of images is needed; avoids a huge intermediary dataset.
    """
    def __init__(self, records, tile=512, samples_per_epoch=1200, seed=13, training=True):
        self.records = records
        self.tile = tile
        self.samples = samples_per_epoch
        self.seed = seed
        self.training = training
        self.epoch = 0
        self.annotators = defaultdict(list)
        for r in records:
            self.annotators[r['stem']].append(r)
        self.unique = sorted(self.annotators)

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch) if self.training else 0

    def __getitem__(self, index):
        import torch
        # Independent RNG even with multiple data-loader workers.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, self.epoch]))
        stem = self.unique[int(rng.integers(len(self.unique)))]
        variants = self.annotators[stem]
        rec = variants[int(rng.integers(len(variants)))]
        im = read_gray(rec['path'])
        inst = polygons_to_instances(rec['ann'], im.shape)
        h, w = im.shape
        t = self.tile
        if h < t or w < t:
            raise ValueError(f'Image {rec["path"]} smaller than tile {t}')
        if rng.random() < .78 and inst.max() > 0:
            ys, xs = np.nonzero(inst)
            k = int(rng.integers(len(ys)))
            y = int(np.clip(int(ys[k]) - rng.integers(t//4, 3*t//4), 0, h-t))
            x = int(np.clip(int(xs[k]) - rng.integers(t//4, 3*t//4), 0, w-t))
        else:
            y, x = int(rng.integers(h-t+1)), int(rng.integers(w-t+1))
        # Image-wide correction before cropping is essential for train/test parity.
        feat = feature_channels(im)[y:y+t, x:x+t]
        inst = inst[y:y+t, x:x+t]
        for _ in range(int(rng.integers(4))):
            feat, inst = np.rot90(feat), np.rot90(inst)
        if rng.random() < .5:
            feat, inst = feat[:, ::-1], inst[:, ::-1]
        if rng.random() < .5:
            feat, inst = feat[::-1], inst[::-1]
        feat = np.ascontiguousarray(feat)
        # Mild photometric augmentation only on the intensity channel.
        if self.training:
            scale = float(rng.uniform(.90, 1.10))
            shift = float(rng.uniform(-.025, .025))
            feat[..., 0] = np.clip(feat[..., 0] * scale + shift, 0, 1)
            # A global affine intensity transform also scales the background
            # subtraction channel around its neutral value of 0.5.
            feat[..., 1] = np.clip(.5 + (feat[..., 1] - .5) * scale, 0, 1)
        fg, boundary = derive_targets(inst)
        return (torch.from_numpy(feat.transpose(2, 0, 1).copy()),
                torch.from_numpy(np.stack([fg, boundary], axis=0)))
