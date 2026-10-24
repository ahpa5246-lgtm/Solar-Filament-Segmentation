"""Fail-closed, pinned-source enhancements for solar filament PQ research.

The upstream project is MIT licensed and pinned by the notebook builder.
These source-to-source modifications are *training-only*: inference checkpoints
remain compatible with the original upstream U-Net and loader. No test data,
leaderboard results, or reserved confirmation labels enter model selection.
"""
from __future__ import annotations

import ast


def _replace_exactly_once(source: str, needle: str, replacement: str, label: str) -> str:
    if source.count(needle) != 1:
        raise ValueError(f"Pinned {label} source has drifted; refusing to patch")
    return source.replace(needle, replacement)


def patch_auto14_sources(train: str, losses: str, tune: str) -> dict[str, str]:
    """Introduce boundary-aware BCE and month-balanced local hyperparameter tuning.

    Boundary emphasis is optional and equals the original upstream loss
    *exactly* when SOLAR_BOUNDARY_WEIGHT=0. The Dice term is unchanged.
    Sample selection is deterministic and spans available validation months.
    """
    train = _replace_exactly_once(
        train,
        "train_stems, val_stems = split_stems(stems, args.val_frac, args.seed)",
        "development_stems, confirmation_stems = split_stems(stems, 0.15, 2026)\n"
        "        train_stems, val_stems = split_stems(development_stems, 0.18, 2027)",
        "train month split",
    )
    train = _replace_exactly_once(
        train,
        'json.dumps({"train": train_stems, "val": val_stems}, indent=1)',
        'json.dumps({"train": train_stems, "val": val_stems, "confirmation": confirmation_stems}, indent=1)',
        "train split ledger",
    )
    losses = _replace_exactly_once(
        losses,
        "import torch\n",
        "import os\nimport torch\n",
        "loss imports",
    )
    losses = _replace_exactly_once(
        losses,
        "        self.dice_weight = dice_weight\n",
        '        self.dice_weight = dice_weight\n'
        '        self.boundary_weight = float(os.environ.get("SOLAR_BOUNDARY_WEIGHT", "0"))\n'
        '        if not 0 <= self.boundary_weight <= 8:\n'
        '            raise ValueError("Boundary weight must be between 0 and 8")\n',
        "loss initializer",
    )
    losses = _replace_exactly_once(
        losses,
        "        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)\n",
        """        if self.boundary_weight == 0:
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        else:
            # Emphasize filament boundaries without erasing thin filaments.
            # The target is a soft multi-annotator consensus; do not binarize it.
            with torch.no_grad():
                upper = F.max_pool2d(target, kernel_size=7, stride=1, padding=3)
                lower = -F.max_pool2d(-target, kernel_size=7, stride=1, padding=3)
                edge = (upper - lower).clamp(0.0, 1.0)
                weight = 1.0 + self.boundary_weight * edge
            pixel_loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight, reduction="none"
            )
            bce = (pixel_loss * weight).sum() / weight.sum()
""",
        "loss forward",
    )
    tune = _replace_exactly_once(
        tune,
        "    if args.max_images:\n        val_stems = val_stems[: args.max_images]\n",
        """    if args.max_images:
        # Head-of-list sampling overrepresents early observation times.
        # Deterministic round-robin across ALL tuning months improves coverage.
        from collections import defaultdict
        from random import Random
        months = defaultdict(list)
        for stem in val_stems:
            months[stem[:6]].append(stem)
        rng = Random(2027)
        for month in sorted(months):
            rng.shuffle(months[month])
        sampled = []
        while len(sampled) < min(args.max_images, len(val_stems)):
            progressed = False
            for month in sorted(months):
                if len(sampled) >= args.max_images:
                    break
                if months[month]:
                    sampled.append(months[month].pop())
                    progressed = True
            if not progressed:
                break
        val_stems = sorted(sampled)
""",
        "tune balanced month sampler",
    )
    revised = {"scripts/train.py": train, "solarseg/losses.py": losses, "scripts/tune.py": tune}
    for path, text in revised.items():
        ast.parse(text, filename=path)
    return revised
