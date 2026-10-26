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


def patch_auto14_postprocess_source(source: str) -> str:
    """Optionally reconnect aligned components; zero gap = upstream identity.

    Modifies only the postprocess layer. Both daily notebook and final notebook
    must apply this identical patch before importing solarseg.postprocess.
    """
    source = _replace_exactly_once(
        source, "import numpy as np\n",
        "import numpy as np\n\nfrom .auto14_geometry import link_aligned_fragments\n",
        "postprocess geometry import",
    )
    source = _replace_exactly_once(
        source, "    merge_dist: int = 0\n",
        "    merge_dist: int = 0\n    link_gap: int = 0\n",
        "postprocess configuration",
    )
    source = _replace_exactly_once(
        source, "    return filter_components(raw, cfg.min_area, cfg.max_instances)\n",
        """    result = filter_components(raw, cfg.min_area, cfg.max_instances)
    if cfg.link_gap:
        return link_aligned_fragments(result, max_gap=cfg.link_gap)
    return result
""",
        "postprocess instance recovery",
    )
    ast.parse(source, filename="solarseg/postprocess.py")
    return source


def patch_auto14_sources(train: str, losses: str, tune: str, postprocess: str | None = None) -> dict[str, str]:
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
        '        if not 0 <= self.boundary_weight <= 8:\n'
        '            raise ValueError("Boundary weight must be between 0 and 8")\n',
        '        if not 0 <= self.boundary_weight <= 8:\n'
        '            raise ValueError("Boundary weight must be between 0 and 8")\n'
        '        self.topology_weight = float(os.environ.get("SOLAR_TOPOLOGY_WEIGHT", "0"))\n'
        '        if not 0 <= self.topology_weight <= 0.5:\n'
        '            raise ValueError("Topology weight must be between 0 and 0.5")\n',
        "loss topology initializer",
    )
    losses = _replace_exactly_once(
        losses,
        "class BCEDiceLoss(nn.Module):\n",
        "\ndef _soft_erode(x):\n    # Directional 3x1 / 1x3 erosion protects long thin structures.\n    return torch.minimum(\n        -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0)),\n        -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1)),\n    )\n\n\ndef _soft_skeleton(x, iterations=3):\n    # Differentiable approximation; compute on TRAINING pixels only.\n    opened = F.max_pool2d(_soft_erode(x), 3, stride=1, padding=1)\n    skeleton = F.relu(x - opened)\n    for _ in range(iterations):\n        x = _soft_erode(x)\n        opened = F.max_pool2d(_soft_erode(x), 3, stride=1, padding=1)\n        detail = F.relu(x - opened)\n        skeleton = skeleton + F.relu(detail - skeleton * detail)\n    return skeleton\n\n\ndef soft_cldice_loss(logits, target):\n    \"\"\"Soft skeleton connectivity; bounded and differentiable for empty masks.\"\"\"\n    probability = torch.sigmoid(logits.float())\n    soft_target = target.float().clamp(0.0, 1.0)\n    pred_skeleton = _soft_skeleton(probability)\n    target_skeleton = _soft_skeleton(soft_target)\n    eps = 1.0\n    precision = ((pred_skeleton * soft_target).sum() + eps) / (pred_skeleton.sum() + eps)\n    sensitivity = ((target_skeleton * probability).sum() + eps) / (target_skeleton.sum() + eps)\n    return 1.0 - (2.0 * precision * sensitivity) / (precision + sensitivity + 1e-6)\n\n\n" + "class BCEDiceLoss(nn.Module):\n",
        "loss topology helper",
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
    losses = _replace_exactly_once(
        losses,
        "        return bce + self.dice_weight * soft_dice_loss(logits, target)\n",
        "        total = bce + self.dice_weight * soft_dice_loss(logits, target)\n"
        "        if self.topology_weight:\n"
        "            total = total + self.topology_weight * soft_cldice_loss(logits, target)\n"
        "        return total\n",
        "loss topology contribution",
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
    if postprocess is not None:
        revised["solarseg/postprocess.py"] = patch_auto14_postprocess_source(postprocess)
    for path, text in revised.items():
        ast.parse(text, filename=path)
    return revised
