"""Date-held-out postprocessing search with a disjoint confirmation subset.

All tuning and confirmation labels come ONLY from the official training JSON.
Scores are not Kaggle scores. Do not use competition test annotations.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .data import load_records, split_records, read_gray, polygons_to_instances
from .metrics import compare, Counts
from .research_metrics import diagnose_instances
from .pipeline import load_model, predict_probabilities
from .postprocess import Params, to_instances


def parameter_grid(profile: str) -> list[Params]:
    if profile == 'baseline':
        return [Params(threshold=t, boundary=b, min_area=a, closing=3, min_seed=12)
                for t, b, a in product([.30, .40, .50], [.50, .65], [20, 50, 100])]
    if profile == 'precision':
        # The v2 baseline has many false positives; explore more selective
        # foreground/area settings rather than simply making masks larger.
        return [Params(threshold=t, boundary=b, min_area=a, closing=3, min_seed=12)
                for t, b, a in product([.40, .50, .60, .70], [.45, .60], [100, 200, 400])]
    raise ValueError(f'Unknown profile: {profile}')


def evaluate_one(gt, pred):
    counts = compare(gt, pred)
    diag = diagnose_instances(gt, pred)
    return counts, diag


def _aggregate(images):
    result = Counts()
    dice = []
    splits, merges = 0, 0
    for image in images:
        for a in image['annotations']:
            c = a['counts']
            result += Counts(sum_iou=c['sum_iou'], tp=c['tp'], fp=c['fp'], fn=c['fn'])
            dice.append(a['matched_dice_proxy'])
            splits += a['likely_splits']
            merges += a['likely_merges']
    return {'pq':float(result.pq),
            'matched_dice_proxy':float(np.mean(dice)) if dice else 0.,
            'tp':result.tp, 'fp':result.fp, 'fn':result.fn,
            'likely_splits':splits, 'likely_merges':merges}


def _bootstrap_pq(images, seed=42, samples=300):
    if not images:
        return [0., 0.]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        chosen = rng.integers(0, len(images), size=len(images))
        scores.append(_aggregate([images[int(i)] for i in chosen])['pq'])
    return [float(x) for x in np.percentile(scores, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/kaggle/input')
    ap.add_argument('--checkpoint', default='/kaggle/working/solar_artifacts/best.pt')
    ap.add_argument('--out', default='/kaggle/working/solar_artifacts')
    ap.add_argument('--max-images', type=int, default=24, help='Observations used for parameter selection')
    ap.add_argument('--audit-images', type=int, default=12, help='Disjoint observations used for confirmation')
    ap.add_argument('--profile', choices=['baseline', 'precision'], default='precision')
    ap.add_argument('--tta-mode', choices=['none', 'flip', 'd4'], default='flip')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--stride', type=int, default=384)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.max_images < 1 or args.audit_images < 1:
        ap.error('Both tuning and confirmation need at least one observation')
    records, _, _ = load_records(args.data)
    _, validation = split_records(records, seed=args.seed)
    groups = defaultdict(list)
    for record in validation:
        groups[record['stem']].append(record)
    rng = np.random.default_rng(args.seed)
    stems = list(rng.permutation(sorted(groups)))
    if len(stems) < args.max_images + args.audit_images:
        ap.error(f'Only {len(stems)} validation observations available; reduce image counts')
    tune_stems = stems[:args.max_images]
    audit_stems = stems[args.max_images:args.max_images + args.audit_images]
    assert set(tune_stems).isdisjoint(audit_stems)
    model, _ = load_model(args.checkpoint, args.device)
    tta = {'none':False, 'flip':True, 'd4':'d4'}[args.tta_mode]
    grid = parameter_grid(args.profile)
    # Include the known v2 setting as an explicit comparison.
    baseline = Params(threshold=.50, boundary=.50, min_area=100, closing=3, min_seed=12)
    if baseline not in grid:
        grid.append(baseline)
    summaries = [Counts() for _ in grid]
    dice_sum = [0. for _ in grid]
    dice_n = [0 for _ in grid]

    for n, stem in enumerate(tune_stems, 1):
        variants = groups[str(stem)]
        image = read_gray(variants[0]['path'])
        probs = predict_probabilities(model, image, args.tile, args.stride,
                                      args.batch, args.device, tta=tta)
        truths = [polygons_to_instances(r['ann'], image.shape) for r in variants]
        for i, params in enumerate(grid):
            pred = to_instances(probs[0], probs[1], params)
            for gt in truths:
                counts, diag = evaluate_one(gt, pred)
                summaries[i] += counts
                dice_sum[i] += diag.matched_dice_proxy
                dice_n[i] += 1
        print(f'tuning {n}/{len(tune_stems)}: {stem}', flush=True)

    ranking = sorted(range(len(grid)), key=lambda i:(summaries[i].pq,
                     dice_sum[i] / max(dice_n[i], 1)), reverse=True)
    results = [{'params':asdict(grid[i]), 'pq':float(summaries[i].pq),
                'matched_dice_proxy':float(dice_sum[i]/max(dice_n[i],1)),
                'tp':summaries[i].tp, 'fp':summaries[i].fp, 'fn':summaries[i].fn}
               for i in ranking]
    chosen = grid[ranking[0]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'tuning_results.json').write_text(json.dumps(results, indent=2))
    (out/'best_params.json').write_text(json.dumps(asdict(chosen), indent=2))
    (out/'tuning_manifest.json').write_text(json.dumps({
        'seed':args.seed, 'profile':args.profile, 'tta_mode':args.tta_mode,
        'tuning_observations':list(map(str,tune_stems)),
        'confirmation_observations':list(map(str,audit_stems))}, indent=2))
    print('Best tuning subset:', results[0], flush=True)

    confirmations = {'chosen':[], 'v2_baseline':[]}
    for n, stem in enumerate(audit_stems, 1):
        variants = groups[str(stem)]
        image = read_gray(variants[0]['path'])
        probs = predict_probabilities(model, image, args.tile, args.stride,
                                      args.batch, args.device, tta=tta)
        truths = [polygons_to_instances(r['ann'], image.shape) for r in variants]
        for name, params in [('chosen',chosen), ('v2_baseline',baseline)]:
            pred = to_instances(probs[0], probs[1], params)
            entry = {'stem':str(stem), 'annotations':[]}
            for gt in truths:
                counts, diag = evaluate_one(gt, pred)
                entry['annotations'].append({
                    'counts':asdict(counts),
                    'matched_dice_proxy':diag.matched_dice_proxy,
                    'likely_splits':diag.likely_splits,
                    'likely_merges':diag.likely_merges})
            confirmations[name].append(entry)
        print(f'confirmation {n}/{len(audit_stems)}: {stem}', flush=True)
    report = {'selected_params':asdict(chosen), 'profile':args.profile,
              'tta_mode':args.tta_mode, 'tune_images':len(tune_stems),
              'confirmation_images':len(audit_stems),
              'note':'All labels from date-heldout TRAIN data; Dice is a local proxy.',
              'results':{
                  key:{**_aggregate(value),
                       'pq_bootstrap_95ci':_bootstrap_pq(value,seed=args.seed)}
                  for key,value in confirmations.items()},
              'per_image':confirmations}
    (out/'confirmation_results.json').write_text(json.dumps(report,indent=2))
    print('Independent threshold confirmation:',report['results'],flush=True)
    print('NOT a Kaggle leaderboard result.',flush=True)


if __name__ == '__main__':
    main()
