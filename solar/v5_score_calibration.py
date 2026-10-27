"""Bounded score calibration of preserved V5 weights; no training or test inference."""
import math

THRESHOLDS = (0.30, 0.45, 0.60)
BASELINE = 0.60


def choose_threshold(tuning):
    if set(tuning) != set(THRESHOLDS):
        raise ValueError('Incomplete preregistered tuning grid')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in tuning.values()):
        raise ValueError('Invalid tuning PQ')
    return max(THRESHOLDS, key=lambda t: (tuning[t], t == BASELINE, t))


def run(annotation, manifest, detector_predictions, plain_predictions,
        detector_checkpoint, refiner_checkpoint, source, out, seconds=1000):
    import csv
    import hashlib
    import json
    import time
    from pathlib import Path
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as coco
    from .v6_resolution_audit import original_detector, select_pilot_dates, refine, DETECTOR_SHA
    from .v6_refiner_audit import load_components, postprocess, references, evaluate, REFINER_SHA
    from .v5_replay import verify_artifacts, V3_ARTIFACT_SHA256, load_predictions, aggregate
    from .v6_cached_search import micro, paired_ci
    from .v6_audit import measure_rles, overlay
    from .v6_diagnostics import instance_profile, aggregate_profiles

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', training=False, test_inference=False,
                  official_submission=False, promote=False,
                  experiment='V5 fixed-weight detector score calibration',
                  thresholds=list(THRESHOLDS), baseline=BASELINE,
                  limitations=['Historical V5 validation reused; not a pristine final holdout.',
                               'Recovered API v3 weights are not independently linked to scored web Version 5.',
                               'Exploratory local comparison only; no automatic promotion or submission.'])
    started = time.perf_counter()
    try:
        verify_artifacts({'training_manifest.csv': Path(manifest),
                          'best_detector_predictions_1024.pkl': Path(detector_predictions),
                          'refined_validation_predictions_2048.pkl': Path(plain_predictions)},
                         V3_ARTIFACT_SHA256)
        report['checkpoint_sha256'] = {}
        for name, path, expected in [('detector', detector_checkpoint, DETECTOR_SHA),
                                     ('refiner', refiner_checkpoint, REFINER_SHA)]:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Checkpoint fingerprint mismatch')
            report['checkpoint_sha256'][name] = actual
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for bounded V5 inference')
        torch.manual_seed(42)
        np.random.seed(42)
        detector = original_detector(source)
        detector.load_state_dict(torch.load(detector_checkpoint, map_location='cpu', weights_only=True)['model'])
        detector = detector.cuda().eval()
        detector.transform.min_size = (1024,)
        detector.transform.max_size = 1024
        helpers = load_components(source)
        refiner = helpers['CropUNet'](pretrained=False)
        refiner.load_state_dict(torch.load(refiner_checkpoint, map_location='cpu', weights_only=True)['model'])
        refiner = refiner.cuda().eval()
        rows = list(csv.DictReader(Path(manifest).open()))
        training = {r['observation_id'] for r in rows if r['split'] == 'train'}
        validation = {r['observation_id']: r for r in rows if r['split'] == 'val'}
        tuning, confirmation = select_pilot_dates(training, validation, None)
        stems = sorted(tuning + confirmation)
        if set(s[:8] for s in tuning) & set(s[:8] for s in confirmation):
            raise ValueError('Tuning/confirmation dates overlap')
        truth = references(annotation, set(stems))
        cached = load_predictions(Path(detector_predictions))
        predictions = {threshold: {} for threshold in THRESHOLDS}
        images = Path(annotation).parent / 'train_images'
        parity = []
        torch.cuda.reset_peak_memory_stats()
        report.update(gpu=torch.cuda.get_device_name(0), used_gpus=1,
                      tuning_observations=len(tuning), confirmation_observations=len(confirmation))
        with torch.inference_mode():
            for number, stem in enumerate(stems, 1):
                if time.perf_counter() - started > seconds:
                    raise RuntimeError('Budget exhausted; no partial comparison accepted')
                image = cv2.imread(str(images / validation[stem]['file_name']), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (2048, 2048):
                    raise ValueError('Native validation image missing')
                resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(resized.astype(np.float32) / 255).unsqueeze(0).repeat(3, 1, 1).cuda()
                raw = detector([tensor])[0]
                scores = raw['scores'].cpu().numpy()
                masks = (raw['masks'][:, 0] >= .4).cpu().numpy().astype(np.uint8)
                for threshold in THRESHOLDS:
                    rles, confidences = [], []
                    for mask, score in zip(masks, scores):
                        if score >= threshold and mask.any():
                            rles.append(coco.encode(np.asfortranarray(mask)))
                            confidences.append(float(score))
                    proposals = postprocess(rles, confidences, 'union')
                    if threshold == BASELINE:
                        parity.append(measure_rles(cached[stem], proposals, shape=(1024, 1024))['pq_local'])
                        if parity[-1] < .99:
                            raise RuntimeError('V5 baseline cache parity failed')
                    predictions[threshold][stem] = refine(refiner, helpers, image, proposals)
                report['processed_observations'] = number
                print(f'V5 fixed-weight calibration {number}/{len(stems)}', flush=True)
        tuning_pq = {t: micro(evaluate(truth, p, tuning)) for t, p in predictions.items()}
        selected = choose_threshold(tuning_pq)
        # Selection is frozen before either confirmation branch is evaluated.
        base = evaluate(truth, predictions[BASELINE], confirmation)
        candidate = evaluate(truth, predictions[selected], confirmation)
        delta = paired_ci(base, candidate)
        profiles = {}
        for name, threshold in [('baseline', BASELINE), ('candidate', selected)]:
            profiles[name] = aggregate_profiles([instance_profile(gt, predictions[threshold][s])
                                                for s in confirmation for gt in truth[s]])
        report.update(status='completed', selected=selected, tuning_pq=tuning_pq,
                      detector_cache_parity_min=min(parity),
                      confirmation_baseline=aggregate(base), confirmation_candidate=aggregate(candidate),
                      paired_macro_delta_ci=delta, error_profiles=profiles,
                      local_gain_supported=(micro(candidate) > micro(base) and delta['low'] > 0),
                      elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
        (out / 'observations.json').write_text(json.dumps({'tuning': tuning, 'confirmation': confirmation}))
        hardest = sorted(confirmation, key=lambda s: sum(r['fp'] + r['fn'] for r in base[s]), reverse=True)[:6]
        for i, stem in enumerate(hardest):
            image = cv2.imread(str(images / validation[stem]['file_name']), cv2.IMREAD_GRAYSCALE)
            gt = np.stack([coco.decode(r) for r in truth[stem][0]]) if truth[stem][0] else np.zeros((0, 2048, 2048), bool)
            for name, threshold in [('baseline', BASELINE), ('candidate', selected)]:
                rles = predictions[threshold][stem]
                masks = np.stack([coco.decode(r) for r in rles]) if rles else np.zeros((0, 2048, 2048), bool)
                overlay(image, gt, masks, out / f'figures/{i:02d}-{name}.png')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        if type(error) in (ValueError, RuntimeError):
            report['diagnosis'] = str(error)[:500]
        raise
    finally:
        (out / 'score_calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
