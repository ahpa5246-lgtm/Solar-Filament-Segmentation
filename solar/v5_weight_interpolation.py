"""Controlled interpolation of the completed V5 detector continuation.

Hypothesis: 16 epochs reduced training loss but damaged end-to-end PQ.
A small interpolation with the frozen original checkpoint may retain the
original proposals while recovering useful trained features. No fitting,
test inference, or official submission occurs here.
"""
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)


def interpolate_states(baseline, trained, alpha):
    """Blend floating parameters, keep original integral counters and metadata."""
    import torch
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid interpolation coefficient")
    if baseline.keys() != trained.keys():
        raise ValueError("Detector state keys changed")
    result = {}
    for key, value in baseline.items():
        other = trained[key]
        if not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor):
            raise TypeError("Expected tensor detector state")
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Detector state tensor shape/dtype mismatch: " + key)
        if torch.is_floating_point(value):
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError("Nonfinite checkpoint tensor: " + key)
            result[key] = value.clone() if alpha == 0 else other.clone() if alpha == 1 else torch.lerp(value, other, alpha)
        else:
            # BatchNorm counters advance during continuation; they are not
            # learnable weights and must not block safe interpolation.
            # Preserve the baseline counter rather than blending integers.
            if not torch.equal(value, other) and not key.endswith(".num_batches_tracked"):
                raise ValueError("Nonfloating checkpoint state changed: " + key)
            result[key] = value.clone()
    return result


def run(annotation, manifest, detector_cache, plain_cache, baseline_checkpoint,
        refiner_checkpoint, source, final_checkpoint, finetune_report,
        expected_best_sha, out, seconds=3300):
    import csv
    import cv2
    import numpy as np
    import torch
    from .v5_finetune import validate_partitions
    from .v5_replay import verify_artifacts, V3_ARTIFACT_SHA256, load_predictions, aggregate
    from .v6_resolution_audit import original_detector, select_pilot_dates, detect, refine, DETECTOR_SHA
    from .v6_refiner_audit import load_components, postprocess, references, evaluate, REFINER_SHA, SOURCE_SHA
    from .v6_cached_search import micro, paired_ci
    from .v6_audit import measure_rles

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="STARTED", training=False, test_inference=False,
                  official_submission=False, architecture="V5 detector + frozen Crop U-Net",
                  hypothesis="Small interpolation may prevent the detector PQ regression from 16-epoch continuation",
                  tuning_alphas=list(ALPHAS), source_version="v2",
                  limitations=["Historical validation was used before; not a pristine independent holdout.",
                               "This is local PQ, not an official Kaggle score."])
    started = time.perf_counter()
    def save():
        pending = out / "interpolation_result.pending"
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(out / "interpolation_result.json")
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for detector/refiner inference")
        torch.manual_seed(2030)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(2)
        cv2.setNumThreads(0)
        if len(expected_best_sha) != 64:
            raise ValueError("Missing selected V5 checkpoint hash")
        prior = json.loads(Path(finetune_report).read_text())
        if (prior.get("status") != "COMPLETE" or prior.get("epochs_completed") != 16
                or prior.get("best_epoch") != 0 or prior.get("best_checkpoint_sha256") != expected_best_sha):
            raise ValueError("V2 immutable result does not match the no-gain continuation")
        for path, expected in ((source, SOURCE_SHA), (baseline_checkpoint, DETECTOR_SHA),
                               (refiner_checkpoint, REFINER_SHA)):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Original V5 artifact fingerprint mismatch")
        report["input_artifact_sha256"] = verify_artifacts({
            "training_manifest.csv": Path(manifest),
            "best_detector_predictions_1024.pkl": Path(detector_cache),
            "refined_validation_predictions_2048.pkl": Path(plain_cache),
        }, V3_ARTIFACT_SHA256)
        last = torch.load(final_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if last.get("epoch") != 16 or last.get("partial_epoch"):
            raise ValueError("Expected fully completed v2 epoch 16 checkpoint")
        base = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)["model"]
        trained = last["model"]
        if base.keys() != trained.keys():
            raise ValueError("Detector architecture changed")
        report["last_checkpoint_sha256"] = hashlib.sha256(Path(final_checkpoint).read_bytes()).hexdigest()
        rows = list(csv.DictReader(Path(manifest).open()))
        fitting = [r["observation_id"] for r in rows if r["split"] == "train"]
        validation = {r["observation_id"]: r for r in rows if r["split"] == "val"}
        tuning, confirmation = select_pilot_dates(fitting, validation, None)
        validate_partitions(fitting, tuning, confirmation)
        report["split_counts"] = {"fitting": len(fitting), "tuning": len(tuning),
                                  "confirmation": len(confirmation)}
        truth = references(annotation, set(tuning + confirmation))
        images = Path(annotation).parent / "train_images"
        cached = load_predictions(Path(detector_cache))
        model = original_detector(source).cuda()
        helpers = load_components(source)
        refiner = helpers["CropUNet"](pretrained=False)
        refiner.load_state_dict(torch.load(refiner_checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        refiner = refiner.cuda().eval()
        for parameter in refiner.parameters():
            parameter.requires_grad_(False)
        def assess(stems, check_parity=False):
            model.eval()
            predictions = {}
            parity = []
            with torch.inference_mode():
                for stem in stems:
                    if time.perf_counter() - started > seconds - 180:
                        raise TimeoutError("Interpolation evaluation budget reached; no model promotion")
                    image = cv2.imread(str(images / validation[stem]["file_name"]), cv2.IMREAD_GRAYSCALE)
                    if image is None or image.shape != (2048, 2048):
                        raise ValueError("Validation image missing or wrong resolution")
                    masks, scores = detect(model, image, 1024)
                    proposals = postprocess(masks, scores, "union")
                    if check_parity:
                        parity.append(measure_rles(cached[stem], proposals, shape=(1024, 1024))["pq_local"])
                        if parity[-1] < 0.99:
                            raise RuntimeError("Epoch-zero detector cache parity failed")
                    predictions[stem] = refine(refiner, helpers, image, proposals)
            return evaluate(truth, predictions, stems), min(parity) if parity else None

        trials = []
        baseline_stats = None
        for alpha in ALPHAS:
            model.load_state_dict(interpolate_states(base, trained, alpha), strict=True)
            stats, parity = assess(tuning, check_parity=(alpha == 0))
            if alpha == 0:
                baseline_stats = stats
                report["baseline_parity_min"] = parity
            trials.append({"alpha": alpha, "tuning_micro_pq": micro(stats)})
            report["tuning_trials"] = trials
            save()
            print("V5_INTERPOLATION_TUNING", json.dumps(trials[-1]), flush=True)
        winner = max(trials, key=lambda t: (t["tuning_micro_pq"], -t["alpha"]))
        alpha = winner["alpha"]
        report["selected_alpha"] = alpha
        report["baseline_tuning_pq"] = trials[0]["tuning_micro_pq"]
        report["selected_tuning_pq"] = winner["tuning_micro_pq"]
        model.load_state_dict(interpolate_states(base, trained, 0.0), strict=True)
        baseline_confirmation, _ = assess(confirmation)
        if alpha > 0 and winner["tuning_micro_pq"] > trials[0]["tuning_micro_pq"]:
            model.load_state_dict(interpolate_states(base, trained, alpha), strict=True)
            selected_confirmation, _ = assess(confirmation)
            interval = paired_ci(baseline_confirmation, selected_confirmation)
            supported = micro(selected_confirmation) > micro(baseline_confirmation) and interval["low"] > 0
        else:
            selected_confirmation = baseline_confirmation
            interval = {"estimate": 0.0, "low": 0.0, "high": 0.0}
            supported = False
        report.update(status="COMPLETE", confirmation_baseline=aggregate(baseline_confirmation),
                      confirmation_candidate=aggregate(selected_confirmation),
                      paired_macro_delta_ci=interval, local_gain_supported=supported,
                      official_public_pq=None, elapsed_seconds=time.perf_counter() - started,
                      next_action="Evaluate independently before full fit; no submission yet" if supported
                                  else "Retain baseline; move to train-only hard-negative detector experiment")
        if supported:
            torch.save({"model": interpolate_states(base, trained, alpha),
                        "interpolation_alpha": alpha, "source_epoch": 16}, out / "candidate_detector.pt")
            report["candidate_checkpoint_sha256"] = hashlib.sha256((out / "candidate_detector.pt").read_bytes()).hexdigest()
        save()
        print("V5_INTERPOLATION_RESULT", json.dumps(report), flush=True)
    except Exception as error:
        report.update(status="INCOMPLETE" if isinstance(error, TimeoutError) else "FAILED",
                      error_type=type(error).__name__, diagnosis=str(error)[:400],
                      elapsed_seconds=time.perf_counter() - started)
        save()
        raise
    return report
