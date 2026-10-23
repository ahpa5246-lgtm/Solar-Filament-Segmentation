# V5 Version 5 source review

Version label `v5` verified with the official SDK; session status COMPLETE. Source download may omit execution outputs; no accuracy inferred.

SHA256: `262946128a7838a5899c2b7966217f796c3a8059f1cb55ee2b5df283e075e13e`

Selected source statements for forensic review (not executed):

```python
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(manifest.index, test_size=0.20, random_state=SEED, stratify=strata)
        for mask_threshold in mask_grid:
            binary_masks = probabilities >= mask_threshold
            for score_threshold in score_grid:
                masks = detector.merge_overlaps(binary_masks[scores >= score_threshold])
                prediction_grids[(score_threshold, mask_threshold)][observation_id] = [encode_mask(mask) for mask in masks]
    calibration_rows.append({"score_threshold": configuration[0], "mask_threshold": configuration[1], "selected_dice": selected_metrics["mean_dice"], "all_dice": all_metrics["mean_dice"], "all_union_dice": all_metrics["union_dice"], "all_f1_50": all_metrics["f1_50"], "count_error": all_metrics["count_error"], "predictions": all_metrics["predictions"]})
calibration["selection_score"] = (calibration["selected_dice"] + calibration["all_dice"]) / 2
best_configuration = (float(best_row["score_threshold"]), float(best_row["mask_threshold"]))
    json.dump({"score_threshold": best_configuration[0], "mask_threshold": best_configuration[1]}, file, indent=2)
def merge_overlapping_rles(rles):
    mode:{obs:merge_overlapping_rles(predictions.get(obs,[]))
    comparison.append({"strategy": name, "selected_dice": selected_metrics["mean_dice"], "all_dice": all_metrics["mean_dice"], "all_union_dice": all_metrics["union_dice"], "all_f1_50": all_metrics["f1_50"], "count_error": all_metrics["count_error"], "predictions": all_metrics["predictions"]})
best_strategy=str(comparison['all_dice'].idxmax())
use_refiner=best_strategy!='Detector'
use_refiner_tta=best_strategy=='Detector + refiner + flip4'
print(f"Use refiner: {use_refiner} | Final refiner epochs: {best_refiner_epoch}")
if use_refiner:
final_configuration = {"detector_epochs": best_detector_epoch, "score_threshold": detector_thresholds["score_threshold"], "mask_threshold": detector_thresholds["mask_threshold"], "overlap_merge": True, "use_refiner": use_refiner, "refiner_flip4": use_refiner_tta, "refiner_epochs": best_refiner_epoch if use_refiner else None, "refiner_threshold": 0.85 if use_refiner else None}
if final_configuration["use_refiner"]:
score_threshold = final_configuration["score_threshold"]
mask_threshold = final_configuration["mask_threshold"]
        keep = output["scores"] >= score_threshold
        detector_masks = (output["masks"][keep, 0] >= mask_threshold).byte().cpu().numpy().astype(bool)
        prediction_rles = merge_overlapping_rles(prediction_rles)
```
