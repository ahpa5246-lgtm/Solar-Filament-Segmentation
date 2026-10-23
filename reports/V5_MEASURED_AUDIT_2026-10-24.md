# Measured V5 audit — 2026-10-24

V5 remains the owner-confirmed best **official Public PQ 0.32**. All measurements below are **local cached tuning-validation replay**, not leaderboard scores. The recovered training artifacts belong to Kaggle API session `minahussain111/solar-filament-research-v5/v3`; its association with the owner's scored submission has not been independently verified. API `v5` is a preserved code snapshot with a rendering log, not the source of the recovered checkpoints.

## Recovered evidence

Read-only authenticated Actions recovered the original training manifest, selected detector predictions, plain-refiner predictions, and configurations. Replay uses only official competition TRAIN polygons. No hidden test labels, model training, or official submission occurred. Raw data, predictions and weights are not committed. SHA256 fingerprints are recorded and enforced before replay.

The original split has **565 training and 142 validation observations**, with no shared physical observation. However, **10 acquisition dates overlap**. Evaluation covers **229 annotator-image variants** across 142 observations. The public reference's all-pairs PQ and canonical one-to-one PQ coincide on these particular cached masks.

| Cached strategy | Local micro PQ | SQ | RQ | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector | 0.362862 | 0.636935 | 0.569701 | 895 | 681 | 671 | 0.567893 | 0.571520 |
| Detector + plain refiner | 0.404278 | 0.671161 | 0.602356 | 946 | 629 | 620 | 0.600635 | 0.604087 |

Refinement increases local micro PQ by **0.041416**, about **11.4% relative**. These are paired cached observations but a paired CI for this comparison has not yet been computed. Separate macro-observation bootstrap estimates (2,000 resamples) are detector 0.360689, 95% CI [0.334518, 0.384997], and plain refiner 0.397622, CI [0.369754, 0.423281]. Macro intervals and micro PQ are different aggregates.

The refiner improves both matched overlap and recognition; removing it would discard a measured benefit. Precision and recall near 0.60 show substantial false positives and misses. Matched-mask median IoU rises from 0.632653 to 0.677419. Split/merge heuristics count 23/51 for the detector and 21/48 after refinement across annotator variants. These heuristics do not prove physical failure causes or establish that merging is the dominant limitation.

## Source findings

- Detector threshold calibration and refiner/flip4 strategy selection use Dice-based objectives, not the public-reference PQ now reproduced by V6.
- Both detector and refiner postprocessing union transitive groups at **any positive overlap**. A synthetic regression demonstrates distinct overlapping masks being merged. The effect on solar PQ remains unmeasured because the cached predictions were already merged.
- Physical observations are deduplicated, but the original split is stratified by site/count rather than grouped by date. Date overlap is now measured rather than hypothesized.
- Saved final configuration: detector epochs 6, score threshold 0.6, mask threshold 0.4, overlap merging enabled, refiner enabled, **flip4 enabled**, refiner epochs 8, refiner threshold 0.85. The surviving validation cache contains **plain refinement only**. Do not apply conclusions to flip4 test predictions without a matching replay.

## Unsuccessful inexpensive experiment retained

Minimum-area filtering tested native-mask cutoffs 0, 32, 64, 128, 256, 512, 1024 and 2048 pixels. The 10 validation observations sharing training dates were excluded. The remaining pool was split by date into **65 tuning observations** and **67 confirmation observations**, with no date overlap.

Cutoffs through 128 tied the baseline; 256 reduced tuning PQ, and larger cutoffs deteriorated substantially. Tie-breaking retained **min_area=0**. Confirmation PQ was **0.406185 for both baseline and selected candidate**, delta 0 and paired macro bootstrap interval [0,0]. This experiment failed to improve PQ and is not promoted. The original V5 strategy was already selected on all its original validation data, so this post-hoc confirmation is not a pristine final holdout for the original model.

## Next experiments justified by evidence

1. Preserve original code and checkpoints. Validate fresh-session artifact mounting and exact input hashes on CPU before using an accelerator.
2. Reconstruct plain and flip4 refinement on original detector crops using the saved validation checkpoint. Recover raw confidence-bearing detector proposals, then compare PQ-based calibration and mask NMS against any-overlap union. Keep the same date-isolated tuning/confirmation protocol and preserve baseline configurations.
3. Inspect held-out overlays and error-size/contrast distributions before deciding whether multiscale resolution or topology refinement addresses actual misses. Do not thicken masks or union neighboring objects merely to increase counts of matches.
4. Compare a genuine alternative instance method only after establishing the corrected baseline, using the same training data, date holdout and compute budget. No alternative framework or model is claimed evaluated yet.

Current quota query reports 30 GPU hours total and 4.2724 used at observation time. It is not a reservation or guarantee. Any inference execution must be bounded and recorded separately from training and official evaluation.

Machine-readable measurements: [V5_CACHED_REPLAY.json](V5_CACHED_REPLAY.json). Original source snapshot: [v5_version5.ipynb](../notebooks/baselines/v5_version5.ipynb). No higher official score is claimed.

## Fresh-session reproduction and bounded refiner experiment

The private immutable dataset cache has verified Kaggle `isPrivate=true` metadata. CPU notebook `solar-filament-research-v6-audit`, version 2, completed with all three source-artifact fingerprints enforced and reproduced the table above. See `V5_PRIVATE_CACHE.json` and `V6_KAGGLE_CPU_EXECUTION.json`.

The next experiment is inference-only: exact archived best-refiner checkpoint, original detector crops, plain versus four-flip logit averaging, thresholds 0.65/0.75/0.85/0.90, and any-overlap union versus mask-IoU NMS versus retaining masks. Selection uses only the fixed date-isolated tuning observations. Only the selected candidate and the original flip4/0.85/union baseline are compared on confirmation observations. Plain replay must reproduce cached micro PQ within 0.005 before selection. Original V5 holdout reuse remains a limitation. No test inference or official submission is performed. Kaggle execution is limited to 1,200 seconds with a minimum remaining quota guard; the application inference budget is 1,000 seconds. Although T4×2 is allocated, only device 0 is used. A separate read-only review verified original crop/TTA parity and identified a runner status check that was corrected before launch.
