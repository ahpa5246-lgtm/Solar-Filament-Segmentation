# Solar filament instance segmentation: working technical report

**Draft, not submitted.** Transfer to the organizers' current four-page template after verifying its instructions and deadline timezone. The owner reports November 15, 2026; this date's precise timezone is not independently verified. Final report/form submission requires explicit owner approval.

## Abstract

We investigate individual solar filament segmentation in full-disk 2048×2048 GONG H-alpha observations. The best owner-confirmed official Public Panoptic Quality (PQ) is **0.32**, from V5: a COCO-pretrained Mask R-CNN followed by a ResNet18 U-Net crop refiner. A forensic replay of recovered training-session artifacts measures local micro PQ of 0.362862 for the detector and 0.404278 after plain refinement. These local tuning-validation results are not official scores and cannot establish a leaderboard improvement. We implement fingerprint-verified replay, date-isolated postprocessing selection, instance diagnostics and bounded checkpoint inference before proposing further training.

## Data and validation protocol

Supervision uses only official competition TRAIN polygons. Test images are reserved for inference; no hidden or externally released test labels are used. Multiple annotator views of the same physical observation remain together. The recovered V5 manifest has 565 training and 142 validation observations, no physical-observation overlap, but ten shared acquisition dates. Its 142 validation observations contain 229 annotation-image variants.

For new cached-mask searches, we exclude validation observations sharing a training date. Seed 42 divides the remaining dates into 65 tuning observations and 67 confirmation observations, with no shared dates. The original model and strategy were already selected using the original validation set; this post-hoc confirmation does not undo that historical reuse. New training must use a fresh date-grouped training/model-selection/confirmation protocol.

## Method

V5 uses torchvision `maskrcnn_resnet50_fpn_v2`, COCO initialization, ResNet50/FPN proposals and instance-mask heads at approximately 1024×1024 detector resolution. Detector predictions are reconstructed at native resolution. Each retained instance defines a square crop with context factor 1.8 and minimum side 96 pixels. The grayscale crop is resized to 256×256 and repeated across three channels for an ImageNet-initialized ResNet18 U-Net encoder/decoder. Refiner probabilities are resized back into the native crop. An empty refined mask falls back to the detector mask.

Saved final configuration specifies six detector epochs, score threshold 0.6, mask threshold 0.4, eight refiner epochs and refiner threshold 0.85. Four-flip inference averages inverse-transformed logits before sigmoid. Both original stages merge transitive groups at any positive mask overlap. Their selection objectives used Dice-based metrics; V6 explicitly measures instance PQ. The exact recovered source, weights and manifest are fingerprinted. The source of recovered artifacts is API session `solar-filament-research-v5/v3`; this is a version of the V5 project notebook, not the historical V3 semantic model. Its association to the owner-confirmed scored submission is not independently established.

The bounded V6 refiner ablation compares plain/four-flip inference, thresholds 0.65/0.75/0.85/0.90 and three postprocessing policies: original overlap union, mask-IoU NMS at 0.5, and retaining masks. Only tuning observations select a candidate; only baseline and the selected candidate are evaluated on confirmation. Plain inference must reproduce cached PQ within 0.005 before interpretation. The run performs no training, test inference or official submission, and has an explicit GPU timeout. Results are reported separately when execution is verified.

## Evaluation and measured results

PQ is the matched IoU sum divided by TP+0.5FP+0.5FN. The retrieved public organizer helper uses strict IoU >0.5 qualifying pairs and unmatched rows/columns, rather than enforcing canonical one-to-one assignment. We retain both calculations and regression-test against the cited public helper. They coincide for the cached predictions measured here. Equivalence to unpublished competition code is not claimed.

| Method | Official Public PQ | Local cached micro PQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| V1 LightGBM | 0.08 | — | — | — | — |
| V2 semantic U-Net + watershed | 0.23 | — | — | — | — |
| V4 edge-guided attention U-Net | 0.24 | — | — | — | — |
| V5 project submission | **0.32** | — | — | — | — |
| Recovered V5 detector | — | 0.362862 | 895 | 681 | 671 |
| Recovered V5 detector + plain refiner | — | 0.404278 | 946 | 629 | 620 |

V3 has no verified official score. Local rows share cached observations and annotator views; their sample differs from the leaderboard. Plain refinement increases local micro PQ by 0.041416. SQ/RQ improve from 0.636935/0.569701 to 0.671161/0.602356. Matched-mask median IoU improves from 0.632653 to 0.677419. Precision/recall remain approximately 0.60. Heuristic split/merge counts decline from 23/51 to 21/48, but these counts do not establish the dominant physical error cause. Per-instance area/elongation profiles and private held-out overlays support further inspection.

A minimum-area ablation tested thresholds 0, 32, 64, 128, 256, 512, 1024 and 2048 pixels on cached plain-refiner predictions. The selected threshold was zero: removing masks did not improve tuning PQ. Confirmation baseline and candidate both scored 0.406185, with paired macro difference interval [0,0]. The unsuccessful experiment is retained. No leaderboard submission followed it.

Uncertainty estimates bootstrap physical observations, averaging annotator-specific PQ within each observation. These are macro-observation intervals, distinct from micro PQ. Detailed metrics, artifact hashes, experiment limitations and resource records are in `V5_CACHED_REPLAY.json`, `V6_KAGGLE_CPU_EXECUTION.json`, and the measured audit.

## Reproducibility, limitations and pending evidence

CPU unit tests cover evaluator parity, date isolation, native mask placement, overlap handling, compressed COCO encoding, unique submission identifiers, checkpoint round trips and notebook metadata. Fresh Kaggle CPU execution reproduced cached metrics using verified private inputs. Original V5 source and successful artifacts remain preserved. Optional new detector training is a separate bounded pilot and has not been presented as a trained improvement.

No alternative architecture has yet completed a fair training comparison. High-resolution proposals, topology supervision and ensembling remain hypotheses. Training cost, inference latency, peak memory and qualitative panels must be filled from completed experiments before final publication. No official score above 0.32, Private score, or independent final holdout result is claimed. The final runnable notebook, model availability and exact frozen CUDA environment must accompany the final selected method.

## References and attribution

- Ahmadzadeh et al., MAGFiLO, Scientific Data 2024, doi:10.1038/s41597-024-03876-y.
- Kirillov et al., Panoptic Segmentation, CVPR 2019, doi:10.1109/CVPR.2019.00963.
- Nomannic, [CV] Solar Filament | Mask R-CNN + U-Net Refiner, Kaggle, Apache-2.0; modifications and license copy are documented in `third_party/NOTICE.md`.
- Organizer self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook.
- Competition rules and final instructions: https://www.kaggle.com/competitions/filament-segmentation-2026/rules.
