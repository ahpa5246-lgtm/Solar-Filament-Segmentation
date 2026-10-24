# Solar Filament Segmentation — automatically generated evidence handoff

Generated only from public aggregate GitHub ledger. **Not** the required final PDF/report-form submission. Empty evidence stays empty.

## Executive summary

The project investigates instance segmentation of solar H-alpha images at native 2048 × 2048 resolution. The official competition criterion is **Panoptic Quality (PQ)** under the organizers' updated August 2026 evaluation; it is **not interchangeable with binary Dice**. Existing confirmed V5 public baseline: **0.32 (prior owner-confirmed V5 public leaderboard; not replicated in this report)**. New official score: **No new official Kaggle score verified**. A 0.70 score is an aspiration, never a guaranteed outcome.

## Dataset, partitioning and leakage prevention

The Kaggle challenge provides 707 training and 180 hidden-test physical images, with multiple expert annotators on some training images. The challenger splits training observations by calendar month to reduce near-duplicate leakage. Approximately 15% of month groups are reserved exclusively for one final independent confirmation, while development data is further split into training and model-selection/tuning. Checkpoint selection and threshold tuning on the development subset are exploratory; we do not interpret the highest tuning PQ as an unbiased performance estimate. The final holdout is not consulted during scheduled training.

## Reproducible training and architecture provenance

The existing repository retains its own V5 detector-plus-crop-refiner code and checkpoints. The 14-day challenger is **derived from a public MIT-licensed U-Net implementation by Farnoosh Memari** (copyright 2026). Upstream repository: https://github.com/FarnooshMemari/solar-filament-segmentation, immutable source commit ef37d2e93d4cc9e248595e791ee39fe520d392d0. The unchanged upstream MIT LICENSE is retained when cloned on Kaggle. This project's contribution is a date-grouped experimental protocol, adaptation, trained checkpoints, diagnostic code, conservative promotion gate, reproducible scheduling and result tracking; the upstream model design must not be claimed as independently invented.

One private Nvidia T4 Kaggle notebook is scheduled for each day 2026-10-24 through 2026-11-06 UTC. Experiments vary U-Net width, resolution, positive class weight, learning exposure and random seed under fixed train/validation partitions. Every experiment preserves best.pt, last.pt, configuration and tuning results privately in Kaggle rather than pushing weights/data into the public repository. GitHub Actions executes the runs under GPU quota constraints.

## Recorded experiments

Attempted/recorded days: **1**. Successful training notebooks: **0**.

| Day | Kaggle training status | Exploratory LOCAL tuning PQ | Private checkpoint SHA256 prefix |
|---|---|---|---|
| 1 | FAILED | n/a | n/a |

The scores in this table use a local open-source evaluator and are subject to selection bias. They are **not** competition leaderboard measurements and cannot establish superiority to V5.

## Locked final confirmation and competition delivery

Chosen candidate (from LOCAL tuning only): **none**. Independent LOCAL confirmation PQ: **Not independently measured**. Final execution state: **not yet executed**. Public official Kaggle result: **No new official Kaggle score verified**.

The frozen one-shot final workflow is scheduled for November 7 UTC. Its promotion rule requires a valid, unused month-grouped confirmation set and local PQ >= 0.36. When those conditions pass, it generates COCO-compressed-RLE predictions for all 180 competition test images, checks IDs/masks/CSV structure, and attempts one official Kaggle upload. No successful run of that workflow is inferred merely because GitHub CI checks passed before November 7. Full evaluation and team eligibility still depend on the competition organizers.

## Limitations, auditability and credit

A fixed open-source semantic U-Net is a useful challenger but may not split adjacent filament instances as effectively as a Mask R-CNN plus crop-refiner system. Thin solar structures, overlapping annotator masks, uncertainty at instance matching IoU > 0.5, test-time distribution shift, and the size of the withheld test sample can all limit generalization. The daily experimental loop explores a finite set of configurations; it is not autonomous architecture discovery. A reported competitor public 0.70–0.71 from mid-2026 must not be presented as independently verified current post-rescoring PQ. Public competition references and evaluation details: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation .

Code and experiment log: https://github.com/ahpa5246-lgtm/Solar-Filament-Segmentation .
Kaggle reference: https://www.kaggle.com/code/minahussain111/solar-filament-research-v5 .
The final four-page technical report and required submission form must separately be filed by November 15, 2026, 11:59 pm CDT. This machine-generated evidence handoff supplies reusable, attributed statements but is **not** an automatic official final-report filing.
