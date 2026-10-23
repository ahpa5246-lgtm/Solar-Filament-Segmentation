# Research plan after Kaggle public score 0.23 (2026-10-23)

## Measured baseline
- v1 LightGBM public score: **0.08** (user-provided Kaggle screenshot).
- v2 two-head U-Net public score: **0.23** after 30 GPU epochs (user-provided Kaggle screenshot and training log).
- v2 validation subset PQ: **0.3051**, 16 observations, TP=145, FP=249, FN=85. Not an official Kaggle score.
- v2 produced 2,570 masks over 180 test images. Count alone does not establish accuracy.
- v3 and v4 are **unrun experimental candidates**. No improvement is claimed yet.

## Evaluation warning
The challenge description emphasizes Panoptic Quality, while public analyses of the custom evaluator discuss instance-wise Dice. Neither our PQ nor our local matched-instance Dice proxy is guaranteed byte-equivalent to the hidden evaluator. Report both, with TP/FP/FN, splits/merges and morphology. Only official Kaggle submissions establish public scores. Never use test ground-truth labels.

## Staged free-GPU experiments

| Experiment | Hypothesis | Cost | Required evidence |
| --- | --- | --- | --- |
| V3 | Reuse v2 trained weights; 25-setting precision-focused postprocessing + four-view TTA | Kaggle GPU, **no retraining** | Tuning, independent confirmation, new CSV |
| V4 | New edge-guided attention model with topology loss disabled, isolating architecture | One Kaggle training run | Best checkpoint, history, confirmation, CSV |
| V5 | Soft-clDice 0.05 on same architecture and split; topology ablation | Kaggle or RTX 3090 | Paired PQ, Dice, FP/FN and centerline continuity |
| V6 | Seeds 42/1337/2026, ensembles only if validated | Local RTX 3090 | Per-seed variance, ensemble vs best single |

Run V3 before V4; never run them simultaneously on free Kaggle resources. Preserve the v2 0.23 submission.

## Validation and experimental controls
1. Date-grouped holdout prevents different annotations of one observation from leaking across train and validation.
2. Validation patches must be deterministic across epochs. Save random seeds, split manifest, architecture and loss settings.
3. V3/V4 select thresholds using 24 date-heldout observations; evaluate on 12 **different** date-heldout observations. Evaluate every annotation view.
4. Report exploratory 95% bootstrap intervals by observation, not by mask. Twelve observations give wide intervals.
5. Use the same TTA for calibration and final prediction; v2 used mismatched TTA settings.
6. Analyze spurious detections, missed filaments, splits/merges, matched-instance Dice proxy and PQ. Review train-validation overlays for barbs and limb darkening.
7. Validate schema and unique IDs before submission; retain every experiment's checkpoint and logs.
8. Use a second seed or fold to confirm substantial gains. Avoid repeated public-leaderboard overfitting.

## Published references
- Solomon et al., **EdgeAttNet: Towards Barb-Aware Filament Segmentation** (2025), https://arxiv.org/abs/2509.02964 and official MIT code https://github.com/dasjar/EdgeAttNet. Reports better multiscale mIoU than U-Net on MAGFiLO. **Not a Kaggle-score comparison.** Our edge-guided bottleneck is an original *inspired ablation*, not their exact architecture.
- Shit et al., **clDice – A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation** (CVPR 2021), https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html. Improves thin-structure continuity in its benchmarks, but may trade precision for recall.
- **Automated High-Precision Recognition of Solar Filaments Based on an Improved U2-Net** (Universe 2024), https://www.mdpi.com/2218-1997/10/10/381. Candidate for later experiments, on different data.
- Official evaluation: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission and organizer self-evaluation notebook https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook.

## Fair-play data policy
Do not use publicly released MAGFiLO ground-truth labels corresponding to competition test images, even if publicly obtainable. Only official **train** annotations may inform supervision and tuning. Test images are inference-only.

## Reproduction
- V3 notebook: notebooks/optimize_v3_kaggle.ipynb — attach official competition input **and the successful v2 notebook output containing best.pt**.
- V4 notebook: notebooks/train_edge_v4_kaggle.ipynb — official competition input only.
- CPU regression tests: pytest -q.
- Save v3_artifacts or v4_artifacts, including model, selected thresholds, confirmation, split manifest, training history and frozen environment.

## Promotion gate
Do not call a candidate better until its notebook completes, confirmation diagnostics are credible, and its **official public score exceeds 0.23**. The private score and qualitative judging may differ.
