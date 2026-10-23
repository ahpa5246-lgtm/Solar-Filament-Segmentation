> Status update 2026-10-23: V5 Version 5 completed on T4 ×2 and achieved owner-confirmed official Public PQ **0.32**; V4 is **0.24**. V3 has no verified score. Historical proposed version labels below are superseded by [the V6 audit](V6_AUDIT_2026-10-23.md). Preserve successful V5 artifacts.

# V4 forensic audit and V5 transfer-learning experiment (2026-10-23)

## Confirmed Kaggle outcomes (user-provided screenshot and run log)

| Version | Architecture | Public Kaggle score | Relative change |
|---|---|---:|---:|
| V1 | LightGBM | 0.08 | — |
| V2 | Two-head U-Net, trained 30 epochs | 0.23 | +0.15 |
| V4 | Edge-guided attention U-Net, trained 30 epochs | **0.24** | +0.01 vs V2 |
| V5 Version 5 | COCO Mask R-CNN + ResNet18 crop refiner | **0.32** | +33.3% vs V4 |

PQ lies in [0,1], so **0.5** (not 5) is the meaningful aspirational target. None of the new candidates has yet demonstrated it.

### V4 execution facts
- 18 synthetic tests passed; 2-sample T4 GPU memory preflight passed.
- 30 epochs completed, checkpoint saved. Lowest validation loss ~0.4858 at epoch 25. Epoch 30 validation loss ~0.4922.
- 24 observation threshold search selected `threshold=0.60, boundary=0.60, min_area=400, closing=3, min_seed=12`.
- Independent 12-observation confirmation with **the V4 model held constant**:
  - Selected postprocessing: PQ 0.2859, matched-instance Dice proxy 0.3213, TP 95, FP 211, FN 50.
  - Legacy V2 *threshold settings on V4 model*: PQ 0.2252, matched Dice proxy 0.2276, TP 99, FP 347, FN 46.
  - This is **not** a V2-vs-V4 architecture comparison; it is a postprocessing ablation within V4.
  - Bootstrap 95% intervals for PQ overlap substantially (chosen 0.219–0.374 vs old settings 0.174–0.296). The small 12-image set cannot establish a generalization gain.
- 2,335 predicted masks over 180 test observations; public Kaggle score 0.24.

### Root-cause hypotheses
1. Semantic U-Net produces a foreground probability map, then watershed heuristics try to recover independent filament objects. FP and splitting remain high; instance identity is not directly supervised.
2. Our architecture is a **lightweight EdgeAttNet-inspired ablation**, not the exact published EdgeAttNet; replacing the bottleneck alone did not produce a strong gain.
3. V4 training sampled 512x512 patches and was selected by BCE/Dice/boundary loss, not end-to-end full-disk instance PQ.
4. Multi-annotator MAGFiLO labels vary; even valid annotations disagree, so a single small confirmation split is noisy.
5. The public metric is documented as PQ by organizers; independent forensic analysis of public sibling scoring code suggests there may be metric implementation differences. We will not use hidden test labels, score exploits, or unsupported score promises.

## V5: transfer-learned **instance** detector plus crop refiner

**Source**: Nomannic, [CV] Solar Filament | Mask R-CNN + U-Net Refiner, Kaggle notebook, **Apache-2.0** (see third_party/NOTICE.md).

Architecture:
- COCO-pretrained `torchvision.maskrcnn_resnet50_fpn_v2` with ResNet50-FPN, RPN anchors tuned to filament scales; trained on official competition TRAIN polygons at 1024².
- Two T4 GPUs using DDP (torchrun); holdout uses unique physical observations, not annotation IDs.
- A separate ImageNet-pretrained ResNet18 crop U-Net refines each detected object on 256² native-resolution crops.
- Detector score and mask thresholds are selected on training-validation observations; refiner vs no-refiner and 4-view TTA vs single view are also selected on validation.
- Native 2048² COCO-RLE submission, checked for 180/180 test observations and unique IDs. Test images are inference-only.

**Not a guaranteed leaderboard gain.** The public source's own notebook currently displays a 0.33 public score on its individual page, while the competition's code listing has shown 0.69 for a related listing. Do not conflate these scores, and do not claim V5 will reach either. Score only after the user's own Kaggle run.

## Reproducibility and compliance
- Generator: `scripts/build_v5_from_apache_reference.py`.
- GitHub Action: `.github/workflows/sync-kaggle.yml`; pulls pinned public notebook **version 1**, verifies source signatures and parses all code cells, builds modified private `minahussain111/solar-filament-research-v5`, and syncs with `--no-run` (no GPU automatically consumed).
- User manually selects Kaggle GPU T4 x2, Internet On, official competition input, then **Save Version → Save & Run All**.
- Keep V2 and V4 as baselines. Report official V5 public score separately; do not tune on private test labels.
- Third-party attribution and Apache 2.0 license retained in generated notebook. Original public notebook is not committed wholesale to our repository.

## Further ablations after V5
- V5-A: detector-only vs detector+refiner vs detector+refiner+flip4; select via disjoint date-heldout images.
- V5-B: scale 1024 vs 1536 **only if GPU and heldout validation justify**, noting anchor rescaling may degrade accuracy.
- V6: exact upstream MIT EdgeAttNet implementation, rather than lightweight imitation, **only with properly documented adaptation and same train/validation split**.
- V7: controlled instance-level detector ensemble or fine-tuning (YOLO11-seg, Mask R-CNN) with independent validation and matched resource budgets.
