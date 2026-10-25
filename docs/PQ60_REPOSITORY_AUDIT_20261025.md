# PQ 0.60 — verifiable peer-repository audit (2026-10-25)

**Official baseline:** Kaggle submission 57571678 scored 0.32 public PQ (authenticated Kaggle API). **Research challenger:** real day-2 GPU training completed in 3,418.9 seconds. Split: 489 train, 111 tuning, 107 untouched confirmation. Its local tuning PQ is **0.40438**, not an official or independent result. Private checkpoint SHA256: d213d3d8bb616a6b9d44fef1219f48727eed148fe68a79e7c0800f534af97a3f.

## Relevant public repositories reviewed (maintainer claims, not independent Kaggle scores)

| Repository | Public score claimed / verified local score | Relevant approach | Caveat |
|---|---|---|---|
| [FarnooshMemari/solar-filament-segmentation](https://github.com/FarnooshMemari/solar-filament-segmentation) | public 0.32 / validation 0.389 | Month-grouped U-Net, soft multi-annotator labels, PQ tuning | MIT code; current research challenger's frozen source |
| [HeShen-1/filament-segmentation-2026](https://github.com/HeShen-1/filament-segmentation-2026) | public 0.35 / mean 5-fold 0.3593 | YOLO-seg at 1536, ResNet crop refiner | MIT; additional model fusion showed negative results |
| [theainerd/filament_segmentation](https://github.com/theainerd/filament_segmentation) | public 0.40 / 5-fold local 0.4519 | Ensemble of EfficientNet segmentation models, instance proposals and a rescorer | MIT source; published pretrained weights require separate permissions review |
| [FELIPEACASTRO/solar-filament-2026](https://github.com/FELIPEACASTRO/solar-filament-2026) | historical public 0.71, NOT verified current PQ | Mask R-CNN and crop refiner, NMS, expert disagreement analysis | Organizer rescored all submissions in August 2026. Historical 0.71 is not proof of current PQ; weights not published |
| [ali-119/Solar-Filament-Segmentation](https://github.com/ali-119/Solar-Filament-Segmentation) | no verified official PQ | EfficientNetB0 fine-tuned semantic U-Net | Semantic Dice is not instance PQ |
| [Agnuxo1/solar-filament-segmentation-2026](https://github.com/Agnuxo1/solar-filament-segmentation-2026) | none established | Preparation and evaluation scaffold | No trained model |
| [ameyypawar/filament-seg](https://github.com/ameyypawar/filament-seg) | not verified | Annotation disagreement, fragmentation-aware matching | Methods need paired evaluation |
| [Wzy633/solar-filament-segmentation-2026](https://github.com/Wzy633/solar-filament-segmentation-2026) | public 0.31 / local 0.35259 | YOLO26 instance segmenter, date grouping, higher resolution inference | Dependency and license obligations must be reviewed before borrowing |

Do not reproduce other teams' private data, unlicensed weights, organizer scorers with uncertain rights or leaderboard files. Read original license/notice before copying code. No external result supports a guarantee that our pipeline can achieve PQ 0.60.

## Concrete execution and acceptance contract

1. Use ONLY the genuinely trained private Kaggle day-two checkpoint, verifying exact SHA256.
2. A separate private inference notebook evaluates all 107 held-out confirmation physical observations, none used for training or postprocessing calibration.
3. Reject if independent local PQ <0.36 or if any data-date leakage, missing artifact or provenance mismatch occurs. Always preserve the old V5 model.
4. Only on a pass, infer all 180 unlabeled test frames, check native 2048x2048 compressed COCO RLE, disjoint instance masks, all image IDs, and final CSV checksum.
5. The CSV must differ from the old V5 CSV (SHA256 94b985ff19e0848d655d0a524265ca67d150fa203531ca18780d3a009cd586d6). Check Kaggle history to avoid duplicate submissions, then allow at most one official upload.
6. Record the actual Kaggle return status, submission ID and public PQ. No metric is asserted without Kaggle scoring.

GitHub CI passing demonstrates software integrity only. A local tuned PQ of 0.40438 is optimistic. Public PQ 0.60 remains **NOT ACHIEVED** unless a future authenticated competition submission reports it.

The permanent 14-day GPU schedule, special early validation, and November 7 final all run under independent fail-closed identity/date gates. Organizer switched to Panoptic Quality in August 2026 and explicitly states a PQ above 0.35 is scientifically valuable. See https://www.kaggle.com/competitions/filament-segmentation-2026/overview.
