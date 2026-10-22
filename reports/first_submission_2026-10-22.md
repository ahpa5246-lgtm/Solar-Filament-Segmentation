# First Kaggle leaderboard submission — 22 October 2026

**Status:** Real predictions exported; Kaggle upload and public score **not verified** yet. Do not describe this as a winning model.

## Dataset
- Official MAGFiLO Kaggle 2026 archive, restored from four checksum-verified pieces in the owner's Google Drive.
- 707 distinct training JPEGs, 180 test JPEGs, 1 annotation JSON (1,154 image annotation sets, 8,199 annotated filaments).
- No test masks or external ground-truth metadata used.

## First CPU prototype
- LightGBM 4.6.0 with 100 trees / 24 leaves, pixelwise binary classification.
- Ten local grayscale, Gaussian background, contrast and texture features.
- 42 training observation stems, with 8 separate held-out stems used for exploratory PQ checks. This first prototype does **not** guarantee day-group separation; use date-grouped holdout for the next model.
- Test inference at 1024 × 1024 for speed; instance masks reconstructed at the official 2048 × 2048 resolution.
- Components filtered by minimum area and morphology; default foreground probability threshold 0.975, fallback thresholds 0.95 / 0.92 / 0.89 applied only to previously empty images.
- Compressed COCO RLE count strings exported in a two-column CSV named `submission.csv`.

## Measurements (not official)
- Exploratory aggregated **PQ ≈ 0.161** on eight held-out stems evaluated at 1024 × 1024. This is **NOT** the official Kaggle metric or public leaderboard result; it is not a robust estimate of generalization.
- Test output contains **639 filament masks** across **177 of 180** test images. Three test images had no detections; absence of rows reflects zero predictions, not missing execution.
- CSV has exactly `filament_id,segmentation_rle`, unique IDs, and valid 2048×2048 RLE count sums; binary-mask encode/decode roundtrip tested.
- CSV byte count: 107,962. SHA-256: `3de22d13bedda88277eb8e63bcfdbeaec549e6c138a6c4aa071712a69603438b`.

## Next experiment
1. Upload `submission.csv` with the authorized Kaggle account to establish an authentic public baseline score.
2. Record Kaggle's score and any validator messages, **without inferring private leaderboard performance**.
3. Train the repository's high-resolution multi-task U-Net in a Kaggle GPU runtime and compare with the official self-evaluation notebook.
4. Tune fragmentation/overmerging, calibration and image-specific brightness variation using observation-date grouped validation, not by overfitting public leaderboard queries.
5. Produce the required 4-page technical report and a fully reproducible public solution before final submission.

The codebase and research report are separate from the official competition acceptance. The user has not yet authorized a Kaggle API session in this assistant runtime, so no actual Kaggle submission was performed.
