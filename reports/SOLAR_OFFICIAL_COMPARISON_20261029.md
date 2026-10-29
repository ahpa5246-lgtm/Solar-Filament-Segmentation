# Official comparison, 29 October 2026

Kaggle API independently rechecked submission 57760055: COMPLETE, Public PQ **0.36**. Previous V5 baseline **0.32**; absolute gain **0.04**. Neither 0.40 nor 0.60 reached.

YOLO11m-seg COCO transfer with multi-annotator fitting examples and guarded training-only background crops replaced Mask R-CNN proposals; the original CropUNet still refined their masks. This combination improved the official score, but these changes were bundled, so no individual cause is established by an official ablation. Confirmation PQ 0.4416 is a local result and overstates the official 0.36.

Original YOLO submission 57759532 was rejected for overlapping masks. Version 18 repairs version 17's frozen predictions by confidence-priority pixel ownership on 27 images; all 1,074 instances remain across 180 images. Original rejected predictions and prior best are preserved separately. No unchanged CSV was resubmitted.

## Next experiment

Reuse the existing refiner notebook to test box-conditioned grayscale/CLAHE refinement, fitting-only negative examples, and explicit rejection of empty predictions. Preserve exact v17 checkpoint inputs in a private dataset because the current v18 notebook only contains repaired predictions. Train on Kaggle, select checkpoint/configuration on tuning data, inspect frozen confirmation, generate all test predictions, and obtain the official score before choosing another approach. Local comparisons guide selection and do not substitute for the official assessment.

Work stays in this Codex conversation. Both discovered ChatGPT Solar scheduled tasks were already paused and remain paused; their tools cannot rebind them to this conversation. No claim is made that an unattended Codex job or model switch was configured.

## Active execution

Existing notebook `minahussain111/solar-filament-research-v6-refiner/v2` is running on Kaggle (launched 2026-10-29 05:41 UTC). Resume monitor and automatic assessment with `/workspace/solar-venv/bin/python scripts/run_seed_refiner.py`; it detects the recorded version and never pushes another version when the ledger exists. Ledger: `reports/SEED_REFINER_EXECUTION_20261029.json`. If Kaggle reports an error, inspect the exact v2 log and fix that error before relaunching. If predictions complete, the controller validates every mask and requests the official assessment once; later invocations only poll the recorded submission.
