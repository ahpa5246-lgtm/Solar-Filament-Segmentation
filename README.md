# Solar Filament Segmentation — IEEE BigDataCup 2026

## V5: pretrained Mask R-CNN + U-Net refiner (best verified baseline)

**Verified public scores (2026-10-23):** V1 0.08, V2 0.23, V4 0.24, **V5 Version 5: 0.32** (confirmed by the owner; not independently re-fetched in this session). V3 has no verified score.

The existing semantic U-Net + watershed pipeline suffers from false-positive and instance-identity errors. V5 changes the *model family*: a COCO-pretrained Mask R-CNN ResNet50-FPN-v2 predicts **separate instances**, followed by an ImageNet-pretrained ResNet18 crop U-Net that refines their masks. A four-flip TTA ablation is selected using official **training-validation** labels, never test labels.

V5 is generated reproducibly from [Nomannic's Apache-2.0 public Kaggle notebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner), with attribution and modified cells documented in [third_party/NOTICE.md](third_party/NOTICE.md). The GitHub Action pulls the pinned upstream notebook, checks its structure and Python syntax, and saves [V5 on Kaggle](https://www.kaggle.com/code/minahussain111/solar-filament-research-v5) with **--no-run**. Run the Kaggle notebook manually using Internet On, official competition input, GPU T4 x2, and Save Version → Save & Run All. The successful Version 5 trained on two T4 GPUs. New runs may take over an hour and must not overwrite the successful artifacts. **No score improvement is promised.**

Detailed forensic report: [V4 failure and V5 experiment](reports/V4_FAILURE_AND_V5_RESEARCH.md).


**Status (2026-10-23): confirmed Kaggle public scores V1 0.08, V2 0.23, V4 0.24, V5 0.32.** Preserve V5 Version 5 and its checkpoints/submission. See [the experiment plan](reports/EXPERIMENT_PLAN_2026.md) and [V4 forensic report](reports/V4_FAILURE_AND_V5_RESEARCH.md).

A reproducible Python research implementation for [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026): identify **individual** solar-filament instances from 2048×2048 grayscale GONG H-alpha images, and export one pycocotools compressed RLE count per instance into `submission.csv`.

The organizers changed the primary metric to **Panoptic Quality (PQ)**, where instance grouping and spurious/missed detections are critical. The competition's research-quality assessment also evaluates morphology, code readability, and reproducibility; leaderboard position is not a sufficient win condition.

## V6 audit and experiment tools

[V6 audit status and blockers](reports/V6_AUDIT_2026-10-23.md) distinguishes source findings from unmeasured failure modes. `solar.v6_forensics` evaluates independent overlapping instance masks, retains every annotation variant, produces IoU/Dice distributions and contour overlays, and bootstraps by physical observation. Original V5 fitting and holdout manifests are mandatory; a newly generated split cannot validate an already trained model.

The self-contained [V6 notebook](notebooks/v6/solar_filament_v6.ipynb) runs audit readiness by default without GPU training. The private Kaggle slug is prepared but deployment is not yet confirmed. A bounded, single-GPU detector pilot is explicitly opt-in, uses date-separated train/tuning/confirmation partitions, selects checkpoints by local PQ, and preserves all V5 files. It is not yet an improved model or a comparison with a different instance-segmentation method.

```bash
python -m pip install -r requirements-v6.txt
python -m solar.v6_forensics --help
python -m solar.v6_train --help
```

## Scientific/methodological choices

1. **Day-grouped holdout** prevents the same observation re-annotated by multiple people (or another observation on the same day) from appearing in both train and validation.
2. **High-resolution training** samples positive-biased 512×512 patches directly from 2048×2048 source images, preserving fine barbs better than globally shrinking the solar disk.
3. **Two-channel input**: physical grayscale H-alpha image plus a broad local dark-feature contrast response; no RGB conversion.
4. **Two-head U-Net** predicts foreground and boundary structure, using Dice + weighted BCE and a boundary auxiliary loss. No pretrained weights or mystery-model API are required.
5. **Instance-aware restoration** extracts markers outside highly probable boundaries and applies watershed to reconstruct complete filaments.
6. **Overlapping inference** uses cosine feathering to avoid seams between tiles. Optional horizontal flip TTA.
7. **PQ-based tuning** uses held-out observation dates and multiple independent annotations; it never optimizes to the public Kaggle leaderboard or accesses hidden evaluation labels.
8. **COCO RLE** is encoded via `pycocotools` with Fortran contiguous binary arrays. Each prediction gets a unique `image_stem_N` ID.

The v2 pipeline was trained and scored on Kaggle (public 0.23); these design choices are not claims of state-of-the-art performance. Stronger candidates must be compared on the same held-out observations and via official Kaggle evaluation. The organizer's published PQ description and some public discussions of custom instance-Dice evaluation do not perfectly align; local metrics are proxies, not official scores.

## Requirements

- Python 3.10+ (CPU environment verified with 3.12.14), `requirements.txt` pinned versions, recommended CUDA GPU with 12+ GiB VRAM (reduce batch if OOM).
- The **original competition dataset** (~751 MB), acquired by an authorized Kaggle account **after accepting the rules**. It is CC BY-NC 4.0 and is **not included or redistributed** here.
- For an easy reproducible GPU run: upload `notebooks/run_on_kaggle.ipynb` to Kaggle, enable GPU + Internet, attach the competition data, then **Run All**. **Kaggle does not execute this automatically just because the repository exists.** The resulting `submission.csv` must still be uploaded via *Submit Prediction*. See the notebook for commands.

## Local CLI

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
# unzip the official Kaggle archive inside ./data or another directory
python -m solar.audit --data ./data
python -m solar.train --data ./data --out ./artifacts --epochs 35 --steps-per-epoch 280 --batch 4
python -m solar.tune --data ./data --checkpoint ./artifacts/best.pt --out ./artifacts --max-images 16
python -m solar.predict --data ./data --checkpoint ./artifacts/best.pt --params ./artifacts/best_params.json --out ./submission.csv --tta
```

The inference command scans **every** test JPEG unless debugging with `--max-images`. Do not submit a debug subset. The Kaggle CSV must contain exactly `filament_id,segmentation_rle` with COCO **compressed RLE counts only**, not size metadata. `submission.csv` is ignored by Git because it contains predictions; share it using Kaggle submission UI.

## Repository layout

```
solar/data.py          MAGFiLO COCO-JSON loading, split, polygon masks, GPU data augmentation
solar/model.py         Multi-task U-Net and BCE + Dice losses
solar/postprocess.py   Boundary-aware watershed and instance selection
solar/pipeline.py      Tiled inference, checkpoint loading, COCO RLE CSV validation
solar/metrics.py       IoU > .5 matched Panoptic Quality and TP/FP/FN
solar/train.py         Training, best-validation-loss checkpoint, seed/split logs
solar/tune.py          Held-out PQ postprocessing search
solar/predict.py       Kaggle submission generator
solar/audit.py         Data integrity statistics
solar/research_metrics.py  Matched-instance Dice proxy, split/merge diagnostics
notebooks/optimize_v3_kaggle.ipynb  Reuse trained v2 weights, precision threshold search, 4-view TTA
notebooks/train_edge_v4_kaggle.ipynb  Edge-guided attention architecture experiment
notebooks/             Kaggle end-to-end execution notebook
reports/               Honest technical report draft, measurements still required
tests/                 Synthetic tests (no Kaggle data or GPU required)
```

## What still requires experimental verification?

**The v2 submission passed the full-data CUDA training and Kaggle public scoring stages** (0.23); its 16-observation held-out PQ was 0.3051, not a leaderboard metric. V4 scored 0.24 and V5 scored 0.32; V3 has no verified score. New CPU synthetic tests validate software behavior, not scientific accuracy. Kaggle GPU execution requires the user's authenticated Kaggle notebook.

Before final evaluation, produce a **four-page technical PDF using the organizers' Overleaf template**, publish the exact final runnable model and dependencies, and submit the required form by the previously recorded **15 November 2026** deadline (precise timezone and current instructions must be reverified on the official page). The prizes additionally depend on eligibility and U.S. bank-transfer compatibility. The competition permits AI development assistance, but entrants must understand and defend the code and method.

## Ethics and licensing

Code in this repository is original and distributed under MIT (OSI approved). Uses PyTorch (BSD), scikit-image (BSD), SciPy (BSD), OpenCV (Apache), NumPy (BSD), and pycocotools (BSD-like permissive); check each pinned upstream license for distribution. Training data © its respective owners: **MAGFiLO / GONG H-alpha dataset CC BY-NC 4.0**, not included in code license. Models trained on the data must obey competition restrictions. No competition-data-derived labels from the test set or additional ground-truth metadata are used.

The original semantic pipeline is independently developed; V5 is an attributed modification of the Apache-2.0 reference described above. Edge-guided attention and soft-clDice experiments are research-inspired and separately identified, not claimed reproductions of the original papers. Never train on ground-truth labels corresponding to the competition test images, even if public copies are available. Useful references: [official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), [MAGFiLO publication](https://doi.org/10.1038/s41597-024-03876-y), [Panoptic Quality](https://doi.org/10.1109/CVPR.2019.00963).
