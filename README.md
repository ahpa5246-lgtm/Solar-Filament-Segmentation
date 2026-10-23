# Solar Filament Segmentation — IEEE BigDataCup 2026

**Status (2026-10-23): Kaggle-confirmed public score 0.23 for the trained two-head U-Net v2 (30 epochs, 180 test images); previous LightGBM baseline 0.08.** The new v3 threshold/TTA and v4 edge-attention experiments are implemented but **not yet run or scored**. See [the experiment plan](reports/EXPERIMENT_PLAN_2026.md).

A reproducible Python research implementation for [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026): identify **individual** solar-filament instances from 2048×2048 grayscale GONG H-alpha images, and export one pycocotools compressed RLE count per instance into `submission.csv`.

The organizers changed the primary metric to **Panoptic Quality (PQ)**, where instance grouping and spurious/missed detections are critical. The competition's research-quality assessment also evaluates morphology, code readability, and reproducibility; leaderboard position is not a sufficient win condition.

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

- Python 3.10–3.11, `requirements.txt` pinned versions, recommended CUDA GPU with 12+ GiB VRAM (reduce batch if OOM).
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

**The v2 submission passed the full-data CUDA training and Kaggle public scoring stages** (0.23); its 16-observation held-out PQ was 0.3051, not a leaderboard metric. The v3/v4 research candidates remain untrained/unscored as applicable. New CPU synthetic tests validate software behavior, not scientific accuracy. Kaggle GPU execution requires the user's authenticated Kaggle notebook.

Before final evaluation, produce a **four-page technical PDF using the organizers' Overleaf template**, publish the exact final runnable model and dependencies, and submit the required form by **15 November 2026, 11:59 PM CDT**. The prizes additionally depend on eligibility and U.S. bank-transfer compatibility. The competition permits AI development assistance, but entrants must understand and defend the code and method.

## Ethics and licensing

Code in this repository is original and distributed under MIT (OSI approved). Uses PyTorch (BSD), scikit-image (BSD), SciPy (BSD), OpenCV (Apache), NumPy (BSD), and pycocotools (BSD-like permissive); check each pinned upstream license for distribution. Training data © its respective owners: **MAGFiLO / GONG H-alpha dataset CC BY-NC 4.0**, not included in code license. Models trained on the data must obey competition restrictions. No competition-data-derived labels from the test set or additional ground-truth metadata are used.

The project is an independently developed implementation with no code copied from other Kaggle competitors' solutions. Edge-guided attention and soft-clDice experiments are research-inspired and separately identified, not claimed reproductions of the original papers. Never train on ground-truth labels corresponding to the competition test images, even if public copies are available. Useful references: [official self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), [MAGFiLO publication](https://doi.org/10.1038/s41597-024-03876-y), [Panoptic Quality](https://doi.org/10.1109/CVPR.2019.00963).
