# Technical report (working draft — NOT READY FOR SUBMISSION)

**Use the organizers' mandatory Overleaf template and four-page limit before submitting. Replace every `[MEASURE]` from actual model outputs. Never invent results.**

## Abstract
We study automatic instance segmentation of solar filaments in the MAGFiLO GONG H-alpha observations. We implement a reproducible patch-based multi-task U-Net with joint foreground and boundary objectives, reconstruct complete instances with marker-controlled watershed and optimize morphology postprocessing against held-out observation-date Panoptic Quality (PQ). Experimental results are pending completion of GPU training and scoring.

## 1 Introduction
Solar filaments are elongated structures visible as dark absorption features in H-alpha. Fine lateral barbs, weak contrast, nonuniform solar limb illumination, and adjacent dark features complicate delineation. The competition evaluates individual filament instances rather than a single binary region. Under the PQ objective, a fragmented filament and false connected components increase false-positive and false-negative counts.

## 2 Data and preprocessing
Use official MAGFiLO v1.0 competition JPEG 2048×2048 grayscale images and COCO polygon annotations. The multiple independently annotated copies of a physical observation remain grouped. For validation, hold out whole YYYYMMDD dates, keeping the split manifest and random seed in the output. Image channels are original grayscale intensity and smoothed-background dark-feature contrast. Tiles are sampled at native resolution with positive foreground oversampling, random geometric orientation and moderate intensity changes; validation does not use test annotations.

## 3 Architecture and training
Our two-input U-Net consists of four encoder downsampling levels and corresponding skip-connected decoder stages, with a two-logit output for foreground and boundaries. Group normalization reduces dependence on large batch sizes. Training uses AdamW, cosine LR schedule, mixed precision when on CUDA, a foreground class-weighted BCE plus soft Dice objective, and an auxiliary boundary-weighted BCE. The best checkpoint is selected by date-heldout loss. Record training time `[MEASURE]`, peak VRAM `[MEASURE]`, epoch `[MEASURE]`, learning rate history `[MEASURE]`.

## 4 Instance reconstruction
Whole-image 2048×2048 predictions are created from overlapping 512×512 native-resolution tiles with tapered averaging. High-confidence boundaries suppress seeds in likely touching regions; markers are obtained by connected-component analysis, then watershed expands seeds across foreground. Tiny masks are suppressed according to parameters selected by genuine held-out PQ. RLE encoding uses pycocotools and column names required by the competition. Report thresholds `[MEASURE]` and representative false splits/merges `[MEASURE]`.

## 5 Evaluation
Calculate PQ as sum of matched IoU divided by TP+0.5FP+0.5FN for one-to-one instance IoU > .5. Show held-out PQ `[MEASURE]`, Dice distribution `[MEASURE]`, IoU distribution `[MEASURE]`, false splits/merges `[MEASURE]`, ablation without boundary head `[MEASURE]`, 512-tile versus lower-res `[MEASURE]`, inference time per 2048×2048 image `[MEASURE]`, and Kaggle public score `[MEASURE]`. Distinguish all local scores from the private test score. Explain why labels differ between annotators.

## 6 Reproducibility and limitations
Source code, pinned libraries, random split manifests, trained weights/checkpoint availability, notebook with an end-to-end pipeline, expected input directory tree and exact CSV output instructions must be disclosed. Data access and U.S. legal payment eligibility follow the competition rules. Limitations include insufficient multi-scale context and uncertainty at solar limbs and over-merged adjacent filaments. Future work includes multi-scale context fusion, cross-annotator consensus and explicit instance embeddings, to be tested against the same held-out dates rather than inferred from a leaderboard alone.

## References
1. Ahmadzadeh et al., MAGFiLO, Scientific Data 2024, doi:10.1038/s41597-024-03876-y.
2. Kirillov et al., Panoptic Segmentation, CVPR 2019, doi:10.1109/CVPR.2019.00963.
3. IEEE BigDataCup 2026 challenge official overview and rule pages.
