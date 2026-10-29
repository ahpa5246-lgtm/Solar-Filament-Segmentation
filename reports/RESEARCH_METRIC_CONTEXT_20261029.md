# Research review: metric and annotation context

The public [topology-safe 0.70 handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff) dates its confirmed result to 5 August 2026 and explicitly describes the leaderboard/evaluator as preliminary. Its public model card documents official-training-only provenance and Apache-2.0 checkpoint licensing. We downloaded and safely inspected its checkpoint tensor dictionaries; we did not execute its notebook, import test labels, or claim its published score as ours.

The later [new-metric pipeline](https://www.kaggle.com/code/ektarr/new-metric-solar-filament) documents the 7 August switch from Dice to Panoptic Quality, requiring IoU > 0.5 for matches. Therefore the older 0.70 result must not be interpreted as 0.70 PQ or used as proof that importing its weights will achieve our current target. Current project scores are verified by fresh official submissions, not notebook titles or leaderboard-era comparisons.

Other inspected public references include [YOLO detect + seeded crop refinement](https://www.kaggle.com/code/timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net) and [classical dark-residual segmentation](https://www.kaggle.com/code/vmerckle/simple-dark-filament-segmentation-0-46). Their reported numbers are not independently verified as current PQ. No copied predictions or external test annotations were used.

The new-metric pipeline suggests physical-observation folds, medoid annotator targets, full detector-to-refiner OOF evaluation, and fold ensembles. These are research leads, not proven improvements in this project. Our fitting-only annotation audit found substantial disagreement between different expert records of the same image. Medoid targets may reduce contradictory supervision, but must undergo training and official competition evaluation before any performance or reliability claim.

At this review, our verified official sequence is 0.32 → 0.36 → 0.37. The current frozen-model resolution experiment has no official result yet. No 0.40 or 0.60 milestone is claimed.
