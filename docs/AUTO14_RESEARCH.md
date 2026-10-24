# Solar AutoResearch — 14-day autonomous Kaggle GPU program

## Objective and scientific boundary

Research toward a public score of 0.70 is an ambition, not a result or guarantee. The owner's previous V5 public best is 0.32. Local validation PQ, Dice, and Kaggle public scoring are not interchangeable.

**Authorized window:** 2026-10-24 through 2026-11-06 inclusive, in UTC. A GitHub Actions workflow schedules 03:17 UTC (06:17 Baghdad) each calendar day and triggers immediately when first merged. GitHub's scheduled runs can be delayed or skipped. Outside the window, no GPU jobs run.

After GitHub connection and Kaggle secret KAGGLE are present, no additional ChatGPT conversation is necessary to **attempt** the scheduled jobs. Whether a job really ran is determined by GitHub Actions and the private Kaggle version, not by the existence of this documentation.

## Actual model family and attribution

The current Mask R-CNN + crop-refiner V5 remains preserved. In a separate parallel challenger track, the program starts with Farnoosh Memari's permissively MIT-licensed U-Net repository, pinned to commit ef37d2e93d4cc9e248595e791ee39fe520d392d0. The GPU notebook clones the pinned revision, checks the MIT license and applies a narrow, fail-closed split patch. See third_party/FARNOOSH_AUTO14_NOTICE.md for full credit.

A deterministic month-grouped split reserves about 15% for confirmation and NEVER uses them in the scheduled training, tuning or checkpoint selection. About 18% of the remaining observations are used for model selection and postprocessing tuning. That reuse makes the daily tuning PQ optimistic. No result should be promoted until an independently checked final assessment.

Fourteen predeclared configurations explore U-Net width, 1024/1536 working resolution, 20–36 epochs, class weights, positive crop fraction and seeds. Days 10/12/14 may exploit the best PRIOR local tuning score. Leaderboard results are never used as a hyperparameter oracle. Each day trains a new checkpoint rather than resuming one checkpoint for two weeks.

## Daily execution, outputs, and controls

1. Confirm today's slot is one of the fourteen days and that it has not already been completed, failed or quota-skipped.
2. Query the Kaggle GPU quota and refuse a new launch if fewer than 2.25 GPU hours remain.
3. Build a different PRIVATE notebook per day, id minahussain111/solar-autoresearch-d01 through d14.
4. Run one notebook with Nvidia T4, maximum server execution 7,200 seconds, within a GitHub job limited to 145 minutes.
5. On successful training and tuning, preserve best.pt, last.pt, training loss, split.json, postprocess.json and a source fingerprint in PRIVATE Kaggle version outputs.
6. Recover a small auto14_result.json and commit only summary metrics, execution state, source revision, training configuration and checkpoint hash to reports/AUTO14_STATE.json. Never commit images, annotation polygons, model weights, test predictions, or secrets.
7. A network-uncertain upload is not blindly attempted again. Preexisting Kaggle jobs are read instead. Failed days are recorded.

**No daily competition submission, no test-set inference, and no test labels.** Each local training score is exploratory. A separate one-shot final workflow is scheduled on 2026-11-07 UTC. It chooses the strongest verified candidate by LOCAL tuning only, evaluates that frozen candidate once against unused month-grouped confirmation labels, and if local confirmation PQ >=0.36 creates a full 180-image test CSV and attempts exactly one idempotency-checked official Kaggle submission. An official score exists only if Kaggle actually returns it. If confirmation fails, no CSV or official submission is attempted; V5 remains preserved.

## Monitor independently

- GitHub execution history: https://github.com/ahpa5246-lgtm/Solar-Filament-Segmentation/actions/workflows/auto14-research.yml
- Aggregate ledger: reports/AUTO14_STATE.json, created after first attempt
- Kaggle private experiments: https://www.kaggle.com/code/minahussain111/solar-autoresearch-d01 (replace d01 by d02...d14)
- Existing V5: https://www.kaggle.com/code/minahussain111/solar-filament-research-v5
- Competition: https://www.kaggle.com/competitions/filament-segmentation-2026

The competition organizers list the final report/repository submission deadline as **November 15, 2026, 11:59 pm CDT**. The separate four-page technical report and submission form remain mandatory for final judging and are not submitted by this training system.

## Published methodological baselines worth learning from

Farnoosh Memari reports Dice 0.713, local PQ 0.389 and Kaggle public score 0.32 for a U-Net baseline. HeShen-1/filament-segmentation-2026 reports local five-fold PQ ~0.359 and public ~0.35 for a YOLO detector plus crop refiner. FELIPEACASTRO/solar-filament-2026 reports official public 0.70–0.71 from an attributed Mask R-CNN + D4 crop-refiner mechanism, but their README warns the trained weights are not included. Their own method is MIT but third-party licenses still apply, and their own scorer-derived implementation is deliberately not redistributed because its upstream licensing was unresolved.

This autonomous track does not yet reproduce 0.70. Some model families are poor prospects compared with the established 0.69 public Mask R-CNN path. Follow-up methodological work should compare our real date-grouped validation with an attributed, legally shareable instance detector; do not misrepresent someone else's checkpoint as original training.

## Operational limitations

Kaggle notebooks require account access, available free GPU quota and working Internet for the exact pinned GitHub source; the process can fail. GitHub Actions is a scheduled execution service, not an independent AI researcher. It cannot autonomously invent architecture changes or guarantee numerical improvements. Repeated tuning on fixed validation is not independent statistical confirmation. Protect final blind evaluation from repeated optimization, and distinguish model ownership of new weights from authorship of borrowed architecture.

## Automatic final evaluation and score retrieval

The separate workflow in .github/workflows/auto14-final.yml runs at 05:31 UTC on November 7, 2026 (the date is hard-guarded in Python). It checks the research ledger, selects the best verified checkpoint without looking at leaderboard scores, mounts only that candidate's private Kaggle output plus the official competition inputs, validates months across train/tune/confirmation, measures local one-to-one PQ on the sealed confirmation set, then conditionally generates a checked compressed-COCO-RLE CSV from the 180 official test images. If all checks pass, GitHub Actions makes ONE Kaggle upload and records the real evaluation response (if received) in reports/AUTO14_FINAL.json. The generated CSV stays inside the private Kaggle notebook solar-autoresearch-final-2026 and the public repository stores only hash, counts and aggregate scores. The finalizing Kaggle notebook performs inference only, not additional model training.

Successful automations do not absolve ownership of the final four-page technical report and submission form. Kaggle quotas, API access, GPU availability, GitHub scheduler timing, training quality, and independent confirmation can prevent scoring. A provisional local PQ of 0.36 is a **decision threshold**, not a claim about the score the hidden Kaggle metric will return. No path promises a score of 0.70.
