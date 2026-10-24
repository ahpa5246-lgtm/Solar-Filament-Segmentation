# Validated NMS Submission Implementation Plan

> Execute natively under the owner's standing authorization; review before GPU deployment.

**Goal:** Produce a real, validated competition test CSV and obtain an official result if supported.
**Architecture:** Fix detector mask NMS at IoU 0.5 and preserve 1024 inference, flip4/0.85 refinement and final union. Confirm on the original date-disjoint pool excluding all 16 pilot-confirmation observations, with no threshold search. Only a positive micro gain and positive paired macro lower CI permit test inference. Use final all-training checkpoints, never validation checkpoints. Keep original V5 recoverable.
**Tech Stack:** Existing torchvision/pycocotools components and authenticated GitHub Actions/Kaggle CLI.
**Spec:** User's master task and reports/V5_MEASURED_AUDIT_2026-10-24.md.

- [ ] Extend resolution audit with fixed-NMS confirmation: full tuning dates, confirmation excluding pilot; test disjointness and exclusion before code.
- [ ] Build/deploy private bounded confirmation notebook; inspect actual results.
- [ ] Prepare a separate private immutable final-checkpoint cache with source fingerprints and original output parity material; preserve existing cache.
- [ ] Implement final inference and CSV validation, including exact test-image inventory, unique IDs, full-size nonempty compressed masks and round-trip decoding. Compare reconstructed original final-pipeline outputs against surviving original predictions before accepting candidate outputs.
- [ ] If confirmation and parity gates pass, execute final inference; retrieve validated CSV and submit once using authorized official API. Persist official result separately from local metrics.
- [ ] Report actual outcome, download link, commits and blockers. Do not claim scores before evaluation or submit final report/form.

Budget: each inference session ≤1200 seconds, one used T4, ≥0.75 GPU-hours remaining. No retraining for this change. No hidden labels or repeated leaderboard tuning. A failed gate retains V5 and prevents submission.
