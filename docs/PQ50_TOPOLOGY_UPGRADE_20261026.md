# Solar PQ 0.50 research — connectivity-aware GPU experiment (2026-10-26)

## Evidence boundary

Official baseline: **0.32 Public PQ** on Kaggle, submission 57571678. Completed independent daily GPU trials: day 2 **0.40438 LOCAL tuning PQ**, day 3 **0.38915 LOCAL tuning PQ**. These are not official scores, and the latter is worse on its identical development protocol. The day-2 model remains the best of those two local experiments; no new official score has been verified. The early-final run returned `ERROR`, so a claimed 0.50 or 0.60 would be false.

## Implemented today

The project now patches the pinned MIT U-Net training loss with a **bounded differentiable soft-clDice** term to test whether filament skeleton continuity helps PQ. The proposed metric target is instance-level PQ, not binary Dice. The term operates only on the **training** multi-annotator soft targets, and has no access to competition test annotations or the sealed date-grouped confirmation months.

* Days **1–3** are immutable historical control notebooks with no boundary or topology configuration.
* Days **4–6** isolate the previously planned boundary-weighted BCE experiments.
* Days **7–14** add topology weights **0.12** or **0.25** alongside bounded boundary BCE to compare whether continuity helps; this is experimental and need not improve PQ.
* Each run must record a fingerprint of patched pinned-source files, the exact topology weight, training completion, best-checkpoint SHA256 and development-only PQ.
* Weight zero is an exact algebraic identity with the original BCE+Dice contribution; existing stored U-Net checkpoints and the independent final inference pipeline remain compatible.
* The run ledger preserves the strongest measured completed local checkpoint, and **never promotes a weaker candidate solely because it was trained later**.

## Numerical verification versus performance

A tiny synthetic two-pixel-wide filament test can prove that soft-clDice penalizes a broken center and produces finite nonzero gradients. It cannot prove higher real Kaggle PQ. The repo's `tests/test_auto14_upgrade.py` verifies that numerical contract, notebook assembly, environment wiring, day bounds, baseline compatibility and source-drift checks. GitHub's regular CPU Unit Tests workflow will also execute that test file; consult its actual run outcome rather than inferring success from a committed test.

## GPU and official score

The existing GitHub Actions workflow `.github/workflows/auto14-research.yml` is scheduled during October 24–November 6, 2026 UTC, launches private Kaggle GPU notebooks with quota safeguards, and preserves completed checkpoints. The already completed October 26 day-three notebook cannot be silently overwritten or retrained under its identity. The changed loss applies to **future** notebooks only.

The independent final selection/evaluation is gated to November 7, 2026, and an official submission is attempted only after leakage checks and confirmation PQ eligibility. If Kaggle rejects an experiment, quota expires, or a model fails confirmation, maintain V5 and report the failure. The finite competition deadline is November 15; a perpetual GPU-training guarantee is impossible.

**Success criterion:** a new Kaggle submission shows **Public PQ >= 0.50**, with its exact submission reference and score. Local metrics, green tests or extra epochs do not meet this criterion.
