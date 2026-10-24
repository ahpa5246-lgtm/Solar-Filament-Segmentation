# Orientation-aware non-painting instance recovery for PQ

Hypothesis: the same solar filament can be predicted as two nearby elongated components separated by weak image contrast. A connected-component postprocessor treats them as independent predictions, increasing PQ's fragmentation penalty. Expanding all masks using dilation can join parallel but distinct filaments, creating false merges.

## Candidate

A deterministic endpoint-matching algorithm uses the longest PCA axis of sufficiently elongated labelled components. It estimates each component's two outer tips, then joins endpoints only if both outward directions align with the connecting vector (cosine >= 0.90) and the gap is within 16, 32 or 48 full-resolution pixels. The connected components inherit one instance ID and retain **exactly the same set of positive pixels** as before. The operation never fills the gap or reads ground-truth/test information.

False positives are controlled with a minimum component area of 80, minimum PCA elongation 2.5, and a one-partner-per-endpoint rule. The family is intentionally more restrictive than generic morphological dilation.

## Guarded tuning

In selected research days 5, 7, 9, 11 and 13, after the baseline threshold/min-area/fragment-merge grid has already been frozen, a second small ablation evaluates four geometrical candidates: zero (baseline), 16, 32 and 48 pixel maximum tip gap.

All four candidates are evaluated on the SAME 24 month-balanced development-only validation images, never on the reserved confirmation months. A new setting is accepted only when its local pooled PQ exceeds its own same-image baseline by at least 0.015 AND increases PQ in at least half of sampled months. Otherwise the output config retains link_gap=0, which exactly preserves original upstream behavior.

The accepted setting and complete ablation summary live in the checkpoint's private postprocess.json; source code and source fingerprints stay in private Kaggle outputs. The Nov 7 single final notebook reconstitutes exactly the same postprocessing implementation before independent confirmation and, conditionally, official test inference.

This is a **conditional experiment, not a demonstrated leaderboard increase**. Performance depends on real annotation geometry, GPU/CPU quota, successful checkpoint completion, and the official scorer. Local tuning PQ remains optimistic.

## Unit checks

Tests verify that aligned collinear gaps can be joined, nearby parallel and perpendicular filaments remain separate, round noise does not merge, all positive pixels are conserved, outputs are deterministic, and both daily and final Kaggle notebooks embed the same source patch. The unit tests do not replace a real competition holdout or official score.

Third-party upstream source: Farnoosh Memari (MIT, pinned commit ef37d2e93d4cc9e248595e791ee39fe520d392d0). Local geometry algorithm and ablation are project additions.
