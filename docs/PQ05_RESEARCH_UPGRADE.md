# PQ 0.50 research upgrade — explicit, testable changes

**Status:** implemented in code; any change in Kaggle leaderboard score remains **UNVERIFIED** until an official competition response is recorded. The last owner-confirmed V5 public PQ was 0.32. A target of 0.50 is ambitious, not a result and not a guarantee.

## Hypothesis

Solar filament polygons are thin and elongated. A union-mask U-Net can obtain good semantic Dice while missing thin boundaries and merging physically distinct instances. The revised research arm weights uncertain local boundaries (7 x 7 spatial target range, using *soft multi-annotator labels*) inside the weighted BCE term, retaining the original soft Dice loss. It never intentionally removes filament pixels. Boundary weights 1.5 / 3.0 / 6.0 are evaluated against the original zero-weight control on previously defined dates, using the existing held-out month split.

This does NOT turn a semantic U-Net into an instance-native model. Connected-component merging and splitting remain an important limitation; a gain in actual PQ is not presumed.

## Changes shipped

1. The unchanged day-1 through day-3 experiment plans form an inference-compatible baseline. Days 4-14 apply predeclared bounded boundary weights. A resumed immutable day-one Kaggle notebook remains compatible with its previously recorded configuration.
2. The private notebook clones Farnoosh Memari's MIT-licensed repository at SHA ef37d2e93d4cc9e248595e791ee39fe520d392d0. The revised training, losses, and tuning files are reconstructed from guarded single-occurrence source patches. Any upstream drift aborts; no permissive fallback.
3. The loss code takes its weight only from the planned training subprocess environment. Weight 0 reproduces the upstream weighted BCE + soft Dice mathematically and numerically. Higher values reweight edge pixels but cannot use held-out annotations.
4. Local postprocess tuning samples up to 24 validation images in a reproducible, balanced round-robin across available months instead of only the first 24 chronological images.
5. The private Kaggle notebook writes SHA256 fingerprints of all three generated source files. The runner refuses a completed boundary-arm result without consistent training settings and a valid patch fingerprint.
6. One final Nov 7 blind confirmation is retained, and any Kaggle submission stays blocked until the original safeguards and independent local PQ threshold are met.

## Verification evidence

- Run pytest -q tests/test_auto14.py tests/test_auto14_upgrade.py tests/test_auto14_final.py on a CPU system with PyTorch available. Tests check exact control-loss equivalence, real gradients, greater weight on incorrect filament boundaries, month balancing, source-drift rejection, deterministic plans, notebook compilation and safety gates.
- Compare **local tuning PQ** for completed arms with the frozen baseline. Tuning PQ reuses training checkpoint selection and threshold search, and is optimistic.
- Evaluate exactly one selected candidate on never-touched month-grouped confirmation data; report TP, FP, FN, SQ, RQ, PQ and labeler multiplicity. An official Kaggle score is a distinct quantity.
- Do NOT infer a public score of 0.50 from an improved Dice, local PQ, PR merge, GitHub CI, Kaggle GPU usage, or existence of a CSV.

## Known limitations

The 14-day daily research schedule has finite GPU quota, may have interrupted private Kaggle jobs, and does not guarantee successful training. The first day's immutable notebook may still use the previous source patch. A final official upload also depends on completed checkpoints, quota and external Kaggle service. The day-1 recovery bug from the earlier runner must resolve before a genuine numerical result can be established.

## Attribution

Third-party model and training pipeline: Farnoosh Memari, MIT. Full notices: third_party/FARNOOSH_AUTO14_NOTICE.md. The boundary-focused research patch and tests are project additions, not claims of ownership of upstream model.
