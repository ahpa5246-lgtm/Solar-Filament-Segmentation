# Fourteen-day challenger: third-party rights and attribution

The automatic GPU experiments clone a pinned public source:
- Farnoosh Memari, [solar-filament-segmentation](https://github.com/FarnooshMemari/solar-filament-segmentation).
- Exact commit: ef37d2e93d4cc9e248595e791ee39fe520d392d0.
- License: MIT; copyright (c) 2026 Farnoosh Memari. The original LICENSE remains in the cloned source and must travel with copies or substantial portions.
- Adaptation: change the seed-dependent validation split into stable, month-disjoint training/tuning/confirmation partitions; run specified training hyperparameter variants. This is an adapted baseline, not independently invented architecture.

The evaluation and non-promotion policy is original to this repository. The original MIT license covers the upstream source only. Additional research references (not copied into this implementation): HeShen-1/filament-segmentation-2026, ameyypawar/filament-seg, FELIPEACASTRO/solar-filament-2026. Do not redistribute the organizer's unlicensed scorer port or private checkpoints.

The official competition train data is used for training and validation only. No test labels are accessed, copied, or inferred. Kaggle privacy, quota, competition rules, and source access must all be respected. Every trained checkpoint is derived using the user's private Kaggle session, not a claim of originality over open-source architecture.
