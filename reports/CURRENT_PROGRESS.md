# Current implementation and verification status — October 24, 2026

GitHub access has recovered: the prepared commits were pushed and authenticated Kaggle execution through Actions succeeded. No user reconnection action is currently needed. V5 remains the owner-confirmed official Public best at **0.32**. No new official submission or competition-model training run is claimed.

## Reproducible software evidence

- Fresh local suite: **51 passed, 1 CUDA-only test skipped**. The workspace has no CUDA.
- `pip check`: no broken requirements.
- A wheel was built and installed outside the checkout. Isolated Python verified the imported package came from that installation.
- Model predictions before and after checkpoint serialization were bitwise identical.
- A synthetic CPU detector step produced five finite losses, nonzero gradients and an actual classifier-weight update. This checks software, not solar accuracy.
- Native 2048×2048 masks survived CSV/compressed-COCO-RLE round trips exactly. Missing observation inventory was rejected.
- All five default V6 notebook code cells executed in a fresh process/directory, using its embedded package. Optional competition training was not exercised.
- GitHub Actions independently passed unit tests and these delivery checks on Python 3.11 and uploaded the `cpu-verification-evidence` artifact: https://github.com/ahpa5246-lgtm/Solar-Filament-Segmentation/actions/runs/40060680129.

Machine-readable source hashes, package versions and local integration results: `CPU_DELIVERY_VERIFICATION.json`. Repeatable commands and reconnect instructions: `docs/VERIFICATION.md`. Implementation commits: `11f13a8` and `b6b7208`.

## Real failure found and fixed

The first resolution launch refused GPU execution because a status-only CPU inspection had overwritten completed replay evidence and removed its artifact hashes. A regression test reproduced this loss before the fix and passed afterward. Inspection now writes a separate status file, including on failure. The successful CPU v2 report was restored byte-for-byte from commit `a47e0f7`; it was not reconstructed from invented metrics. The retry completed successfully. The failed launch remains in Git/Actions history.

## Completed model experiments

All values below are LOCAL validation, distinct from official Public PQ 0.32.

| Experiment | Observation scope | Result | Decision |
|---|---|---|---|
| Cached detector versus plain refiner | 142 original validation observations | Micro PQ 0.362862 → 0.404278; paired macro gain CI [0.027194, 0.046946] | Preserve refiner |
| Minimum native-mask area | 65 tuning / 67 confirmation | Selected zero; confirmation micro PQ unchanged at 0.406185 | No filtering promoted |
| Flip4/threshold/overlap ablation | 65 tuning / 67 confirmation | Confirmation micro PQ 0.407370 → 0.420812; paired macro gain CI [-0.016739, 0.013660] | Failed predeclared gate; original threshold retained |
| Original detector resolution/overlap pilot | 16 tuning / 16 confirmation | Selected 1024/mask NMS; confirmation micro PQ 0.408629 → 0.424046; paired macro gain CI [0.004045, 0.060616] | Positive small pilot; needs broader confirmation |

The resolution pilot reconstructed the original detector masks with minimum parity **1.0** across all 32 observations. It used original source/checkpoint fingerprints, no training or test inference, one T4 of two visible devices, 120.764 seconds for inference/reconstruction, and peak CUDA allocation 4,073,027,584 bytes (~3.79 GiB). Native 2048-resolution tuning PQ was only 0.266869 with union and 0.258865 with NMS, versus 0.421627/0.436670 at 1024. Increased inference resolution is therefore not adopted. The pilot uses historically reused validation data and only 16 confirmation observations; it does not establish final generalization or a new official record.

The earlier full refiner replay used 199.526 inference/reconstruction seconds and peak CUDA allocation 548,147,200 bytes. These measurements exclude session startup/evaluation and are not total billed GPU time.

## Private Kaggle notebooks and evidence

- Main V6 readiness notebook: https://www.kaggle.com/code/minahussain111/solar-filament-research-v6.
- CPU replay v2 completed: https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-audit.
- Refiner replay v1 completed: https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-refiner.
- Resolution pilot v1 completed: https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-resolution.

Aggregate results are in `V6_KAGGLE_CPU_EXECUTION.json`, `V6_KAGGLE_REFINER_EXECUTION.json`, and `V6_KAGGLE_RESOLUTION_EXECUTION.json`. Private outputs retain qualitative GT/prediction panels. Raw training labels, images, weights and credentials are not published in Git.

## Remaining research and delivery work

Broader validation of detector NMS, inspection of private qualitative outputs, an alternative-instance-architecture training comparison, fresh date-grouped independent final confirmation, final-model test inference, official evaluation of any new submission, and the organizer-template four-page final report remain outstanding. The recovered training session's association to the owner-confirmed scored V5 submission is not independently established. No alternative model or score above 0.32 is claimed. The report deadline timezone/template still require official verification; no final report/form was submitted.
