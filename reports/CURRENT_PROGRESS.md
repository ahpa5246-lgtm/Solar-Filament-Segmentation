# Current implementation and experiment status

**Update October 24, 2026:** GitHub fetch/push access recovered and commits `9da615a`/`c2d44c5` were deployed. Fresh local verification: 51 tests passed, one CUDA-only test skipped. Built-wheel checkpoint/inference/CSV verification and all five default notebook code cells executed successfully outside the checkout. See `CPU_DELIVERY_VERIFICATION.json` and `docs/VERIFICATION.md`. The first resolution launch safely stopped before GPU use because a status-only job had overwritten CPU replay evidence. That reporting bug was reproduced, fixed, independently reviewed, and the original completed v2 evidence restored unchanged. The deployment notes below describe the earlier block; resolution execution is being retried after that fix.

The project remains in progress. V5 is preserved as the owner-confirmed best official Public PQ **0.32**. No new official submission or competition-model training run was performed in this work; a synthetic CPU optimizer step verifies software behavior only.

## Completed and published

- `42a54c585144ed98bfd525fa2925c7d15baf4b33`: private bounded GPU refiner replay, archived checkpoint reconstruction, reusable replay module and tests.
- `0a5b897`: Apache license text and corrected derivative attribution.
- `2aad17a`: native mask-area/elongation error diagnostics and paired refiner uncertainty.
- `8f9af4b`: measured evidence and rewritten technical-report draft.
- `bcbe937`: Actions-persisted completed GPU experiment evidence.
- Unit CI run https://github.com/ahpa5246-lgtm/Solar-Filament-Segmentation/actions/runs/40038805924 completed successfully for the published revision. Subsequent local changes have not run in remote CI.

Private Kaggle notebooks:

- https://www.kaggle.com/code/minahussain111/solar-filament-research-v6 (code synchronized; default readiness notebook).
- https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-audit (CPU replay v2 completed).
- https://www.kaggle.com/code/minahussain111/solar-filament-research-v6-refiner (GPU replay v1 completed).

## Measured experiments

| Experiment | Local result | Decision |
|---|---|---|
| Cached detector versus plain refiner, 142 observations | Micro PQ 0.362862 → 0.404278; paired macro gain CI [0.027194, 0.046946] | Preserve refiner; not an official-score comparison |
| Minimum native-mask area | Selected zero; confirmation micro PQ unchanged at 0.406185 | No filtering promoted |
| GPU flip4/threshold/overlap ablation | Confirmation micro PQ 0.407370 → 0.420812; paired macro gain CI [-0.016739, 0.013660] | Failed predeclared promotion gate; retain original configuration |

GPU refiner replay used no training, one Tesla T4 of two allocated GPUs, 199.526 seconds for inference/reconstruction and peak allocated CUDA memory 548,147,200 bytes. The runtime used torch 2.11.0+cu128. Plain reconstruction exactly reproduced cached PQ. Private outputs retain predictions and six qualitative panels. These outputs are not publicly committed.

## Prepared locally, not deployed

`9da615ac37899d608fd7df1732232100a1a3769d` implements the native-resolution pilot, reviewed before deployment. It compares exact archived V5 detector weights at 1024/2048 resolution and union/mask NMS, with fixed 16 tuning plus 16 confirmation observations, fingerprint guards and detector-mask parity checks. No optimization is selected from confirmation. The run is inference-only, with a 1,000-second application budget and 1,200-second server timeout. The new private target slug `minahussain111/solar-filament-research-v6-resolution` has **not** been created or executed.

Runnable package: `notebooks/v6-resolution/solar_filament_resolution_pilot.ipynb` and adjacent `kernel-metadata.json`. Builder: `scripts/build_v6_resolution_notebook.py`. Workflow: `.github/workflows/resolution-audit.yml`; pushing this new workflow to main initiates the bounded experiment through the existing KAGGLE secret. Raw credentials are neither needed in chat nor stored in the notebook.

Final local verification after notebook generation: **50 passed, 1 CUDA test skipped**; `pip check` reports no broken requirements; `git diff --check` passed. CPU tests do not establish native-resolution GPU viability or segmentation accuracy.

## Earlier deployment blocker (access now restored)

GitHub smart-HTTP access changed from working to HTTP 401 during the next push. A credential supplied by the runtime was also rejected; no credential values were printed, stored in Git, or requested from the owner. The available runtime has no Kaggle API binding, so a GitHub-authentication failure also blocks launching new authenticated Kaggle operations through Actions. Read-only public GitHub HTML remains accessible.

Restore this environment's GitHub repository connection/authentication. Then rebase the local commits on origin/main and push them; the prepared workflow handles upload and execution. No manual Kaggle notebook download/upload is necessary. Until that succeeds, the resolution experiment remains unexecuted.

## Still outstanding

Native-resolution GPU results, direct inspection of private qualitative panels, an alternative-instance-architecture training comparison, fresh date-grouped independent final confirmation, full-data final-model inference, official evaluation of a new submission, and the organizer-template four-page final report remain outstanding. The model-card endpoint for a screened Mask2Former alternative was inaccessible here; no alternate architecture is claimed evaluated. The report deadline timezone/template still require official verification. No final report/form submission was made or authorized.
