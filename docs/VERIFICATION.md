# Reconnecting GitHub and reproducing verification

## Connection recovery

On October 24, 2026, Git fetch and push succeeded again in this environment. Reconnecting is currently unnecessary. If authentication fails again:

1. Open the GitHub connection in ChatGPT/Codex settings and reconnect it. Depending on the client, this appears under Apps, Connected apps, or GitHub connections.
2. On GitHub, open **Settings → Applications → Installed GitHub Apps**, select the OpenAI/Codex app, and choose **Configure**. Confirm repository access includes `ahpa5246-lgtm/Solar-Filament-Segmentation`. Complete organization authorization if GitHub requests it.
3. Return to the cloud task and refresh/reopen the environment. Verify repository access with `git fetch origin main`, then retry the prepared push. Share only the error message if it fails, never a token.

GitHub connection authentication is separate from Kaggle authentication. Authenticated Kaggle operations continue to use the existing repository secret `KAGGLE` through GitHub Actions; no raw credential is needed in chat.

## Repeat the CPU checks

In the existing cloud workspace:

```bash
cd /workspace/Solar-Filament-Segmentation
OMP_NUM_THREADS=2 /workspace/solar-venv/bin/python -m pytest -ra --junitxml=/tmp/solar-tests.xml
/workspace/solar-venv/bin/python -m pip check
OMP_NUM_THREADS=2 /workspace/solar-venv/bin/python scripts/verify_repository.py --out /tmp/solar-delivery.json
```

The last command stages the Python package outside the checkout, builds a wheel, installs it into a temporary target, and runs a separate isolated Python process. It verifies imports actually come from the delivered wheel, checks bitwise identical predictions before/after checkpoint loading, reconstructs native 2048×2048 instance masks, exports compressed COCO CSV, decodes every row, and rejects missing image inventory. It also performs one CPU forward/backward/optimizer step on a synthetic filament, requires all five detector losses to be finite, and verifies the classifier weights actually change. The model in this software check has random weights; its outputs and losses are not a quality benchmark or a competition training run.

It also builds the default V6 notebook and executes every code cell in a fresh directory and process, confirming imports use the embedded package. Default execution requests no training or GPU. This does not exercise the optional training branch.

GitHub Actions repeats unit tests and delivery checks and uploads the JUnit XML and machine-readable delivery report under the `cpu-verification-evidence` artifact. That evidence is separate from solar validation, GPU execution, and official scores.

## An actual workflow failure caught and corrected

The first native-resolution launch refused to run before allocating GPU because the completed CPU replay evidence no longer contained artifact hashes. Investigation found that `--inspect-only` overwrote the execution report with a status-only report. A regression reproduced the data loss before the fix. Inspection now writes a separate status file, including on error; the workflow persists that separate file. The completed v2 report was restored byte-for-byte from commit `a47e0f7`, retaining its original fingerprints and measured results. No metrics were regenerated or invented during restoration.

## Limits of the evidence

Passing tests demonstrates specific software behavior, not perfection. The local workspace has no CUDA, so its dedicated CUDA smoke test skips. Real checkpoint GPU replay is independently recorded in `reports/V6_KAGGLE_REFINER_EXECUTION.json`. It completed on T4 hardware, exactly reproduced the plain cached metric, and evaluated all 142 validation observations. Its selected threshold failed the predeclared promotion gate because the paired macro-observation confidence interval includes zero. V5 remains the owner-confirmed official Public best at 0.32. The native-resolution pilot also completed on 32 real observations with exact detector-cache parity. It selected 1024/mask NMS: confirmation local micro PQ improved from 0.408629 to 0.424046 on 16 observations, with paired macro gain CI [0.004045, 0.060616]. This small, historically reused holdout is not an official-score result. Final-model training, alternative architecture comparisons and any new official submission remain separate work.

Latest independent CI, including the synthetic CPU optimizer update: https://github.com/ahpa5246-lgtm/Solar-Filament-Segmentation/actions/runs/40060680129. Its downloadable artifact is `cpu-verification-evidence`.
