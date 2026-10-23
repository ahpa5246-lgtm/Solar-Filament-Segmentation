# GitHub → Kaggle sync (set up once)

This repository includes [sync-kaggle.yml](../.github/workflows/sync-kaggle.yml). It syncs both research notebooks to **new** private Kaggle notebook IDs, never overwrites the successful v2 notebook, and uses `--no-run` to avoid spending GPU quota on every GitHub commit.

## One-time Kaggle/GitHub setup

1. On Kaggle, open **Settings → API** and create a legacy API token/key if offered. Do **not** paste the key into chat, an issue, a notebook, or a tracked file. Keep the token confidential.
2. On GitHub, open **Solar-Filament-Segmentation → Settings → Secrets and variables → Actions → New repository secret**.
3. Add `KAGGLE_USERNAME` with the value `minahussain111`.
4. Add `KAGGLE_KEY` with your Kaggle API key (secret). Use only your own account. Kaggle may instead offer a new token format; verify its compatibility with the Kaggle CLI's `KAGGLE_KEY` authentication.
5. Open **Actions → Sync research notebooks to Kaggle (no GPU run) → Run workflow → both**.

Once set up, GitHub automatically syncs V3 and V4 when either notebook changes on `main`. Updates to `solar/*.py` do **not** trigger sync because the notebook already clones the latest GitHub source when executed.

Kaggle destinations:
- V3: https://www.kaggle.com/code/minahussain111/solar-filament-research-v3
- V4: https://www.kaggle.com/code/minahussain111/solar-filament-research-v4

The workflow attaches competition source `filament-segmentation-2026`. V3 also attaches the **previous v2 notebook output** `minahussain111/notebookbde8e3d22c`, which must contain `solar_artifacts/best.pt`. Verify the notebook Input pane shows these sources after sync.

## Safety and limitations

- The workflow **saves** the notebook; it does not execute training or submit to the competition. Manually check **Internet On**, **GPU T4**, and Inputs, then use **Save Version → Save & Run All** in Kaggle.
- Some Kaggle users report API-uploaded notebooks may have Internet disabled despite metadata. If Kaggle prevents Internet access for the API-synced version, enable it in the Kaggle UI before running; if unavailable, use the manually imported notebook or a self-contained offline bundle.
- Kaggle CLI/API availability and the `--no-run` option can change. Check the GitHub Actions run logs for errors; a GitHub workflow file alone does not establish a working authenticated integration.
- The GitHub connector in this chat can edit code, but it is **not** an authenticated Kaggle connector. Kaggle execution, scoring, and secrets require user-controlled Kaggle/GitHub permissions.
- Secrets are referenced via GitHub's secret store; they are never committed to the repository. Rotate a compromised key promptly.
- Do not enable automatic GPU runs on every push: a source edit can accidentally start many expensive sessions.
