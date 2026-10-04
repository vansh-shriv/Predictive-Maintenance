# Phase 9 — Testing & CI

## Objective
Make quality checks automatic and reproducible: a clean lint baseline, a test suite that runs on a bare machine (no dataset, no registry), and a GitHub Actions pipeline that lints, tests and builds/smoke-tests the containers.

## What was done
- **Ruff configuration** (`pyproject.toml`): explicit rule set `E, F, W, I, B, UP` (pyflakes, pycodestyle, isort, bugbear, pyupgrade), line length 100, py310 target. Explicit because ruff 0.16's *default* rules flagged 21 style-only items (e.g. `dict()` calls) and would shift again with future releases. Notebooks are exempt from `E501`/`E702` only.
- **Lint pass:** 32 findings fixed (long lines, import order, ambiguous variable name `l` ×2, `zip(strict=)`). `ruff check .` is clean. Behaviour was unchanged and re-verified by the test suite.
- **pytest markers:** `data` (needs C-MAPSS in `data/raw`) and `registry` (needs a champion in the local `mlflow.db`) on `test_loader.py` and `test_serving_parity.py`. CI runs `pytest -m "not data and not registry"`.
- **New tests**
  - `tests/test_training_smoke.py` — runs `train_xgb.run` and `train_deep.run` (LSTM, CNN) on tiny synthetic arrays against a temporary MLflow store: checks metrics, MLflow params/tags, prediction shapes, and that the logged training loss actually decreases.
  - `tests/test_api.py` — unwritable prediction log must not break `/predict`.
- **Pinned `requirements.txt`** to the exact versions used in development (`torch>=2.14.1` as a floor, CI installs the CPU build first).
- **`.github/workflows/ci.yml`** — two jobs on push to `main` and on PRs:
  1. `lint-and-test`: Python 3.12, pip cache, CPU torch, `pip install -r requirements.txt && pip install -e .`, `ruff check .`, `pytest -m "not data and not registry"`.
  2. `docker`: `docker compose --profile monitor build`, start `mlflow` + `api`, wait for MLflow health, assert the API returns **503 `degraded`** (no champion registered in a fresh stack), then `compose down -v`. Read-only `contents` permission, no secrets, concurrency cancels superseded runs.
- **A real bug found while designing CI:** on a fresh Linux checkout Docker creates the bind-mounted `./logs` as root-owned, so the non-root API user could not append to the prediction log and `/predict` would have returned 500. Logging is now best-effort (`OSError` is caught and warned), with a test.

## Verification
| Check | Result |
|---|---|
| `ruff check .` | All checks passed |
| Full suite on dev machine | 30 passed (includes data + registry tests) |
| CI-mode suite (`-m "not data and not registry"`) in a **clean copy of the repo** (git-tracked + untracked files, no `data/`, no `mlflow.db`, `PYTHONPATH` pointed at the copy) | **27 passed**, 3 deselected |
| `pip install --dry-run -r requirements.txt` | no changes (pins equal the current environment) |
| Workflow YAML | parses; both jobs/steps as intended |

## Limitations — read before trusting the green tick
- **The GitHub Actions workflow has not been executed.** I cannot run GitHub's runners here; what was verified is each ingredient locally (lint, the CI-mode test selection in a clean copy, the compose build + degraded-API smoke test from Phase 8). First-run failures are possible, most likely from: the CPU-torch install on Ubuntu, runner time/disk for the Docker builds (monitor image ≈ 1.5 GB), or a missing `docker compose` flag. Check the first run's logs.
- CI tests do **not** train on real C-MAPSS or check model quality, and do not run the serving-parity test; those need the dataset and a registry. A model-quality gate (e.g. "champion test RMSE ≤ 15 on FD001") belongs in a scheduled/manual workflow with data access — not implemented.
- The Docker job only proves the stack boots and reports "no model" correctly; it does not promote a model and predict.
- No coverage measurement, no type checking (mypy), no dependency/vulnerability scanning, no pre-commit hooks.
- The training smoke tests take ~1 minute (MLflow `pt2` model export dominates).
- Only Linux CI; local development was on Windows. Windows-specific issues found along the way (cp1252 console, clock resolution) are fixed but not guarded by CI.

## How to run locally
```powershell
.venv\Scripts\Activate.ps1
ruff check .
pytest -m "not data and not registry"     # what CI runs
pytest                                     # everything (needs data + registry)
pytest -m data                             # only dataset-dependent tests
```

## Git steps (you run these)
```powershell
git add .
git commit -m "ci: ruff config and lint pass, test markers, training smoke tests, pinned deps, GitHub Actions workflow (phase 9)"
git push
```
After pushing, open the repository's **Actions** tab and check the first run; tell me about any failure and I will fix it.

## Next
Phase 10 — Pipeline automation & final docs: a single reproducible entry point for the whole flow (download → build → train → select → register → promote), a retraining-trigger design tied to the drift monitor, and a final README (architecture diagram, results, how to run) plus project retrospective.
