# Phase 5 — Experiment tracking, tuning & model registry

## Objective
Find out whether the deep models can close the gap to XGBoost, quantify seed noise, try an ensemble, and register a champion model in the MLflow Model Registry using a selection rule that does not peek at the test set.

## What was done
- **Config plumbing:** `paths.dataset_dir()` + `build_dataset.ensure_dataset()` — any `(window, cap)` variant is built on demand into `data/processed/FD001_w<W>_c<C>/`. `train_xgb.py` and `train_deep.py` were refactored into callable `run(...)` functions (return metrics, predictions, run id) with `--window --cap --seeds --stage` flags; deep training also gained `--loss huber`, `--hidden`, `--no-bn`.
- `training/sweep.py` — one-factor-at-a-time sweep around the defaults (13 runs, seed 0, `stage=sweep`).
- New validation metric **`val_rmse_deg`**: RMSE only on windows with capped RUL < cap (why: see below).
- Multi-seed **final runs** (`stage=final`, 3 seeds): XGBoost w30, XGBoost w50, LSTM w50.
- `training/ensemble.py` — averages XGBoost + LSTM predictions (evaluation only, `stage=ensemble`).
- `training/registry.py` — groups runs by identical params (seed excluded), ranks configs by **mean `val_rmse_deg`**, registers the lowest-seed run of the winner as `rul-champion` with alias `champion`; test metrics are attached as tags for reporting only.

## Why a new validation metric
Plain `val_rmse` averages over every window of the validation engines, which is dominated by the flat early-life plateau (RUL = cap). The test protocol is one window per engine at an arbitrary cut point. Window 50 vs 30 showed val RMSE 9.8 vs 12.7 but nearly identical test RMSE — a sign val was rewarding something the test does not. `val_rmse_deg` (degradation phase only) is a closer proxy, though still imperfect (see Caveats).

## Results — FD001 (test = official last-window protocol)

Single-seed sweep (seed 0; `val_deg` = val_rmse_deg; **val is only comparable at cap 125**):
| Model | Config | val_deg | test RMSE |
|---|---|---|---|
| XGB | w30 (default) | 13.82 | 13.52 |
| XGB | w20 | 16.53 | 16.75 |
| XGB | w50 | 10.71 | 13.80 |
| XGB | cap 100 (w30) | 9.75 (not comparable) | 17.30 |
| XGB | cap 150 (w30) | 17.59 (not comparable) | 16.75 |
| LSTM | default | 15.01 | 15.64 |
| LSTM | w50 | 13.54 | 16.42 |
| LSTM | hidden 128 | 13.59 | 15.24 |
| LSTM | Huber loss | 13.97 | 16.01 |
| LSTM | lr 3e-4 | 13.98 | 15.68 |
| CNN | default (BN) | 17.66 | 19.06 |
| CNN | no BatchNorm | 17.34 | 18.15 |
| CNN | no BN, w50 | 17.63 | 17.27 |

Multi-seed finals (3 seeds, mean ± std):
| Model | mean val_deg | **test RMSE** | test NASA |
|---|---|---|---|
| XGBoost w30 | 13.76 | **13.62 ± 0.17** | **270.4 ± 6.7** |
| XGBoost w50 | **10.72** | 13.87 ± 0.12 | 311.5 ± 4.7 |
| LSTM w50 | 12.52 | 16.05 ± 0.38 | 396.6 ± 48.4 |
| Ensemble (XGB w50 + LSTM w50, 50/50) | 10.62 | 14.20 ± 0.31 | 312.6 ± 27.5 |
| (Phase 4) LSTM w30 | – | 15.54 ± 0.10 | 437.9 ± 37.1 |
| (Phase 4) CNN w30 | – | 19.46 ± 0.40 | 951.3 ± 151.9 |

## Conclusions
1. **XGBoost remains the best model family on FD001.** Tuning moved the LSTM only within ~15.2–16.4 test RMSE; the CNN stayed ≥ 17. The deep-vs-tree gap (≈ 2 RMSE) is far larger than seed noise.
2. **RUL cap 125 is the best of {100, 125, 150}** on test RMSE for XGBoost (13.5 vs 17.3 / 16.8). Window 20 is clearly worse; 30 vs 50 is a tie on test.
3. **Ensembling did not help**: 14.20 vs 13.87 (XGB w50 alone) — the weaker LSTM drags the average down. Not registered.
4. **Champion = XGBoost, window 50** (registry v1, alias `champion`), chosen by the pre-declared rule (lowest mean `val_rmse_deg`). Loading it back from the registry reproduces its logged test RMSE (13.80 for the registered seed-0 run).

## Caveats
- **Val and test disagree on w30 vs w50.** Val strongly prefers w50 (10.7 vs 13.8) but test slightly prefers w30 (13.62 vs 13.87). The 0.25 gap is about 1.5× the seed std and the test set has only 100 engines, so it is probably noise — but it means the champion is *not provably better than w30* on test. We keep the rule rather than switching after seeing test results, which would be test-set selection.
- The sweep is one-factor-at-a-time on a single seed; interactions are unexplored. Deep models could still improve with a proper search (Optuna) — not done.
- Only FD001. FD002–FD004 not tuned (the FD002 baseline had a big val→test gap, still open).
- Phase 3/4 runs (before `val_rmse_deg` existed) are excluded from selection because they lack the metric.
- The 4 early XGB sweep runs (made before `val_rmse_deg` was added) were deleted from MLflow and re-run.

## Issues hit
- A scripted string-replace silently failed on part of `registry.py`; the file was rewritten cleanly. A group-by key built with `DataFrame.agg("|".join)` crashed on mixed param dtypes; replaced with an explicit row loop.

## How to reproduce
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.training.sweep
python -m pmaint.training.train_xgb --seeds 3 --window 30 --stage final
python -m pmaint.training.train_xgb --seeds 3 --window 50 --stage final
python -m pmaint.training.train_deep --model lstm --seeds 3 --window 50 --stage final
python -m pmaint.training.ensemble --seeds 3 --xgb-window 50 --lstm-window 50
python -m pmaint.training.registry --dry-run     # inspect ranking
python -m pmaint.training.registry               # register champion
mlflow ui --backend-store-uri sqlite:///mlflow.db  # Models tab -> rul-champion
```
Total compute: ~40 min on CPU. `mlflow.db` is local and git-ignored, so the registered model exists only on this machine until Phase 8 (shared MLflow server in docker-compose).

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(mlops): config sweep, multi-seed finals, ensemble check and MLflow champion registry (phase 5)"
git push
```

## Next
Phase 6 — FastAPI serving: `/health`, `/predict` (accepts a window of raw sensor readings, applies the saved `Preprocessor` + window tabularisation, returns RUL), loading `models:/rul-champion@champion`.
