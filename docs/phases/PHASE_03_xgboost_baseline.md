# Phase 3 — XGBoost baseline

## Objective
Establish a strong, cheap reference model and the evaluation + experiment-tracking plumbing that the LSTM / 1D-CNN (Phase 4) will be compared against.

## What was done
- `training/metrics.py` — `rmse`, `nasa_score` (asymmetric: late predictions penalised more, `exp(d/10)-1` vs `exp(-d/13)-1`, d = pred − true), `evaluate`.
- `features/tabular.py` — `window_to_tabular`: each (30, F) window → 4F features per window: **last value, mean, std, linear slope** per sensor (15 sensors → 60 features on FD001).
- `training/tracking.py` — shared MLflow setup: SQLite backend `mlflow.db` in the repo root (git-ignored), experiment `rul-cmapss`. Override with `MLFLOW_TRACKING_URI`.
- `training/train_xgb.py` — trains `XGBRegressor` with early stopping on the engine-wise validation set, evaluates on val + official test protocol, logs params / metrics / model / preprocessor to MLflow.
- `tests/test_metrics_tabular.py` — metric and feature tests (9 tests pass in total).

## Results (single seed 42, verified by running)
| Subset | val RMSE (capped, all windows) | **test RMSE** | **test NASA score** | best iter |
|---|---|---|---|---|
| FD001 | 12.65 | **13.58** | **267.6** | 205 |
| FD002 | 14.42 | 26.48 | 7745.7 | 310 |

- FD001 is in the range reported in the literature for tree-based models (RMSE ≈ 13–18), so the pipeline looks sane. This is the number the deep model must beat.
- Top FD001 features: `mean_s_4`, `last_s_11`, `last_s_4`, `mean_s_3`, `last_s_9` — the usual HPC-degradation sensors (T50, Ps30, ...), i.e. the model uses physically meaningful signals.

## How to read the metrics
- **Test** = official protocol: one prediction per engine at its last observed cycle, scored against the true *uncapped* RUL. Compare models on this.
- **Val** = every window of held-out engines against *capped* RUL. Used for early stopping / sanity only. `val_nasa_score` is a sum over thousands of windows, so it is **not comparable** to the test score.
- Predictions are clipped at ≥ 0 before scoring.

## Caveats / observations
- FD002: val 14.4 vs test 26.5 — a big generalisation gap. Likely causes: test engines are cut early (shortest history is 21 cycles, so windows are heavily padded), and per-regime scaling with KMeans may be imperfect. Not a bug in FD001 pipeline; worth revisiting after the deep model.
- Single seed and single validation split → differences of ~1 RMSE are within noise. Phase 5 will add repeated runs.
- Hyperparameters are reasonable defaults, **not tuned**. Tuning is deferred so the baseline stays honest and cheap.

## How to reproduce
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.data.build_dataset --subset FD001
python -m pmaint.training.train_xgb --subset FD001
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
pytest
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(training): XGBoost baseline with RMSE/NASA score and MLflow tracking (phase 3)"
git push
```
`mlflow.db` and `mlruns/` are git-ignored.

## Next
Phase 4 — PyTorch LSTM and 1D-CNN on the same windows, same metrics, logged to the same MLflow experiment for a direct comparison.
