# Phase 2 — Preprocessing & feature engineering

## Objective
Turn raw C-MAPSS tables into model-ready, leakage-free arrays: capped RUL targets, scaled informative sensors, sliding windows, and an engine-wise train/validation split.

## What was done
- `features/rul.py` — `cap_rul()`, piecewise-linear target (default cap **125** cycles).
- `features/preprocessing.py` — `Preprocessor`: drops constant sensors (std ≤ 1e-6), standardises each sensor **within its operating regime**. `n_regimes=1` for FD001/FD003; `n_regimes=6` (KMeans on `setting_1..3`) for FD002/FD004. Saved with joblib so the API (Phase 6) applies the identical transform.
- `features/windows.py` — `split_units()` (split by engine id), `make_train_windows()` (one window per cycle, front-padded with the first row), `make_last_windows()` (final window per test engine, the official protocol).
- `data/build_dataset.py` — CLI that wires it together and writes `data/processed/<subset>/dataset.npz` + `preprocessor.joblib`.
- `tests/test_features.py` — 4 unit tests on a synthetic frame.

## Design choices (and why)
| Choice | Reason |
|---|---|
| RUL cap = 125 | Early life shows no measurable degradation; an uncapped target makes the model chase noise. Literature standard for FD001. Configurable via `--cap`. |
| Validation targets are also capped; **test targets stay uncapped** | Test RUL is the real label we are scored against. |
| Scaler fit on training engines only | Prevents val/test statistics leaking into training. |
| Split by engine (80/20, seed 42) | Windows from one engine are highly correlated; row-wise splits would inflate scores. |
| Window = 30, front-padding | Shortest test engine history: 31 cycles (FD001), 21 (FD002), 38 (FD003), 19 (FD004) — so FD002/FD004 need padding. Window size is a hyperparameter to tune later. |
| Standardise per regime | In FD002/FD004 the operating condition dominates sensor variance and hides degradation. |
| Windows built after scaling | Cheap and equivalent; no cross-engine mixing since windows are built per engine. |

## Results (verified by running)
| Subset | X_train | X_val | X_test | Features |
|---|---|---|---|---|
| FD001 | (16779, 30, 15) | (3852, 30, 15) | (100, 30, 15) | 15 (s_1,5,10,16,18,19 dropped) |
| FD002 | (43083, 30, 21) | (10676, 30, 21) | (259, 30, 21) | 21 |

- FD001 train targets max = 125 (cap applied); test targets 7–145 (uncapped, as expected).
- Scaled data mean ≈ -0.05, std ≈ 0.9 on train/val; test mean ≈ -0.02 (no obvious shift).
- `pytest`: 6 passed.

## Issues hit along the way
- `pip install -e .` had produced a **non-editable copy** in `.venv` (stale `build/` dir), so new modules were not importable. Fixed with `pip uninstall pmaint && pip install -e .`; `build/` and `dist/` added to `.gitignore`.
- pandas returned a read-only array from `to_numpy()`; fixed with `copy=True`.

## How to reproduce
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.data.build_dataset --subset FD001
python -m pmaint.data.build_dataset --subset FD002   # optional
pytest
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(features): RUL capping, per-regime scaling, windowing and dataset builder (phase 2)"
git push
```

## Next
Phase 3 — XGBoost baseline: flatten/aggregate windows into tabular features, train, evaluate with RMSE + NASA score, first MLflow runs.
