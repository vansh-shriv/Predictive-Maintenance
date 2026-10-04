# Phase 4 — Deep models (LSTM and 1D-CNN)

## Objective
Train sequence models on the same windows, with the same metrics and MLflow experiment as the XGBoost baseline, and compare fairly.

## What was done
- `models/nets.py` — two PyTorch regressors (input `(B, 30, F)` → RUL in cycles):
  - `LSTMRegressor`: 2-layer LSTM (hidden 64, dropout 0.2) → last time step → MLP head.
  - `CNN1DRegressor`: 3× Conv1d(k=5, 64 ch) + BatchNorm + ReLU → mean+max pooling over time → MLP head.
  - Both output `raw * RUL_cap`, so the network works in normalised units and predictions come out in cycles.
- `training/train_deep.py` — Adam (lr 1e-3, wd 1e-5), MSE on RUL/cap, grad-clip 1.0, ReduceLROnPlateau, early stopping (patience 10) on validation RMSE, best-weights restore. `--seeds N` runs seeds 0..N-1, each as its own MLflow run, and prints mean ± std. Logs per-epoch curves, final metrics, the model (with input example) and the preprocessor.
- `tests/test_nets.py` — forward-shape and gradient-flow tests (11 tests pass in total).

## Results — FD001, official test protocol (3 seeds each)
| Model | Test RMSE | Test NASA score |
|---|---|---|
| **XGBoost baseline** (1 seed, Phase 3) | **13.58** | **267.6** |
| LSTM | 15.54 ± 0.10 | 437.9 ± 37.1 |
| 1D-CNN | 19.46 ± 0.40 | 951.3 ± 151.9 |

**Conclusion: with the current setup the XGBoost baseline beats both deep models.** The seed-to-seed spread (≈0.1–0.4 RMSE) is far smaller than the gaps (≈2 and ≈6 RMSE), so this is not noise. Validation tells the same story (XGB 12.65, LSTM ≈13.4, CNN ≈17.9).

## Interpretation (hypotheses, not verified)
- The hand-built features (window mean/slope) with a 30-cycle window are already very informative on FD001; small data (80 training engines) favours trees.
- The CNN stopped early (14–18 epochs, best around epoch 4–8) and has the weakest validation score; BatchNorm with a small, highly correlated training set and the noisy plateau-based scheduler are plausible culprits. Not investigated yet.
- Literature reports LSTM/CNN reaching RMSE ≈ 12–14 on FD001, typically with longer windows, tuning, and sometimes larger models — so there is likely headroom.

## What was NOT done
- No hyperparameter tuning (window size, hidden size, lr, cap, loss). Models are first-pass defaults, so "XGBoost wins" applies to these settings only.
- Only FD001 was trained for deep models (CPU only: ~1–2 min per run).
- XGBoost has a single seed; its variance is untested.

## Issues hit
- `mlflow.pytorch.log_model` (MLflow 3.x, default `pt2` format) requires an `input_example`; added `input_example=Xte[:2]`. The first LSTM run crashed at logging time because of this and was re-run.
- The runs of that crashed attempt may remain in `mlflow.db` as FAILED/partial — harmless; can be deleted in the UI.

## How to reproduce
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.data.build_dataset --subset FD001
python -m pmaint.training.train_deep --model lstm --subset FD001 --seeds 3
python -m pmaint.training.train_deep --model cnn  --subset FD001 --seeds 3
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(models): LSTM and 1D-CNN RUL regressors with MLflow-tracked training (phase 4)"
git push
```

## Next
Phase 5 — Experiment tracking & registry. Proposed scope: small, logged hyperparameter sweeps (window, cap, hidden size) to see whether deep models can close the gap, XGBoost multi-seed, then register the best model in the MLflow Model Registry with a `champion` alias for the API. Decision point for you: see the question below.
