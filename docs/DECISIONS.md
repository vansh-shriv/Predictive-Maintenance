# Decision Log

Every notable choice, idea, and trade-off, newest at the bottom.

## D-001 — Project layout (Phase 0)
- `src/pmaint/` as an installable package (src layout) → clean imports, testable.
- `data/raw` and `data/processed` are git-ignored; reproduced by scripts.
- `docs/phases/` holds one markdown file per phase (what was done, why, how to reproduce).
- `notebooks/` for EDA only; production logic lives in `src/`.

## D-002 — Tooling (Phase 0)
- Python 3.10+ with a virtualenv (`.venv`).
- PyTorch for LSTM/1D-CNN, XGBoost for baseline, MLflow for tracking, FastAPI + uvicorn for serving, Evidently for drift.
- Dependencies are pinned loosely in `requirements.txt`; revisit pinning in Phase 9/10.

## Ideas parking lot
- Piecewise-linear RUL cap (~125 cycles) is standard for C-MAPSS; evaluate in Phase 2.
- Evaluate on FD001 first; extend to FD002–FD004 (multiple operating conditions) later.
- NASA asymmetric scoring function alongside RMSE (late predictions are penalised more).

## D-003 — Data source and loading (Phase 1)
- Source: NASA PHM S3 zip (`phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan...zip`); contains a nested `CMAPSSData.zip`, handled by the downloader. If the URL ever dies, fall back to the NASA open-data portal.
- Column names: `unit, cycle, setting_1..3, s_1..s_21`. Files have trailing whitespace → parsed with `sep=r"\s+"` and empty columns dropped.
- Raw data stays untouched; the uncapped RUL is derived in code (`add_train_rul`), capping deferred to Phase 2.
- Start modelling on FD001; the code is subset-agnostic so FD002–FD004 can be added later.

## Ideas parking lot (additions)
- FD002/FD004 need per-operating-regime normalisation (cluster settings 1–3 into 6 regimes, e.g. KMeans).
- Consider a notebook-free `scripts/` EDA report later if the notebook proves awkward to review in git (outputs are not committed).

## D-004 — Preprocessing design (Phase 2)
- RUL cap 125, window 30, 80/20 engine-wise validation split (seed 42); all configurable in `build_dataset.py`.
- Scaling statistics come from training engines only; the same `Preprocessor` object is persisted and will be reused by the API.
- Metrics (Phase 3+) are computed on the **uncapped** test RUL at each test engine's last window.
- Windows are materialised in memory (FD001 ≈ 16.8k×30×15 float32 ≈ 30 MB) — fine for FD001–FD004; revisit with a lazy Dataset if memory grows.

## Ideas parking lot (additions)
- Tune window size (20/30/50) and RUL cap (100/125/150) as MLflow experiments in Phase 5.
- Exponential smoothing / rolling stats as extra features for XGBoost (Phase 3).
- Environment gotcha: use `.venv` explicitly; the shell default `python` is a different conda env.

## D-005 — Baseline & tracking design (Phase 3)
- Baseline features: last / mean / std / slope per sensor over the 30-cycle window. Simple, interpretable, no sequence model needed.
- Primary comparison metric: **test RMSE** (official last-window protocol), NASA score as secondary. Val metrics only for early stopping.
- MLflow backend: local SQLite (`mlflow.db`) rather than the deprecated-style `./mlruns` file store; a server URI can be swapped in via env var in Phase 8 (docker-compose).
- Baseline FD001 reference: **RMSE 13.58 / NASA 267.6**.

## Ideas parking lot (additions)
- Investigate the FD002 val→test gap (short test histories, regime clustering quality).
- Multiple seeds + confidence intervals before claiming deep model > XGBoost.
- Hyperparameter search (Optuna) as a Phase 5 option, logged as nested MLflow runs.

## D-006 — Deep model design & honest comparison (Phase 4)
- Networks emit cycles (`raw * cap`); loss is computed on RUL/cap for stable gradients.
- Early stopping on validation RMSE with best-weight restore; 3 seeds per model to separate signal from noise.
- Result: XGBoost (13.58) < LSTM (15.54 ± 0.10) < CNN (19.46 ± 0.40) in test RMSE on FD001. We report this as-is; the deep models were untuned.
- The registry candidate must be chosen by measured test/validation results, not by which model is "more modern". If tuning does not close the gap, XGBoost is the champion and the deep models are documented as a negative result.

## Ideas parking lot (additions)
- Tuning candidates: window 50, hidden 128, lr 3e-4, Huber loss, no BatchNorm (use LayerNorm/dropout) for CNN, cap 100/130.
- Ensembling XGBoost + LSTM (average) — often beats either alone.
- Feed hand-crafted features (slope etc.) to the LSTM as extra channels.
- Delete the failed MLflow run from the first (crashed) LSTM attempt for a clean UI.

## D-007 — Model selection protocol & champion (Phase 5)
- Selection uses mean validation val_rmse_deg across 3 seeds among runs with the same RUL cap; test metrics are reporting-only. The rule was fixed before the multi-seed finals were run.
- Champion: XGBoost, window 50, cap 125 (registry rul-champion v1, alias champion). Test RMSE 13.87 +/- 0.12 (w30 would have been 13.62 +/- 0.17, within noise, not switched).
- Registered artifact is a single model run (seed 0 of the winning config) plus its preprocessing/preprocessor.joblib artifact. Phase 6 must load both from the same run.
- The serving window size is therefore 50, not 30: the API must front-pad short histories to 50 like training.
- Deep models and the ensemble are documented as a negative result on FD001.

## Ideas parking lot (additions)
- Revisit the val/test mismatch: a validation protocol that mimics test (random truncation point per val engine, repeated) would give a less noisy selector.
- Optuna search for XGBoost / LSTM; evaluate FD003 (2 fault modes) and FD002/4 (6 regimes).
- Quantile or interval predictions for maintenance decisions (late predictions are costlier, see the NASA score).

## D-008 — Serving design (Phase 6)
- API input is raw telemetry (24 columns), not engineered features: scaling, windowing and feature extraction live server-side so clients cannot diverge from training.
- The window size is read from the registered run's params, never hard-coded; the preprocessor comes from the same run's artifacts, so model, scaler and window are always a consistent set.
- Stateless API: caller sends history per request; short histories are front-padded and flagged with padded=true.
- The model-load failure keeps the process alive and /health returns 503 with the reason.
- Prediction log (JSONL, opt-in via PMAINT_PRED_LOG) stores the last reading + prediction per request as the data source for Phase 7.
- Parity test tolerance is 0.5 cycles because tree splits amplify 1e-7 float noise.

## Ideas parking lot (additions)
- Enforce or warn on a minimum history (e.g. < 30 cycles); a 5-cycle history for a nearly failed engine returned 110 vs true 20.
- Per-engine server-side buffers so clients can post one reading per cycle.
- Batch endpoint, API key auth, request-size limits, Prometheus metrics.
- Hot-reload when the champion alias changes.

## D-009 — Drift monitoring design (Phase 7)
- Monitor the last raw reading per request scaled with the training Preprocessor, plus predicted_rul. Reference = 60 FD001 engines x 3 random cuts; current traffic is judged against it.
- Drift test = normalised Wasserstein distance (effect size), thresholds 0.6 for sensors and 0.2 for predictions. The default K-S p-value test was rejected after it gave 3 false alarms in 20 normal runs: rows from the same engine are not independent and 15 parallel tests inflate false hits.
- Alert levels: critical >= 30% of sensors drifted; warning >= 1 sensor or prediction drift; insufficient_data < 50 rows. CLI exit codes 0/1/2.
- Simulated "normal" traffic must come from the same population as the reference (disjoint engines, uniform cuts). Using the early-truncated test engines produced a legitimate but misleading drift alarm.
- Measured: 2/40 false warnings on fresh normal seeds (5%), 10/10 sensor-bias detections, 3/3 aged-fleet critical. Thresholds are calibrated on simulation only and must be recalibrated on real traffic.

## Ideas parking lot (additions)
- Delayed-label accuracy monitoring: when an engine fails, compare past predictions with the true RUL (rolling RMSE) and trigger retraining.
- Drift on window-level features (slope, rolling mean) and multivariate/correlation drift.
- Schedule the monitor (cron / GitHub Actions / compose job), keep report history, send alerts.
- Retraining trigger: critical drift or rolling RMSE above a limit -> rerun sweep + registry (Phase 10 pipeline idea).
