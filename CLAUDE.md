# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
RUL prediction on NASA C-MAPSS turbofan data: LSTM / 1D-CNN vs XGBoost baseline, served with FastAPI, tracked in MLflow, drift-monitored with Evidently. See `docs/ROADMAP.md` for the phase plan.

## Working agreement (important)
- Build **one phase at a time**. At the end of each phase: write/update `docs/phases/PHASE_XX_*.md`, log decisions/ideas in `docs/DECISIONS.md`, then give the user commit message(s) and **stop until the user confirms they have committed and pushed**. Do not run git commit/push yourself.
- Document every move, step, and idea in the markdown docs as it happens.
- Report negative results honestly (e.g. deep models lost to XGBoost on FD001). Select models on validation metrics only; test metrics are for reporting.

## Environment
Use the project venv (`.venv\Scripts\python.exe`); the shell's default `python` is a different conda env without the dependencies. Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence an MLflow banner.

## Commands
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .     # must be editable; a stale build/ dir breaks it
pytest                                  # all tests
pytest tests/test_api.py::test_predict_ok_and_padding_flag   # single test
ruff check .

python -m pmaint.data.download                        # C-MAPSS -> data/raw
python -m pmaint.data.build_dataset --subset FD001 [--window 50 --cap 125]   # -> data/processed/
python -m pmaint.training.train_xgb --seeds 3 --window 50 --stage final
python -m pmaint.training.train_deep --model lstm --seeds 3
python -m pmaint.training.sweep                       # one-factor sweep, stage=sweep
python -m pmaint.training.registry [--dry-run]        # rank configs by mean val_rmse_deg, register champion
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn pmaint.serving.app:app --port 8000            # serves models:/rul-champion@champion
python -m pmaint.serving.sample_request --unit 24 > payload.json   # then POST to /predict
python -m pmaint.monitoring.simulate --scenario sensor_bias        # normal | sensor_bias | aged_fleet -> logs/sim_*.jsonl
python -m pmaint.monitoring.drift --current logs/sim_sensor_bias.jsonl --html reports/drift.html   # exit 0 ok / 1 warning / 2 critical

# Docker (Phase 8)
docker compose up -d --build                          # mlflow:5000 + api:8000
python -m pmaint.training.promote --src sqlite:///mlflow.db --dst http://localhost:5000   # copy champion into the container registry
python -m pmaint.monitoring.replay --scenario sensor_bias   # send simulated traffic to the API
docker compose --profile monitor run --rm monitor     # drift job; exit 0/1/2
docker compose down                                   # keeps the registry volume (-v deletes it)
```

## Architecture
src layout, package `pmaint` in `src/pmaint/`: data -> features -> models -> training -> serving / monitoring. Notebooks are for EDA only.

Things that span several files:
- **Processed datasets** (`data/processed/<subset>[_w<W>_c<C>]/`): `dataset.npz` + `preprocessor.joblib`, built by `data/build_dataset.py` (`ensure_dataset` builds a `(window, cap)` variant on demand; path logic in `paths.dataset_dir`). Split is by engine; scaler is fit on training engines only; per-operating-regime scaling (KMeans, 6 regimes) for FD002/FD004.
- **Targets**: train/val RUL is capped (default 125); test RUL is the true uncapped value at each engine's last cycle. Test metrics use only the last window per engine (official protocol).
- **Experiment tracking**: all training goes through `training/tracking.setup_mlflow()` (SQLite `mlflow.db`, experiment `rul-cmapss`; override with `MLFLOW_TRACKING_URI`). Runs carry `tags.stage` (baseline/deep/sweep/final/ensemble) and `params.rul_cap`; each run logs its `preprocessing/preprocessor.joblib`.
- **Model selection**: `training/registry.py` groups runs by params (seed excluded), ranks by mean `val_rmse_deg` (validation RMSE on degradation-phase windows) and registers the winner as `rul-champion@champion`. Validation metrics are only comparable at equal RUL cap.
- **Serving** (`serving/`): `Predictor.from_registry` loads the champion, reads `window` from the registered run's params and the preprocessor from that run's artifacts. `/predict` takes raw readings (24 columns), applies the *same* Preprocessor -> last-`window` front-padded window -> tabular features -> model. `tests/test_serving_parity.py` guards against training/serving skew. Set `PMAINT_PRED_LOG=<file.jsonl>` to log each prediction (input for Phase 7 drift monitoring).
- Serving currently supports only tabular (xgboost) champions.
- **Drift monitoring** (`monitoring/`): `traffic.py` generates API-format log records via the real `Predictor` (reference = FD001 engines `unit % 5 < 3`, live = the rest; test engines are NOT used because they are truncated early in life). `drift.py` runs Evidently with a normalised-Wasserstein test (thresholds in module constants; the default K-S test gave false alarms on correlated rows) and maps results to ok/warning/critical.
- **Containers** (`docker/`, `docker-compose.yml`): one multi-stage Dockerfile (`serving`, `monitor` targets) + `Dockerfile.mlflow`. Serving deps are exact pins (`docker/requirements-serving.txt`, `xgboost-cpu`, `mlflow-skinny`) because the model/preprocessor are pickles; keep them in sync with the training env. The API image has no torch/Evidently. The API retries model loading every 15 s, so start order vs. `promote` does not matter. `PMAINT_HOME` sets the data root inside containers; the monitor needs `data/raw` mounted.
