# Roadmap

Goal: RUL (Remaining Useful Life) prediction on NASA C-MAPSS turbofan data with an LSTM / 1D-CNN, benchmarked against an XGBoost baseline. Served with FastAPI, tracked in MLflow, drift-monitored with Evidently.

Workflow: one phase at a time. Each phase ends with a docs entry in `docs/phases/` and a suggested commit message. The next phase starts only after the user has committed and pushed.

| Phase | Name | Output |
|-------|------|--------|
| 0 ✅ | Project scaffolding & docs | Folder structure, README, .gitignore, requirements, docs skeleton |
| 1 ✅ | Data acquisition & EDA | C-MAPSS download script, loader, EDA notebook, data notes |
| 2 ✅ | Preprocessing & feature engineering | RUL labels (piecewise-linear cap), scaling, sliding windows, train/val split by engine |
| 3 ✅ | XGBoost baseline | Baseline model, RMSE + NASA scoring function, first MLflow runs |
| 4 ✅ | Deep model (LSTM and/or 1D-CNN) | PyTorch models, training loop, comparison vs baseline in MLflow |
| 5 ✅ | Experiment tracking & model registry | MLflow params/metrics/artifacts, registered "Production" model |
| 6 | Serving API | FastAPI `/predict`, `/health`, schema validation, loads model from registry |
| 7 | Drift monitoring | Evidently reports (data + prediction drift), simulated drift scenario |
| 8 | Containerization | Dockerfile(s), docker-compose (API + MLflow) |
| 9 | Testing & CI | pytest, ruff, GitHub Actions |
| 10 | Pipeline automation & final docs | Reproducible pipeline (e.g. DVC/Makefile), retraining trigger idea, final README |

Phases may be re-ordered or split as we learn; changes are logged in `docs/DECISIONS.md`.
