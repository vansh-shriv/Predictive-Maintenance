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
