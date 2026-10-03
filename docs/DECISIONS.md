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
