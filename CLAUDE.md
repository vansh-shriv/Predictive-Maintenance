# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
RUL prediction on NASA C-MAPSS turbofan data: LSTM / 1D-CNN vs XGBoost baseline, served with FastAPI, tracked in MLflow, drift-monitored with Evidently. See `docs/ROADMAP.md` for the phase plan.

## Working agreement (important)
- Build **one phase at a time**. At the end of each phase: write/update `docs/phases/PHASE_XX_*.md`, log decisions/ideas in `docs/DECISIONS.md`, then give the user commit message(s) and **stop until the user confirms they have committed and pushed**. Do not run git commit/push yourself.
- Document every move, step, and idea in the markdown docs as it happens.

## Commands
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .
pytest                      # all tests
pytest tests/test_x.py::test_name   # single test
ruff check .
```

## Architecture
src layout: package `pmaint` in `src/pmaint/` (data → features → models → training → serving / monitoring). Notebooks are for EDA only; logic lives in `src/`. `data/raw` and `data/processed` are git-ignored and reproduced by scripts.

Update this file as phases add real commands (training entry points, `mlflow ui`, `uvicorn` app path, docker-compose).
