# Phase 0 — Project scaffolding & docs

## Objective
Create a clean, conventional repo skeleton so every later phase has an obvious home.

## What was done
- Created folder structure (see README).
- Added `.gitignore`, `requirements.txt`, `pyproject.toml`, `README.md`, `CLAUDE.md`.
- Added `docs/ROADMAP.md`, `docs/DECISIONS.md`, and this file.

## Why
- src layout + `pyproject.toml` lets us `pip install -e .` so notebooks, API, and tests share one import path.
- Docs-first so each move is recorded as it happens.

## How to reproduce / verify
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -c "import pmaint; print(pmaint.__version__)"
```

## Git steps (you run these)
```powershell
cd D:\D\AI-ML\PredictiveMaintenance
git init
git branch -M main
git add .
git commit -m "chore: scaffold project structure and phase docs (phase 0)"
git remote add origin <your-repo-url>
git push -u origin main
```

## Next
Phase 1 — Data acquisition & EDA.
