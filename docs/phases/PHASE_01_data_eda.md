# Phase 1 — Data acquisition & EDA

## Objective
Get C-MAPSS reproducibly into `data/raw`, load it into clean DataFrames, and understand its structure before modelling.

## What was done
- `src/pmaint/paths.py` — central project paths.
- `src/pmaint/data/download.py` — downloads the NASA PHM zip, extracts the (nested) txt files to `data/raw`. Idempotent; `--force` re-downloads.
- `src/pmaint/data/loader.py` — `load_subset("FD00x")` → `(train, test, rul_test)` with named columns (`unit`, `cycle`, `setting_1..3`, `s_1..s_21`); `add_train_rul()` adds the uncapped linear RUL.
- `notebooks/01_eda.ipynb` — EDA (not executed in repo; run it locally).
- `tests/test_loader.py` — shape/RUL sanity tests (skipped if data is absent).

## Dataset facts (verified by running the loader)
| Subset | Train rows | Test rows | Train/Test units | Life min/mean/max (cycles) | Op. conditions | Constant columns |
|---|---|---|---|---|---|---|
| FD001 | 20,631 | 13,096 | 100 / 100 | 128 / 206 / 362 | 1 | setting_3, s_1, s_5, s_10, s_16, s_18, s_19 |
| FD002 | 53,759 | 33,991 | 260 / 259 | 128 / 207 / 378 | 6 | none (regimes mask them) |
| FD003 | 24,720 | 16,596 | 100 / 100 | 145 / 247 / 525 | 1 | setting_3, s_1, s_5, s_16, s_18, s_19 |
| FD004 | 61,249 | 41,214 | 249 / 248 | 128 / 246 / 543 | 6 | none |

- No missing values in any train file.
- Train = full run-to-failure. Test = trajectories cut before failure; `RUL_FDxxx.txt` gives the true RUL at the last test cycle (FD001: 7–145).
- FD001/FD003: one operating condition; FD002/FD004: six, so sensors must be normalised **per operating regime** (Phase 2).
- FD001 = 1 fault mode (HPC degradation); FD003 = 2 fault modes (HPC + fan).

## Implications for later phases
- Drop constant sensors for FD001/FD003 (they carry no signal).
- Engine lifetimes vary widely (128–543), so RUL labels should be capped (piecewise-linear) — early life shows no degradation.
- Splits **must be by engine (`unit`)**, never by row, to avoid leakage.
- The official evaluation uses only the last window of each test engine.

## How to reproduce
```powershell
pip install -e .
python -m pmaint.data.download
jupyter notebook notebooks/01_eda.ipynb
pytest tests/test_loader.py
```
Note: `pytest` was not installed in the dev environment, so the test logic was checked with an equivalent inline script; run `pip install -r requirements.txt` to use pytest properly.

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(data): add C-MAPSS downloader, loader, EDA notebook and docs (phase 1)"
git push
```
`data/raw` is git-ignored, so the dataset is not committed.

## Next
Phase 2 — Preprocessing: capped RUL labels, sensor selection, per-regime scaling, sliding windows, engine-wise validation split.
