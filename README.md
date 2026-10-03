# Predictive Maintenance — RUL Prediction with MLOps

Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan dataset using an LSTM / 1D-CNN, compared against an XGBoost baseline. Served with FastAPI, tracked in MLflow, monitored for drift with Evidently.

Status: built phase by phase — see [docs/ROADMAP.md](docs/ROADMAP.md). Decisions and ideas: [docs/DECISIONS.md](docs/DECISIONS.md). Per-phase write-ups: [docs/phases/](docs/phases/).

## Structure
```
data/raw, data/processed   # git-ignored, produced by scripts
notebooks/                 # EDA only
src/pmaint/
  data/                    # download, loading, windowing
  features/                # RUL labels, scaling
  models/                  # xgboost baseline, LSTM / 1D-CNN
  training/                # train + evaluate + MLflow logging
  serving/                 # FastAPI app
  monitoring/              # Evidently drift reports
tests/
docker/
docs/
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```
