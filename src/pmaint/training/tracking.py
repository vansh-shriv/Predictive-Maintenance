"""Shared MLflow configuration (local SQLite backend; overridable via MLFLOW_TRACKING_URI)."""
import os

import mlflow

from pmaint.paths import ROOT

EXPERIMENT = "rul-cmapss"


def setup_mlflow(experiment: str = EXPERIMENT) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(experiment)
