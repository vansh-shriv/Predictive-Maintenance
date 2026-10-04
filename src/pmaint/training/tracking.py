"""Shared MLflow configuration (local SQLite backend; overridable via MLFLOW_TRACKING_URI)."""
import os

import mlflow

from pmaint.paths import ROOT

EXPERIMENT = "rul-cmapss"


def setup_mlflow(experiment: str = EXPERIMENT) -> None:
    default_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
    mlflow.set_experiment(experiment)
