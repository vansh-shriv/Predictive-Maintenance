"""Turns raw telemetry into a RUL prediction using the registered champion + its preprocessor."""
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from pmaint.data.loader import SENSOR_COLS, SETTING_COLS
from pmaint.features.preprocessing import Preprocessor
from pmaint.features.tabular import window_to_tabular
from pmaint.features.windows import make_last_windows
from pmaint.training.registry import ALIAS, MODEL_NAME
from pmaint.training.tracking import setup_mlflow

RAW_COLS = SETTING_COLS + SENSOR_COLS


class Predictor:
    """Preprocessor + window + tabular features + model. Must mirror training exactly."""

    def __init__(self, model, preprocessor: Preprocessor, window: int, meta: dict):
        self.model, self.pre, self.window, self.meta = model, preprocessor, window, meta
        self.feature_names = list(preprocessor.feature_cols_)

    @classmethod
    def from_registry(cls, name: str = MODEL_NAME, alias: str = ALIAS) -> "Predictor":
        setup_mlflow()  # honours MLFLOW_TRACKING_URI
        client = MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        model_type = mv.tags.get("model_type", "unknown")
        if model_type != "xgboost":
            raise RuntimeError(f"Serving supports tabular xgboost models, got '{model_type}'")
        run = client.get_run(mv.run_id)
        window = int(run.data.params["window"])
        local = client.download_artifacts(mv.run_id, "preprocessing/preprocessor.joblib")
        pre = Preprocessor.load(Path(local))
        model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
        meta = {"model_name": name, "version": str(mv.version), "run_id": mv.run_id,
                "model_type": model_type}
        return cls(model, pre, window, meta)

    def predict(self, readings: pd.DataFrame) -> tuple[float, bool]:
        """readings: chronological raw rows with RAW_COLS. Returns (rul, padded)."""
        df = readings[RAW_COLS].astype(float).reset_index(drop=True)
        scaled = self.pre.transform(df)
        scaled["unit"] = 1
        X, _ = make_last_windows(scaled, self.pre.feature_cols_, window=self.window)
        feats, _ = window_to_tabular(X, self.feature_names)
        rul = float(np.clip(self.model.predict(feats), 0, None)[0])
        return rul, len(df) < self.window


def prediction_log_path() -> Path | None:
    p = os.environ.get("PMAINT_PRED_LOG")
    return Path(p) if p else None
