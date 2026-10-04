"""Shared test doubles. Imported as `helpers` (tests/ is on pytest's pythonpath, see pyproject)."""
import numpy as np
import pandas as pd

from pmaint.features.preprocessing import Preprocessor
from pmaint.serving.predictor import RAW_COLS, Predictor

WINDOW = 10


class StubModel:
    def predict(self, X):
        return np.full(len(X), 42.0)


def stub_predictor() -> Predictor:
    """A Predictor with a constant model (42 cycles) and a real Preprocessor fit on noise."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, len(RAW_COLS))), columns=RAW_COLS)
    pre = Preprocessor(1).fit(df)
    meta = {"model_name": "rul-champion", "version": "1", "run_id": "abc", "model_type": "xgboost"}
    return Predictor(StubModel(), pre, WINDOW, meta)


def payload(n: int) -> dict:
    """A valid /predict request body with n random readings."""
    rng = np.random.default_rng(1)
    rows = [
        dict(zip(RAW_COLS, rng.normal(size=len(RAW_COLS)).tolist(), strict=True))
        for _ in range(n)
    ]
    return {"engine_id": "e1", "readings": rows}
