import numpy as np
import pandas as pd
from helpers import stub_predictor

from pmaint.monitoring import drift, traffic

FEATS = [f"s_{i}" for i in range(1, 11)]


def _frame(rng, n, shift=None, pred_mean=70.0):
    df = pd.DataFrame(rng.normal(size=(n, len(FEATS))), columns=FEATS)
    for col, d in (shift or {}).items():
        df[col] += d
    df["predicted_rul"] = rng.normal(pred_mean, 20, size=n)
    return df


def test_no_drift_is_ok():
    rng = np.random.default_rng(0)
    r = drift.analyse(_frame(rng, 300), _frame(rng, 300), FEATS)
    assert r["level"] == "ok" and r["drifted_features"] == [] and not r["prediction_drift"]


def test_localised_shift_warns_and_names_the_sensor():
    rng = np.random.default_rng(0)
    r = drift.analyse(_frame(rng, 300), _frame(rng, 300, shift={"s_3": 2.0}), FEATS)
    assert r["level"] == "warning" and r["drifted_features"] == ["s_3"]


def test_broad_shift_is_critical():
    rng = np.random.default_rng(0)
    shift = {c: 2.0 for c in FEATS[:5]}
    r = drift.analyse(_frame(rng, 300), _frame(rng, 300, shift=shift, pred_mean=20), FEATS)
    assert r["level"] == "critical" and r["prediction_drift"]


def test_too_few_rows_is_flagged():
    rng = np.random.default_rng(0)
    r = drift.analyse(_frame(rng, 300), _frame(rng, 10), FEATS)
    assert r["level"] == "insufficient_data"


def test_simulated_record_matches_api_log_schema():
    """Simulated traffic must be interchangeable with what POST /predict writes to the log."""
    from pmaint.serving.predictor import RAW_COLS

    rng = np.random.default_rng(0)
    hist = pd.DataFrame(rng.normal(size=(40, len(RAW_COLS))), columns=RAW_COLS)
    rec = traffic._record(stub_predictor(), "e1", hist)
    expected = {"ts", "engine_id", "cycles", "predicted_rul", "model_version", "last_reading"}
    assert set(rec) == expected
    assert set(rec["last_reading"]) == set(RAW_COLS) and rec["cycles"] == 40
