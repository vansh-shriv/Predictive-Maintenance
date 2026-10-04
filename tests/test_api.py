import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pmaint.data.loader import SENSOR_COLS, SETTING_COLS
from pmaint.features.preprocessing import Preprocessor
from pmaint.serving import app as app_module
from pmaint.serving.predictor import RAW_COLS, Predictor

WINDOW = 10


class StubModel:
    def predict(self, X):
        return np.full(len(X), 42.0)


def _stub_predictor():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, len(RAW_COLS))), columns=RAW_COLS)
    pre = Preprocessor(1).fit(df)
    meta = {"model_name": "rul-champion", "version": "1", "run_id": "abc", "model_type": "xgboost"}
    return Predictor(StubModel(), pre, WINDOW, meta)


def _payload(n):
    rng = np.random.default_rng(1)
    rows = [dict(zip(RAW_COLS, rng.normal(size=len(RAW_COLS)).tolist())) for _ in range(n)]
    return {"engine_id": "e1", "readings": rows}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Predictor, "from_registry", classmethod(lambda cls: _stub_predictor()))
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["window"] == WINDOW and info["version"] == "1"


def test_predict_ok_and_padding_flag(client):
    r = client.post("/predict", json=_payload(3)).json()
    assert r["predicted_rul"] == 42.0 and r["padded"] is True and r["cycles_received"] == 3
    r = client.post("/predict", json=_payload(25)).json()
    assert r["padded"] is False and r["engine_id"] == "e1"


def test_predict_validation_errors(client):
    assert client.post("/predict", json={"readings": []}).status_code == 422
    bad = _payload(2)
    del bad["readings"][0][SENSOR_COLS[0]]
    assert client.post("/predict", json=bad).status_code == 422


def test_degraded_when_model_missing(monkeypatch):
    def boom(cls):
        raise RuntimeError("no registry")

    monkeypatch.setattr(Predictor, "from_registry", classmethod(boom))
    with TestClient(app_module.app) as c:
        h = c.get("/health")
        assert h.status_code == 503 and "no registry" in h.json()["error"]
        assert c.post("/predict", json=_payload(2)).status_code == 503


def test_prediction_log(client, monkeypatch, tmp_path):
    log = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PMAINT_PRED_LOG", str(log))
    client.post("/predict", json=_payload(2))
    assert len(log.read_text().strip().splitlines()) == 1


def test_setting_cols_in_schema():
    assert set(SETTING_COLS) <= set(RAW_COLS)
