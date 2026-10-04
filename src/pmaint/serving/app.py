"""FastAPI service: uvicorn pmaint.serving.app:app

Env: MLFLOW_TRACKING_URI (default: local sqlite mlflow.db), PMAINT_PRED_LOG (optional JSONL path
that receives one record per prediction -- input for the Phase 7 drift monitor).
"""
import json
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pmaint.serving.predictor import RAW_COLS, Predictor, prediction_log_path
from pmaint.serving.schemas import ModelInfo, PredictRequest, PredictResponse

log = logging.getLogger("pmaint.api")


RELOAD_INTERVAL_S = 15  # retry a failed model load at most this often


def _try_load(app: FastAPI) -> None:
    app.state.last_attempt = time.monotonic()
    try:
        app.state.predictor = Predictor.from_registry()
        app.state.load_error = None
        log.info("Loaded %s", app.state.predictor.meta)
    except Exception as e:  # keep the process up so /health can report the problem
        app.state.load_error = f"{type(e).__name__}: {e}"
        log.warning("Model load failed: %s", app.state.load_error)


def _retry_if_needed() -> None:
    """If no model is loaded (e.g. registry was still empty at startup), retry periodically."""
    if app.state.predictor is None and time.monotonic() - app.state.last_attempt >= RELOAD_INTERVAL_S:
        _try_load(app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    _try_load(app)
    yield


app = FastAPI(title="Turbofan RUL API", version="0.1.0", lifespan=lifespan)


def _predictor() -> Predictor:
    _retry_if_needed()
    p = app.state.predictor
    if p is None:
        raise HTTPException(503, f"Model not loaded: {app.state.load_error}")
    return p


@app.get("/health")
def health():
    _retry_if_needed()
    ok = app.state.predictor is not None
    body = {"status": "ok" if ok else "degraded", "model_loaded": ok}
    if not ok:
        body["error"] = app.state.load_error
    return body if ok else JSONResponse(body, status_code=503)


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    p = _predictor()
    return ModelInfo(**p.meta, window=p.window, features=p.feature_names)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = _predictor()
    df = pd.DataFrame([r.model_dump() for r in req.readings])[RAW_COLS]
    rul, padded = p.predict(df)

    log_path = prediction_log_path()
    if log_path is not None:
        rec = {"ts": time.time(), "engine_id": req.engine_id, "cycles": len(df),
               "predicted_rul": rul, "model_version": p.meta["version"],
               "last_reading": df.iloc[-1].to_dict()}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return PredictResponse(
        engine_id=req.engine_id, predicted_rul=rul, cycles_received=len(df), window=p.window,
        padded=padded, model_version=p.meta["version"], model_run_id=p.meta["run_id"],
    )
