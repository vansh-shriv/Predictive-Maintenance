# Phase 6 — Serving API (FastAPI)

## Objective
Expose the registered champion (`models:/rul-champion@champion`) behind an HTTP API that accepts **raw telemetry** and returns a RUL — reproducing the training pipeline exactly.

## What was done
- `serving/schemas.py` — pydantic models. `Reading` is generated from the 24 raw column names (`setting_1..3`, `s_1..s_21`), all required finite floats. `PredictRequest` = optional `engine_id` + 1–1000 chronological `readings`.
- `serving/predictor.py` — `Predictor.from_registry()`: resolves the alias, reads `window` from the registered run's params, downloads that run's `preprocessing/preprocessor.joblib`, loads the model via `mlflow.pyfunc`. `predict()` = Preprocessor.transform → `make_last_windows` (front-pads short histories) → `window_to_tabular` → model → clip ≥ 0. Refuses non-xgboost champions with a clear error.
- `serving/app.py` — endpoints:
  - `GET /health` → 200 `{status: ok}` or **503** with the load error (process stays up so the problem is visible).
  - `GET /model-info` → name, version, run id, window (50), feature list.
  - `POST /predict` → `predicted_rul`, `cycles_received`, `window`, `padded`, `model_version`, `model_run_id`. Validation failures → 422; model not loaded → 503.
  - Optional JSONL prediction log (`PMAINT_PRED_LOG`), one record per call with the last reading — this will feed Phase 7 drift monitoring.
- `serving/sample_request.py` — builds a payload from a real test engine for demos/curl.
- `tests/test_api.py` (6 tests, stubbed model — no MLflow needed) and `tests/test_serving_parity.py` (real champion vs offline pipeline).

## Verification (real server, real champion)
```
GET /health        -> {"status":"ok","model_loaded":true}
GET /model-info    -> rul-champion v1, window 50, xgboost
POST unit 1  (31 cycles, padded)  -> 113.3   (true RUL 112)
POST unit 24 (186 cycles)         -> 24.7    (true RUL 20)
POST {"readings":[{"s_2":1}]}     -> 422
3 requests logged to the JSONL prediction log
```
`pytest`: 20 passed.

## Training/serving skew check
`test_serving_parity` feeds 10 raw FD001 test engines through the API pipeline and compares with predictions from the offline `dataset.npz` features. Inputs agree to ~1e-7 (float32 rounding from batch-vs-single reductions) but XGBoost output differed by up to **0.08 cycles** when a feature sits on a split threshold. Tolerance set to 0.5 cycles; real skew (wrong scaler, wrong window) would show up as many cycles.

## Caveats / limitations
- **Short histories give unreliable predictions.** Sending only the last 5 cycles of engine 24 (true RUL 20) returned **109.9**: front-padding with the first row makes the window look flat, so the mean/slope features say "no degradation". The `padded` flag is returned so clients can discount such predictions; a hard minimum history or a confidence warning is a candidate improvement (parked in DECISIONS).
- The API is **stateless**: the caller sends the history each time. A production design might store per-engine buffers.
- No authentication, rate limiting, or batch endpoint yet.
- Model is loaded once at startup from the local `mlflow.db` registry; changing the `champion` alias needs a restart. Docker/remote MLflow comes in Phase 8.
- `fastapi.testclient` emits a deprecation warning suggesting `httpx2`; harmless for now.

## How to run
```powershell
.venv\Scripts\Activate.ps1
$env:PMAINT_PRED_LOG = "logs/predictions.jsonl"   # optional; logs/ is git-ignored
uvicorn pmaint.serving.app:app --port 8000
# in another shell:
python -m pmaint.serving.sample_request --unit 24 > payload.json
curl.exe -X POST http://localhost:8000/predict -H "content-type: application/json" -d "@payload.json"
# interactive docs: http://localhost:8000/docs
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(serving): FastAPI RUL service loading the MLflow champion, with skew test (phase 6)"
git push
```
`CLAUDE.md` was also updated with the real commands and cross-file architecture notes.

## Next
Phase 7 — Drift monitoring with Evidently: reference data from training, "production" data from the prediction log, data-drift and prediction-drift reports, and a simulated drift scenario (e.g. sensor offset/degradation shift) to prove the monitor fires.
