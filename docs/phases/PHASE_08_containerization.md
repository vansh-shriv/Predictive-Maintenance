# Phase 8 — Containerization (Docker + docker-compose)

## Objective
Make the service reproducible and independent of the developer machine: the API and a shared MLflow tracking/registry server run in containers, the champion lives in the server's registry (not in a local `mlflow.db`), and the drift monitor runs as a one-shot job.

## What was done
- `docker/Dockerfile` — multi-stage: `base` (python 3.12-slim, `libgomp1` for xgboost, non-root user `app`, source copied, `PYTHONPATH=/app/src`) → `serving` (uvicorn + HEALTHCHECK on `/health`) and `monitor` (adds Evidently; CMD runs the drift CLI).
- `docker/requirements-serving.txt` — **exact pins** matching the training environment (xgboost/scikit-learn pickles are version sensitive). Uses `mlflow-skinny` (HTTP client only) and `xgboost-cpu`.
- `docker/Dockerfile.mlflow` — MLflow 3.16.1 server: SQLite backend + proxied artifact store, both on a named volume.
- `docker-compose.yml` — services `mlflow` (healthcheck), `api` (waits for healthy mlflow, `MLFLOW_TRACKING_URI=http://mlflow:5000`, `./logs` mounted for the prediction log) and `monitor` (profile; mounts `./logs`, `./reports`, read-only `./data/raw`).
- `training/promote.py` — copies `rul-champion@champion` from one registry to another (model, preprocessor artifact, params, metrics, tags, provenance tags). This is the bridge from "trained on my laptop" to "served from the shared registry".
- `serving/app.py` — model load now **retries every 15 s** while no model is loaded, so the API recovers on its own when the registry is populated after startup. Test added.
- `paths.py` — `PMAINT_HOME` env override for the data root inside containers.
- `monitoring/replay.py` + `traffic.histories()` — replay simulated engine histories over HTTP against a running API (stack smoke test).
- `.dockerignore` keeps data, venv, mlruns, logs, notebooks, tests and docs out of the build context.

## Verification (real stack, Docker 28.5)
| Check | Result |
|---|---|
| `docker compose up -d --build` | mlflow healthy, api starts |
| API before promotion | `/health` 503 `degraded` (registry empty) — as designed |
| `promote` then wait ~15 s | API self-loads: `/health` 200, `/model-info` → window 50, xgboost |
| Prediction parity | unit 1 → **113.2662**, unit 24 → **24.7470**, identical to the local (non-Docker) API, also after the CPU-only image rebuild |
| Replay 200 `normal` requests over HTTP | 200 OK (~21 s), prediction log written by the container |
| `monitor` container on that log | `level: ok`, exit 0, HTML report written |
| Replay 200 `sensor_bias` requests then monitor | exit **1** (warning) — drift detected end to end |
| Restart mlflow container | registry/alias still present (named volume `pmaint_mlflow-data`) |
| `pytest` | 26 passed |

## Image sizes
| Image | Size |
|---|---|
| pmaint-api | 847 MB (was 1.63 GB) |
| pmaint-monitor | 1.55 GB (was 2.33 GB) |
| pmaint-mlflow | 1.27 GB |

The first API image was 1.63 GB because the default Linux xgboost wheel drags in ~345 MB of CUDA/NCCL libraries that a CPU-only service never uses; switching to `xgboost-cpu==3.4.1` removed them with bit-identical predictions. torch is deliberately **not** in the API image (the champion is XGBoost).

## Issues hit and fixes
- `promote` crashed at the end of its MLflow run with `UnicodeEncodeError` (MLflow prints an emoji; Windows consoles default to cp1252). The model was not registered, but a RUNNING run was left on the server. Fixed by reconfiguring stdout to UTF-8 in the script; the stray run was deleted via the MLflow API.
- Retry-timer comparison used `>`; on Windows the monotonic clock resolution (~15 ms) made a zero-interval test flaky. Changed to `>=`.
- MLflow logs a harmless warning on model load: `psutil` / `rich` "uninstalled" — they were recorded as inferred model requirements but the XGBoost flavor does not use them (predictions verified identical).

## Caveats / not done
- The registry contents are a **copy**: `promote` creates a new run/version on the server; run ids differ from the local ones (provenance is in tags `promoted_from_run` / `promoted_from_uri`).
- No authentication or TLS on MLflow or the API; ports are published on localhost for development only.
- The API still loads the model only at startup/after failures; changing the `champion` alias on a *healthy* API needs `docker compose restart api`.
- SQLite + local-volume artifacts is fine for one machine; a team setup would use Postgres + object storage.
- The monitor image is large (Evidently pulls plotly/scipy/etc.) and generates its reference from `data/raw`, which must be mounted; there is no scheduler yet.
- Not tested: Linux/macOS hosts, GPU, `--platform` arm64 builds.
- Other containers visible on this machine belong to unrelated projects and were not touched.

## How to run
```powershell
mkdir logs, reports -ErrorAction SilentlyContinue
docker compose up -d --build
.venv\Scripts\python.exe -m pmaint.training.promote --src sqlite:///mlflow.db --dst http://localhost:5000
curl.exe http://localhost:8000/health            # ok after ~15 s
# MLflow UI: http://localhost:5000     API docs: http://localhost:8000/docs
python -m pmaint.monitoring.replay --scenario sensor_bias        # send traffic to the container API
docker compose --profile monitor run --rm monitor                # exit code 0/1/2, reports/drift.html
docker compose down                                              # keep data; add -v to delete the registry volume
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(docker): API, MLflow server and drift-monitor containers with compose and champion promotion (phase 8)"
git push
```
`logs/` and `reports/*.html` stay ignored. The `docker compose` stack is currently stopped (`docker compose down`); the registry volume is kept.

## Next
Phase 9 — Testing & CI: ruff configuration and a clean lint pass, pytest split into fast unit tests vs. tests needing data/registry, and a GitHub Actions workflow (lint, tests, Docker build; no secrets).
