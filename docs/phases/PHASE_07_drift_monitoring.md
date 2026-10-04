# Phase 7 — Drift monitoring (Evidently)

## Objective
Detect when live traffic stops looking like the data the model was built on, using the API's prediction log, and prove the monitor fires on realistic faults without crying wolf on normal traffic.

## What was done
- `monitoring/traffic.py` — simulated production traffic. Each record is produced by a real `Predictor.predict()` call on a truncated history (random cut point), in exactly the format `POST /predict` logs (`ts, engine_id, cycles, predicted_rul, model_version, last_reading`). A test checks the schema matches.
- `monitoring/simulate.py` — CLI that writes a scenario log (`logs/sim_<scenario>.jsonl`).
- `monitoring/drift.py` — builds a reference set, converts logs to monitored columns, runs the Evidently `DataDriftPreset`, applies an alert policy, optionally writes the HTML report. CLI exit code: **0 ok / 1 warning / 2 critical** (so cron/CI can act on it).
- `tests/test_drift.py` — 5 tests (synthetic data, no MLflow): no-drift → ok, single shifted sensor → warning naming it, broad shift → critical, too few rows flagged, log-schema parity.

## What is monitored
- **Inputs:** the last raw reading of each request, **scaled with the training Preprocessor** (so operating regimes are normalised and the comparison is on the same scale the model sees). 15 informative sensors.
- **Output:** `predicted_rul`.
- **Reference:** 60 FD001 engines (`unit % 5 < 3`), 3 random cut points each → 180 rows.
- **No labels in production**, so drift is a *proxy* for degradation, not a measurement of error. True RUL only becomes known after an engine fails; a delayed-label accuracy check is a future item.

## Scenarios (current traffic = the other 40 engines, 5 cuts each → 200 rows)
| Scenario | What it simulates | Expected |
|---|---|---|
| `normal` | same population, different engines | ok |
| `sensor_bias` | calibration fault: `s_4, s_11, s_12` shifted by +1.5 training-σ | warning (localised fault) |
| `aged_fleet` | engines in the last 15% of life | critical (real population shift) |

## Alert policy
Per column: **normalised Wasserstein distance** (distance in reference-std units) > **0.6** for sensors, > **0.2** for `predicted_rul`.
- `critical`: ≥ 30% of sensor columns drifted
- `warning`: ≥ 1 sensor column drifted, or prediction drift
- `insufficient_data`: < 50 current rows
- else `ok`

## Results
Decision-relevant numbers (largest per-sensor distance; "pred" = predicted_rul distance):
| Scenario | runs | max sensor distance (min / median / max) | pred distance (median) | outcome |
|---|---|---|---|---|
| normal | 30 seeds (tuning) | 0.16 / 0.24 / 0.46 | 0.08 | – |
| sensor_bias | 6 | 1.53 / 1.58 / 1.67 | 0.25 | – |
| aged_fleet | 3 | 1.39 / 1.42 / 1.43 | 1.52 | – |

Validation on **fresh seeds not used for tuning** (final thresholds):
| Scenario | runs | outcomes |
|---|---|---|
| normal | 40 | 38 ok, **2 false warnings (5%)** — both prediction-drift only, no sensor flagged |
| sensor_bias | 10 | 10/10 warning, always flags exactly `s_4, s_11, s_12` + prediction drift (mean prediction 76 → ~65) |
| aged_fleet | 3 | 3/3 critical, 14 of 15 sensors drifted, prediction mean 15 vs 77 |

End-to-end CLI check: normal → exit 0, sensor_bias → exit 1, aged_fleet → critical; HTML reports written (~4.7 MB each, git-ignored).

## Why Wasserstein and not Evidently's default K-S test (a failed first attempt)
The first version used Evidently's default per-column test (K-S, p < 0.05) with "≥ 2 drifted columns = warning". On fresh seeds it produced **3 false alarms in 20 normal runs (2 critical)**. Cause: rows are not independent — several cuts come from the same engine and engines differ in baseline wear — so the test is over-confident, and 15 simultaneous tests add multiple-comparison noise. Switching to an **effect-size** measure (how far the distribution moved, in std units) removed the problem: normal traffic stays below ~0.46 while real faults sit above 1.3, a 3× gap.

An earlier simulation bug also produced a false *critical* on "normal": using the unseen test engines as "normal" traffic made predictions average 105 vs 74 in the reference, because test engines are truncated early in life (different life-stage mix). The monitor was right that the populations differ; the simulation was wrong. Reference and current traffic now come from the same population (disjoint engines).

## Caveats
- **Thresholds were tuned on simulated data** (30 normal / 6 bias seeds) and then checked on fresh seeds, but real traffic may behave differently. Expect to recalibrate on real operations data.
- Sensitivity: a mean shift under roughly **0.6 σ** per sensor will not be flagged; this is a deliberate trade-off against false alarms.
- A ~5% false-warning rate remains, driven by prediction drift (threshold 0.2 sits just above the normal maximum of 0.17).
- Reference and "live" engines were both part of the model's training set (disjoint engines, but the model saw them), so prediction distributions are slightly sharper than real unseen traffic. This does not affect the input-drift logic.
- Monitors only the *last reading*; slow trend/slope drift and multivariate drift (correlation changes) are not covered.
- FD001 only (one operating regime). Multi-regime subsets are handled by scaling, but untested.
- The monitor is a batch CLI; scheduling, storage of reports over time and alert delivery are not implemented.

## How to reproduce
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.monitoring.simulate --scenario sensor_bias      # -> logs/sim_sensor_bias.jsonl
python -m pmaint.monitoring.drift --current logs/sim_sensor_bias.jsonl --html reports/drift_sensor_bias.html
echo $LASTEXITCODE                                               # 0 ok / 1 warning / 2 critical
# or monitor the real API log:
$env:PMAINT_PRED_LOG = "logs/predictions.jsonl"; uvicorn pmaint.serving.app:app
python -m pmaint.monitoring.drift --current logs/predictions.jsonl
pytest tests/test_drift.py
```
Open the HTML report in a browser for per-column distribution plots.

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(monitoring): Evidently drift monitor with traffic simulator, alert policy and tests (phase 7)"
git push
```
`logs/` and `reports/*.html` are git-ignored.

## Next
Phase 8 — Containerization: Dockerfile for the API, docker-compose with an MLflow tracking server (shared registry) + API (+ optional monitor job), so the champion no longer depends on a local `mlflow.db`.
