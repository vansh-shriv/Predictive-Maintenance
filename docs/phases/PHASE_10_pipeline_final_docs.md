# Phase 10 — Pipeline automation & final documentation

## Objective
Replace the hand-run command sequence with one reproducible entry point, define what drift should trigger (and what it should not), make re-runs safe for the registered champion, and finish the documentation.

## What was done
- **`python -m pmaint.pipeline`** — steps `data → train → register → promote` (always run in that order; any comma-separated subset via `--steps`).
  - `data`: idempotent download + processed arrays for every window used.
  - `train`: multi-seed `final` runs for the Phase 5 candidate pool (XGBoost windows 30 and 50, LSTM window 50), each tagged with a **pipeline id**.
  - `register`: ranks only *this pipeline's* candidates by mean validation metric and registers the winner.
  - `promote`: copies the champion into another registry (`--promote-to`).
  - `--quick` = 1 seed, XGBoost only (smoke run, ~40 s); `--skip-deep` = XGBoost candidates only; `--force` bypasses the gate.
- **Champion safety gate** (`training/registry.py`): an existing champion is replaced only if the candidate's mean validation metric beats it by ≥ 0.1 cycles (`MIN_GAIN`). Re-running the pipeline is therefore idempotent and a worse retrain can never silently take over. The champion's metric is read from a version tag (`config_mean_metric`) or, for v1, from its run.
- **Pipeline-id tag** on runs (`tags` argument added to `train_xgb.run` / `train_deep.run`) so a rerun's candidate pool is not polluted by earlier runs with identical params.
- **Retrain trigger** (`monitoring/trigger.py`): turns the drift verdict into an action; exit code 0 none / 1 investigate / 2 retrain; `--execute` runs `train + register` only when the action is `retrain`.
- **Docs:** rewritten `README.md` (architecture diagram, results, quick start, limitations), new `docs/RETROSPECTIVE.md`, updated `ROADMAP.md`, `DECISIONS.md`, `CLAUDE.md`.
- **Tests:** `tests/test_pipeline_logic.py` — step parsing/validation, promote requires a destination, the gate (no champion / clearly better / tie / within margin / worse), and the five decision-policy cases. Total suite: **38 tests**.

## Retrain policy — why not "drift → retrain"
| Drift verdict | Action | Reasoning |
|---|---|---|
| ok / insufficient data | none | nothing to act on; collect more traffic |
| warning (localised: a few sensors and/or predictions) | **investigate** | typically a sensor/calibration fault; retraining on faulty readings would bake the fault into the model |
| critical (≥ 30 % of sensors) + **new labelled data** | **retrain** | the population really changed and there is data to learn it from |
| critical, no new labelled data | **investigate** | retraining on the same data cannot fix a population shift |

The retrain still has to pass the registry gate, so it cannot make the deployed model worse by validation metrics. Checked on the Phase 7 simulated logs: `normal` → none (exit 0), `sensor_bias` → investigate (exit 1), `aged_fleet` → investigate (exit 1), `aged_fleet` + `--new-labelled-data` → retrain (exit 2; `--execute` not run).

## Verification
| Check | Result |
|---|---|
| Full pipeline `python -m pmaint.pipeline` (data, train 3 seeds × 3 configs incl. LSTM, register) | exit 0; **every metric identical to Phase 5** (e.g. XGB w50 s0 val_deg 10.71 / test 13.80; LSTM w50 s0 test 16.42 / 417.64). Ranking reproduced: XGB w50 10.72 → LSTM w50 12.52 → XGB w30 13.76. Gate: *champion unchanged*. |
| `python -m pmaint.pipeline --quick` | 40 s; gate: *champion unchanged* |
| `--steps promote --promote-to http://localhost:5000` against the compose stack | champion copied (registry v2 there); API loaded it by itself; `/predict` unit 24 → 24.747 (same as before) |
| Retrain trigger on 4 scenarios | as in the table above |
| `ruff check .` | clean |
| `pytest` (bare, local) | 38 passed |
| `pytest -m "not data and not registry"` in a clean copy (no data, no `mlflow.db`) | 35 passed, 3 deselected |
| GitHub Actions for this phase | **both jobs green** (lint-and-test, docker), confirmed after the push |

## Caveats
- The pipeline is a sequential Python script, not a DAG/orchestrator: no caching of finished steps beyond "dataset exists", no parallel training, no resume after a crash mid-train (rerun creates new runs under a new pipeline id).
- The training candidate pool is hard-coded (the Phase 5 selection). Changing it means editing `pipeline.py`; there is no config file.
- Full pipeline ≈ 15–20 min on CPU (LSTM dominates); CI deliberately does not run it.
- The trigger's `retrain` outcome needs someone to assert `--new-labelled-data`; nothing in this project produces new labelled data, so the retrain path was exercised only up to the decision.
- Gate margin (0.1 cycles) is a judgement call, not derived from a noise estimate; seed std of the validation metric is ~0.01 for XGBoost and larger for the LSTM.
- The promoted registry copy is a new run/version (provenance in tags), so version numbers differ between registries.

## How to run
```powershell
.venv\Scripts\Activate.ps1
python -m pmaint.pipeline --quick                      # smoke
python -m pmaint.pipeline                              # full reproduction
python -m pmaint.pipeline --steps promote --promote-to http://localhost:5000
python -m pmaint.monitoring.trigger --current logs/sim_aged_fleet.jsonl [--new-labelled-data] [--execute]
```

## Git steps (you run these)
```powershell
git add .
git commit -m "feat(pipeline): one-command pipeline, champion safety gate, retrain trigger, final README and retrospective (phase 10)"
git push
```
Then check the Actions tab for both jobs.

## Project complete
All 11 phases (0–10) are done. Remaining ideas and known gaps are in `docs/RETROSPECTIVE.md` ("What I would do next") and the parking lots in `docs/DECISIONS.md`.
