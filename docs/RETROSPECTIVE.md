# Retrospective

An honest account of the project: what was delivered, what turned out differently from the plan, the mistakes made and how each was caught, and what is still missing.

## Delivered
A complete MLOps loop for RUL prediction on C-MAPSS FD001: reproducible data/feature pipeline, tracked experiments and a validation-based model registry with a safety gate, a FastAPI service that serves the registered champion from raw telemetry, a drift monitor with a documented alert/retrain policy, Docker/Compose packaging, 38 automated tests, and CI (lint + tests + container smoke test) that is green on GitHub. One command (`python -m pmaint.pipeline`) reproduces training through registration.

## The modelling result (and why it matters)
The brief was an LSTM/1D-CNN *against* an XGBoost baseline. **The baseline won**: XGBoost 13.6–13.9 test RMSE vs LSTM 15.5–16.1 and CNN 19.5 on FD001, stable across seeds (gaps ≫ seed std). Tuning the deep models (window, width, loss, learning rate, BatchNorm) moved them only within ~15–16 (LSTM) and ≥ 17 (CNN); averaging with XGBoost made things worse than XGBoost alone. This was reported as-is and the registry picks by validation metrics, not by which architecture is more fashionable. The likely reasons (small training set of 80 engines, strong hand-crafted window features, little tuning) are hypotheses, not findings.

## What went wrong, and how it was caught
| # | Phase | Problem | How it was found | Fix / lesson |
|---|---|---|---|---|
| 1 | 2 | `pip install -e .` had produced a stale non-editable copy; new modules weren't importable | Tests failed with `No module named pmaint.features...` | Reinstall editable; `build/` ignored. Verify the install mode, not just "pip succeeded". |
| 2 | 5 | Plain validation RMSE rewarded a longer window by 3 RMSE while test RMSE didn't move | Window 50 vs 30: val 9.8 vs 12.7, test ≈ equal | Added `val_rmse_deg` (degradation-phase windows only); still imperfect and documented as such. |
| 3 | 5 | Risk of picking a lucky single seed as champion | Design review of the sweep | Registry ranks configs by **mean over seeds**; the rule was fixed before the final runs. |
| 4 | 6 | Training/serving skew risk (scaler, window, features) | Parity test on 10 real engines | Inputs agreed to 1e-7, yet XGBoost output moved up to 0.08 cycles because a tree split flipped; tolerance set at 0.5 cycles with the reason written down. |
| 5 | 7 | Default K-S drift test: 3 false alarms in 20 normal runs | Calibrating on fresh seeds | Rows from one engine aren't independent and 15 parallel tests inflate false hits → switched to a Wasserstein effect-size measure (normal ≤ 0.46 vs faults ≥ 1.3). |
| 6 | 7 | "Normal" simulated traffic raised a critical alarm | Prediction means 105 vs 74 | Test engines are truncated early in life; the *simulation* was wrong, the monitor was right. Reference and live traffic now come from the same population. |
| 7 | 8 | 1.6 GB API image | Inspecting site-packages | xgboost's Linux wheel pulled ~345 MB of CUDA libs; `xgboost-cpu` → 847 MB, bit-identical predictions. |
| 8 | 8 | `promote` crashed on an emoji (Windows cp1252) leaving a stray RUNNING run | Real run failed at the end | Force UTF-8 stdout; deleted the stray run. |
| 9 | 9 | Prediction-log write could fail with a root-owned bind mount on Linux, turning `/predict` into a 500 | Reasoning about a fresh CI/Linux checkout | Logging is best-effort; test added. |
| 10 | 9 | CI failed on test collection (`from tests.test_api import …`); my local runs used `python -m pytest`, which hides it | First GitHub run | Shared doubles in `tests/helpers.py` + pytest `pythonpath`. **Lesson: reproduce CI with the exact CI command**, my "clean copy" check had the same blind spot. |

## What I would do next (in rough priority)
1. **Close the validation/test gap.** Build a validation protocol that mimics the test (random truncation point per held-out engine, repeated) to get a selector that agrees with the test metric and has less noise than 100 test engines.
2. **Give the deep models a fair chance:** a proper search (Optuna), longer windows, hand-crafted features as extra channels, and FD003–FD004 where more complex dynamics might favour sequence models.
3. **Delayed-label monitoring:** when an engine fails, compare its past predictions with the true RUL (rolling RMSE). Drift is only a proxy; this measures real degradation and is the right trigger for retraining.
4. **Production hardening:** auth/TLS, per-engine server-side buffers, a minimum-history rule or confidence output (a 5-cycle history gave 110 vs a true 20), alias hot-reload, Postgres + object storage for MLflow, a scheduled monitor with report history and alerts.
5. **CI model-quality gate:** a scheduled workflow with dataset access that retrains and fails if the champion's test RMSE regresses beyond a margin; add coverage, mypy, and image/dependency scanning.
6. **Other subsets:** FD002/FD004 show a large validation-to-test gap that was not investigated (short test histories, regime clustering quality).

## Process notes
- Building phase by phase with a commit after each made every claim traceable to a commit, a doc, and (mostly) a command that reproduces it.
- Negative results were kept in the docs instead of being tuned away; several of the most useful findings (items 2, 5, 6, 10 above) were corrections to my own earlier assumptions.
- Thresholds and policies that depend on real traffic (drift, retrain) are explicitly marked as calibrated on simulation.
