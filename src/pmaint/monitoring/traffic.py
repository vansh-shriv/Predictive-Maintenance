"""Simulated production traffic. Produces JSONL records in exactly the format the API logs.

Reference and "normal" traffic use disjoint halves of the FD001 training engines (see generate());
drift scenarios perturb or re-sample the "normal" half. Each record comes from a real Predictor call
on a truncated history (a random cut point per request), mirroring POST /predict.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmaint.data.loader import load_subset
from pmaint.serving.predictor import RAW_COLS, Predictor

SCENARIOS = ("normal", "sensor_bias", "aged_fleet")
BIAS_SENSORS = ("s_4", "s_11", "s_12")  # three of the most important sensors in the champion
BIAS_SIGMAS = 1.5  # shift in training-set standard deviations
MIN_HISTORY = 30


def _record(predictor: Predictor, engine_id: str, hist: pd.DataFrame) -> dict:
    rul, _ = predictor.predict(hist[RAW_COLS])
    return {"ts": time.time(), "engine_id": engine_id, "cycles": len(hist),
            "predicted_rul": rul, "model_version": predictor.meta["version"],
            "last_reading": hist[RAW_COLS].iloc[-1].astype(float).to_dict()}


def _cuts(rng, length, n, lo_frac=0.0):
    lo = max(MIN_HISTORY, int(np.ceil(lo_frac * length)))
    if lo >= length:
        return [length] * n
    return rng.integers(lo, length + 1, size=n).tolist()


def histories(scenario: str = "normal", subset: str = "FD001",
              cuts_per_engine: int | None = None, seed: int = 0):
    """Yield (engine_id, truncated raw history DataFrame) for one scenario."""
    if scenario not in SCENARIOS + ("reference",):
        raise ValueError(f"scenario must be one of {SCENARIOS + ('reference',)}")
    rng = np.random.default_rng(seed)
    train, _, _ = load_subset(subset)
    lo_frac = 0.0

    # Reference and "current" traffic are drawn from the SAME population (full-life engines with
    # uniformly random cut points) but from disjoint engines: unit % 5 < 3 -> reference, else live.
    # Test engines are deliberately not used: they are truncated early in life, so their life-stage
    # mix differs from the reference and produces a (correct but uninteresting) drift alarm.
    in_ref = train.unit % 5 < 3
    if scenario == "reference":
        df = train[in_ref]
        cuts_per_engine = cuts_per_engine or 3
    else:
        df = train[~in_ref]
        cuts_per_engine = cuts_per_engine or 5
        if scenario == "aged_fleet":  # engines in the last 15% of their life
            lo_frac = 0.85
    units = df.unit.unique()

    if scenario == "sensor_bias":
        sigma = train[list(BIAS_SENSORS)].std()
        df = df.copy()
        for s in BIAS_SENSORS:
            df[s] = df[s] + BIAS_SIGMAS * sigma[s]

    for u in units:
        g = df[df.unit == u].sort_values("cycle")
        for cut in _cuts(rng, len(g), cuts_per_engine, lo_frac):
            yield f"{subset}-{scenario}-u{u}", g.iloc[:cut]


def generate(predictor: Predictor, scenario: str = "normal", subset: str = "FD001",
             cuts_per_engine: int | None = None, seed: int = 0) -> list[dict]:
    """Return log records for one scenario (predictions made in-process)."""
    return [_record(predictor, eid, hist)
            for eid, hist in histories(scenario, subset, cuts_per_engine, seed)]


def write_jsonl(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
