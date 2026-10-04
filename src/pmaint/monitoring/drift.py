"""Drift monitoring with Evidently.

python -m pmaint.monitoring.drift --current logs/predictions.jsonl [--html reports/drift.html]

Compares the model's live inputs (last raw reading per request, scaled with the *training*
preprocessor so operating regimes are normalised) and its predictions against a reference set
built from held-out validation engines. Exit code: 0 ok, 1 warning, 2 critical -- usable from cron/CI.
There are no ground-truth labels in production, so drift is a *proxy* for model degradation.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from pmaint.serving.predictor import RAW_COLS, Predictor
from pmaint.monitoring import traffic
from pmaint.training.tracking import setup_mlflow

# Alert policy (see docs/phases/PHASE_07_drift_monitoring.md for the rationale).
CRITICAL_SHARE = 0.30  # fraction of monitored columns drifted -> critical
WARNING_MIN_COLS = 1   # drifted feature columns -> warning (an effect-size test has few false hits)
DRIFT_METHOD = "wasserstein"  # effect size (in reference-std units), not a p-value
DRIFT_THRESHOLD = 0.6   # feature column 'drifted' if normalised Wasserstein distance exceeds this
PREDICTION_THRESHOLD = 0.2  # same, for predicted_rul (its distances are naturally smaller)
MIN_ROWS = 50          # below this the tests are too weak to trust


def to_frame(records: list[dict], predictor: Predictor) -> pd.DataFrame:
    """Log records -> scaled sensor values + prediction (columns Evidently monitors)."""
    raw = pd.DataFrame([r["last_reading"] for r in records])[RAW_COLS]
    scaled = predictor.pre.transform(raw)[predictor.feature_names]
    scaled["predicted_rul"] = [r["predicted_rul"] for r in records]
    return scaled


def analyse(reference: pd.DataFrame, current: pd.DataFrame, feature_cols: list[str],
            html_path: Path | None = None) -> dict:
    cols = feature_cols + ["predicted_rul"]
    dd = DataDefinition(numerical_columns=cols)
    snap = Report([DataDriftPreset(num_method=DRIFT_METHOD, num_threshold=DRIFT_THRESHOLD,
                    per_column_threshold={"predicted_rul": PREDICTION_THRESHOLD})]).run(
        Dataset.from_pandas(current[cols], data_definition=dd),
        Dataset.from_pandas(reference[cols], data_definition=dd),
    )
    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(html_path))

    drifted, scores = [], {}
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("ValueDrift"):
            col, v, thr = cfg["column"], m["value"], cfg["threshold"]
            scores[col] = v
            is_drift = v < thr if "p_value" in cfg["method"] else v > thr
            if is_drift:
                drifted.append(col)

    feat_drifted = [c for c in drifted if c != "predicted_rul"]
    prediction_drift = "predicted_rul" in drifted
    share = len(feat_drifted) / len(feature_cols)
    if len(current) < MIN_ROWS:
        level = "insufficient_data"
    elif share >= CRITICAL_SHARE:
        level = "critical"
    elif len(feat_drifted) >= WARNING_MIN_COLS or prediction_drift:
        level = "warning"
    else:
        level = "ok"
    return {"level": level, "n_reference": len(reference), "n_current": len(current),
            "drifted_features": feat_drifted, "drift_share": round(share, 3),
            "prediction_drift": prediction_drift,
            "scores": {k: float(f"{v:.3g}") for k, v in scores.items()}}


def run(current_records: list[dict], predictor: Predictor, subset="FD001",
        html_path: Path | None = None) -> dict:
    reference = to_frame(traffic.generate(predictor, "reference", subset), predictor)
    current = to_frame(current_records, predictor)
    return analyse(reference, current, predictor.feature_names, html_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", required=True, help="JSONL prediction log")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--html", default=None, help="write the Evidently HTML report here")
    a = ap.parse_args()
    setup_mlflow()
    result = run(traffic.read_jsonl(Path(a.current)), Predictor.from_registry(), a.subset, a.html)
    print({k: v for k, v in result.items() if k != "scores"})
    sys.exit({"ok": 0, "warning": 1, "critical": 2}.get(result["level"], 3))
