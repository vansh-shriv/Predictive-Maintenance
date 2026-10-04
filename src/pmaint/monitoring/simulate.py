"""Write simulated prediction logs: python -m pmaint.monitoring.simulate --scenario sensor_bias

Scenarios: normal (live half of the training engines), sensor_bias (calibration drift on 3 sensors),
aged_fleet (engines in the last 15% of life). Output has the same format as the API's log.
"""
import argparse

from pmaint.monitoring import traffic
from pmaint.paths import ROOT
from pmaint.serving.predictor import Predictor
from pmaint.training.tracking import setup_mlflow

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=traffic.SCENARIOS, default="normal")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--cuts", type=int, default=None, help="requests per engine (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    setup_mlflow()
    recs = traffic.generate(Predictor.from_registry(), a.scenario, a.subset, a.cuts, a.seed)
    out = a.out or ROOT / "logs" / f"sim_{a.scenario}.jsonl"
    traffic.write_jsonl(recs, out)
    print(f"wrote {len(recs)} records to {out}")
