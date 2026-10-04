"""Replay simulated traffic against a running API.

python -m pmaint.monitoring.replay --scenario normal

Sends the same truncated histories that `simulate` scores in-process, but over HTTP to
POST /predict, so the service writes its own prediction log (PMAINT_PRED_LOG). Useful as a
stack smoke test.
"""
import argparse

import httpx

from pmaint.monitoring import traffic
from pmaint.serving.predictor import RAW_COLS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--scenario", choices=traffic.SCENARIOS, default="normal")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--cuts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N requests")
    a = ap.parse_args()

    n = failed = 0
    with httpx.Client(base_url=a.url, timeout=30) as c:
        for eid, hist in traffic.histories(a.scenario, a.subset, a.cuts, a.seed):
            body = {"engine_id": eid, "readings": hist[RAW_COLS].to_dict("records")}
            r = c.post("/predict", json=body)
            n += 1
            failed += r.status_code != 200
            if a.limit and n >= a.limit:
                break
    print(f"sent {n} requests, {failed} failed")
