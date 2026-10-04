"""Build a sample /predict payload from a C-MAPSS test engine.

python -m pmaint.serving.sample_request --unit 24 [--cycles 80] > payload.json
"""
import argparse
import json

from pmaint.data.loader import load_subset
from pmaint.serving.predictor import RAW_COLS


def build_payload(subset="FD001", unit=1, cycles=None) -> tuple[dict, float]:
    """Return (payload, true_rul_at_last_cycle)."""
    _, test, rul = load_subset(subset)
    g = test[test.unit == unit].sort_values("cycle")
    if cycles:
        g = g.tail(cycles)
    return ({"engine_id": f"{subset}-unit{unit}", "readings": g[RAW_COLS].to_dict("records")},
            float(rul.loc[unit]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--cycles", type=int, default=None, help="keep only the last N cycles")
    a = ap.parse_args()
    payload, true_rul = build_payload(a.subset, a.unit, a.cycles)
    print(json.dumps(payload))
