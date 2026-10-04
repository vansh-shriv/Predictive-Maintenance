"""Turn a drift verdict into an action (retrain / investigate / nothing).

python -m pmaint.monitoring.trigger --current logs/predictions.jsonl
                                    [--new-labelled-data] [--execute]

Exit code: 0 none, 1 investigate, 2 retrain.

Policy (why it is not simply "drift -> retrain"):
  ok / insufficient_data  -> none         (nothing to act on / collect more traffic first)
  warning                 -> investigate  (localised: typically a sensor fault or a prediction
                                           shift. Retraining on faulty readings would bake the
                                           fault into the model -- a human must look first.)
  critical                -> retrain, but only if NEW LABELLED DATA exists. Broad drift means the
                             population changed; retraining on the same old data cannot help.
                             Without new labels the answer is 'investigate'.
No automatic action replaces the champion unconditionally: the pipeline's register step still
applies the validation-metric safety gate.
"""
import argparse
import json
import sys
from pathlib import Path

NONE, INVESTIGATE, RETRAIN = "none", "investigate", "retrain"
EXIT_CODES = {NONE: 0, INVESTIGATE: 1, RETRAIN: 2}


def decide(drift_result: dict, new_labelled_data: bool = False) -> dict:
    level = drift_result["level"]
    if level in ("ok", "insufficient_data"):
        why = ("not enough live traffic to judge" if level == "insufficient_data"
               else "no meaningful drift")
        return {"action": NONE, "reason": why}
    if level == "warning":
        cols = ", ".join(drift_result.get("drifted_features", [])) or "predictions only"
        return {"action": INVESTIGATE,
                "reason": f"localised drift ({cols}); check sensors / data quality "
                          "before any retrain"}
    if new_labelled_data:  # critical
        return {"action": RETRAIN,
                "reason": f"broad drift ({drift_result['drift_share']:.0%} of sensors) and new "
                          "labelled data is available"}
    return {"action": INVESTIGATE,
            "reason": f"broad drift ({drift_result['drift_share']:.0%} of sensors) but no new "
                      "labelled data: retraining on the old data would not help"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", required=True, help="JSONL prediction log")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--new-labelled-data", action="store_true",
                    help="assert that new run-to-failure data has been added to the training set")
    ap.add_argument("--execute", action="store_true",
                    help="if the action is 'retrain', run the pipeline (train + register)")
    a = ap.parse_args()

    from pmaint import pipeline
    from pmaint.monitoring import drift, traffic
    from pmaint.serving.predictor import Predictor
    from pmaint.training.tracking import setup_mlflow

    setup_mlflow()
    result = drift.run(traffic.read_jsonl(Path(a.current)), Predictor.from_registry(), a.subset)
    decision = decide(result, a.new_labelled_data)
    print(json.dumps({"drift": {k: v for k, v in result.items() if k != "scores"},
                      "decision": decision}, indent=2))
    if decision["action"] == RETRAIN and a.execute:
        pipeline.run(("train", "register"), a.subset)
    sys.exit(EXIT_CODES[decision["action"]])
