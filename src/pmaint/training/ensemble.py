"""Average XGBoost and LSTM predictions.

python -m pmaint.training.ensemble --xgb-window 50 --lstm-window 30

Both component models are (re)trained for the given seed, then predictions are averaged. Val and
test rows align across window sizes (one window per cycle / one per test engine), so mixing
windows is valid. Evaluation only -- the champion registry holds single, servable models.
"""
import argparse

import mlflow
import numpy as np

from pmaint.data.build_dataset import ensure_dataset
from pmaint.training import train_deep, train_xgb
from pmaint.training.metrics import evaluate, rmse_degrading
from pmaint.training.tracking import setup_mlflow


def main(subset, seeds, xgb_window, lstm_window, lstm_kw, w_xgb):
    setup_mlflow()
    all_metrics = []
    for seed in seeds:
        x = train_xgb.run(subset, seed, xgb_window, 125, stage="ensemble-component")
        lr = train_deep.run("lstm", subset, seed, lstm_window, 125, model_kw=lstm_kw,
                            stage="ensemble-component")
        d = np.load(ensure_dataset(subset, 30, 125) / "dataset.npz")  # same engines/targets
        pv = w_xgb * x["pred_val"] + (1 - w_xgb) * lr["pred_val"]
        pt = w_xgb * x["pred_test"] + (1 - w_xgb) * lr["pred_test"]
        with mlflow.start_run(run_name=f"ensemble-{subset}-s{seed}"):
            mlflow.set_tags({"model": "ensemble", "subset": subset, "stage": "ensemble"})
            mlflow.log_params({"subset": subset, "seed": seed, "w_xgb": w_xgb, "rul_cap": 125,
                               "xgb_window": xgb_window, "lstm_window": lstm_window,
                               "xgb_run": x["run_id"], "lstm_run": lr["run_id"]})
            m = {f"val_{k}": v for k, v in evaluate(d["y_val"], pv).items()}
            m.update({f"test_{k}": v for k, v in evaluate(d["y_test"], pt).items()})
            m["val_rmse_deg"] = rmse_degrading(d["y_val"], pv, 125)
            mlflow.log_metrics(m)
        print(f"[ensemble s{seed}] " + " ".join(f"{k}={v:.2f}" for k, v in m.items()))
        all_metrics.append(m)
    for k in ("test_rmse", "test_nasa_score", "val_rmse_deg"):
        v = np.array([m[k] for m in all_metrics])
        print(f"ensemble {subset} {k}: {v.mean():.2f} +/- {v.std():.2f} (n={len(v)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--xgb-window", type=int, default=30)
    ap.add_argument("--lstm-window", type=int, default=30)
    ap.add_argument("--lstm-hidden", type=int, default=64)
    ap.add_argument("--w-xgb", type=float, default=0.5)
    a = ap.parse_args()
    main(a.subset, range(a.seeds), a.xgb_window, a.lstm_window,
         {"hidden": a.lstm_hidden, "layers": 2, "dropout": 0.2}, a.w_xgb)
