"""XGBoost: python -m pmaint.training.train_xgb --subset FD001 --seeds 3 [--window 30 --cap 125]"""
import argparse

import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor

from pmaint.data.build_dataset import ensure_dataset
from pmaint.features.tabular import window_to_tabular
from pmaint.training.metrics import evaluate, rmse_degrading
from pmaint.training.tracking import setup_mlflow

PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, early_stopping_rounds=50,
)


def run(subset="FD001", seed=42, window=30, cap=125, stage="baseline", params=None) -> dict:
    """Train one XGBoost model inside an MLflow run. Returns metrics, preds and run_id."""
    params = {**PARAMS, **(params or {})}
    pdir = ensure_dataset(subset, window, cap)
    d = np.load(pdir / "dataset.npz")
    names = d["features"].tolist()
    Xtr, cols = window_to_tabular(d["X_train"], names)
    Xva, _ = window_to_tabular(d["X_val"], names)
    Xte, _ = window_to_tabular(d["X_test"], names)

    with mlflow.start_run(run_name=f"xgb-{subset}-w{window}-c{cap}-s{seed}") as r:
        mlflow.set_tags({"model": "xgboost", "subset": subset, "stage": stage})
        mlflow.log_params({**params, "subset": subset, "seed": seed, "n_features": len(cols),
                           "window": window, "rul_cap": cap})
        model = XGBRegressor(**params, random_state=seed, n_jobs=-1)
        model.fit(Xtr, d["y_train"], eval_set=[(Xva, d["y_val"])], verbose=False)

        # val targets are capped RUL; test targets are the true (uncapped) RUL at the last cycle.
        pred_va = model.predict(Xva)
        pred_te = np.clip(model.predict(Xte), 0, None)
        metrics = {f"val_{k}": v for k, v in evaluate(d["y_val"], pred_va).items()}
        metrics.update({f"test_{k}": v for k, v in evaluate(d["y_test"], pred_te).items()})
        metrics["val_rmse_deg"] = rmse_degrading(d["y_val"], pred_va, cap)
        metrics["best_iteration"] = int(model.best_iteration)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model", input_example=Xte[:2])
        mlflow.log_artifact(str(pdir / "preprocessor.joblib"), "preprocessing")
        shown = " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
        print(f"[xgb w{window} c{cap} s{seed}] {shown}")
        return {"metrics": metrics, "pred_val": pred_va, "pred_test": pred_te,
                "run_id": r.info.run_id}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--stage", default="baseline")
    a = ap.parse_args()
    setup_mlflow()
    res = [run(a.subset, s, a.window, a.cap, a.stage)["metrics"] for s in range(a.seeds)]
    for k in ("test_rmse", "test_nasa_score"):
        v = np.array([r[k] for r in res])
        print(f"xgb {a.subset} {k}: {v.mean():.2f} +/- {v.std():.2f} (n={len(v)})")
