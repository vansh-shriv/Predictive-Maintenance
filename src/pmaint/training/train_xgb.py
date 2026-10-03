"""XGBoost baseline: python -m pmaint.training.train_xgb --subset FD001"""
import argparse

import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor

from pmaint.features.tabular import window_to_tabular
from pmaint.paths import PROCESSED_DIR
from pmaint.training.metrics import evaluate
from pmaint.training.tracking import setup_mlflow

PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, early_stopping_rounds=50,
)


def main(subset: str = "FD001", seed: int = 42) -> dict:
    d = np.load(PROCESSED_DIR / subset / "dataset.npz")
    names = d["features"].tolist()
    Xtr, cols = window_to_tabular(d["X_train"], names)
    Xva, _ = window_to_tabular(d["X_val"], names)
    Xte, _ = window_to_tabular(d["X_test"], names)

    setup_mlflow()
    with mlflow.start_run(run_name=f"xgb-baseline-{subset}"):
        mlflow.set_tags({"model": "xgboost", "subset": subset, "stage": "baseline"})
        mlflow.log_params({**PARAMS, "subset": subset, "seed": seed, "n_features": len(cols),
                           "window": d["X_train"].shape[1]})

        model = XGBRegressor(**PARAMS, random_state=seed, n_jobs=-1)
        model.fit(Xtr, d["y_train"], eval_set=[(Xva, d["y_val"])], verbose=False)

        # val targets are capped RUL; test targets are the true (uncapped) RUL at the last cycle.
        m_val = evaluate(d["y_val"], model.predict(Xva))
        pred_te = np.clip(model.predict(Xte), 0, None)
        m_te = evaluate(d["y_test"], pred_te)
        metrics = {f"val_{k}": v for k, v in m_val.items()}
        metrics.update({f"test_{k}": v for k, v in m_te.items()})
        metrics["best_iteration"] = int(model.best_iteration)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")
        mlflow.log_artifact(str(PROCESSED_DIR / subset / "preprocessor.joblib"), "preprocessing")

        imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:10]
        print("Top features:", [(c, round(float(v), 3)) for c, v in imp])
        print({k: round(v, 3) for k, v in metrics.items()})
        return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001")
    main(ap.parse_args().subset)
