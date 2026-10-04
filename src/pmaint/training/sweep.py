"""One-factor-at-a-time sweep around the default configs, all logged to MLflow (stage='sweep').

python -m pmaint.training.sweep [--only xgb|lstm|cnn]

Selection must use *validation* RMSE, and only among runs with the same RUL cap (cap changes the
validation targets, so val RMSE is not comparable across caps). Test metrics are for reporting.
"""
import argparse

from pmaint.training import train_deep, train_xgb
from pmaint.training.tracking import setup_mlflow

SUBSET = "FD001"

XGB_CONFIGS = [  # (window, cap); first entry of each list is the default config
    (30, 125), (20, 125), (50, 125), (30, 100), (30, 150),
]
LSTM_CONFIGS = [
    dict(), dict(window=50), dict(kw=dict(hidden=128)), dict(loss="huber"), dict(lr=3e-4),
]
CNN_CONFIGS = [
    dict(), dict(kw=dict(batch_norm=False)), dict(window=50, kw=dict(batch_norm=False)),
]


def main(only=None):
    setup_mlflow()
    if only in (None, "xgb"):
        for w, c in XGB_CONFIGS:
            train_xgb.run(SUBSET, seed=0, window=w, cap=c, stage="sweep")
    if only in (None, "lstm"):
        for cfg in LSTM_CONFIGS:
            kw = {"hidden": 64, "layers": 2, "dropout": 0.2, **cfg.get("kw", {})}
            train_deep.run("lstm", SUBSET, 0, cfg.get("window", 30), 125, lr=cfg.get("lr", 1e-3),
                           loss=cfg.get("loss", "mse"), model_kw=kw, stage="sweep")
    if only in (None, "cnn"):
        for cfg in CNN_CONFIGS:
            kw = {"channels": 64, "kernel": 5, "dropout": 0.2, **cfg.get("kw", {})}
            train_deep.run("cnn", SUBSET, 0, cfg.get("window", 30), 125, model_kw=kw, stage="sweep")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["xgb", "lstm", "cnn"])
    main(ap.parse_args().only)
