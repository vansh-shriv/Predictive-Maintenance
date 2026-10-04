"""Train LSTM / 1D-CNN: python -m pmaint.training.train_deep --model lstm --subset FD001 --seeds 3"""
import argparse
import copy
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pmaint.features.rul import DEFAULT_RUL_CAP
from pmaint.models.nets import build_model
from pmaint.paths import PROCESSED_DIR
from pmaint.training.metrics import evaluate
from pmaint.training.tracking import setup_mlflow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X, batch=2048) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)


def run(model_name, subset, seed, epochs, lr, batch_size, patience, hidden_kw) -> dict:
    set_seed(seed)
    d = np.load(PROCESSED_DIR / subset / "dataset.npz")
    Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in
                                    ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    model = build_model(model_name, Xtr.shape[2], **hidden_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    params = dict(model=model_name, subset=subset, seed=seed, epochs=epochs, lr=lr,
                  batch_size=batch_size, patience=patience, window=Xtr.shape[1],
                  n_features=Xtr.shape[2], rul_cap=DEFAULT_RUL_CAP, **hidden_kw)
    with mlflow.start_run(run_name=f"{model_name}-{subset}-s{seed}"):
        mlflow.set_tags({"model": model_name, "subset": subset, "stage": "deep"})
        mlflow.log_params(params)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                # loss in normalised units (RUL / cap) for stable gradients
                loss = loss_fn(model(xb) / model.scale, yb / model.scale)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(xb)
            val_rmse = evaluate(yva, predict(model, Xva))["rmse"]
            sched.step(val_rmse)
            mlflow.log_metrics({"train_loss": tot / len(Xtr), "val_rmse_epoch": val_rmse}, step=ep)
            if val_rmse < best - 1e-4:
                best, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)

        m_val = evaluate(yva, predict(model, Xva))
        m_te = evaluate(yte, np.clip(predict(model, Xte), 0, None))
        metrics = {f"val_{k}": v for k, v in m_val.items()}
        metrics.update({f"test_{k}": v for k, v in m_te.items()})
        metrics["epochs_run"] = ep + 1
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model.eval(), name="model", input_example=Xte[:2])
        mlflow.log_artifact(str(PROCESSED_DIR / subset / "preprocessor.joblib"), "preprocessing")
        print(f"[{model_name} s{seed}] " + " ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
        return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10)
    a = ap.parse_args()

    setup_mlflow()
    hidden = {"hidden": 64, "layers": 2, "dropout": 0.2} if a.model == "lstm" else {
        "channels": 64, "kernel": 5, "dropout": 0.2}
    res = [run(a.model, a.subset, s, a.epochs, a.lr, a.batch_size, a.patience, hidden)
           for s in range(a.seeds)]
    for k in ("test_rmse", "test_nasa_score"):
        v = np.array([r[k] for r in res])
        print(f"{a.model} {a.subset} {k}: {v.mean():.2f} +/- {v.std():.2f} (n={len(v)})")
