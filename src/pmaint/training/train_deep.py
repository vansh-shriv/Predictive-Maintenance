"""Train LSTM / 1D-CNN: python -m pmaint.training.train_deep --model lstm --seeds 3 [--window 50 ...]"""
import argparse
import copy
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pmaint.data.build_dataset import ensure_dataset
from pmaint.models.nets import build_model
from pmaint.training.metrics import evaluate, rmse_degrading
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


def run(model_name="lstm", subset="FD001", seed=0, window=30, cap=125, epochs=60, lr=1e-3,
        batch_size=256, patience=10, loss="mse", model_kw=None, stage="deep") -> dict:
    set_seed(seed)
    pdir = ensure_dataset(subset, window, cap)
    d = np.load(pdir / "dataset.npz")
    Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in
                                    ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    model_kw = {**(model_kw or {}), "scale": cap}
    model = build_model(model_name, Xtr.shape[2], **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss() if loss == "mse" else nn.HuberLoss(delta=0.1)  # delta in RUL/cap units
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    params = dict(model=model_name, subset=subset, seed=seed, epochs=epochs, lr=lr, loss=loss,
                  batch_size=batch_size, patience=patience, window=window,
                  n_features=Xtr.shape[2], rul_cap=cap, **model_kw)
    with mlflow.start_run(run_name=f"{model_name}-{subset}-w{window}-s{seed}") as r:
        mlflow.set_tags({"model": model_name, "subset": subset, "stage": stage})
        mlflow.log_params(params)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                # loss in normalised units (RUL / cap) for stable gradients
                l = loss_fn(model(xb) / model.scale, yb / model.scale)
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += l.item() * len(xb)
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

        pred_va = predict(model, Xva)
        pred_te = np.clip(predict(model, Xte), 0, None)
        metrics = {f"val_{k}": v for k, v in evaluate(yva, pred_va).items()}
        metrics.update({f"test_{k}": v for k, v in evaluate(yte, pred_te).items()})
        metrics["val_rmse_deg"] = rmse_degrading(yva, pred_va, cap)
        metrics["epochs_run"] = ep + 1
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model.eval(), name="model", input_example=Xte[:2])
        mlflow.log_artifact(str(pdir / "preprocessor.joblib"), "preprocessing")
        print(f"[{model_name} w{window} s{seed}] " + " ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
        return {"metrics": metrics, "pred_val": pred_va, "pred_test": pred_te, "run_id": r.info.run_id}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--hidden", type=int, default=64, help="LSTM hidden size / CNN channels")
    ap.add_argument("--no-bn", action="store_true", help="CNN without BatchNorm")
    ap.add_argument("--stage", default="deep")
    a = ap.parse_args()

    setup_mlflow()
    kw = ({"hidden": a.hidden, "layers": 2, "dropout": 0.2} if a.model == "lstm"
          else {"channels": a.hidden, "kernel": 5, "dropout": 0.2, "batch_norm": not a.no_bn})
    res = [run(a.model, a.subset, s, a.window, a.cap, a.epochs, a.lr, a.batch_size, a.patience,
               a.loss, kw, a.stage)["metrics"] for s in range(a.seeds)]
    for k in ("test_rmse", "test_nasa_score"):
        v = np.array([r[k] for r in res])
        print(f"{a.model} {a.subset} {k}: {v.mean():.2f} +/- {v.std():.2f} (n={len(v)})")
