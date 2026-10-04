"""End-to-end smoke tests of the training entry points on tiny synthetic data (no dataset needed).

They exercise: windows -> model fit -> metrics -> MLflow logging (tmp sqlite store) for XGBoost and
the PyTorch models, so a broken refactor is caught in CI even without C-MAPSS.
"""
import mlflow
import numpy as np
import pytest

from pmaint.training import train_deep, train_xgb
from pmaint.training.tracking import setup_mlflow

WINDOW, N_FEAT = 10, 4


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)

    def split(n):
        X = rng.normal(size=(n, WINDOW, N_FEAT)).astype(np.float32)
        y = np.clip(60 + 20 * X[:, -1, 0] + rng.normal(size=n), 0, 125).astype(np.float32)
        return X, y

    Xtr, ytr = split(300)
    Xva, yva = split(80)
    Xte, yte = split(20)
    pdir = tmp_path / "processed"
    pdir.mkdir()
    np.savez(pdir / "dataset.npz", X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte,
             y_test=yte, features=np.array([f"s_{i}" for i in range(N_FEAT)]))
    (pdir / "preprocessor.joblib").write_bytes(b"placeholder")  # only logged as an artifact

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    for mod in (train_xgb, train_deep):
        monkeypatch.setattr(mod, "ensure_dataset", lambda *a, **k: pdir)
    setup_mlflow("smoke")
    yield
    mlflow.end_run()


def test_xgb_run_logs_metrics_and_model(synthetic):
    res = train_xgb.run("FD001", seed=0, window=WINDOW, params={"n_estimators": 30,
                                                               "early_stopping_rounds": 5})
    m = res["metrics"]
    assert {"val_rmse", "test_rmse", "test_nasa_score", "val_rmse_deg"} <= set(m)
    assert res["pred_test"].shape == (20,) and (res["pred_test"] >= 0).all()
    logged = mlflow.get_run(res["run_id"])
    assert logged.data.params["rul_cap"] == "125" and logged.data.tags["stage"] == "baseline"


@pytest.mark.parametrize("name,kw", [("lstm", {"hidden": 8, "layers": 1}),
                                     ("cnn", {"channels": 8})])
def test_deep_run_trains_and_logs(synthetic, name, kw):
    res = train_deep.run(name, "FD001", seed=0, window=WINDOW, epochs=8, lr=1e-2, batch_size=64,
                         patience=8, model_kw=kw)
    m = res["metrics"]
    assert np.isfinite(m["val_rmse"]) and np.isfinite(m["test_rmse"])
    assert res["pred_val"].shape == (80,) and res["pred_test"].shape == (20,)
    hist = mlflow.MlflowClient().get_metric_history(res["run_id"], "train_loss")
    losses = [h.value for h in sorted(hist, key=lambda h: h.step)]
    assert len(losses) >= 2 and losses[-1] < losses[0]  # the optimiser is actually learning
