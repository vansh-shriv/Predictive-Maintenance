"""Build processed arrays for one subset: python -m pmaint.data.build_dataset --subset FD001"""
import argparse

import numpy as np

from pmaint.data.loader import add_train_rul, load_subset
from pmaint.features.preprocessing import Preprocessor
from pmaint.features.rul import DEFAULT_RUL_CAP, cap_rul
from pmaint.features.windows import make_last_windows, make_train_windows, split_units
from pmaint.paths import dataset_dir

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def build(subset="FD001", window=30, cap=DEFAULT_RUL_CAP, val_frac=0.2, seed=42):
    train, test, rul_test = load_subset(subset)
    train = add_train_rul(train)
    train["rul"] = cap_rul(train["rul"], cap)

    train_units, val_units = split_units(train.unit.unique(), val_frac, seed)
    # Fit the scaler on training engines only (no validation/test leakage).
    pre = Preprocessor(N_REGIMES[subset], seed).fit(train[train.unit.isin(train_units)])
    train_s, test_s = pre.transform(train), pre.transform(test)
    cols = pre.feature_cols_

    Xtr, ytr, _ = make_train_windows(train_s[train_s.unit.isin(train_units)], cols, window=window)
    Xva, yva, _ = make_train_windows(train_s[train_s.unit.isin(val_units)], cols, window=window)
    Xte, te_units = make_last_windows(test_s, cols, window=window)
    yte = rul_test.loc[te_units].to_numpy(dtype=np.float32)  # true (uncapped) RUL

    out = dataset_dir(subset, window, cap)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz", X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        X_test=Xte, y_test=yte, train_units=train_units, val_units=val_units,
        test_units=te_units, features=np.array(cols),
    )
    pre.save(out / "preprocessor.joblib")
    print(f"{subset}: train {Xtr.shape} val {Xva.shape} test {Xte.shape} features={len(cols)}")
    return out


def ensure_dataset(subset="FD001", window=30, cap=DEFAULT_RUL_CAP):
    """Return the processed dir for this config, building it first if missing."""
    out = dataset_dir(subset, window, cap)
    if not (out / "dataset.npz").exists():
        build(subset, window, cap)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001", choices=list(N_REGIMES))
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=int, default=DEFAULT_RUL_CAP)
    a = ap.parse_args()
    build(a.subset, a.window, a.cap)
