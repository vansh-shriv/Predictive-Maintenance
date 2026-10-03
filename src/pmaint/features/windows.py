"""Sliding-window construction and engine-wise splitting."""
import numpy as np
import pandas as pd


def split_units(units, val_frac: float = 0.2, seed: int = 42):
    """Split engine ids (never rows) into train / validation sets."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    return np.sort(units[n_val:]), np.sort(units[:n_val])


def _pad_front(arr: np.ndarray, window: int) -> np.ndarray:
    """Left-pad short histories by repeating the first row."""
    if len(arr) >= window:
        return arr[-window:]
    pad = np.repeat(arr[:1], window - len(arr), axis=0)
    return np.vstack([pad, arr])


def make_train_windows(df: pd.DataFrame, feature_cols, target_col: str = "rul", window: int = 30):
    """One window ending at every cycle of every engine (short starts are front-padded).

    Returns X (N, window, F) float32, y (N,) float32, units (N,) int.
    """
    X, y, u = [], [], []
    for unit, g in df.groupby("unit"):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        tgt = g[target_col].to_numpy(dtype=np.float32)
        for i in range(len(g)):
            X.append(_pad_front(feats[: i + 1], window))
            y.append(tgt[i])
            u.append(unit)
    return np.stack(X), np.array(y, dtype=np.float32), np.array(u)


def make_last_windows(df: pd.DataFrame, feature_cols, window: int = 30):
    """Final window of every engine -- the official C-MAPSS test protocol.

    Returns X (n_units, window, F) and the unit ids in the same order.
    """
    X, u = [], []
    for unit, g in df.groupby("unit"):
        X.append(_pad_front(g[feature_cols].to_numpy(dtype=np.float32), window))
        u.append(unit)
    return np.stack(X), np.array(u)
