"""Collapse (N, window, F) sensor windows into flat tabular features for tree models."""
import numpy as np


def window_to_tabular(X: np.ndarray, feature_names=None):
    """Per sensor: last value, window mean, window std, and linear-trend slope per cycle.

    Returns (N, 4F) float32 and the column names.
    """
    n, w, f = X.shape
    t = np.arange(w, dtype=np.float32)
    t_c = t - t.mean()
    slope = np.einsum("nwf,w->nf", X, t_c) / np.sum(t_c**2)
    feats = np.concatenate([X[:, -1, :], X.mean(axis=1), X.std(axis=1), slope], axis=1)
    names = feature_names if feature_names is not None else [f"f{i}" for i in range(f)]
    cols = [f"{s}_{n_}" for s in ("last", "mean", "std", "slope") for n_ in names]
    return feats.astype(np.float32), cols
