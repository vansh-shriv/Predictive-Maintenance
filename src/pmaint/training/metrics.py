"""Evaluation metrics for RUL prediction."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric C-MAPSS score (lower is better). Late predictions (d > 0) cost more.

    s = sum(exp(-d/13) - 1) for d < 0, sum(exp(d/10) - 1) for d >= 0, with d = pred - true.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def evaluate(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}


def rmse_degrading(y_true_capped, y_pred, cap: float) -> float:
    """RMSE on windows already in the degradation phase (capped RUL < cap).

    Plain validation RMSE is dominated by the flat early-life plateau, which the test protocol
    (one late-ish window per engine) barely contains. This is a closer validation proxy.
    """
    y_true_capped = np.asarray(y_true_capped)
    m = y_true_capped < cap
    return rmse(y_true_capped[m], np.asarray(y_pred)[m])
