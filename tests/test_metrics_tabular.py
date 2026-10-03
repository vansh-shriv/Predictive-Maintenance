import numpy as np
import pytest

from pmaint.features.tabular import window_to_tabular
from pmaint.training.metrics import nasa_score, rmse


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_nasa_score_is_asymmetric_and_zero_when_perfect():
    assert nasa_score([50], [50]) == 0.0
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # late costs more
    assert nasa_score([50], [60]) == pytest.approx(np.exp(1) - 1)


def test_window_to_tabular():
    X = np.zeros((2, 5, 3), dtype=np.float32)
    X[0, :, 0] = np.arange(5)  # perfect +1 slope on sensor 0
    feats, cols = window_to_tabular(X, ["a", "b", "c"])
    assert feats.shape == (2, 12) and len(cols) == 12
    assert feats[0, cols.index("last_a")] == 4
    assert feats[0, cols.index("mean_a")] == 2
    assert feats[0, cols.index("slope_a")] == pytest.approx(1.0)
    assert feats[1].sum() == 0
