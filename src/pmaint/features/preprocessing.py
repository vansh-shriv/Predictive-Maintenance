"""Sensor selection + per-operating-regime scaling. Fit on train only, reuse everywhere."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pmaint.data.loader import SENSOR_COLS, SETTING_COLS

REGIME_SETTINGS = SETTING_COLS  # all 3 settings define the operating regime


class Preprocessor:
    """Drops constant sensors, then standardises each sensor within its operating regime.

    n_regimes=1 (FD001/FD003) is plain global standardisation; n_regimes=6 (FD002/FD004)
    clusters the settings with KMeans and scales per cluster.
    """

    def __init__(self, n_regimes: int = 1, seed: int = 42):
        self.n_regimes = n_regimes
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        self.feature_cols_ = [c for c in SENSOR_COLS if train[c].std() > 1e-6]
        if self.n_regimes > 1:
            self.kmeans_ = KMeans(self.n_regimes, n_init=10, random_state=self.seed).fit(
                train[REGIME_SETTINGS]
            )
        reg = self._regimes(train)
        feats = train[self.feature_cols_].to_numpy(dtype=float)
        self.mean_, self.std_ = {}, {}
        for r in np.unique(reg):
            x = feats[reg == r]
            std = x.std(axis=0, ddof=1) if len(x) > 1 else np.ones(x.shape[1])
            self.mean_[r] = x.mean(axis=0)
            self.std_[r] = np.where(std < 1e-6, 1.0, std)
        return self

    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(df[REGIME_SETTINGS])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        reg = self._regimes(df)
        x = df[self.feature_cols_].to_numpy(dtype=float, copy=True)
        for r in np.unique(reg):
            m = reg == r
            x[m] = (x[m] - self.mean_[r]) / self.std_[r]
        out[self.feature_cols_] = x
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # Persistence: reused by the serving API in Phase 6.
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)
