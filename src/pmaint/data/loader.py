"""Load raw C-MAPSS files into DataFrames."""
from pathlib import Path

import pandas as pd

from pmaint.paths import RAW_DIR

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _read(path: Path) -> pd.DataFrame:
    # Files are space-separated with trailing whitespace -> use regex sep, drop empty cols.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS
    return df


def load_subset(subset: str = "FD001", raw_dir: Path = RAW_DIR):
    """Return (train, test, rul_test).

    train: full run-to-failure trajectories.
    test: trajectories truncated before failure.
    rul_test: Series indexed by unit with the true RUL at the last test cycle.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    train = _read(raw_dir / f"train_{subset}.txt")
    test = _read(raw_dir / f"test_{subset}.txt")
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None, names=["rul"])
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return train, test, rul["rul"]


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Add linear (uncapped) RUL column: max_cycle - cycle. Capping happens in Phase 2."""
    out = train.copy()
    out["rul"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out
