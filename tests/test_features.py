import numpy as np
import pandas as pd

from pmaint.features.preprocessing import Preprocessor
from pmaint.features.rul import cap_rul
from pmaint.features.windows import make_last_windows, make_train_windows, split_units


def _df():
    rows = []
    for unit, n in ((1, 5), (2, 8)):
        for c in range(1, n + 1):
            r = dict(unit=unit, cycle=c, setting_1=0.0, setting_2=0.0, setting_3=100.0,
                     s_1=1.0, s_2=float(c), s_3=float(c) * 2, rul=n - c)
            r.update({f"s_{i}": 5.0 for i in range(4, 22)})  # constant sensors
            rows.append(r)
    return pd.DataFrame(rows)


def test_cap_rul():
    assert list(cap_rul(np.array([0, 100, 200]), 125)) == [0, 100, 125]


def test_split_units_disjoint():
    tr, va = split_units(range(1, 101), 0.2)
    assert len(va) == 20 and not set(tr) & set(va)


def test_preprocessor_drops_constants_and_standardises():
    df = _df()
    pre = Preprocessor(1).fit(df)
    assert pre.feature_cols_ == ["s_2", "s_3"]
    out = pre.transform(df)
    assert abs(out["s_2"].mean()) < 1e-6 and abs(out["s_2"].std(ddof=1) - 1) < 1e-6


def test_windows_shapes_and_padding():
    df = _df()
    X, y, u = make_train_windows(df, ["s_2", "s_3"], window=4)
    assert X.shape == (13, 4, 2) and y.shape == (13,)
    assert (X[0] == X[0][0]).all()  # first window is fully padded with row 0
    Xl, ul = make_last_windows(df, ["s_2", "s_3"], window=6)
    assert Xl.shape == (2, 6, 2) and list(ul) == [1, 2]
    assert Xl[0, -1, 0] == 5.0  # last row of unit 1
