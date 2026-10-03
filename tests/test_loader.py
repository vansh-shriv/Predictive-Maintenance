import pytest

from pmaint.data.loader import COLUMNS, add_train_rul, load_subset
from pmaint.paths import RAW_DIR

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "train_FD001.txt").exists(), reason="run pmaint.data.download first"
)


def test_fd001_shapes():
    train, test, rul = load_subset("FD001")
    assert list(train.columns) == COLUMNS
    assert train.unit.nunique() == 100 and test.unit.nunique() == 100
    assert len(rul) == 100


def test_train_rul_ends_at_zero():
    train, _, _ = load_subset("FD001")
    out = add_train_rul(train)
    assert (out.groupby("unit").rul.min() == 0).all()
    assert out.rul.max() == out.groupby("unit").cycle.max().max() - 1
