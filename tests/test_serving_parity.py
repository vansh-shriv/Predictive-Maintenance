"""Training/serving skew guard: the API pipeline must reproduce the offline pipeline exactly."""
import numpy as np
import pytest

from pmaint.paths import PROCESSED_DIR, RAW_DIR, ROOT

pytestmark = [
    pytest.mark.data,
    pytest.mark.registry,
    pytest.mark.skipif(
        not ((ROOT / "mlflow.db").exists() and (RAW_DIR / "test_FD001.txt").exists()
             and (PROCESSED_DIR / "FD001_w50_c125" / "dataset.npz").exists()),
        reason="needs registered champion + FD001 data (run Phases 1-5)",
    ),
]


def test_api_pipeline_matches_offline_test_predictions():
    from pmaint.data.loader import load_subset
    from pmaint.features.tabular import window_to_tabular
    from pmaint.serving.predictor import RAW_COLS, Predictor

    p = Predictor.from_registry()
    d = np.load(PROCESSED_DIR / f"FD001_w{p.window}_c125" / "dataset.npz")
    feats, _ = window_to_tabular(d["X_test"], d["features"].tolist())
    offline = np.clip(p.model.predict(feats), 0, None)

    _, test, _ = load_subset("FD001")
    for i, unit in enumerate(d["test_units"][:10]):
        g = test[test.unit == unit].sort_values("cycle")
        online, _ = p.predict(g[RAW_COLS])
        # Inputs agree to ~1e-7 (float32 rounding from batch-vs-single reductions), but a tree split
        # sitting on a threshold can flip, giving differences of ~0.1 cycle. Real skew is >> 1.
        assert online == pytest.approx(float(offline[i]), abs=0.5)
