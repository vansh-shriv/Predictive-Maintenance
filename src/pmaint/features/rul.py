"""RUL target construction."""
import numpy as np

DEFAULT_RUL_CAP = 125


def cap_rul(rul, cap: int = DEFAULT_RUL_CAP):
    """Piecewise-linear RUL: early life is treated as 'healthy' (constant = cap)."""
    return np.minimum(rul, cap)
