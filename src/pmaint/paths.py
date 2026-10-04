import os
from pathlib import Path

# PMAINT_HOME lets containers (where the package is not under the repo root) set the data root.
ROOT = Path(os.environ.get("PMAINT_HOME") or Path(__file__).resolve().parents[2])
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def dataset_dir(subset: str, window: int = 30, cap: int = 125) -> Path:
    """Default config keeps the plain subset folder; variants get a suffixed folder."""
    if (window, cap) == (30, 125):
        return PROCESSED_DIR / subset
    return PROCESSED_DIR / f"{subset}_w{window}_c{cap}"
