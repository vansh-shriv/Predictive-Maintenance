"""Download and extract the NASA C-MAPSS turbofan dataset into data/raw."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from pmaint.paths import RAW_DIR

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
EXPECTED = [f"{kind}_FD00{i}.txt" for i in range(1, 5) for kind in ("train", "test", "RUL")]


def _extract(zip_path: Path, dest: Path) -> None:
    """Extract txt files, descending into a nested CMAPSSData.zip if present."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".zip"):
                nested = dest / Path(name).name
                nested.write_bytes(zf.read(name))
                _extract(nested, dest)
                nested.unlink()
            elif name.endswith(".txt"):
                (dest / Path(name).name).write_bytes(zf.read(name))


def download(dest: Path = RAW_DIR, force: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if not force and all((dest / f).exists() for f in EXPECTED):
        print(f"C-MAPSS already present in {dest}")
        return dest
    zip_path = dest / "cmapss_download.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    _extract(zip_path, dest)
    zip_path.unlink()
    missing = [f for f in EXPECTED if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"Missing files after extraction: {missing}")
    print(f"Extracted {len(EXPECTED)} files to {dest}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    download(force=ap.parse_args().force)
