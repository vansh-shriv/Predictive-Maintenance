"""One reproducible entry point for the whole flow.

python -m pmaint.pipeline                         # data, train, register  (full: with LSTM)
python -m pmaint.pipeline --quick                 # 1 seed, XGBoost only (smoke run, ~1 min)
python -m pmaint.pipeline --steps data,train      # any comma-separated subset, in canonical order
python -m pmaint.pipeline --steps promote --promote-to http://localhost:5000

Steps (always executed in this order):
  data      download C-MAPSS (idempotent) and build the processed arrays for every window used
  train     multi-seed "final" runs of every candidate config, tagged with a pipeline id
  register  rank this pipeline's candidates by mean validation metric; register as the champion
            only if it beats the current one by MIN_GAIN (so reruns are idempotent)
  promote   copy the champion into another MLflow registry (e.g. the docker-compose server)

Candidates reproduce the Phase 5 selection pool: XGBoost windows 30 and 50, plus LSTM window 50.
"""
import argparse
import sys
import time

import mlflow

from pmaint.data import build_dataset, download
from pmaint.training import promote, registry, train_deep, train_xgb
from pmaint.training.tracking import setup_mlflow

STEPS = ("data", "train", "register", "promote")
DEFAULT_STEPS = ("data", "train", "register")
XGB_WINDOWS = (30, 50)
LSTM_WINDOW = 50
LSTM_KW = {"hidden": 64, "layers": 2, "dropout": 0.2}
CAP = 125


def parse_steps(spec: str) -> list[str]:
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = wanted - set(STEPS)
    if unknown:
        raise ValueError(f"unknown step(s) {sorted(unknown)}; choose from {STEPS}")
    return [s for s in STEPS if s in wanted]  # canonical order regardless of input order


def new_pipeline_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run(steps=DEFAULT_STEPS, subset="FD001", seeds=3, with_deep=True, pipeline_id=None,
        promote_to=None, force=False) -> dict:
    pipeline_id = pipeline_id or new_pipeline_id()
    result = {"pipeline_id": pipeline_id, "steps": list(steps)}
    print(f"== pipeline {pipeline_id}: steps={list(steps)} subset={subset} seeds={seeds} "
          f"deep={with_deep}")

    if "data" in steps:
        print("-- data")
        download.download()
        for w in sorted({*XGB_WINDOWS, *([LSTM_WINDOW] if with_deep else [])}):
            build_dataset.ensure_dataset(subset, w, CAP)

    if "train" in steps or "register" in steps:
        setup_mlflow()

    if "train" in steps:
        print("-- train")
        tags = {"pipeline_id": pipeline_id}
        for w in XGB_WINDOWS:
            for s in range(seeds):
                train_xgb.run(subset, s, w, CAP, stage="final", tags=tags)
        if with_deep:
            for s in range(seeds):
                train_deep.run("lstm", subset, s, LSTM_WINDOW, CAP, model_kw=dict(LSTM_KW),
                               stage="final", tags=tags)

    if "register" in steps:
        print("-- register")
        # With a train step in this invocation, only this pipeline's runs are candidates.
        pid = pipeline_id if "train" in steps else None
        mv = registry.register_best(subset, CAP, pipeline_id=pid, force=force)
        result["registered_version"] = None if mv is None else mv.version

    if "promote" in steps:
        if not promote_to:
            raise ValueError("--promote-to is required for the promote step")
        print("-- promote")
        setup_mlflow()
        promote.promote(mlflow.get_tracking_uri(), promote_to)

    return result


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(DEFAULT_STEPS))
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-deep", action="store_true", help="XGBoost candidates only")
    ap.add_argument("--quick", action="store_true", help="1 seed and no deep model (smoke run)")
    ap.add_argument("--pipeline-id", default=None)
    ap.add_argument("--promote-to", default=None,
                    help="destination MLflow URI for the promote step")
    ap.add_argument("--force", action="store_true", help="bypass the champion safety gate")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # MLflow prints emoji; avoid cp1252 crashes on Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return run(parse_steps(a.steps), a.subset, 1 if a.quick else a.seeds,
               not (a.quick or a.skip_deep), a.pipeline_id, a.promote_to, a.force)


if __name__ == "__main__":
    main()
