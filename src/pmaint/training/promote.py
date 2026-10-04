"""Copy the champion from one MLflow registry to another.

Example: local sqlite -> docker-compose server.

python -m pmaint.training.promote --src sqlite:///mlflow.db --dst http://localhost:5000

The model, its preprocessor artifact, params, metrics and tags are re-logged as a new run on the
destination and registered as `rul-champion` with alias `champion`. Provenance is kept in tags
(`promoted_from_run`, `promoted_from_uri`). Idempotent only in the sense that each call creates a
new version on the destination.
"""
import argparse
import sys
import tempfile

import mlflow
from mlflow import MlflowClient

from pmaint.training.registry import ALIAS, MODEL_NAME
from pmaint.training.tracking import EXPERIMENT


def promote(src: str, dst: str, name: str = MODEL_NAME, alias: str = ALIAS):
    src_client = MlflowClient(tracking_uri=src, registry_uri=src)
    mv = src_client.get_model_version_by_alias(name, alias)
    run = src_client.get_run(mv.run_id)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = src_client.download_artifacts(mv.run_id, "model", tmp)
        pre_path = src_client.download_artifacts(
            mv.run_id, "preprocessing/preprocessor.joblib", tmp)

        mlflow.set_tracking_uri(dst)
        mlflow.set_registry_uri(dst)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"promoted-{run.info.run_name}") as r:
            mlflow.log_params(run.data.params)
            mlflow.log_metrics(run.data.metrics)
            mlflow.set_tags({k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")})
            mlflow.set_tags({"promoted_from_run": mv.run_id, "promoted_from_uri": src})
            mlflow.log_artifact(pre_path, "preprocessing")
            mlflow.log_artifacts(model_dir, "model")
            new_run = r.info.run_id

    dst_client = MlflowClient(tracking_uri=dst, registry_uri=dst)
    new_mv = mlflow.register_model(f"runs:/{new_run}/model", name)
    dst_client.set_registered_model_alias(name, alias, new_mv.version)
    for k, v in mv.tags.items():
        dst_client.set_model_version_tag(name, new_mv.version, k, v)
    print(f"Promoted {name}@{alias}: {src} v{mv.version} -> "
          f"{dst} v{new_mv.version} (run {new_run})")
    return new_mv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    # MLflow prints emoji; Windows consoles default to cp1252 and would crash at end of run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    promote(a.src, a.dst)
