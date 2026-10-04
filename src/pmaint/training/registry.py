"""Pick the best *configuration* by mean validation metric over seeds and register it as champion.

python -m pmaint.training.registry [--subset FD001] [--cap 125] [--metric val_rmse_deg]
                                   [--stages final] [--dry-run]

Runs are grouped by identical params (seed excluded) and ranked by the mean validation metric, so
a lucky single seed cannot win. The lowest-seed run of the winning config is registered. Only runs
with the same RUL cap are comparable on validation metrics. Test metrics are attached to the model
version for reporting but never used to select.
"""
import argparse

import mlflow
from mlflow import MlflowClient

from pmaint.training.tracking import EXPERIMENT, setup_mlflow

MODEL_NAME = "rul-champion"
ALIAS = "champion"


def candidates(subset: str, cap: int, metric: str, stages=("final",)):
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT], output_format="pandas",
        filter_string=f"params.subset = '{subset}' and params.rul_cap = '{cap}' "
                      "and attributes.status = 'FINISHED'",
    )
    col = f"metrics.{metric}"
    if runs.empty or col not in runs:
        return runs.iloc[0:0]
    runs = runs[runs["tags.stage"].isin(stages)].dropna(subset=[col]).copy()
    if runs.empty:
        return runs
    keys = [c for c in runs.columns if c.startswith("params.") and c != "params.seed"]
    runs["config"] = [
        "|".join(f"{k}={row[k]}" for k in keys) for _, row in runs.iterrows()
    ]
    g = runs.groupby("config")
    runs["cfg_mean"] = g[col].transform("mean")
    runs["cfg_test_mean"] = g["metrics.test_rmse"].transform("mean")
    runs["cfg_n"] = g[col].transform("size")
    runs["seed"] = runs["params.seed"].astype(int)
    return runs.sort_values(["cfg_mean", "seed"])


def register_best(subset="FD001", cap=125, metric="val_rmse_deg", dry_run=False,
                  stages=("final",)):
    ranked = candidates(subset, cap, metric, stages)
    if ranked.empty:
        raise SystemExit("No candidate runs found; run the final-stage training first.")
    table = ranked.drop_duplicates("config")[
        ["tags.model", "params.window", "cfg_n", "cfg_mean", "cfg_test_mean"]]
    table.columns = ["model", "window", "n_seeds", f"mean_{metric}", "mean_test_rmse"]
    print(table.to_string(index=False))
    best = ranked.iloc[0]
    print(f"\nChampion config: {best['tags.model']} window={best['params.window']} "
          f"mean {metric}={best['cfg_mean']:.3f}; run {best['run_id']} (seed {best['seed']})")
    if dry_run:
        return None
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    for k, v in {"selected_by": f"mean {metric}", "window": best["params.window"],
                 "model_type": best["tags.model"],
                 "test_rmse_config_mean": f"{best['cfg_test_mean']:.3f}"}.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
    print(f"Registered {MODEL_NAME} v{mv.version} with alias '{ALIAS}'")
    return mv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--metric", default="val_rmse_deg")
    ap.add_argument("--stages", nargs="+", default=["final"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    setup_mlflow()
    register_best(a.subset, a.cap, a.metric, a.dry_run, tuple(a.stages))
