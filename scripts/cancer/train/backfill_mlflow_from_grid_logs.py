#!/usr/bin/env python3
"""
Backfill MLflow runs from grid_results/*/logs (train.log + eval.log).

Use when jobs ran offline or MLflow tracking URI was unreachable (e.g. SSH tunnel down).
Creates the same metric keys as live training where possible:
  - train stage (ct_iql_em): best/outer_iter, best/val_mae_uns
  - eval stage (eval): eval/tau{T}/mae_uns, mae_norm, rmse_uns, rmse_norm

Example (on ThinkStation A, mlflow server on :5000):
  python scripts/cancer/train/backfill_mlflow_from_grid_logs.py \\
    --grid-root grid_results/em_iql_grid_her0/gamma_4 \\
    --experiment em_iql_grid_her0 \\
    --mlflow-uri http://127.0.0.1:5000 \\
    --gamma 4 \\
    --dry-run

  # apply
  python scripts/cancer/train/backfill_mlflow_from_grid_logs.py \\
    --grid-root grid_results/em_iql_grid_her0/gamma_4 \\
    --experiment em_iql_grid_her0 \\
    --mlflow-uri http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

TAG_DIR = re.compile(r"^(.+)_seed(\d+)$")
RE_EVAL_MAE = re.compile(
    r"MAE normalized:\s*([\d.eE+-]+)\s*\|\s*MAE unscaled:\s*([\d.eE+-]+)"
    r"(?:\s*\|\s*RMSE unscaled:\s*([\d.eE+-]+))?"
)
RE_TRAIN_DONE = re.compile(
    r"EM training done\.\s+best_outer=(\d+)\s+best_\w+=([\d.eE+-]+)"
)
RE_GLOBAL_RMSE = re.compile(
    r"Global RMSE on stacked batches \(normalized space\):\s*([\d.eE+-]+)"
)


@dataclass
class ParsedJob:
    tag: str
    combo_id: str
    seed: str
    eval_mae_norm: Optional[float]
    eval_mae_uns: Optional[float]
    eval_rmse_uns: Optional[float]
    eval_rmse_norm: Optional[float]
    best_outer: Optional[int]
    best_val_mae_uns: Optional[float]


def _parse_tag_dir(name: str) -> Optional[Tuple[str, str]]:
    m = TAG_DIR.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_job(log_dir: Path) -> Optional[ParsedJob]:
    tag = log_dir.name
    parsed = _parse_tag_dir(tag)
    if parsed is None:
        return None
    combo_id, seed = parsed

    eval_mae_norm = eval_mae_uns = eval_rmse_uns = eval_rmse_norm = None
    eval_log = log_dir / "eval.log"
    if eval_log.is_file():
        text = eval_log.read_text(errors="replace")
        m = RE_EVAL_MAE.search(text)
        if m:
            eval_mae_norm = float(m.group(1))
            eval_mae_uns = float(m.group(2))
            if m.group(3):
                eval_rmse_uns = float(m.group(3))
        rm = RE_GLOBAL_RMSE.search(text)
        if rm:
            eval_rmse_norm = float(rm.group(1))

    best_outer = best_val = None
    train_log = log_dir / "train.log"
    if train_log.is_file():
        text = train_log.read_text(errors="replace")
        m = RE_TRAIN_DONE.search(text)
        if m:
            best_outer = int(m.group(1))
            best_val = float(m.group(2))

    if eval_mae_uns is None and best_val is None:
        return None

    return ParsedJob(
        tag=tag,
        combo_id=combo_id,
        seed=seed,
        eval_mae_norm=eval_mae_norm,
        eval_mae_uns=eval_mae_uns,
        eval_rmse_uns=eval_rmse_uns,
        eval_rmse_norm=eval_rmse_norm,
        best_outer=best_outer,
        best_val_mae_uns=best_val,
    )


def _run_exists(
    client,
    experiment_id: str,
    *,
    combo_id: str,
    seed: str,
    stage: str,
) -> bool:
    filt = (
        f"tags.combo_id = '{combo_id}' AND tags.seed = '{seed}' "
        f"AND tags.stage = '{stage}' AND attributes.status = 'FINISHED'"
    )
    runs = client.search_runs([experiment_id], filter_string=filt, max_results=1)
    return len(runs) > 0


def backfill(
    *,
    grid_root: Path,
    experiment: str,
    mlflow_uri: str,
    gamma: str,
    eval_tau: int,
    skip_existing: bool,
    train_runs: bool,
    dry_run: bool,
) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    logs_root = grid_root / "logs"
    if not logs_root.is_dir():
        print(f"ERROR: logs dir not found: {logs_root}", file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = client.create_experiment(experiment)
        print(f"Created experiment: {experiment} (id={exp_id})")
    else:
        exp_id = exp.experiment_id

    jobs = []
    for d in sorted(logs_root.iterdir()):
        if not d.is_dir():
            continue
        job = parse_job(d)
        if job is not None:
            jobs.append(job)

    n_train = n_eval = n_skip = 0
    for job in jobs:
        base_tags = {
            "stage": "",
            "seed": job.seed,
            "gamma": gamma,
            "combo_id": job.combo_id,
            "dataset": "cancer_sim_cont",
            "backfill": "grid_logs",
            "grid_tag": job.tag,
        }

        if train_runs and job.best_val_mae_uns is not None:
            if skip_existing and _run_exists(
                client, exp_id, combo_id=job.combo_id, seed=job.seed, stage="ct_iql_em"
            ):
                n_skip += 1
            elif dry_run:
                print(f"[dry-run] train  {job.tag}  best_val={job.best_val_mae_uns:.6f}")
                n_train += 1
            else:
                run_name = f"ct_iql_em_seed{job.seed}_g{gamma}_{job.combo_id}_backfill"
                with mlflow.start_run(
                    experiment_id=exp_id, run_name=run_name, tags={**base_tags, "stage": "ct_iql_em"}
                ):
                    mlflow.set_tag("backfill", "grid_logs")
                    step = job.best_outer or 0
                    mlflow.log_metrics(
                        {
                            "best/outer_iter": float(step),
                            "best/val_mae_uns": float(job.best_val_mae_uns),
                        },
                        step=int(step),
                    )
                n_train += 1
                print(f"[train]  {job.tag}")

        if job.eval_mae_uns is not None:
            if skip_existing and _run_exists(
                client, exp_id, combo_id=job.combo_id, seed=job.seed, stage="eval"
            ):
                n_skip += 1
            elif dry_run:
                print(f"[dry-run] eval   {job.tag}  mae_uns={job.eval_mae_uns:.6f}")
                n_eval += 1
            else:
                run_name = f"eval_seed{job.seed}_g{gamma}_{job.combo_id}_backfill"
                metrics: Dict[str, float] = {
                    f"eval/tau{eval_tau}/mae_uns": float(job.eval_mae_uns),
                }
                if job.eval_mae_norm is not None:
                    metrics[f"eval/tau{eval_tau}/mae_norm"] = float(job.eval_mae_norm)
                if job.eval_rmse_uns is not None:
                    metrics[f"eval/tau{eval_tau}/rmse_uns"] = float(job.eval_rmse_uns)
                if job.eval_rmse_norm is not None:
                    metrics[f"eval/tau{eval_tau}/rmse_norm"] = float(job.eval_rmse_norm)

                with mlflow.start_run(
                    experiment_id=exp_id,
                    run_name=run_name,
                    tags={
                        **base_tags,
                        "stage": "eval",
                        "eval_split": "val",
                        "eval_tau_list": str(eval_tau),
                    },
                ):
                    mlflow.set_tag("backfill", "grid_logs")
                    mlflow.log_metrics(metrics, step=0)
                n_eval += 1
                print(f"[eval]   {job.tag}")

    print(
        f"\nParsed {len(jobs)} jobs from {logs_root}\n"
        f"  train runs created: {n_train}\n"
        f"  eval runs created:  {n_eval}\n"
        f"  skipped (existing): {n_skip}\n"
        f"  experiment: {experiment} @ {mlflow_uri}"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill MLflow from grid_results logs")
    p.add_argument(
        "--grid-root",
        type=Path,
        required=True,
        help="e.g. grid_results/em_iql_grid_her0/gamma_4",
    )
    p.add_argument("--experiment", default="em_iql_grid_her0")
    p.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    p.add_argument("--gamma", default="4")
    p.add_argument("--eval-tau", type=int, default=6)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip combo_id+seed+stage if a FINISHED run already exists (default: true)",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force create runs even if one already exists for that combo+seed+stage",
    )
    p.add_argument(
        "--train-runs",
        action="store_true",
        help="Also backfill ct_iql_em train runs (best/val_mae_uns from train.log)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = args.grid_root
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    raise SystemExit(
        backfill(
            grid_root=root,
            experiment=args.experiment,
            mlflow_uri=args.mlflow_uri,
            gamma=str(args.gamma),
            eval_tau=args.eval_tau,
            skip_existing=args.skip_existing,
            train_runs=args.train_runs,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
