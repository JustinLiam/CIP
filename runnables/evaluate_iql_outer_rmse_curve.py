"""
Post-hoc outer-checkpoint diagnostics for CT+IQL EM.

Loads ct_iql_em_outerXXXX.pt checkpoints from a directory, evaluates each
checkpoint on tau=1..6 with the corrected IQL planner evaluator, and writes:

  - outer_rmse_curve.csv          long-form per outer/tau metrics
  - outer_rmse_curve_summary.csv  per-outer mean/max RMSE
  - outer_rmse_curve.png          RMSE vs outer curves
"""
import ast
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import matplotlib
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics
from src.models.inference_model import InferenceModel
from src.utils.em_ckpt import is_em_checkpoint, load_em_for_eval
from src.utils.stable_iql_em_defaults import stable_select
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _list_from_config(value: Any, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return list(default)
        if raw.startswith("["):
            value = ast.literal_eval(raw)
        else:
            value = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in value]


def _str_list_from_config(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return list(default)
        if raw.startswith("["):
            value = ast.literal_eval(raw)
        else:
            value = [x.strip() for x in raw.split(",") if x.strip()]
    return [str(v) for v in value]


def _resolve_path(value: str, original_cwd: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else original_cwd / p


def _outer_from_checkpoint(path: Path) -> int:
    m = re.search(r"outer(\d+)", path.name)
    if m:
        return int(m.group(1))
    obj = torch.load(str(path), map_location="cpu")
    return int(obj.get("outer_iter", -1))


def _load_meta(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if not is_em_checkpoint(obj):
        raise ValueError(f"Not an EM checkpoint: {path}")
    return {
        "outer": int(obj.get("outer_iter", _outer_from_checkpoint(path))),
        "val_score": obj.get("val_score", ""),
        "val_metric": obj.get("val_metric", ""),
        "checkpoint_type": obj.get("checkpoint_type", ""),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    tau_values = sorted({int(r["tau"]) for r in rows})
    plt.figure(figsize=(9, 5.5))
    for tau in tau_values:
        tau_rows = sorted((r for r in rows if int(r["tau"]) == tau), key=lambda r: int(r["outer"]))
        plt.plot(
            [int(r["outer"]) for r in tau_rows],
            [float(r["rmse_uns"]) for r in tau_rows],
            marker="o",
            linewidth=1.3,
            markersize=3,
            label=f"tau={tau}",
        )

    summary_sorted = sorted(summary_rows, key=lambda r: int(r["outer"]))
    plt.plot(
        [int(r["outer"]) for r in summary_sorted],
        [float(r["mean_rmse_uns"]) for r in summary_sorted],
        color="black",
        marker="s",
        linewidth=2.2,
        markersize=3.5,
        label="mean tau1-6",
    )
    plt.xlabel("EM outer")
    plt.ylabel("RMSE unscaled")
    plt.title("Outer Checkpoint RMSE Curve")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    ckpt_dir_raw = str(OmegaConf.select(args, "exp.outer_curve_ckpt_dir", default="")).strip()
    if not ckpt_dir_raw:
        raise ValueError("Set exp.outer_curve_ckpt_dir to a directory containing ct_iql_em_outer*.pt")
    ckpt_dir = _resolve_path(ckpt_dir_raw, original_cwd)
    pattern = str(OmegaConf.select(args, "exp.outer_curve_ckpt_pattern", default="ct_iql_em_outer*.pt"))
    ckpts = sorted(ckpt_dir.glob(pattern), key=_outer_from_checkpoint)
    max_ckpts = OmegaConf.select(args, "exp.outer_curve_max_checkpoints", default=None)
    if max_ckpts is not None:
        ckpts = ckpts[: int(max_ckpts)]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched {ckpt_dir / pattern}")

    tau_list = _list_from_config(
        OmegaConf.select(args, "exp.outer_curve_tau_list", default=None),
        default=_list_from_config(stable_select(args, "exp.iql_eval_tau_list"), [1, 2, 3, 4, 5, 6]),
    )
    worlds = tuple(
        w.strip()
        for w in _str_list_from_config(
            OmegaConf.select(args, "exp.outer_curve_worlds", default=None),
            default=_str_list_from_config(stable_select(args, "exp.em_val_worlds"), ["sim"]),
        )
        if w.strip()
    )
    sel_world = str(OmegaConf.select(args, "exp.outer_curve_selection_world", default=stable_select(args, "exp.em_val_selection_world", worlds[0])))

    out_dir_raw = str(OmegaConf.select(args, "exp.outer_curve_output_dir", default="")).strip()
    out_dir = _resolve_path(out_dir_raw, original_cwd) if out_dir_raw else ckpt_dir / "outer_rmse_curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(args.dataset.static_size) > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    split_name = "test" if bool(args.exp.test) else "val"
    fold = dataset_collection.test_f if bool(args.exp.test) else dataset_collection.val_f
    inference_model = InferenceModel(args).to(device)
    max_tau = float(stable_select(args, "exp.max_tau"))
    autoregressive_eval = bool(stable_select(args, "exp.iql_eval_autoregressive"))
    val_bs = int(stable_select(args, "exp.iql_val_batch_size") or args.exp.batch_size_val)
    base_seed = int(args.exp.seed)

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    logger.info(
        "Evaluating %d checkpoints from %s on split=%s taus=%s",
        len(ckpts),
        ckpt_dir,
        split_name,
        tau_list,
    )

    for ckpt in ckpts:
        meta = _load_meta(ckpt)
        outer = int(meta["outer"])
        logger.info("Loading outer=%d checkpoint=%s", outer, ckpt)
        planner = load_em_for_eval(inference_model, str(ckpt), device)
        tau_rmse = []
        for tau in tau_list:
            set_seed(base_seed + int(tau) * 1009)
            metrics = aggregate_iql_planner_metrics(
                planner,
                inference_model,
                dataset_collection,
                fold,
                args,
                device=device,
                tau=int(tau),
                max_tau=max_tau,
                autoregressive_eval=autoregressive_eval,
                val_batch_size=val_bs,
                log_batches=False,
                worlds=worlds,
            )
            per_world = metrics.get("per_world", {worlds[0]: metrics})
            m = per_world[sel_world]
            tau_rmse.append(float(m["rmse_uns"]))
            rows.append(
                {
                    "outer": outer,
                    "split": split_name,
                    "tau": int(tau),
                    "mae_uns": float(m["mae_uns"]),
                    "rmse_uns": float(m["rmse_uns"]),
                    "rmse_norm": float(m["rmse_norm"]),
                    "val_score": meta["val_score"],
                    "val_metric": meta["val_metric"],
                    "checkpoint_type": meta["checkpoint_type"],
                    "checkpoint": str(ckpt),
                }
            )
            logger.info("outer=%d tau=%d rmse_uns=%.6f mae_uns=%.6f", outer, tau, m["rmse_uns"], m["mae_uns"])
        summary_rows.append(
            {
                "outer": outer,
                "split": split_name,
                "mean_rmse_uns": sum(tau_rmse) / len(tau_rmse),
                "max_rmse_uns": max(tau_rmse),
                "min_rmse_uns": min(tau_rmse),
                "checkpoint": str(ckpt),
            }
        )

    long_csv = out_dir / "outer_rmse_curve.csv"
    summary_csv = out_dir / "outer_rmse_curve_summary.csv"
    png_path = out_dir / "outer_rmse_curve.png"
    _write_csv(
        long_csv,
        rows,
        [
            "outer",
            "split",
            "tau",
            "mae_uns",
            "rmse_uns",
            "rmse_norm",
            "val_score",
            "val_metric",
            "checkpoint_type",
            "checkpoint",
        ],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["outer", "split", "mean_rmse_uns", "max_rmse_uns", "min_rmse_uns", "checkpoint"],
    )
    _plot_curve(rows, summary_rows, png_path)
    logger.info("Wrote %s", long_csv)
    logger.info("Wrote %s", summary_csv)
    logger.info("Wrote %s", png_path)


if __name__ == "__main__":
    main()
