"""Per-sample terminal-error diagnostics for CT+IQL checkpoints.

This mirrors ``eval_iql_planner.py`` rollout semantics, but writes distribution
statistics that explain the gap between MAE and RMSE.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Distribution

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_iql_planner import (  # noqa: E402
    _extend_h_work_after_one_step,
    _iql_augmented_state,
    _policy_to_sim_interval_torch,
    _resolve_eval_tau_list,
    _resolve_iql_ckpt,
    _sim_actions_to_tanh_batch,
    _unscaled_cancer_volume_np,
)
from src.data.cip_dataset import CIPDataset, get_dataloader  # noqa: E402
from src.data.iql_dataset_builder import align_h_t_static_to_history  # noqa: E402
from src.models.inference_model import InferenceModel  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.planners.iql_planner import IQLPlanner  # noqa: E402
from src.utils.em_ckpt import is_em_checkpoint, load_em_for_eval  # noqa: E402
from src.utils.inference_ckpt import load_inference_checkpoint  # noqa: E402
from src.utils.stable_iql_em_defaults import stable_select  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _top_share(values: np.ndarray, frac: float) -> float:
    if values.size == 0:
        return float("nan")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * frac)))
    return float(np.sort(values)[-k:].sum() / total)


def _to_list(raw, default: Iterable[int]) -> List[int]:
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return list(default)
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in list(raw)]


def _load_planner_and_encoder(args: DictConfig, original_cwd: Path, device: str):
    inference_model = InferenceModel(args).to(device)
    em_eval_ckpt = str(stable_select(args, "exp.em_eval_ckpt")).strip()
    planner_path = _resolve_iql_ckpt(args, original_cwd)
    em_path = Path(em_eval_ckpt) if em_eval_ckpt else planner_path
    if em_eval_ckpt and not em_path.is_absolute():
        em_path = original_cwd / em_path

    use_em = False
    if em_path.is_file():
        obj = torch.load(str(em_path), map_location="cpu")
        use_em = is_em_checkpoint(obj)

    if use_em:
        logger.info("Loading combined EM checkpoint from %s", em_path)
        planner = load_em_for_eval(inference_model, str(em_path), device)
    else:
        iql_ckpt = str(OmegaConf.select(args, "exp.iql_inference_ckpt", default=""))
        load_inference_checkpoint(inference_model, iql_ckpt, device)
        inference_model.eval()
        if not planner_path.exists():
            raise FileNotFoundError(
                f"IQL checkpoint not found: {planner_path}. Set exp.iql_eval_ckpt / exp.em_eval_ckpt."
            )
        planner = IQLPlanner.from_checkpoint(str(planner_path), device=device)
    inference_model.eval()
    planner.actor.eval()
    return inference_model, planner, em_path


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(
        str(original_cwd), args["exp"]["processed_data_dir"]
    )

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    if args.exp.test:
        data = dataset_collection.test_f.data
        fold = dataset_collection.test_f
        split_name = "test"
    else:
        data = dataset_collection.val_f.data
        fold = dataset_collection.val_f
        split_name = "val"

    inference_model, planner, ckpt_path = _load_planner_and_encoder(args, original_cwd, device)
    max_action = float(planner.cfg.max_action)
    max_tau = float(stable_select(args, "exp.max_tau"))
    autoregressive_eval = bool(stable_select(args, "exp.iql_eval_autoregressive"))
    batch_size = int(OmegaConf.select(args, "exp.batch_size_val", default=128))
    tau_list = _resolve_eval_tau_list(args)
    original_exp_tau = int(OmegaConf.select(args, "exp.tau", default=max(tau_list)))
    mean_ser, std_ser = dataset_collection.train_scaling_params

    label = str(OmegaConf.select(args, "exp.error_diag_label", default="iql"))
    out_dir_raw = str(OmegaConf.select(args, "exp.error_diag_output_dir", default="")).strip()
    out_dir = Path(out_dir_raw) if out_dir_raw else original_cwd / "error_diagnostics"
    if not out_dir.is_absolute():
        out_dir = original_cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    try:
        for tau in tau_list:
            args.exp.tau = int(tau)
            try:
                dataloader = get_dataloader(
                    CIPDataset(data, args, train=False),
                    batch_size=batch_size,
                    shuffle=False,
                )
            finally:
                args.exp.tau = original_exp_tau

            pred_norm_chunks = []
            true_norm_chunks = []
            logger.info(
                "Error diagnostic rollout | label=%s split=%s tau=%d autoregressive=%s",
                label,
                split_name,
                tau,
                autoregressive_eval,
            )

            with torch.no_grad():
                for batch in dataloader:
                    H_t, targets = batch
                    H_t = align_h_t_static_to_history(H_t)
                    for key in H_t:
                        H_t[key] = H_t[key].to(device)
                    for key in targets:
                        targets[key] = targets[key].to(device)

                    if autoregressive_eval:
                        eval_target = targets["outputs"][:, -1, :]
                        H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                        a_prev_sim = gather_last_valid(
                            H_work["current_treatments"], H_work.get("active_entries")
                        ).clone()
                        planned = []
                        for step in range(tau):
                            H_work = align_h_t_static_to_history(H_work)
                            z, _, _ = inference_model.ct_hidden_history(H_work)
                            a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                            obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
                            po = planner.actor(obs)
                            ma = planner.actor.max_action
                            if isinstance(po, Distribution):
                                a_raw = torch.clamp(ma * po.mean, -ma, ma)
                            else:
                                a_raw = torch.clamp(po * ma, -ma, ma)
                            a_sim = _policy_to_sim_interval_torch(a_raw, max_action)
                            planned.append(a_sim)
                            y_np = fold.simulate_output_after_actions(
                                H_work,
                                a_sim.unsqueeze(1),
                                dataset_collection.train_scaling_params,
                            )
                            y_norm = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                            _extend_h_work_after_one_step(
                                H_work, a_sim, y_norm, mean_ser, std_ser, torch.device(device)
                            )
                            a_prev_sim = a_sim
                        a_seq = torch.stack(planned, dim=1).contiguous()
                    else:
                        raise NotImplementedError("This diagnostic currently expects autoregressive eval.")

                    pred_norm = fold.simulate_output_after_actions(
                        H_t, a_seq, dataset_collection.train_scaling_params
                    )
                    true_norm = targets["outputs"][:, -1, :].detach().cpu().numpy()
                    pred_norm_chunks.append(pred_norm)
                    true_norm_chunks.append(true_norm)

            pred_norm_all = np.concatenate(pred_norm_chunks, axis=0)
            true_norm_all = np.concatenate(true_norm_chunks, axis=0)
            pred_uns = _unscaled_cancer_volume_np(pred_norm_all, mean_ser, std_ser).reshape(-1)
            true_uns = _unscaled_cancer_volume_np(true_norm_all, mean_ser, std_ser).reshape(-1)
            err = pred_uns - true_uns
            ae = np.abs(err)
            se = err ** 2
            n = int(ae.size)

            row = {
                "label": label,
                "split": split_name,
                "tau": int(tau),
                "n": n,
                "mae": float(ae.mean()),
                "rmse": float(np.sqrt(se.mean())),
                "mean_error": float(err.mean()),
                "median_ae": _percentile(ae, 50),
                "p75_ae": _percentile(ae, 75),
                "p90_ae": _percentile(ae, 90),
                "p95_ae": _percentile(ae, 95),
                "p99_ae": _percentile(ae, 99),
                "max_ae": float(ae.max()) if n else float("nan"),
                "rmse_over_mae": float(np.sqrt(se.mean()) / (ae.mean() + 1e-12)),
                "top1pct_mse_share": _top_share(se, 0.01),
                "top5pct_mse_share": _top_share(se, 0.05),
                "top10pct_mse_share": _top_share(se, 0.10),
                "top1pct_mae_share": _top_share(ae, 0.01),
                "top5pct_mae_share": _top_share(ae, 0.05),
                "checkpoint": str(ckpt_path),
            }
            summary_rows.append(row)
            logger.info(
                "tau=%d n=%d mae=%.6f rmse=%.6f p95=%.6f p99=%.6f top5_mse=%.3f",
                tau,
                n,
                row["mae"],
                row["rmse"],
                row["p95_ae"],
                row["p99_ae"],
                row["top5pct_mse_share"],
            )

            for idx, (p, y, e, a) in enumerate(zip(pred_uns, true_uns, err, ae)):
                sample_rows.append(
                    {
                        "label": label,
                        "split": split_name,
                        "tau": int(tau),
                        "sample_idx": idx,
                        "pred_unscaled": float(p),
                        "target_unscaled": float(y),
                        "error": float(e),
                        "abs_error": float(a),
                        "squared_error": float(e * e),
                    }
                )
    finally:
        args.exp.tau = original_exp_tau

    summary_path = out_dir / f"{label}_{split_name}_summary.csv"
    samples_path = out_dir / f"{label}_{split_name}_samples.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", samples_path)


if __name__ == "__main__":
    main()
