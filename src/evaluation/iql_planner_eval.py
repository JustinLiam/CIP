"""
Aggregate IQL planner metrics on a CIP split (val/test), matching ``eval_iql_planner.py`` protocol.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributions import Distribution

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history, dataset_actions_to_tanh_policy_space
from src.models.inference_model import InferenceModel
from src.planners.iql_planner import IQLPlanner

logger = logging.getLogger(__name__)


def _actions_to_sim_interval(raw: np.ndarray, max_action: float) -> np.ndarray:
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return np.clip((raw + max_action) / denom, 0.0, 1.0).astype(np.float32)


def _policy_to_sim_interval_torch(raw: torch.Tensor, max_action: float) -> torch.Tensor:
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return torch.clamp((raw + max_action) / denom, 0.0, 1.0)


def _sim_actions_to_tanh_batch(a_sim: torch.Tensor, max_action: float) -> torch.Tensor:
    if max_action <= 0:
        return a_sim
    a = torch.clamp(a_sim, 0.0, max_action)
    return 2.0 * a - max_action


def _iql_augmented_state(
    z: torch.Tensor,
    eval_target: torch.Tensor,
    step: int,
    eval_tau: int,
    max_tau: float,
    a_prev_tanh: torch.Tensor,
) -> torch.Tensor:
    bsz = z.size(0)
    steps_left = float(eval_tau - step)
    delta = torch.full((bsz, 1), steps_left / max_tau, device=z.device, dtype=z.dtype)
    return torch.cat([z, eval_target, delta, a_prev_tanh], dim=-1)


def _unscaled_cancer_volume_np(y_norm: np.ndarray, mean_ser, std_ser) -> np.ndarray:
    m = float(mean_ser["cancer_volume"])
    s = float(std_ser["cancer_volume"])
    return y_norm.astype(np.float64) * s + m


def _extend_h_work_after_one_step(
    H: dict,
    a_sim: torch.Tensor,
    y_norm: torch.Tensor,
    mean_ser,
    std_ser,
    device: torch.device,
) -> None:
    B = a_sim.size(0)
    y_col = y_norm.view(B, 1)
    y_ch = y_col.unsqueeze(-1)
    y_uns = y_col * float(std_ser["cancer_volume"]) + float(mean_ser["cancer_volume"])

    last_curr = H["current_treatments"][:, -1:, :].clone()
    last_out = H["outputs"][:, -1:, :].clone()

    H["prev_treatments"] = torch.cat([H["prev_treatments"], last_curr], dim=1)
    H["current_treatments"] = torch.cat([H["current_treatments"], a_sim.unsqueeze(1)], dim=1)
    H["outputs"] = torch.cat([H["outputs"], y_ch], dim=1)
    H["prev_outputs"] = torch.cat([H["prev_outputs"], last_out], dim=1)
    ae = H["active_entries"]
    H["active_entries"] = torch.cat(
        [ae, torch.ones(B, 1, ae.size(-1), device=device, dtype=ae.dtype)], dim=1
    )

    H["cancer_volume"] = torch.cat([H["cancer_volume"], y_col], dim=1)

    uo = H["unscaled_outputs"]
    y_u = y_uns.unsqueeze(-1) if uo.dim() == 3 else y_uns
    H["unscaled_outputs"] = torch.cat([uo, y_u], dim=1)

    H["chemo_application"] = torch.cat([H["chemo_application"], a_sim[:, 0:1]], dim=1)
    H["radio_application"] = torch.cat([H["radio_application"], a_sim[:, 1:2]], dim=1)

    if "static_features" in H:
        sf = H["static_features"]
        if sf.dim() == 3:
            last = sf[:, -1:, :].expand(-1, 1, -1)
            H["static_features"] = torch.cat([sf, last], dim=1)

    if "current_covariates" in H:
        cc = H["current_covariates"]
        ext = cc[:, -1:, :].clone()
        ext[:, :, 0:1] = y_ch
        H["current_covariates"] = torch.cat([cc, ext], dim=1)


@torch.no_grad()
def aggregate_iql_planner_metrics(
    planner: IQLPlanner,
    inference_model: InferenceModel,
    dataset_collection: Any,
    fold: Any,
    args: DictConfig,
    *,
    device: str,
    tau: int,
    max_tau: float,
    autoregressive_eval: bool,
    val_batch_size: int = 128,
    log_batches: bool = False,
    return_series: bool = False,
    include_factual_traj_rmse: bool = False,
) -> Dict[str, Any]:
    """
    Full pass over ``fold``'s dataloader; returns global MAE/RMSE (normalized and unscaled tumor volume).

    Matches the aggregation in ``runnables/eval_iql_planner.py``.
    """
    data = fold.data
    max_action = float(planner.cfg.max_action)
    mean_ser, std_ser = dataset_collection.train_scaling_params

    dataloader = get_dataloader(CIPDataset(data, args, train=False), batch_size=val_batch_size, shuffle=False)

    ture_output_list = []
    output_after_actions_list = []
    factual_output_list = [] if include_factual_traj_rmse else None
    batch_rmse_plan: list = []
    batch_rmse_fact: list = [] if include_factual_traj_rmse else None

    was_training = planner.actor.training
    planner.actor.eval()
    inference_model.eval()

    try:
        for i, batch in enumerate(dataloader):
            H_t, targets = batch
            H_t = align_h_t_static_to_history(H_t)
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            if autoregressive_eval:
                eval_target = targets["outputs"][:, -1, :]
                H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                a_prev_sim = H_work["current_treatments"][:, -1, :].clone()
                planned = []
                for step in range(tau):
                    H_work = align_h_t_static_to_history(H_work)
                    z, _, _ = inference_model.ct_hidden_history(H_work)
                    a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                    obs = _iql_augmented_state(z, eval_target, step, tau, max_tau, a_prev_tanh)
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
                z, _, _ = inference_model.ct_hidden_history(H_t)
                z_np = z.detach().cpu().numpy()
                eval_target_np = targets["outputs"][:, -1, :].detach().cpu().numpy()
                a_prev_raw = H_t["current_treatments"][:, -1, :].detach().cpu().numpy()
                a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                bsz = z_np.shape[0]
                delta_scalar = float(tau - 0) / max_tau
                delta_vec = np.array([delta_scalar], dtype=np.float32)
                a_rows = []
                for b in range(bsz):
                    obs_b = np.concatenate([z_np[b], eval_target_np[b], delta_vec, a_prev_feat[b]], axis=0)
                    a_rows.append(planner.act(obs_b))
                a_raw = np.stack(a_rows, axis=0)
                a_sim = _actions_to_sim_interval(a_raw, max_action)
                a_seq = (
                    torch.tensor(a_sim, device=device, dtype=torch.float32)
                    .unsqueeze(1)
                    .expand(-1, tau, -1)
                    .contiguous()
                )

            output_after_actions = fold.simulate_output_after_actions(
                H_t, a_seq, dataset_collection.train_scaling_params
            )
            ture_output = targets["outputs"][:, -1, :].detach().cpu().numpy()

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)

            if include_factual_traj_rmse:
                true_actions = targets["current_treatments"]
                ture_output_actions = fold.simulate_output_after_actions(
                    H_t, true_actions, dataset_collection.train_scaling_params
                )
                factual_output_list.append(ture_output_actions)
                loss_2 = float(np.sqrt(((ture_output_actions - ture_output) ** 2).mean()))
                batch_rmse_fact.append(loss_2)

            if log_batches:
                loss = float(np.sqrt(((output_after_actions - ture_output) ** 2).mean()))
                batch_rmse_plan.append(loss)
                if include_factual_traj_rmse and batch_rmse_fact is not None:
                    logger.info(
                        f"Batch {i} RMSE (IQL plan): {loss:.6f}, RMSE (factual actions): {batch_rmse_fact[-1]:.6f}"
                    )
                else:
                    logger.info(f"Batch {i} RMSE (IQL plan): {loss:.6f}")
    finally:
        planner.actor.train(was_training)

    ture_output_list = np.concatenate(ture_output_list, axis=0)
    output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)

    rmse_norm = float(np.sqrt(((output_after_actions_list - ture_output_list) ** 2).mean()))
    rmse_factual_norm = None
    if include_factual_traj_rmse and factual_output_list is not None:
        factual_output_list = np.concatenate(factual_output_list, axis=0)
        rmse_factual_norm = float(
            np.sqrt(((factual_output_list - ture_output_list) ** 2).mean())
        )

    iql_y_norm = output_after_actions_list.reshape(-1)
    true_y_norm = ture_output_list.reshape(-1)
    iql_y_uns = _unscaled_cancer_volume_np(output_after_actions_list, mean_ser, std_ser).reshape(-1)
    true_y_uns = _unscaled_cancer_volume_np(ture_output_list, mean_ser, std_ser).reshape(-1)

    mae_norm = float(np.mean(np.abs(iql_y_norm - true_y_norm)))
    mae_uns = float(np.mean(np.abs(iql_y_uns - true_y_uns)))
    rmse_uns = float(np.sqrt(np.mean((iql_y_uns - true_y_uns) ** 2)))

    try:
        std = float(dataset_collection.train_scaling_params[1]["cancer_volume"])
    except Exception:
        std = 1.0

    out: Dict[str, Any] = {
        "mae_norm": mae_norm,
        "mae_uns": mae_uns,
        "rmse_norm": rmse_norm,
        "rmse_uns": rmse_uns,
        "rmse_norm_x_std": rmse_norm * std,
        "mean_batch_rmse_plan": float(np.mean(batch_rmse_plan)) if batch_rmse_plan else None,
        "mean_batch_rmse_factual": float(np.mean(batch_rmse_fact)) if batch_rmse_fact else None,
        "rmse_factual_norm": rmse_factual_norm,
    }
    if return_series:
        out["iql_y_norm"] = iql_y_norm
        out["true_y_norm"] = true_y_norm
        out["iql_y_uns"] = iql_y_uns
        out["true_y_uns"] = true_y_uns
    return out
