"""
Aggregate IQL planner metrics on a CIP split (val/test), matching ``eval_iql_planner.py`` protocol.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history, dataset_actions_to_tanh_policy_space
from src.evaluation.iql_action_selection import select_iql_policy_action
from src.models.inference_model import InferenceModel
from src.models.sequence_utils import gather_last_valid
from src.planners.iql_planner import IQLPlanner
from src.utils.stable_iql_em_defaults import stable_select

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
    a = torch.clamp(a_sim, 0.0, 1.0)
    return (2.0 * a - 1.0) * float(max_action)


def _iql_augmented_state(
    planner: IQLPlanner,
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
    return planner.build_state(z, eval_target, delta, a_prev_tanh)


def _unscaled_cancer_volume_np(y_norm: np.ndarray, mean_ser, std_ser) -> np.ndarray:
    m = float(mean_ser["cancer_volume"])
    s = float(std_ser["cancer_volume"])
    return y_norm.astype(np.float64) * s + m


def _split_scaling_params(scaling_params):
    if isinstance(scaling_params, dict) and "output_means" in scaling_params:
        means = np.asarray(scaling_params["output_means"], dtype=np.float64).reshape(1, -1)
        stds = np.asarray(scaling_params["output_stds"], dtype=np.float64).reshape(1, -1)
        return means, stds
    mean_ser, std_ser = scaling_params
    if hasattr(mean_ser, "__getitem__") and "cancer_volume" in mean_ser:
        return (
            np.asarray([[float(mean_ser["cancer_volume"])]], dtype=np.float64),
            np.asarray([[float(std_ser["cancer_volume"])]], dtype=np.float64),
        )
    return (
        np.asarray(mean_ser, dtype=np.float64).reshape(1, -1),
        np.asarray(std_ser, dtype=np.float64).reshape(1, -1),
    )


def _unscale_outputs_np(y_norm: np.ndarray, scaling_params) -> np.ndarray:
    means, stds = _split_scaling_params(scaling_params)
    return np.asarray(y_norm, dtype=np.float64) * stds + means


def _unscale_outputs_torch(y_norm: torch.Tensor, scaling_params, device: torch.device) -> torch.Tensor:
    means, stds = _split_scaling_params(scaling_params)
    mean_t = torch.as_tensor(means, dtype=y_norm.dtype, device=device)
    std_t = torch.as_tensor(stds, dtype=y_norm.dtype, device=device)
    return y_norm * std_t + mean_t


def _fingerprint_np(arr: np.ndarray, max_items: int = 64) -> str:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size > max_items:
        flat = flat[:max_items]
    flat = np.round(flat, 6)
    return hashlib.sha1(flat.tobytes()).hexdigest()[:16]


def _extend_h_work_after_one_step(
    H: dict,
    a_sim: torch.Tensor,
    y_norm: torch.Tensor,
    scaling_params,
    device: torch.device,
    next_observation: Dict[str, np.ndarray] | None = None,
) -> None:
    B = a_sim.size(0)
    if next_observation is not None and "outputs" in next_observation:
        y_norm = torch.as_tensor(next_observation["outputs"], device=device, dtype=torch.float32)
    y_step = y_norm.view(B, -1)
    y_ch = y_step.unsqueeze(1)
    y_uns = _unscale_outputs_torch(y_step, scaling_params, device)

    def _next_tensor(key: str) -> torch.Tensor | None:
        if next_observation is None or key not in next_observation:
            return None
        return torch.as_tensor(next_observation[key], device=device, dtype=torch.float32).view(B, 1, -1)

    active = H.get("active_entries")
    last_curr = gather_last_valid(H["current_treatments"], active).unsqueeze(1).clone()
    last_out = gather_last_valid(H["outputs"], active).unsqueeze(1).clone()

    next_action = _next_tensor("current_treatments")
    H["prev_treatments"] = torch.cat([H["prev_treatments"], last_curr], dim=1)
    H["current_treatments"] = torch.cat(
        [H["current_treatments"], next_action if next_action is not None else a_sim.unsqueeze(1)],
        dim=1,
    )
    H["outputs"] = torch.cat([H["outputs"], y_ch], dim=1)
    H["prev_outputs"] = torch.cat([H["prev_outputs"], last_out], dim=1)
    ae = H["active_entries"]
    next_active = _next_tensor("active_entries")
    H["active_entries"] = torch.cat(
        [ae, next_active if next_active is not None else torch.ones(B, 1, ae.size(-1), device=device, dtype=ae.dtype)],
        dim=1,
    )

    if "sequence_lengths" in H:
        H["sequence_lengths"] = H["sequence_lengths"] + 1

    next_sim_day = _next_tensor("sim_day")
    if "sim_day" in H and next_sim_day is not None:
        H["sim_day"] = torch.cat([H["sim_day"], next_sim_day], dim=1)
    elif "sim_day" in H:
        last = H["sim_day"][:, -1:, :].clone()
        H["sim_day"] = torch.cat([H["sim_day"], last + 1.0], dim=1)
    for meta_key in ("sim_episode_id", "sim_seed", "sim_county_id"):
        next_meta = _next_tensor(meta_key)
        if meta_key in H and next_meta is not None:
            H[meta_key] = torch.cat([H[meta_key], next_meta], dim=1)
        elif meta_key in H:
            last = H[meta_key][:, -1:, :].clone()
            H[meta_key] = torch.cat([H[meta_key], last], dim=1)

    if "cancer_volume" in H:
        H["cancer_volume"] = torch.cat([H["cancer_volume"], y_step[:, 0:1]], dim=1)

    if "unscaled_outputs" in H:
        uo = H["unscaled_outputs"]
        next_u = _next_tensor("unscaled_outputs")
        y_u = next_u if next_u is not None and uo.dim() == 3 else y_uns.unsqueeze(1) if uo.dim() == 3 else y_uns
        H["unscaled_outputs"] = torch.cat([uo, y_u], dim=1)
    if "prev_unscaled_outputs" in H:
        last_u = gather_last_valid(H["unscaled_outputs"][:, :-1, :], active).unsqueeze(1).clone()
        H["prev_unscaled_outputs"] = torch.cat([H["prev_unscaled_outputs"], last_u], dim=1)

    if "chemo_application" in H:
        H["chemo_application"] = torch.cat([H["chemo_application"], a_sim[:, 0:1]], dim=1)
    if "radio_application" in H:
        H["radio_application"] = torch.cat([H["radio_application"], a_sim[:, 1:2]], dim=1)

    if "static_features" in H:
        sf = H["static_features"]
        if sf.dim() == 3:
            last = sf[:, -1:, :].expand(-1, 1, -1)
            H["static_features"] = torch.cat([sf, last], dim=1)

    next_cov = _next_tensor("current_covariates")
    if next_cov is not None:
        if "current_covariates" in H:
            H["current_covariates"] = torch.cat([H["current_covariates"], next_cov], dim=1)
        if "vitals" in H:
            H["vitals"] = torch.cat([H["vitals"], next_cov], dim=1)
        if "next_covariates" in H:
            H["next_covariates"] = torch.cat([H["next_covariates"], next_cov], dim=1)
        if "next_vitals" in H:
            H["next_vitals"] = torch.cat([H["next_vitals"], next_cov], dim=1)
        return

    if "vitals" in H and "future_vitals" in H and H["future_vitals"].size(1) > 0:
        next_vitals = H["future_vitals"][:, :1, :]
        H["vitals"] = torch.cat([H["vitals"], next_vitals], dim=1)
        H["future_vitals"] = H["future_vitals"][:, 1:, :]
        if "current_covariates" in H:
            H["current_covariates"] = torch.cat([H["current_covariates"], next_vitals], dim=1)
        if "next_covariates" in H:
            H["next_covariates"] = H["next_covariates"][:, 1:, :]
        if "next_vitals" in H:
            H["next_vitals"] = H["next_vitals"][:, 1:, :]
        return

    if "current_covariates" in H:
        cc = H["current_covariates"]
        ext = cc[:, -1:, :].clone()
        ext[:, :, 0:y_step.size(-1)] = y_ch
        H["current_covariates"] = torch.cat([cc, ext], dim=1)
        if "vitals" in H:
            H["vitals"] = torch.cat([H["vitals"], ext], dim=1)


def _rollout_one_step_y(
    H_work: Dict,
    a_sim: torch.Tensor,
    *,
    fold,
    scaling_params,
    device: str,
) -> torch.Tensor:
    """
    One-step outcome y_{t+1} in normalized (train-scaling) space given current H_work
    and a simulator-interval action a_sim in [0, 1].

    Returns a ``[B, 1]`` (or ``[B, y_dim]``) tensor on ``device``.
    """
    y_np = fold.simulate_output_after_actions(H_work, a_sim.unsqueeze(1), scaling_params)
    return torch.as_tensor(y_np, device=device, dtype=torch.float32)


def _rollout_one_step(
    H_work: Dict,
    a_sim: torch.Tensor,
    *,
    fold,
    scaling_params,
    device: str,
) -> tuple[torch.Tensor, Dict[str, np.ndarray] | None]:
    if hasattr(fold, "simulate_next_after_action"):
        next_obs = fold.simulate_next_after_action(H_work, a_sim, scaling_params)
        y = torch.as_tensor(next_obs["outputs"], device=device, dtype=torch.float32)
        return y, next_obs
    return _rollout_one_step_y(
        H_work, a_sim, fold=fold, scaling_params=scaling_params, device=device
    ), None


def _simulate_a_seq_final_y(
    H_t: Dict,
    a_seq: torch.Tensor,
    *,
    fold,
    scaling_params,
) -> np.ndarray:
    """
    Replay ``a_seq`` (shape ``[B, tau, A]``) from ``H_t`` in the dataset simulator and
    return y_{t+tau} as a ``[B, y_dim]`` numpy array in normalized train-scaling space.
    """
    return fold.simulate_output_after_actions(H_t, a_seq, scaling_params)


def _stats_np(x: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _make_action_grid(action_dim: int, grid_points: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    points = max(2, int(grid_points))
    vals = torch.linspace(0.0, 1.0, points, device=device, dtype=dtype)
    grids = torch.cartesian_prod(*([vals] * int(action_dim)))
    return grids.contiguous()


@torch.no_grad()
def _q_grid_action_diagnostics(
    planner: IQLPlanner,
    obs: torch.Tensor,
    action_grid_sim: torch.Tensor,
    max_action: float,
) -> Dict[str, np.ndarray]:
    bsz = obs.size(0)
    grid_n = action_grid_sim.size(0)
    action_grid_policy = _sim_actions_to_tanh_batch(action_grid_sim, max_action)
    q_chunks = []
    chunk = 4096
    for start in range(0, grid_n, chunk):
        end = min(start + chunk, grid_n)
        g = end - start
        obs_rep = obs.unsqueeze(1).expand(bsz, g, obs.size(-1)).reshape(bsz * g, obs.size(-1))
        act_rep = action_grid_policy[start:end].unsqueeze(0).expand(bsz, g, action_grid_policy.size(-1)).reshape(bsz * g, action_grid_policy.size(-1))
        q_chunks.append(planner.qf(obs_rep, act_rep).view(bsz, g).detach())
    q_grid = torch.cat(q_chunks, dim=1)
    best_idx = torch.argmax(q_grid, dim=1)
    q_argmax = action_grid_sim[best_idx]
    grid_mean = action_grid_sim.mean(dim=1).view(1, -1)
    x = grid_mean - grid_mean.mean(dim=1, keepdim=True)
    y = q_grid - q_grid.mean(dim=1, keepdim=True)
    denom = (x * x).sum(dim=1).clamp(min=1e-8)
    slope = (x * y).sum(dim=1) / denom
    return {
        "q_argmax": q_argmax.detach().cpu().numpy(),
        "q_slope": slope.detach().cpu().numpy(),
        "q_best": q_grid.max(dim=1).values.detach().cpu().numpy(),
    }


@torch.no_grad()
def _sim_best_action_proxy(
    H_t: Dict[str, torch.Tensor],
    target: torch.Tensor,
    action_grid_sim: torch.Tensor,
    *,
    tau: int,
    fold,
    scaling_params,
    device: str,
) -> np.ndarray:
    bsz = target.size(0)
    errors = []
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    target_np = target.detach().cpu().numpy()
    try:
        for cand in action_grid_sim:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            a_seq = cand.view(1, 1, -1).expand(bsz, int(tau), cand.numel()).contiguous()
            y = _simulate_a_seq_final_y(
                H_t, a_seq,
                fold=fold, scaling_params=scaling_params,
            )
            err = np.sqrt(np.mean((y - target_np) ** 2, axis=1))
            errors.append(err)
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    err_grid = np.stack(errors, axis=1)
    best_idx = np.argmin(err_grid, axis=1)
    return action_grid_sim.detach().cpu().numpy()[best_idx]


def _append_action_diag(acc: Dict[str, list], key: str, value: np.ndarray) -> None:
    acc.setdefault(key, []).append(np.asarray(value, dtype=np.float32))


def _finalize_action_diagnostics(acc: Dict[str, list]) -> Dict[str, Any]:
    if not acc or not acc.get("planned"):
        return {}
    planned = np.concatenate(acc["planned"], axis=0)
    factual = np.concatenate(acc["factual"], axis=0)
    planned_flat = planned.reshape(-1, planned.shape[-1])
    factual_flat = factual.reshape(-1, factual.shape[-1])
    diff_flat = planned_flat - factual_flat
    out: Dict[str, Any] = {
        "planned_mean": float(planned_flat.mean()),
        "factual_mean": float(factual_flat.mean()),
        "planned_minus_factual_mean": float(planned_flat.mean() - factual_flat.mean()),
        "action_rmse": float(np.sqrt(np.mean(diff_flat ** 2))),
        "action_mae": float(np.mean(np.abs(diff_flat))),
        "planned": _stats_np(planned_flat),
        "factual": _stats_np(factual_flat),
        "planned_minus_factual": _stats_np(diff_flat),
    }
    if acc.get("q_argmax"):
        q_argmax = np.concatenate(acc["q_argmax"], axis=0)
        out["q_argmax_mean"] = float(q_argmax.mean())
        out["planned_minus_q_argmax_mean"] = float(planned[:, 0, :].mean() - q_argmax.mean())
        out["q_argmax"] = _stats_np(q_argmax)
    if acc.get("q_slope"):
        q_slope = np.concatenate(acc["q_slope"], axis=0)
        out["q_slope_mean"] = float(q_slope.mean())
        out["q_slope"] = _stats_np(q_slope)
    if acc.get("sim_best_proxy"):
        sim_best = np.concatenate(acc["sim_best_proxy"], axis=0)
        out["sim_best_proxy_mean"] = float(sim_best.mean())
        out["planned_minus_sim_best_proxy_mean"] = float(planned[:, 0, :].mean() - sim_best.mean())
        out["sim_best_proxy"] = _stats_np(sim_best)
    return out


@torch.no_grad()
def _collect_batch_action_diagnostics(
    planner: IQLPlanner,
    H_t: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    first_obs: torch.Tensor,
    a_seq: torch.Tensor,
    *,
    tau: int,
    fold,
    scaling_params,
    action_grid_sim: torch.Tensor,
    max_action: float,
    device: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        "planned": a_seq.detach().cpu().numpy(),
        "factual": targets["current_treatments"].detach().cpu().numpy(),
    }
    q_diag = _q_grid_action_diagnostics(planner, first_obs, action_grid_sim, max_action)
    out.update(q_diag)
    out["sim_best_proxy"] = _sim_best_action_proxy(
        H_t, targets["outputs"][:, -1, :], action_grid_sim,
        tau=tau, fold=fold, scaling_params=scaling_params,
        device=device,
    )
    return out


def _compute_rollout_metrics(
    output_after_actions_list: list,
    ture_output_list: list,
    factual_output_list: list | None,
    scaling_params,
    std: float,
    batch_rmse_plan: list,
    batch_rmse_fact: list | None,
    return_series: bool,
) -> Dict[str, Any]:
    """Aggregate per-batch simulator closed-loop outputs into MAE/RMSE metrics."""
    pred_arr = np.concatenate(output_after_actions_list, axis=0)
    true_arr = np.concatenate(ture_output_list, axis=0)

    rmse_norm = float(np.sqrt(((pred_arr - true_arr) ** 2).mean()))
    rmse_factual_norm = None
    mae_factual_norm = None
    mae_factual_uns = None
    if factual_output_list is not None and len(factual_output_list) > 0:
        fact_arr = np.concatenate(factual_output_list, axis=0)
        rmse_factual_norm = float(np.sqrt(((fact_arr - true_arr) ** 2).mean()))
        fact_y_uns = _unscale_outputs_np(fact_arr, scaling_params).reshape(-1)
        true_y_uns_full = _unscale_outputs_np(true_arr, scaling_params).reshape(-1)
        mae_factual_norm = float(np.mean(np.abs(fact_arr.reshape(-1) - true_arr.reshape(-1))))
        mae_factual_uns = float(np.mean(np.abs(fact_y_uns - true_y_uns_full)))

    iql_y_norm = pred_arr.reshape(-1)
    true_y_norm = true_arr.reshape(-1)
    iql_y_uns = _unscale_outputs_np(pred_arr, scaling_params).reshape(-1)
    true_y_uns = _unscale_outputs_np(true_arr, scaling_params).reshape(-1)

    mae_norm = float(np.mean(np.abs(iql_y_norm - true_y_norm)))
    mae_uns = float(np.mean(np.abs(iql_y_uns - true_y_uns)))
    rmse_uns = float(np.sqrt(np.mean((iql_y_uns - true_y_uns) ** 2)))
    out: Dict[str, Any] = {
        "mae_norm": mae_norm,
        "mae_uns": mae_uns,
        "rmse_norm": rmse_norm,
        "rmse_uns": rmse_uns,
        "rmse_norm_x_std": rmse_norm * std,
        "mean_batch_rmse_plan": float(np.mean(batch_rmse_plan)) if batch_rmse_plan else None,
        "mean_batch_rmse_factual": float(np.mean(batch_rmse_fact)) if batch_rmse_fact else None,
        "rmse_factual_norm": rmse_factual_norm,
        "mae_factual_norm": mae_factual_norm,
        "mae_factual_uns": mae_factual_uns,
    }
    if return_series:
        out["iql_y_norm"] = iql_y_norm
        out["true_y_norm"] = true_y_norm
        out["iql_y_uns"] = iql_y_uns
        out["true_y_uns"] = true_y_uns
    return out


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
    debug_panel: bool = False,
    action_diagnostics: bool = False,
    action_grid_points: int = 11,
    action_diag_max_batches: int | None = None,
    sample_seed: int | None = None,
) -> Dict[str, Any]:
    """
    Full pass over ``fold``'s dataloader. The planner is evaluated as a closed-loop
    policy: after each selected action, the dataset simulator returns the next outcome
    and the updated history is used for the next action. Returned RMSE/MAE are computed
    against the fold targets in normalized and unscaled spaces.
    """
    data = fold.data
    max_action = float(planner.cfg.max_action)
    action_selector = str(stable_select(args, "exp.iql_eval_action_selector"))
    action_candidate_actions = int(stable_select(args, "exp.iql_eval_candidate_actions"))
    action_q_bc_penalty = float(stable_select(args, "exp.iql_eval_q_bc_penalty"))
    action_candidate_noise_std = float(stable_select(args, "exp.iql_eval_candidate_noise_std"))
    action_hold_days = max(1, int(OmegaConf.select(args, "dataset.action_hold_days", default=1)))
    scaling_params = dataset_collection.train_scaling_params

    original_exp_tau = int(OmegaConf.select(args, "exp.tau", default=tau))
    args.exp.tau = int(tau)
    try:
        dataloader = get_dataloader(
            CIPDataset(data, args, train=False, sample_seed=sample_seed),
            batch_size=val_batch_size,
            shuffle=False,
        )
    finally:
        args.exp.tau = original_exp_tau

    collect_series = bool(return_series or debug_panel)
    ture_output_list: list = []
    output_after_actions_list: list = []
    factual_output_list: list = []
    batch_rmse_plan: list = []
    batch_rmse_fact: list = []
    action_diag_acc: Dict[str, list] = {}
    action_diag_batches = 0
    action_grid_sim = None
    if action_diagnostics:
        action_grid_sim = _make_action_grid(
            int(planner.cfg.action_dim), int(action_grid_points), device, torch.float32
        )
    debug_payload: Dict[str, Any] | None = {} if debug_panel else None

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

            ture_output = targets["outputs"][:, -1, :].detach().cpu().numpy()
            ture_output_list.append(ture_output)
            debug_info: Dict[str, Any] | None = debug_payload if debug_payload is not None and i == 0 else None

            if autoregressive_eval:
                eval_target = targets["outputs"][:, -1, :]
                H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                a_prev_sim = gather_last_valid(
                    H_work["current_treatments"], H_work.get("active_entries")
                ).clone()
                planned = []
                history_checks = []
                first_obs = None
                held_action_sim = None
                closed_loop_output_after_actions = None
                for step in range(tau):
                    H_work = align_h_t_static_to_history(H_work)
                    z, _, _ = inference_model.ct_hidden_history(H_work)
                    pre_len = None
                    prev_out = None
                    z_before = None
                    if debug_info is not None and len(history_checks) < 2:
                        pre_len = int(H_work["outputs"].size(1))
                        prev_out = gather_last_valid(
                            H_work["outputs"], H_work.get("active_entries")
                        ).detach().clone()
                        z_before = z.detach().clone()
                    a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                    obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
                    if step == 0:
                        first_obs = obs.detach()
                    if action_hold_days > 1 and held_action_sim is not None and (step % action_hold_days) != 0:
                        a_sim = held_action_sim.clone()
                    else:
                        a_raw = select_iql_policy_action(
                            planner,
                            obs,
                            selector=action_selector,
                            candidate_actions=action_candidate_actions,
                            q_bc_penalty=action_q_bc_penalty,
                            candidate_noise_std=action_candidate_noise_std,
                        )
                        a_sim = _policy_to_sim_interval_torch(a_raw, max_action)
                        held_action_sim = a_sim.detach().clone()
                    planned.append(a_sim)
                    y_norm, next_observation = _rollout_one_step(
                        H_work, a_sim,
                        fold=fold, scaling_params=scaling_params, device=device,
                    )
                    _extend_h_work_after_one_step(
                        H_work,
                        a_sim,
                        y_norm,
                        scaling_params,
                        torch.device(device),
                        next_observation=next_observation,
                    )
                    closed_loop_output_after_actions = y_norm.detach().cpu().numpy()
                    if debug_info is not None and len(history_checks) < 2:
                        post_len = int(H_work["outputs"].size(1))
                        appended_action_ok = bool(torch.allclose(
                            H_work["current_treatments"][:, -1, :], a_sim, atol=1e-6, rtol=1e-5
                        ))
                        appended_output_ok = bool(torch.allclose(
                            H_work["outputs"][:, -1, :], y_norm.view(y_norm.size(0), -1), atol=1e-6, rtol=1e-5
                        ))
                        prev_output_ok = bool(torch.allclose(
                            H_work["prev_outputs"][:, -1, :], prev_out, atol=1e-6, rtol=1e-5
                        ))
                        H_probe = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_work.items()}
                        H_probe = align_h_t_static_to_history(H_probe)
                        z_after, _, _ = inference_model.ct_hidden_history(H_probe)
                        z_delta = float(torch.norm(z_after - z_before, dim=-1).mean().item())
                        ok = bool(
                            (post_len == pre_len + 1)
                            and appended_action_ok
                            and appended_output_ok
                            and prev_output_ok
                        )
                        history_checks.append({
                            "step": int(step),
                            "len_before": pre_len,
                            "len_after": post_len,
                            "appended_action_ok": appended_action_ok,
                            "appended_output_ok": appended_output_ok,
                            "prev_output_ok": prev_output_ok,
                            "z_delta_mean_l2": z_delta,
                            "ok": ok,
                        })
                    a_prev_sim = a_sim
                a_seq = torch.stack(planned, dim=1).contiguous()
                if first_obs is None:
                    raise RuntimeError("action diagnostics expected a first observation but rollout produced none.")
                if debug_info is not None:
                    debug_info["history_checks"] = history_checks
                    debug_info["history_updates_ok"] = bool(
                        history_checks and all(bool(item["ok"]) for item in history_checks)
                    )
            else:
                z, _, _ = inference_model.ct_hidden_history(H_t)
                z_np = z.detach().cpu().numpy()
                eval_target_np = targets["outputs"][:, -1, :].detach().cpu().numpy()
                a_prev_raw = gather_last_valid(
                    H_t["current_treatments"], H_t.get("active_entries")
                ).detach().cpu().numpy()
                a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                bsz = z_np.shape[0]
                delta_scalar = float(tau - 0) / max_tau
                delta_vec = np.array([delta_scalar], dtype=np.float32)
                delta_full = torch.full((bsz, 1), delta_scalar, device=device, dtype=torch.float32)
                a_prev_full = torch.as_tensor(a_prev_feat, device=device, dtype=torch.float32)
                first_obs = planner.build_state(z, targets["outputs"][:, -1, :], delta_full, a_prev_full).detach()
                a_rows = []
                for b in range(bsz):
                    z_b = torch.as_tensor(z_np[b:b + 1], device=device, dtype=torch.float32)
                    target_b = torch.as_tensor(eval_target_np[b:b + 1], device=device, dtype=torch.float32)
                    delta_b = torch.as_tensor(delta_vec.reshape(1, 1), device=device, dtype=torch.float32)
                    a_prev_b = torch.as_tensor(a_prev_feat[b:b + 1], device=device, dtype=torch.float32)
                    obs_b = planner.build_state(z_b, target_b, delta_b, a_prev_b)
                    a_b = select_iql_policy_action(
                        planner,
                        obs_b,
                        selector=action_selector,
                        candidate_actions=action_candidate_actions,
                        q_bc_penalty=action_q_bc_penalty,
                        candidate_noise_std=action_candidate_noise_std,
                    )
                    a_rows.append(a_b.detach().cpu().numpy().reshape(-1))
                a_raw = np.stack(a_rows, axis=0)
                a_sim = _actions_to_sim_interval(a_raw, max_action)
                a_seq = (
                    torch.tensor(a_sim, device=device, dtype=torch.float32)
                    .unsqueeze(1)
                    .expand(-1, tau, -1)
                    .contiguous()
                )
                closed_loop_output_after_actions = None
                if debug_info is not None:
                    debug_info["history_checks"] = []
                    debug_info["history_updates_ok"] = True

            if action_diagnostics and action_grid_sim is not None:
                max_diag_batches = action_diag_max_batches
                can_collect = max_diag_batches is None or action_diag_batches < int(max_diag_batches)
                if can_collect:
                    batch_diag = _collect_batch_action_diagnostics(
                        planner, H_t, targets, first_obs, a_seq,
                        tau=tau, fold=fold, scaling_params=scaling_params,
                        action_grid_sim=action_grid_sim, max_action=max_action, device=device,
                    )
                    for key, value in batch_diag.items():
                        _append_action_diag(action_diag_acc, key, value)
                    action_diag_batches += 1

            if debug_info is not None:
                a_seq_np = a_seq.detach().cpu().numpy()
                debug_info["action_sequence"] = {
                    "fingerprint": _fingerprint_np(a_seq_np[:2, :min(tau, 4), :]),
                    "mean": float(a_seq_np.mean()),
                    "std": float(a_seq_np.std()),
                    "min": float(a_seq_np.min()),
                    "max": float(a_seq_np.max()),
                }

            if closed_loop_output_after_actions is None:
                closed_loop_output_after_actions = _simulate_a_seq_final_y(
                    H_t, a_seq,
                    fold=fold, scaling_params=scaling_params,
                )
            output_after_actions = closed_loop_output_after_actions
            output_after_actions_list.append(output_after_actions)

            if include_factual_traj_rmse:
                true_actions = targets["current_treatments"]
                factual_y = _simulate_a_seq_final_y(
                    H_t, true_actions,
                    fold=fold, scaling_params=scaling_params,
                )
                factual_output_list.append(factual_y)
                loss_fact = float(np.sqrt(((factual_y - ture_output) ** 2).mean()))
                batch_rmse_fact.append(loss_fact)

            if log_batches:
                loss = float(np.sqrt(((output_after_actions - ture_output) ** 2).mean()))
                batch_rmse_plan.append(loss)
                if include_factual_traj_rmse:
                    logger.info(
                        f"Batch {i} RMSE (plan): {loss:.6f}, "
                        f"RMSE (factual): {batch_rmse_fact[-1]:.6f}"
                    )
                else:
                    logger.info(f"Batch {i} RMSE (plan): {loss:.6f}")
    finally:
        planner.actor.train(was_training)

    try:
        std = float(dataset_collection.train_scaling_params[1]["cancer_volume"])
    except Exception:
        std = 1.0

    out = _compute_rollout_metrics(
        output_after_actions_list=output_after_actions_list,
        ture_output_list=ture_output_list,
        factual_output_list=(factual_output_list if include_factual_traj_rmse else None),
        scaling_params=scaling_params,
        std=std,
        batch_rmse_plan=batch_rmse_plan,
        batch_rmse_fact=(batch_rmse_fact if include_factual_traj_rmse else None),
        return_series=collect_series,
    )
    if action_diagnostics:
        action_diag = _finalize_action_diagnostics(action_diag_acc)
        if action_diag:
            out["action_diagnostics"] = action_diag

    if debug_payload is not None:
        pred_y_uns = np.asarray(out.get("iql_y_uns"))
        true_y_uns = np.asarray(out.get("true_y_uns"))
        mae_recomputed = float(np.mean(np.abs(pred_y_uns - true_y_uns)))
        same_shape = tuple(pred_y_uns.shape) == tuple(true_y_uns.shape)
        metric_ok = bool(
            same_shape and abs(mae_recomputed - float(out["mae_uns"])) <= 1e-10
        )
        debug_payload["metric_alignment"] = {
            "pred_shape": list(pred_y_uns.shape),
            "true_shape": list(true_y_uns.shape),
            "pred_hash": _fingerprint_np(pred_y_uns),
            "true_hash": _fingerprint_np(true_y_uns),
            "mae_uns_logged": float(out["mae_uns"]),
            "mae_uns_recomputed": mae_recomputed,
            "mae_uns_abs_diff": float(abs(mae_recomputed - float(out["mae_uns"]))),
            "same_shape": same_shape,
        }
        debug_payload["summary_flags"] = {
            "history_updates_ok": bool(debug_payload.get("history_updates_ok", True)),
            "action_sequence_changes": True,
            "metric_alignment_ok": bool(metric_ok),
        }
        if not return_series:
            out.pop("iql_y_norm", None)
            out.pop("true_y_norm", None)
            out.pop("iql_y_uns", None)
            out.pop("true_y_uns", None)
        out["debug_panel"] = debug_payload
    return out
