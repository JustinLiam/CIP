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

# Valid values for the ``world`` argument used by A/B rollout helpers.
#   - "sim":       use the cancer simulator (oracle, synthetic-benchmark only).
#   - "predictor": use the learned OutcomePredictor loaded from ct_best_encoder.pt
#                  (model-based OPE; works on real data too).
_VALID_WORLDS = ("sim", "predictor")


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


def _rollout_one_step_y(
    world: str,
    H_work: Dict,
    a_sim: torch.Tensor,
    *,
    fold,
    scaling_params,
    inference_model: InferenceModel,
    device: str,
) -> torch.Tensor:
    """
    One-step outcome y_{t+1} in normalized (train-scaling) space given current H_work
    and a simulator-interval action a_sim in [0, 1].

    - world="sim":       calls the oracle ``simulate_output_after_actions``.
    - world="predictor": uses the jointly-trained ``OutcomePredictor`` on (Z_t, a_sim).

    Returns a ``[B, 1]`` (or ``[B, y_dim]``) tensor on ``device``.
    """
    if world == "sim":
        y_np = fold.simulate_output_after_actions(H_work, a_sim.unsqueeze(1), scaling_params)
        return torch.as_tensor(y_np, device=device, dtype=torch.float32)
    if world == "predictor":
        if not getattr(inference_model, "_outcome_predictor_loaded", False):
            logger.warning(
                "predictor-world rollout requested but outcome_predictor weights were "
                "not loaded; results reflect random-initialized predictor."
            )
        z, _, _ = inference_model.ct_hidden_history(H_work)
        za = torch.cat([z, a_sim], dim=-1)
        y_norm = inference_model.outcome_predictor(za)
        return y_norm.detach().to(device=device, dtype=torch.float32)
    raise ValueError(f"unknown world: {world!r}; expected one of {_VALID_WORLDS}")


def _simulate_a_seq_final_y(
    world: str,
    H_t: Dict,
    a_seq: torch.Tensor,
    *,
    fold,
    scaling_params,
    inference_model: InferenceModel,
    device: str,
    mean_ser,
    std_ser,
) -> np.ndarray:
    """
    Closed-loop rollout of ``a_seq`` (shape ``[B, tau, A]``) starting from ``H_t`` and
    return y_{t+tau} as a ``[B, 1]`` numpy array in normalized (train-scaling) space.

    For the "sim" world this is a single oracle call (same noise resampling as before).
    For the "predictor" world we manually unroll predictor one step at a time, mirroring
    ``_extend_h_work_after_one_step`` so z_t gets refreshed against the updated H_work.
    """
    if world == "sim":
        return fold.simulate_output_after_actions(H_t, a_seq, scaling_params)
    if world == "predictor":
        H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
        tau_local = int(a_seq.size(1))
        y_last: torch.Tensor | None = None
        for step in range(tau_local):
            H_work = align_h_t_static_to_history(H_work)
            a_step = a_seq[:, step, :].contiguous()
            y_norm = _rollout_one_step_y(
                "predictor", H_work, a_step,
                fold=fold, scaling_params=scaling_params,
                inference_model=inference_model, device=device,
            )
            _extend_h_work_after_one_step(H_work, a_step, y_norm, mean_ser, std_ser, torch.device(device))
            y_last = y_norm
        assert y_last is not None
        return y_last.detach().cpu().numpy()
    raise ValueError(f"unknown world: {world!r}; expected one of {_VALID_WORLDS}")


def _compute_world_metrics(
    output_after_actions_list: list,
    ture_output_list: list,
    factual_output_list: list | None,
    mean_ser,
    std_ser,
    std: float,
    batch_rmse_plan: list,
    batch_rmse_fact: list | None,
    return_series: bool,
) -> Dict[str, Any]:
    """Aggregate a single world's per-batch results into MAE/RMSE metrics."""
    pred_arr = np.concatenate(output_after_actions_list, axis=0)
    true_arr = np.concatenate(ture_output_list, axis=0)

    rmse_norm = float(np.sqrt(((pred_arr - true_arr) ** 2).mean()))
    rmse_factual_norm = None
    mae_factual_norm = None
    mae_factual_uns = None
    if factual_output_list is not None and len(factual_output_list) > 0:
        fact_arr = np.concatenate(factual_output_list, axis=0)
        rmse_factual_norm = float(np.sqrt(((fact_arr - true_arr) ** 2).mean()))
        fact_y_uns = _unscaled_cancer_volume_np(fact_arr, mean_ser, std_ser).reshape(-1)
        true_y_uns_full = _unscaled_cancer_volume_np(true_arr, mean_ser, std_ser).reshape(-1)
        mae_factual_norm = float(np.mean(np.abs(fact_arr.reshape(-1) - true_arr.reshape(-1))))
        mae_factual_uns = float(np.mean(np.abs(fact_y_uns - true_y_uns_full)))

    iql_y_norm = pred_arr.reshape(-1)
    true_y_norm = true_arr.reshape(-1)
    iql_y_uns = _unscaled_cancer_volume_np(pred_arr, mean_ser, std_ser).reshape(-1)
    true_y_uns = _unscaled_cancer_volume_np(true_arr, mean_ser, std_ser).reshape(-1)

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
    worlds: tuple = ("sim",),
) -> Dict[str, Any]:
    """
    Full pass over ``fold``'s dataloader; returns global MAE/RMSE (normalized and unscaled tumor volume).

    ``worlds`` controls which dynamics are used for the closed-loop rollout:

      - "sim":       the synthetic cancer simulator oracle (legacy behavior).
      - "predictor": the learned OutcomePredictor loaded from ``ct_best_encoder.pt``
                     (model-based OPE; valid on real data).

    When a single world is requested, the returned dict is backward compatible (flat
    metrics at the top level). When multiple are requested, top-level metrics correspond
    to the first world in ``worlds`` and ``out["per_world"]`` holds a
    ``{world_name: metric_dict}`` mapping.

    RNG alignment: within each batch, the NumPy global state is snapshotted after the
    DataLoader has drawn ``__getitem__`` samples, then restored before every world's
    rollout. That way both worlds see identical H_t/targets and the simulator noise
    samples from the same starting state in every world (the predictor world happens
    to be deterministic, but we still restore to keep state transitions identical
    across subsequent batches).
    """
    assert len(worlds) >= 1, "worlds must contain at least one entry"
    for w in worlds:
        if w not in _VALID_WORLDS:
            raise ValueError(f"unknown world {w!r}; valid: {_VALID_WORLDS}")

    data = fold.data
    max_action = float(planner.cfg.max_action)
    mean_ser, std_ser = dataset_collection.train_scaling_params
    scaling_params = dataset_collection.train_scaling_params

    dataloader = get_dataloader(CIPDataset(data, args, train=False), batch_size=val_batch_size, shuffle=False)

    ture_output_list: list = []
    per_world_pred: Dict[str, list] = {w: [] for w in worlds}
    per_world_fact: Dict[str, list] = {w: [] for w in worlds} if include_factual_traj_rmse else {w: [] for w in worlds}
    per_world_batch_rmse_plan: Dict[str, list] = {w: [] for w in worlds}
    per_world_batch_rmse_fact: Dict[str, list] = {w: [] for w in worlds}

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

            np_state_pre = np.random.get_state()
            torch_state_pre = torch.get_rng_state()

            for world in worlds:
                np.random.set_state(np_state_pre)
                torch.set_rng_state(torch_state_pre)

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
                        y_norm = _rollout_one_step_y(
                            world, H_work, a_sim,
                            fold=fold, scaling_params=scaling_params,
                            inference_model=inference_model, device=device,
                        )
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

                output_after_actions = _simulate_a_seq_final_y(
                    world, H_t, a_seq,
                    fold=fold, scaling_params=scaling_params,
                    inference_model=inference_model, device=device,
                    mean_ser=mean_ser, std_ser=std_ser,
                )
                per_world_pred[world].append(output_after_actions)

                if include_factual_traj_rmse:
                    true_actions = targets["current_treatments"]
                    factual_y = _simulate_a_seq_final_y(
                        world, H_t, true_actions,
                        fold=fold, scaling_params=scaling_params,
                        inference_model=inference_model, device=device,
                        mean_ser=mean_ser, std_ser=std_ser,
                    )
                    per_world_fact[world].append(factual_y)
                    loss_fact = float(np.sqrt(((factual_y - ture_output) ** 2).mean()))
                    per_world_batch_rmse_fact[world].append(loss_fact)

                if log_batches:
                    loss = float(np.sqrt(((output_after_actions - ture_output) ** 2).mean()))
                    per_world_batch_rmse_plan[world].append(loss)
                    if include_factual_traj_rmse:
                        logger.info(
                            f"Batch {i} [world={world}] RMSE (plan): {loss:.6f}, "
                            f"RMSE (factual): {per_world_batch_rmse_fact[world][-1]:.6f}"
                        )
                    else:
                        logger.info(f"Batch {i} [world={world}] RMSE (plan): {loss:.6f}")
    finally:
        planner.actor.train(was_training)

    try:
        std = float(dataset_collection.train_scaling_params[1]["cancer_volume"])
    except Exception:
        std = 1.0

    per_world_metrics: Dict[str, Dict[str, Any]] = {}
    for world in worlds:
        per_world_metrics[world] = _compute_world_metrics(
            output_after_actions_list=per_world_pred[world],
            ture_output_list=ture_output_list,
            factual_output_list=(per_world_fact[world] if include_factual_traj_rmse else None),
            mean_ser=mean_ser,
            std_ser=std_ser,
            std=std,
            batch_rmse_plan=per_world_batch_rmse_plan[world],
            batch_rmse_fact=per_world_batch_rmse_fact[world] if include_factual_traj_rmse else None,
            return_series=return_series,
        )

    primary = worlds[0]
    out: Dict[str, Any] = dict(per_world_metrics[primary])
    if len(worlds) > 1:
        out["per_world"] = per_world_metrics
    return out
