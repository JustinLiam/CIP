"""
RQ3 diagnostic: behavioral support and value extrapolation on Tumor.

The script is evaluation-only. It compares the checkpoint actor with a global
critic-grid maximizer while keeping the encoder, critic, simulator, targets,
and evaluation histories fixed.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runnables.plot_iql_action_support import _load_models, _prepare_dataset
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.ct_transition_dataset import CTEstepDataset, collate_ct_estep_batch
from src.data.iql_dataset_builder import align_h_t_static_to_history
from src.data.iql_raw_transition_dataset import _sample_target_indices
from src.evaluation.iql_action_selection import select_iql_policy_action
from src.evaluation.iql_action_support import (
    SupportIndex,
    deduplicate_context_actions,
    make_action_grid_sim,
    q_grid_argmax_action,
)
from src.evaluation.iql_planner_eval import (
    _build_decision_history_view,
    _extend_h_work_after_one_step,
    _iql_augmented_state,
    _policy_to_sim_interval_torch,
    _rollout_one_step,
    _sim_actions_to_tanh_batch,
    _unscaled_cancer_volume_np,
)
from src.models.sequence_utils import gather_last_valid
from src.utils.stable_iql_em_defaults import stable_select
from src.utils.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rq3_support_value")

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _clone_dict(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _as_int_list(value: Any, default: Iterable[int]) -> List[int]:
    if value is None:
        return [int(x) for x in default]
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        text = value.strip().strip("[]")
        value = [part.strip() for part in text.split(",") if part.strip()]
    return [int(x) for x in value]


def _resolve_output_dir(args: DictConfig, original_cwd: Path) -> Path:
    raw = str(OmegaConf.select(args, "exp.rq3_out_dir"))
    path = Path(raw)
    if not path.is_absolute():
        path = original_cwd / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_loader(
    args: DictConfig,
    data: Dict[str, np.ndarray],
    *,
    tau: int,
    batch_size: int,
    sample_seed: int,
):
    old_tau = int(OmegaConf.select(args, "exp.tau", default=tau))
    args.exp.tau = int(tau)
    try:
        dataset = CIPDataset(data, args, train=False, sample_seed=sample_seed)
        loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
        n_samples = len(dataset)
        history_lengths = np.asarray(dataset.history_lengths, dtype=np.int64)
    finally:
        args.exp.tau = old_tau
    return loader, n_samples, history_lengths


@torch.no_grad()
def _build_support_index(
    *,
    train_data: Dict[str, np.ndarray],
    inference_model,
    planner,
    device: str,
    batch_size: int,
    k: int,
    cache_path: Path,
) -> Tuple[SupportIndex, Dict[str, Any]]:
    action_dim = int(planner.cfg.action_dim)
    output_dim = int(planner.cfg.output_dim or 1)
    z_dim = int(planner.cfg.state_dim) - output_dim - 1 - action_dim
    if z_dim <= 0:
        raise ValueError(f"Invalid z_dim={z_dim}")

    if cache_path.exists():
        obj = np.load(cache_path, allow_pickle=False)
        index = SupportIndex.from_arrays(
            context_raw=obj["context_raw"],
            behavior_actions_sim=obj["behavior_actions_sim"],
            z_dim=z_dim,
            action_dim=action_dim,
            include_delta=False,
            k=k,
            z_weight=1.0,
            prev_action_weight=1.0,
            delta_weight=0.5,
            size_before_dedup=int(obj["size_before_dedup"].item()),
            deduplicate=True,
            context_mean=obj["context_mean"],
            context_std=obj["context_std"],
        )
        return index, {
            "cache_hit": True,
            "size_before_dedup": int(obj["size_before_dedup"].item()),
            "size_after_dedup": int(index.size_after_dedup),
            "used_sklearn": bool(index.used_sklearn),
        }

    transition_dataset = CTEstepDataset(train_data)
    loader = DataLoader(
        transition_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ct_estep_batch,
    )
    contexts: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    inference_model.eval()
    for batch_idx, batch in enumerate(loader):
        H_t = align_h_t_static_to_history(_to_device(batch["H_t"], device))
        z, _, _ = inference_model.ct_hidden_history(H_t)
        patient_idx = batch["patient_index"].cpu().numpy()
        time_idx = batch["time_index"].cpu().numpy()
        previous = train_data["current_treatments"][patient_idx, time_idx - 1, :]
        current = train_data["current_treatments"][patient_idx, time_idx, :]
        contexts.append(
            np.concatenate(
                [z.detach().cpu().numpy(), previous.astype(np.float32)],
                axis=1,
            ).astype(np.float32)
        )
        actions.append(current.astype(np.float32))
        if (batch_idx + 1) % 25 == 0:
            logger.info(
                "Support encoding: %d/%d batches",
                batch_idx + 1,
                len(loader),
            )

    context_raw = np.concatenate(contexts, axis=0).astype(np.float32)
    behavior_actions = np.concatenate(actions, axis=0).astype(np.float32)
    size_before = int(context_raw.shape[0])
    context_raw, behavior_actions, _ = deduplicate_context_actions(
        context_raw,
        behavior_actions,
        decimals=6,
    )
    index = SupportIndex.from_arrays(
        context_raw=context_raw,
        behavior_actions_sim=behavior_actions,
        z_dim=z_dim,
        action_dim=action_dim,
        include_delta=False,
        k=k,
        z_weight=1.0,
        prev_action_weight=1.0,
        delta_weight=0.5,
        size_before_dedup=size_before,
        deduplicate=True,
    )
    np.savez_compressed(
        cache_path,
        context_raw=index.context_raw,
        behavior_actions_sim=index.behavior_actions_sim,
        context_mean=index.context_mean,
        context_std=index.context_std,
        size_before_dedup=np.asarray(size_before, dtype=np.int64),
    )
    return index, {
        "cache_hit": False,
        "size_before_dedup": size_before,
        "size_after_dedup": int(index.size_after_dedup),
        "used_sklearn": bool(index.used_sklearn),
    }


def _training_reward_std(
    *,
    data: Dict[str, np.ndarray],
    seed: int,
    max_tau: float,
    target_horizons: List[int],
    samples_per_transition: int,
    reward_clip: float,
    decision_interval_days: int,
) -> Tuple[float, Dict[str, float]]:
    rng = np.random.RandomState(int(seed))
    rewards: List[float] = []
    n_patients = int(data["current_treatments"].shape[0])
    for i in range(n_patients):
        length = int(data["active_entries"][i].sum())
        if length < 3:
            continue
        last_idx = length - 1
        for t in range(1, length - 1):
            if int(decision_interval_days) > 1:
                day = (
                    int(np.asarray(data["sim_day"][i, t]).reshape(-1)[0])
                    if "sim_day" in data
                    else int(t)
                )
                if day % int(decision_interval_days) != 0:
                    continue
            indices = _sample_target_indices(
                t=t,
                last_idx=last_idx,
                max_tau=max_tau,
                samples_per_transition=samples_per_transition,
                rng=rng,
                target_sampling="horizon_aligned",
                target_horizons=target_horizons,
            )
            y_next = data["outputs"][i, t, :].astype(np.float32)
            for t_target in indices:
                target = data["outputs"][i, t_target, :].astype(np.float32)
                reward = -float(np.mean(np.abs(y_next - target)))
                if reward_clip > 0:
                    reward = float(np.clip(reward, -reward_clip, reward_clip))
                rewards.append(reward)
    arr = np.asarray(rewards, dtype=np.float32)
    std = float(arr.std()) + 1e-8
    return std, {
        "count": int(arr.size),
        "raw_mean": float(arr.mean()),
        "raw_std": float(arr.std()),
        "raw_min": float(arr.min()),
        "raw_max": float(arr.max()),
        "scaled_mean": float((arr / std).mean()),
        "scaled_std": float((arr / std).std()),
    }


def _support_context(
    support_index: SupportIndex,
    z: torch.Tensor,
    previous_action_sim: torch.Tensor,
) -> np.ndarray:
    return support_index.build_context(
        z.detach().cpu().numpy(),
        previous_action_sim.detach().cpu().numpy(),
        None,
    )


def _action_min_distance(
    support_index: SupportIndex,
    context_raw: np.ndarray,
    action_sim: torch.Tensor,
) -> np.ndarray:
    local_actions = support_index.query(context_raw)["actions"]
    action_np = action_sim.detach().cpu().numpy()
    return np.linalg.norm(local_actions - action_np[:, None, :], axis=2).min(axis=1)


@torch.no_grad()
def _evaluate_rollout(
    *,
    method: str,
    planner,
    inference_model,
    dataset_collection,
    fold,
    args: DictConfig,
    support_index: SupportIndex,
    device: str,
    tau: int,
    max_tau: float,
    sample_seed: int,
    batch_size: int,
    action_grid_sim: torch.Tensor,
    q_chunk_size: int,
    reward_std: float,
    reward_clip: float,
    discount: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    set_seed(int(sample_seed))
    loader, expected_n, history_lengths = _make_loader(
        args,
        fold.data,
        tau=tau,
        batch_size=batch_size,
        sample_seed=sample_seed,
    )
    scaling_params = dataset_collection.train_scaling_params
    mean_ser, std_ser = scaling_params
    max_action = float(planner.cfg.max_action)
    selector = str(stable_select(args, "exp.iql_eval_action_selector"))
    candidate_actions = int(stable_select(args, "exp.iql_eval_candidate_actions"))
    q_bc_penalty = float(stable_select(args, "exp.iql_eval_q_bc_penalty"))
    candidate_noise_std = float(stable_select(args, "exp.iql_eval_candidate_noise_std"))

    all_actions: List[np.ndarray] = []
    all_distances: List[np.ndarray] = []
    all_q: List[np.ndarray] = []
    all_rewards: List[np.ndarray] = []
    all_returns: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_target: List[np.ndarray] = []

    for batch_idx, (H_t, targets) in enumerate(loader):
        H_t = align_h_t_static_to_history(_to_device(H_t, device))
        targets = _to_device(targets, device)
        eval_target = targets["outputs"][:, -1, :]
        H_work = _clone_dict(H_t)
        previous_action_sim = gather_last_valid(
            H_work["current_treatments"],
            H_work.get("active_entries"),
        ).clone()

        step_actions: List[np.ndarray] = []
        step_distances: List[np.ndarray] = []
        step_q: List[np.ndarray] = []
        step_rewards: List[np.ndarray] = []
        y_last: Optional[torch.Tensor] = None

        for step in range(int(tau)):
            H_work = align_h_t_static_to_history(H_work)
            H_policy = align_h_t_static_to_history(_build_decision_history_view(H_work))
            z, _, _ = inference_model.ct_hidden_history(H_policy)
            previous_action_policy = _sim_actions_to_tanh_batch(
                previous_action_sim,
                max_action,
            )
            obs = _iql_augmented_state(
                planner,
                z,
                eval_target,
                step,
                tau,
                max_tau,
                previous_action_policy,
            )
            context = _support_context(support_index, z, previous_action_sim)

            if method == "actor":
                action_policy = select_iql_policy_action(
                    planner,
                    obs,
                    selector=selector,
                    candidate_actions=candidate_actions,
                    q_bc_penalty=q_bc_penalty,
                    candidate_noise_std=candidate_noise_std,
                )
                action_sim = _policy_to_sim_interval_torch(action_policy, max_action)
            elif method == "qgrid":
                action_sim = q_grid_argmax_action(
                    planner,
                    obs,
                    action_grid_sim,
                    max_action,
                    device=device,
                    q_chunk_size=q_chunk_size,
                )
                action_policy = _sim_actions_to_tanh_batch(action_sim, max_action)
            elif method == "factual":
                action_sim = targets["current_treatments"][:, step, :].contiguous()
                action_policy = _sim_actions_to_tanh_batch(action_sim, max_action)
            else:
                raise ValueError(f"Unknown method={method!r}")

            q_value = planner.qf(obs, action_policy)
            min_distance = _action_min_distance(
                support_index,
                context,
                action_sim,
            )
            y_last, next_observation = _rollout_one_step(
                H_work,
                action_sim,
                fold=fold,
                scaling_params=scaling_params,
                device=device,
            )
            raw_reward = -torch.mean(torch.abs(y_last - eval_target), dim=-1)
            if reward_clip > 0:
                raw_reward = torch.clamp(raw_reward, -reward_clip, reward_clip)
            scaled_reward = raw_reward / float(reward_std)

            step_actions.append(action_sim.detach().cpu().numpy())
            step_distances.append(min_distance.astype(np.float32))
            step_q.append(q_value.detach().cpu().numpy().astype(np.float32))
            step_rewards.append(scaled_reward.detach().cpu().numpy().astype(np.float32))
            _extend_h_work_after_one_step(
                H_work,
                action_sim,
                y_last,
                scaling_params,
                torch.device(device),
                next_observation=next_observation,
            )
            previous_action_sim = action_sim

        if y_last is None:
            raise RuntimeError("Rollout produced no outcomes")
        reward_array = np.stack(step_rewards, axis=1).astype(np.float32)
        return_array = np.zeros_like(reward_array)
        running = np.zeros(reward_array.shape[0], dtype=np.float32)
        for step in range(int(tau) - 1, -1, -1):
            running = reward_array[:, step] + float(discount) * running
            return_array[:, step] = running

        all_actions.append(np.stack(step_actions, axis=1).astype(np.float32))
        all_distances.append(np.stack(step_distances, axis=1).astype(np.float32))
        all_q.append(np.stack(step_q, axis=1).astype(np.float32))
        all_rewards.append(reward_array)
        all_returns.append(return_array)
        all_pred.append(y_last.detach().cpu().numpy().astype(np.float32))
        all_target.append(eval_target.detach().cpu().numpy().astype(np.float32))
        # The paper's standard Tumor evaluation also replays the factual action
        # sequence after every planned batch. Besides producing its diagnostic
        # RMSE, that call advances the stochastic simulator RNG before the next
        # CIP batch. Retain it here so actor RMSE and sampled histories reproduce
        # the main evaluation path exactly.
        fold.simulate_output_after_actions(
            H_t,
            targets["current_treatments"],
            scaling_params,
        )
        if (batch_idx + 1) % 5 == 0:
            logger.info("%s rollout: %d/%d batches", method, batch_idx + 1, len(loader))

    actions = np.concatenate(all_actions, axis=0)
    distances = np.concatenate(all_distances, axis=0)
    q_values = np.concatenate(all_q, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    returns = np.concatenate(all_returns, axis=0)
    pred_norm = np.concatenate(all_pred, axis=0)
    target_norm = np.concatenate(all_target, axis=0)
    pred_uns = _unscaled_cancer_volume_np(pred_norm, mean_ser, std_ser).reshape(-1)
    target_uns = _unscaled_cancer_volume_np(target_norm, mean_ser, std_ser).reshape(-1)
    q_over = np.maximum(q_values - returns, 0.0)

    arrays = {
        "actions": actions,
        "min_distance": distances,
        "q": q_values,
        "reward": rewards,
        "sim_return": returns,
        "q_over_positive": q_over,
        "pred_y_norm": pred_norm,
        "target_y_norm": target_norm,
    }
    summary = {
        "method": method,
        "n_samples": int(actions.shape[0]),
        "n_decisions": int(distances.size),
        "expected_n_samples": int(expected_n),
        "history_lengths": history_lengths.tolist(),
        "rmse_norm": float(np.sqrt(np.mean((pred_norm - target_norm) ** 2))),
        "rmse_uns": float(np.sqrt(np.mean((pred_uns - target_uns) ** 2))),
        "q_over": float(q_over.mean()),
        "q_mae": float(np.mean(np.abs(q_values - returns))),
        "q_bias": float(np.mean(q_values - returns)),
        "q_mean": float(q_values.mean()),
        "sim_return_mean": float(returns.mean()),
        "min_distance_mean": float(distances.mean()),
        "min_distance_p95": float(np.quantile(distances, 0.95)),
    }
    return arrays, summary


def _git_head(original_cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(original_cwd),
            text=True,
        ).strip()
    except Exception:
        return ""


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    original_cwd = Path(get_original_cwd())
    out_dir = _resolve_output_dir(args, original_cwd)
    seed = int(args.exp.seed)
    tau = int(OmegaConf.select(args, "exp.rq3_tau", default=12))
    max_tau = float(stable_select(args, "exp.max_tau"))
    batch_size = int(OmegaConf.select(args, "exp.rq3_batch_size", default=128))
    support_batch_size = int(OmegaConf.select(args, "exp.rq3_support_batch_size", default=512))
    support_k = int(OmegaConf.select(args, "exp.rq3_support_k", default=256))
    threshold_quantile = float(
        OmegaConf.select(args, "exp.rq3_threshold_quantile", default=0.95)
    )
    grid_points = int(OmegaConf.select(args, "exp.rq3_grid_points", default=31))
    q_chunk_size = int(OmegaConf.select(args, "exp.rq3_q_chunk_size", default=8192))
    device = str(OmegaConf.select(args, "exp.device", default="cuda"))
    expected_beta = float(OmegaConf.select(args, "exp.rq3_expected_beta", default=2.0))

    set_seed(seed)
    dataset_collection = _prepare_dataset(args)
    inference_model, planner, planner_ckpt, encoder_ckpt, is_em = _load_models(
        args,
        original_cwd,
        device,
    )
    if not is_em:
        raise ValueError("RQ3 diagnostic requires a combined Full CRIPO EM checkpoint")
    if abs(float(planner.cfg.beta) - expected_beta) > 1e-8:
        raise ValueError(
            f"Checkpoint beta={planner.cfg.beta} does not match expected beta={expected_beta}"
        )
    if str(planner.cfg.actor_update).lower() != "awr":
        raise ValueError(f"Expected AWR actor, got {planner.cfg.actor_update}")

    cache_path = out_dir / "support_index.npz"
    support_index, support_meta = _build_support_index(
        train_data=dataset_collection.train_f.data,
        inference_model=inference_model,
        planner=planner,
        device=device,
        batch_size=support_batch_size,
        k=support_k,
        cache_path=cache_path,
    )

    reward_clip = float(stable_select(args, "exp.iql_reward_clip"))
    reward_scale = str(stable_select(args, "exp.iql_reward_scale")).lower()
    if reward_scale != "auto":
        raise ValueError(f"Expected reward_scale=auto, got {reward_scale}")
    reward_type = str(stable_select(args, "exp.iql_reward_type")).lower()
    if reward_type != "negative_outcome":
        raise ValueError(f"Expected negative_outcome reward, got {reward_type}")
    target_horizons = _as_int_list(
        stable_select(args, "exp.iql_target_horizons"),
        default=range(1, 7),
    )
    samples_per_transition = int(
        stable_select(args, "exp.em_her_samples_per_transition")
    )
    decision_interval_days = int(
        stable_select(args, "exp.iql_decision_interval_days")
    )
    reward_std, reward_meta = _training_reward_std(
        data=dataset_collection.train_f.data,
        seed=seed,
        max_tau=max_tau,
        target_horizons=target_horizons,
        samples_per_transition=samples_per_transition,
        reward_clip=reward_clip,
        decision_interval_days=decision_interval_days,
    )
    logger.info("Training reward reconstruction: %s", reward_meta)

    action_grid_sim = make_action_grid_sim(
        int(planner.cfg.action_dim),
        grid_points,
        device=device,
        dtype=torch.float32,
    )
    factual_arrays, factual_summary = _evaluate_rollout(
        method="factual",
        planner=planner,
        inference_model=inference_model,
        dataset_collection=dataset_collection,
        fold=dataset_collection.val_f,
        args=args,
        support_index=support_index,
        device=device,
        tau=tau,
        max_tau=max_tau,
        sample_seed=seed,
        batch_size=batch_size,
        action_grid_sim=action_grid_sim,
        q_chunk_size=q_chunk_size,
        reward_std=reward_std,
        reward_clip=reward_clip,
        discount=float(planner.cfg.discount),
    )
    threshold = float(
        np.quantile(factual_arrays["min_distance"], threshold_quantile)
    )
    factual_summary["low_support_threshold"] = threshold
    factual_summary["low_support_rate"] = float(
        np.mean(factual_arrays["min_distance"] > threshold)
    )
    np.savez_compressed(out_dir / "validation_factual.npz", **factual_arrays)

    method_summaries: Dict[str, Any] = {}
    for method in ("actor", "qgrid"):
        arrays, summary = _evaluate_rollout(
            method=method,
            planner=planner,
            inference_model=inference_model,
            dataset_collection=dataset_collection,
            fold=dataset_collection.test_f,
            args=args,
            support_index=support_index,
            device=device,
            tau=tau,
            max_tau=max_tau,
            sample_seed=seed,
            batch_size=batch_size,
            action_grid_sim=action_grid_sim,
            q_chunk_size=q_chunk_size,
            reward_std=reward_std,
            reward_clip=reward_clip,
            discount=float(planner.cfg.discount),
        )
        summary["low_support_threshold"] = threshold
        summary["low_support_rate"] = float(
            np.mean(arrays["min_distance"] > threshold)
        )
        method_summaries[method] = summary
        np.savez_compressed(out_dir / f"test_{method}.npz", **arrays)
        logger.info("%s summary: %s", method, summary)

    metadata = {
        "protocol": {
            "dataset": "Tumor",
            "kappa": int(args.dataset.coeff),
            "tau": tau,
            "split_threshold": "validation factual actions",
            "threshold_statistic": f"quantile_{threshold_quantile:.3f}",
            "support_definition": (
                "minimum Euclidean action distance among K nearest training "
                "behavior contexts; context=(encoded history, previous action)"
            ),
            "support_k": support_k,
            "grid_points_per_action_dim": grid_points,
            "q_over_definition": "mean(max(Q_hat(s,a)-discounted_simulator_return,0))",
            "q_over_scope": "all test rollout decision points",
            "standard_eval_factual_replay": True,
            "reward_type": reward_type,
            "reward_scale": reward_scale,
            "reward_clip": reward_clip,
            "discount": float(planner.cfg.discount),
            "target_horizons_training": target_horizons,
            "sample_seed": seed,
        },
        "checkpoint": {
            "path": str(planner_ckpt),
            "encoder_path": str(encoder_ckpt),
            "beta": float(planner.cfg.beta),
            "actor_update": str(planner.cfg.actor_update),
            "outer_iter": int(
                torch.load(str(planner_ckpt), map_location="cpu").get(
                    "outer_iter",
                    -1,
                )
            ),
        },
        "support_index": support_meta,
        "training_reward": reward_meta,
        "validation_factual": factual_summary,
        "test": method_summaries,
        "git_head": _git_head(original_cwd),
    }
    with (out_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)

    with (out_dir / "result.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "method",
                "low_support_rate",
                "q_over",
                "q_mae",
                "q_bias",
                "rmse_uns",
                "rmse_norm",
                "n_samples",
                "n_decisions",
                "threshold",
            ],
        )
        writer.writeheader()
        for method, summary in method_summaries.items():
            writer.writerow(
                {
                    "seed": seed,
                    "method": method,
                    "low_support_rate": summary["low_support_rate"],
                    "q_over": summary["q_over"],
                    "q_mae": summary["q_mae"],
                    "q_bias": summary["q_bias"],
                    "rmse_uns": summary["rmse_uns"],
                    "rmse_norm": summary["rmse_norm"],
                    "n_samples": summary["n_samples"],
                    "n_decisions": summary["n_decisions"],
                    "threshold": threshold,
                }
            )
    logger.info("RQ3 result saved to %s", out_dir / "result.json")


if __name__ == "__main__":
    main()
