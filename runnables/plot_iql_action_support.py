"""
Create local empirical action-support diagnostics for the IQL planner.

Example:
  python runnables/plot_iql_action_support.py \
    +dataset=cancer_sim_cont +model=vcip \
    exp.test=false exp.tau=6 \
    +exp.action_support_K=256 \
    +exp.action_support_r=0.075 \
    +exp.action_support_eta=0.01 \
    +exp.action_support_include_delta=false \
    +exp.action_support_eval_batch_size=128 \
    +exp.action_support_max_eval_samples=512 \
    +exp.action_support_grid_points=31 \
    +exp.action_support_q_chunk_size=8192 \
    +exp.action_support_out_dir=plots/iql_action_support

The diagnostics are visualization-only and do not modify training behavior.
All action-support quantities are empirical local action support in simulator
action space [0, 1], not global action-range diagnostics.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader  # noqa: E402
from src.data.iql_dataset_builder import align_h_t_static_to_history, build_iql_transitions_from_ct  # noqa: E402
from src.evaluation.iql_action_selection import select_iql_policy_action  # noqa: E402
from src.evaluation.iql_action_support import (  # noqa: E402
    SupportIndex,
    batch_local_support_metrics,
    local_support_metrics,
    make_action_grid_sim,
    q_grid_argmax_action,
)
from src.evaluation.iql_planner_eval import (  # noqa: E402
    _build_decision_history_view,
    _extend_h_work_after_one_step,
    _iql_augmented_state,
    _policy_to_sim_interval_torch,
    _rollout_one_step_y,
    _sim_actions_to_tanh_batch,
    _unscaled_cancer_volume_np,
)
from src.models.inference_model import InferenceModel  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.planners.iql_planner import IQLPlanner  # noqa: E402
from src.utils.em_ckpt import is_em_checkpoint, load_em_for_eval  # noqa: E402
from src.utils.inference_ckpt import load_inference_checkpoint  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _hash_json(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _resolve_path(value: str, original_cwd: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else original_cwd / p


def _list_from_config(value: Any, default: Optional[Iterable[int]] = None) -> Optional[List[int]]:
    if value is None:
        return None if default is None else [int(x) for x in default]
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None if default is None else [int(x) for x in default]
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        value = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in value]


def _array_fingerprint(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha1": h.hexdigest()[:16]}


def _data_fingerprint(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    keys = [
        "current_treatments",
        "prev_treatments",
        "outputs",
        "active_entries",
        "static_features",
    ]
    return {k: _array_fingerprint(data[k]) for k in keys if k in data}


def _checkpoint_fingerprint(path_value: str | Path | None, original_cwd: Path) -> Dict[str, Any]:
    if path_value is None or str(path_value).strip() == "":
        return {"path": "", "exists": False}
    path = _resolve_path(str(path_value), original_cwd)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolve_iql_ckpt(args: DictConfig, original_cwd: Path) -> Path:
    explicit = str(OmegaConf.select(args, "exp.iql_eval_ckpt", default="")).strip()
    if explicit:
        return _resolve_path(explicit, original_cwd)
    seed = int(args.exp.seed)
    gamma = int(args.dataset.coeff)
    return original_cwd / "iql_models" / f"seed_{seed}" / f"gamma_{gamma}" / "iql_planner.pt"


def _prepare_dataset(args: DictConfig):
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(args["dataset"]["static_size"]) > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)
    return dataset_collection


def _load_models(args: DictConfig, original_cwd: Path, device: str):
    inference_model = InferenceModel(args).to(device)
    em_eval_ckpt = str(OmegaConf.select(args, "exp.em_eval_ckpt", default="")).strip()
    planner_path = _resolve_iql_ckpt(args, original_cwd)
    em_path = _resolve_path(em_eval_ckpt, original_cwd) if em_eval_ckpt else planner_path

    use_em = False
    if em_path.is_file():
        probe = torch.load(str(em_path), map_location="cpu")
        use_em = is_em_checkpoint(probe)

    if use_em:
        planner = load_em_for_eval(inference_model, str(em_path), device)
        encoder_checkpoint = em_path
        planner_checkpoint = em_path
    else:
        if em_eval_ckpt and not em_path.exists():
            raise FileNotFoundError(f"EM checkpoint not found: {em_path}")
        inference_ckpt = str(OmegaConf.select(args, "exp.iql_inference_ckpt", default="")).strip()
        load_inference_checkpoint(inference_model, inference_ckpt, device)
        if not planner_path.exists():
            raise FileNotFoundError(
                f"IQL checkpoint not found: {planner_path}. Set exp.iql_eval_ckpt / exp.em_eval_ckpt or train first."
            )
        planner = IQLPlanner.from_checkpoint(str(planner_path), device=device)
        encoder_checkpoint = _resolve_path(inference_ckpt, original_cwd) if inference_ckpt else None
        planner_checkpoint = planner_path

    inference_model.eval()
    planner.actor.eval()
    return inference_model, planner, planner_checkpoint, encoder_checkpoint, bool(use_em)


def _make_eval_dataloader(args: DictConfig, data: Dict[str, np.ndarray], batch_size: int, tau: int):
    original_tau = int(OmegaConf.select(args, "exp.tau", default=tau))
    args.exp.tau = int(tau)
    try:
        return get_dataloader(CIPDataset(data, args, train=False), batch_size=int(batch_size), shuffle=False)
    finally:
        args.exp.tau = original_tau


def _slice_batch(batch: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.size(0) >= n:
            out[key] = value[:n].contiguous()
        else:
            out[key] = value
    return out


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _clone_tensor_dict(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in batch.items()}


def _action_selector_kwargs(args: DictConfig) -> Dict[str, Any]:
    return {
        "selector": str(OmegaConf.select(args, "exp.iql_eval_action_selector", default="mean")),
        "candidate_actions": int(OmegaConf.select(args, "exp.iql_eval_candidate_actions", default=16)),
        "q_bc_penalty": float(OmegaConf.select(args, "exp.iql_eval_q_bc_penalty", default=0.0)),
        "candidate_noise_std": float(OmegaConf.select(args, "exp.iql_eval_candidate_noise_std", default=0.25)),
    }


def _support_context_from_state(
    support_index: SupportIndex,
    z: torch.Tensor,
    previous_action_sim: torch.Tensor,
    *,
    tau: int,
    step: int,
    max_tau: float,
) -> np.ndarray:
    bsz = z.size(0)
    delta = np.full((bsz, 1), float(tau - step) / float(max_tau), dtype=np.float32)
    return support_index.build_context(
        z.detach().cpu().numpy(),
        previous_action_sim.detach().cpu().numpy(),
        delta,
    )


def _build_or_load_support_index(
    *,
    args: DictConfig,
    dataset_collection,
    inference_model: InferenceModel,
    planner: IQLPlanner,
    original_cwd: Path,
    out_dir: Path,
    device: str,
    support_cache_enabled: bool,
    encoder_checkpoint: Optional[Path],
) -> Tuple[SupportIndex, Dict[str, Any], Path]:
    max_action = float(planner.cfg.max_action)
    action_dim = int(planner.cfg.action_dim)
    output_dim = int(args.dataset.output_size)
    z_dim = int(planner.cfg.state_dim) - output_dim - 1 - action_dim
    if z_dim <= 0:
        raise ValueError(f"Invalid parsed z_dim={z_dim} from planner state_dim={planner.cfg.state_dim}")

    k = int(OmegaConf.select(args, "exp.action_support_K", default=256))
    include_delta = bool(OmegaConf.select(args, "exp.action_support_include_delta", default=False))
    deduplicate = bool(OmegaConf.select(args, "exp.action_support_deduplicate", default=True))
    z_weight = float(OmegaConf.select(args, "exp.action_support_z_weight", default=1.0))
    prev_action_weight = float(OmegaConf.select(args, "exp.action_support_prev_action_weight", default=1.0))
    delta_weight = float(OmegaConf.select(args, "exp.action_support_delta_weight", default=0.5))
    max_patients_raw = OmegaConf.select(args, "exp.action_support_max_patients", default=1000)
    max_patients = None if max_patients_raw is None else int(max_patients_raw)
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    target_horizons = _list_from_config(OmegaConf.select(args, "exp.iql_target_horizons", default=None), default=None)
    target_sampling = str(OmegaConf.select(args, "exp.iql_target_sampling", default="horizon_aligned"))

    support_meta = {
        "kind": "support_index",
        "train_data": _data_fingerprint(dataset_collection.train_f.data),
        "encoder_checkpoint": _checkpoint_fingerprint(encoder_checkpoint, original_cwd),
        "max_patients": max_patients,
        "include_delta": include_delta,
        "context_weights": {
            "z": z_weight,
            "previous_action": prev_action_weight,
            "delta": delta_weight,
        },
        "deduplicate": deduplicate,
        "target_horizons": target_horizons,
        "target_sampling": target_sampling,
        "max_tau": max_tau,
        "samples_per_transition": int(OmegaConf.select(args, "exp.em_her_samples_per_transition", default=1)),
        "max_action": max_action,
        "z_dim": z_dim,
        "output_dim": output_dim,
        "action_dim": action_dim,
        "seed": int(args.exp.seed),
    }
    support_key = _hash_json(support_meta)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"support_index_{support_key}.npz"

    if support_cache_enabled and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        cached_meta = json.loads(str(loaded["metadata"].item()))
        support_index = SupportIndex.from_arrays(
            context_raw=loaded["context_raw"],
            behavior_actions_sim=loaded["behavior_actions_sim"],
            z_dim=z_dim,
            action_dim=action_dim,
            include_delta=include_delta,
            k=k,
            z_weight=z_weight,
            prev_action_weight=prev_action_weight,
            delta_weight=delta_weight,
            size_before_dedup=int(cached_meta["size_before_dedup"]),
            deduplicate=deduplicate,
            context_mean=loaded["context_mean"],
            context_std=loaded["context_std"],
        )
        cached_meta["cache_hit"] = True
        cached_meta["used_sklearn"] = bool(support_index.used_sklearn)
        cached_meta["size_after_dedup"] = int(support_index.size_after_dedup)
        return support_index, cached_meta, cache_path

    set_seed(int(args.exp.seed))
    transitions = build_iql_transitions_from_ct(
        data=dataset_collection.train_f.data,
        inference_model=inference_model,
        device=device,
        reward_type=str(OmegaConf.select(args, "exp.iql_reward_type", default="negative_outcome")),
        max_patients=max_patients,
        max_action=max_action,
        dataset_actions_unit_interval=bool(OmegaConf.select(args, "exp.iql_dataset_actions_unit_interval", default=True)),
        max_tau=max_tau,
        reward_clip=float(OmegaConf.select(args, "exp.iql_reward_clip", default=3.0)),
        reward_scale=str(OmegaConf.select(args, "exp.iql_reward_scale", default="auto")),
        reward_huber_delta=float(OmegaConf.select(args, "exp.iql_reward_huber_delta", default=1.0)),
        samples_per_transition=int(OmegaConf.select(args, "exp.em_her_samples_per_transition", default=1)),
        target_sampling=target_sampling,
        target_horizons=target_horizons,
        horizon_terminal_done=bool(OmegaConf.select(args, "exp.iql_horizon_terminal_done", default=True)),
    )
    support_index = SupportIndex.from_iql_transitions(
        transitions,
        z_dim=z_dim,
        output_dim=output_dim,
        action_dim=action_dim,
        max_action=max_action,
        include_delta=include_delta,
        k=k,
        z_weight=z_weight,
        prev_action_weight=prev_action_weight,
        delta_weight=delta_weight,
        deduplicate=deduplicate,
    )
    support_meta.update(
        {
            "cache_hit": False,
            "cache_key": support_key,
            "size_before_dedup": int(support_index.size_before_dedup),
            "size_after_dedup": int(support_index.size_after_dedup),
            "used_sklearn": bool(support_index.used_sklearn),
        }
    )
    if support_cache_enabled:
        np.savez_compressed(
            cache_path,
            context_raw=support_index.context_raw,
            behavior_actions_sim=support_index.behavior_actions_sim,
            context_mean=support_index.context_mean,
            context_std=support_index.context_std,
            metadata=np.asarray(json.dumps(support_meta, default=_json_default)),
        )
    return support_index, support_meta, cache_path


@torch.no_grad()
def collect_same_state_panel_diagnostics(
    *,
    planner: IQLPlanner,
    inference_model: InferenceModel,
    dataset_collection,
    fold,
    args: DictConfig,
    support_index: SupportIndex,
    action_grid_sim: torch.Tensor,
    device: str,
    tau: int,
    max_tau: float,
    batch_size: int,
    max_eval_samples: int,
    q_chunk_size: int,
) -> Dict[str, np.ndarray]:
    data = fold.data
    dataloader = _make_eval_dataloader(args, data, batch_size, tau)
    scaling_params = dataset_collection.train_scaling_params
    max_action = float(planner.cfg.max_action)
    selector_kwargs = _action_selector_kwargs(args)
    r = float(OmegaConf.select(args, "exp.action_support_r", default=0.075))
    eta = float(OmegaConf.select(args, "exp.action_support_eta", default=0.01))

    arrays: Dict[str, List[np.ndarray]] = {
        "panel_ours_actions": [],
        "panel_qgrid_actions": [],
        "panel_local_median_actions": [],
        "panel_local_actions": [],
        "panel_local_quantiles": [],
        "panel_ours_mass": [],
        "panel_qgrid_mass": [],
        "panel_local_median_mass": [],
        "panel_ours_min_distance": [],
        "panel_qgrid_min_distance": [],
        "panel_local_median_min_distance": [],
    }
    seen = 0
    for batch in dataloader:
        if seen >= max_eval_samples:
            break
        H_t, targets = batch
        remaining = int(max_eval_samples - seen)
        bsz = int(targets["outputs"].shape[0])
        if bsz > remaining:
            H_t = _slice_batch(H_t, remaining)
            targets = _slice_batch(targets, remaining)
            bsz = remaining
        seen += bsz

        H_t = align_h_t_static_to_history(_to_device(H_t, device))
        targets = _to_device(targets, device)
        eval_target = targets["outputs"][:, -1, :]
        H_work = _clone_tensor_dict(H_t)
        a_prev_sim = gather_last_valid(H_work["current_treatments"], H_work.get("active_entries")).clone()

        batch_steps: Dict[str, List[np.ndarray]] = {k: [] for k in arrays}
        for step in range(int(tau)):
            H_work = align_h_t_static_to_history(H_work)
            H_policy = align_h_t_static_to_history(_build_decision_history_view(H_work))
            z, _, _ = inference_model.ct_hidden_history(H_policy)
            a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
            obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
            ours_raw = select_iql_policy_action(planner, obs, **selector_kwargs)
            ours_sim = _policy_to_sim_interval_torch(ours_raw, max_action)
            qgrid_sim = q_grid_argmax_action(
                planner,
                obs,
                action_grid_sim,
                max_action,
                device=device,
                q_chunk_size=q_chunk_size,
            )

            context_raw = _support_context_from_state(
                support_index,
                z,
                a_prev_sim,
                tau=tau,
                step=step,
                max_tau=max_tau,
            )
            local_actions = support_index.query(context_raw)["actions"]
            ours_np = ours_sim.detach().cpu().numpy()
            qgrid_np = qgrid_sim.detach().cpu().numpy()
            median_np = np.median(local_actions, axis=1).astype(np.float32)

            ours_metrics = batch_local_support_metrics(local_actions, ours_np, r=r, eta=eta)
            qgrid_metrics = batch_local_support_metrics(local_actions, qgrid_np, r=r, eta=eta)
            median_metrics = batch_local_support_metrics(local_actions, median_np, r=r, eta=eta)

            batch_steps["panel_ours_actions"].append(ours_np)
            batch_steps["panel_qgrid_actions"].append(qgrid_np)
            batch_steps["panel_local_median_actions"].append(median_np)
            batch_steps["panel_local_actions"].append(local_actions)
            batch_steps["panel_local_quantiles"].append(ours_metrics["quantiles"])
            batch_steps["panel_ours_mass"].append(ours_metrics["small_ball_mass"])
            batch_steps["panel_qgrid_mass"].append(qgrid_metrics["small_ball_mass"])
            batch_steps["panel_local_median_mass"].append(median_metrics["small_ball_mass"])
            batch_steps["panel_ours_min_distance"].append(ours_metrics["knn_min_distance"])
            batch_steps["panel_qgrid_min_distance"].append(qgrid_metrics["knn_min_distance"])
            batch_steps["panel_local_median_min_distance"].append(median_metrics["knn_min_distance"])

            y_norm = _rollout_one_step_y(
                H_work,
                ours_sim,
                fold=fold,
                scaling_params=scaling_params,
                device=device,
            )
            _extend_h_work_after_one_step(H_work, ours_sim, y_norm, scaling_params, torch.device(device))
            a_prev_sim = ours_sim

        for key, value in batch_steps.items():
            if key == "panel_local_actions":
                arrays[key].append(np.stack(value, axis=1))
            elif key == "panel_local_quantiles":
                arrays[key].append(np.stack(value, axis=1))
            else:
                arrays[key].append(np.stack(value, axis=1))

    if seen == 0:
        raise RuntimeError("No evaluation samples were collected for action-support diagnostics.")
    return {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}


@torch.no_grad()
def evaluate_method_closed_loop(
    *,
    method_name: str,
    planner: IQLPlanner,
    inference_model: InferenceModel,
    dataset_collection,
    fold,
    args: DictConfig,
    support_index: SupportIndex,
    action_grid_sim: torch.Tensor,
    device: str,
    tau: int,
    max_tau: float,
    batch_size: int,
    max_eval_samples: int,
    q_chunk_size: int,
    eval_seed: int,
    external_actions: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    set_seed(int(eval_seed))
    dataloader = _make_eval_dataloader(args, fold.data, batch_size, tau)
    scaling_params = dataset_collection.train_scaling_params
    mean_ser, std_ser = scaling_params
    max_action = float(planner.cfg.max_action)
    selector_kwargs = _action_selector_kwargs(args)
    r = float(OmegaConf.select(args, "exp.action_support_r", default=0.075))
    eta = float(OmegaConf.select(args, "exp.action_support_eta", default=0.01))

    action_rows: List[np.ndarray] = []
    mass_rows: List[np.ndarray] = []
    ood_rows: List[np.ndarray] = []
    min_dist_rows: List[np.ndarray] = []
    pred_rows: List[np.ndarray] = []
    true_rows: List[np.ndarray] = []
    seen = 0

    for batch in dataloader:
        if seen >= max_eval_samples:
            break
        H_t, targets = batch
        remaining = int(max_eval_samples - seen)
        bsz = int(targets["outputs"].shape[0])
        if bsz > remaining:
            H_t = _slice_batch(H_t, remaining)
            targets = _slice_batch(targets, remaining)
            bsz = remaining
        if external_actions is not None and seen + bsz > external_actions.shape[0]:
            raise ValueError(
                f"External action sequence for {method_name!r} has only {external_actions.shape[0]} rows, "
                f"but at least {seen + bsz} are needed."
            )

        H_t = align_h_t_static_to_history(_to_device(H_t, device))
        targets = _to_device(targets, device)
        eval_target = targets["outputs"][:, -1, :]
        H_work = _clone_tensor_dict(H_t)
        a_prev_sim = gather_last_valid(H_work["current_treatments"], H_work.get("active_entries")).clone()
        step_actions: List[np.ndarray] = []
        step_masses: List[np.ndarray] = []
        step_oods: List[np.ndarray] = []
        step_min_dist: List[np.ndarray] = []
        y_last: Optional[torch.Tensor] = None

        for step in range(int(tau)):
            H_work = align_h_t_static_to_history(H_work)
            H_policy = align_h_t_static_to_history(_build_decision_history_view(H_work))
            z, _, _ = inference_model.ct_hidden_history(H_policy)
            a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
            obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
            context_raw = _support_context_from_state(
                support_index,
                z,
                a_prev_sim,
                tau=tau,
                step=step,
                max_tau=max_tau,
            )
            local_actions = support_index.query(context_raw)["actions"]

            if method_name == "ours":
                action_raw = select_iql_policy_action(planner, obs, **selector_kwargs)
                action_sim = _policy_to_sim_interval_torch(action_raw, max_action)
            elif method_name == "qgrid":
                action_sim = q_grid_argmax_action(
                    planner,
                    obs,
                    action_grid_sim,
                    max_action,
                    device=device,
                    q_chunk_size=q_chunk_size,
                )
            elif method_name == "factual":
                action_sim = targets["current_treatments"][:, step, :].contiguous()
            elif method_name == "local_median":
                median_np = np.median(local_actions, axis=1).astype(np.float32)
                action_sim = torch.as_tensor(median_np, device=device, dtype=torch.float32)
            elif external_actions is not None:
                action_np = external_actions[seen : seen + bsz, step, :].astype(np.float32)
                action_sim = torch.as_tensor(action_np, device=device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown closed-loop method: {method_name!r}")

            action_np = action_sim.detach().cpu().numpy()
            metrics = batch_local_support_metrics(local_actions, action_np, r=r, eta=eta)
            step_actions.append(action_np)
            step_masses.append(metrics["small_ball_mass"])
            step_oods.append(metrics["is_ood"].astype(np.float32))
            step_min_dist.append(metrics["knn_min_distance"])

            y_last = _rollout_one_step_y(
                H_work,
                action_sim,
                fold=fold,
                scaling_params=scaling_params,
                device=device,
            )
            _extend_h_work_after_one_step(H_work, action_sim, y_last, scaling_params, torch.device(device))
            a_prev_sim = action_sim

        assert y_last is not None
        action_rows.append(np.stack(step_actions, axis=1))
        mass_rows.append(np.stack(step_masses, axis=1))
        ood_rows.append(np.stack(step_oods, axis=1))
        min_dist_rows.append(np.stack(step_min_dist, axis=1))
        pred_rows.append(y_last.detach().cpu().numpy())
        true_rows.append(eval_target.detach().cpu().numpy())
        seen += bsz

    pred_y_norm = np.concatenate(pred_rows, axis=0).astype(np.float32)
    true_y_norm = np.concatenate(true_rows, axis=0).astype(np.float32)
    pred_y_uns = _unscaled_cancer_volume_np(pred_y_norm, mean_ser, std_ser).reshape(-1)
    true_y_uns = _unscaled_cancer_volume_np(true_y_norm, mean_ser, std_ser).reshape(-1)
    rmse_norm = float(np.sqrt(np.mean((pred_y_norm.reshape(-1) - true_y_norm.reshape(-1)) ** 2)))
    rmse_uns = float(np.sqrt(np.mean((pred_y_uns - true_y_uns) ** 2)))
    actions = np.concatenate(action_rows, axis=0).astype(np.float32)
    masses = np.concatenate(mass_rows, axis=0).astype(np.float32)
    ood = np.concatenate(ood_rows, axis=0).astype(np.float32)
    min_dist = np.concatenate(min_dist_rows, axis=0).astype(np.float32)

    arrays = {
        f"method_{method_name}_actions": actions,
        f"method_{method_name}_support_mass": masses,
        f"method_{method_name}_ood": ood,
        f"method_{method_name}_min_distance": min_dist,
        f"method_{method_name}_pred_y_norm": pred_y_norm,
        f"method_{method_name}_true_y_norm": true_y_norm,
    }
    summary = {
        "method": method_name,
        "n_samples": int(actions.shape[0]),
        "ood_rate": float(ood.mean()),
        "support_mass_mean": float(masses.mean()),
        "support_mass_min": float(masses.min()),
        "knn_min_distance_mean": float(min_dist.mean()),
        "rmse_norm": rmse_norm,
        "rmse_uns": rmse_uns,
    }
    return arrays, summary


def _load_external_actions(path_raw: str, *, tau: int, action_dim: int, original_cwd: Path) -> Dict[str, np.ndarray]:
    path_raw = str(path_raw or "").strip()
    if not path_raw:
        return {}
    path = _resolve_path(path_raw, original_cwd)
    if not path.exists():
        raise FileNotFoundError(f"External action sequence file not found: {path}")
    out: Dict[str, np.ndarray] = {}
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        for key in loaded.files:
            arr = np.asarray(loaded[key], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[1] >= tau and arr.shape[2] == action_dim:
                name = key[:-8] if key.endswith("_actions") else key
                out[f"external_{name}"] = np.clip(arr[:, :tau, :], 0.0, 1.0)
    else:
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] < tau or arr.shape[2] != action_dim:
            raise ValueError(f"External action array must have shape [N, >=tau, {action_dim}], got {arr.shape}")
        out["external"] = np.clip(arr[:, :tau, :], 0.0, 1.0)
    if not out:
        raise ValueError(f"No valid [N, >=tau, {action_dim}] action arrays found in {path}")
    return out


def _diagnostic_cache_metadata(
    *,
    args: DictConfig,
    support_key: str,
    planner_checkpoint: Path,
    original_cwd: Path,
    split_name: str,
    tau: int,
    external_actions_path: str,
) -> Dict[str, Any]:
    return {
        "kind": "diagnostic_rollout",
        "support_key": support_key,
        "planner_checkpoint": _checkpoint_fingerprint(planner_checkpoint, original_cwd),
        "grid_points": int(OmegaConf.select(args, "exp.action_support_grid_points", default=31)),
        "eval_split": split_name,
        "tau": int(tau),
        "seed": int(args.exp.seed),
        "rollout": "closed_loop_simulator",
        "rmse_metric": str(OmegaConf.select(args, "exp.action_support_rmse_metric", default="rmse_norm")),
        "eval_batch_size": int(OmegaConf.select(args, "exp.action_support_eval_batch_size", default=128)),
        "max_eval_samples": int(OmegaConf.select(args, "exp.action_support_max_eval_samples", default=512)),
        "q_chunk_size": int(OmegaConf.select(args, "exp.action_support_q_chunk_size", default=8192)),
        "K": int(OmegaConf.select(args, "exp.action_support_K", default=256)),
        "r": float(OmegaConf.select(args, "exp.action_support_r", default=0.075)),
        "eta": float(OmegaConf.select(args, "exp.action_support_eta", default=0.01)),
        "action_selector": _action_selector_kwargs(args),
        "external_actions": _checkpoint_fingerprint(external_actions_path, original_cwd),
    }


def _run_or_load_diagnostics(
    *,
    args: DictConfig,
    planner: IQLPlanner,
    inference_model: InferenceModel,
    dataset_collection,
    fold,
    support_index: SupportIndex,
    support_meta: Dict[str, Any],
    planner_checkpoint: Path,
    original_cwd: Path,
    out_dir: Path,
    device: str,
    split_name: str,
    tau: int,
    cache_enabled: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Path]:
    rmse_metric = str(OmegaConf.select(args, "exp.action_support_rmse_metric", default="rmse_norm")).strip()
    if rmse_metric not in ("rmse_norm", "rmse_uns"):
        raise ValueError("exp.action_support_rmse_metric must be one of: rmse_norm, rmse_uns")
    grid_points = int(OmegaConf.select(args, "exp.action_support_grid_points", default=31))
    eval_batch_size = int(OmegaConf.select(args, "exp.action_support_eval_batch_size", default=128))
    max_eval_samples = int(OmegaConf.select(args, "exp.action_support_max_eval_samples", default=512))
    q_chunk_size = int(OmegaConf.select(args, "exp.action_support_q_chunk_size", default=8192))
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    external_path = str(OmegaConf.select(args, "exp.action_support_external_actions_path", default="")).strip()

    action_grid_sim = make_action_grid_sim(
        int(planner.cfg.action_dim),
        grid_points,
        device=device,
        dtype=torch.float32,
    )
    diag_meta = _diagnostic_cache_metadata(
        args=args,
        support_key=str(support_meta.get("cache_key", _hash_json(support_meta))),
        planner_checkpoint=planner_checkpoint,
        original_cwd=original_cwd,
        split_name=split_name,
        tau=tau,
        external_actions_path=external_path,
    )
    diag_key = _hash_json(diag_meta)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"diagnostic_rollout_{diag_key}.npz"

    if cache_enabled and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        arrays = {k: loaded[k] for k in loaded.files if k != "metadata"}
        cached_meta = json.loads(str(loaded["metadata"].item()))
        cached_meta["cache_hit"] = True
        return arrays, cached_meta, cache_path

    set_seed(int(args.exp.seed) + 1701)
    panel_arrays = collect_same_state_panel_diagnostics(
        planner=planner,
        inference_model=inference_model,
        dataset_collection=dataset_collection,
        fold=fold,
        args=args,
        support_index=support_index,
        action_grid_sim=action_grid_sim,
        device=device,
        tau=tau,
        max_tau=max_tau,
        batch_size=eval_batch_size,
        max_eval_samples=max_eval_samples,
        q_chunk_size=q_chunk_size,
    )

    method_summaries: Dict[str, Any] = {}
    method_arrays: Dict[str, np.ndarray] = {}
    methods = ["ours", "qgrid", "factual"]
    if bool(OmegaConf.select(args, "exp.action_support_include_local_median", default=True)):
        methods.append("local_median")
    external_actions = _load_external_actions(
        external_path,
        tau=tau,
        action_dim=int(planner.cfg.action_dim),
        original_cwd=original_cwd,
    )

    method_eval_seed = int(args.exp.seed) + 3001
    for method in methods:
        arrays, summary = evaluate_method_closed_loop(
            method_name=method,
            planner=planner,
            inference_model=inference_model,
            dataset_collection=dataset_collection,
            fold=fold,
            args=args,
            support_index=support_index,
            action_grid_sim=action_grid_sim,
            device=device,
            tau=tau,
            max_tau=max_tau,
            batch_size=eval_batch_size,
            max_eval_samples=max_eval_samples,
            q_chunk_size=q_chunk_size,
            eval_seed=method_eval_seed,
        )
        method_arrays.update(arrays)
        method_summaries[method] = summary

    for method, ext_actions in external_actions.items():
        arrays, summary = evaluate_method_closed_loop(
            method_name=method,
            planner=planner,
            inference_model=inference_model,
            dataset_collection=dataset_collection,
            fold=fold,
            args=args,
            support_index=support_index,
            action_grid_sim=action_grid_sim,
            device=device,
            tau=tau,
            max_tau=max_tau,
            batch_size=eval_batch_size,
            max_eval_samples=max_eval_samples,
            q_chunk_size=q_chunk_size,
            eval_seed=method_eval_seed,
            external_actions=ext_actions,
        )
        method_arrays.update(arrays)
        method_summaries[method] = summary

    arrays = {**panel_arrays, **method_arrays}
    diag_meta.update(
        {
            "cache_hit": False,
            "cache_key": diag_key,
            "method_summaries": method_summaries,
        }
    )
    if cache_enabled:
        np.savez_compressed(
            cache_path,
            **arrays,
            metadata=np.asarray(json.dumps(diag_meta, default=_json_default)),
        )
    return arrays, diag_meta, cache_path


def _select_representative(arrays: Dict[str, np.ndarray], args: DictConfig) -> Dict[str, Any]:
    ours_mass = arrays["panel_ours_mass"]
    qgrid_mass = arrays["panel_qgrid_mass"]
    n_samples, tau = ours_mass.shape
    sample_cfg = OmegaConf.select(args, "exp.action_support_sample_index", default=None)
    step_cfg = OmegaConf.select(args, "exp.action_support_step_index", default=None)

    if sample_cfg is not None and step_cfg is not None:
        sample_index = int(sample_cfg)
        step_index = int(step_cfg)
        criterion = "manual sample_index and step_index"
    elif sample_cfg is not None:
        sample_index = int(sample_cfg)
        diff = ours_mass[sample_index] - qgrid_mass[sample_index]
        step_index = int(np.argmax(diff))
        criterion = "manual sample_index; auto step maximizing ours_support_mass - qgrid_support_mass"
    elif step_cfg is not None:
        step_index = int(step_cfg)
        diff = ours_mass[:, step_index] - qgrid_mass[:, step_index]
        sample_index = int(np.argmax(diff))
        criterion = "manual step_index; auto sample maximizing ours_support_mass - qgrid_support_mass"
    else:
        eta = float(OmegaConf.select(args, "exp.action_support_eta", default=0.01))
        diff = ours_mass - qgrid_mass
        low_q = qgrid_mass < eta
        if np.any(low_q):
            score = np.where(low_q, diff, -np.inf)
            criterion = "auto: qgrid_support_mass < eta, then maximize ours_support_mass - qgrid_support_mass"
        else:
            score = diff
            criterion = "auto: maximize ours_support_mass - qgrid_support_mass"
        flat = int(np.argmax(score))
        sample_index, step_index = np.unravel_index(flat, ours_mass.shape)
        sample_index, step_index = int(sample_index), int(step_index)

    if not (0 <= sample_index < n_samples):
        raise ValueError(f"sample_index={sample_index} out of range [0, {n_samples - 1}]")
    if not (0 <= step_index < tau):
        raise ValueError(f"step_index={step_index} out of range [0, {tau - 1}]")

    local = arrays["panel_local_actions"][sample_index, step_index]
    r = float(OmegaConf.select(args, "exp.action_support_r", default=0.075))
    eta = float(OmegaConf.select(args, "exp.action_support_eta", default=0.01))
    ours_metrics = local_support_metrics(local, arrays["panel_ours_actions"][sample_index, step_index], r=r, eta=eta)
    qgrid_metrics = local_support_metrics(local, arrays["panel_qgrid_actions"][sample_index, step_index], r=r, eta=eta)
    median_metrics = local_support_metrics(
        local,
        arrays["panel_local_median_actions"][sample_index, step_index],
        r=r,
        eta=eta,
    )
    return {
        "sample_index": sample_index,
        "step_index": step_index,
        "selection_criterion": criterion,
        "ours_support_mass": float(ours_mass[sample_index, step_index]),
        "qgrid_support_mass": float(qgrid_mass[sample_index, step_index]),
        "ours_support_metrics": ours_metrics,
        "qgrid_support_metrics": qgrid_metrics,
        "local_median_support_metrics": median_metrics,
    }


def _plot_panel_a(ax, arrays: Dict[str, np.ndarray], selected: Dict[str, Any]) -> None:
    s = int(selected["sample_index"])
    k = int(selected["step_index"])
    local = arrays["panel_local_actions"][s, k]
    ours = arrays["panel_ours_actions"][s, k]
    qgrid = arrays["panel_qgrid_actions"][s, k]
    median = arrays["panel_local_median_actions"][s, k]
    ax.hist2d(local[:, 0], local[:, 1], bins=24, range=[[0, 1], [0, 1]], cmap="Greys", alpha=0.55)
    ax.scatter(local[:, 0], local[:, 1], s=12, c="0.55", alpha=0.45, edgecolors="none", label="Local behavior")
    ax.scatter([ours[0]], [ours[1]], marker="*", s=180, c="#d62728", edgecolors="black", linewidths=0.5, label="Ours")
    ax.scatter([qgrid[0]], [qgrid[1]], marker="x", s=110, c="#1f77b4", linewidths=2.0, label="Q-grid argmax")
    ax.scatter([median[0]], [median[1]], marker="o", s=70, c="#2ca02c", edgecolors="black", linewidths=0.5, label="Local median")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Chemotherapy action")
    ax.set_ylabel("Radiotherapy action")
    ax.set_title("(a) Empirical local action support")
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.grid(alpha=0.2)


def _plot_panel_b(axes, arrays: Dict[str, np.ndarray], selected: Dict[str, Any]) -> None:
    s = int(selected["sample_index"])
    steps = np.arange(arrays["panel_ours_actions"].shape[1])
    quant = arrays["panel_local_quantiles"][s]
    labels = ["Chemotherapy action", "Radiotherapy action"]
    for dim, ax in enumerate(axes):
        ax.fill_between(steps, quant[:, 0, dim], quant[:, 4, dim], color="0.75", alpha=0.45, label="Local 5%-95%")
        ax.fill_between(steps, quant[:, 1, dim], quant[:, 3, dim], color="0.45", alpha=0.35, label="Local 25%-75%")
        ax.plot(steps, arrays["panel_ours_actions"][s, :, dim], "-*", color="#d62728", label="Ours")
        ax.plot(steps, arrays["panel_qgrid_actions"][s, :, dim], "-x", color="#1f77b4", label="Q-grid argmax")
        ax.plot(steps, arrays["panel_local_median_actions"][s, :, dim], "-o", color="#2ca02c", label="Local median")
        ax.set_ylim(0, 1)
        ax.set_ylabel(labels[dim])
        ax.grid(alpha=0.2)
    axes[0].set_title("(b) Same-state closed-loop support envelope")
    axes[-1].set_xlabel("Rollout step")
    axes[0].legend(fontsize=8, ncol=2, loc="best", frameon=True)


def _plot_panel_c(ax, method_summaries: Dict[str, Dict[str, Any]], rmse_metric: str) -> None:
    labels = {
        "ours": "Ours",
        "qgrid": "Q-grid argmax",
        "factual": "Factual",
        "local_median": "Local median",
    }
    colors = {
        "ours": "#d62728",
        "qgrid": "#1f77b4",
        "factual": "#7f7f7f",
        "local_median": "#2ca02c",
    }
    for method, summary in method_summaries.items():
        x = float(summary["ood_rate"])
        y = float(summary[rmse_metric])
        label = labels.get(method, method.replace("_", " "))
        ax.scatter([x], [y], s=80, color=colors.get(method, "black"), edgecolors="black", linewidths=0.5)
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Empirical OOD action rate")
    ax.set_ylabel(f"Terminal {rmse_metric}")
    ax.set_title("(c) OOD rate vs terminal RMSE")
    ax.grid(alpha=0.25)


def _save_figures(out_dir: Path, arrays: Dict[str, np.ndarray], selected: Dict[str, Any], method_summaries: Dict[str, Any], rmse_metric: str) -> None:
    fig = plt.figure(figsize=(15.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_b0 = fig.add_subplot(gs_b[0, 0])
    ax_b1 = fig.add_subplot(gs_b[1, 0], sharex=ax_b0)
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_panel_a(ax_a, arrays, selected)
    _plot_panel_b([ax_b0, ax_b1], arrays, selected)
    _plot_panel_c(ax_c, method_summaries, rmse_metric)
    fig.suptitle("Empirical local action support diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "local_action_support_intro.png", dpi=300)
    fig.savefig(out_dir / "local_action_support_intro.pdf")
    plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(5.2, 4.6))
    _plot_panel_a(ax_a, arrays, selected)
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "panel_a_local_joint_support.png", dpi=300)
    fig_a.savefig(out_dir / "panel_a_local_joint_support.pdf")
    plt.close(fig_a)

    fig_b, axes_b = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    _plot_panel_b(axes_b, arrays, selected)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "panel_b_action_envelope.png", dpi=300)
    fig_b.savefig(out_dir / "panel_b_action_envelope.pdf")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(5.4, 4.6))
    _plot_panel_c(ax_c, method_summaries, rmse_metric)
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "panel_c_ood_rmse.png", dpi=300)
    fig_c.savefig(out_dir / "panel_c_ood_rmse.pdf")
    plt.close(fig_c)


def _range_summary(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    return {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}


def _caption_text(rmse_metric: str) -> str:
    return (
        "Empirical local action support diagnostics for target-conditioned IQL. "
        "Panel (a) shows local 2D joint behavior-action support near one representative decision state. "
        "Panel (b) shows the same state's closed-loop support envelope along the Ours rollout, with Q-grid "
        "argmax evaluated at those same Ours-generated states. Panel (c) compares method-specific closed-loop "
        f"empirical OOD action rate against terminal {rmse_metric}. Lower-left is better."
    )


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    seed = int(args.exp.seed)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = _resolve_path(
        str(OmegaConf.select(args, "exp.action_support_out_dir", default="plots/iql_action_support")),
        original_cwd,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_collection = _prepare_dataset(args)
    split_name = "test" if bool(OmegaConf.select(args, "exp.test", default=False)) else "val"
    fold = dataset_collection.test_f if split_name == "test" else dataset_collection.val_f
    inference_model, planner, planner_checkpoint, encoder_checkpoint, use_em = _load_models(args, original_cwd, device)

    action_dim = int(planner.cfg.action_dim)
    if action_dim != 2:
        raise ValueError(
            f"Panel (a) local 2D joint action support only supports action_dim == 2; got action_dim={action_dim}."
        )

    tau = int(OmegaConf.select(args, "exp.tau", default=6))
    rmse_metric = str(OmegaConf.select(args, "exp.action_support_rmse_metric", default="rmse_norm")).strip()
    if rmse_metric not in ("rmse_norm", "rmse_uns"):
        raise ValueError("exp.action_support_rmse_metric must be one of: rmse_norm, rmse_uns")

    cache_enabled = bool(OmegaConf.select(args, "exp.action_support_cache", default=True))
    support_index, support_meta, support_cache_path = _build_or_load_support_index(
        args=args,
        dataset_collection=dataset_collection,
        inference_model=inference_model,
        planner=planner,
        original_cwd=original_cwd,
        out_dir=out_dir,
        device=device,
        support_cache_enabled=cache_enabled,
        encoder_checkpoint=encoder_checkpoint,
    )
    arrays, diag_meta, diag_cache_path = _run_or_load_diagnostics(
        args=args,
        planner=planner,
        inference_model=inference_model,
        dataset_collection=dataset_collection,
        fold=fold,
        support_index=support_index,
        support_meta=support_meta,
        planner_checkpoint=planner_checkpoint,
        original_cwd=original_cwd,
        out_dir=out_dir,
        device=device,
        split_name=split_name,
        tau=tau,
        cache_enabled=cache_enabled,
    )
    method_summaries = diag_meta["method_summaries"]
    selected = _select_representative(arrays, args)

    all_panel_actions = np.concatenate(
        [
            arrays["panel_ours_actions"].reshape(-1, action_dim),
            arrays["panel_qgrid_actions"].reshape(-1, action_dim),
            arrays["panel_local_median_actions"].reshape(-1, action_dim),
            arrays["panel_local_actions"].reshape(-1, action_dim),
        ],
        axis=0,
    )
    summary = {
        "diagnostic": "iql_local_empirical_action_support",
        "caption": _caption_text(rmse_metric),
        "split": split_name,
        "tau": int(tau),
        "rollout": "closed_loop_simulator",
        "seed": seed,
        "device": device,
        "use_em_checkpoint": bool(use_em),
        "planner_checkpoint": str(planner_checkpoint),
        "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint is not None else "",
        "support_cache_path": str(support_cache_path),
        "diagnostic_cache_path": str(diag_cache_path),
        "support_index": support_meta,
        "representative_selection": selected,
        "rmse_metric_for_panel_c": rmse_metric,
        "methods": method_summaries,
        "ranges": {
            "plotted_actions": _range_summary(all_panel_actions),
            "local_quantiles": _range_summary(arrays["panel_local_quantiles"]),
        },
        "config": {
            "K": int(OmegaConf.select(args, "exp.action_support_K", default=256)),
            "r": float(OmegaConf.select(args, "exp.action_support_r", default=0.075)),
            "eta": float(OmegaConf.select(args, "exp.action_support_eta", default=0.01)),
            "include_delta": bool(OmegaConf.select(args, "exp.action_support_include_delta", default=False)),
            "eval_batch_size": int(OmegaConf.select(args, "exp.action_support_eval_batch_size", default=128)),
            "max_eval_samples": int(OmegaConf.select(args, "exp.action_support_max_eval_samples", default=512)),
            "q_chunk_size": int(OmegaConf.select(args, "exp.action_support_q_chunk_size", default=8192)),
            "grid_points": int(OmegaConf.select(args, "exp.action_support_grid_points", default=31)),
        },
    }

    _save_figures(out_dir, arrays, selected, method_summaries, rmse_metric)
    summary_path = out_dir / "support_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    (out_dir / "caption.txt").write_text(_caption_text(rmse_metric) + "\n")
    np.savez_compressed(
        out_dir / "support_diagnostics.npz",
        **arrays,
        support_summary_json=np.asarray(json.dumps(summary, default=_json_default)),
    )

    print("IQL empirical local action support diagnostics complete")
    print(f"Output directory: {out_dir}")
    print(f"Plotted actions min/max: {summary['ranges']['plotted_actions']}")
    print(f"Local quantiles min/max: {summary['ranges']['local_quantiles']}")
    print("OOD rates: " + json.dumps({k: v["ood_rate"] for k, v in method_summaries.items()}, sort_keys=True))
    print(f"Selected RMSE metric: {rmse_metric}")
    print(
        "Representative sample/step: "
        f"{selected['sample_index']}/{selected['step_index']} ({selected['selection_criterion']})"
    )
    print(
        "Support index size before/after deduplication: "
        f"{support_index.size_before_dedup}/{support_index.size_after_dedup}"
    )


if __name__ == "__main__":
    main()
