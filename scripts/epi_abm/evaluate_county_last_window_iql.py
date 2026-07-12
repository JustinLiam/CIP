"""Evaluate EM IQL checkpoints on EpiABM county rollouts.

This script is intentionally separate from the training code. It evaluates one
decision point per county. By default it keeps the historical last-window
protocol for backwards compatibility; pass ``--window-mode fixed-start`` to
compare multiple horizons from the same decision day.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.iql_dataset_builder import align_h_t_static_to_history  # noqa: E402
from src.evaluation.iql_action_selection import select_iql_policy_action  # noqa: E402
from src.evaluation.iql_planner_eval import (  # noqa: E402
    _extend_h_work_after_one_step,
    _iql_augmented_state,
    _policy_to_sim_interval_torch,
    _unscale_outputs_np,
)
from src.models.inference_model import InferenceModel  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.utils.em_ckpt import load_em_for_eval  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402


def _parse_label_path(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint spec must be label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty checkpoint label in {item!r}")
        out[label] = Path(raw_path).expanduser()
    return out


def _stable_int(*parts: object, modulo: int = 2_000_000_000) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _to_device_history(H: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.clone().to(device) if torch.is_tensor(v) else v) for k, v in H.items()}


def _cpu_history(H: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in H.items()}


def _slice_county_window(
    data: dict,
    row_idx: int,
    tau: int,
    *,
    window_mode: str,
    decision_day: int | None = None,
) -> Tuple[dict, dict, int]:
    n_rows = int(np.asarray(data["sequence_lengths"]).shape[0])
    seq_len = int(np.asarray(data["sequence_lengths"])[row_idx])
    if window_mode == "last":
        selected_day = seq_len - int(tau)
    elif window_mode == "fixed-start":
        if decision_day is None:
            raise ValueError("--decision-day is required when --window-mode fixed-start")
        selected_day = int(decision_day)
    else:
        raise ValueError(f"Unknown window_mode={window_mode!r}")

    if selected_day <= 0:
        raise ValueError(f"row_idx={row_idx} has invalid decision_day={selected_day}")
    if selected_day + int(tau) > seq_len:
        raise ValueError(
            f"row_idx={row_idx} has seq_len={seq_len}, decision_day={selected_day}, tau={tau}; "
            "future horizon exceeds available trajectory"
        )

    H: dict = {}
    targets: dict = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[0] != n_rows:
            continue
        if arr.ndim >= 2 and arr.shape[1] >= seq_len:
            H[key] = torch.as_tensor(arr[row_idx:row_idx + 1, :selected_day]).float()
            targets[key] = torch.as_tensor(arr[row_idx:row_idx + 1, selected_day:selected_day + tau]).float()
        elif arr.ndim == 1:
            H[key] = torch.as_tensor(arr[row_idx:row_idx + 1])

    H["sequence_lengths"] = torch.as_tensor([selected_day]).long()
    if "active_entries" in H:
        H["active_entries"] = torch.ones_like(H["active_entries"])
    return H, targets, selected_day


def _county_id(data: dict, row_idx: int) -> str:
    value = np.asarray(data["sim_county_id"])[row_idx, 0, 0]
    return f"{int(float(value)):05d}"


def _output_mean_std(scaling_params) -> Tuple[np.ndarray, np.ndarray]:
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


def _scale_outputs_np(y_uns: np.ndarray, scaling_params) -> np.ndarray:
    means, stds = _output_mean_std(scaling_params)
    return ((np.asarray(y_uns, dtype=np.float64) - means) / stds).astype(np.float32)


def _outcome_transform(scaling_params) -> str:
    if isinstance(scaling_params, dict):
        value = str(scaling_params.get("outcome_transform", "raw_cases_zscore")).strip().lower()
        if value in {"per10k", "per_10k", "per10k_cases", "per_10k_cases", "per10k_cases_zscore", "per_10k_cases_zscore"}:
            return "per10k_cases_zscore"
    return "raw_cases_zscore"


def _model_units_to_raw_cases(values: np.ndarray, scaling_params, population: float | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if _outcome_transform(scaling_params) == "per10k_cases_zscore":
        if population is None or population <= 0:
            return np.full_like(arr, np.nan, dtype=np.float64)
        return arr * float(population) / 10000.0
    return arr


def _model_units_to_per_capita(
    values: np.ndarray,
    scaling_params,
    population: float | None,
    base: float,
) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if _outcome_transform(scaling_params) == "per10k_cases_zscore":
        scale = float(base) / 10000.0
        return float(arr.reshape(-1).mean()) * scale
    if population is None or population <= 0:
        return None
    return float(arr.reshape(-1).mean()) / float(population) * float(base)


def _population_from_row(data: dict, row_idx: int) -> float | None:
    if "static_features" not in data:
        return None
    arr = np.asarray(data["static_features"])
    if arr.shape[0] <= row_idx:
        return None
    if arr.ndim == 3:
        value = float(arr[row_idx, 0, 0])
    elif arr.ndim == 2:
        value = float(arr[row_idx, 0])
    else:
        return None
    population = value * 100000.0
    if not np.isfinite(population) or population <= 0:
        return None
    return population


def _make_eval_target_norm(
    H: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    scaling_params,
    *,
    target_mode: str,
    target_scale: float,
    target_value: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    factual_final_norm = targets["outputs"][:, -1, :].detach().cpu().numpy().astype(np.float32)
    factual_final_uns = _unscale_outputs_np(factual_final_norm, scaling_params)

    if target_mode == "factual_final":
        target_uns = factual_final_uns
    elif target_mode in {"half_factual_final", "relative_factual_final"}:
        target_uns = factual_final_uns * float(target_scale)
    elif target_mode == "relative_current":
        current_norm = H["outputs"][:, -1, :].detach().cpu().numpy().astype(np.float32)
        target_uns = _unscale_outputs_np(current_norm, scaling_params) * float(target_scale)
    elif target_mode == "absolute_final":
        if target_value is None:
            raise ValueError("--target-value is required when --target-mode absolute_final")
        target_uns = np.full_like(factual_final_uns, float(target_value), dtype=np.float64)
    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}")

    target_uns = np.maximum(target_uns, 0.0)
    target_norm = _scale_outputs_np(target_uns, scaling_params)
    return target_norm.astype(np.float32), target_uns.astype(np.float64)


def _select_action(
    *,
    label: str,
    inference_model: InferenceModel,
    planner,
    H_work: Dict[str, torch.Tensor],
    eval_target_norm: torch.Tensor,
    step: int,
    tau: int,
    max_tau: float,
    selector: str,
    candidate_actions: int,
    q_bc_penalty: float,
    candidate_noise_std: float,
    eval_seed: int,
    county: str,
    device: str,
) -> torch.Tensor:
    H_work = align_h_t_static_to_history(H_work)
    with torch.no_grad():
        z, _, _ = inference_model.ct_hidden_history(H_work)
        target = eval_target_norm
        a_prev = gather_last_valid(H_work["current_treatments"], H_work.get("active_entries"))
        a_prev_tanh = (2.0 * torch.clamp(a_prev, 0.0, 1.0) - 1.0) * float(planner.actor.max_action)
        obs = _iql_augmented_state(planner, z, target, step, tau, max_tau, a_prev_tanh)

        # Keep candidate-action sampling invariant to checkpoint labels. Otherwise
        # the same checkpoint evaluated as "old_td" vs "old10_td" can receive
        # different q_sample candidates and produce different ABM rollouts.
        seed = int(eval_seed) + _stable_int("candidate_action", county, tau, step)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        raw = select_iql_policy_action(
            planner,
            obs,
            selector=selector,
            candidate_actions=candidate_actions,
            q_bc_penalty=q_bc_penalty,
            candidate_noise_std=candidate_noise_std,
        )
        return _policy_to_sim_interval_torch(raw, float(planner.actor.max_action)).detach()


def _rollout_policy(
    *,
    fold,
    dataset_collection,
    inference_model: InferenceModel,
    planner,
    H: Dict[str, torch.Tensor],
    eval_target_norm: np.ndarray,
    label: str,
    county: str,
    tau: int,
    max_tau: float,
    action_hold_days: int,
    selector: str,
    candidate_actions: int,
    q_bc_penalty: float,
    candidate_noise_std: float,
    eval_seed: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    H_work = _to_device_history(H, device)
    eval_target_dev = torch.as_tensor(eval_target_norm, dtype=torch.float32, device=device)
    planned_actions: List[np.ndarray] = []
    pred_norm: List[float] = []
    held_action = None

    for step in range(int(tau)):
        if held_action is None or (step % int(action_hold_days)) == 0:
            held_action = _select_action(
                label=label,
                inference_model=inference_model,
                planner=planner,
                H_work=H_work,
                eval_target_norm=eval_target_dev,
                step=step,
                tau=tau,
                max_tau=max_tau,
                selector=selector,
                candidate_actions=candidate_actions,
                q_bc_penalty=q_bc_penalty,
                candidate_noise_std=candidate_noise_std,
                eval_seed=eval_seed,
                county=county,
                device=device,
            )

        next_observation = fold.simulate_next_after_action(
            _cpu_history(H_work),
            held_action.detach().cpu(),
            dataset_collection.train_scaling_params,
        )
        y_norm = torch.as_tensor(
            next_observation["outputs"],
            device=device,
            dtype=torch.float32,
        ).view(1, -1)
        _extend_h_work_after_one_step(
            H_work,
            held_action,
            y_norm,
            dataset_collection.train_scaling_params,
            torch.device(device),
            next_observation=next_observation,
        )
        planned_actions.append(held_action.detach().cpu().numpy().reshape(-1))
        pred_norm.append(float(y_norm[0, 0].detach().cpu()))

    return np.asarray(pred_norm, dtype=np.float32).reshape(1, tau, 1), np.asarray(planned_actions, dtype=np.float32)


def _rollout_factual(fold, dataset_collection, H: dict, targets: dict, tau: int) -> np.ndarray:
    actions = targets["current_treatments"].detach().cpu().numpy().astype(np.float32)
    return fold.simulate_output_after_actions(
        H,
        actions[:, :tau, :],
        dataset_collection.train_scaling_params,
        return_daily=True,
    )


def _release_county_cache(fold, county: str) -> None:
    """Release heavyweight ABM runner/snapshot state once a county is complete."""
    cache = getattr(fold, "simulator_cache", None)
    try:
        if cache is not None:
            cache.release_county(county)
    except Exception as exc:  # diagnostic cleanup should not invalidate metrics
        print(json.dumps({
            "event": "release_county_warning",
            "county": county,
            "error": repr(exc),
        }), flush=True)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _metric_row(
    *,
    split: str,
    tau: int,
    county: str,
    label: str,
    pred_norm_daily: np.ndarray,
    true_norm_daily: np.ndarray,
    factual_norm_daily: np.ndarray,
    target_norm_final: np.ndarray,
    planned_actions: np.ndarray | None,
    scaling_params,
    population: float | None,
) -> dict:
    pred_final = pred_norm_daily[:, -1, :]
    true_final = true_norm_daily[:, -1, :]
    factual_final = factual_norm_daily[:, -1, :]
    target_final = np.asarray(target_norm_final, dtype=np.float32)

    transform = _outcome_transform(scaling_params)
    pred_daily_model = _unscale_outputs_np(pred_norm_daily, scaling_params)
    true_daily_model = _unscale_outputs_np(true_norm_daily, scaling_params)
    factual_daily_model = _unscale_outputs_np(factual_norm_daily, scaling_params)
    pred_model = _unscale_outputs_np(pred_final, scaling_params)
    true_model = _unscale_outputs_np(true_final, scaling_params)
    factual_model = _unscale_outputs_np(factual_final, scaling_params)
    target_model = _unscale_outputs_np(target_final, scaling_params)

    pred_daily_uns = _model_units_to_raw_cases(pred_daily_model, scaling_params, population)
    true_daily_uns = _model_units_to_raw_cases(true_daily_model, scaling_params, population)
    factual_daily_uns = _model_units_to_raw_cases(factual_daily_model, scaling_params, population)
    pred_uns = _model_units_to_raw_cases(pred_model, scaling_params, population)
    true_uns = _model_units_to_raw_cases(true_model, scaling_params, population)
    factual_uns = _model_units_to_raw_cases(factual_model, scaling_params, population)
    target_uns = _model_units_to_raw_cases(target_model, scaling_params, population)

    pred_cum_model = pred_daily_model.sum(axis=1)
    true_cum_model = true_daily_model.sum(axis=1)
    factual_cum_model = factual_daily_model.sum(axis=1)
    pred_cum_uns = pred_daily_uns.sum(axis=1)
    true_cum_uns = true_daily_uns.sum(axis=1)
    factual_cum_uns = factual_daily_uns.sum(axis=1)

    def _scalar(value: np.ndarray) -> float:
        return float(np.asarray(value, dtype=np.float64).reshape(-1).mean())

    def _per_capita_model(value: np.ndarray, base: float) -> float | None:
        return _model_units_to_per_capita(value, scaling_params, population, base)

    target_distance = np.abs(pred_uns - target_uns)
    factual_target_distance = np.abs(factual_uns - target_uns)
    target_distance_model = np.abs(pred_model - target_model)
    factual_target_distance_model = np.abs(factual_model - target_model)
    policy_vs_factual_final_improvement = factual_uns - pred_uns
    policy_vs_factual_cum_improvement = factual_cum_uns - pred_cum_uns
    policy_vs_factual_final_improvement_model = factual_model - pred_model
    policy_vs_factual_cum_improvement_model = factual_cum_model - pred_cum_model
    return {
        "event": "county_done",
        "split": split,
        "tau": int(tau),
        "county": county,
        "label": label,
        "population": float(population) if population is not None else None,
        "outcome_transform": transform,
        "rmse_norm": float(np.sqrt(((pred_final - true_final) ** 2).mean())),
        "mae_norm": float(np.abs(pred_final - true_final).mean()),
        "rmse_uns": float(np.sqrt(((pred_uns - true_uns) ** 2).mean())),
        "mae_uns": float(np.abs(pred_uns - true_uns).mean()),
        "rmse_factual_norm": float(np.sqrt(((factual_final - true_final) ** 2).mean())),
        "rmse_factual_uns": float(np.sqrt(((factual_uns - true_uns) ** 2).mean())),
        "mae_factual_uns": float(np.abs(factual_uns - true_uns).mean()),
        "rmse_cumulative_uns": float(np.sqrt(((pred_cum_uns - true_cum_uns) ** 2).mean())),
        "rmse_factual_cumulative_uns": float(np.sqrt(((factual_cum_uns - true_cum_uns) ** 2).mean())),
        "pred_final_uns": _scalar(pred_uns),
        "true_final_uns": _scalar(true_uns),
        "factual_final_uns": _scalar(factual_uns),
        "target_final_uns": _scalar(target_uns),
        "pred_cumulative_uns": _scalar(pred_cum_uns),
        "true_cumulative_uns": _scalar(true_cum_uns),
        "factual_cumulative_uns": _scalar(factual_cum_uns),
        "target_distance_uns": _scalar(target_distance),
        "factual_target_distance_uns": _scalar(factual_target_distance),
        "target_improvement_uns": _scalar(factual_target_distance - target_distance),
        "policy_vs_factual_final_improvement_uns": _scalar(policy_vs_factual_final_improvement),
        "policy_vs_factual_cumulative_improvement_uns": _scalar(policy_vs_factual_cum_improvement),
        "pred_final_model_units": _scalar(pred_model),
        "true_final_model_units": _scalar(true_model),
        "factual_final_model_units": _scalar(factual_model),
        "target_final_model_units": _scalar(target_model),
        "target_distance_model_units": _scalar(target_distance_model),
        "factual_target_distance_model_units": _scalar(factual_target_distance_model),
        "target_improvement_model_units": _scalar(factual_target_distance_model - target_distance_model),
        "policy_vs_factual_final_improvement_model_units": _scalar(policy_vs_factual_final_improvement_model),
        "policy_vs_factual_cumulative_improvement_model_units": _scalar(policy_vs_factual_cum_improvement_model),
        "pred_final_per_10k": _per_capita_model(pred_model, 10000.0),
        "true_final_per_10k": _per_capita_model(true_model, 10000.0),
        "factual_final_per_10k": _per_capita_model(factual_model, 10000.0),
        "target_final_per_10k": _per_capita_model(target_model, 10000.0),
        "pred_cumulative_per_10k": _per_capita_model(pred_cum_model, 10000.0),
        "true_cumulative_per_10k": _per_capita_model(true_cum_model, 10000.0),
        "factual_cumulative_per_10k": _per_capita_model(factual_cum_model, 10000.0),
        "target_distance_per_10k": _per_capita_model(target_distance_model, 10000.0),
        "factual_target_distance_per_10k": _per_capita_model(factual_target_distance_model, 10000.0),
        "target_improvement_per_10k": _per_capita_model(factual_target_distance_model - target_distance_model, 10000.0),
        "policy_vs_factual_final_improvement_per_10k": _per_capita_model(
            policy_vs_factual_final_improvement_model, 10000.0
        ),
        "policy_vs_factual_cumulative_improvement_per_10k": _per_capita_model(
            policy_vs_factual_cum_improvement_model, 10000.0
        ),
        "pred_final_per_100k": _per_capita_model(pred_model, 100000.0),
        "true_final_per_100k": _per_capita_model(true_model, 100000.0),
        "factual_final_per_100k": _per_capita_model(factual_model, 100000.0),
        "target_final_per_100k": _per_capita_model(target_model, 100000.0),
        "pred_cumulative_per_100k": _per_capita_model(pred_cum_model, 100000.0),
        "true_cumulative_per_100k": _per_capita_model(true_cum_model, 100000.0),
        "factual_cumulative_per_100k": _per_capita_model(factual_cum_model, 100000.0),
        "target_distance_per_100k": _per_capita_model(target_distance_model, 100000.0),
        "factual_target_distance_per_100k": _per_capita_model(factual_target_distance_model, 100000.0),
        "target_improvement_per_100k": _per_capita_model(factual_target_distance_model - target_distance_model, 100000.0),
        "policy_vs_factual_final_improvement_per_100k": _per_capita_model(
            policy_vs_factual_final_improvement_model, 100000.0
        ),
        "policy_vs_factual_cumulative_improvement_per_100k": _per_capita_model(
            policy_vs_factual_cum_improvement_model, 100000.0
        ),
        "action_mean": float(planned_actions.mean()) if planned_actions is not None and planned_actions.size else None,
        "action_std": float(planned_actions.std()) if planned_actions is not None and planned_actions.size else None,
        "action_min": float(planned_actions.min()) if planned_actions is not None and planned_actions.size else None,
        "action_max": float(planned_actions.max()) if planned_actions is not None and planned_actions.size else None,
    }


def _summarize(rows: List[dict]) -> List[dict]:
    out = []
    keys = sorted({(r["split"], r["tau"], r["label"]) for r in rows})

    def _values(group: List[dict], key: str) -> List[float]:
        vals = []
        for row in group:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                vals.append(value)
        return vals

    def _mean(group: List[dict], key: str) -> float | None:
        vals = _values(group, key)
        return float(np.mean(vals)) if vals else None

    def _median(group: List[dict], key: str) -> float | None:
        vals = _values(group, key)
        return float(np.median(vals)) if vals else None

    def _p75(group: List[dict], key: str) -> float | None:
        vals = _values(group, key)
        return float(np.percentile(vals, 75)) if vals else None

    def _max(group: List[dict], key: str) -> float | None:
        vals = _values(group, key)
        return float(np.max(vals)) if vals else None

    def _root_mean_square(group: List[dict], key: str) -> float | None:
        vals = _values(group, key)
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else None

    for split, tau, label in keys:
        group = [r for r in rows if (r["split"], r["tau"], r["label"]) == (split, tau, label)]
        if not group:
            continue
        out.append({
            "split": split,
            "tau": int(tau),
            "label": label,
            "n_counties": len({r["county"] for r in group}),
            "n_eval_points": len(group),
            "rmse_norm": _root_mean_square(group, "rmse_norm"),
            "mae_norm": _mean(group, "mae_norm"),
            "rmse_uns": _root_mean_square(group, "rmse_uns"),
            "mae_uns": _mean(group, "mae_uns"),
            "rmse_factual_norm": _root_mean_square(group, "rmse_factual_norm"),
            "rmse_factual_uns": _root_mean_square(group, "rmse_factual_uns"),
            "median_county_rmse_uns": _median(group, "rmse_uns"),
            "p75_county_rmse_uns": _p75(group, "rmse_uns"),
            "max_county_rmse_uns": _max(group, "rmse_uns"),
            "median_county_rmse_factual_uns": _median(group, "rmse_factual_uns"),
            "rmse_cumulative_uns": _root_mean_square(group, "rmse_cumulative_uns"),
            "rmse_factual_cumulative_uns": _root_mean_square(group, "rmse_factual_cumulative_uns"),
            "mean_pred_final_uns": _mean(group, "pred_final_uns"),
            "median_pred_final_uns": _median(group, "pred_final_uns"),
            "mean_true_final_uns": _mean(group, "true_final_uns"),
            "mean_factual_final_uns": _mean(group, "factual_final_uns"),
            "mean_target_final_uns": _mean(group, "target_final_uns"),
            "mean_pred_cumulative_uns": _mean(group, "pred_cumulative_uns"),
            "median_pred_cumulative_uns": _median(group, "pred_cumulative_uns"),
            "mean_true_cumulative_uns": _mean(group, "true_cumulative_uns"),
            "mean_factual_cumulative_uns": _mean(group, "factual_cumulative_uns"),
            "mean_target_distance_uns": _mean(group, "target_distance_uns"),
            "median_target_distance_uns": _median(group, "target_distance_uns"),
            "mean_factual_target_distance_uns": _mean(group, "factual_target_distance_uns"),
            "mean_target_improvement_uns": _mean(group, "target_improvement_uns"),
            "median_target_improvement_uns": _median(group, "target_improvement_uns"),
            "mean_policy_vs_factual_final_improvement_uns": _mean(
                group, "policy_vs_factual_final_improvement_uns"
            ),
            "median_policy_vs_factual_final_improvement_uns": _median(
                group, "policy_vs_factual_final_improvement_uns"
            ),
            "mean_policy_vs_factual_cumulative_improvement_uns": _mean(
                group, "policy_vs_factual_cumulative_improvement_uns"
            ),
            "median_policy_vs_factual_cumulative_improvement_uns": _median(
                group, "policy_vs_factual_cumulative_improvement_uns"
            ),
            "mean_pred_final_per_100k": _mean(group, "pred_final_per_100k"),
            "median_pred_final_per_100k": _median(group, "pred_final_per_100k"),
            "mean_factual_final_per_100k": _mean(group, "factual_final_per_100k"),
            "median_factual_final_per_100k": _median(group, "factual_final_per_100k"),
            "mean_target_final_per_100k": _mean(group, "target_final_per_100k"),
            "median_target_final_per_100k": _median(group, "target_final_per_100k"),
            "mean_pred_cumulative_per_100k": _mean(group, "pred_cumulative_per_100k"),
            "median_pred_cumulative_per_100k": _median(group, "pred_cumulative_per_100k"),
            "mean_factual_cumulative_per_100k": _mean(group, "factual_cumulative_per_100k"),
            "median_factual_cumulative_per_100k": _median(group, "factual_cumulative_per_100k"),
            "mean_target_distance_per_100k": _mean(group, "target_distance_per_100k"),
            "median_target_distance_per_100k": _median(group, "target_distance_per_100k"),
            "mean_pred_final_per_10k": _mean(group, "pred_final_per_10k"),
            "median_pred_final_per_10k": _median(group, "pred_final_per_10k"),
            "mean_factual_final_per_10k": _mean(group, "factual_final_per_10k"),
            "median_factual_final_per_10k": _median(group, "factual_final_per_10k"),
            "mean_target_final_per_10k": _mean(group, "target_final_per_10k"),
            "median_target_final_per_10k": _median(group, "target_final_per_10k"),
            "mean_pred_cumulative_per_10k": _mean(group, "pred_cumulative_per_10k"),
            "median_pred_cumulative_per_10k": _median(group, "pred_cumulative_per_10k"),
            "mean_factual_cumulative_per_10k": _mean(group, "factual_cumulative_per_10k"),
            "median_factual_cumulative_per_10k": _median(group, "factual_cumulative_per_10k"),
            "mean_target_distance_per_10k": _mean(group, "target_distance_per_10k"),
            "median_target_distance_per_10k": _median(group, "target_distance_per_10k"),
            "mean_policy_vs_factual_final_improvement_per_10k": _mean(
                group, "policy_vs_factual_final_improvement_per_10k"
            ),
            "median_policy_vs_factual_final_improvement_per_10k": _median(
                group, "policy_vs_factual_final_improvement_per_10k"
            ),
            "mean_policy_vs_factual_cumulative_improvement_per_10k": _mean(
                group, "policy_vs_factual_cumulative_improvement_per_10k"
            ),
            "median_policy_vs_factual_cumulative_improvement_per_10k": _median(
                group, "policy_vs_factual_cumulative_improvement_per_10k"
            ),
            "mean_policy_vs_factual_final_improvement_per_100k": _mean(
                group, "policy_vs_factual_final_improvement_per_100k"
            ),
            "median_policy_vs_factual_final_improvement_per_100k": _median(
                group, "policy_vs_factual_final_improvement_per_100k"
            ),
            "mean_policy_vs_factual_cumulative_improvement_per_100k": _mean(
                group, "policy_vs_factual_cumulative_improvement_per_100k"
            ),
            "median_policy_vs_factual_cumulative_improvement_per_100k": _median(
                group, "policy_vs_factual_cumulative_improvement_per_100k"
            ),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", action="append", default=[], help="label=path; can be repeated")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--taus", nargs="+", type=int, default=[7, 14, 21])
    parser.add_argument("--window-mode", choices=["last", "fixed-start"], default="last")
    parser.add_argument("--decision-day", type=int, default=None, help="History length for fixed-start eval.")
    parser.add_argument(
        "--target-mode",
        choices=[
            "factual_final",
            "half_factual_final",
            "relative_factual_final",
            "relative_current",
            "absolute_final",
        ],
        default="factual_final",
    )
    parser.add_argument("--target-scale", type=float, default=0.5)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--factual-only", action="store_true", help="Run only factual replay sanity rows.")
    parser.add_argument("--include-factual-row", action="store_true", help="Also write factual replay rows.")
    parser.add_argument("--selector", default="q_sample")
    parser.add_argument("--candidate-actions", type=int, default=64)
    parser.add_argument("--q-bc-penalty", type=float, default=1.0)
    parser.add_argument("--candidate-noise-std", type=float, default=0.25)
    parser.add_argument("--eval-seed", type=int, default=20260708)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--abm-device", default="cpu")
    parser.add_argument("--epi-root", default=None, help="Override cfg.dataset.epi_root, useful for isolated runtimes.")
    parser.add_argument("--processed-data-dir", default=None, help="Override cfg.dataset.processed_data_dir.")
    parser.add_argument("--counties", nargs="+", default=None, help="Override cfg.dataset.counties.")
    parser.add_argument("--counties-from-epicf-csv", default=None, help="Override cfg.dataset.counties_from_epicf_csv.")
    parser.add_argument("--generate-if-missing", action="store_true", help="Allow dataset cache generation.")
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate dataset cache before eval.")
    parser.add_argument("--cache-version", default=None, help="Override cfg.dataset.cache_version.")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Override cfg.dataset.seed.")
    parser.add_argument(
        "--outcome-transform",
        choices=["raw_cases_zscore", "per10k_cases_zscore"],
        default=None,
        help="Override cfg.dataset.outcome_transform.",
    )
    parser.add_argument("--max-counties", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=0, help="Inclusive row offset within each split.")
    parser.add_argument("--row-end", type=int, default=None, help="Exclusive row offset within each split.")
    args = parser.parse_args()
    if not args.ckpt and not args.factual_only:
        raise ValueError("At least one --ckpt is required unless --factual-only is set.")
    if args.window_mode == "fixed-start" and args.decision_day is None:
        raise ValueError("--decision-day is required when --window-mode fixed-start")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "county_metrics.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    if args.epi_root is not None:
        cfg.dataset.epi_root = args.epi_root
    if args.processed_data_dir is not None:
        cfg.dataset.processed_data_dir = args.processed_data_dir
    if args.counties is not None:
        cfg.dataset.counties = list(args.counties)
        cfg.dataset.counties_from_epicf_csv = None
        cfg.dataset.county = str(args.counties[0])
    if args.counties_from_epicf_csv is not None:
        cfg.dataset.counties_from_epicf_csv = args.counties_from_epicf_csv
        cfg.dataset.counties = None
    if args.generate_if_missing:
        cfg.dataset.generate_if_missing = True
    if args.force_regenerate:
        cfg.dataset.force_regenerate = True
    if args.cache_version is not None:
        cfg.dataset.cache_version = args.cache_version
    if args.dataset_seed is not None:
        cfg.dataset.seed = int(args.dataset_seed)
    if args.outcome_transform is not None:
        cfg.dataset.outcome_transform = args.outcome_transform
    cfg.dataset.device = args.abm_device
    cfg.exp.device = args.model_device
    cfg.exp.iql_eval_action_selector = args.selector
    cfg.exp.iql_eval_candidate_actions = args.candidate_actions
    cfg.exp.iql_eval_q_bc_penalty = args.q_bc_penalty
    cfg.exp.iql_eval_candidate_noise_std = args.candidate_noise_std

    set_seed(int(cfg.exp.seed))
    dataset_collection = instantiate(cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(cfg.dataset.static_size) > 0 and len(dataset_collection.train_f.data["static_features"].shape) == 2:
        dataset_collection = repeat_static(dataset_collection)

    device = args.model_device if torch.cuda.is_available() and args.model_device.startswith("cuda") else "cpu"
    ckpt_paths = _parse_label_path(args.ckpt)
    models = {}
    for label, ckpt_path in ckpt_paths.items():
        path = ckpt_path if ckpt_path.is_absolute() else Path.cwd() / ckpt_path
        inference_model = InferenceModel(cfg).to(device)
        planner = load_em_for_eval(inference_model, str(path), device)
        inference_model.eval()
        planner.actor.eval()
        models[label] = (inference_model, planner, str(path))

    manifest_path.write_text(json.dumps({
        "config": str(Path(args.config).resolve()),
        "ckpts": {k: v[2] for k, v in models.items()},
        "splits": args.splits,
        "taus": args.taus,
        "window_mode": args.window_mode,
        "decision_day": args.decision_day,
        "target_mode": args.target_mode,
        "target_scale": args.target_scale,
        "target_value": args.target_value,
        "factual_only": bool(args.factual_only),
        "include_factual_row": bool(args.include_factual_row),
        "selector": args.selector,
        "candidate_actions": args.candidate_actions,
        "q_bc_penalty": args.q_bc_penalty,
        "candidate_noise_std": args.candidate_noise_std,
        "candidate_seed_scope": "shared_by_county_tau_step",
        "eval_seed": args.eval_seed,
        "model_device": device,
        "abm_device": args.abm_device,
        "epi_root": str(cfg.dataset.epi_root),
        "processed_data_dir": str(cfg.dataset.processed_data_dir),
        "counties": list(cfg.dataset.get("counties", []) or []),
        "counties_from_epicf_csv": cfg.dataset.get("counties_from_epicf_csv", None),
        "generate_if_missing": bool(cfg.dataset.generate_if_missing),
        "force_regenerate": bool(cfg.dataset.force_regenerate),
        "cache_version": cfg.dataset.get("cache_version", None),
        "dataset_seed": int(cfg.dataset.seed),
        "outcome_transform": cfg.dataset.get("outcome_transform", "raw_cases_zscore"),
        "scaling_outcome_transform": (
            dataset_collection.train_scaling_params.get("outcome_transform", "raw_cases_zscore")
            if isinstance(dataset_collection.train_scaling_params, dict)
            else None
        ),
        "row_start": args.row_start,
        "row_end": args.row_end,
        "max_counties": args.max_counties,
        "started_at": time.time(),
    }, indent=2, sort_keys=True) + "\n")

    all_rows: List[dict] = []
    action_hold_days = max(1, int(OmegaConf.select(cfg, "dataset.action_hold_days", default=1)))
    max_tau = float(OmegaConf.select(cfg, "exp.max_tau", default=max(args.taus)))

    with metrics_path.open("w") as metrics_f:
        for split in args.splits:
            if split == "train":
                fold = dataset_collection.train_f
            elif split == "val":
                fold = dataset_collection.val_f
            elif split == "test":
                fold = dataset_collection.test_f
            else:
                raise ValueError(f"Unknown split={split!r}; expected train, val, or test.")
            data = fold.data
            n_rows = int(np.asarray(data["sequence_lengths"]).shape[0])
            row_indices = list(range(n_rows))
            row_start = max(0, int(args.row_start))
            row_end = n_rows if args.row_end is None else min(n_rows, int(args.row_end))
            if row_start > row_end:
                raise ValueError(f"row-start={row_start} is greater than row-end={row_end}")
            row_indices = row_indices[row_start:row_end]
            if args.max_counties is not None:
                row_indices = row_indices[:int(args.max_counties)]

            for tau in args.taus:
                cfg.exp.tau = int(tau)
                print(json.dumps({
                    "event": "split_tau_start",
                    "split": split,
                    "tau": int(tau),
                    "n_counties": len(row_indices),
                    "labels": list(models),
                    "window_mode": args.window_mode,
                    "decision_day": args.decision_day,
                    "target_mode": args.target_mode,
                }), flush=True)
                for local_idx, row_idx in enumerate(row_indices):
                    county = _county_id(data, row_idx)
                    H, targets, decision_day = _slice_county_window(
                        data,
                        row_idx,
                        int(tau),
                        window_mode=args.window_mode,
                        decision_day=args.decision_day,
                    )
                    true_norm_daily = targets["outputs"].detach().cpu().numpy().astype(np.float32)
                    target_norm_final, _ = _make_eval_target_norm(
                        H,
                        targets,
                        dataset_collection.train_scaling_params,
                        target_mode=args.target_mode,
                        target_scale=args.target_scale,
                        target_value=args.target_value,
                    )
                    population = _population_from_row(data, row_idx)
                    factual_norm_daily = _rollout_factual(fold, dataset_collection, H, targets, int(tau))
                    if args.factual_only or args.include_factual_row:
                        row = _metric_row(
                            split=split,
                            tau=int(tau),
                            county=county,
                            label="factual_replay",
                            pred_norm_daily=factual_norm_daily,
                            true_norm_daily=true_norm_daily,
                            factual_norm_daily=factual_norm_daily,
                            target_norm_final=target_norm_final,
                            planned_actions=None,
                            scaling_params=dataset_collection.train_scaling_params,
                            population=population,
                        )
                        row.update({
                            "idx": int(local_idx),
                            "row_idx": int(row_idx),
                            "decision_day": int(decision_day),
                            "window_mode": args.window_mode,
                            "target_mode": args.target_mode,
                            "target_scale": float(args.target_scale),
                            "target_value": args.target_value,
                            "elapsed_sec": 0.0,
                        })
                        all_rows.append(row)
                        metrics_f.write(json.dumps(row, sort_keys=True) + "\n")
                        metrics_f.flush()
                        summary_path.write_text(json.dumps(_summarize(all_rows), indent=2, sort_keys=True) + "\n")
                        print(json.dumps(row, sort_keys=True), flush=True)
                    if args.factual_only:
                        _release_county_cache(fold, county)
                        continue

                    for label, (inference_model, planner, _) in models.items():
                        started = time.time()
                        pred_norm_daily, planned_actions = _rollout_policy(
                            fold=fold,
                            dataset_collection=dataset_collection,
                            inference_model=inference_model,
                            planner=planner,
                            H=H,
                            eval_target_norm=target_norm_final,
                            label=label,
                            county=county,
                            tau=int(tau),
                            max_tau=max_tau,
                            action_hold_days=action_hold_days,
                            selector=args.selector,
                            candidate_actions=args.candidate_actions,
                            q_bc_penalty=args.q_bc_penalty,
                            candidate_noise_std=args.candidate_noise_std,
                            eval_seed=args.eval_seed,
                            device=device,
                        )
                        row = _metric_row(
                            split=split,
                            tau=int(tau),
                            county=county,
                            label=label,
                            pred_norm_daily=pred_norm_daily,
                            true_norm_daily=true_norm_daily,
                            factual_norm_daily=factual_norm_daily,
                            target_norm_final=target_norm_final,
                            planned_actions=planned_actions,
                            scaling_params=dataset_collection.train_scaling_params,
                            population=population,
                        )
                        row.update({
                            "idx": int(local_idx),
                            "row_idx": int(row_idx),
                            "decision_day": int(decision_day),
                            "window_mode": args.window_mode,
                            "target_mode": args.target_mode,
                            "target_scale": float(args.target_scale),
                            "target_value": args.target_value,
                            "elapsed_sec": round(time.time() - started, 3),
                        })
                        all_rows.append(row)
                        metrics_f.write(json.dumps(row, sort_keys=True) + "\n")
                        metrics_f.flush()
                        summary_path.write_text(json.dumps(_summarize(all_rows), indent=2, sort_keys=True) + "\n")
                        print(json.dumps(row, sort_keys=True), flush=True)
                    _release_county_cache(fold, county)
                print(json.dumps({
                    "event": "split_tau_complete",
                    "split": split,
                    "tau": int(tau),
                    "summary": _summarize(all_rows),
                }), flush=True)

    summary = _summarize(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "eval_done", "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
