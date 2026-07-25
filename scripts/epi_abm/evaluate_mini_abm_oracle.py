"""Evaluate a shared nine-action EpiABM oracle without changing formal runs.

The oracle is deliberately restricted to constant action plans from
``{0, 0.5, 1}^2``.  Every method sees the same GPU factual replay and the same
ABM candidate trajectories.  Model checkpoints only rank this shared grid.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions import Distribution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.epi_abm.evaluate_county_last_window_iql import (  # noqa: E402
    _county_id,
    _output_mean_std,
    _population_from_row,
    _release_county_cache,
    _rollout_factual,
    _slice_county_window,
    _to_device_history,
)
from scripts.epi_abm.evaluate_county_major_iql import (  # noqa: E402
    _dataset_signature,
    _load_job_models,
    _protocol_hash,
)
from src.data.iql_dataset_builder import align_h_t_static_to_history  # noqa: E402
from src.evaluation.iql_planner_eval import _iql_augmented_state  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402


DEFAULT_METHODS = ("full_cripo", "conv1d_only", "attention_only")


def action_grid(levels: Sequence[float]) -> np.ndarray:
    values = tuple(float(value) for value in levels)
    if not values:
        raise ValueError("At least one action level is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Action levels must be in [0, 1], got {values}")
    return np.asarray(list(itertools.product(values, repeat=2)), dtype=np.float32)


def training_return_loss(
    pred_norm_daily: np.ndarray,
    target_norm_final: float,
    *,
    tau: int,
    discount: float,
    reward_clip: float,
) -> float:
    """Return the positive loss whose negative is the implemented IQL return."""
    values = np.asarray(pred_norm_daily, dtype=np.float64).reshape(-1)[: int(tau)]
    if values.size != int(tau):
        raise ValueError(f"Expected {tau} rollout values, got {values.size}")
    distance = np.abs(values - float(target_norm_final))
    if float(reward_clip) > 0.0:
        distance = np.minimum(distance, float(reward_clip))
    weights = np.power(float(discount), np.arange(int(tau), dtype=np.float64))
    return float(np.dot(weights, distance))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _normalize_jobs(
    source_manifest: Path,
    *,
    methods: Iterable[str],
    selector: str,
    q_bc_penalty: float,
) -> List[dict]:
    raw = json.loads(source_manifest.read_text())
    allowed = set(methods)
    jobs = []
    for item in raw.get("jobs", []):
        method = str(item.get("method", ""))
        if method not in allowed:
            continue
        ckpts = dict(item.get("ckpts", {}))
        if len(ckpts) != 1:
            raise ValueError(
                f"Mini oracle requires one validation-selected checkpoint for {item.get('id')}; "
                f"got {sorted(ckpts)}"
            )
        config = _absolute(str(item["config"]), source_manifest.parent)
        normalized_ckpts = {
            str(label): str(_absolute(str(path), source_manifest.parent))
            for label, path in ckpts.items()
        }
        missing = [path for path in [config, *map(Path, normalized_ckpts.values())] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files for {item.get('id')}: {missing}")
        jobs.append({
            "id": str(item["id"]),
            "method": method,
            "seed": int(item["seed"]),
            "config": str(config),
            "ckpts": normalized_ckpts,
            "selector": str(selector),
            "candidate_actions": 9,
            "q_bc_penalty": float(q_bc_penalty),
            "candidate_noise_std": 0.0,
            "eval_seed": 0,
        })
    expected = {(method, seed) for method in allowed for seed in (10, 101, 1010, 10101, 101010)}
    observed = {(job["method"], job["seed"]) for job in jobs}
    if observed != expected:
        raise ValueError(
            f"Expected 3 methods x 5 seeds from validation-selected manifest; "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    return sorted(jobs, key=lambda job: (job["method"], job["seed"]))


def _load_existing(path: Path, key_fields: Sequence[str]) -> tuple[List[dict], set]:
    rows: List[dict] = []
    keys = set()
    if not path.exists():
        return rows, keys
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = tuple(row[field] for field in key_fields)
        if key in keys:
            raise ValueError(f"Duplicate key in {path}:{line_number}: {key}")
        keys.add(key)
        rows.append(row)
    return rows, keys


def _unscale_daily(norm_daily: np.ndarray, scaling_params) -> np.ndarray:
    means, stds = _output_mean_std(scaling_params)
    return np.asarray(norm_daily, dtype=np.float64) * stds + means


def _observed_action_support(dataset_collection) -> set[tuple[float, float]]:
    data = dataset_collection.train_f.data
    actions = np.asarray(data["current_treatments"], dtype=np.float64)
    active = np.asarray(data.get("active_entries", np.ones(actions.shape[:-1] + (1,))))
    valid = actions[active[..., 0] > 0.0]
    return {tuple(np.round(row[:2], 6).tolist()) for row in valid}


@torch.no_grad()
def _score_grid(
    *,
    inference_model,
    planner,
    history: Dict[str, torch.Tensor],
    target_norm_final: float,
    tau: int,
    max_tau: float,
    grid_sim: np.ndarray,
    q_bc_penalty: float,
    device: str,
) -> dict:
    h_work = align_h_t_static_to_history(_to_device_history(history, device))
    z, _, _ = inference_model.ct_hidden_history(h_work)
    target = torch.full((z.size(0), 1), float(target_norm_final), dtype=z.dtype, device=z.device)
    a_prev = gather_last_valid(h_work["current_treatments"], h_work.get("active_entries"))
    max_action = float(planner.actor.max_action)
    a_prev_tanh = (2.0 * torch.clamp(a_prev, 0.0, 1.0) - 1.0) * max_action
    obs = _iql_augmented_state(planner, z, target, 0, int(tau), float(max_tau), a_prev_tanh)

    policy_out = planner.actor(obs)
    actor_mean_tanh = policy_out.mean if isinstance(policy_out, Distribution) else policy_out
    grid_tanh_np = (2.0 * np.asarray(grid_sim, dtype=np.float32) - 1.0) * max_action
    grid_tanh = torch.as_tensor(grid_tanh_np, dtype=obs.dtype, device=obs.device)
    obs_rep = obs.expand(grid_tanh.size(0), obs.size(-1))
    q_values = planner.qf(obs_rep, grid_tanh).reshape(-1)
    dist_sq = (grid_tanh / max_action - actor_mean_tanh.expand_as(grid_tanh)).pow(2).sum(dim=-1)
    regularized = q_values - q_values.new_tensor(float(q_bc_penalty)) * dist_sq
    return {
        "q_values": q_values.detach().cpu().double().tolist(),
        "bc_distance_sq": dist_sq.detach().cpu().double().tolist(),
        "regularized_scores": regularized.detach().cpu().double().tolist(),
        "q_choice": int(torch.argmax(q_values).item()),
        "regularized_choice": int(torch.argmax(regularized).item()),
        "actor_mean_sim": torch.clamp(
            (actor_mean_tanh / max_action + 1.0) / 2.0, 0.0, 1.0
        ).detach().cpu().double().reshape(-1).tolist(),
    }


def _candidate_metrics(
    *,
    candidate_norm: np.ndarray,
    factual_norm: np.ndarray,
    grid: np.ndarray,
    taus: Sequence[int],
    scaling_params,
    discount: float,
    reward_clip: float,
    support: set[tuple[float, float]],
) -> Dict[int, List[dict]]:
    candidate_model = _unscale_daily(candidate_norm, scaling_params)
    factual_model = _unscale_daily(factual_norm, scaling_params)
    by_tau: Dict[int, List[dict]] = {}
    for tau in taus:
        target_norm = float(np.asarray(factual_norm)[0, int(tau) - 1, 0])
        target_model = float(np.asarray(factual_model)[0, int(tau) - 1, 0])
        rows = []
        for action_index, action in enumerate(grid):
            pred_norm = candidate_norm[action_index, : int(tau), 0]
            pred_model = candidate_model[action_index, : int(tau), 0]
            factual_prefix = factual_model[0, : int(tau), 0]
            rows.append({
                "action_index": int(action_index),
                "action": [float(action[0]), float(action[1])],
                "in_observed_support": tuple(np.round(action, 6).tolist()) in support,
                "training_return_loss": training_return_loss(
                    pred_norm,
                    target_norm,
                    tau=int(tau),
                    discount=float(discount),
                    reward_clip=float(reward_clip),
                ),
                "terminal_error_per_10k": abs(float(pred_model[-1]) - target_model),
                "trajectory_rmse_per_10k": float(
                    np.sqrt(np.mean(np.square(pred_model - factual_prefix)))
                ),
                "pred_final_per_10k": float(pred_model[-1]),
                "factual_target_per_10k": target_model,
            })
        by_tau[int(tau)] = rows
    return by_tau


def _argmin(rows: Sequence[dict], field: str) -> int:
    return min(range(len(rows)), key=lambda index: (float(rows[index][field]), index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--taus", nargs="+", type=int, default=[7, 14, 21])
    parser.add_argument("--action-levels", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--decision-day", type=int, default=161)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--reward-clip", type=float, default=3.0)
    parser.add_argument("--selector", default="explicit_grid_q_bc")
    parser.add_argument("--q-bc-penalty", type=float, default=1.0)
    parser.add_argument("--model-device", required=True)
    parser.add_argument("--abm-device", required=True)
    parser.add_argument("--epi-root", required=True)
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--cache-version", required=True)
    parser.add_argument("--dataset-seed", type=int, default=100)
    parser.add_argument("--outcome-transform", default="per10k_cases_zscore")
    parser.add_argument("--row-indices", nargs="+", type=int, default=None)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=None)
    args = parser.parse_args()

    if args.selector != "explicit_grid_q_bc":
        raise ValueError("The mini oracle protocol requires selector=explicit_grid_q_bc")
    taus = sorted({int(tau) for tau in args.taus})
    if not taus or taus[0] <= 0:
        raise ValueError(f"Invalid horizons: {taus}")
    max_eval_tau = max(taus)
    grid = action_grid(args.action_levels)
    if grid.shape != (9, 2):
        raise ValueError(f"The formal mini oracle requires exactly 9 actions, got {grid.shape}")

    source_manifest = Path(args.source_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    shard_dir = out_dir / args.split / "shards" / args.worker_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = _normalize_jobs(
        source_manifest,
        methods=args.methods,
        selector=args.selector,
        q_bc_penalty=args.q_bc_penalty,
    )

    base_cfg = OmegaConf.load(jobs[0]["config"])
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.dataset.epi_root = args.epi_root
    base_cfg.dataset.processed_data_dir = args.processed_data_dir
    base_cfg.dataset.cache_version = args.cache_version
    base_cfg.dataset.seed = int(args.dataset_seed)
    base_cfg.dataset.outcome_transform = args.outcome_transform
    base_cfg.dataset.device = args.abm_device
    base_cfg.dataset.generate_if_missing = False
    base_cfg.dataset.force_regenerate = False
    base_cfg.exp.device = args.model_device
    expected_signature = _dataset_signature(base_cfg)
    for job in jobs[1:]:
        cfg = OmegaConf.load(job["config"])
        OmegaConf.set_struct(cfg, False)
        cfg.dataset.processed_data_dir = args.processed_data_dir
        cfg.dataset.cache_version = args.cache_version
        cfg.dataset.seed = int(args.dataset_seed)
        cfg.dataset.outcome_transform = args.outcome_transform
        if _dataset_signature(cfg) != expected_signature:
            raise ValueError(f"Dataset protocol mismatch for {job['id']}")

    set_seed(int(base_cfg.exp.seed))
    dataset_collection = instantiate(base_cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if (
        int(base_cfg.dataset.static_size) > 0
        and len(dataset_collection.train_f.data["static_features"].shape) == 2
    ):
        dataset_collection = repeat_static(dataset_collection)
    dataset_collection.simulator_cache.configure_snapshot_days([int(args.decision_day)])
    fold = dataset_collection.val_f if args.split == "val" else dataset_collection.test_f
    data = fold.data
    n_rows = int(np.asarray(data["sequence_lengths"]).shape[0])
    if args.row_indices is None:
        end = n_rows if args.row_end is None else min(n_rows, int(args.row_end))
        row_indices = list(range(max(0, int(args.row_start)), end))
    else:
        row_indices = [int(index) for index in args.row_indices]
    if any(index < 0 or index >= n_rows for index in row_indices):
        raise ValueError(f"row indices must be in 0..{n_rows - 1}: {row_indices}")

    support = _observed_action_support(dataset_collection)
    device = args.model_device if torch.cuda.is_available() else "cpu"
    protocol_family = {
        "schema": "epi_abm_nine_action_mini_oracle_v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "methods": list(args.methods),
        "split": args.split,
        "taus": taus,
        "decision_day": int(args.decision_day),
        "action_grid": grid.tolist(),
        "plan_type": "constant_action_over_horizon",
        "oracle_objective": "discounted_clipped_distance_to_gpu_factual_horizon_endpoint",
        "discount": float(args.discount),
        "reward_clip": float(args.reward_clip),
        "reward_scale": "omitted_positive_auto_scale_does_not_change_argmin",
        "selector": args.selector,
        "q_bc_penalty": float(args.q_bc_penalty),
        "target_reference": "same_device_gpu_factual_replay",
        "common_random_numbers": True,
        "model_device_backend": device.split(":", 1)[0],
        "abm_device_backend": args.abm_device.split(":", 1)[0],
        "dataset_signature": expected_signature,
        "processed_data_dir": str(args.processed_data_dir),
        "cache_version": str(args.cache_version),
        "dataset_seed": int(args.dataset_seed),
        "outcome_transform": str(args.outcome_transform),
        "observed_train_action_support": [list(value) for value in sorted(support)],
    }
    protocol = {
        **protocol_family,
        "worker_id": args.worker_id,
        "row_indices": row_indices,
        "model_device": device,
        "abm_device": args.abm_device,
        "epi_root": str(Path(args.epi_root).resolve()),
    }
    family_hash = _protocol_hash(protocol_family)
    protocol_hash = _protocol_hash(protocol)
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("protocol_hash") != protocol_hash:
            raise ValueError(f"Resume protocol mismatch: {manifest_path}")
    manifest_path.write_text(json.dumps({
        **protocol,
        "protocol_family_hash": family_hash,
        "protocol_hash": protocol_hash,
        "jobs": jobs,
        "started_at": time.time(),
    }, indent=2, sort_keys=True) + "\n")

    oracle_path = shard_dir / "oracle_rows.jsonl"
    model_path = shard_dir / "model_rows.jsonl"
    oracle_rows, oracle_done = _load_existing(
        oracle_path, ("split", "row_idx", "tau", "action_index")
    )
    model_rows, model_done = _load_existing(
        model_path, ("split", "row_idx", "tau", "job_id")
    )
    max_tau_for_model = float(OmegaConf.select(base_cfg, "exp.max_tau", default=max_eval_tau))

    for row_idx in row_indices:
        county = _county_id(data, row_idx)
        expected_oracle = {
            (args.split, row_idx, tau, action_index)
            for tau in taus for action_index in range(len(grid))
        }
        expected_model = {
            (args.split, row_idx, tau, job["id"])
            for tau in taus for job in jobs
        }
        if expected_oracle <= oracle_done and expected_model <= model_done:
            continue
        started = time.time()
        print(json.dumps({
            "event": "mini_oracle_county_start",
            "split": args.split,
            "row_idx": row_idx,
            "county": county,
        }), flush=True)
        try:
            history, targets, decision_day = _slice_county_window(
                data,
                row_idx,
                max_eval_tau,
                window_mode="fixed-start",
                decision_day=int(args.decision_day),
            )
            factual_norm = _rollout_factual(
                fold, dataset_collection, history, targets, max_eval_tau
            )
            candidate_norm = []
            for action in grid:
                actions = np.tile(action.reshape(1, 1, 2), (1, max_eval_tau, 1))
                candidate_norm.append(
                    fold.simulate_output_after_actions(
                        history,
                        actions,
                        dataset_collection.train_scaling_params,
                        return_daily=True,
                    )[0]
                )
            candidate_norm_np = np.stack(candidate_norm, axis=0).astype(np.float32)
            metrics_by_tau = _candidate_metrics(
                candidate_norm=candidate_norm_np,
                factual_norm=factual_norm,
                grid=grid,
                taus=taus,
                scaling_params=dataset_collection.train_scaling_params,
                discount=float(args.discount),
                reward_clip=float(args.reward_clip),
                support=support,
            )
            factual_actions = targets["current_treatments"].detach().cpu().numpy()[0]
            factual_constant = bool(
                np.allclose(factual_actions, factual_actions[:1], atol=1e-7, rtol=0.0)
            )
            factual_grid_index = None
            factual_candidate_max_abs_diff_norm = None
            if factual_constant:
                matches = np.where(np.all(np.isclose(grid, factual_actions[0], atol=1e-7), axis=1))[0]
                factual_grid_index = int(matches[0]) if matches.size else None
                if factual_grid_index is not None:
                    factual_candidate_max_abs_diff_norm = float(np.max(np.abs(
                        candidate_norm_np[factual_grid_index]
                        - factual_norm[0]
                    )))
                    if factual_candidate_max_abs_diff_norm != 0.0:
                        raise AssertionError(
                            "The constant factual grid action did not exactly reproduce the "
                            f"GPU factual replay for county={county}: "
                            f"max_abs_diff_norm={factual_candidate_max_abs_diff_norm}"
                        )

            with oracle_path.open("a") as handle:
                for tau in taus:
                    for candidate in metrics_by_tau[tau]:
                        key = (args.split, row_idx, tau, candidate["action_index"])
                        if key in oracle_done:
                            continue
                        row = {
                            **candidate,
                            "split": args.split,
                            "row_idx": int(row_idx),
                            "county": county,
                            "population": _population_from_row(data, row_idx),
                            "tau": int(tau),
                            "decision_day": int(decision_day),
                            "factual_action_constant": factual_constant,
                            "factual_action": factual_actions[0, :2].astype(float).tolist(),
                            "factual_grid_index": factual_grid_index,
                            "factual_candidate_max_abs_diff_norm": factual_candidate_max_abs_diff_norm,
                            "worker_id": args.worker_id,
                        }
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                        oracle_rows.append(row)
                        oracle_done.add(key)

            with model_path.open("a") as handle:
                for job in jobs:
                    needed = [
                        tau for tau in taus
                        if (args.split, row_idx, tau, job["id"]) not in model_done
                    ]
                    if not needed:
                        continue
                    cfg, models = _load_job_models(job, device)
                    try:
                        label, (inference_model, planner) = next(iter(models.items()))
                        for tau in needed:
                            candidate_rows = metrics_by_tau[tau]
                            target_norm = float(factual_norm[0, tau - 1, 0])
                            scores = _score_grid(
                                inference_model=inference_model,
                                planner=planner,
                                history=history,
                                target_norm_final=target_norm,
                                tau=tau,
                                max_tau=max_tau_for_model,
                                grid_sim=grid,
                                q_bc_penalty=float(args.q_bc_penalty),
                                device=device,
                            )
                            train_oracle = _argmin(candidate_rows, "training_return_loss")
                            terminal_oracle = _argmin(candidate_rows, "terminal_error_per_10k")
                            trajectory_oracle = _argmin(candidate_rows, "trajectory_rmse_per_10k")
                            selected = int(scores["regularized_choice"])
                            q_selected = int(scores["q_choice"])
                            row = {
                                "split": args.split,
                                "row_idx": int(row_idx),
                                "county": county,
                                "tau": int(tau),
                                "decision_day": int(decision_day),
                                "job_id": job["id"],
                                "method": job["method"],
                                "seed": int(job["seed"]),
                                "checkpoint_label": label,
                                "checkpoint": next(iter(job["ckpts"].values())),
                                "train_oracle_action_index": train_oracle,
                                "terminal_oracle_action_index": terminal_oracle,
                                "trajectory_oracle_action_index": trajectory_oracle,
                                "selected_action_index": selected,
                                "selected_action": candidate_rows[selected]["action"],
                                "selected_in_observed_support": candidate_rows[selected]["in_observed_support"],
                                "q_selected_action_index": q_selected,
                                "q_selected_action": candidate_rows[q_selected]["action"],
                                "q_selected_in_observed_support": candidate_rows[q_selected]["in_observed_support"],
                                "training_return_regret": float(
                                    candidate_rows[selected]["training_return_loss"]
                                    - candidate_rows[train_oracle]["training_return_loss"]
                                ),
                                "q_training_return_regret": float(
                                    candidate_rows[q_selected]["training_return_loss"]
                                    - candidate_rows[train_oracle]["training_return_loss"]
                                ),
                                "terminal_regret_per_10k": float(
                                    candidate_rows[selected]["terminal_error_per_10k"]
                                    - candidate_rows[terminal_oracle]["terminal_error_per_10k"]
                                ),
                                "trajectory_regret_per_10k": float(
                                    candidate_rows[selected]["trajectory_rmse_per_10k"]
                                    - candidate_rows[trajectory_oracle]["trajectory_rmse_per_10k"]
                                ),
                                "selected_training_return_loss": candidate_rows[selected]["training_return_loss"],
                                "selected_terminal_error_per_10k": candidate_rows[selected]["terminal_error_per_10k"],
                                "selected_trajectory_rmse_per_10k": candidate_rows[selected]["trajectory_rmse_per_10k"],
                                "train_oracle_agreement": selected == train_oracle,
                                "q_train_oracle_agreement": q_selected == train_oracle,
                                "terminal_oracle_agreement": selected == terminal_oracle,
                                "factual_grid_index": factual_grid_index,
                                "factual_action_constant": factual_constant,
                                "factual_candidate_max_abs_diff_norm": factual_candidate_max_abs_diff_norm,
                                "q_values": scores["q_values"],
                                "bc_distance_sq": scores["bc_distance_sq"],
                                "regularized_scores": scores["regularized_scores"],
                                "actor_mean_sim": scores["actor_mean_sim"],
                                "worker_id": args.worker_id,
                            }
                            if row["training_return_regret"] < -1e-10:
                                raise AssertionError(f"Negative oracle regret: {row}")
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                            model_rows.append(row)
                            model_done.add((args.split, row_idx, tau, job["id"]))
                    finally:
                        del models
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        finally:
            _release_county_cache(fold, county)
        print(json.dumps({
            "event": "mini_oracle_county_complete",
            "split": args.split,
            "row_idx": row_idx,
            "county": county,
            "elapsed_sec": round(time.time() - started, 3),
        }), flush=True)

    expected_oracle_count = len(row_indices) * len(taus) * len(grid)
    expected_model_count = len(row_indices) * len(taus) * len(jobs)
    summary = {
        "schema": "epi_abm_nine_action_mini_oracle_shard_summary_v1",
        "protocol_family_hash": family_hash,
        "protocol_hash": protocol_hash,
        "worker_id": args.worker_id,
        "split": args.split,
        "row_indices": row_indices,
        "oracle_rows": len(oracle_rows),
        "model_rows": len(model_rows),
        "expected_oracle_rows": expected_oracle_count,
        "expected_model_rows": expected_model_count,
        "complete": len(oracle_rows) == expected_oracle_count and len(model_rows) == expected_model_count,
        "finished_at": time.time(),
    }
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not summary["complete"]:
        raise RuntimeError(f"Incomplete mini-oracle shard: {summary}")
    print(json.dumps({"event": "mini_oracle_worker_done", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
