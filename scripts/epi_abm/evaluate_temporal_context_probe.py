"""Probe local and long-range temporal use with cached mini-oracle outcomes.

This experiment never runs the ABM.  It reuses the shared nine-action oracle
trajectories and only re-scores the validation-selected model checkpoints after
history truncation or temporal-block occlusion.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    _slice_county_window,
    _to_device_history,
)
from scripts.epi_abm.evaluate_county_major_iql import (  # noqa: E402
    _dataset_signature,
    _load_job_models,
    _protocol_hash,
)
from scripts.epi_abm.evaluate_mini_abm_oracle import (  # noqa: E402
    DEFAULT_METHODS,
    _normalize_jobs,
    action_grid,
)
from src.data.iql_dataset_builder import align_h_t_static_to_history  # noqa: E402
from src.evaluation.iql_planner_eval import _iql_augmented_state  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402


DEFAULT_LENGTHS = (7, 14, 28, 56, 112, 161)
DEFAULT_OCCLUSION_BLOCKS = (
    ("recent_1_7", 1, 7),
    ("local_8_28", 8, 28),
    ("mid_29_56", 29, 56),
    ("long_57_112", 57, 112),
    ("remote_113_161", 113, 161),
)
OCCLUSION_KEYS = (
    "vitals",
    "current_covariates",
    "prev_treatments",
    "prev_outputs",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_oracle_split(oracle_run_dir: Path, split: str) -> Tuple[Dict[tuple, dict], Dict[tuple, dict]]:
    split_dir = oracle_run_dir / split / "shards"
    oracle: Dict[tuple, dict] = {}
    baseline: Dict[tuple, dict] = {}
    for path in sorted(split_dir.glob("*/oracle_rows.jsonl")):
        for row in _read_jsonl(path):
            key = (int(row["row_idx"]), int(row["tau"]), int(row["action_index"]))
            if key in oracle:
                raise ValueError(f"Duplicate oracle key {key} from {path}")
            oracle[key] = row
    for path in sorted(split_dir.glob("*/model_rows.jsonl")):
        for row in _read_jsonl(path):
            key = (str(row["job_id"]), int(row["row_idx"]), int(row["tau"]))
            if key in baseline:
                raise ValueError(f"Duplicate baseline key {key} from {path}")
            baseline[key] = row
    if len(oracle) != 23 * 3 * 9:
        raise ValueError(f"Expected 621 oracle rows for {split}, got {len(oracle)}")
    if len(baseline) != 23 * 3 * 15:
        raise ValueError(f"Expected 1035 model rows for {split}, got {len(baseline)}")
    return oracle, baseline


def truncate_history(history: Dict[str, torch.Tensor], history_length: int) -> Dict[str, torch.Tensor]:
    """Keep the most recent ``history_length`` active timesteps."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if "prev_treatments" not in history:
        raise KeyError("prev_treatments")
    total = int(history["prev_treatments"].shape[1])
    length = min(int(history_length), total)
    out: Dict[str, torch.Tensor] = {}
    for key, value in history.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == total:
            out[key] = value[:, total - length :].clone()
        elif torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value
    out["sequence_lengths"] = torch.as_tensor([length], dtype=torch.long)
    if "active_entries" in out:
        out["active_entries"] = torch.ones_like(out["active_entries"])
    return out


def occlude_recency_block(
    history: Dict[str, torch.Tensor],
    recency_start: int,
    recency_end: int,
    *,
    keys: Iterable[str] = OCCLUSION_KEYS,
) -> Dict[str, torch.Tensor]:
    """Replace a recency block by the county-specific mean outside that block.

    Recency one is the final observed timestep.  Static covariates, metadata,
    active masks, and the planner's explicit previous-action channel are left
    untouched.  The replacement removes within-block dynamics while preserving
    each county's feature scale.
    """
    if recency_start < 1 or recency_end < recency_start:
        raise ValueError((recency_start, recency_end))
    total = int(history["prev_treatments"].shape[1])
    start = max(0, total - int(recency_end))
    end = min(total, total - int(recency_start) + 1)
    if end <= start:
        raise ValueError(
            f"Occlusion block {recency_start}..{recency_end} is outside history length {total}"
        )
    out = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in history.items()
    }
    for key in keys:
        value = out.get(key)
        if not torch.is_tensor(value) or value.dim() < 3 or value.shape[1] != total:
            continue
        keep = torch.cat((value[:, :start], value[:, end:]), dim=1)
        if keep.shape[1] == 0:
            raise ValueError(f"Cannot occlude the entire history for key={key}")
        baseline = keep.mean(dim=1, keepdim=True)
        value[:, start:end] = baseline.expand(-1, end - start, -1)
    return out


def centered_relative_change(reference: Sequence[float], perturbed: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    alt = np.asarray(perturbed, dtype=np.float64)
    ref = ref - ref.mean()
    alt = alt - alt.mean()
    return float(np.linalg.norm(alt - ref) / max(np.linalg.norm(ref), 1e-12))


@torch.no_grad()
def _encode_history(inference_model, history: Dict[str, torch.Tensor], device: str) -> dict:
    h_work = align_h_t_static_to_history(_to_device_history(history, device))
    z, _, _ = inference_model.ct_hidden_history(h_work)
    a_prev = gather_last_valid(h_work["current_treatments"], h_work.get("active_entries"))
    return {"z": z, "a_prev": a_prev}


@torch.no_grad()
def _score_encoded_grid(
    *,
    encoded: dict,
    planner,
    target_norm_final: float,
    tau: int,
    max_tau: float,
    grid_sim: np.ndarray,
    q_bc_penalty: float,
) -> dict:
    z = encoded["z"]
    a_prev = encoded["a_prev"]
    target = torch.full((z.size(0), 1), float(target_norm_final), dtype=z.dtype, device=z.device)
    max_action = float(planner.actor.max_action)
    a_prev_tanh = (2.0 * torch.clamp(a_prev, 0.0, 1.0) - 1.0) * max_action
    obs = _iql_augmented_state(planner, z, target, 0, int(tau), float(max_tau), a_prev_tanh)
    policy_out = planner.actor(obs)
    actor_mean = policy_out.mean if isinstance(policy_out, Distribution) else policy_out
    grid_tanh = torch.as_tensor(
        (2.0 * np.asarray(grid_sim, dtype=np.float32) - 1.0) * max_action,
        dtype=obs.dtype,
        device=obs.device,
    )
    obs_rep = obs.expand(grid_tanh.size(0), obs.size(-1))
    q_values = planner.qf(obs_rep, grid_tanh).reshape(-1)
    dist_sq = (grid_tanh / max_action - actor_mean.expand_as(grid_tanh)).pow(2).sum(dim=-1)
    regularized = q_values - q_values.new_tensor(float(q_bc_penalty)) * dist_sq
    return {
        "q_values": q_values.detach().cpu().double().tolist(),
        "regularized_scores": regularized.detach().cpu().double().tolist(),
        "q_choice": int(torch.argmax(q_values).item()),
        "regularized_choice": int(torch.argmax(regularized).item()),
        "z": z.detach().cpu().double().reshape(-1).tolist(),
    }


def _candidate_rows(oracle: Dict[tuple, dict], row_idx: int, tau: int) -> List[dict]:
    return [oracle[(int(row_idx), int(tau), index)] for index in range(9)]


def _argmin(rows: Sequence[dict], field: str, indices: Sequence[int]) -> int:
    return min(indices, key=lambda index: (float(rows[index][field]), int(index)))


def _choice_metrics(rows: Sequence[dict], choice: int, support_indices: Sequence[int]) -> dict:
    all_indices = list(range(len(rows)))
    oracle_choice = _argmin(rows, "training_return_loss", all_indices)
    support_choice = _argmin(rows, "training_return_loss", support_indices)
    return {
        "training_return_regret": float(
            rows[choice]["training_return_loss"] - rows[oracle_choice]["training_return_loss"]
        ),
        "terminal_regret_per_10k": float(
            rows[choice]["terminal_error_per_10k"]
            - min(float(row["terminal_error_per_10k"]) for row in rows)
        ),
        "trajectory_regret_per_10k": float(
            rows[choice]["trajectory_rmse_per_10k"]
            - min(float(row["trajectory_rmse_per_10k"]) for row in rows)
        ),
        "oracle_action_index": int(oracle_choice),
        "in_support_oracle_action_index": int(support_choice),
    }


def _best_support_choice(scores: Sequence[float], support_indices: Sequence[int]) -> int:
    return max(support_indices, key=lambda index: (float(scores[index]), -int(index)))


def _target_norm(candidate_rows: Sequence[dict], scaling_params) -> float:
    means, stds = _output_mean_std(scaling_params)
    target = float(candidate_rows[0]["factual_target_per_10k"])
    return float((target - float(means.reshape(-1)[0])) / float(stds.reshape(-1)[0]))


def _load_existing(path: Path, fields: Sequence[str]) -> Tuple[List[dict], set]:
    rows = []
    keys = set()
    if not path.exists():
        return rows, keys
    for line_no, row in enumerate(_read_jsonl(path), start=1):
        key = tuple(row[field] for field in fields)
        if key in keys:
            raise ValueError(f"Duplicate key at {path}:{line_no}: {key}")
        rows.append(row)
        keys.add(key)
    return rows, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--oracle-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--history-lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--taus", nargs="+", type=int, default=[7, 14, 21])
    parser.add_argument("--decision-day", type=int, default=161)
    parser.add_argument("--q-bc-penalty", type=float, default=1.0)
    parser.add_argument("--model-device", required=True)
    parser.add_argument("--epi-root", required=True)
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--cache-version", required=True)
    parser.add_argument("--dataset-seed", type=int, default=100)
    parser.add_argument("--outcome-transform", default="per10k_cases_zscore")
    parser.add_argument("--row-indices", nargs="+", type=int, default=None)
    parser.add_argument("--job-indices", nargs="+", type=int, default=None)
    parser.add_argument("--skip-occlusion", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not str(args.model_device).startswith("cuda"):
        raise RuntimeError("The formal temporal context probe requires a CUDA model device")
    lengths = sorted({int(value) for value in args.history_lengths})
    taus = sorted({int(value) for value in args.taus})
    if lengths[-1] != int(args.decision_day):
        raise ValueError("history-lengths must include decision-day as the full-context reference")

    source_manifest = Path(args.source_manifest).expanduser().resolve()
    oracle_run_dir = Path(args.oracle_run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    shard_dir = out_dir / args.split / "shards" / args.worker_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    all_jobs = _normalize_jobs(
        source_manifest,
        methods=args.methods,
        selector="explicit_grid_q_bc",
        q_bc_penalty=float(args.q_bc_penalty),
    )
    if args.job_indices is None:
        jobs = all_jobs
    else:
        jobs = [all_jobs[int(index)] for index in args.job_indices]
    if not jobs:
        raise ValueError("No jobs selected")

    base_cfg = OmegaConf.load(all_jobs[0]["config"])
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.dataset.epi_root = args.epi_root
    base_cfg.dataset.processed_data_dir = args.processed_data_dir
    base_cfg.dataset.cache_version = args.cache_version
    base_cfg.dataset.seed = int(args.dataset_seed)
    base_cfg.dataset.outcome_transform = args.outcome_transform
    base_cfg.dataset.device = "cpu"
    base_cfg.dataset.generate_if_missing = False
    base_cfg.dataset.force_regenerate = False
    base_cfg.exp.device = args.model_device
    expected_signature = _dataset_signature(base_cfg)

    set_seed(int(base_cfg.exp.seed))
    dataset_collection = instantiate(base_cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(base_cfg.dataset.static_size) > 0 and len(
        dataset_collection.train_f.data["static_features"].shape
    ) == 2:
        dataset_collection = repeat_static(dataset_collection)
    fold = dataset_collection.val_f if args.split == "val" else dataset_collection.test_f
    data = fold.data
    n_rows = int(np.asarray(data["sequence_lengths"]).shape[0])
    row_indices = list(range(n_rows)) if args.row_indices is None else [int(x) for x in args.row_indices]
    if any(index < 0 or index >= n_rows for index in row_indices):
        raise ValueError(f"row indices must be in 0..{n_rows - 1}: {row_indices}")

    oracle, source_baseline = _load_oracle_split(oracle_run_dir, args.split)
    support_indices = sorted({
        int(index)
        for (_row, _tau, index), row in oracle.items()
        if bool(row["in_observed_support"])
    })
    if support_indices != [0, 2, 6, 8]:
        raise ValueError(f"Unexpected observed action support: {support_indices}")

    grid = action_grid([0.0, 0.5, 1.0])
    protocol_family = {
        "schema": "epi_abm_temporal_context_probe_v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "oracle_run_dir": str(oracle_run_dir),
        "methods": list(args.methods),
        "history_lengths": lengths,
        "taus": taus,
        "decision_day": int(args.decision_day),
        "action_grid": grid.tolist(),
        "q_bc_penalty": float(args.q_bc_penalty),
        "occlusion_blocks": [list(block) for block in DEFAULT_OCCLUSION_BLOCKS],
        "occlusion_keys": list(OCCLUSION_KEYS),
        "occlusion_replacement": "county_channel_mean_outside_block",
        "oracle_reused_without_abm_rerun": True,
        "model_device_backend": "cuda",
        "dataset_signature": expected_signature,
    }
    protocol = {
        **protocol_family,
        "split": args.split,
        "worker_id": args.worker_id,
        "model_device": args.model_device,
        "row_indices": row_indices,
        "job_ids": [job["id"] for job in jobs],
        "skip_occlusion": bool(args.skip_occlusion),
    }
    family_hash = _protocol_hash(protocol_family)
    protocol_hash = _protocol_hash(protocol)
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("protocol_hash") != protocol_hash:
            raise ValueError(f"Resume protocol mismatch: {manifest_path}")
    manifest_path.write_text(json.dumps({
        **protocol,
        "protocol_family_hash": family_hash,
        "protocol_hash": protocol_hash,
        "jobs": jobs,
        "started_at": time.time(),
    }, indent=2, sort_keys=True) + "\n")

    context_path = shard_dir / "context_rows.jsonl"
    occlusion_path = shard_dir / "occlusion_rows.jsonl"
    context_rows, context_done = _load_existing(
        context_path, ("split", "job_id", "row_idx", "tau", "history_length")
    )
    occlusion_rows, occlusion_done = _load_existing(
        occlusion_path, ("split", "job_id", "row_idx", "tau", "block")
    )
    max_tau = float(OmegaConf.select(base_cfg, "exp.max_tau", default=max(taus)))
    parity_max_abs = 0.0

    for job in jobs:
        print(json.dumps({"event": "context_job_start", "split": args.split, "job_id": job["id"]}), flush=True)
        cfg, models = _load_job_models(job, args.model_device)
        try:
            label, (inference_model, planner) = next(iter(models.items()))
            for row_idx in row_indices:
                history, _targets, decision_day = _slice_county_window(
                    data,
                    row_idx,
                    max(taus),
                    window_mode="fixed-start",
                    decision_day=int(args.decision_day),
                )
                county = _county_id(data, row_idx)
                full_scores: Dict[int, dict] = {}
                with context_path.open("a") as handle:
                    for history_length in lengths:
                        needed = [
                            tau for tau in taus
                            if (args.split, job["id"], row_idx, tau, history_length) not in context_done
                        ]
                        if not needed and history_length != int(args.decision_day):
                            continue
                        context = truncate_history(history, history_length)
                        encoded = _encode_history(inference_model, context, args.model_device)
                        for tau in taus:
                            candidates = _candidate_rows(oracle, row_idx, tau)
                            scores = _score_encoded_grid(
                                encoded=encoded,
                                planner=planner,
                                target_norm_final=_target_norm(candidates, dataset_collection.train_scaling_params),
                                tau=tau,
                                max_tau=max_tau,
                                grid_sim=grid,
                                q_bc_penalty=float(args.q_bc_penalty),
                            )
                            if history_length == int(args.decision_day):
                                full_scores[tau] = scores
                                source = source_baseline[(job["id"], row_idx, tau)]
                                q_diff = float(np.max(np.abs(
                                    np.asarray(scores["q_values"]) - np.asarray(source["q_values"])
                                )))
                                reg_diff = float(np.max(np.abs(
                                    np.asarray(scores["regularized_scores"])
                                    - np.asarray(source["regularized_scores"])
                                )))
                                parity_max_abs = max(parity_max_abs, q_diff, reg_diff)
                                if max(q_diff, reg_diff) > 1e-4:
                                    raise AssertionError(
                                        f"Full-context parity failed for {job['id']}/{county}/tau{tau}: "
                                        f"q={q_diff} regularized={reg_diff}"
                                    )
                            key = (args.split, job["id"], row_idx, tau, history_length)
                            if key in context_done:
                                continue
                            regularized_choice = int(scores["regularized_choice"])
                            q_choice = int(scores["q_choice"])
                            support_choice = _best_support_choice(
                                scores["regularized_scores"], support_indices
                            )
                            support_q_choice = _best_support_choice(scores["q_values"], support_indices)
                            row = {
                                "split": args.split,
                                "job_id": job["id"],
                                "method": job["method"],
                                "seed": int(job["seed"]),
                                "checkpoint_label": label,
                                "row_idx": int(row_idx),
                                "county": county,
                                "tau": int(tau),
                                "decision_day": int(decision_day),
                                "history_length": int(history_length),
                                "regularized_choice": regularized_choice,
                                "q_choice": q_choice,
                                "in_support_regularized_choice": support_choice,
                                "in_support_q_choice": support_q_choice,
                                **_choice_metrics(candidates, regularized_choice, support_indices),
                                "q_training_return_regret": _choice_metrics(
                                    candidates, q_choice, support_indices
                                )["training_return_regret"],
                                "in_support_training_return_regret": float(
                                    candidates[support_choice]["training_return_loss"]
                                    - candidates[_argmin(
                                        candidates, "training_return_loss", support_indices
                                    )]["training_return_loss"]
                                ),
                                "in_support_q_training_return_regret": float(
                                    candidates[support_q_choice]["training_return_loss"]
                                    - candidates[_argmin(
                                        candidates, "training_return_loss", support_indices
                                    )]["training_return_loss"]
                                ),
                                "q_values": scores["q_values"],
                                "regularized_scores": scores["regularized_scores"],
                                "z": scores["z"],
                                "worker_id": args.worker_id,
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                            context_rows.append(row)
                            context_done.add(key)

                if args.skip_occlusion:
                    continue
                if len(full_scores) != len(taus):
                    raise RuntimeError(f"Missing full-context scores for {job['id']}/{county}")
                with occlusion_path.open("a") as handle:
                    for block, start, end in DEFAULT_OCCLUSION_BLOCKS:
                        if all(
                            (args.split, job["id"], row_idx, tau, block) in occlusion_done
                            for tau in taus
                        ):
                            continue
                        occluded = occlude_recency_block(history, start, end)
                        encoded = _encode_history(inference_model, occluded, args.model_device)
                        for tau in taus:
                            key = (args.split, job["id"], row_idx, tau, block)
                            if key in occlusion_done:
                                continue
                            candidates = _candidate_rows(oracle, row_idx, tau)
                            scores = _score_encoded_grid(
                                encoded=encoded,
                                planner=planner,
                                target_norm_final=_target_norm(candidates, dataset_collection.train_scaling_params),
                                tau=tau,
                                max_tau=max_tau,
                                grid_sim=grid,
                                q_bc_penalty=float(args.q_bc_penalty),
                            )
                            baseline = full_scores[tau]
                            choice = int(scores["regularized_choice"])
                            base_choice = int(baseline["regularized_choice"])
                            metrics = _choice_metrics(candidates, choice, support_indices)
                            base_metrics = _choice_metrics(candidates, base_choice, support_indices)
                            row = {
                                "split": args.split,
                                "job_id": job["id"],
                                "method": job["method"],
                                "seed": int(job["seed"]),
                                "row_idx": int(row_idx),
                                "county": county,
                                "tau": int(tau),
                                "block": block,
                                "recency_start": int(start),
                                "recency_end": int(end),
                                "regularized_choice": choice,
                                "baseline_regularized_choice": base_choice,
                                "choice_changed": choice != base_choice,
                                "q_choice_changed": int(scores["q_choice"]) != int(baseline["q_choice"]),
                                "q_centered_relative_change": centered_relative_change(
                                    baseline["q_values"], scores["q_values"]
                                ),
                                "regularized_centered_relative_change": centered_relative_change(
                                    baseline["regularized_scores"], scores["regularized_scores"]
                                ),
                                "z_relative_change": float(
                                    np.linalg.norm(np.asarray(scores["z"]) - np.asarray(baseline["z"]))
                                    / max(np.linalg.norm(np.asarray(baseline["z"])), 1e-12)
                                ),
                                "training_return_regret": metrics["training_return_regret"],
                                "baseline_training_return_regret": base_metrics["training_return_regret"],
                                "training_return_regret_delta": float(
                                    metrics["training_return_regret"]
                                    - base_metrics["training_return_regret"]
                                ),
                                "worker_id": args.worker_id,
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                            occlusion_rows.append(row)
                            occlusion_done.add(key)
        finally:
            del models
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps({"event": "context_job_complete", "split": args.split, "job_id": job["id"]}), flush=True)

    expected_context = len(jobs) * len(row_indices) * len(taus) * len(lengths)
    expected_occlusion = 0 if args.skip_occlusion else (
        len(jobs) * len(row_indices) * len(taus) * len(DEFAULT_OCCLUSION_BLOCKS)
    )
    summary = {
        "schema": "epi_abm_temporal_context_probe_shard_v1",
        "split": args.split,
        "worker_id": args.worker_id,
        "protocol_family_hash": family_hash,
        "protocol_hash": protocol_hash,
        "context_rows": len(context_rows),
        "expected_context_rows": expected_context,
        "occlusion_rows": len(occlusion_rows),
        "expected_occlusion_rows": expected_occlusion,
        "full_context_parity_max_abs": parity_max_abs,
        "complete": len(context_rows) == expected_context and len(occlusion_rows) == expected_occlusion,
        "finished_at": time.time(),
    }
    (shard_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["complete"]:
        raise RuntimeError(f"Incomplete temporal context probe: {summary}")
    print(json.dumps({"event": "context_worker_done", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
