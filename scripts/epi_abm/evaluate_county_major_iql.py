"""Evaluate many EpiABM method/seed jobs while keeping one county resident."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.epi_abm.evaluate_county_last_window_iql import (  # noqa: E402
    _county_id,
    _make_eval_target_norm,
    _metric_row,
    _parse_label_path,
    _population_from_row,
    _release_county_cache,
    _rollout_factual,
    _rollout_policy,
    _scale_outputs_np,
    _slice_county_window,
    _summarize,
)
from src.models.inference_model import InferenceModel  # noqa: E402
from src.utils.em_ckpt import load_em_for_eval  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402


def _absolute(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _protocol_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            row["split"],
            int(row["row_idx"]),
            int(row["tau"]),
            row["label"],
        )
        if key in seen:
            raise ValueError(f"Duplicate result key in {path}:{line_number}: {key}")
        seen.add(key)
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_target_rows(path: Path) -> Dict[tuple, dict]:
    rows: Dict[tuple, dict] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["split"]),
            int(row["seed"]),
            int(row["row_idx"]),
            int(row["tau"]),
            str(row["county"]),
            int(row["decision_day"]),
        )
        if key in rows:
            previous = rows[key]
            for field in ("target_final_model_units", "target_final_per_10k"):
                if abs(float(previous[field]) - float(row[field])) > 1.0e-10:
                    raise ValueError(
                        f"Duplicate external target mismatch in {path}:"
                        f"{line_number}: {key} {field}"
                    )
            continue
        rows[key] = row
    return rows


def _external_target_norm(
    external_targets: Dict[tuple, dict],
    *,
    split: str,
    seed: int,
    row_idx: int,
    tau: int,
    county: str,
    decision_day: int,
    target_scale: float,
    scaling_params,
) -> np.ndarray:
    key = (
        str(split),
        int(seed),
        int(row_idx),
        int(tau),
        str(county),
        int(decision_day),
    )
    row = external_targets.get(key)
    if row is None:
        raise KeyError(f"Missing external target for key={key}")
    target_model_units = np.asarray(
        [[float(row["target_final_model_units"]) * float(target_scale)]],
        dtype=np.float64,
    )
    return _scale_outputs_np(target_model_units, scaling_params).astype(np.float32)


def _load_job_models(job: dict, device: str):
    config_path = Path(job["config"])
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.exp.device = device
    cfg.exp.iql_eval_action_selector = job["selector"]
    cfg.exp.iql_eval_candidate_actions = int(job["candidate_actions"])
    cfg.exp.iql_eval_q_bc_penalty = float(job["q_bc_penalty"])
    cfg.exp.iql_eval_candidate_noise_std = float(job["candidate_noise_std"])
    set_seed(int(cfg.exp.seed))

    models = {}
    ckpt_paths = _parse_label_path(
        [f"{label}={path}" for label, path in job["ckpts"].items()]
    )
    for label, ckpt_path in ckpt_paths.items():
        inference_model = InferenceModel(cfg).to(device)
        planner = load_em_for_eval(inference_model, str(ckpt_path), device)
        inference_model.eval()
        planner.actor.eval()
        models[label] = (inference_model, planner)
    return cfg, models


def _normalize_jobs(raw_jobs: List[dict], manifest_dir: Path, args) -> List[dict]:
    jobs = []
    seen_ids = set()
    seen_outputs = set()
    for raw in raw_jobs:
        job_id = str(raw["id"])
        if job_id in seen_ids:
            raise ValueError(f"Duplicate job id: {job_id}")
        seen_ids.add(job_id)
        config = _absolute(raw["config"], manifest_dir)
        out_dir = _absolute(raw["out_dir"], manifest_dir)
        if out_dir in seen_outputs:
            raise ValueError(f"Multiple jobs share out_dir={out_dir}")
        seen_outputs.add(out_dir)
        ckpts = {
            str(label): str(_absolute(path, manifest_dir))
            for label, path in raw["ckpts"].items()
        }
        if not ckpts:
            raise ValueError(f"Job {job_id} has no checkpoints")
        missing = [path for path in [config, *map(Path, ckpts.values())] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Job {job_id} is missing files: {missing}")
        jobs.append({
            "id": job_id,
            "method": str(raw.get("method", "cripo")),
            "seed": int(raw["seed"]),
            "config": str(config),
            "out_dir": str(out_dir),
            "ckpts": ckpts,
            "selector": str(raw.get("selector", args.selector)),
            "candidate_actions": int(raw.get("candidate_actions", args.candidate_actions)),
            "q_bc_penalty": float(raw.get("q_bc_penalty", args.q_bc_penalty)),
            "candidate_noise_std": float(
                raw.get("candidate_noise_std", args.candidate_noise_std)
            ),
            "eval_seed": int(raw.get("eval_seed", args.eval_seed)),
        })
    return jobs


def _dataset_signature(cfg) -> str:
    data = OmegaConf.to_container(cfg.dataset, resolve=True)
    for key in ("device", "epi_root"):
        data.pop(key, None)
    return _protocol_hash(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--taus", nargs="+", type=int, required=True)
    parser.add_argument("--decision-day", type=int, required=True)
    parser.add_argument("--window-mode", choices=["fixed-start"], default="fixed-start")
    parser.add_argument("--target-mode", default="factual_final")
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument(
        "--external-target-file",
        default=None,
        help=(
            "Immutable GPU factual-replay target JSONL. Values are looked up by "
            "split/seed/row/tau/county/decision-day and multiplied by --target-scale."
        ),
    )
    parser.add_argument("--selector", default="q_sample")
    parser.add_argument("--candidate-actions", type=int, default=64)
    parser.add_argument("--q-bc-penalty", type=float, default=1.0)
    parser.add_argument("--candidate-noise-std", type=float, default=0.25)
    parser.add_argument("--eval-seed", type=int, default=20260708)
    parser.add_argument("--model-device", required=True)
    parser.add_argument("--abm-device", required=True)
    parser.add_argument("--epi-root", required=True)
    parser.add_argument("--processed-data-dir", default=None)
    parser.add_argument("--cache-version", default=None)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--outcome-transform", default=None)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--row-indices", nargs="+", type=int, default=None)
    parser.add_argument(
        "--no-persistent-abm-session",
        dest="persistent_abm_session",
        action="store_false",
        help="Use replay-from-snapshot for every policy day instead of a resident rollout.",
    )
    parser.set_defaults(persistent_abm_session=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    jobs = _normalize_jobs(manifest["jobs"], manifest_path.parent, args)
    if not jobs:
        raise ValueError("County-major manifest contains no jobs")

    base_cfg = OmegaConf.load(jobs[0]["config"])
    OmegaConf.set_struct(base_cfg, False)
    if args.processed_data_dir is not None:
        base_cfg.dataset.processed_data_dir = args.processed_data_dir
    if args.cache_version is not None:
        base_cfg.dataset.cache_version = args.cache_version
    if args.dataset_seed is not None:
        base_cfg.dataset.seed = int(args.dataset_seed)
    if args.outcome_transform is not None:
        base_cfg.dataset.outcome_transform = args.outcome_transform
    base_cfg.dataset.epi_root = args.epi_root
    base_cfg.dataset.device = args.abm_device
    base_cfg.dataset.generate_if_missing = False
    base_cfg.dataset.force_regenerate = False
    base_cfg.exp.device = args.model_device

    expected_dataset_signature = _dataset_signature(base_cfg)
    for job in jobs[1:]:
        cfg = OmegaConf.load(job["config"])
        OmegaConf.set_struct(cfg, False)
        if args.processed_data_dir is not None:
            cfg.dataset.processed_data_dir = args.processed_data_dir
        if args.cache_version is not None:
            cfg.dataset.cache_version = args.cache_version
        if args.dataset_seed is not None:
            cfg.dataset.seed = int(args.dataset_seed)
        if args.outcome_transform is not None:
            cfg.dataset.outcome_transform = args.outcome_transform
        if _dataset_signature(cfg) != expected_dataset_signature:
            raise ValueError(
                f"Job {job['id']} does not share the base dataset protocol"
            )

    set_seed(int(base_cfg.exp.seed))
    external_target_path = None
    external_target_sha256 = None
    external_targets = None
    if args.external_target_file is not None:
        external_target_path = Path(args.external_target_file).expanduser().resolve()
        if not external_target_path.is_file():
            raise FileNotFoundError(
                f"Missing external target file: {external_target_path}"
            )
        external_targets = _load_external_target_rows(external_target_path)
        external_target_sha256 = _sha256_file(external_target_path)

    dataset_collection = instantiate(base_cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if (
        int(base_cfg.dataset.static_size) > 0
        and len(dataset_collection.train_f.data["static_features"].shape) == 2
    ):
        dataset_collection = repeat_static(dataset_collection)
    dataset_collection.simulator_cache.configure_snapshot_days([args.decision_day])

    if args.split == "val":
        fold = dataset_collection.val_f
    else:
        fold = dataset_collection.test_f
    data = fold.data
    n_rows = int(np.asarray(data["sequence_lengths"]).shape[0])
    if args.row_indices is None:
        row_end = n_rows if args.row_end is None else min(n_rows, int(args.row_end))
        row_indices = list(range(max(0, int(args.row_start)), row_end))
    else:
        row_indices = [int(row_idx) for row_idx in args.row_indices]
    invalid = [row_idx for row_idx in row_indices if row_idx < 0 or row_idx >= n_rows]
    if invalid:
        raise ValueError(f"row indices outside 0..{n_rows - 1}: {invalid}")

    device = (
        args.model_device
        if torch.cuda.is_available() and args.model_device.startswith("cuda")
        else "cpu"
    )
    protocol_family = {
        "schema": "epi_abm_county_major_v3",
        "split": args.split,
        "taus": [int(tau) for tau in args.taus],
        "decision_day": int(args.decision_day),
        "window_mode": args.window_mode,
        "target_mode": args.target_mode,
        "target_scale": float(args.target_scale),
        "target_value": args.target_value,
        "target_reference": (
            "external_target_file" if external_targets is not None else "cached_target"
        ),
        "external_target_file": (
            str(external_target_path) if external_target_path is not None else None
        ),
        "external_target_sha256": external_target_sha256,
        "model_device_backend": device.split(":", 1)[0],
        "abm_device_backend": args.abm_device.split(":", 1)[0],
        "processed_data_dir": str(base_cfg.dataset.processed_data_dir),
        "cache_version": str(base_cfg.dataset.cache_version),
        "dataset_seed": int(base_cfg.dataset.seed),
        "outcome_transform": str(base_cfg.dataset.outcome_transform),
        "dataset_signature": expected_dataset_signature,
        "snapshot_days": [0, int(args.decision_day)],
        "persistent_abm_session": bool(args.persistent_abm_session),
    }
    protocol = {
        **protocol_family,
        "worker_id": args.worker_id,
        "row_indices": row_indices,
        "model_device": device,
        "abm_device": args.abm_device,
        "epi_root": str(Path(args.epi_root).resolve()),
    }
    protocol_family_hash = _protocol_hash(protocol_family)
    protocol_hash = _protocol_hash(protocol)

    states = {}
    for job in jobs:
        shard_dir = Path(job["out_dir"]) / "county_major_shards" / args.worker_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = shard_dir / "county_metrics.jsonl"
        summary_path = shard_dir / "summary.json"
        job_manifest_path = shard_dir / "manifest.json"
        existing = _load_existing_rows(metrics_path)
        existing_manifest = None
        if job_manifest_path.exists():
            existing_manifest = json.loads(job_manifest_path.read_text())
            if existing_manifest.get("protocol_hash") != protocol_hash:
                raise ValueError(
                    f"Resume protocol mismatch for {job['id']}: {job_manifest_path}"
                )
            if existing_manifest.get("ckpts") != job["ckpts"]:
                raise ValueError(
                    f"Resume checkpoint mismatch for {job['id']}: {job_manifest_path}"
                )
        job_manifest_path.write_text(json.dumps({
            **protocol,
            "protocol_family_hash": protocol_family_hash,
            "protocol_hash": protocol_hash,
            "job_id": job["id"],
            "config": job["config"],
            "ckpts": job["ckpts"],
            "selector": job["selector"],
            "candidate_actions": job["candidate_actions"],
            "q_bc_penalty": job["q_bc_penalty"],
            "candidate_noise_std": job["candidate_noise_std"],
            "eval_seed": job["eval_seed"],
            "started_at": (
                existing_manifest.get("started_at")
                if existing_manifest is not None
                else time.time()
            ),
            "updated_at": time.time(),
        }, indent=2, sort_keys=True) + "\n")
        states[job["id"]] = {
            "metrics_path": metrics_path,
            "summary_path": summary_path,
            "rows": existing,
            "done": {
                (
                    row["split"],
                    int(row["row_idx"]),
                    int(row["tau"]),
                    row["label"],
                )
                for row in existing
            },
        }

    action_hold_days = max(
        1, int(OmegaConf.select(base_cfg, "dataset.action_hold_days", default=1))
    )
    max_tau = float(OmegaConf.select(base_cfg, "exp.max_tau", default=max(args.taus)))

    for row_idx in row_indices:
        county = _county_id(data, row_idx)
        county_complete = all(
            all(
                (args.split, row_idx, int(tau), label)
                in states[job["id"]]["done"]
                for tau in args.taus
                for label in job["ckpts"]
            )
            for job in jobs
        )
        if county_complete:
            print(json.dumps({
                "event": "county_major_county_resume_skip",
                "split": args.split,
                "county": county,
                "row_idx": row_idx,
            }), flush=True)
            continue
        county_started = time.time()
        print(json.dumps({
            "event": "county_major_county_start",
            "split": args.split,
            "county": county,
            "row_idx": row_idx,
            "jobs": [job["id"] for job in jobs],
        }), flush=True)
        try:
            contexts = {}
            factual_started = time.time()
            for tau in args.taus:
                H, targets, decision_day = _slice_county_window(
                    data,
                    row_idx,
                    int(tau),
                    window_mode=args.window_mode,
                    decision_day=args.decision_day,
                )
                true_norm_daily = (
                    targets["outputs"].detach().cpu().numpy().astype(np.float32)
                )
                target_norm_final, _ = _make_eval_target_norm(
                    H,
                    targets,
                    dataset_collection.train_scaling_params,
                    target_mode=args.target_mode,
                    target_scale=args.target_scale,
                    target_value=args.target_value,
                )
                factual_norm_daily = _rollout_factual(
                    fold, dataset_collection, H, targets, int(tau)
                )
                contexts[int(tau)] = {
                    "H": H,
                    "true_norm_daily": true_norm_daily,
                    "target_norm_final": target_norm_final,
                    "factual_norm_daily": factual_norm_daily,
                    "decision_day": decision_day,
                    "population": _population_from_row(data, row_idx),
                }
            print(json.dumps({
                "event": "county_major_factual_ready",
                "county": county,
                "row_idx": row_idx,
                "elapsed_sec": round(time.time() - factual_started, 3),
            }), flush=True)

            for job in jobs:
                state = states[job["id"]]
                needed_labels = [
                    label
                    for label in job["ckpts"]
                    if any(
                        (args.split, row_idx, int(tau), label) not in state["done"]
                        for tau in args.taus
                    )
                ]
                if not needed_labels:
                    continue
                active_job = dict(job)
                active_job["ckpts"] = {
                    label: job["ckpts"][label] for label in needed_labels
                }
                job_started = time.time()
                cfg, models = _load_job_models(active_job, device)
                try:
                    with state["metrics_path"].open("a") as metrics_file:
                        for label, (inference_model, planner) in models.items():
                            for tau in args.taus:
                                key = (args.split, row_idx, int(tau), label)
                                if key in state["done"]:
                                    continue
                                cfg.exp.tau = int(tau)
                                context = contexts[int(tau)]
                                target_norm_final = context["target_norm_final"]
                                target_reference = "cached_target"
                                if external_targets is not None:
                                    target_norm_final = _external_target_norm(
                                        external_targets,
                                        split=args.split,
                                        seed=int(job["seed"]),
                                        row_idx=int(row_idx),
                                        tau=int(tau),
                                        county=county,
                                        decision_day=int(context["decision_day"]),
                                        target_scale=float(args.target_scale),
                                        scaling_params=(
                                            dataset_collection.train_scaling_params
                                        ),
                                    )
                                    target_reference = "external_target_file"
                                started = time.time()
                                pred_norm_daily, planned_actions = _rollout_policy(
                                    fold=fold,
                                    dataset_collection=dataset_collection,
                                    inference_model=inference_model,
                                    planner=planner,
                                    H=context["H"],
                                    eval_target_norm=target_norm_final,
                                    label=label,
                                    county=county,
                                    tau=int(tau),
                                    max_tau=max_tau,
                                    action_hold_days=action_hold_days,
                                    selector=job["selector"],
                                    candidate_actions=job["candidate_actions"],
                                    q_bc_penalty=job["q_bc_penalty"],
                                    candidate_noise_std=job["candidate_noise_std"],
                                    eval_seed=job["eval_seed"],
                                    device=device,
                                    persistent_abm_session=args.persistent_abm_session,
                                )
                                row = _metric_row(
                                    split=args.split,
                                    tau=int(tau),
                                    county=county,
                                    label=label,
                                    pred_norm_daily=pred_norm_daily,
                                    true_norm_daily=context["true_norm_daily"],
                                    factual_norm_daily=context["factual_norm_daily"],
                                    target_norm_final=target_norm_final,
                                    planned_actions=planned_actions,
                                    scaling_params=dataset_collection.train_scaling_params,
                                    population=context["population"],
                                )
                                row.update({
                                    "idx": int(row_idx),
                                    "row_idx": int(row_idx),
                                    "decision_day": int(context["decision_day"]),
                                    "window_mode": args.window_mode,
                                    "target_mode": args.target_mode,
                                    "target_scale": float(args.target_scale),
                                    "target_value": args.target_value,
                                    "target_reference": target_reference,
                                    "external_target_file": (
                                        str(external_target_path)
                                        if external_target_path is not None
                                        else None
                                    ),
                                    "external_target_sha256": external_target_sha256,
                                    "elapsed_sec": round(time.time() - started, 3),
                                    "job_id": job["id"],
                                    "worker_id": args.worker_id,
                                })
                                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                                metrics_file.flush()
                                state["rows"].append(row)
                                state["done"].add(key)
                                print(json.dumps(row, sort_keys=True), flush=True)
                    state["summary_path"].write_text(
                        json.dumps(
                            _summarize(state["rows"]), indent=2, sort_keys=True
                        ) + "\n"
                    )
                finally:
                    del models
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(json.dumps({
                    "event": "county_major_job_county_complete",
                    "job_id": job["id"],
                    "county": county,
                    "row_idx": row_idx,
                    "elapsed_sec": round(time.time() - job_started, 3),
                }), flush=True)
        finally:
            _release_county_cache(fold, county)
        print(json.dumps({
            "event": "county_major_county_complete",
            "split": args.split,
            "county": county,
            "row_idx": row_idx,
            "elapsed_sec": round(time.time() - county_started, 3),
        }), flush=True)

    for job in jobs:
        state = states[job["id"]]
        state["summary_path"].write_text(
            json.dumps(_summarize(state["rows"]), indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "event": "county_major_eval_done",
        "worker_id": args.worker_id,
        "jobs": {
            job["id"]: len(states[job["id"]]["rows"])
            for job in jobs
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
