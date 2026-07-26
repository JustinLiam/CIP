"""Measure deployed CRIPO parameters and synchronized closed-loop latency."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics
from src.models.inference_model import InferenceModel
from src.utils.em_ckpt import load_em_for_eval
from src.utils.stable_iql_em_defaults import stable_select
from src.utils.utils import repeat_static, set_seed, to_float


OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _unique_numel(*modules) -> int:
    seen = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            pointer = parameter.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += parameter.numel()
    return int(total)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs/")
def main(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)
    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args.exp.processed_data_dir = os.path.join(
        str(original_cwd), str(args.exp.processed_data_dir)
    )

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if int(args.dataset.static_size) > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    fold = dataset_collection.test_f if bool(args.exp.test) else dataset_collection.val_f
    inference_model = InferenceModel(args).to(device)
    checkpoint = Path(str(args.exp.em_eval_ckpt))
    if not checkpoint.is_absolute():
        checkpoint = original_cwd / checkpoint
    planner = load_em_for_eval(inference_model, str(checkpoint), device)
    inference_model.eval()
    planner.actor.eval()

    params_deploy = _unique_numel(
        inference_model.ct_history_encoder,
        inference_model.projection_head,
        planner.actor,
    )
    batch_size = int(OmegaConf.select(args, "exp.batch_size_val", default=128))
    max_tau = float(stable_select(args, "exp.max_tau"))
    autoregressive = bool(stable_select(args, "exp.iql_eval_autoregressive"))
    output_dim = int(args.dataset.output_size)

    def run_tau(tau: int) -> dict:
        _sync()
        started = time.perf_counter_ns()
        metrics = aggregate_iql_planner_metrics(
            planner,
            inference_model,
            dataset_collection,
            fold,
            args,
            device=device,
            tau=tau,
            max_tau=max_tau,
            autoregressive_eval=autoregressive,
            val_batch_size=batch_size,
            log_batches=False,
            return_series=True,
            include_factual_traj_rmse=False,
            sample_seed=int(args.exp.seed),
        )
        _sync()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        n_values = int(np.asarray(metrics["true_y_norm"]).size)
        episodes = max(1, n_values // max(output_dim, 1))
        return {
            "tau": tau,
            "episodes": episodes,
            "elapsed_ms": elapsed_ms,
            "episode_ms": elapsed_ms / episodes,
            "decision_ms": elapsed_ms / (episodes * tau),
        }

    # Warm CUDA kernels and the simulator path outside the reported timing.
    run_tau(1)
    timing = [run_tau(tau) for tau in (1, 6, 12)]
    payload = {
        "dataset": str(args.dataset.name),
        "seed": int(args.exp.seed),
        "checkpoint": str(checkpoint),
        "device": device,
        "batch_size": batch_size,
        "params_deploy": params_deploy,
        "timing": timing,
    }
    output = Path(str(args.exp.efficiency_output))
    if not output.is_absolute():
        output = original_cwd / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
