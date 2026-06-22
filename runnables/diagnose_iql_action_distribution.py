"""
Diagnose IQL planned actions against factual target actions under the eval rollout.

This script mirrors eval_iql_planner.py's autoregressive action generation, but
records action-space statistics instead of outcome RMSE.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Distribution

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runnables.eval_iql_planner import (  # noqa: E402
    _extend_h_work_after_one_step,
    _iql_augmented_state,
    _policy_to_sim_interval_torch,
    _sim_actions_to_tanh_batch,
)
from src.data.cip_dataset import CIPDataset, get_dataloader  # noqa: E402
from src.data.iql_dataset_builder import align_h_t_static_to_history  # noqa: E402
from src.models.inference_model import InferenceModel  # noqa: E402
from src.models.sequence_utils import gather_last_valid  # noqa: E402
from src.utils.em_ckpt import load_em_for_eval  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _resolve_eval_tau_list(args: DictConfig) -> List[int]:
    raw = OmegaConf.select(args, "exp.iql_eval_tau_list", default=None)
    if raw is not None:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("["):
                raw = raw.strip("[]").split(",")
            else:
                raw = raw.split(",")
        taus = [int(t) for t in list(raw) if str(t).strip()]
        if taus:
            return taus
    return [int(args.exp.tau)]


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    set_seed(int(args.exp.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    if bool(OmegaConf.select(args, "exp.test", default=False)):
        data = dataset_collection.test_f.data
        fold = dataset_collection.test_f
        split_name = "test"
    else:
        data = dataset_collection.val_f.data
        fold = dataset_collection.val_f
        split_name = "val"

    em_eval_ckpt = str(OmegaConf.select(args, "exp.em_eval_ckpt", default="")).strip()
    if not em_eval_ckpt:
        raise ValueError("Set exp.em_eval_ckpt to an EM checkpoint.")
    em_path = Path(em_eval_ckpt)
    if not em_path.is_absolute():
        em_path = original_cwd / em_path
    if not em_path.is_file():
        raise FileNotFoundError(f"EM checkpoint not found: {em_path}")

    inference_model = InferenceModel(args).to(device)
    planner = load_em_for_eval(inference_model, str(em_path), device)
    inference_model.eval()
    max_action = float(planner.cfg.max_action)
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    mean_ser, std_ser = dataset_collection.train_scaling_params
    tau_list = _resolve_eval_tau_list(args)
    batch_size = int(OmegaConf.select(args, "exp.batch_size_val", default=128))
    max_batches = OmegaConf.select(args, "exp.action_diag_max_batches", default=None)
    max_batches = None if max_batches is None else int(max_batches)
    original_exp_tau = int(OmegaConf.select(args, "exp.tau", default=max(tau_list)))

    results = {
        "checkpoint": str(em_path),
        "split": split_name,
        "seed": int(args.exp.seed),
        "max_action": max_action,
        "taus": {},
    }

    for tau in tau_list:
        args.exp.tau = int(tau)
        try:
            dataloader = get_dataloader(CIPDataset(data, args, train=False), batch_size=batch_size, shuffle=False)
        finally:
            args.exp.tau = original_exp_tau

        planned_chunks = []
        factual_chunks = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                H_t, targets = batch
                H_t = align_h_t_static_to_history(H_t)
                for key in H_t:
                    H_t[key] = H_t[key].to(device)
                for key in targets:
                    targets[key] = targets[key].to(device)

                eval_target = targets["outputs"][:, -1, :]
                H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                a_prev_sim = gather_last_valid(
                    H_work["current_treatments"], H_work.get("active_entries")
                ).clone()
                planned = []
                for step in range(tau):
                    H_work = align_h_t_static_to_history(H_work)
                    z, _, _ = inference_model.ct_hidden_history(H_work)
                    a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                    obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
                    policy_out = planner.actor(obs)
                    actor_max = planner.actor.max_action
                    if isinstance(policy_out, Distribution):
                        a_raw = torch.clamp(actor_max * policy_out.mean, -actor_max, actor_max)
                    else:
                        a_raw = torch.clamp(policy_out * actor_max, -actor_max, actor_max)
                    a_sim = _policy_to_sim_interval_torch(a_raw, max_action)
                    planned.append(a_sim)
                    y_np = fold.simulate_output_after_actions(
                        H_work,
                        a_sim.unsqueeze(1),
                        dataset_collection.train_scaling_params,
                    )
                    y_norm = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                    _extend_h_work_after_one_step(H_work, a_sim, y_norm, mean_ser, std_ser, torch.device(device))
                    a_prev_sim = a_sim

                planned_seq = torch.stack(planned, dim=1).detach().cpu().numpy()
                factual_seq = targets["current_treatments"].detach().cpu().numpy()
                planned_chunks.append(planned_seq)
                factual_chunks.append(factual_seq)

        planned_arr = np.concatenate(planned_chunks, axis=0)
        factual_arr = np.concatenate(factual_chunks, axis=0)
        diff = planned_arr - factual_arr
        tau_result = {
            "action_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "action_mae": float(np.mean(np.abs(diff))),
            "planned": _stats(planned_arr),
            "factual": _stats(factual_arr),
            "diff": _stats(diff),
            "steps": [],
        }
        for step in range(tau):
            step_diff = diff[:, step, :]
            tau_result["steps"].append(
                {
                    "step": step + 1,
                    "rmse": float(np.sqrt(np.mean(step_diff ** 2))),
                    "mae": float(np.mean(np.abs(step_diff))),
                    "planned": _stats(planned_arr[:, step, :]),
                    "factual": _stats(factual_arr[:, step, :]),
                    "diff": _stats(step_diff),
                }
            )
        results["taus"][str(tau)] = tau_result

    out_path = str(OmegaConf.select(args, "exp.action_diag_out", default=""))
    text = json.dumps(results, indent=2, sort_keys=True)
    if out_path:
        p = Path(out_path)
        if not p.is_absolute():
            p = original_cwd / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        logger.info("Wrote action diagnostics to %s", p)
    print(text)


if __name__ == "__main__":
    main()
