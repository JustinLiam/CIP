"""
Diagnose action shift for CT+IQL EM checkpoints.

The script is diagnostic-only. It does not participate in training/evaluation
unless invoked directly. It reports validation/test rollout action summaries,
Q(s,a) grid preferences, simulator best-action proxy, and replay-level
advantage/weight/action correlations.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ct_transition_dataset import _covariate_stream_dim  # noqa: E402
from src.data.iql_raw_transition_dataset import IQLRawReplayBuffer, build_iql_raw_transitions  # noqa: E402
from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics  # noqa: E402
from src.models.ct_encoder_weight import CTEncoderWeightModel  # noqa: E402
from src.models.inference_model import InferenceModel  # noqa: E402
from src.planners.iql_planner import _cap_renormalize_weights  # noqa: E402
from src.utils.em_ckpt import load_em_ct_model, load_em_for_eval  # noqa: E402
from src.utils.stable_iql_em_defaults import stable_select  # noqa: E402
from src.utils.utils import repeat_static, set_seed, to_float  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _coerce_int_list(raw: Any, default: Optional[Iterable[int]] = None) -> List[int]:
    if raw is None:
        return list(default or [])
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return list(default or [])
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in raw]


def _resolve_tau_list(args: DictConfig) -> List[int]:
    raw = OmegaConf.select(args, "exp.action_diag_tau_list", default=None)
    if raw is None:
        raw = stable_select(args, "exp.iql_eval_tau_list")
    default_taus = _coerce_int_list(stable_select(args, "exp.iql_eval_tau_list"), default=[1, 2, 3, 4, 5, 6])
    taus = _coerce_int_list(raw, default=default_taus)
    return taus or default_taus


def _stats(x: np.ndarray) -> Dict[str, float]:
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    if spearman:
        x = _rankdata(x)
        y = _rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_bins(action_mean: np.ndarray, values: Dict[str, np.ndarray], n_bins: int = 5) -> List[Dict[str, Any]]:
    action_mean = np.asarray(action_mean, dtype=np.float64).reshape(-1)
    qs = np.quantile(action_mean, np.linspace(0.0, 1.0, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            mask = (action_mean >= lo) & (action_mean <= hi)
        else:
            mask = (action_mean >= lo) & (action_mean < hi)
        row: Dict[str, Any] = {
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "action_mean": float(action_mean[mask].mean()) if mask.any() else None,
        }
        for key, arr in values.items():
            arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            row[f"{key}_mean"] = float(arr[mask].mean()) if mask.any() else None
            row[f"{key}_std"] = float(arr[mask].std()) if mask.any() else None
        out.append(row)
    return out


def _policy_to_sim_np(action_policy: np.ndarray, max_action: float) -> np.ndarray:
    if max_action <= 0:
        return action_policy.astype(np.float32)
    return np.clip((action_policy + max_action) / (2.0 * max_action), 0.0, 1.0).astype(np.float32)


def _prepare_dataset(args: DictConfig):
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)
    return dataset_collection


def _resolve_ckpt(args: DictConfig, original_cwd: Path, seed: int) -> Path:
    raw = str(stable_select(args, "exp.em_eval_ckpt")).strip()
    if not raw:
        raise ValueError("Set exp.em_eval_ckpt to an EM checkpoint path or a template containing {seed}.")
    raw = raw.format(seed=seed)
    path = Path(raw)
    if not path.is_absolute():
        path = original_cwd / path
    if not path.is_file():
        raise FileNotFoundError(f"EM checkpoint not found for seed {seed}: {path}")
    return path


def _replay_diagnostics(args: DictConfig, dataset_collection, ckpt_path: Path, planner, device: str) -> Dict[str, Any]:
    max_action = float(planner.cfg.max_action)
    max_tau = float(stable_select(args, "exp.max_tau"))
    max_patients = OmegaConf.select(args, "exp.action_diag_replay_max_patients", default=256)
    max_patients = None if max_patients is None else int(max_patients)
    sample_n = int(OmegaConf.select(args, "exp.action_diag_replay_samples", default=4096))

    raw = build_iql_raw_transitions(
        dataset_collection.train_f.data,
        reward_type=str(stable_select(args, "exp.iql_reward_type")),
        max_patients=max_patients,
        max_action=max_action,
        dataset_actions_unit_interval=bool(stable_select(args, "exp.iql_dataset_actions_unit_interval")),
        max_tau=max_tau,
        reward_clip=float(stable_select(args, "exp.iql_reward_clip")),
        reward_scale=str(stable_select(args, "exp.iql_reward_scale")),
        reward_huber_delta=float(stable_select(args, "exp.iql_reward_huber_delta")),
        samples_per_transition=int(stable_select(args, "exp.em_her_samples_per_transition")),
        target_sampling=str(stable_select(args, "exp.iql_target_sampling")),
        target_horizons=stable_select(args, "exp.iql_target_horizons"),
        horizon_terminal_done=bool(stable_select(args, "exp.iql_horizon_terminal_done")),
        seed=int(args.exp.seed) + 9109,
    )
    if not raw:
        return {"n_transitions": 0}

    ds_dict = OmegaConf.to_container(args["dataset"], resolve=True)
    ct_model = CTEncoderWeightModel(args, _covariate_stream_dim(ds_dict)).to(device)
    load_em_ct_model(ct_model, str(ckpt_path), device)
    ct_model.eval()
    replay = IQLRawReplayBuffer(raw, device=device)
    batch = replay.sample(min(sample_n, replay.size))

    with torch.no_grad():
        z_t, a_t = ct_model.encode(batch.H_t)
        _, w_raw = ct_model.compute_weights(z_t, a_t, detach_z=True, uniform=False)
        w = _cap_renormalize_weights(w_raw.detach(), planner.cfg.weight_max)
        states = planner.build_state(z_t, batch.y_target, batch.delta_t_norm, batch.a_prev_tanh)
        q = planner.qf(states, batch.action)
        v = planner.vf(states)
        adv = q - v
        exp_adv = torch.exp(float(planner.cfg.beta) * adv).clamp(max=float(planner.cfg.adv_max))

    action_sim = _policy_to_sim_np(batch.action.detach().cpu().numpy(), max_action)
    action_mean = action_sim.mean(axis=1)
    arrays = {
        "adv": adv.detach().cpu().numpy(),
        "exp_adv": exp_adv.detach().cpu().numpy(),
        "w_raw": w_raw.detach().cpu().numpy(),
        "w": w.detach().cpu().numpy(),
        "reward": batch.reward.detach().cpu().numpy(),
    }
    correlations = {}
    for key, arr in arrays.items():
        correlations[f"action_{key}_pearson"] = _corr(action_mean, arr)
        correlations[f"action_{key}_spearman"] = _corr(action_mean, arr, spearman=True)

    return {
        "n_transitions": int(replay.size),
        "n_sampled": int(action_mean.shape[0]),
        "action": _stats(action_mean),
        "adv": _stats(arrays["adv"]),
        "exp_adv": _stats(arrays["exp_adv"]),
        "w_raw": _stats(arrays["w_raw"]),
        "w": _stats(arrays["w"]),
        "reward": _stats(arrays["reward"]),
        "correlations": correlations,
        "action_quantile_bins": _quantile_bins(action_mean, arrays),
    }


def _run_one_seed(args: DictConfig, seed: int, original_cwd: Path) -> Dict[str, Any]:
    args = copy.deepcopy(args)
    args.exp.seed = int(seed)
    set_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_collection = _prepare_dataset(args)
    ckpt_path = _resolve_ckpt(args, original_cwd, seed)

    inference_model = InferenceModel(args).to(device)
    planner = load_em_for_eval(inference_model, str(ckpt_path), device)
    inference_model.eval()
    planner.actor.eval()

    split_name = "test" if bool(OmegaConf.select(args, "exp.test", default=False)) else "val"
    fold = dataset_collection.test_f if split_name == "test" else dataset_collection.val_f
    tau_list = _resolve_tau_list(args)
    grid_points = int(OmegaConf.select(args, "exp.action_diag_grid_points", default=stable_select(args, "exp.iql_val_action_grid_points")))
    max_batches = OmegaConf.select(
        args,
        "exp.action_diag_max_batches",
        default=stable_select(args, "exp.iql_val_action_diag_max_batches"),
    )
    max_batches = None if max_batches is None else int(max_batches)
    val_bs = int(OmegaConf.select(args, "exp.batch_size_val", default=128))
    max_tau = float(stable_select(args, "exp.max_tau"))
    autoreg = bool(stable_select(args, "exp.iql_eval_autoregressive"))

    per_tau = {}
    for tau in tau_list:
        metrics = aggregate_iql_planner_metrics(
            planner,
            inference_model,
            dataset_collection,
            fold,
            args,
            device=device,
            tau=int(tau),
            max_tau=max_tau,
            autoregressive_eval=autoreg,
            val_batch_size=val_bs,
            worlds=("sim",),
            action_diagnostics=True,
            action_grid_points=grid_points,
            action_diag_max_batches=max_batches,
        )
        per_tau[str(tau)] = {
            "mae_uns": float(metrics["mae_uns"]),
            "rmse_uns": float(metrics["rmse_uns"]),
            "action_diagnostics": metrics.get("action_diagnostics", {}),
        }

    return {
        "seed": int(seed),
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "max_action": float(planner.cfg.max_action),
        "actor_update": str(planner.cfg.actor_update),
        "tau": per_tau,
        "replay_diagnostics": _replay_diagnostics(args, dataset_collection, ckpt_path, planner, device),
    }


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    seeds = _coerce_int_list(OmegaConf.select(args, "exp.action_diag_seeds", default=None), default=[20, 202, 2020])
    results = {
        "diagnostic": "ct_iql_action_shift",
        "seeds": seeds,
        "tau_list": _resolve_tau_list(args),
        "note": "Diagnostic only; do not use test split to select hyperparameters.",
        "results": [],
    }
    for seed in seeds:
        logger.info("Running action diagnostics for seed=%s", seed)
        results["results"].append(_run_one_seed(args, int(seed), original_cwd))

    text = json.dumps(results, indent=2, sort_keys=True)
    out_path = str(OmegaConf.select(args, "exp.action_diag_out", default="")).strip()
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
