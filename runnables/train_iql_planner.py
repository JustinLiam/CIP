import json
import logging
import os
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.iql_dataset_builder import build_iql_transitions_from_ct
from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics
from src.models.inference_model import InferenceModel
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig, TransitionReplayBuffer
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _state_dict_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_dict_to_cpu(v) for k, v in obj.items()}
    return obj


def _spearman_rho(a: List[float], b: List[float]) -> Optional[float]:
    """Pearson correlation on ranks; None when undefined (< 2 samples or zero variance)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2 or aa.size != bb.size:
        return None
    ra = aa.argsort().argsort().astype(np.float64)
    rb = bb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return None
    return float((ra * rb).sum() / denom)


def _top_k_overlap(
    metric_a: List[float], metric_b: List[float], k: int
) -> Optional[float]:
    """Fraction of top-K (lowest-metric) indices shared between two ranking metrics."""
    if k <= 0 or len(metric_a) == 0 or len(metric_a) != len(metric_b):
        return None
    k = min(k, len(metric_a))
    idx_a = set(np.argsort(np.asarray(metric_a))[:k].tolist())
    idx_b = set(np.argsort(np.asarray(metric_b))[:k].tolist())
    return float(len(idx_a & idx_b) / k)


def _should_run_iql_val(step: int, total_updates: int, every: int) -> bool:
    if every <= 0:
        return False
    if step % every == 0:
        return True
    if step == total_updates and total_updates % every != 0:
        return True
    return False


@contextmanager
def _isolated_rng(seed: int):
    """
    Run a block with deterministic numpy/torch RNG state derived from ``seed``,
    then restore the previous global RNG state. This keeps the val protocol
    (random history offsets in ``CIPDataset``) identical across checkpoints
    without perturbing the training-time replay sampling sequence.
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(args.exp.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    iql_updates = int(OmegaConf.select(args, "exp.iql_updates", default=5000))
    iql_batch_size = int(OmegaConf.select(args, "exp.iql_batch_size", default=256))
    iql_max_patients = OmegaConf.select(args, "exp.iql_max_patients", default=None)
    iql_max_patients = None if iql_max_patients is None else int(iql_max_patients)
    iql_ckpt = OmegaConf.select(args, "exp.iql_inference_ckpt", default="")

    original_cwd = get_original_cwd()
    args["exp"]["processed_data_dir"] = os.path.join(original_cwd, args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    inference_model = InferenceModel(args).to(device)
    load_inference_checkpoint(inference_model, iql_ckpt, device)
    inference_model.eval()

    iql_max_action = float(OmegaConf.select(args, "exp.iql_max_action", default=1.0))
    iql_unit_actions = bool(OmegaConf.select(args, "exp.iql_dataset_actions_unit_interval", default=True))
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))

    transitions = build_iql_transitions_from_ct(
        data=dataset_collection.train_f.data,
        inference_model=inference_model,
        device=device,
        reward_type=str(OmegaConf.select(args, "exp.iql_reward_type", default="negative_outcome_mse")),
        max_patients=iql_max_patients,
        max_action=iql_max_action,
        dataset_actions_unit_interval=iql_unit_actions,
        max_tau=max_tau,
    )
    logger.info(f"Built transitions: {len(transitions['states'])}")

    log_window = int(OmegaConf.select(args, "exp.iql_log_window", default=200))
    iql_max_grad = OmegaConf.select(args, "exp.iql_max_grad_norm", default=None)
    iql_max_grad = None if iql_max_grad is None else float(iql_max_grad)

    planner_cfg = IQLPlannerConfig(
        state_dim=int(transitions["states"].shape[1]),
        action_dim=int(transitions["actions"].shape[1]),
        max_action=iql_max_action,
        hidden_dim=int(OmegaConf.select(args, "exp.iql_hidden_dim", default=256)),
        n_hidden=int(OmegaConf.select(args, "exp.iql_n_hidden", default=2)),
        iql_tau=float(OmegaConf.select(args, "exp.iql_tau", default=0.7)),
        beta=float(OmegaConf.select(args, "exp.iql_beta", default=3.0)),
        discount=float(OmegaConf.select(args, "exp.iql_discount", default=0.99)),
        tau=float(OmegaConf.select(args, "exp.iql_target_tau", default=0.005)),
        actor_lr=float(OmegaConf.select(args, "exp.iql_actor_lr", default=3e-4)),
        qf_lr=float(OmegaConf.select(args, "exp.iql_qf_lr", default=3e-4)),
        vf_lr=float(OmegaConf.select(args, "exp.iql_vf_lr", default=3e-4)),
        max_steps=iql_updates,
        deterministic_actor=bool(OmegaConf.select(args, "exp.iql_deterministic", default=False)),
        actor_dropout=OmegaConf.select(args, "exp.iql_actor_dropout", default=None),
        max_grad_norm=iql_max_grad,
        device=device,
    )

    planner = IQLPlanner(planner_cfg)
    replay = TransitionReplayBuffer(transitions, device=device)

    iql_val_every = int(OmegaConf.select(args, "exp.iql_val_every", default=0))
    val_metric_key = str(OmegaConf.select(args, "exp.iql_val_metric", default="mae_uns")).strip().lower()
    if val_metric_key not in ("mae_uns", "mae_norm"):
        raise ValueError(f"exp.iql_val_metric must be mae_uns or mae_norm, got {val_metric_key!r}")
    val_bs_cfg = OmegaConf.select(args, "exp.iql_val_batch_size", default=None)
    val_bs = int(val_bs_cfg) if val_bs_cfg is not None else int(
        OmegaConf.select(args, "exp.batch_size_val", default=128)
    )
    eval_tau = int(args.exp.tau)
    autoreg = bool(OmegaConf.select(args, "exp.iql_eval_autoregressive", default=True))
    save_last_ckpt = bool(OmegaConf.select(args, "exp.iql_val_save_last", default=True))
    val_seed = int(args.exp.seed) + 17  # fixed offset; do not collide with training seed

    # A/B world selection.
    val_worlds_raw = OmegaConf.select(args, "exp.iql_val_worlds", default=["sim"])
    if isinstance(val_worlds_raw, str):
        val_worlds: Tuple[str, ...] = tuple(val_worlds_raw.split(","))
    else:
        val_worlds = tuple(str(w).strip() for w in val_worlds_raw)
    for w in val_worlds:
        if w not in ("sim", "predictor"):
            raise ValueError(f"exp.iql_val_worlds contains invalid world: {w!r}")

    selection_world = str(OmegaConf.select(args, "exp.iql_val_selection_world", default=val_worlds[0])).strip()
    if selection_world not in val_worlds:
        raise ValueError(
            f"exp.iql_val_selection_world={selection_world!r} must be one of exp.iql_val_worlds={val_worlds}"
        )
    debug_panel_enabled = bool(OmegaConf.select(args, "exp.iql_debug_panel", default=False))

    # Default save dir: iql_models/seed_${s}/gamma_${g}/. Override via exp.iql_save_dir
    # so grid search / parallel workers don't clobber the canonical file used by
    # the main pipeline or by each other.
    _iql_save_dir_override = OmegaConf.select(args, "exp.iql_save_dir", default=None)
    if _iql_save_dir_override:
        model_dir = Path(str(_iql_save_dir_override))
        if not model_dir.is_absolute():
            model_dir = Path(get_original_cwd()) / model_dir
    else:
        model_dir = Path(get_original_cwd()) / "iql_models" / f"seed_{args.exp.seed}" / f"gamma_{int(args.dataset.coeff)}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "iql_planner.pt"
    last_path = model_dir / "iql_planner_last.pt"
    trace_path = model_dir / "iql_val_trace.json"
    debug_panel_path = model_dir / "iql_debug_panel.jsonl"

    # Warn early if predictor requested but its weights weren't loaded.
    if "predictor" in val_worlds and not getattr(inference_model, "_outcome_predictor_loaded", False):
        logger.warning(
            "exp.iql_val_worlds includes 'predictor' but outcome_predictor weights were not "
            "loaded from the CT checkpoint. Predictor-world metrics will reflect a randomly "
            "initialized network (i.e. they are meaningless). Re-train CT so that "
            "ct_best_encoder.pt contains 'outcome_predictor'."
        )

    if iql_val_every > 0:
        logger.info(
            f"Periodic val enabled: every {iql_val_every} steps on val_f, "
            f"metric={val_metric_key}, val_batch_size={val_bs}, autoregressive={autoreg}, "
            f"val_seed={val_seed}, worlds={val_worlds}, selection_world={selection_world!r}"
        )
    else:
        logger.info("Periodic val disabled (iql_val_every=0); will save last-step checkpoint only.")
    if debug_panel_enabled:
        if debug_panel_path.exists():
            debug_panel_path.unlink()
        logger.info(f"IQL val debug panel enabled; writing JSONL records to {debug_panel_path}")

    best_per_world: Dict[str, Dict[str, Any]] = {
        w: {"metric": float("inf"), "state": None, "step": -1} for w in val_worlds
    }
    val_trace: List[Dict[str, Any]] = []
    prev_action_fingerprints: Dict[str, Optional[str]] = {w: None for w in val_worlds}

    loss_keys = ("actor_loss", "q_loss", "value_loss")
    loss_buf = {k: deque(maxlen=max(1, log_window)) for k in loss_keys}

    for step in range(1, iql_updates + 1):
        batch = replay.sample(iql_batch_size)
        logs = planner.train_step(batch)
        for k in loss_keys:
            loss_buf[k].append(logs[k])
        if step % 200 == 0 or step == 1:
            parts = [f"[{step}/{iql_updates}]"]
            for k in loss_keys:
                last = logs[k]
                mean = float(np.mean(loss_buf[k]))
                parts.append(f"{k}={last:.4f} (mean_{min(step, log_window)}={mean:.4f})")
            logger.info(", ".join(parts))

        if _should_run_iql_val(step, iql_updates, iql_val_every):
            with _isolated_rng(val_seed):
                metrics = aggregate_iql_planner_metrics(
                    planner,
                    inference_model,
                    dataset_collection,
                    dataset_collection.val_f,
                    args,
                    device=device,
                    tau=eval_tau,
                    max_tau=max_tau,
                    autoregressive_eval=autoreg,
                    val_batch_size=val_bs,
                    log_batches=False,
                    worlds=val_worlds,
                    debug_panel=debug_panel_enabled,
                )

            per_world = metrics.get("per_world", {val_worlds[0]: metrics})

            trace_entry: Dict[str, Any] = {"step": step}
            improved_worlds: List[str] = []
            for w in val_worlds:
                m_w = float(per_world[w][val_metric_key])
                trace_entry[w] = {
                    "mae_norm": float(per_world[w]["mae_norm"]),
                    "mae_uns": float(per_world[w]["mae_uns"]),
                    "rmse_norm": float(per_world[w]["rmse_norm"]),
                }
                if m_w < best_per_world[w]["metric"]:
                    best_per_world[w]["metric"] = m_w
                    best_per_world[w]["state"] = _state_dict_to_cpu(planner.state_dict())
                    best_per_world[w]["step"] = step
                    improved_worlds.append(w)
            val_trace.append(trace_entry)

            parts = [f"[val step {step}/{iql_updates}]"]
            for w in val_worlds:
                m_w = per_world[w][val_metric_key]
                tag = "*" if w in improved_worlds else ""
                parts.append(
                    f"{w}:{val_metric_key}={m_w:.6f}"
                    f"(mae_norm={per_world[w]['mae_norm']:.6f}){tag}"
                )
            logger.info(" ".join(parts))
            if debug_panel_enabled and "debug_panel" in metrics:
                panel = metrics["debug_panel"]
                panel["step"] = int(step)
                panel["val_metric"] = val_metric_key
                panel["worlds"] = list(val_worlds)
                panel["selection_world"] = selection_world
                action_flag = True
                for w in val_worlds:
                    world_panel = panel.get("per_world", {}).setdefault(w, {})
                    action_info = world_panel.get("action_sequence", {})
                    fingerprint = action_info.get("fingerprint")
                    prev_fp = prev_action_fingerprints.get(w)
                    changed_vs_prev = bool(fingerprint is not None and (prev_fp is None or fingerprint != prev_fp))
                    world_panel["action_sequence_changed_vs_prev"] = changed_vs_prev
                    if fingerprint is not None:
                        prev_action_fingerprints[w] = str(fingerprint)
                    action_flag = action_flag and changed_vs_prev
                panel["summary_flags"]["action_sequence_changes"] = bool(action_flag)
                with open(debug_panel_path, "a") as f:
                    f.write(json.dumps(panel) + "\n")

    # -------------------------------------------------------------------
    # Cross-world diagnostics: how well does predictor-ranking track sim-ranking?
    # -------------------------------------------------------------------
    if len(val_worlds) >= 2 and len(val_trace) >= 2:
        logger.info("=" * 80)
        logger.info("Cross-world checkpoint-ranking diagnostics "
                    f"(#val_points={len(val_trace)}, metric={val_metric_key})")
        series: Dict[str, List[float]] = {
            w: [float(entry[w][val_metric_key]) for entry in val_trace] for w in val_worlds
        }
        steps_list = [int(entry["step"]) for entry in val_trace]

        for i in range(len(val_worlds)):
            for j in range(i + 1, len(val_worlds)):
                w_a, w_b = val_worlds[i], val_worlds[j]
                rho = _spearman_rho(series[w_a], series[w_b])
                rho_str = f"{rho:.4f}" if rho is not None else "NaN"
                logger.info(f"  Spearman rho({w_a}, {w_b}) = {rho_str}")
                for K in (1, 3, 5):
                    ov = _top_k_overlap(series[w_a], series[w_b], K)
                    ov_str = f"{ov:.3f}" if ov is not None else "NaN"
                    logger.info(f"  Top-{K} overlap({w_a}, {w_b}) = {ov_str}")

        # Regret: picking by world w_sel, measured against w_ref
        for w_ref in val_worlds:
            ref_series = series[w_ref]
            best_ref_idx = int(np.argmin(ref_series))
            best_ref_val = ref_series[best_ref_idx]
            for w_sel in val_worlds:
                if w_sel == w_ref:
                    continue
                picked_idx = int(np.argmin(series[w_sel]))
                picked_step = steps_list[picked_idx]
                picked_ref_val = ref_series[picked_idx]
                regret = picked_ref_val - best_ref_val
                logger.info(
                    f"  Regret: pick-by-{w_sel} vs best-by-{w_ref} "
                    f"(ref={w_ref}:{val_metric_key}) = {regret:+.6f} "
                    f"(picked step {picked_step}: {picked_ref_val:.6f}; "
                    f"best step {steps_list[best_ref_idx]}: {best_ref_val:.6f})"
                )
        logger.info("=" * 80)

    # -------------------------------------------------------------------
    # Save checkpoints + val trace.
    # -------------------------------------------------------------------
    logger.info(f"IQL planner checkpoints will be saved under: {model_dir}")

    any_val_improved = any(bw["state"] is not None for bw in best_per_world.values())

    if iql_val_every > 0 and any_val_improved:
        # Primary checkpoint = checkpoint selected by the configured selection world.
        sel = best_per_world[selection_world]
        if sel["state"] is None:
            logger.warning(
                f"No val improvement recorded for selection_world={selection_world!r}; "
                "falling back to last-step weights for the primary checkpoint."
            )
            torch.save(_state_dict_to_cpu(planner.state_dict()), ckpt_path)
        else:
            torch.save(sel["state"], ckpt_path)
            logger.info(
                f"Saved BEST IQL planner (selection_world={selection_world!r}) to {ckpt_path} "
                f"({val_metric_key}={sel['metric']:.6f} at step {sel['step']})"
            )

        # Per-world best checkpoints (for offline A/B comparison; do not overwrite primary).
        for w, bw in best_per_world.items():
            if bw["state"] is None:
                continue
            per_path = model_dir / f"iql_planner_best_{w}.pt"
            torch.save(bw["state"], per_path)
            logger.info(
                f"Saved per-world BEST IQL planner [{w}] to {per_path} "
                f"({val_metric_key}={bw['metric']:.6f} at step {bw['step']})"
            )

        if save_last_ckpt:
            torch.save(_state_dict_to_cpu(planner.state_dict()), last_path)
            logger.info(f"Saved LAST-step IQL planner to {last_path}")
    else:
        torch.save(_state_dict_to_cpu(planner.state_dict()), ckpt_path)
        logger.info(
            f"Saved IQL planner to {ckpt_path} "
            "(periodic val disabled or no val improvement; last-step weights)"
        )

    if val_trace:
        trace_payload = {
            "val_metric": val_metric_key,
            "worlds": list(val_worlds),
            "selection_world": selection_world,
            "tau": eval_tau,
            "max_tau": max_tau,
            "val_seed": val_seed,
            "history": val_trace,
            "best_per_world": {
                w: {"metric": bw["metric"], "step": bw["step"]}
                for w, bw in best_per_world.items()
            },
        }
        with open(trace_path, "w") as f:
            json.dump(trace_payload, f, indent=2)
        logger.info(f"Saved val trace to {trace_path}")


if __name__ == "__main__":
    main()
