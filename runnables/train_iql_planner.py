import logging
import os
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

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

    if iql_val_every > 0:
        logger.info(
            f"Periodic val enabled: every {iql_val_every} steps on val_f, "
            f"metric={val_metric_key}, val_batch_size={val_bs}, autoregressive={autoreg}, "
            f"val_seed={val_seed}"
        )
    else:
        logger.info("Periodic val disabled (iql_val_every=0); will save last-step checkpoint only.")

    best_metric = float("inf")
    best_state_cpu = None
    best_step = -1

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
                )
            m = float(metrics[val_metric_key])
            improved = m < best_metric
            tag = " *best" if improved else ""
            logger.info(
                f"[val step {step}/{iql_updates}] {val_metric_key}={m:.6f} "
                f"(mae_norm={metrics['mae_norm']:.6f}, mae_uns={metrics['mae_uns']:.6f}, "
                f"rmse_norm={metrics['rmse_norm']:.6f}){tag}"
            )
            if improved:
                best_metric = m
                best_state_cpu = _state_dict_to_cpu(planner.state_dict())
                best_step = step

    model_dir = Path(get_original_cwd()) / "iql_models" / f"seed_{args.exp.seed}" / f"gamma_{int(args.dataset.coeff)}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "iql_planner.pt"
    last_path = model_dir / "iql_planner_last.pt"

    if iql_val_every > 0 and best_state_cpu is not None:
        torch.save(best_state_cpu, ckpt_path)
        logger.info(
            f"Saved BEST IQL planner to {ckpt_path} "
            f"({val_metric_key}={best_metric:.6f} at step {best_step})"
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


if __name__ == "__main__":
    main()
