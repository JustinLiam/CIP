"""
E-step / M-step loops for CT+IQL EM training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.iql_raw_transition_dataset import IQLRawReplayBuffer
from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.planners.iql_planner import IQLPlanner


@dataclass
class EMTrainConfig:
    align_mode: str = "sinkhorn"
    sinkhorn_blur: float = 0.01
    w_clip: Optional[float] = 1.0
    m_batch_size: int = 256
    warmup_outer_iters: int = 0
    log_m_every: int = 50
    e_epochs: int = 5
    e_batch_size: int = 512


def run_e_step_full(
    model: CTEncoderWeightModel,
    loader: DataLoader,
    optimizer_w: torch.optim.Optimizer,
    cfg: EMTrainConfig,
    device: str,
    *,
    outer_iter: int,
    outer_seed: int,
) -> Dict[str, float]:
    """CTD-style full-dataset E-step: always trains WeightNet (encoder frozen)."""
    return model.e_step_full_dataset(
        loader,
        optimizer_w,
        align_mode=cfg.align_mode,
        sinkhorn_blur=cfg.sinkhorn_blur,
        e_epochs=cfg.e_epochs,
        train_batch_size=cfg.e_batch_size,
        w_clip=cfg.w_clip,
        device=device,
        outer_seed=outer_seed,
    )


def run_m_step_steps(
    model: CTEncoderWeightModel,
    planner: IQLPlanner,
    encoder_optimizer: torch.optim.Optimizer,
    replay: IQLRawReplayBuffer,
    num_steps: int,
    cfg: EMTrainConfig,
    *,
    outer_iter: int,
) -> Dict[str, float]:
    """M-step: freeze WeightNet; weighted V→Q→π; Q-step updates encoder."""
    model.train()
    model.weight_net.eval()
    planner.actor.train()
    planner.qf.train()
    planner.vf.train()

    # Warmup: uniform w in M-step only; encoder still updated via Q-loss.
    uniform = outer_iter <= cfg.warmup_outer_iters
    keys = ("value_loss", "q_loss", "actor_loss", "w_mean", "w_std")
    sums = {k: 0.0 for k in keys}

    for step in range(1, num_steps + 1):
        batch = replay.sample(cfg.m_batch_size)
        logs = planner.m_step_weighted(
            batch,
            encoder_model=model,
            encoder_optimizer=encoder_optimizer,
            uniform_weights=uniform,
        )
        for k in keys:
            sums[k] += logs.get(k, 0.0)

    nb = max(num_steps, 1)
    return {k: sums[k] / nb for k in keys}
