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
    k_inner: int = 1
    w_clip: Optional[float] = 1.0
    m_batch_size: int = 256
    warmup_outer_iters: int = 0
    log_m_every: int = 50
    e_epochs: int = 1


def run_e_step_epoch(
    model: CTEncoderWeightModel,
    loader: DataLoader,
    optimizer_w: torch.optim.Optimizer,
    cfg: EMTrainConfig,
    device: str,
    *,
    outer_iter: int,
) -> Dict[str, float]:
    """E-step: freeze encoder + IQL; update WeightNet only."""
    model.eval()
    uniform = outer_iter <= cfg.warmup_outer_iters
    n = 0
    sums = {"align_pre": 0.0, "align_post": 0.0, "w_ess_frac": 0.0, "w_std": 0.0}

    for batch in loader:
        H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
        metrics = model.e_step_batch(
            H_t,
            align_mode=cfg.align_mode,
            sinkhorn_blur=cfg.sinkhorn_blur,
            k_inner=cfg.k_inner,
            optimizer_w=optimizer_w,
            w_clip=cfg.w_clip,
            uniform_weights=uniform,
        )
        for k in sums:
            sums[k] += metrics[k]
        n += 1

    nb = max(n, 1)
    return {k: sums[k] / nb for k in sums}


def run_e_step_epochs(
    model: CTEncoderWeightModel,
    loader: DataLoader,
    optimizer_w: torch.optim.Optimizer,
    cfg: EMTrainConfig,
    device: str,
    *,
    outer_iter: int,
) -> Dict[str, float]:
    """Run ``cfg.e_epochs`` full passes over the E-step loader; average batch metrics."""
    n_epochs = max(1, int(cfg.e_epochs))
    agg = {"align_pre": 0.0, "align_post": 0.0, "w_ess_frac": 0.0, "w_std": 0.0}
    for _ in range(n_epochs):
        ep = run_e_step_epoch(
            model, loader, optimizer_w, cfg, device, outer_iter=outer_iter
        )
        for k in agg:
            agg[k] += ep[k]
    return {k: agg[k] / n_epochs for k in agg}


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
    """M-step: freeze WeightNet; weighted V→Q→π per batch; Q-step updates encoder."""
    model.train()
    model.weight_net.eval()
    planner.actor.train()
    planner.qf.train()
    planner.vf.train()

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
