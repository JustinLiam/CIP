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
    use_weight_net: bool = True
    align_mode: str = "sinkhorn"
    sinkhorn_blur: float = 0.01
    w_clip: Optional[float] = 1.0
    weight_max: Optional[float] = None
    m_batch_size: int = 256
    warmup_outer_iters: int = 0
    log_m_every: int = 50
    e_epochs: int = 5
    e_batch_size: int = 512
    encoder_diagnostics: bool = False
    encoder_diagnostics_every: int = 50


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
        weight_max=cfg.weight_max,
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

    # The uniform ablation keeps the complete M-step but never reads WeightNet.
    uniform = (not cfg.use_weight_net) or outer_iter <= cfg.warmup_outer_iters
    keys = ("value_loss", "q_loss", "actor_loss", "w_mean", "w_std")
    sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    diag_every = max(1, int(cfg.encoder_diagnostics_every))

    for step in range(1, num_steps + 1):
        batch = replay.sample(cfg.m_batch_size)
        collect_diag = bool(cfg.encoder_diagnostics) and (
            step % diag_every == 0 or step == num_steps
        )
        logs = planner.m_step_weighted(
            batch,
            encoder_model=model,
            encoder_optimizer=encoder_optimizer,
            uniform_weights=uniform,
            collect_encoder_diagnostics=collect_diag,
        )
        for k, value in logs.items():
            if k not in sums:
                sums[k] = 0.0
                counts[k] = 0
            sums[k] += float(value)
            counts[k] += 1

    nb = max(num_steps, 1)
    out = {}
    for k, value in sums.items():
        denom = nb if k in keys else max(counts.get(k, 0), 1)
        out[k] = value / denom
    if cfg.encoder_diagnostics:
        out["enc_diag_collected_steps"] = float(
            max(counts.get("enc_update_norm/projection", 0), counts.get("enc_grad_norm/projection", 0))
        )
    return out
