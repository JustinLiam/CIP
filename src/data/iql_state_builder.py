"""Online state construction for CT+IQL EM (planning-oriented)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from src.models.ct_encoder_weight import CTEncoderWeightModel


def build_augmented_state(
    z_t: torch.Tensor,
    y_target: torch.Tensor,
    delta_t_norm: torch.Tensor,
    a_prev_tanh: torch.Tensor,
) -> torch.Tensor:
    """
    s = concat(Z_t, y_target, delta_t_norm, a_prev) matching legacy IQL state layout.
    Shapes: z [B,z], y [B,y], delta [B,1], a_prev [B,a].
    """
    if delta_t_norm.dim() == 1:
        delta_t_norm = delta_t_norm.unsqueeze(-1)
    return torch.cat([z_t, y_target, delta_t_norm, a_prev_tanh], dim=-1)


def encode_history_batch(
    model: CTEncoderWeightModel,
    H_t: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns Z_t [B,z_dim], A_t [B,a_dim]."""
    return model.encode(H_t)
