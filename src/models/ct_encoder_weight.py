"""
CT encoder + WeightNet for CT+IQL EM training (no OutcomePredictor).

E-step: update WeightNet only (encoder frozen).
M-step: encoder updated via weighted IQL Q-loss (see IQLPlanner.m_step_weighted).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.models.ct_deconfound import WeightNet, build_covariate_x
from src.models.ct_history_encoder import CTHistoryEncoder, ProjectionHead
from src.utils.utils import (
    compute_mmd_weighted,
    compute_weighted_wasserstein_joint_marginal_flat,
)


def _cfg_sel(cfg, key: str, default):
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, key, default=default)
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def alignment_loss(
    Z_t: torch.Tensor,
    A_t: torch.Tensor,
    w: torch.Tensor,
    mode: str,
    sinkhorn_blur: float,
) -> torch.Tensor:
    B = Z_t.size(0)
    perm = torch.randperm(B, device=Z_t.device)
    joint_rep = torch.cat([Z_t, A_t], dim=-1)
    marginal_rep = torch.cat([Z_t, A_t[perm]], dim=-1)
    if mode == "mmd":
        return compute_mmd_weighted(joint_rep, marginal_rep, w)
    if mode == "sinkhorn":
        return compute_weighted_wasserstein_joint_marginal_flat(
            joint_rep, marginal_rep, w, blur=sinkhorn_blur
        )
    raise ValueError(f"Unknown ct_align_loss: {mode}")


class CTEncoderWeightModel(nn.Module):
    """Encoder + WeightNet only (planning-oriented EM; no outcome predictor)."""

    def __init__(self, cfg, x_dim: int):
        super().__init__()
        ds = cfg["dataset"]
        md = cfg["model"]
        self.cfg = cfg
        self.treatment_dim = int(ds["treatment_size"])
        self.output_dim = int(ds["output_size"])
        self.static_size = int(ds["static_size"])
        self.z_dim = int(md["z_dim"])
        dropout = float(_cfg_sel(cfg, "exp.dropout", 0.1))
        num_layers = int(md["inference"]["num_layers"])

        self.ct_encoder = CTHistoryEncoder(
            x_dim=x_dim,
            a_dim=self.treatment_dim,
            y_dim=self.output_dim,
            static_dim=self.static_size,
            d_model=64,
            num_heads=4,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.projection = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=self.z_dim)
        wh = int(_cfg_sel(cfg, "exp.ct_weight_hidden", 64))
        self.weight_net = WeightNet(self.z_dim, self.treatment_dim, hidden_dim=wh)

    def encode(self, H_t: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns Z_t [B, z_dim], A_t [B, a_dim]."""
        x = build_covariate_x(H_t, self.cfg)
        a_prev = H_t["prev_treatments"]
        y_prev = H_t["prev_outputs"]
        active = H_t.get("active_entries")
        static = H_t.get("static_features")

        ct_rep = self.ct_encoder(
            x=x,
            a=a_prev,
            y=y_prev,
            active_entries=active,
            static_features=static,
        )
        Z_seq = self.projection(ct_rep)
        Z_t = Z_seq[:, -1, :]
        A_t = H_t["current_treatments"][:, -1, :]
        return Z_t, A_t

    def compute_weights(
        self,
        Z_t: torch.Tensor,
        A_t: torch.Tensor,
        *,
        detach_z: bool = True,
        uniform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits_w [B], w [B] with batch softmax * B.
        ``uniform=True`` returns all-ones weights (warmup).
        """
        if uniform:
            b = Z_t.size(0)
            device, dtype = Z_t.device, Z_t.dtype
            logits = torch.zeros(b, device=device, dtype=dtype)
            w = torch.ones(b, device=device, dtype=dtype)
            return logits, w
        z_in = Z_t.detach() if detach_z else Z_t
        za = torch.cat([z_in, A_t], dim=-1)
        logits_w = self.weight_net(za)
        w = F.softmax(logits_w, dim=0) * float(Z_t.size(0))
        return logits_w, w

    def e_step_batch(
        self,
        H_t: Dict[str, torch.Tensor],
        *,
        align_mode: str,
        sinkhorn_blur: float,
        k_inner: int,
        optimizer_w: torch.optim.Optimizer,
        w_clip: Optional[float] = 1.0,
        uniform_weights: bool = False,
    ) -> Dict[str, float]:
        """
        E-step on one batch: freeze encoder, update WeightNet only.
        """
        with torch.no_grad():
            Z_t, A_t = self.encode(H_t)
        Z_det = Z_t.detach()

        if uniform_weights:
            return {
                "align_pre": 0.0,
                "align_post": 0.0,
                "w_ess_frac": 1.0,
                "w_std": 0.0,
            }

        loss_align_pre = None
        for i in range(max(1, int(k_inner))):
            optimizer_w.zero_grad(set_to_none=True)
            _, w_i = self.compute_weights(Z_det, A_t, detach_z=True)
            loss_align = alignment_loss(Z_det, A_t, w_i, align_mode, sinkhorn_blur)
            if i == 0:
                loss_align_pre = float(loss_align.detach())
            loss_align.backward()
            if w_clip is not None and w_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.weight_net.parameters(), max_norm=float(w_clip))
            optimizer_w.step()

        with torch.no_grad():
            _, w_final = self.compute_weights(Z_det, A_t, detach_z=True)
            loss_align_post = alignment_loss(Z_det, A_t, w_final, align_mode, sinkhorn_blur)
            w_det = w_final.detach()
            b = float(w_det.numel())
            w_sum = float(w_det.sum())
            w_sq = float((w_det * w_det).sum())
            ess = (w_sum * w_sum) / (w_sq * b + 1e-12) if b > 0 else 1.0
            w_std = float(w_det.std(unbiased=False)) if b > 1 else 0.0

        return {
            "align_pre": float(loss_align_pre if loss_align_pre is not None else loss_align_post),
            "align_post": float(loss_align_post),
            "w_ess_frac": ess,
            "w_std": w_std,
        }

    def encoder_parameters(self):
        return list(self.ct_encoder.parameters()) + list(self.projection.parameters())

    def state_dict_encoder(self) -> Dict[str, Any]:
        return {
            "ct_history_encoder": self.ct_encoder.state_dict(),
            "projection_head": self.projection.state_dict(),
            "weight_net": self.weight_net.state_dict(),
        }

    def load_state_dict_encoder(self, state: Dict[str, Any], *, strict: bool = True) -> None:
        self.ct_encoder.load_state_dict(state["ct_history_encoder"], strict=strict)
        self.projection.load_state_dict(state["projection_head"], strict=strict)
        if "weight_net" in state:
            self.weight_net.load_state_dict(state["weight_net"], strict=strict)
