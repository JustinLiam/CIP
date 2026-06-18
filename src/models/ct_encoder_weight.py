"""
CT encoder + WeightNet for CT+IQL EM training (no OutcomePredictor).

E-step: update WeightNet only (encoder frozen); CTD-style full-dataset fit per outer.
M-step: encoder updated via weighted IQL Q-loss (see IQLPlanner.m_step_weighted).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
    logits_w: torch.Tensor,
    mode: str,
    sinkhorn_blur: float,
    *,
    perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alignment loss; ``logits_w`` are raw WeightNet outputs (single softmax in sinkhorn)."""
    b = Z_t.size(0)
    if perm is None:
        perm = torch.randperm(b, device=Z_t.device)
    joint_rep = torch.cat([Z_t, A_t], dim=-1)
    marginal_rep = torch.cat([Z_t, A_t[perm]], dim=-1)
    if mode == "mmd":
        w = F.softmax(logits_w.reshape(-1), dim=0) * float(b)
        return compute_mmd_weighted(joint_rep, marginal_rep, w)
    if mode == "sinkhorn":
        return compute_weighted_wasserstein_joint_marginal_flat(
            joint_rep, marginal_rep, logits_w, blur=sinkhorn_blur
        )
    raise ValueError(f"Unknown ct_align_loss: {mode}")


def _weight_health(
    logits_w: torch.Tensor,
    active_mask: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """ESS fraction and weight std on active samples (mean(w)=1 per batch)."""
    b = logits_w.numel()
    w = F.softmax(logits_w.reshape(-1), dim=0) * float(b)
    if active_mask is not None and active_mask.numel() == b:
        act = active_mask.reshape(-1) > 0.5
        w_act = w[act]
        if w_act.numel() == 0:
            return 1.0, 0.0
        w_det = w_act.detach()
        n_act = float(w_det.numel())
        w_sum = float(w_det.sum())
        w_sq = float((w_det * w_det).sum())
        ess = (w_sum * w_sum) / (w_sq * n_act + 1e-12)
        w_std = float(w_det.std(unbiased=False)) if n_act > 1 else 0.0
        return ess, w_std
    w_det = w.detach()
    w_sum = float(w_det.sum())
    w_sq = float((w_det * w_det).sum())
    ess = (w_sum * w_sum) / (w_sq * float(b) + 1e-12) if b > 0 else 1.0
    w_std = float(w_det.std(unbiased=False)) if b > 1 else 0.0
    return ess, w_std


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
        local_conv_layers = int(_cfg_sel(cfg, "model.inference.local_conv_layers", 0))
        local_conv_kernel_size = int(_cfg_sel(cfg, "model.inference.local_conv_kernel_size", 6))
        local_conv_dilation = int(_cfg_sel(cfg, "model.inference.local_conv_dilation", 1))

        self.ct_encoder = CTHistoryEncoder(
            x_dim=x_dim,
            a_dim=self.treatment_dim,
            y_dim=self.output_dim,
            static_dim=self.static_size,
            d_model=64,
            num_heads=4,
            num_layers=num_layers,
            dropout=dropout,
            local_conv_layers=local_conv_layers,
            local_conv_kernel_size=local_conv_kernel_size,
            local_conv_dilation=local_conv_dilation,
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
        logits_w [B], w [B] with batch softmax * B (mean(w)=1).
        ``uniform=True`` for M-step warmup only (E-step always trains WeightNet).
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

    @torch.no_grad()
    def _encode_full_dataset(
        self,
        loader: DataLoader,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all E-step transitions: Z_det, A, active_mask at query step."""
        self.eval()
        z_parts, a_parts, act_parts = [], [], []
        for batch in loader:
            H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
            Z_t, A_t = self.encode(H_t)
            active_t = H_t["active_entries"][:, -1, 0]
            z_parts.append(Z_t.detach())
            a_parts.append(A_t)
            act_parts.append(active_t)
        return torch.cat(z_parts, 0), torch.cat(a_parts, 0), torch.cat(act_parts, 0)

    def _mean_align_on_chunks(
        self,
        joint_rep: torch.Tensor,
        marginal_rep: torch.Tensor,
        *,
        align_mode: str,
        sinkhorn_blur: float,
        chunk_size: int,
    ) -> float:
        n = joint_rep.size(0)
        total = 0.0
        n_chunks = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            idx = slice(start, end)
            logits = self.weight_net(joint_rep[idx])
            loss = compute_weighted_wasserstein_joint_marginal_flat(
                joint_rep[idx], marginal_rep[idx], logits, blur=sinkhorn_blur
            )
            total += float(loss.detach())
            n_chunks += 1
        return total / max(n_chunks, 1)

    def e_step_full_dataset(
        self,
        loader: DataLoader,
        optimizer_w: torch.optim.Optimizer,
        *,
        align_mode: str = "sinkhorn",
        sinkhorn_blur: float,
        e_epochs: int,
        train_batch_size: int,
        w_clip: Optional[float] = 1.0,
        device: str,
        outer_seed: int = 0,
    ) -> Dict[str, float]:
        """
        CTD-style E-step: encode full train set, fit WeightNet on fixed joint vs shuffled-marginal
        pairs for ``e_epochs`` passes (batched), then leave weights for detached M-step use.
        """
        if align_mode != "sinkhorn":
            raise ValueError(f"EM E-step supports sinkhorn only, got {align_mode!r}")

        Z_det, A_all, active_all = self._encode_full_dataset(loader, device)
        n = Z_det.size(0)
        if n == 0:
            return {"align_pre": 0.0, "align_post": 0.0, "w_ess_frac": 1.0, "w_std": 0.0, "n_samples": 0.0}

        # Generator device must match ``device=`` in randperm (cuda tensors need cuda generator).
        gen = torch.Generator(device=Z_det.device)
        gen.manual_seed(int(outer_seed))
        perm = torch.randperm(n, generator=gen, device=Z_det.device)
        joint_rep = torch.cat([Z_det, A_all], dim=-1)
        marginal_rep = torch.cat([Z_det, A_all[perm]], dim=-1)

        chunk = max(1, int(train_batch_size))
        align_pre = self._mean_align_on_chunks(
            joint_rep,
            marginal_rep,
            align_mode=align_mode,
            sinkhorn_blur=sinkhorn_blur,
            chunk_size=chunk,
        )

        self.weight_net.train()
        n_epochs = max(1, int(e_epochs))
        for _ in range(n_epochs):
            order = torch.randperm(n, device=Z_det.device)
            for start in range(0, n, chunk):
                idx = order[start : start + chunk]
                joint_b = joint_rep[idx]
                marg_b = marginal_rep[idx]
                optimizer_w.zero_grad(set_to_none=True)
                logits = self.weight_net(joint_b)
                loss_align = compute_weighted_wasserstein_joint_marginal_flat(
                    joint_b, marg_b, logits, blur=sinkhorn_blur
                )
                loss_align.backward()
                if w_clip is not None and w_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.weight_net.parameters(), max_norm=float(w_clip)
                    )
                optimizer_w.step()

        self.weight_net.eval()
        align_post = self._mean_align_on_chunks(
            joint_rep,
            marginal_rep,
            align_mode=align_mode,
            sinkhorn_blur=sinkhorn_blur,
            chunk_size=chunk,
        )

        with torch.no_grad():
            logits_all = self.weight_net(joint_rep)
        w_ess, w_std = _weight_health(logits_all, active_all)

        return {
            "align_pre": align_pre,
            "align_post": align_post,
            "w_ess_frac": w_ess,
            "w_std": w_std,
            "n_samples": float(n),
        }

    def e_step_batch(
        self,
        H_t: Dict[str, torch.Tensor],
        *,
        align_mode: str,
        sinkhorn_blur: float,
        k_inner: int,
        optimizer_w: torch.optim.Optimizer,
        w_clip: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Single-batch E-step (tests / legacy); always updates WeightNet."""
        with torch.no_grad():
            Z_t, A_t = self.encode(H_t)
        Z_det = Z_t.detach()
        active_t = H_t["active_entries"][:, -1, 0]

        loss_align_pre = None
        for i in range(max(1, int(k_inner))):
            optimizer_w.zero_grad(set_to_none=True)
            logits_i = self.weight_net(torch.cat([Z_det, A_t], dim=-1))
            loss_align = alignment_loss(
                Z_det, A_t, logits_i, align_mode, sinkhorn_blur
            )
            if i == 0:
                loss_align_pre = float(loss_align.detach())
            loss_align.backward()
            if w_clip is not None and w_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.weight_net.parameters(), max_norm=float(w_clip))
            optimizer_w.step()

        with torch.no_grad():
            logits_final = self.weight_net(torch.cat([Z_det, A_t], dim=-1))
            loss_align_post = alignment_loss(
                Z_det, A_t, logits_final, align_mode, sinkhorn_blur
            )
            w_ess, w_std = _weight_health(logits_final, active_t)

        return {
            "align_pre": float(loss_align_pre if loss_align_pre is not None else loss_align_post),
            "align_post": float(loss_align_post),
            "w_ess_frac": w_ess,
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
