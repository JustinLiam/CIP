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
from src.models.sequence_utils import gather_last_valid, last_valid_mask
from src.utils.stable_iql_em_defaults import stable_default
from src.utils.utils import (
    compute_mmd_weighted,
    compute_weighted_wasserstein_joint_marginal_flat,
)


def _stratified_permutation(
    strata: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Permute rows only within each stratum, preserving its empirical marginal."""
    strata = strata.reshape(-1)
    permutation = torch.arange(strata.numel(), device=strata.device)
    for value in torch.unique(strata, sorted=True):
        indices = torch.nonzero(strata == value, as_tuple=False).reshape(-1)
        if indices.numel() > 1:
            local_order = torch.randperm(
                indices.numel(), generator=generator, device=strata.device
            )
            permutation[indices] = indices[local_order]
    return permutation


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
    return alignment_loss_from_representations(
        joint_rep,
        marginal_rep,
        logits_w,
        mode,
        sinkhorn_blur,
    )


def alignment_loss_from_representations(
    joint_rep: torch.Tensor,
    marginal_rep: torch.Tensor,
    logits_w: torch.Tensor,
    mode: str,
    sinkhorn_blur: float,
) -> torch.Tensor:
    """Compute the selected WeightNet alignment loss on fixed joint/marginal pairs."""
    b = joint_rep.size(0)
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
    diagnostics = _weight_diagnostics(logits_w, active_mask)
    return diagnostics["w_ess_frac"], diagnostics["w_std"]


def _weight_diagnostics(
    logits_w: torch.Tensor,
    active_mask: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Weight concentration diagnostics on active samples."""
    b = logits_w.numel()
    w = F.softmax(logits_w.reshape(-1), dim=0) * float(b)
    if active_mask is not None and active_mask.numel() == b:
        act = active_mask.reshape(-1) > 0.5
        w = w[act]
    w_det = w.detach()
    if w_det.numel() == 0:
        return {
            "w_ess_frac": 1.0,
            "w_std": 0.0,
            "w_var": 0.0,
            "w_max": 1.0,
            "w_p50": 1.0,
            "w_p90": 1.0,
            "w_p95": 1.0,
            "w_p99": 1.0,
        }
    n = float(w_det.numel())
    w_sum = float(w_det.sum())
    w_sq = float((w_det * w_det).sum())
    w_std = float(w_det.std(unbiased=False)) if n > 1 else 0.0
    quantiles = torch.quantile(
        w_det,
        w_det.new_tensor([0.50, 0.90, 0.95, 0.99]),
    )
    return {
        "w_ess_frac": (w_sum * w_sum) / (w_sq * n + 1e-12),
        "w_std": w_std,
        "w_var": w_std * w_std,
        "w_max": float(w_det.max()),
        "w_p50": float(quantiles[0]),
        "w_p90": float(quantiles[1]),
        "w_p95": float(quantiles[2]),
        "w_p99": float(quantiles[3]),
    }


class CTEncoderWeightModel(nn.Module):
    """Encoder + WeightNet only (planning-oriented EM; no outcome predictor)."""

    def __init__(self, cfg, x_dim: int):
        super().__init__()
        ds = cfg["dataset"]
        self.cfg = cfg
        self.treatment_dim = int(ds["treatment_size"])
        self.output_dim = int(ds["output_size"])
        self.static_size = int(ds["static_size"])
        self.z_dim = int(_cfg_sel(cfg, "model.z_dim", stable_default("model.z_dim")))
        dropout = float(_cfg_sel(cfg, "exp.dropout", 0.1))
        num_layers = int(_cfg_sel(cfg, "model.inference.num_layers", stable_default("model.inference.num_layers")))
        local_conv_layers = int(
            _cfg_sel(cfg, "model.inference.local_conv_layers", stable_default("model.inference.local_conv_layers"))
        )
        local_conv_kernel_size = int(
            _cfg_sel(
                cfg,
                "model.inference.local_conv_kernel_size",
                stable_default("model.inference.local_conv_kernel_size"),
            )
        )
        local_conv_dilation = int(
            _cfg_sel(cfg, "model.inference.local_conv_dilation", stable_default("model.inference.local_conv_dilation"))
        )

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
        Z_t = gather_last_valid(Z_seq, active)
        A_t = gather_last_valid(H_t["current_treatments"], active)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all E-step transitions, retaining their decision-time stratum."""
        self.eval()
        z_parts, a_parts, act_parts, time_parts = [], [], [], []
        for batch in loader:
            H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
            Z_t, A_t = self.encode(H_t)
            active_t = last_valid_mask(H_t["active_entries"])
            time_index = batch.get("time_index")
            if time_index is None:
                # Legacy/custom loaders do not carry metadata. The last valid
                # row is the decision index under the processed-data contract.
                time_index = H_t["active_entries"].sum(dim=1).reshape(-1).long() - 1
            else:
                time_index = time_index.to(device=device, dtype=torch.long).reshape(-1)
            z_parts.append(Z_t.detach())
            a_parts.append(A_t)
            act_parts.append(active_t)
            time_parts.append(time_index)
        return (
            torch.cat(z_parts, 0),
            torch.cat(a_parts, 0),
            torch.cat(act_parts, 0),
            torch.cat(time_parts, 0),
        )

    def _mean_align_on_chunks(
        self,
        joint_rep: torch.Tensor,
        marginal_rep: torch.Tensor,
        *,
        align_mode: str,
        sinkhorn_blur: float,
        chunk_size: int,
        strata: Optional[torch.Tensor] = None,
    ) -> float:
        n = joint_rep.size(0)
        total = 0.0
        total_samples = 0
        if strata is None:
            groups = [torch.arange(n, device=joint_rep.device)]
        else:
            groups = [
                torch.nonzero(strata == value, as_tuple=False).reshape(-1)
                for value in torch.unique(strata, sorted=True)
            ]
        for group in groups:
            for start in range(0, group.numel(), chunk_size):
                idx = group[start : start + chunk_size]
                if idx.numel() < 2:
                    continue
                logits = self.weight_net(joint_rep[idx])
                loss = alignment_loss_from_representations(
                    joint_rep[idx],
                    marginal_rep[idx],
                    logits,
                    align_mode,
                    sinkhorn_blur,
                )
                count = int(idx.numel())
                total += float(loss.detach()) * count
                total_samples += count
        return total / max(total_samples, 1)

    def _mean_align_over_time(
        self,
        joint_rep: torch.Tensor,
        marginal_rep: torch.Tensor,
        time_index: torch.Tensor,
        *,
        sinkhorn_blur: float,
    ) -> float:
        """CTD-style mean of full-stratum Sinkhorn losses over decision times."""
        losses = []
        for time_value in torch.unique(time_index, sorted=True):
            idx = torch.nonzero(
                time_index == time_value, as_tuple=False
            ).reshape(-1)
            if idx.numel() < 2:
                continue
            logits = self.weight_net(joint_rep[idx])
            loss = alignment_loss_from_representations(
                joint_rep[idx],
                marginal_rep[idx],
                logits,
                "sinkhorn",
                sinkhorn_blur,
            )
            losses.append(float(loss.detach()))
        return sum(losses) / max(len(losses), 1)

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
        if align_mode not in {"sinkhorn", "mmd"}:
            raise ValueError(
                f"EM E-step supports align_mode='sinkhorn' or 'mmd', got {align_mode!r}"
            )

        Z_det, A_all, active_all, time_all = self._encode_full_dataset(loader, device)
        n_total = Z_det.size(0)
        valid = active_all > 0.5
        Z_det = Z_det[valid]
        A_all = A_all[valid]
        active_all = active_all[valid]
        time_all = time_all[valid]
        n = Z_det.size(0)
        if n == 0:
            return {
                "align_pre": 0.0,
                "align_post": 0.0,
                "w_ess_frac": 1.0,
                "w_std": 0.0,
                "n_samples": 0.0,
                "n_samples_total": float(n_total),
                "n_time_strata": 0.0,
            }

        # Generator device must match ``device=`` in randperm (cuda tensors need cuda generator).
        gen = torch.Generator(device=Z_det.device)
        gen.manual_seed(int(outer_seed))
        # The CTD reference constructs the product-of-marginals surrogate by
        # shuffling treatments across patients independently at every time.
        # Mixing all patient-time rows lets WeightNet learn temporal imbalance.
        perm = _stratified_permutation(time_all, generator=gen)
        joint_rep = torch.cat([Z_det, A_all], dim=-1)
        marginal_rep = torch.cat([Z_det, A_all[perm]], dim=-1)

        chunk = max(1, int(train_batch_size))
        if align_mode == "sinkhorn":
            align_pre = self._mean_align_over_time(
                joint_rep,
                marginal_rep,
                time_all,
                sinkhorn_blur=sinkhorn_blur,
            )
        else:
            align_pre = self._mean_align_on_chunks(
                joint_rep,
                marginal_rep,
                align_mode=align_mode,
                sinkhorn_blur=sinkhorn_blur,
                chunk_size=chunk,
                strata=time_all,
            )

        self.weight_net.train()
        n_epochs = max(1, int(e_epochs))
        optimizer_steps = 0
        if align_mode == "sinkhorn":
            time_groups = [
                torch.nonzero(time_all == time_value, as_tuple=False).reshape(-1)
                for time_value in torch.unique(time_all, sorted=True)
            ]
            time_groups = [group for group in time_groups if group.numel() >= 2]
            for _ in range(n_epochs):
                optimizer_w.zero_grad(set_to_none=True)
                for group in time_groups:
                    logits = self.weight_net(joint_rep[group])
                    loss_align = alignment_loss_from_representations(
                        joint_rep[group],
                        marginal_rep[group],
                        logits,
                        align_mode,
                        sinkhorn_blur,
                    )
                    (loss_align / float(len(time_groups))).backward()
                if w_clip is not None and w_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.weight_net.parameters(), max_norm=float(w_clip)
                    )
                optimizer_w.step()
                optimizer_steps += 1
        else:
            for _ in range(n_epochs):
                unique_times = torch.unique(time_all, sorted=True)
                time_order = unique_times[
                    torch.randperm(unique_times.numel(), device=Z_det.device)
                ]
                for time_value in time_order:
                    group = torch.nonzero(
                        time_all == time_value, as_tuple=False
                    ).reshape(-1)
                    group = group[torch.randperm(group.numel(), device=Z_det.device)]
                    for start in range(0, group.numel(), chunk):
                        idx = group[start : start + chunk]
                        if idx.numel() < 2:
                            continue
                        joint_b = joint_rep[idx]
                        marg_b = marginal_rep[idx]
                        optimizer_w.zero_grad(set_to_none=True)
                        logits = self.weight_net(joint_b)
                        loss_align = alignment_loss_from_representations(
                            joint_b,
                            marg_b,
                            logits,
                            align_mode,
                            sinkhorn_blur,
                        )
                        loss_align.backward()
                        if w_clip is not None and w_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.weight_net.parameters(), max_norm=float(w_clip)
                            )
                        optimizer_w.step()
                        optimizer_steps += 1

        self.weight_net.eval()
        if align_mode == "sinkhorn":
            align_post = self._mean_align_over_time(
                joint_rep,
                marginal_rep,
                time_all,
                sinkhorn_blur=sinkhorn_blur,
            )
        else:
            align_post = self._mean_align_on_chunks(
                joint_rep,
                marginal_rep,
                align_mode=align_mode,
                sinkhorn_blur=sinkhorn_blur,
                chunk_size=chunk,
                strata=time_all,
            )

        with torch.no_grad():
            logits_all = self.weight_net(joint_rep)
        weight_diagnostics = _weight_diagnostics(logits_all, active_all)

        return {
            "align_pre": align_pre,
            "align_post": align_post,
            **weight_diagnostics,
            "n_samples": float(n),
            "n_samples_total": float(n_total),
            "n_time_strata": float(torch.unique(time_all).numel()),
            "e_optimizer_steps": float(optimizer_steps),
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
        active_t = last_valid_mask(H_t["active_entries"])

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
