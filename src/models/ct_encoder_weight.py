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
    """Legacy single-stratum diagnostics from raw log-weights."""
    b = logits_w.numel()
    w = F.softmax(logits_w.reshape(-1), dim=0) * float(b)
    if active_mask is not None and active_mask.numel() == b:
        act = active_mask.reshape(-1) > 0.5
        w = w[act]
    return _weight_diagnostics_from_weights(w)


def _weight_diagnostics_from_weights(w: torch.Tensor) -> Dict[str, float]:
    """Weight concentration diagnostics for already-normalized weights."""
    w_det = w.reshape(-1).detach()
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


def _cap_mean_one_weights(w: torch.Tensor, weight_max: Optional[float]) -> torch.Tensor:
    """Cap positive weights while preserving their sum exactly when feasible."""
    if weight_max is None or float(weight_max) <= 0:
        return w
    cap = float(weight_max)
    if cap < 1.0:
        raise ValueError("weight_max must be >= 1.0 when enabled.")
    if cap == 1.0:
        return torch.ones_like(w)

    target = float(w.numel())
    lo = w.new_zeros(())
    hi = w.new_ones(())
    while float(torch.clamp(w * hi, max=cap).sum().detach()) < target:
        hi = hi * 2.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        if float(torch.clamp(w * mid, max=cap).sum().detach()) < target:
            lo = mid
        else:
            hi = mid
    return torch.clamp(w * hi, max=cap)


def normalize_log_weights_by_stratum(
    logits_w: torch.Tensor,
    strata: torch.Tensor,
    *,
    weight_max: Optional[float] = None,
) -> torch.Tensor:
    """Apply exactly one softmax within each decision-time stratum."""
    logits = logits_w.reshape(-1)
    strata = strata.reshape(-1).to(device=logits.device)
    if logits.numel() != strata.numel():
        raise ValueError(
            f"logits and strata must have equal length, got {logits.numel()} and {strata.numel()}"
        )
    weights = torch.empty_like(logits)
    for value in torch.unique(strata, sorted=True):
        idx = torch.nonzero(strata == value, as_tuple=False).reshape(-1)
        group_w = F.softmax(logits[idx], dim=0) * float(idx.numel())
        weights[idx] = _cap_mean_one_weights(group_w, weight_max)
    return weights


def _stratified_weight_diagnostics(
    weights: torch.Tensor,
    strata: torch.Tensor,
) -> Dict[str, float]:
    """Summarize ESS and spread within decision-time strata."""
    ess_values, std_values = [], []
    for value in torch.unique(strata, sorted=True):
        idx = torch.nonzero(strata == value, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        diagnostics = _weight_diagnostics_from_weights(weights[idx])
        ess_values.append(diagnostics["w_ess_frac"])
        std_values.append(diagnostics["w_std"])
    if not ess_values:
        return {
            "w_time_ess_mean": 1.0,
            "w_time_ess_min": 1.0,
            "w_time_std_mean": 0.0,
            "w_time_std_max": 0.0,
        }
    return {
        "w_time_ess_mean": float(sum(ess_values) / len(ess_values)),
        "w_time_ess_min": float(min(ess_values)),
        "w_time_std_mean": float(sum(std_values) / len(std_values)),
        "w_time_std_max": float(max(std_values)),
    }


def _standardized_za_dependence(
    z: torch.Tensor,
    a: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute weighted cross-correlation between latent state and action."""
    w = weights.reshape(-1, 1)
    denom = w.sum().clamp_min(1e-8)
    z_mean = (w * z).sum(0, keepdim=True) / denom
    a_mean = (w * a).sum(0, keepdim=True) / denom
    z_centered = z - z_mean
    a_centered = a - a_mean
    z_std = torch.sqrt((w * z_centered.square()).sum(0) / denom).clamp_min(1e-6)
    a_std = torch.sqrt((w * a_centered.square()).sum(0) / denom).clamp_min(1e-6)
    cross = (w * z_centered).transpose(0, 1) @ a_centered / denom
    corr = cross / (z_std[:, None] * a_std[None, :])
    return corr.abs().mean()


def _stratified_za_dependence(
    z: torch.Tensor,
    a: torch.Tensor,
    weights: torch.Tensor,
    strata: torch.Tensor,
) -> float:
    values = []
    for value in torch.unique(strata, sorted=True):
        idx = torch.nonzero(strata == value, as_tuple=False).reshape(-1)
        if idx.numel() < 2:
            continue
        values.append(float(_standardized_za_dependence(z[idx], a[idx], weights[idx]).detach()))
    return float(sum(values) / max(len(values), 1))


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
        self.fixed_transition_weights: Dict[Tuple[int, int], float] = {}

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
        strata: Optional[torch.Tensor] = None,
        weight_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits_w [B], w [B] normalized once within each supplied time stratum.
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
        if strata is None:
            raise ValueError("Non-uniform WeightNet inference requires decision-time strata.")
        w = normalize_log_weights_by_stratum(
            logits_w, strata, weight_max=weight_max
        )
        return logits_w, w

    @torch.no_grad()
    def _encode_full_dataset(
        self,
        loader: DataLoader,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all E-step transitions, retaining patient and decision-time keys."""
        self.eval()
        z_parts, a_parts, act_parts, patient_parts, time_parts = [], [], [], [], []
        row_offset = 0
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
            patient_index = batch.get("patient_index")
            if patient_index is None:
                patient_index = torch.arange(
                    row_offset,
                    row_offset + Z_t.size(0),
                    device=device,
                    dtype=torch.long,
                )
            else:
                patient_index = patient_index.to(
                    device=device, dtype=torch.long
                ).reshape(-1)
            row_offset += Z_t.size(0)
            z_parts.append(Z_t.detach())
            a_parts.append(A_t)
            act_parts.append(active_t)
            patient_parts.append(patient_index)
            time_parts.append(time_index)
        return (
            torch.cat(z_parts, 0),
            torch.cat(a_parts, 0),
            torch.cat(act_parts, 0),
            torch.cat(patient_parts, 0),
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
        align_mode: str,
        sinkhorn_blur: float,
    ) -> float:
        """Mean full-stratum alignment loss over decision times."""
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
                align_mode,
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
        weight_max: Optional[float] = None,
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

        Z_det, A_all, active_all, patient_all, time_all = self._encode_full_dataset(
            loader, device
        )
        n_total = Z_det.size(0)
        valid = active_all > 0.5
        Z_det = Z_det[valid]
        A_all = A_all[valid]
        active_all = active_all[valid]
        patient_all = patient_all[valid]
        time_all = time_all[valid]
        n = Z_det.size(0)
        if n == 0:
            self.fixed_transition_weights = {}
            return {
                "align_pre": 0.0,
                "align_post": 0.0,
                "w_ess_frac": 1.0,
                "w_std": 0.0,
                "n_samples": 0.0,
                "n_samples_total": float(n_total),
                "n_time_strata": 0.0,
                "align_reduction": 0.0,
                "align_relative_reduction": 0.0,
                "za_dependence_pre": 0.0,
                "za_dependence_post": 0.0,
            }

        # Generator device must match ``device=`` in randperm (cuda tensors need cuda generator).
        gen = torch.Generator(device=Z_det.device)
        gen.manual_seed(int(outer_seed))
        # The CTD reference constructs the product-of-marginals surrogate by
        # shuffling treatments across patients independently at every time.
        # Keeping time strata separate preserves that conditional contract.
        perm = _stratified_permutation(time_all, generator=gen)
        joint_rep = torch.cat([Z_det, A_all], dim=-1)
        marginal_rep = torch.cat([Z_det, A_all[perm]], dim=-1)

        align_pre = self._mean_align_over_time(
            joint_rep,
            marginal_rep,
            time_all,
            align_mode=align_mode,
            sinkhorn_blur=sinkhorn_blur,
        )

        self.weight_net.train()
        n_epochs = max(1, int(e_epochs))
        optimizer_steps = 0
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

        self.weight_net.eval()
        align_post = self._mean_align_over_time(
            joint_rep,
            marginal_rep,
            time_all,
            align_mode=align_mode,
            sinkhorn_blur=sinkhorn_blur,
        )

        with torch.no_grad():
            logits_all = self.weight_net(joint_rep)
            fixed_weights = normalize_log_weights_by_stratum(
                logits_all,
                time_all,
                weight_max=weight_max,
            )
        self.fixed_transition_weights = {
            (int(patient), int(time)): float(weight)
            for patient, time, weight in zip(
                patient_all.detach().cpu().tolist(),
                time_all.detach().cpu().tolist(),
                fixed_weights.detach().cpu().tolist(),
            )
        }
        if len(self.fixed_transition_weights) != n:
            raise ValueError(
                "E-step patient/time keys are not unique; cannot attach fixed replay weights."
            )

        weight_diagnostics = _weight_diagnostics_from_weights(fixed_weights)
        time_diagnostics = _stratified_weight_diagnostics(fixed_weights, time_all)
        uniform_weights = torch.ones_like(fixed_weights)
        za_dependence_pre = _stratified_za_dependence(
            Z_det, A_all, uniform_weights, time_all
        )
        za_dependence_post = _stratified_za_dependence(
            Z_det, A_all, fixed_weights, time_all
        )
        align_reduction = float(align_pre - align_post)
        align_relative_reduction = align_reduction / max(abs(float(align_pre)), 1e-12)

        return {
            "align_pre": align_pre,
            "align_post": align_post,
            "align_reduction": align_reduction,
            "align_relative_reduction": align_relative_reduction,
            **weight_diagnostics,
            **time_diagnostics,
            "za_dependence_pre": za_dependence_pre,
            "za_dependence_post": za_dependence_post,
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
