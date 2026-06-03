"""Density-ratio weighting for IQL (NeurIPS 2023 DW-IQL style)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_EPS = 1e-8


def _init_linear(layer: nn.Linear, gain: float = 0.01) -> None:
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class _ScalarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        dims = [input_dim, *([hidden_dim] * n_hidden), 1]
        layers = []
        for i in range(len(dims) - 2):
            lin = nn.Linear(dims[i], dims[i + 1])
            _init_linear(lin)
            layers.extend([lin, nn.ReLU()])
        out = nn.Linear(dims[-2], dims[-1])
        _init_linear(out, gain=0.01)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _stable_mean_one_from_logw(log_w: torch.Tensor) -> torch.Tensor:
    """Numerically stable mean-one weights for WeightNet training (grad through mean)."""
    log_w = log_w.reshape(-1)
    w = torch.exp(log_w - log_w.max().detach())
    return w / (w.mean() + _EPS)


def _make_iql_safe_weight(
    train_weight: torch.Tensor,
    weight_min: float,
    weight_max: float,
) -> torch.Tensor:
    """Detached, clamped, renormalized weights for IQL losses only."""
    w = train_weight.detach().clamp(min=weight_min, max=weight_max)
    return w / (w.mean() + _EPS)


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"Non-finite values in {name}: {bad} / {tensor.numel()} elements")


class DensityRatioWeightNet(nn.Module):
    """
    Learns transition-level density ratios via state and state-action logits.
    Supports two modes:
      - "reference": paper-consistent decomposition with separated w(s), w(s,a)
                     and unified weighting semantics across return/flow/KL.
      - "cip": legacy mean-one exp weighting used in this repository.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        mode: str = "reference",
        weight_temp: float = 1.0,
        clip_ratio: float = 0.2,
        flow_discount: float = 1.0,
        use_done_mask: bool = True,
        weight_min: float = 0.05,
        weight_max: float = 10.0,
    ):
        super().__init__()
        self.state_net = _ScalarMLP(state_dim, hidden_dim, n_hidden)
        self.sa_net = _ScalarMLP(state_dim + action_dim, hidden_dim, n_hidden)
        self.mode = str(mode).lower()
        self.weight_temp = float(weight_temp)
        self.clip_ratio = float(clip_ratio)
        self.flow_discount = float(flow_discount)
        self.use_done_mask = bool(use_done_mask)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        phi_s = self.state_net(state)
        psi_sa = self.sa_net(torch.cat([state, action], dim=-1))
        phi_next = self.state_net(next_state)

        log_w_s = phi_s
        log_w_sa = phi_s + psi_sa
        log_w_next = phi_next

        if self.mode == "reference":
            lo = -(1.0 + self.clip_ratio)
            hi = 1.0 + self.clip_ratio
            s_clipped = phi_s.clamp(min=lo, max=hi)
            sa_clipped = psi_sa.clamp(min=lo, max=hi)
            next_s_clipped = phi_next.clamp(min=lo, max=hi)

            # Shared log-space semantics:
            #   log w(s)     ~ s_clipped / T
            #   log ratio(a|s) ~ sa_clipped / T
            #   log w(s,a)   = log w(s) + log ratio(a|s)
            t = max(self.weight_temp, _EPS)
            log_w_s_ref = s_clipped / t
            log_ratio_ref = sa_clipped / t
            log_w_sa_ref = log_w_s_ref + log_ratio_ref
            log_w_next_ref = next_s_clipped / t

            # Mean-one normalization keeps one consistent weighting family
            # for reward / flow / KL and matches existing IQL mean-loss usage.
            w_s_train = _stable_mean_one_from_logw(log_w_s_ref)
            w_sa_train = _stable_mean_one_from_logw(log_w_sa_ref)
            w_next_train = _stable_mean_one_from_logw(log_w_next_ref)
            w_s_iql = _make_iql_safe_weight(w_s_train, self.weight_min, self.weight_max)
            w_sa_iql = _make_iql_safe_weight(w_sa_train, self.weight_min, self.weight_max)
            clipped_ratio = s_clipped + sa_clipped
        elif self.mode == "cip":
            w_s_train = _stable_mean_one_from_logw(log_w_s)
            w_sa_train = _stable_mean_one_from_logw(log_w_sa)
            w_next_train = _stable_mean_one_from_logw(log_w_next)

            w_s_iql = _make_iql_safe_weight(w_s_train, self.weight_min, self.weight_max)
            w_sa_iql = _make_iql_safe_weight(w_sa_train, self.weight_min, self.weight_max)
            s_clipped = phi_s
            sa_clipped = psi_sa
            clipped_ratio = log_w_sa
        else:
            raise ValueError(f"Unknown DW mode: {self.mode!r}")

        return {
            "w_s_train": w_s_train,
            "w_sa_train": w_sa_train,
            "w_next_train": w_next_train,
            "w_s_iql": w_s_iql,
            "w_sa_iql": w_sa_iql,
            "log_w_s": log_w_s,
            "log_w_sa": log_w_sa,
            "log_w_next": log_w_next,
            "s_ratio": phi_s,
            "sa_ratio": psi_sa,
            "s_clipped_ratio": s_clipped,
            "sa_clipped_ratio": sa_clipped,
            "clipped_ratio": clipped_ratio,
        }

    def compute_dw_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: Optional[torch.Tensor] = None,
        lambda_flow: float = 1.0,
        lambda_kl: float = 0.01,
        center_reward: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weights = self.forward(state, action, next_state)
        w_sa_train = weights["w_sa_train"]
        w_next_train = weights["w_next_train"]
        w_sa_iql = weights["w_sa_iql"]

        r = reward.reshape(-1)
        if center_reward:
            r = r - r.mean().detach()

        if self.mode == "reference":
            # Unified weighting semantics:
            # - return term uses transition weight w(s,a)
            # - flow compares w(s') and w(s,a) in the same normalized family
            # - KL regularizes that same transition weight
            return_term = -(w_sa_train * r).mean()
            flow_sq = (self.flow_discount * w_sa_train - w_next_train) ** 2
            if done is None or not self.use_done_mask:
                flow_penalty = flow_sq.mean()
            else:
                mask = (1.0 - done.reshape(-1).float()).clamp(min=0.0, max=1.0)
                denom = mask.sum() + _EPS
                flow_penalty = (flow_sq * mask).sum() / denom
            kl_penalty = (w_sa_train * torch.log(w_sa_train + _EPS)).mean()
        else:
            return_term = -(w_sa_train * r).mean()
            flow_sq = (w_next_train - w_sa_train) ** 2
            if done is None:
                flow_penalty = flow_sq.mean()
            else:
                mask = (1.0 - done.reshape(-1).float()).clamp(min=0.0, max=1.0)
                denom = mask.sum() + _EPS
                flow_penalty = (flow_sq * mask).sum() / denom
            kl_penalty = (w_sa_train * torch.log(w_sa_train + _EPS)).mean()
        dw_loss = return_term + float(lambda_flow) * flow_penalty + float(lambda_kl) * kl_penalty

        _check_finite("dw_loss", dw_loss.unsqueeze(0))
        _check_finite("w_sa_train", w_sa_train)
        _check_finite("w_next_train", w_next_train)
        _check_finite("w_sa_iql", w_sa_iql)

        log_w_sa = weights["log_w_sa"].reshape(-1).detach()
        stats = weight_statistics(w_sa_iql, reward=r)
        stats.update(
            {
                "dw_loss": float(dw_loss.item()),
                "dw_return_term": float(return_term.item()),
                "dw_flow_penalty": float(flow_penalty.item()),
                "dw_kl_penalty": float(kl_penalty.item()),
                "dw_w_s_train_mean": float(weights["w_s_train"].mean().item()),
                "dw_w_sa_train_mean": float(w_sa_train.mean().item()),
                "dw_w_next_train_mean": float(w_next_train.mean().item()),
                "dw_w_s_iql_mean": float(weights["w_s_iql"].mean().item()),
                "dw_w_sa_iql_mean": float(w_sa_iql.mean().item()),
                "dw_log_w_sa_mean": float(log_w_sa.mean().item()),
                "dw_log_w_sa_std": float(log_w_sa.std(unbiased=False).item())
                if log_w_sa.numel() > 1
                else 0.0,
                "dw_log_w_sa_min": float(log_w_sa.min().item()),
                "dw_log_w_sa_max": float(log_w_sa.max().item()),
            }
        )
        return dw_loss, stats


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).detach().float()
    b = b.reshape(-1).detach().float()
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(torch.sqrt((a * a).sum() * (b * b).sum()).item())
    if denom <= 0.0:
        return 0.0
    return float((a * b).sum().item() / denom)


def weight_statistics(
    weights: torch.Tensor,
    reward: torch.Tensor | None = None,
    td_error: torch.Tensor | None = None,
) -> Dict[str, float]:
    w = weights.reshape(-1).detach().float()
    n = max(int(w.numel()), 1)
    ess = float((w.sum() ** 2) / ((w * w).sum() + _EPS))
    out = {
        "weight_mean": float(w.mean().item()),
        "weight_std": float(w.std(unbiased=False).item()) if w.numel() > 1 else 0.0,
        "weight_min": float(w.min().item()) if w.numel() else 0.0,
        "weight_max": float(w.max().item()) if w.numel() else 0.0,
        "weight_p95": float(torch.quantile(w, 0.95).item()) if w.numel() else 0.0,
        "weight_p99": float(torch.quantile(w, 0.99).item()) if w.numel() else 0.0,
        "weight_ess": ess / float(n),
    }
    if reward is not None:
        out["corr_weight_reward"] = _safe_corr(w, reward)
    if td_error is not None:
        out["corr_weight_td_error"] = _safe_corr(w, td_error)
    return out
