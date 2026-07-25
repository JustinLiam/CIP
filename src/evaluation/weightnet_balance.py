"""Held-out representation-intervention balance metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _normalized_weights(weights: torch.Tensor) -> torch.Tensor:
    w = weights.reshape(-1).double().clamp_min(0.0)
    total = w.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return torch.full_like(w, 1.0 / max(w.numel(), 1))
    return w / total


def _weighted_center(matrix: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    row_mean = matrix @ probability
    grand_mean = probability @ row_mean
    return matrix - row_mean[:, None] - row_mean[None, :] + grand_mean


def weighted_distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Biased weighted distance correlation for continuous multivariate inputs."""
    if x.size(0) < 2:
        return 0.0
    x = x.reshape(x.size(0), -1).double()
    y = y.reshape(y.size(0), -1).double()
    p = _normalized_weights(weights)
    pair_weight = p[:, None] * p[None, :]
    x_centered = _weighted_center(torch.cdist(x, x), p)
    y_centered = _weighted_center(torch.cdist(y, y), p)
    covariance_sq = (pair_weight * x_centered * y_centered).sum().clamp_min(0.0)
    x_variance_sq = (pair_weight * x_centered.square()).sum().clamp_min(0.0)
    y_variance_sq = (pair_weight * y_centered.square()).sum().clamp_min(0.0)
    denominator = torch.sqrt(x_variance_sq * y_variance_sq)
    if float(denominator) <= 1.0e-15:
        return 0.0
    correlation_sq = (covariance_sq / denominator).clamp(0.0, 1.0)
    return float(torch.sqrt(correlation_sq))


def _rbf_kernel(x: torch.Tensor) -> torch.Tensor:
    squared_distance = torch.cdist(x.double(), x.double()).square()
    positive = squared_distance[squared_distance > 0]
    bandwidth_sq = positive.median() if positive.numel() else squared_distance.new_tensor(1.0)
    bandwidth_sq = bandwidth_sq.clamp_min(1.0e-12)
    return torch.exp(-squared_distance / (2.0 * bandwidth_sq))


def weighted_normalized_hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Normalized weighted HSIC with per-stratum median RBF bandwidths."""
    if x.size(0) < 2:
        return 0.0
    p = _normalized_weights(weights)
    pair_weight = p[:, None] * p[None, :]
    k_centered = _weighted_center(_rbf_kernel(x.reshape(x.size(0), -1)), p)
    l_centered = _weighted_center(_rbf_kernel(y.reshape(y.size(0), -1)), p)
    cross = (pair_weight * k_centered * l_centered).sum()
    k_norm = (pair_weight * k_centered.square()).sum().clamp_min(0.0)
    l_norm = (pair_weight * l_centered.square()).sum().clamp_min(0.0)
    denominator = torch.sqrt(k_norm * l_norm)
    if float(denominator) <= 1.0e-15:
        return 0.0
    return float((cross / denominator).clamp(0.0, 1.0))


def weighted_mean_absolute_cross_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Mean absolute weighted Pearson correlation across all feature pairs."""
    p = _normalized_weights(weights)[:, None]
    x = x.reshape(x.size(0), -1).double()
    y = y.reshape(y.size(0), -1).double()
    x_centered = x - (p * x).sum(0, keepdim=True)
    y_centered = y - (p * y).sum(0, keepdim=True)
    x_std = torch.sqrt((p * x_centered.square()).sum(0)).clamp_min(1.0e-12)
    y_std = torch.sqrt((p * y_centered.square()).sum(0)).clamp_min(1.0e-12)
    covariance = (p * x_centered).transpose(0, 1) @ y_centered
    correlation = covariance / (x_std[:, None] * y_std[None, :])
    return float(correlation.abs().mean())


def _dependence_metrics(
    z: torch.Tensor,
    action: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, float]:
    return {
        "distance_correlation": weighted_distance_correlation(z, action, weights),
        "normalized_hsic": weighted_normalized_hsic(z, action, weights),
        "mean_abs_cross_correlation": weighted_mean_absolute_cross_correlation(
            z, action, weights
        ),
    }


def stratified_balance_metrics(
    z: torch.Tensor,
    action: torch.Tensor,
    weights: torch.Tensor,
    time_index: torch.Tensor,
    *,
    min_samples: int = 8,
    shuffle_seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Compute equal-time-macro balance metrics under uniform/learned/shuffled weights."""
    z = z.detach().cpu()
    action = action.detach().cpu()
    weights = weights.detach().cpu().reshape(-1)
    time_index = time_index.detach().cpu().reshape(-1)
    uniform = torch.ones_like(weights)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    rows: List[Dict[str, float]] = []

    for time_value in torch.unique(time_index, sorted=True):
        idx = torch.nonzero(time_index == time_value, as_tuple=False).reshape(-1)
        if idx.numel() < int(min_samples):
            continue
        learned_w = weights[idx]
        shuffled_w = learned_w[torch.randperm(idx.numel(), generator=generator)]
        row: Dict[str, float] = {
            "time": float(time_value),
            "n": float(idx.numel()),
        }
        for label, current_weights in (
            ("uniform", uniform[idx]),
            ("weighted", learned_w),
            ("shuffled", shuffled_w),
        ):
            metrics = _dependence_metrics(z[idx], action[idx], current_weights)
            for key, value in metrics.items():
                row[f"{label}_{key}"] = value
        weight_sum = learned_w.sum()
        row["ess_fraction"] = float(
            (weight_sum.square() / learned_w.square().sum().clamp_min(1.0e-12))
            / float(idx.numel())
        )
        row["weight_std"] = float(learned_w.double().std(unbiased=False))
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No decision-time stratum has at least min_samples={min_samples}."
        )

    aggregate: Dict[str, float] = {
        "n_samples": float(sum(row["n"] for row in rows)),
        "n_time_strata": float(len(rows)),
        "min_samples_per_stratum": float(min_samples),
        "ess_fraction_mean": float(
            sum(row["ess_fraction"] for row in rows) / len(rows)
        ),
        "ess_fraction_min": float(min(row["ess_fraction"] for row in rows)),
        "weight_std_mean": float(
            sum(row["weight_std"] for row in rows) / len(rows)
        ),
        "weight_std_max": float(max(row["weight_std"] for row in rows)),
    }
    metric_names = (
        "distance_correlation",
        "normalized_hsic",
        "mean_abs_cross_correlation",
    )
    for metric in metric_names:
        for label in ("uniform", "weighted", "shuffled"):
            key = f"{label}_{metric}"
            aggregate[key] = float(sum(row[key] for row in rows) / len(rows))
        uniform_value = aggregate[f"uniform_{metric}"]
        weighted_value = aggregate[f"weighted_{metric}"]
        aggregate[f"relative_reduction_{metric}"] = (
            (uniform_value - weighted_value) / max(abs(uniform_value), 1.0e-12)
        )
    return aggregate, rows
