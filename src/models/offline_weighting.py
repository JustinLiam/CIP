"""
Periodic offline WeightNet training for CT deconfounding (CTD-NKO-style).

Learns dataset/time-level balancing weights on frozen encoder latents:
  joint   = concat([Z_t, A_t])
  marginal = concat([Z_t, A_perm_t])  (A shuffled within time step when possible)

Default input to OfflineWeightNet is [Z, A] (joint sample features), matching the
balancing target P_w(Z, A) vs P(Z)P(A).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.ct_transition_dataset import CTTransitionDataset, collate_ct_batch
from src.utils.utils import compute_mmd_weighted

logger = logging.getLogger(__name__)


class OfflineWeightNet(nn.Module):
    """
    MLP on concat([Z_t, A_t]) -> scalar **raw logit**.

    No sigmoid / softplus on output; all mass normalization happens in
    :func:`normalize_weights` during alignment and when building the weight table.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def normalize_weights(
    logits: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    mode: str = "softmax_time",
) -> torch.Tensor:
    """
    Normalize raw logits into non-negative weights [N, 1].

    ``softmax_time``: per ``group_ids`` (e.g. time_id), softmax then multiply by
    group size so within-group mean ≈ 1.
    ``mean_one``: global weights scaled to mean 1.
    """
    w = logits.reshape(-1)
    n = w.numel()
    if n == 0:
        return w.new_zeros(0, 1)

    if mode == "mean_one":
        w_pos = F.softmax(w, dim=0) * float(n)
        return w_pos.reshape(-1, 1)

    if mode != "softmax_time":
        raise ValueError(f"Unknown ct_weight_normalize mode: {mode}")

    if group_ids is None:
        w_pos = F.softmax(w, dim=0) * float(n)
        return w_pos.reshape(-1, 1)

    gids = group_ids.reshape(-1).long()
    out = torch.zeros_like(w)
    for gid in gids.unique().tolist():
        mask = gids == int(gid)
        idx = torch.where(mask)[0]
        lg = w[idx]
        n_g = lg.numel()
        if n_g <= 0:
            continue
        out[idx] = F.softmax(lg, dim=0) * float(n_g)
    return out.reshape(-1, 1)


def compute_weighted_alignment_loss(
    joint: torch.Tensor,
    marginal: torch.Tensor,
    weight_logits: torch.Tensor,
    metric: str = "sinkhorn",
    blur: float = 0.01,
    group_ids: Optional[torch.Tensor] = None,
    normalize_mode: str = "softmax_time",
) -> torch.Tensor:
    """
    Weighted alignment between joint (source, weighted) and marginal (target, uniform).

    joint, marginal: [N, D]
    weight_logits: [N] or [N, 1] raw logits from OfflineWeightNet
    """
    if joint.shape != marginal.shape:
        raise ValueError(
            f"joint and marginal must match shape, got {tuple(joint.shape)} vs {tuple(marginal.shape)}"
        )
    w = normalize_weights(weight_logits, group_ids=group_ids, mode=normalize_mode).reshape(-1)
    n = joint.size(0)
    if n == 0:
        return joint.new_zeros(())

    if metric == "mmd":
        return compute_mmd_weighted(joint, marginal, w)

    if metric == "sinkhorn":
        try:
            from src.utils.utils import compute_weighted_wasserstein_joint_marginal_flat

            return compute_weighted_wasserstein_joint_marginal_flat(
                joint, marginal, w, blur=blur
            )
        except (ImportError, Exception) as exc:
            logger.warning(
                "Sinkhorn alignment failed (%s); falling back to MMD.", exc
            )
            return compute_mmd_weighted(joint, marginal, w)

    raise ValueError(f"Unknown alignment metric: {metric}")


def _time_level_weight_diagnostics(
    weights: torch.Tensor,
    time_ids: torch.Tensor,
) -> Dict[str, float]:
    """Per-time-step ESS and global low-weight fractions (softmax_time collapse check)."""
    w = weights.reshape(-1).detach().float().cpu()
    tids = time_ids.reshape(-1).long().cpu()
    n = int(w.numel())
    if n == 0:
        return {
            "min_ess_by_time": 1.0,
            "mean_ess_by_time": 1.0,
            "low_weight_frac_001": 0.0,
            "low_weight_frac_005": 0.0,
        }
    ess_list = []
    for gid in tids.unique().tolist():
        wg = w[tids == int(gid)]
        n_g = int(wg.numel())
        if n_g <= 0:
            continue
        s = float(wg.sum())
        sq = float((wg * wg).sum())
        ess_list.append((s * s) / (sq * n_g + 1e-12))
    if not ess_list:
        ess_list = [1.0]
    return {
        "min_ess_by_time": float(min(ess_list)),
        "mean_ess_by_time": float(sum(ess_list) / len(ess_list)),
        "low_weight_frac_001": float((w < 0.01).sum()) / float(n),
        "low_weight_frac_005": float((w < 0.05).sum()) / float(n),
    }


def _weight_diagnostics(weights: torch.Tensor, time_ids: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Summary stats for normalized weights [N] or [N, 1]."""
    w = weights.reshape(-1).detach().float().cpu()
    n = max(int(w.numel()), 1)
    if w.numel() == 0:
        return {
            "weight_mean": 0.0,
            "weight_std": 0.0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "weight_p01": 0.0,
            "weight_p05": 0.0,
            "weight_p50": 0.0,
            "weight_p95": 0.0,
            "weight_p99": 0.0,
            "ess_frac": 1.0,
            "min_ess_by_time": 1.0,
            "mean_ess_by_time": 1.0,
            "low_weight_frac_001": 0.0,
            "low_weight_frac_005": 0.0,
        }
    w_sum = float(w.sum())
    w_sq = float((w * w).sum())
    ess = (w_sum * w_sum) / (w_sq * n + 1e-12)
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99])
    pct = torch.quantile(w, qs).tolist()
    out = {
        "weight_mean": float(w.mean()),
        "weight_std": float(w.std(unbiased=False)),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_p01": pct[0],
        "weight_p05": pct[1],
        "weight_p50": pct[2],
        "weight_p95": pct[3],
        "weight_p99": pct[4],
        "ess_frac": float(ess),
    }
    if time_ids is not None:
        out.update(_time_level_weight_diagnostics(weights, time_ids))
    return out


def standardize_joint_marginal(
    joint: torch.Tensor,
    marginal: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z-score using mean/std from **joint** only; apply the same affine map to marginal.
    Reduces scale mismatch between Z and A blocks before Sinkhorn/MMD.
    """
    mean = joint.mean(dim=0, keepdim=True)
    std = joint.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (joint - mean) / std, (marginal - mean) / std, mean.squeeze(0), std.squeeze(0)


def _is_binary_treatment(A: torch.Tensor, atol: float = 1e-4) -> bool:
    """Heuristic: all treatment entries near {0, 1}."""
    flat = A.reshape(-1).float()
    if flat.numel() == 0:
        return False
    near0 = (flat.abs() < atol) | ((flat - 1.0).abs() < atol)
    return bool(near0.all())


def _weighted_corr(x: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if w is None:
        w = torch.ones_like(x)
    w = w.reshape(-1).float().clamp_min(0.0)
    s = w.sum() + 1e-12
    w = w / s
    mx = (w * x).sum()
    my = (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx).pow(2)).sum().clamp_min(1e-12)
    vy = (w * (y - my).pow(2)).sum().clamp_min(1e-12)
    return float((cov / torch.sqrt(vx * vy)).item())


def _z_a_correlation_summary(
    Z: torch.Tensor,
    A: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    abs_corrs = []
    for j in range(Z.size(1)):
        for k in range(A.size(1)):
            abs_corrs.append(abs(_weighted_corr(Z[:, j], A[:, k], w)))
    if not abs_corrs:
        return 0.0, 0.0
    return float(sum(abs_corrs) / len(abs_corrs)), float(max(abs_corrs))


def _weighted_lstsq_r2(Z: torch.Tensor, A: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    """Weighted multi-output R2 (A predicted from Z with bias)."""
    n = Z.size(0)
    if n < 2:
        return float("nan")
    device, dtype = Z.device, Z.dtype
    if w is None:
        w = torch.ones(n, device=device, dtype=dtype)
    w = w.reshape(-1).float().clamp_min(0.0)
    s = w.sum() + 1e-12
    w = w / s
    sw = w.sqrt().reshape(-1, 1)
    Z1 = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), Z], dim=1)
    Zw = Z1 * sw
    Aw = A * sw
    coef = torch.linalg.lstsq(Zw, Aw).solution
    pred = Z1 @ coef
    ss_res = (w.reshape(-1, 1) * (A - pred).pow(2)).sum()
    a_mean = (w.reshape(-1, 1) * A).sum(dim=0, keepdim=True)
    ss_tot = (w.reshape(-1, 1) * (A - a_mean).pow(2)).sum()
    return float((1.0 - ss_res / (ss_tot + 1e-8)).item())


def _weighted_lstsq_mse(Z: torch.Tensor, A: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    n = Z.size(0)
    if n < 2:
        return float("nan")
    device, dtype = Z.device, Z.dtype
    if w is None:
        w = torch.ones(n, device=device, dtype=dtype)
    w = w.reshape(-1).float().clamp_min(0.0)
    s = w.sum() + 1e-12
    w = w / s
    sw = w.sqrt().reshape(-1, 1)
    Z1 = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), Z], dim=1)
    coef = torch.linalg.lstsq(Z1 * sw, A * sw).solution
    pred = Z1 @ coef
    return float((w.reshape(-1, 1) * (A - pred).pow(2)).sum().item())


def _weighted_binary_accuracy(Z: torch.Tensor, A: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    """Per-dim linear scores -> sigmoid; mean weighted accuracy across treatment dims."""
    n = Z.size(0)
    if n < 2:
        return float("nan")
    device, dtype = Z.device, Z.dtype
    if w is None:
        w = torch.ones(n, device=device, dtype=dtype)
    w = w.reshape(-1).float().clamp_min(0.0)
    s = w.sum() + 1e-12
    w = w / s
    accs = []
    for k in range(A.size(1)):
        y = A[:, k].float()
        Z1 = torch.cat([torch.ones(n, 1, device=device, dtype=dtype), Z], dim=1)
        sw = w.sqrt().reshape(-1, 1)
        coef = torch.linalg.lstsq(Z1 * sw, y.reshape(-1, 1) * sw).solution
        prob = (Z1 @ coef).sigmoid().squeeze(-1)
        pred = (prob > 0.5).float()
        accs.append((w * (pred == y).float()).sum())
    return float(torch.stack(accs).mean().item())


def compute_balance_diagnostics(
    Z: torch.Tensor,
    A: torch.Tensor,
    weights: torch.Tensor,
    *,
    treatment_mode: str = "continuous",
) -> Dict[str, float]:
    """
    Post-refresh diagnostics: treatment predictability + Z–A correlation.
    Does not affect training loss.
    """
    w = weights.reshape(-1).float()
    mode = str(treatment_mode).lower()
    use_binary = mode in ("multiclass", "multilabel", "binary") or _is_binary_treatment(A)

    out: Dict[str, float] = {}
    if use_binary:
        out["treat_metric"] = "acc"
        out["treat_pred_unweighted"] = _weighted_binary_accuracy(Z, A, None)
        out["treat_pred_weighted"] = _weighted_binary_accuracy(Z, A, w)
    else:
        out["treat_metric"] = "r2"
        out["treat_pred_unweighted"] = _weighted_lstsq_r2(Z, A, None)
        out["treat_pred_weighted"] = _weighted_lstsq_r2(Z, A, w)
        out["treat_mse_unweighted"] = _weighted_lstsq_mse(Z, A, None)
        out["treat_mse_weighted"] = _weighted_lstsq_mse(Z, A, w)

    (
        out["mean_abs_corr_unweighted"],
        out["max_abs_corr_unweighted"],
    ) = _z_a_correlation_summary(Z, A, None)
    out["mean_abs_corr_weighted"], out["max_abs_corr_weighted"] = _z_a_correlation_summary(Z, A, w)
    return out


def shuffle_actions_for_marginal(
    A: torch.Tensor,
    time_ids: torch.Tensor,
    use_time_shuffle: bool = True,
) -> torch.Tensor:
    """Build A_perm by shuffling rows within each time_id (or globally)."""
    A_perm = A.clone()
    n = A.size(0)
    if n <= 1:
        return A_perm
    tids = time_ids.reshape(-1).long()
    if use_time_shuffle:
        for gid in tids.unique().tolist():
            mask = tids == int(gid)
            idx = torch.where(mask)[0]
            if idx.numel() <= 1:
                continue
            perm_local = idx[torch.randperm(idx.numel(), device=A.device)]
            A_perm[mask] = A[perm_local]
    else:
        perm = torch.randperm(n, device=A.device)
        A_perm = A[perm]
    return A_perm


def _align_on_subset(
    joint_samples: torch.Tensor,
    marginal_samples: torch.Tensor,
    time_ids: torch.Tensor,
    weight_net: OfflineWeightNet,
    metric: str,
    blur: float,
    normalize_mode: str,
    max_n: int = 4096,
) -> float:
    """Alignment diagnostic on a subsample (Sinkhorn is costly at full N)."""
    n = joint_samples.size(0)
    if n > max_n:
        idx = torch.randperm(n, device=joint_samples.device)[:max_n]
        jb, mb, tb = joint_samples[idx], marginal_samples[idx], time_ids[idx]
    else:
        jb, mb, tb = joint_samples, marginal_samples, time_ids
    with torch.no_grad():
        logits = weight_net(jb)
        loss = compute_weighted_alignment_loss(
            jb, mb, logits, metric=metric, blur=blur, group_ids=tb, normalize_mode=normalize_mode
        )
    return float(loss.item())


def train_offline_weightnet(
    joint_samples: torch.Tensor,
    marginal_samples: torch.Tensor,
    time_ids: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    metric: str,
    blur: float,
    device: torch.device,
    normalize_mode: str = "softmax_time",
    logger_obj: Optional[logging.Logger] = None,
) -> Tuple[OfflineWeightNet, Dict[str, float]]:
    """
    Train OfflineWeightNet on global joint/marginal pairs.
    Returns trained module and diagnostics dict.
    """
    log = logger_obj or logger
    joint_samples = joint_samples.to(device)
    marginal_samples = marginal_samples.to(device)
    time_ids = time_ids.to(device)
    n, dim = joint_samples.shape
    if n == 0:
        raise ValueError("train_offline_weightnet: empty sample set")

    weight_net = OfflineWeightNet(dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(weight_net.parameters(), lr=float(lr))

    align_initial = _align_on_subset(
        joint_samples, marginal_samples, time_ids, weight_net, metric, blur, normalize_mode
    )

    dataset = TensorDataset(joint_samples, marginal_samples, time_ids)
    loader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=True, drop_last=False)

    weight_net.train()
    for _ in range(int(epochs)):
        for jb, mb, tb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = weight_net(jb)
            loss = compute_weighted_alignment_loss(
                jb,
                mb,
                logits,
                metric=metric,
                blur=blur,
                group_ids=tb,
                normalize_mode=normalize_mode,
            )
            loss.backward()
            optimizer.step()

    weight_net.eval()
    with torch.no_grad():
        logits_f = weight_net(joint_samples)
        weights_f = normalize_weights(logits_f, group_ids=time_ids, mode=normalize_mode)
        align_final = _align_on_subset(
            joint_samples, marginal_samples, time_ids, weight_net, metric, blur, normalize_mode
        )

    diag = _weight_diagnostics(weights_f, time_ids=time_ids)
    diag["align_initial"] = align_initial
    diag["align_final"] = align_final
    diag["align_drop"] = align_initial - align_final
    if abs(align_initial) > 1e-12:
        diag["align_drop_rel"] = (align_initial - align_final) / align_initial
    else:
        diag["align_drop_rel"] = 0.0
    log.debug(
        "offline_weightnet: align %.4f -> %.4f (drop %.4f)",
        align_initial,
        align_final,
        diag["align_drop"],
    )
    return weight_net, diag


@torch.no_grad()
def collect_latents_for_weight_refresh(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Freeze encoder; collect active transition latents over the full loader.

    Uses per-sample ``time_id`` to index the query step (not ``[:, -1]`` on padded batches).

    Returns dict with patient_ids, time_ids, Z, A, active (all filtered to active only).
    """
    from src.models.ct_deconfound import build_covariate_x

    model.eval()
    pids: List[torch.Tensor] = []
    tids: List[torch.Tensor] = []
    zs: List[torch.Tensor] = []
    as_: List[torch.Tensor] = []
    total_collected = 0
    active_kept = 0

    for batch in dataloader:
        H_t = {k: v.to(device) for k, v in batch["H_t"].items()}
        pid_b = batch["patient_id"].to(device)
        tid_b = batch["time_id"].to(device).long()
        B = tid_b.size(0)
        total_collected += B
        row_idx = torch.arange(B, device=device)
        active_mask = H_t["active_entries"][row_idx, tid_b, 0] > 0.5
        active_kept += int(active_mask.sum().item())
        if not active_mask.any():
            continue

        x = build_covariate_x(H_t, model.cfg)
        ct_rep = model.ct_encoder(
            x=x,
            a=H_t["prev_treatments"],
            y=H_t["prev_outputs"],
            active_entries=H_t.get("active_entries"),
            static_features=H_t.get("static_features"),
        )
        Z_seq = model.projection(ct_rep)
        Z_t = Z_seq[row_idx, tid_b]
        A_t = H_t["current_treatments"][row_idx, tid_b]

        pids.append(pid_b[active_mask].cpu())
        tids.append(tid_b[active_mask].cpu())
        zs.append(Z_t[active_mask].cpu())
        as_.append(A_t[active_mask].cpu())

    if not zs:
        empty = torch.zeros(0, dtype=torch.long)
        z_dim = getattr(model, "z_dim", 1)
        a_dim = getattr(model, "treatment_dim", 1)
        return {
            "patient_ids": empty,
            "time_ids": empty,
            "Z": torch.zeros(0, z_dim),
            "A": torch.zeros(0, a_dim),
            "active": torch.zeros(0, dtype=torch.bool),
            "total_collected": total_collected,
            "active_kept": 0,
            "inactive_dropped": total_collected,
            "unique_patients": 0,
            "unique_times": 0,
        }

    pid_cat = torch.cat(pids, dim=0)
    tid_cat = torch.cat(tids, dim=0)
    return {
        "patient_ids": pid_cat,
        "time_ids": tid_cat,
        "Z": torch.cat(zs, dim=0),
        "A": torch.cat(as_, dim=0),
        "active": torch.ones(pid_cat.numel(), dtype=torch.bool),
        "total_collected": total_collected,
        "active_kept": active_kept,
        "inactive_dropped": total_collected - active_kept,
        "unique_patients": int(pid_cat.unique().numel()),
        "unique_times": int(tid_cat.unique().numel()),
    }


def build_weight_table(
    patient_ids: torch.Tensor,
    time_ids: torch.Tensor,
    weights: torch.Tensor,
    num_patients: int,
    max_seq_len: int,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """
    Scatter per-sample weights into [num_patients, max_seq_len, 1].
    Inactive / missing entries keep ``default_weight``.
    """
    table = torch.full((num_patients, max_seq_len, 1), float(default_weight), dtype=torch.float32)
    w = weights.reshape(-1).float().cpu()
    pid = patient_ids.reshape(-1).long().cpu()
    tid = time_ids.reshape(-1).long().cpu()
    for i in range(pid.numel()):
        pi, ti = int(pid[i]), int(tid[i])
        if 0 <= pi < num_patients and 0 <= ti < max_seq_len:
            table[pi, ti, 0] = w[i]
    return table


def refresh_offline_weights_for_dataset(
    model: nn.Module,
    train_ds: CTTransitionDataset,
    device: torch.device,
    *,
    hidden_dim: int = 16,
    weight_epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 0.1,
    metric: str = "sinkhorn",
    blur: float = 0.01,
    use_time_shuffle: bool = True,
    normalize_mode: str = "softmax_time",
    num_workers: int = 0,
    cache_dir: str = "",
    epoch: int = 0,
    treatment_mode: str = "continuous",
    logger_obj: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full refresh: collect latents, train OfflineWeightNet, build weight table, attach to dataset.
    """
    log = logger_obj or logger
    # Full-dataset latent collection (separate from WeightNet training minibatch size).
    collect_bs = max(256, min(len(train_ds), 2048))
    collate_loader = DataLoader(
        train_ds,
        batch_size=collect_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ct_batch,
        drop_last=False,
    )
    collected = collect_latents_for_weight_refresh(model, collate_loader, device)
    n = collected["Z"].size(0)
    if n == 0:
        raise RuntimeError("refresh_offline_weights: no active training samples collected")

    Z = collected["Z"].to(device)
    A = collected["A"].to(device)
    time_ids = collected["time_ids"].to(device)
    patient_ids = collected["patient_ids"]

    A_perm = shuffle_actions_for_marginal(A, time_ids, use_time_shuffle=use_time_shuffle)
    joint = torch.cat([Z, A], dim=-1)
    marginal = torch.cat([Z, A_perm], dim=-1)
    joint_std, marginal_std, _, _ = standardize_joint_marginal(joint, marginal)

    diag_collect = {
        "total_collected": int(collected.get("total_collected", n)),
        "active_kept": int(collected.get("active_kept", n)),
        "inactive_dropped": int(collected.get("inactive_dropped", 0)),
        "unique_patients": int(collected.get("unique_patients", 0)),
        "unique_times": int(collected.get("unique_times", 0)),
    }

    weight_net, diag = train_offline_weightnet(
        joint_std,
        marginal_std,
        time_ids,
        hidden_dim=hidden_dim,
        epochs=weight_epochs,
        batch_size=batch_size,
        lr=lr,
        metric=metric,
        blur=blur,
        device=device,
        normalize_mode=normalize_mode,
        logger_obj=log,
    )

    with torch.no_grad():
        logits = weight_net(joint_std)
        weights = normalize_weights(logits, group_ids=time_ids, mode=normalize_mode)

    balance = compute_balance_diagnostics(Z, A, weights, treatment_mode=treatment_mode)
    time_diag = _time_level_weight_diagnostics(weights, collected["time_ids"])
    diag.update(diag_collect)
    diag.update(balance)
    diag.update(time_diag)

    n_pat = train_ds.data["current_treatments"].shape[0]
    max_len = train_ds.data["current_treatments"].shape[1]
    table = build_weight_table(patient_ids, collected["time_ids"], weights, n_pat, max_len)

    train_ds.set_weight_table(table)

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        out_file = cache_path / f"weight_table_epoch{epoch}.pt"
        torch.save(
            {
                "weight_table": table,
                "patient_ids": patient_ids,
                "time_ids": collected["time_ids"],
                "weights": weights.cpu(),
                "diagnostics": diag,
                "epoch": epoch,
            },
            out_file,
        )
        log.info("[W-refresh] saved weight table to %s", out_file)

    diag["N"] = int(n)
    return table, diag


@torch.no_grad()
def compute_val_balance_diagnostics(
    model: nn.Module,
    val_ds: CTTransitionDataset,
    device: torch.device,
    *,
    align_mode: str = "sinkhorn",
    blur: float = 0.01,
    use_time_shuffle: bool = True,
    treatment_mode: str = "continuous",
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Val-only diagnostics (w=1 on loss path). Collects val latents and reports
    alignment / treatment predictability / Z–A correlation without training weights.
    """
    collect_bs = max(256, min(len(val_ds), 2048))
    loader = DataLoader(
        val_ds,
        batch_size=collect_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ct_batch,
        drop_last=False,
    )
    collected = collect_latents_for_weight_refresh(model, loader, device)
    n = collected["Z"].size(0)
    if n == 0:
        return {}
    Z = collected["Z"].to(device)
    A = collected["A"].to(device)
    time_ids = collected["time_ids"].to(device)
    w_uniform = torch.ones(n, device=device)
    A_perm = shuffle_actions_for_marginal(A, time_ids, use_time_shuffle=use_time_shuffle)
    joint = torch.cat([Z, A], dim=-1)
    marginal = torch.cat([Z, A_perm], dim=-1)
    joint_std, marginal_std, _, _ = standardize_joint_marginal(joint, marginal)
    if align_mode == "mmd":
        val_align = float(compute_mmd_weighted(joint_std, marginal_std, w_uniform).item())
    else:
        val_align = float(
            compute_weighted_alignment_loss(
                joint_std,
                marginal_std,
                torch.zeros(n, device=device),
                metric=align_mode,
                blur=blur,
                group_ids=time_ids,
                normalize_mode="mean_one",
            ).item()
        )
    out = compute_balance_diagnostics(Z, A, w_uniform, treatment_mode=treatment_mode)
    out["offline_val_align_last"] = val_align
    out["val_align_uniform"] = val_align
    out["N_val"] = int(n)
    return out


def log_weight_refresh(epoch: int, diag: Dict[str, Any], metric: str, logger_obj: Optional[logging.Logger] = None):
    """Print [W-refresh] line per user spec."""
    log = logger_obj or logger
    align_drop_rel = float(diag.get("align_drop_rel", float("nan")))
    log.info(
        "[W-refresh] epoch=%s N=%s total_collected=%s active_kept=%s inactive_dropped=%s "
        "unique_patients=%s unique_times=%s metric=%s align_initial=%.6f align_final=%.6f "
        "align_drop=%.6f align_drop_rel=%.2f%% ESS=%.4f std=%.4f min=%.4f max=%.4f "
        "p01=%.4f p05=%.4f p50=%.4f p95=%.4f p99=%.4f "
        "min_ess_by_time=%.4f mean_ess_by_time=%.4f low_w_frac<0.01=%.4f low_w_frac<0.05=%.4f",
        epoch,
        diag.get("N", "?"),
        diag.get("total_collected", "?"),
        diag.get("active_kept", "?"),
        diag.get("inactive_dropped", "?"),
        diag.get("unique_patients", "?"),
        diag.get("unique_times", "?"),
        metric,
        diag.get("align_initial", float("nan")),
        diag.get("align_final", float("nan")),
        diag.get("align_drop", float("nan")),
        100.0 * align_drop_rel,
        diag.get("ess_frac", float("nan")),
        diag.get("weight_std", float("nan")),
        diag.get("weight_min", float("nan")),
        diag.get("weight_max", float("nan")),
        diag.get("weight_p01", float("nan")),
        diag.get("weight_p05", float("nan")),
        diag.get("weight_p50", float("nan")),
        diag.get("weight_p95", float("nan")),
        diag.get("weight_p99", float("nan")),
        diag.get("min_ess_by_time", float("nan")),
        diag.get("mean_ess_by_time", float("nan")),
        diag.get("low_weight_frac_001", float("nan")),
        diag.get("low_weight_frac_005", float("nan")),
    )
    log_w_balance(diag, prefix="[W-balance]", logger_obj=log)


def log_w_balance(diag: Dict[str, Any], prefix: str = "[W-balance]", logger_obj: Optional[logging.Logger] = None):
    log = logger_obj or logger
    metric = diag.get("treat_metric", "r2")
    if metric == "acc":
        log.info(
            "%s treatment_pred acc: unweighted=%.4f weighted=%.4f | "
            "corr mean_abs unweighted=%.4f weighted=%.4f | max_abs unweighted=%.4f weighted=%.4f",
            prefix,
            diag.get("treat_pred_unweighted", float("nan")),
            diag.get("treat_pred_weighted", float("nan")),
            diag.get("mean_abs_corr_unweighted", float("nan")),
            diag.get("mean_abs_corr_weighted", float("nan")),
            diag.get("max_abs_corr_unweighted", float("nan")),
            diag.get("max_abs_corr_weighted", float("nan")),
        )
    else:
        log.info(
            "%s treatment_pred R2: unweighted=%.4f weighted=%.4f MSE: unweighted=%.6f weighted=%.6f | "
            "corr mean_abs unweighted=%.4f weighted=%.4f | max_abs unweighted=%.4f weighted=%.4f",
            prefix,
            diag.get("treat_pred_unweighted", float("nan")),
            diag.get("treat_pred_weighted", float("nan")),
            diag.get("treat_mse_unweighted", float("nan")),
            diag.get("treat_mse_weighted", float("nan")),
            diag.get("mean_abs_corr_unweighted", float("nan")),
            diag.get("mean_abs_corr_weighted", float("nan")),
            diag.get("max_abs_corr_unweighted", float("nan")),
            diag.get("max_abs_corr_weighted", float("nan")),
        )


def log_w_used(extras: Dict[str, Any], logger_obj: Optional[logging.Logger] = None):
    """Log dataset/time-level weights actually read from the dataloader (offline_periodic train)."""
    log = logger_obj or logger
    if "w_used_p50" not in extras:
        return
    log.info(
        "[W-used] (dataset/time-level table) mean=%.4f std=%.4f min=%.4f max=%.4f ESS=%.4f "
        "p01=%.4f p05=%.4f p50=%.4f p95=%.4f p99=%.4f",
        extras.get("w_used_mean", float("nan")),
        extras.get("w_used_std", float("nan")),
        extras.get("w_used_min", float("nan")),
        extras.get("w_used_max", float("nan")),
        extras.get("w_used_ess_frac", float("nan")),
        extras.get("w_used_p01", float("nan")),
        extras.get("w_used_p05", float("nan")),
        extras.get("w_used_p50", float("nan")),
        extras.get("w_used_p95", float("nan")),
        extras.get("w_used_p99", float("nan")),
    )
