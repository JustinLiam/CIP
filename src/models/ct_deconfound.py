"""
CT encoder + WeightNet + Predictor for ctd.md standalone training.
Does not modify TransformerMultiInputBlock internals.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.models.ct_history_encoder import CTHistoryEncoder, ProjectionHead


def active_weight_from_logits(logits_w: torch.Tensor, active_t: torch.Tensor) -> torch.Tensor:
    """
    Mask inactive batch rows before softmax so padding does not share mass;
    scale by ``sum(active_t)`` (not ``B``).

    Aligns with ``forward`` supervised mask ``H_t['active_entries'][:, -1, 0]``.
    Degenerate batches with no active row return zeros.
    """
    n_act = active_t.detach().float().sum()
    if float(n_act) <= 0.0:
        return torch.zeros_like(logits_w)
    logits_m = logits_w.masked_fill(active_t.detach() < 0.5, torch.tensor(-1e4, device=logits_w.device, dtype=logits_w.dtype))
    return F.softmax(logits_m, dim=0) * n_act


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


def build_covariate_x(H_t: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    """Match ``InferenceModel.build_H_t`` covariate stream (x_input to CT)."""
    static_size = int(cfg["dataset"]["static_size"])
    predict_x = bool(cfg["dataset"].get("predict_X", False))
    autoregressive = bool(cfg["dataset"].get("autoregressive", False))

    if static_size > 0:
        if predict_x:
            x = torch.cat((H_t["vitals"], H_t["static_features"]), dim=-1)
        else:
            x = H_t["static_features"]
    else:
        if "current_covariates" in H_t:
            x = H_t["current_covariates"]
        elif "vitals" in H_t:
            x = H_t["vitals"]
        else:
            raise KeyError("Need current_covariates or vitals when static_size==0")

    if autoregressive:
        x = torch.cat((x, H_t["prev_outputs"]), dim=-1)
    x = torch.cat((x, H_t["prev_treatments"]), dim=-1)
    return x


class WeightNet(nn.Module):
    """MLP(Z_t, A_t) -> scalar; batch normalization via softmax in training code."""

    def __init__(self, z_dim: int, a_dim: int, hidden_dim: int = 64):
        super().__init__()
        d = z_dim + a_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.net(z_a).squeeze(-1)


class OutcomePredictor(nn.Module):
    """MLP(Z_t, A_t) -> Y_{t+1}."""

    def __init__(self, z_dim: int, a_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        d = z_dim + a_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.net(z_a)


class OutcomeDecoder(nn.Module):
    """MLP(Z) -> Y; used for k-step latent rollout decoding (no action in head)."""

    def __init__(self, z_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentDynamicsPredictor(nn.Module):
    """MLP(Z_t, A_t) -> Z_{t+1}; residual form stabilizes one-step latent transitions."""

    def __init__(self, z_dim: int, a_dim: int, hidden_dim: int = 64, residual: bool = True):
        super().__init__()
        d = z_dim + a_dim
        self.residual = bool(residual)
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z_t, a_t], dim=-1))
        return z_t + delta if self.residual else delta


class CTDeconfoundModel(nn.Module):
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
        num_layers = int(md["ct_model"]["num_layers"])

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
        ph = int(_cfg_sel(cfg, "exp.ct_predictor_hidden", 64))
        dh = int(_cfg_sel(cfg, "exp.ct_dyn_hidden", ph))
        dyn_residual = bool(_cfg_sel(cfg, "exp.ct_dyn_residual", True))
        self.weight_net = WeightNet(self.z_dim, self.treatment_dim, hidden_dim=wh)
        self.predictor = OutcomePredictor(self.z_dim, self.treatment_dim, self.output_dim, hidden_dim=ph)
        self.z_dynamics = LatentDynamicsPredictor(
            self.z_dim, self.treatment_dim, hidden_dim=dh, residual=dyn_residual
        )
        self.outcome_decoder = OutcomeDecoder(self.z_dim, self.output_dim, hidden_dim=ph)

    def encode(self, H_t: Dict[str, torch.Tensor]) -> tuple:
        """
        Returns:
            Z_t: [B, z_dim] last-step representation
            A_t: [B, a_dim] current treatment at last valid step
        """
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

    def forward(
        self,
        H_t: Dict[str, torch.Tensor],
        y_next: torch.Tensor,
        w_fixed: torch.Tensor = None,
    ):
        """
        Returns
        -------
        loss_pred_w : weighted MSE (WeightNet-reweighted)  — original objective.
        Z_t, A_t, w, logits_w, y_hat : encoder/predictor outputs (unchanged).
        loss_pred_anchor : UN-WEIGHTED MSE on active samples (uniform reweighting).
            Use this as an anchor / calibration loss to counteract the systematic
            offset bias that arises when WeightNet concentrates mass on a sub-population
            whose y-mean differs from the true population y-mean. Combine in train_ct.py
            as ``(1 - a) * loss_pred_w + a * loss_pred_anchor`` with ``a = ct_anchor_weight``.

        If ``w_fixed`` is provided (offline_periodic / none modes), skip online WeightNet
        and use the precomputed per-sample weights. Inactive rows should have w=0.
        """
        Z_t, A_t = self.encode(H_t)

        active_t = H_t["active_entries"][:, -1, 0].float()
        n_act = active_t.sum() + 1e-8

        if w_fixed is not None:
            w = w_fixed.reshape(-1).to(device=Z_t.device, dtype=Z_t.dtype) * active_t
            logits_w = w
        else:
            # Online E-step path: WeightNet on [Z_t.detach(), A_t].
            za_for_w = torch.cat([Z_t.detach(), A_t], dim=-1)
            logits_raw = self.weight_net(za_for_w)
            w = active_weight_from_logits(logits_raw, active_t)
            logits_w = logits_raw.masked_fill(active_t.detach() < 0.5, logits_raw.new_tensor(-1e4))

        # ==============================================================
        # 2. M-Step 分支：为 Predictor 提供特征
        # 这里的 Z_t 保持正常连接，负责接收 loss_pred 回传的梯度更新 Encoder
        # ==============================================================
        za_for_y = torch.cat([Z_t, A_t], dim=-1)
        y_hat = self.predictor(za_for_y)
        se = (y_hat - y_next).pow(2).mean(dim=-1)
        # Weighted (WeightNet-reweighted) M-step loss — softmax 仅在 active 上归一，
        # 分母 n_act 与活跃样本均值一致。
        #
        # 为什么需要 weighted 和 unweighted 两个 loss（loss_pred_w, loss_pred_anchor）？
        #
        # Weighted loss 是用 WeightNet 输出的权重 w 对 MSE 进行重加权，本意是缓解混杂影响，实现更“公平”或去偏的预测。
        # 理论上，如果 w 完美刻画混杂，weighted loss 就反映了消除混杂下的理想风险。
        #
        # 然而，实际训练时，WeightNet 可能受到数据分布/容量限制等影响，w 往往难以完全精确建模混杂机制。
        # 这时，w 可能只在训练样本的某些子群上取高值，导致预测模型最后只对这些子群拟合得很好，
        # 但对整体（观测分布）则出现“系统性偏置” —— 也就是模型预测整体抬高或降低（offset bias）。
        #
        # 这种偏置并不是完全由于“WeightNet 不行”，而是数据本身偏倚、欠充分、或混杂太复杂时，任何 weighting 方法都难以完全恢复整体正确分布，
        # 加之 finite sample + NN 优化动力学更易造成 w 集中——这时 weighted loss 已不能单独说明模型整体表现。
        #
        # unweighted loss（anchor loss）则直接反映在实际观测分布下的均值误差，体现了全体样本下的校准能力。
        # 若只依赖 weighted loss 优化，最终模型甚至可能产生很大 population-level 偏差；引入 anchor loss，可同时兼顾均值准确性和变量去偏。
        #
        w_denom = (w.detach() * active_t).sum() + 1e-8
        loss_pred_w = (w.detach() * se * active_t).sum() / w_denom
        # anchor loss: 只用均匀权重，辅助校正 offset bias，增强泛化
        loss_pred_anchor = (se * active_t).sum() / n_act

        return loss_pred_w, Z_t, A_t, w, logits_w, y_hat, loss_pred_anchor

    def latent_dynamics_loss(
        self,
        H_t: Dict[str, torch.Tensor],
        H_t_next: Dict[str, torch.Tensor],
        *,
        detach_target: bool = True,
    ) -> tuple:
        """
        One-step latent dynamics consistency:
          z_next_pred = g(z_t, a_t)
          z_next_tgt  = encoder(H_{t+1})
        and regress the prediction toward the target latent.

        ``detach_target`` keeps the target branch as a stable teacher signal, mirroring
        CTD-NKO's Koopman-style "predict next hidden state, match encoded next state".
        """
        Z_t, A_t = self.encode(H_t)
        z_next_pred = self.z_dynamics(Z_t, A_t)
        with torch.set_grad_enabled(not detach_target):
            z_next_tgt, _ = self.encode(H_t_next)
        if detach_target:
            z_next_tgt = z_next_tgt.detach()
        # 提取当前时间步是否有效的掩码（active_t）
        active_t = H_t["active_entries"][:, -1, 0] 
        # 只针对 active_t == 1 的样本计算误差，并求平均
        loss_dyn = ((z_next_pred - z_next_tgt).pow(2).mean(dim=-1) * active_t).sum() / (active_t.sum() + 1e-8)
        # loss_dyn = F.mse_loss(z_next_pred, z_next_tgt)
        return loss_dyn, z_next_pred, z_next_tgt

    def weighted_prediction_loss(
        self, H_t: Dict[str, torch.Tensor], y_target: torch.Tensor, w_fixed: torch.Tensor
    ) -> tuple:
        """
        Multi-horizon (k=2, k=3) M-step loss with a FIXED weight ``w_fixed`` from k=1.
        Mirrors ``forward``'s API: returns ``(loss_w, y_hat, loss_anchor)`` so the
        caller can blend them with the same ``ct_anchor_weight``.
        """
        Z_t, A_t = self.encode(H_t)
        za = torch.cat([Z_t, A_t], dim=-1)
        y_hat = self.predictor(za)
        se = (y_hat - y_target).pow(2).mean(dim=-1)
        active_t = H_t["active_entries"][:, -1, 0].float()
        w_denom = (w_fixed * active_t).sum() + 1e-8
        loss_w = (w_fixed * se * active_t).sum() / w_denom
        loss_anchor = (se * active_t).sum() / (active_t.sum() + 1e-8)
        return loss_w, y_hat, loss_anchor

    def rollout_dynamics_loss(
        self,
        H_t: Dict[str, torch.Tensor],
        a_seq: torch.Tensor,
        a_seq_mask: torch.Tensor,
        horizon_k: torch.Tensor,
        y_future: torch.Tensor,
        H_future_list: Optional[List[Dict[str, torch.Tensor]]] = None,
        *,
        Z_t: Optional[torch.Tensor] = None,
        latent_weight: float = 0.1,
        decode_each_step: bool = False,
    ) -> tuple:
        """
        Action-conditioned latent rollout: z_{t+j} = g(z_{t+j-1}, a_{t+j-1}), decode y_{t+k}.

        ``horizon_k`` is 1-based; loss_y compares decoded y at step k to ``y_future``.
        Optional ``H_future_list[j-1]`` provides stop-gradient latent targets at H_{t+j}.
        Pass ``Z_t`` from a prior ``encode(H_t)`` to avoid a duplicate encoder forward.
        Rollout depth truncates to ``max(horizon_k)`` in the batch (loss-exact).
        """
        del decode_each_step  # reserved for per-step decode supervision in later versions
        if Z_t is None:
            Z_t, _ = self.encode(H_t)
        active_t = H_t["active_entries"][:, -1, 0].float()
        n_act = active_t.sum() + 1e-8

        B, k_max, _ = a_seq.shape
        k_roll = int(horizon_k.long().max().clamp(min=1, max=k_max).item())

        z_roll = Z_t
        z_by_step = [Z_t]
        y_by_step = []
        for j in range(k_roll):
            a_j = a_seq[:, j, :]
            z_roll = self.z_dynamics(z_roll, a_j)
            z_by_step.append(z_roll)
            y_by_step.append(self.outcome_decoder(z_roll))

        y_hat_stack = torch.stack(y_by_step, dim=1)  # [B, k_roll, y_dim]
        idx = (horizon_k.long().clamp(min=1, max=k_roll) - 1).view(B, 1, 1).expand(
            -1, 1, y_hat_stack.size(-1)
        )
        y_hat_k = y_hat_stack.gather(1, idx).squeeze(1)
        se_y = (y_hat_k - y_future).pow(2).mean(dim=-1)
        loss_y = (se_y * active_t).sum() / n_act

        z_tgt_cached: List[Optional[torch.Tensor]] = []
        loss_z = torch.zeros((), device=Z_t.device, dtype=Z_t.dtype)
        if H_future_list is not None and latent_weight > 0.0:
            z_losses = []
            hk = horizon_k.long()
            n_future = min(len(H_future_list), k_roll)
            for j in range(1, n_future + 1):
                H_fj = H_future_list[j - 1]
                if H_fj is None:
                    z_tgt_cached.append(None)
                    continue
                with torch.no_grad():
                    z_tgt, _ = self.encode(H_fj)
                z_tgt_cached.append(z_tgt)
                z_pred = z_by_step[j]
                se_z = (z_pred - z_tgt).pow(2).mean(dim=-1)
                step_mask = (hk >= j).float() * a_seq_mask[:, j - 1] * active_t
                denom = step_mask.sum() + 1e-8
                z_losses.append((se_z * step_mask).sum() / denom)
            if z_losses:
                loss_z = torch.stack(z_losses).mean()

        loss_total = loss_y + float(latent_weight) * loss_z
        with torch.no_grad():
            if k_roll < k_max:
                z_roll_ext = z_by_step[-1]
                for j in range(k_roll, k_max):
                    z_roll_ext = self.z_dynamics(z_roll_ext, a_seq[:, j, :])
                    z_by_step.append(z_roll_ext)

            mae_norm = ((y_hat_k - y_future).abs().mean(dim=-1) * active_t).sum() / n_act
            mean_k = float((horizon_k.float() * active_t).sum() / n_act)
            k_frac_eq1 = float(((horizon_k == 1).float() * active_t).sum() / n_act)

            def _masked_mean_l2(z: torch.Tensor) -> float:
                zn = z.pow(2).sum(dim=-1).sqrt()
                return float((zn * active_t).sum() / n_act)

            z_norm_init = _masked_mean_l2(Z_t)
            z_stack = torch.stack(z_by_step, dim=1)  # [B, Kmax+1, z_dim]; index j = z_hat_{t+j}
            idx_k = horizon_k.long().clamp(1, k_max).view(B, 1, 1).expand(-1, 1, self.z_dim)
            z_at_hk = z_stack.gather(1, idx_k).squeeze(1)
            z_norm_at_hk = _masked_mean_l2(z_at_hk)
            zn_init_per = Z_t.pow(2).sum(dim=-1).sqrt()
            zn_at_hk_per = z_at_hk.pow(2).sum(dim=-1).sqrt()
            ratio_per = zn_at_hk_per / (zn_init_per + 1e-8)
            z_norm_ratio = float((ratio_per * active_t).sum() / n_act)
            z_shrink_frac = float(((ratio_per < 0.5).float() * active_t).sum() / n_act)
            z_norm_at_kmax = _masked_mean_l2(z_by_step[-1])

            z_rel_err_hk = float("nan")
            z_cos_hk = float("nan")
            if H_future_list is not None and len(H_future_list) > 0:
                # horizon_k <= k_roll, so z_tgt at index hk-1 only needs futures 1..k_roll.
                n_tgt = min(len(H_future_list), k_roll)
                z_tgt_list: List[torch.Tensor] = []
                for j in range(n_tgt):
                    if j < len(z_tgt_cached) and z_tgt_cached[j] is not None:
                        z_tgt_list.append(z_tgt_cached[j])
                    else:
                        with torch.no_grad():
                            z_j, _ = self.encode(H_future_list[j])
                        z_tgt_list.append(z_j)
                z_tgt_stack = torch.stack(z_tgt_list, dim=1)
                idx_tgt = (horizon_k.long().clamp(1, len(H_future_list)) - 1).view(
                    B, 1, 1
                ).expand(-1, 1, self.z_dim)
                z_tgt_hk = z_tgt_stack.gather(1, idx_tgt).squeeze(1)
                err = (z_at_hk - z_tgt_hk).pow(2).sum(dim=-1).sqrt()
                tgt_norm = z_tgt_hk.pow(2).sum(dim=-1).sqrt()
                z_rel_err_hk = float(((err / (tgt_norm + 1e-8)) * active_t).sum() / n_act)
                cos = F.cosine_similarity(z_at_hk, z_tgt_hk, dim=-1)
                z_cos_hk = float((cos * active_t).sum() / n_act)

            # Per-step ||z|| / ||z_t|| to spot monotonic shrink across rollout depth.
            step_ratios = []
            for j in range(1, len(z_by_step)):
                step_ratios.append(_masked_mean_l2(z_by_step[j]) / (z_norm_init + 1e-8))

        diag = {
            "loss_rollout_y": float(loss_y.detach()),
            "loss_rollout_z": float(loss_z.detach()) if torch.is_tensor(loss_z) else float(loss_z),
            "rollout_mae_norm": float(mae_norm),
            "mean_k": mean_k,
            "k_frac_eq1": k_frac_eq1,
            "z_norm_init": z_norm_init,
            "z_norm_at_hk": z_norm_at_hk,
            "z_norm_ratio": z_norm_ratio,
            "z_norm_at_kmax": z_norm_at_kmax,
            "z_norm_step_ratio_last": float(step_ratios[-1]) if step_ratios else float("nan"),
            "z_rel_err_hk": z_rel_err_hk,
            "z_cos_hk": z_cos_hk,
            "z_shrink_frac": z_shrink_frac,
        }
        return loss_total, diag
