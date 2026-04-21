"""
CT encoder + WeightNet + Predictor for ctd.md standalone training.
Does not modify TransformerMultiInputBlock internals.
"""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.models.ct_history_encoder import CTHistoryEncoder, ProjectionHead


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
        ph = int(_cfg_sel(cfg, "exp.ct_predictor_hidden", 64))
        dh = int(_cfg_sel(cfg, "exp.ct_dyn_hidden", ph))
        dyn_residual = bool(_cfg_sel(cfg, "exp.ct_dyn_residual", True))
        self.weight_net = WeightNet(self.z_dim, self.treatment_dim, hidden_dim=wh)
        self.predictor = OutcomePredictor(self.z_dim, self.treatment_dim, self.output_dim, hidden_dim=ph)
        self.z_dynamics = LatentDynamicsPredictor(
            self.z_dim, self.treatment_dim, hidden_dim=dh, residual=dyn_residual
        )

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

    def forward(self, H_t: Dict[str, torch.Tensor], y_next: torch.Tensor):
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
        """
        Z_t, A_t = self.encode(H_t)
        
        # ==============================================================
        # 1. E-Step 分支：为 WeightNet 提供特征
        # 必须对 Z_t 进行 .detach()，彻底切断 loss_align 流向 Encoder 的路径
        # ==============================================================
        za_for_w = torch.cat([Z_t.detach(), A_t], dim=-1)
        logits_w = self.weight_net(za_for_w)
        w = F.softmax(logits_w, dim=0) * float(Z_t.size(0))
        
        # ==============================================================
        # 2. M-Step 分支：为 Predictor 提供特征
        # 这里的 Z_t 保持正常连接，负责接收 loss_pred 回传的梯度更新 Encoder
        # ==============================================================
        za_for_y = torch.cat([Z_t, A_t], dim=-1)
        y_hat = self.predictor(za_for_y)
        se = (y_hat - y_next).pow(2).mean(dim=-1)
        # 提取当前时间步的 active 掩码 [B]
        active_t = H_t["active_entries"][:, -1, 0]
        # Weighted (WeightNet-reweighted) M-step loss — original behaviour.
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
        loss_pred_w = (w.detach() * se * active_t).sum() / (active_t.sum() + 1e-8)
        # anchor loss: 只用均匀权重，辅助校正 offset bias，增强泛化
        loss_pred_anchor = (se * active_t).sum() / (active_t.sum() + 1e-8)

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
        loss_dyn = F.mse_loss(z_next_pred, z_next_tgt)
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
        active_t = H_t["active_entries"][:, -1, 0]
        loss_w = (w_fixed * se * active_t).sum() / (active_t.sum() + 1e-8)
        loss_anchor = (se * active_t).sum() / (active_t.sum() + 1e-8)
        return loss_w, y_hat, loss_anchor
