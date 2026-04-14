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
        self.weight_net = WeightNet(self.z_dim, self.treatment_dim, hidden_dim=wh)
        self.predictor = OutcomePredictor(self.z_dim, self.treatment_dim, self.output_dim, hidden_dim=ph)

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
        
        # 必须对 w 进行 .detach()，防止 loss_pred 流向 WeightNet
        loss_pred = (w.detach() * se).mean()
        
        return loss_pred, Z_t, A_t, w, logits_w, y_hat
