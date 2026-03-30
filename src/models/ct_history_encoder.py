import torch
import torch.nn as nn
from src.models.utils_transformer import (
    AbsolutePositionalEncoding,
    RelativePositionalEncoding,
    TransformerMultiInputBlock,
)

class CTHistoryEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        a_dim,
        y_dim,
        static_dim=0,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=512,
        use_relative_positional_encoding=True,
        max_relative_position=64,
    ):
        """
        x_dim: 协变量(vitals + static_features)的维度
        a_dim: 治疗记录(treatments)的维度
        y_dim: 历史结果(outputs)的维度
        d_model: Transformer内部的隐藏层维度
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_size = d_model // num_heads
        self.static_dim = static_dim

        # 1. 独立特征嵌入层 (Feature Embeddings)
        self.x_enc = nn.Linear(x_dim, d_model)
        self.a_enc = nn.Linear(a_dim, d_model)
        self.y_enc = nn.Linear(y_dim, d_model)
        self.static_input_transformation = (
            nn.Linear(static_dim, d_model) if static_dim and static_dim > 0 else None
        )
        # 为与 CT 主实现对齐：支持 absolute / relative positional encoding
        self.self_positional_encoding = None
        self.self_positional_encoding_k = None
        self.self_positional_encoding_v = None
        if use_relative_positional_encoding:
            self.self_positional_encoding_k = RelativePositionalEncoding(
                max_relative_position=max_relative_position,
                d_model=self.head_size,
                trainable=False,
            )
            self.self_positional_encoding_v = RelativePositionalEncoding(
                max_relative_position=max_relative_position,
                d_model=self.head_size,
                trainable=False,
            )
        else:
            self.self_positional_encoding = AbsolutePositionalEncoding(
                max_len=max_seq_len, d_model=d_model, trainable=False
            )

        # 2-3. 使用 CT 的 multi-input block（每层同时做 self/cross attention）
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerMultiInputBlock(
                    hidden=d_model,
                    attn_heads=num_heads,
                    head_size=self.head_size,
                    feed_forward_hidden=d_model * 4,
                    dropout=dropout,
                    attn_dropout=dropout,
                    self_positional_encoding_k=self.self_positional_encoding_k,
                    self_positional_encoding_v=self.self_positional_encoding_v,
                    n_inputs=3,
                    disable_cross_attention=False,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, a, y, active_entries=None, static_features=None):
        """
        x, a, y 形状均为: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.size()
        if active_entries is None:
            active_entries = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)

        # 与 CT 一致：三个子网络输入分别映射后在每层 block 中做 self/cross attention
        x_t = self.a_enc(a)  # treatment stream
        x_o = self.y_enc(y)  # outcome stream
        x_v = self.x_enc(x)  # covariate stream

        if self.self_positional_encoding is not None:
            x_t = x_t + self.self_positional_encoding(x_t)
            x_o = x_o + self.self_positional_encoding(x_o)
            x_v = x_v + self.self_positional_encoding(x_v)

        # 与 CT 一致：静态通道作为逐层偏置项注入
        if self.static_input_transformation is not None and static_features is not None:
            if static_features.dim() == 3:
                static_features = static_features[:, 0, :]
            x_s = self.static_input_transformation(static_features).unsqueeze(1)
        else:
            x_s = torch.zeros(batch_size, 1, self.d_model, device=x.device, dtype=x.dtype)

        for block in self.transformer_blocks:
            x_t, x_o, x_v = block(
                (x_t, x_o, x_v),
                x_s,
                active_entries_treat_outcomes=active_entries,
                active_entries_vitals=active_entries,
            )

        # 与 CT 三分支聚合方式一致（等权平均）
        final_rep = (x_t + x_o + x_v) / 3.0
        return final_rep

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        input_dim: 对应 CTHistoryEncoder 的 d_model
        output_dim: 对应 VCIP 中的 z_dim (你的 s_t 的最终维度)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 增加 LayerNorm 提升训练稳定性
            nn.ELU(),                 # ELU 是 VCIP 原本习惯使用的激活函数
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        return self.net(x)