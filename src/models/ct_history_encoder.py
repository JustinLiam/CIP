import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils_transformer import (
    AbsolutePositionalEncoding,
    RelativePositionalEncoding,
    TransformerMultiInputBlock,
)


class LocalConvMultiInputBlock(nn.Module):
    """Decision ConvFormer-style local token mixer for CT's separated streams."""

    def __init__(
        self,
        hidden,
        feed_forward_hidden,
        dropout,
        kernel_size=6,
        dilation=1,
        n_inputs=3,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("local convolution kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("local convolution dilation must be >= 1")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.n_inputs = int(n_inputs)

        self.conv_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_inputs)])
        self.local_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden,
                    out_channels=hidden,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    groups=hidden,
                )
                for _ in range(n_inputs)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_inputs)])
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, feed_forward_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feed_forward_hidden, hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(n_inputs)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_mask(x, active_entries):
        if active_entries is None:
            return x
        return x * active_entries.to(device=x.device, dtype=x.dtype)

    def _causal_depthwise_conv(self, x, conv):
        pad_left = (self.kernel_size - 1) * self.dilation
        x_ch_first = F.pad(x.transpose(1, 2), (pad_left, 0))
        return conv(x_ch_first).transpose(1, 2)

    def forward(self, x_tov, x_s, active_entries_treat_outcomes, active_entries_vitals=None):
        assert len(x_tov) == self.n_inputs
        if self.n_inputs == 2:
            masks = (active_entries_treat_outcomes, active_entries_treat_outcomes)
        else:
            masks = (
                active_entries_treat_outcomes,
                active_entries_treat_outcomes,
                active_entries_vitals if active_entries_vitals is not None else active_entries_treat_outcomes,
            )

        outputs = []
        for idx, (x, mask) in enumerate(zip(x_tov, masks)):
            x = self._apply_mask(x, mask)
            conv_in = self._apply_mask(self.conv_norms[idx](x), mask)
            x = x + self.dropout(self._causal_depthwise_conv(conv_in, self.local_convs[idx]))
            x = self._apply_mask(x, mask)

            # Static covariates are global context, not temporal tokens; inject them before the FFN.
            ffn_in = self.ffn_norms[idx](x + x_s)
            x = x + self.feed_forwards[idx](ffn_in)
            x = self._apply_mask(x, mask)
            outputs.append(x)

        return tuple(outputs)


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
        local_conv_layers=0,
        local_conv_kernel_size=6,
        local_conv_dilation=1,
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
        self.local_conv_layers = max(0, min(int(local_conv_layers), int(num_layers)))
        self.global_attention_layers = int(num_layers) - self.local_conv_layers

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

        # 2. 底层使用 Decision ConvFormer-style local causal depthwise convolution.
        self.local_blocks = nn.ModuleList(
            [
                LocalConvMultiInputBlock(
                    hidden=d_model,
                    feed_forward_hidden=d_model * 4,
                    dropout=dropout,
                    kernel_size=local_conv_kernel_size,
                    dilation=local_conv_dilation,
                    n_inputs=3,
                )
                for _ in range(self.local_conv_layers)
            ]
        )

        # 3. 高层保留 CT 的 multi-input block（self/cross attention）。
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
                for _ in range(self.global_attention_layers)
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

        for block in self.local_blocks:
            x_t, x_o, x_v = block(
                (x_t, x_o, x_v),
                x_s,
                active_entries_treat_outcomes=active_entries,
                active_entries_vitals=active_entries,
            )

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
