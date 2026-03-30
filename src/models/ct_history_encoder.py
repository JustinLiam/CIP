import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """标准的位置编码，Transformer处理时序必须依赖此模块"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CTHistoryEncoder(nn.Module):
    def __init__(self, x_dim, a_dim, y_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        """
        x_dim: 协变量(vitals + static_features)的维度
        a_dim: 治疗记录(treatments)的维度
        y_dim: 历史结果(outputs)的维度
        d_model: Transformer内部的隐藏层维度
        """
        super().__init__()
        self.d_model = d_model

        # 1. 独立特征嵌入层 (Feature Embeddings)
        self.x_enc = nn.Linear(x_dim, d_model)
        self.a_enc = nn.Linear(a_dim, d_model)
        self.y_enc = nn.Linear(y_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 2. 自注意力编码器 (Self-Attention)
        # batch_first=True 保证输入输出形状为 [batch_size, seq_len, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_x = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.self_attn_a = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.self_attn_y = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 交叉注意力机制 (Cross-Attention)
        # 在 PyTorch 中，TransformerDecoderLayer 就是完美的 Cross-Attention 实现
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        # X 关注 A (协变量如何受治疗影响)
        self.cross_attn_xa = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # X 关注 Y (协变量如何受历史结果影响)
        self.cross_attn_xy = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码 (Causal Mask) —— 极其重要，防止信息泄露(偷看未来)"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, a, y):
        """
        x, a, y 形状均为: [batch_size, seq_len, dim]
        """
        seq_len = x.size(1)
        # 生成因果掩码 [seq_len, seq_len]
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Step 1: 投影并加上位置编码
        x_emb = self.pos_encoder(self.x_enc(x))
        a_emb = self.pos_encoder(self.a_enc(a))
        y_emb = self.pos_encoder(self.y_enc(y))

        # Step 2: 独立的时序演化 (Self-Attention)
        x_rep = self.self_attn_x(x_emb, mask=mask)
        a_rep = self.self_attn_a(a_emb, mask=mask)
        y_rep = self.self_attn_y(y_emb, mask=mask)

        # Step 3: 交叉注意力融合 (Cross-Attention)
        # tgt 是 Query，memory 是 Key/Value。两者都需要 mask，防止 t 时刻看到 t+1 的记忆。
        xa_rep = self.cross_attn_xa(tgt=x_rep, memory=a_rep, tgt_mask=mask, memory_mask=mask)
        xy_rep = self.cross_attn_xy(tgt=x_rep, memory=y_rep, tgt_mask=mask, memory_mask=mask)

        # Step 4: 聚合表示 (平均池化或拼接)
        # 参考 Causal Transformer 论文，我们将所有学到的表示取平均
        final_rep = (x_rep + a_rep + y_rep + xa_rep + xy_rep) / 5.0

        return final_rep  # 输出形状: [batch_size, seq_len, d_model]

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