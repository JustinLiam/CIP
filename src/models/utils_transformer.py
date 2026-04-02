import torch
from torch import nn
import math
import torch.nn.functional as F


def get_fixed_sin_cos_encodings(d_model, max_len):
    """
    Sin-cos fixed positional encodddings
    Args:
        d_model: hidden state dimensionality
        max_len: max sequence length
    Returns: PE
    """
    assert d_model % 2 == 0
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, trainable=False):
        super().__init__()
        self.max_len = max_len
        self.trainable = trainable
        if trainable:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            self.register_buffer('pe', get_fixed_sin_cos_encodings(d_model, max_len))

    def forward(self, x):
        batch_size = x.size(0)
        actual_len = x.shape[1]
        assert actual_len <= self.max_len

        pe = self.pe.weight if self.trainable else self.pe
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)[:, :actual_len, :]

    def get_pe(self, position):
        pe = self.pe.weight if self.trainable else self.pe
        return pe[position]


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码（Relative Positional Encoding）模块，主要用于实现Transformer结构中的相对位置信息建模。

    这个类的作用是生成一个长度为[query_len, key_len, d_model]的编码张量，使得模型可以利用序列中两两元素之间的相对距离信息，从而增强模型对时序或结构关系的理解能力。常用于自注意力（self-attention）或交叉注意力（cross-attention）中作为额外的“偏置”项。

    参数说明：
        max_relative_position: 能感知的最大正负相对距离。例如设为16，则能区分[-16, 16]范围的关系。
        d_model: 每个相对距离编码的维度。
        trainable: 如果为True，编码参数可学习；否则使用固定的sin-cos编码。
        cross_attn: 是否用于cross-attention（解码器用，和self-attention的位置编码略有不同）。

    工作原理简述：
        - 初始化时构造一个编码lookup表（可以是可学习embedding，也可以是固定的sin-cos）。
        - 前向传播时，根据query/key长度构造一个距离矩阵，把每个query和key之间的相对距离都编码出来。
        - 然后查表得到每对query-key间的相对位置向量（形状为 [query_len, key_len, d_model]）。
        - 该位置编码可用于后续的注意力得分偏置。
    """
    def __init__(self, max_relative_position: int, d_model: int, trainable=False, cross_attn=False):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.trainable = trainable
        self.cross_attn = cross_attn
        # cross_attn为False时允许[-max, +max]，共2*max+1种相对位置；为True时只考虑[0, max]，常用于Decoder侧的跨注意力mask
        self.num_embeddings = (max_relative_position * 2 + 1) if not cross_attn else (max_relative_position + 1)
        if trainable:
            self.embeddings_table = nn.Embedding(self.num_embeddings, d_model)
        else:
            # 默认固定编码长度，也用2*max+1，无论cross_attn与否（简化实现，实际用时只会索引到需要范围）
            self.register_buffer('embeddings_table', get_fixed_sin_cos_encodings(d_model, max_relative_position * 2 + 1))

    def forward(self, length_q, length_k, device=None):
        embeddings_table = self.embeddings_table.weight if self.trainable else self.embeddings_table
        dev = device if device is not None else embeddings_table.device

        if self.cross_attn:
            distance_mat = torch.arange(length_k - 1, -1, -1, device=dev, dtype=torch.long)[None, :] + torch.arange(
                length_q, device=dev, dtype=torch.long
            )[:, None]
        else:
            distance_mat = torch.arange(length_k, device=dev, dtype=torch.long)[None, :] - torch.arange(
                length_q, device=dev, dtype=torch.long
            )[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        if not self.cross_attn:
            distance_mat_clipped = distance_mat_clipped + self.max_relative_position
        final_mat = distance_mat_clipped.long()
        embeddings = embeddings_table[final_mat]

        # Non-trainable far-away encodings
        # embeddings[(final_mat == final_mat.min()) | (final_mat == final_mat.max())] = 0.0
        return embeddings


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    """
    EDCT / Causal Transformer attention core.

    - Optional relative position bias on scores (R_k) and on the weighted values (R_v), same as the reference CT code.
    - ``one_direction=True`` applies a **causal** (lower-triangular) mask on the last two dims so position i cannot
      attend to j > i. Per project CT constraints, self-attention and **aligned** cross-attention between X/A/Y streams
      should use causal masking; keep ``one_direction=False`` only for non-causal cases (e.g. decoder attending a full
      encoder memory).
    - Padding ``mask`` (1 = keep, 0 = block) is combined with the causal mask before softmax.
    """

    def __init__(self,
                 positional_encoding_k: RelativePositionalEncoding = None,
                 positional_encoding_v: RelativePositionalEncoding = None):
        super(Attention, self).__init__()
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v

    def forward(self, query, key, value, mask=None, dropout=None, one_direction=False):
        d_k = query.size(-1)
        device = query.device

        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.positional_encoding_k is not None:
            R_k = self.positional_encoding_k(query.size(2), key.size(2), device=device)
            scores = scores + torch.einsum('b h q d, q k d -> b h q k', query, R_k)

        scores = scores / math.sqrt(d_k)

        neg_inf = torch.finfo(scores.dtype).min

        if mask is not None:
            if mask.dtype != torch.bool:
                scores = scores.masked_fill(mask == 0, neg_inf)
            else:
                scores = scores.masked_fill(~mask, neg_inf)

        if one_direction:
            lq, lk = query.size(2), key.size(2)
            causal = torch.tril(torch.ones((lq, lk), device=device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, lq, lk), neg_inf)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = torch.matmul(p_attn, value)

        if self.positional_encoding_v is not None:
            R_v = self.positional_encoding_v(query.size(2), value.size(2), device=device)
            output = output + torch.einsum('b h q v, q v d -> b h q d', p_attn, R_v)

        return output, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, head_size=None, dropout=0.0, positional_encoding_k=None, positional_encoding_v=None,
                 final_layer=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, self.num_heads * self.head_size) for _ in range(3)])
        self.attention = Attention(positional_encoding_k, positional_encoding_v)
        self.dropout = nn.Dropout(p=dropout)
        if final_layer:
            self.final_layer = nn.Linear(self.num_heads * self.head_size, d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, one_direction=True):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => num_heads x d_k
        query_, key_, value_ = [layer(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                                for layer, x in zip(self.linear_layers, (query, key, value))]
        # 2) apply self_attention on all the projected vectors in batch.
        x, attn = self.attention(query_, key_, value_, mask=mask, dropout=self.dropout, one_direction=one_direction)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        if hasattr(self, 'final_layer'):
            x = self.final_layer(x)

        return self.layer_norm(x + query)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, final_layer=True, **kwargs):
        super().__init__()
        # self.layer_norm = LayerNorm(hidden)   - already used in MultiHeadedAttention and PositionwiseFeedForward
        self.self_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                   dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                   positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, active_entries):
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        x = self.self_attention(x, x, x, self_att_mask, True)
        x = self.feed_forward(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None,
                 cross_positional_encoding_k=None, cross_positional_encoding_v=None, final_layer=False, **kwargs):
        super().__init__()
        self.layer_norm = LayerNorm(hidden)
        self.self_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                   dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                   positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.cross_attention = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                    dropout=attn_dropout, positional_encoding_k=cross_positional_encoding_k,
                                                    positional_encoding_v=cross_positional_encoding_v, final_layer=final_layer)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, encoder_x, active_entries, active_encoder_br):
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        cross_att_mask = (active_encoder_br.unsqueeze(1) * active_entries).unsqueeze(1)

        x = self.self_attention(x, x, x, self_att_mask, True)
        x = self.cross_attention(x, encoder_x, encoder_x, cross_att_mask, False)
        x = self.feed_forward(x)
        return x


class TransformerMultiInputBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout,
                 self_positional_encoding_k=None, self_positional_encoding_v=None, n_inputs=2, final_layer=False,
                 disable_cross_attention=False, isolate_subnetwork='', **kwargs):
        super().__init__()
        self.n_inputs = n_inputs
        self.disable_cross_attention = disable_cross_attention
        self.isolate_subnetwork = isolate_subnetwork
        self.self_attention_o = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        self.self_attention_t = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                     dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                     positional_encoding_v=self_positional_encoding_v, final_layer=final_layer)
        if not disable_cross_attention:
            self.cross_attention_ot = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)
            self.cross_attention_to = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                           dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                           positional_encoding_v=self_positional_encoding_v,
                                                           final_layer=final_layer)

        if n_inputs == 3:
            self.self_attention_v = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                         dropout=attn_dropout, positional_encoding_k=self_positional_encoding_k,
                                                         positional_encoding_v=self_positional_encoding_v, final_layer=final_layer
                                                         )
            if not disable_cross_attention:
                self.cross_attention_tv = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vt = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_ov = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)
                self.cross_attention_vo = MultiHeadedAttention(num_heads=attn_heads, d_model=hidden, head_size=head_size,
                                                               dropout=attn_dropout,
                                                               positional_encoding_k=self_positional_encoding_k,
                                                               positional_encoding_v=self_positional_encoding_v,
                                                               final_layer=final_layer)

        self.feed_forwards = [PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
                              for _ in range(n_inputs)]
        self.feed_forwards = nn.ModuleList(self.feed_forwards)

        self.n_inputs = n_inputs

    def forward(self, x_tov, x_s, active_entries_treat_outcomes, active_entries_vitals=None):
        assert len(x_tov) == self.n_inputs
        if self.n_inputs == 2:
            x_t, x_o = x_tov
        else:
            x_t, x_o, x_v = x_tov

        self_att_mask_ot = active_entries_treat_outcomes.repeat(1, 1, x_t.size(1)).unsqueeze(1)
        cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot

        x_t_ = self.self_attention_t(x_t, x_t, x_t, self_att_mask_ot, True)
        x_to_ = self.cross_attention_to(x_t_, x_o, x_o, cross_att_mask_ot, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'o' else x_t_

        x_o_ = self.self_attention_o(x_o, x_o, x_o, self_att_mask_ot, True)
        x_ot_ = self.cross_attention_ot(x_o_, x_t, x_t, cross_att_mask_to, True) if not self.disable_cross_attention \
            and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 't' else x_o_

        if self.n_inputs == 2:
            out_t = self.feed_forwards[0](x_to_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_s)

            return out_t, out_o

        else:
            self_att_mask_v = active_entries_vitals.repeat(1, 1, x_v.size(1)).unsqueeze(1)
            cross_att_mask_ot_v = (active_entries_vitals.squeeze(-1).unsqueeze(1) * active_entries_treat_outcomes).unsqueeze(1)
            cross_att_mask_v_ot = (active_entries_treat_outcomes.squeeze(-1).unsqueeze(1) * active_entries_vitals).unsqueeze(1)

            x_tv_ = self.cross_attention_tv(x_t_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 't' and self.isolate_subnetwork != 'v' else 0.0
            x_ov_ = self.cross_attention_ov(x_o_, x_v, x_v, cross_att_mask_ot_v, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'o' and self.isolate_subnetwork != 'v' else 0.0

            x_v_ = self.self_attention_v(x_v, x_v, x_v, self_att_mask_v, True)
            x_vt_ = self.cross_attention_vt(x_v_, x_t, x_t, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 't' else x_v_
            x_vo_ = self.cross_attention_vo(x_v_, x_o, x_o, cross_att_mask_v_ot, True) if not self.disable_cross_attention \
                and self.isolate_subnetwork != 'v' and self.isolate_subnetwork != 'o' else 0.0

            out_t = self.feed_forwards[0](x_to_ + x_tv_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_ov_ + x_s)
            out_v = self.feed_forwards[2](x_vt_ + x_vo_ + x_s)

            return out_t, out_o, out_v
