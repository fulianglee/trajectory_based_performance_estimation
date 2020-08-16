import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from util import add_checkpoint_util, add_util, get_nn_params


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:  # square shape mask
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N: int):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.h = n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        # if mask is not None:
        #     # Same mask applied to all h heads.
        mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details). https://arxiv.org/abs/1607.06450 """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Classifier(nn.Module):
    """standard classifier"""

    def __init__(self, in_size, out_size, hidden=2048):
        super(self.__class__, self).__init__()
        self.ln0 = nn.Linear(in_size, hidden)
        self.ln1 = nn.Linear(hidden, out_size)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.ac(self.ln0(x))
        return F.log_softmax(self.ln1(x), dim=-1)


class BaseModelV9(nn.Module):
    def __init__(self, num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout):
        super(BaseModelV9, self).__init__()
        self.embeddings = nn.ModuleList(
            [Embeddings(num_embed_dim, num_code) for num_code, num_embed_dim in num_dim_pairs]
        )
        self.position = PositionalEncoding(num_dim_pairs[0][1], dropout=dropout)
        d_model = sum([num_embed_dim for num_code, num_embed_dim in num_dim_pairs])

        attn = MultiHeadedAttention(n_head, d_model, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout=dropout)
        self.encoder_layers = clones(encoder_layer, n_layer)
        self.norm = LayerNorm(encoder_layer.size)
        self.clf = Classifier(d_model, n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, xs, mask, sizes):
        # xs :: [n_batch, seq_len, n_feature]
        embeds = torch.cat([self.position(
            embedding(xs.select(-1, i))
        ) if i == 0 else embedding(
            xs.select(-1, i)
        ) for i, embedding in enumerate(self.embeddings)], dim=-1)

        for layer in self.encoder_layers:
            embeds = layer(embeds, mask)
        x = self.norm(embeds)

        return self.clf(x[torch.arange(len(sizes)), sizes])


class BaseModelV9a(nn.Module):
    def __init__(self, num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout):
        super(BaseModelV9a, self).__init__()
        self.embeddings = nn.ModuleList(
            [Embeddings(num_embed_dim, num_code) for num_code, num_embed_dim in num_dim_pairs]
        )
        self.position = PositionalEncodingV2(num_dim_pairs[0][1], dropout=dropout)
        d_model = sum([num_embed_dim for num_code, num_embed_dim in num_dim_pairs])

        attn = MultiHeadedAttention(n_head, d_model, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout=dropout)
        self.encoder_layers = clones(encoder_layer, n_layer)
        self.norm = LayerNorm(encoder_layer.size)
        self.clf = Classifier(d_model, n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, xs, mask, sizes):
        # xs :: [n_batch, seq_len, n_feature]
        embeds = torch.cat([self.position(
            embedding(xs.select(-1, i)), sizes
        ) if i == 0 else embedding(
            xs.select(-1, i)
        ) for i, embedding in enumerate(self.embeddings)], dim=-1)

        for layer in self.encoder_layers:
            embeds = layer(embeds, mask)
        x = self.norm(embeds)

        return self.clf(x[torch.arange(len(sizes)), sizes])


class BaseModelV9b(nn.Module):
    """positional encoding reversed + weight sharing """

    def __init__(self, num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout):
        super(BaseModelV9b, self).__init__()
        self.embeddings = nn.ModuleList(
            [Embeddings(num_embed_dim, num_code) for num_code, num_embed_dim in num_dim_pairs]
        )
        self.position = PositionalEncodingV2(num_dim_pairs[0][1], dropout=dropout)
        self.n_layer = n_layer

        d_model = sum([num_embed_dim for num_code, num_embed_dim in num_dim_pairs])
        attn = MultiHeadedAttention(n_head, d_model, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout=dropout)
        self.norm = LayerNorm(self.encoder_layer.size)
        self.clf = Classifier(d_model, n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, xs, mask, sizes):
        # xs :: [n_batch, seq_len, n_feature]
        embeds = torch.cat([self.position(
            embedding(xs.select(-1, i)), sizes
        ) if i == 0 else embedding(
            xs.select(-1, i)
        ) for i, embedding in enumerate(self.embeddings)], dim=-1)

        for _ in range(self.n_layer):
            embeds = self.encoder_layer(embeds, mask)
        x = self.norm(embeds)

        return self.clf(x[torch.arange(len(sizes)), sizes])


class PositionalEncodingV2(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingV2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.flip(0)
        self.register_buffer('pe', pe)

    def forward(self, x, sizes):
        pos = torch.zeros_like(x)
        for i, j in enumerate(sizes):
            pos[i, :j] = self.pe[-j:]
        x = x + pos
        return self.dropout(x)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def subsequent_mask_batch(sizes):
    "Mask out subsequent positions."
    size = max(sizes)
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_masks = [subsequent_mask.copy() for _ in sizes]
    for s, mask in zip(sizes, subsequent_masks):
        mask[:, s:] = 1
    subsequent_masks = np.vstack(subsequent_masks)
    return torch.from_numpy(subsequent_masks) == 0
