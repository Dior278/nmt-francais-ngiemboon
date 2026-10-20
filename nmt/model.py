"""Encoder–decoder Transformer, sized for a few thousand parallel sentences.

Pre-norm, GELU, and tied embeddings: that combination trains more stably
and generalizes better here than a wider post-norm network.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep positions where mask is true.

    -1e9 does not fit in float16, so AMP on GPU crashes in masked_fill.
    A fully blocked row would also make softmax NaN; those scores stay 0.
    """
    allowed = mask if mask.dtype == torch.bool else mask != 0
    scores = scores.masked_fill(~allowed, float("-inf"))
    fully_blocked = ~allowed.any(dim=-1, keepdim=True)
    return scores.masked_fill(fully_blocked, 0.0)


def causal_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True where a decoder query may read a key. Shape (batch, 1, time, time)."""
    valid_key = tokens.ne(pad_id).unsqueeze(1).unsqueeze(2)
    length = tokens.size(1)
    causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
    return valid_key & causal


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def combine(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, -1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.split(self.W_q(q)), self.split(self.W_k(k)), self.split(self.W_v(v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = _mask_scores(scores, mask)
        weights = self.drop(torch.softmax(scores, dim=-1))
        return self.W_o(self.combine(torch.matmul(weights, v)))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask))
        h = self.norm2(x)
        return x + self.dropout(self.ff(h))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, src_mask))
        h = self.norm3(x)
        return x + self.dropout(self.ff(h))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # * sqrt(d_model) in encode() expects embeddings of std d_model^-0.5,
        # otherwise the first logits explode and the loss stays meaningless.
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()

    def max_positions(self) -> int:
        return int(self.pe.pe.size(1))

    def make_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        # Key padding, not query padding: a pad token must not be read.
        src_mask = src.ne(self.pad_id).unsqueeze(1).unsqueeze(2)
        return src_mask, causal_mask(tgt, self.pad_id)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pe(self.embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.dropout(self.pe(self.embed(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return F.linear(self.dec_norm(x), self.embed.weight)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.make_masks(src, tgt)
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, tgt_mask)
