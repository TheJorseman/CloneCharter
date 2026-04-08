"""AutoregressiveDecoder — causal transformer decoder for chart token generation.

Conditioning:
  - instrument_ids [B]: 0=guitar, 1=bass, 2=drums → Embedding added to all positions
  - difficulty_ids [B]: 0–6 → Embedding added to all positions

Beat alignment:
  - beat_ids [B, T]: maps each token position to its corresponding beat in audio_context
  - Used to add a learnable ALiBi-like distance bias in cross-attention:
      attn_bias[b, t, n] = -exp(log_distance_scale) * |beat_ids[b,t] - n|
    This gives a soft prior toward the current beat without hard-selecting it.

Architecture:
  - Pre-norm (LayerNorm before each sub-layer) for stable training from scratch
  - Flash Attention 2 via scaled_dot_product_attention for causal self-attention
  - Standard nn.MultiheadAttention for cross-attention (with beat distance bias)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from auto_charter.model.config import AutoCharterConfig


class _DecoderLayer(nn.Module):
    """Single pre-norm decoder layer with causal self-attn + cross-attn."""

    def __init__(self, config: AutoCharterConfig) -> None:
        super().__init__()
        d = config.d_model

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        # Self-attention (causal) — we use F.scaled_dot_product_attention manually
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

        # Cross-attention (to audio encoder output)
        self.cross_q = nn.Linear(d, d, bias=False)
        self.cross_k = nn.Linear(d, d, bias=False)
        self.cross_v = nn.Linear(d, d, bias=False)
        self.cross_out = nn.Linear(d, d, bias=False)

        # Feed-forward
        self.ff1 = nn.Linear(d, config.d_ff)
        self.ff2 = nn.Linear(config.d_ff, d)

        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, T, d] → [B, n_heads, T, head_dim]"""
        B, T, d = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """[B, n_heads, T, head_dim] → [B, T, d]"""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        x: Tensor,                  # [B, T, d]
        audio_context: Tensor,      # [B, N, d]
        beat_distance_bias: Tensor, # [B, n_heads, T, N]
        key_padding_mask: Tensor,   # [B, N] bool, True = valid beat (NOT pad)
    ) -> Tensor:
        # 1. Causal self-attention (Flash Attention 2 path via SDPA)
        residual = x
        x = self.norm1(x)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        # is_causal=True automatically applies causal mask
        x_sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        x = residual + self.dropout(self.out_proj(self._merge_heads(x_sa)))

        # 2. Cross-attention to audio_context with beat distance bias
        residual = x
        x = self.norm2(x)
        cq = self._split_heads(self.cross_q(x))               # [B, H, T, hd]
        ck = self._split_heads(self.cross_k(audio_context))   # [B, H, N, hd]
        cv = self._split_heads(self.cross_v(audio_context))   # [B, H, N, hd]

        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(cq, ck.transpose(-2, -1)) / scale  # [B, H, T, N]
        attn_logits = attn_logits + beat_distance_bias

        # Mask padded beats (True=valid → flip for additive mask)
        # key_padding_mask: [B, N] True=valid → expand to [B, 1, 1, N]
        pad_mask = ~key_padding_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        # Handle all-inf rows (full padding) → zero
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        x_ca = torch.matmul(attn_weights, cv)                 # [B, H, T, hd]
        x = residual + self.dropout(self.cross_out(self._merge_heads(x_ca)))

        # 3. Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff2(self.dropout(self.act(self.ff1(x))))
        x = residual + self.dropout(x)

        return x


class AutoregressiveDecoder(nn.Module):
    def __init__(self, config: AutoCharterConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        self.pos_enc = nn.Embedding(config.max_seq_len, d)

        # Conditioning embeddings (added to every token position)
        self.instr_emb = nn.Embedding(config.n_instruments, d)
        self.diff_emb = nn.Embedding(config.n_difficulties, d)

        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.n_dec_layers)])
        self.layer_norm_out = nn.LayerNorm(d)

        # Learnable distance scale for beat-distance cross-attention bias
        # Initialized so that distance of 1 beat gives penalty of ~0.1
        self.log_distance_scale = nn.Parameter(torch.tensor(math.log(0.1)))

        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)

        # Tie input/output embeddings
        self.lm_head.weight = self.token_emb.weight

    def _build_beat_distance_bias(
        self,
        beat_ids: Tensor,           # [B, T] int64
        N: int,                     # number of encoder beats
    ) -> Tensor:                    # [B, n_heads, T, N]
        """
        Compute ALiBi-like bias based on distance from each token's beat.
        bias[b, h, t, n] = -scale * |beat_ids[b,t] - n|
        """
        T = beat_ids.shape[1]
        device = beat_ids.device

        n_range = torch.arange(N, device=device, dtype=torch.float32)  # [N]
        beat_ids_f = beat_ids.float()                                    # [B, T]

        # Compute distance: [B, T, N]
        dist = torch.abs(beat_ids_f.unsqueeze(2) - n_range.unsqueeze(0).unsqueeze(0))

        scale = torch.exp(self.log_distance_scale)
        bias = -scale * dist  # [B, T, N]

        # Expand to heads: [B, n_heads, T, N]
        return bias.unsqueeze(1).expand(-1, self.config.n_heads, -1, -1)

    def forward(
        self,
        input_ids: Tensor,          # [B, T] int64
        beat_ids: Tensor,           # [B, T] int64 — beat index per token
        audio_context: Tensor,      # [B, N, d_model]
        instrument_ids: Tensor,     # [B] int64
        difficulty_ids: Tensor,     # [B] int64
        beat_padding_mask: Tensor,  # [B, N] bool, True=valid
    ) -> Tensor:                    # [B, T, vocab_size] logits
        B, T = input_ids.shape
        device = input_ids.device
        N = audio_context.shape[1]

        # Token embeddings
        tok_emb = self.token_emb(input_ids)                              # [B, T, d]
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_enc(pos_ids)                                  # [B, T, d]

        # Conditioning: instrument + difficulty, added to every position
        cond = self.instr_emb(instrument_ids) + self.diff_emb(difficulty_ids)  # [B, d]
        x = tok_emb + pos_emb + cond.unsqueeze(1)                        # [B, T, d]
        x = self.dropout(x)

        # Beat distance bias for cross-attention
        beat_bias = self._build_beat_distance_bias(beat_ids.clamp(0, N - 1), N)  # [B, H, T, N]

        for layer in self.layers:
            x = layer(x, audio_context, beat_bias, beat_padding_mask)

        x = self.layer_norm_out(x)
        return self.lm_head(x)  # [B, T, vocab_size]
