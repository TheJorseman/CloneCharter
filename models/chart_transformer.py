from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from models.cnn_frontend import AudioCNNFrontEnd
from models.conditioning import ConditioningEncoder


@dataclass
class ModelConfig:
    d_model: int = 768
    enc_layers: int = 8
    dec_layers: int = 12
    n_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    vocab_size: int = 693
    enc_max_len: int = 512      # max encoder sequence length (7 prefix + 505 audio tokens)
    dec_max_len: int = 2048     # max decoder sequence length
    mert_dim: int = 768
    pad_token_id: int = 3       # <PAD>
    bos_token_id: int = 0       # <BOS>
    eos_token_id: int = 1       # <EOS>
    label_smoothing: float = 0.1
    # Gradient checkpointing: apply every N layers (0 = disabled)
    enc_ckpt_every: int = 2
    dec_ckpt_every: int = 3


class ChartTransformer(nn.Module):
    """
    Encoder-Decoder transformer for Clone Hero chart generation.

    Encoder:
        - AudioCNNFrontEnd collapses log-mel [B, 512, T] → [B, T//16, d_model]
        - ConditioningEncoder provides 7 prefix tokens from metadata
        - TransformerEncoder with bidirectional self-attention

    Decoder:
        - Autoregressive over vocab_size tokens
        - Cross-attention to encoder output
        - Weight tying between token embedding and output projection
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Encoder components ──────────────────────────────────────────────
        self.cnn_frontend = AudioCNNFrontEnd(config.d_model)
        self.conditioning = ConditioningEncoder(config.d_model, config.mert_dim)
        self.enc_pos_emb = nn.Embedding(config.enc_max_len, config.d_model)
        self.enc_dropout = nn.Dropout(config.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.enc_layers)

        # ── Decoder components ──────────────────────────────────────────────
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dec_pos_emb = nn.Embedding(config.dec_max_len, config.d_model)
        self.dec_dropout = nn.Dropout(config.dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.dec_layers)

        # Output projection with weight tying
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        std = self.config.d_model ** -0.5
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.enc_pos_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.dec_pos_emb.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.tok_emb.weight:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Encoder ─────────────────────────────────────────────────────────────

    def encode(
        self,
        log_mel: Tensor,            # [B, 512, T_padded]
        mert_emb: Tensor,           # [B, mert_dim]
        bpm: Tensor,                # [B, 1]
        ts: Tensor,                 # [B, 1]
        resolution: Tensor,         # [B, 1]
        offset: Tensor,             # [B, 1]
        instrument_idx: Tensor,     # [B]
        difficulty_idx: Tensor,     # [B]
        enc_padding_mask: Optional[Tensor] = None,  # [B, enc_max_len] True=ignore
    ) -> Tensor:
        """Returns enc_out: [B, enc_max_len, d_model]"""
        # CNN: [B, 512, T] → [B, T//16, d_model]
        audio_tokens = self.cnn_frontend(log_mel)  # [B, S_audio, d_model]

        # Conditioning prefix: [B, 7, d_model]
        prefix = self.conditioning(
            mert_emb, bpm, ts, resolution, offset, instrument_idx, difficulty_idx
        )

        # Concatenate prefix + audio tokens → [B, 7 + S_audio, d_model]
        x = torch.cat([prefix, audio_tokens], dim=1)

        # Truncate or pad to enc_max_len
        B, S, D = x.shape
        if S > self.config.enc_max_len:
            x = x[:, : self.config.enc_max_len, :]
        elif S < self.config.enc_max_len:
            pad = torch.zeros(B, self.config.enc_max_len - S, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # Positional embedding
        positions = torch.arange(self.config.enc_max_len, device=x.device).unsqueeze(0)
        x = self.enc_dropout(x + self.enc_pos_emb(positions))

        # Encoder forward with optional gradient checkpointing
        if self.config.enc_ckpt_every > 0 and self.training:
            x = self._encoder_with_ckpt(x, enc_padding_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=enc_padding_mask)

        return x  # [B, enc_max_len, d_model]

    def _encoder_with_ckpt(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        for i, layer in enumerate(self.encoder.layers):
            if i % self.config.enc_ckpt_every == 0:
                x = checkpoint(layer, x, None, mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=mask)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x

    # ── Decoder ─────────────────────────────────────────────────────────────

    def decode(
        self,
        tgt_tokens: Tensor,                     # [B, S]
        enc_out: Tensor,                        # [B, enc_max_len, d_model]
        enc_padding_mask: Optional[Tensor],     # [B, enc_max_len]
        tgt_key_padding_mask: Optional[Tensor] = None,  # [B, S]
    ) -> Tensor:
        """Returns logits: [B, S, vocab_size]"""
        B, S = tgt_tokens.shape

        positions = torch.arange(S, device=tgt_tokens.device).unsqueeze(0)
        x = self.dec_dropout(self.tok_emb(tgt_tokens) + self.dec_pos_emb(positions))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        if self.config.dec_ckpt_every > 0 and self.training:
            x = self._decoder_with_ckpt(x, enc_out, tgt_mask, enc_padding_mask, tgt_key_padding_mask)
        else:
            x = self.decoder(
                x,
                enc_out,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=enc_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        return self.output_proj(x)  # [B, S, vocab_size]

    def _decoder_with_ckpt(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        for i, layer in enumerate(self.decoder.layers):
            if i % self.config.dec_ckpt_every == 0:
                x = checkpoint(
                    layer, x, memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    # ── Full forward (training) ──────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Expects a batch produced by ChartCollator.
        Returns {'logits': [B, S, vocab_size], 'loss': scalar}.
        """
        enc_out = self.encode(
            log_mel=batch["log_mel"],
            mert_emb=batch["mert_emb"],
            bpm=batch["bpm"],
            ts=batch["ts"],
            resolution=batch["resolution"],
            offset=batch["offset"],
            instrument_idx=batch["instrument_idx"],
            difficulty_idx=batch["difficulty_idx"],
            enc_padding_mask=batch.get("enc_padding_mask"),
        )

        logits = self.decode(
            tgt_tokens=batch["decoder_input_ids"],
            enc_out=enc_out,
            enc_padding_mask=batch.get("enc_padding_mask"),
            tgt_key_padding_mask=batch.get("decoder_attention_mask") == 0
            if batch.get("decoder_attention_mask") is not None
            else None,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            batch["decoder_labels"].reshape(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

        return {"logits": logits, "loss": loss}

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "ChartTransformer":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
