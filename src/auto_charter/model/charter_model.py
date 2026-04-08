"""AutoCharterModel — full encoder-decoder transformer for chart generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from auto_charter.model.audio_encoder import AudioEncoder
from auto_charter.model.config import AutoCharterConfig
from auto_charter.model.token_decoder import AutoregressiveDecoder
from auto_charter.vocab.tokens import Vocab


@dataclass
class AutoCharterOutput:
    loss: Optional[Tensor]    # scalar, None during inference
    logits: Tensor            # [B, T, vocab_size]
    encoder_hidden: Tensor    # [B, N, d_model] — audio context


class AutoCharterModel(nn.Module):
    def __init__(self, config: AutoCharterConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = AutoregressiveDecoder(config)

    def forward(
        self,
        # Encoder inputs
        mert_embeddings: Tensor,        # [B, N, 768]
        logmel_frames: Tensor,          # [B, N, 32, 128]
        bpm_at_beat: Tensor,            # [B, N]
        time_sig_num: Tensor,           # [B, N] int
        time_sig_den: Tensor,           # [B, N] int
        beat_duration_s: Tensor,        # [B, N]
        beat_padding_mask: Tensor,      # [B, N] bool, True=valid
        # Decoder inputs
        input_ids: Tensor,              # [B, T]
        beat_ids: Tensor,               # [B, T]
        instrument_ids: Tensor,         # [B]
        difficulty_ids: Tensor,         # [B]
        # Optional: labels for loss computation
        labels: Optional[Tensor] = None,  # [B, T]
        # Unused but accepted for compatibility with collator output
        attention_mask: Optional[Tensor] = None,
        beat_attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> AutoCharterOutput:
        # Use beat_attention_mask as beat_padding_mask if provided
        if beat_attention_mask is not None:
            beat_padding_mask = beat_attention_mask

        audio_context = self.encoder(
            mert_embeddings=mert_embeddings,
            logmel_frames=logmel_frames,
            bpm_at_beat=bpm_at_beat,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            beat_duration_s=beat_duration_s,
            beat_padding_mask=beat_padding_mask,
        )

        logits = self.decoder(
            input_ids=input_ids,
            beat_ids=beat_ids,
            audio_context=audio_context,
            instrument_ids=instrument_ids,
            difficulty_ids=difficulty_ids,
            beat_padding_mask=beat_padding_mask,
        )

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )

        return AutoCharterOutput(loss=loss, logits=logits, encoder_hidden=audio_context)

    @torch.no_grad()
    def generate(
        self,
        mert_embeddings: Tensor,        # [1, N, 768]
        logmel_frames: Tensor,          # [1, N, 32, 128]
        bpm_at_beat: Tensor,            # [1, N]
        time_sig_num: Tensor,           # [1, N]
        time_sig_den: Tensor,           # [1, N]
        beat_duration_s: Tensor,        # [1, N]
        beat_padding_mask: Tensor,      # [1, N]
        instrument_id: int,
        difficulty_id: int,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[int]:
        """Autoregressive generation. Returns list of token IDs (including BOS/EOS)."""
        device = mert_embeddings.device
        self.eval()

        # Run encoder once
        audio_context = self.encoder(
            mert_embeddings=mert_embeddings,
            logmel_frames=logmel_frames,
            bpm_at_beat=bpm_at_beat,
            time_sig_num=time_sig_num,
            time_sig_den=time_sig_den,
            beat_duration_s=beat_duration_s,
            beat_padding_mask=beat_padding_mask,
        )  # [1, N, d]

        instr_tensor = torch.tensor([instrument_id], device=device)
        diff_tensor = torch.tensor([difficulty_id], device=device)

        # Start with BOS
        generated = [Vocab.BOS]
        current_beat_id = 0
        max_beats = audio_context.shape[1] - 1

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)  # [1, T]
            beat_ids_t = torch.tensor(
                [self._compute_beat_ids(generated, max_beats)],
                dtype=torch.long,
                device=device,
            )  # [1, T]

            logits = self.decoder(
                input_ids=input_ids,
                beat_ids=beat_ids_t,
                audio_context=audio_context,
                instrument_ids=instr_tensor,
                difficulty_ids=diff_tensor,
                beat_padding_mask=beat_padding_mask,
            )  # [1, T, V]

            next_token_logits = logits[0, -1, :]  # [V]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                next_token_logits = self._top_k_filter(next_token_logits, top_k)
            if top_p < 1.0:
                next_token_logits = self._top_p_filter(next_token_logits, top_p)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(int(next_token))

            if next_token == Vocab.EOS:
                break

        return generated

    @staticmethod
    def _compute_beat_ids(tokens: list[int], max_beat: int) -> list[int]:
        """Scan token sequence and assign beat_id to each position."""
        beat_id = 0
        result = []
        for tok in tokens:
            result.append(min(beat_id, max_beat))
            if tok == Vocab.BEAT_BOUNDARY:
                beat_id += 1
        return result

    @staticmethod
    def _top_k_filter(logits: Tensor, k: int) -> Tensor:
        values, _ = torch.topk(logits, k)
        threshold = values[-1]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logits: Tensor, p: float) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > p
        sorted_logits[remove_mask] = float("-inf")
        logits_out = torch.full_like(logits, float("-inf"))
        logits_out.scatter_(0, sorted_indices, sorted_logits)
        return logits_out

    def save_pretrained(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.json")
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: Path | str) -> "AutoCharterModel":
        path = Path(path)
        config = AutoCharterConfig.load(path / "config.json")
        model = cls(config)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
