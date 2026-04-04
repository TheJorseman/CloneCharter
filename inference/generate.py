"""
Autoregressive generation: greedy decode and beam search.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def greedy_decode(
    model,
    enc_out: Tensor,
    enc_padding_mask: Optional[Tensor],
    bos_id: int = 0,
    eos_id: int = 1,
    pad_id: int = 3,
    max_new_tokens: int = 2044,
) -> List[int]:
    """
    Greedy autoregressive decoding.

    Args:
        model:             ChartTransformer (eval mode)
        enc_out:           [1, enc_len, d_model]
        enc_padding_mask:  [1, enc_len]
        max_new_tokens:    maximum tokens to generate after BOS

    Returns:
        List of token IDs (including BOS and EOS)
    """
    device = enc_out.device
    tokens = [bos_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(tgt, enc_out, enc_padding_mask)  # [1, S, vocab]
            next_id = logits[0, -1].argmax().item()
            tokens.append(next_id)
            if next_id == eos_id:
                break

    return tokens


def beam_search(
    model,
    enc_out: Tensor,
    enc_padding_mask: Optional[Tensor],
    bos_id: int = 0,
    eos_id: int = 1,
    pad_id: int = 3,
    beam_size: int = 4,
    max_new_tokens: int = 2044,
    length_penalty: float = 0.6,
) -> List[int]:
    """
    Beam search decoding.

    Args:
        beam_size:       number of parallel hypotheses
        length_penalty:  score = log_prob / (length ** length_penalty)

    Returns:
        Token IDs of the best beam (including BOS and EOS)
    """
    device = enc_out.device

    # Each beam: (token_ids, cumulative_log_prob)
    beams: List[tuple] = [([bos_id], 0.0)]
    completed: List[tuple] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if not beams:
                break

            candidates: List[tuple] = []

            for tokens, score in beams:
                if tokens[-1] == eos_id:
                    completed.append((tokens, score))
                    continue

                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model.decode(tgt, enc_out, enc_padding_mask)   # [1, S, vocab]
                log_probs = F.log_softmax(logits[0, -1], dim=-1)        # [vocab]

                topk = log_probs.topk(beam_size * 2)
                for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((tokens + [tok], score + lp))

            if not candidates:
                break

            # Keep top beam_size by length-penalised score
            candidates.sort(
                key=lambda x: x[1] / max(len(x[0]) ** length_penalty, 1e-8),
                reverse=True,
            )
            beams = candidates[:beam_size]

            # Early exit if all beams ended with EOS
            if all(b[0][-1] == eos_id for b in beams):
                completed.extend(beams)
                break

    completed.extend(beams)
    if not completed:
        return [bos_id, eos_id]

    best = max(
        completed,
        key=lambda x: x[1] / max(len(x[0]) ** length_penalty, 1e-8),
    )
    return best[0]
