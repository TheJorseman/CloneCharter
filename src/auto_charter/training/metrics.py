"""Training and evaluation metrics for AutoCharterModel.

Metrics:
  token_accuracy  — fraction of non-PAD tokens where argmax == label
  perplexity      — exp(cross_entropy_loss) on non-PAD tokens
  note_f1         — note-level F1 score (requires decoding token sequences)
  beat_accuracy   — fraction of beats with matching note sets
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from auto_charter.vocab.tokens import Vocab


def token_accuracy(
    logits: Tensor,     # [B, T, V]
    labels: Tensor,     # [B, T]
    pad_id: int = Vocab.PAD,
) -> float:
    """Fraction of non-PAD label positions where argmax(logits) matches label."""
    # Shift: logits[:-1] predicts labels[1:]
    shift_logits = logits[:, :-1, :]    # [B, T-1, V]
    shift_labels = labels[:, 1:]        # [B, T-1]

    preds = shift_logits.argmax(dim=-1)  # [B, T-1]
    mask = shift_labels != pad_id        # [B, T-1] bool
    if mask.sum() == 0:
        return 0.0
    correct = ((preds == shift_labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


def perplexity(
    logits: Tensor,     # [B, T, V]
    labels: Tensor,     # [B, T]
    pad_id: int = Vocab.PAD,
) -> float:
    """exp(cross_entropy) on non-PAD tokens. Lower = better."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_id, reduction="mean")
    return torch.exp(loss).item()


def note_f1(
    predicted_token_seqs: list[list[int]],
    target_token_seqs: list[list[int]],
) -> dict[str, float]:
    """Note-level F1 score.

    For each sequence pair, decodes tokens to (tick, frozenset[pitches]) note events,
    then computes precision/recall/F1 by exact match on (tick, pitches).

    Returns:
        {"note_f1": float, "note_precision": float, "note_recall": float}
    """
    from auto_charter.tokenizer.decoder import decode_tokens

    total_tp = 0
    total_pred = 0
    total_target = 0

    for pred_tokens, tgt_tokens in zip(predicted_token_seqs, target_token_seqs):
        pred_notes = _decode_to_note_set(pred_tokens, decode_tokens)
        tgt_notes = _decode_to_note_set(tgt_tokens, decode_tokens)

        tp = len(pred_notes & tgt_notes)
        total_tp += tp
        total_pred += len(pred_notes)
        total_target += len(tgt_notes)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_target if total_target > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"note_f1": f1, "note_precision": precision, "note_recall": recall}


def beat_accuracy(
    predicted_token_seqs: list[list[int]],
    target_token_seqs: list[list[int]],
) -> float:
    """Fraction of beats where the note sets match exactly.

    A beat is the token span between two consecutive BEAT_BOUNDARY tokens.
    """
    from auto_charter.tokenizer.decoder import decode_tokens

    total_beats = 0
    correct_beats = 0

    for pred_tokens, tgt_tokens in zip(predicted_token_seqs, target_token_seqs):
        pred_beats = _split_into_beats(pred_tokens)
        tgt_beats = _split_into_beats(tgt_tokens)

        n = min(len(pred_beats), len(tgt_beats))
        total_beats += max(len(pred_beats), len(tgt_beats))
        for pb, tb in zip(pred_beats[:n], tgt_beats[:n]):
            pred_notes = _beat_notes(pb)
            tgt_notes = _beat_notes(tb)
            if pred_notes == tgt_notes:
                correct_beats += 1

    if total_beats == 0:
        return 0.0
    return correct_beats / total_beats


def compute_all(
    logits: Tensor,
    labels: Tensor,
    pad_id: int = Vocab.PAD,
    greedy_tokens: list[list[int]] | None = None,
    target_tokens: list[list[int]] | None = None,
) -> dict[str, float]:
    """Compute all available metrics.

    Args:
        logits: [B, T, V] model output logits
        labels: [B, T] target token ids
        greedy_tokens: Optional pre-decoded greedy token sequences for note_f1/beat_acc
        target_tokens: Optional target token sequences for note_f1/beat_acc
    """
    metrics: dict[str, float] = {
        "token_accuracy": token_accuracy(logits, labels, pad_id),
        "perplexity": perplexity(logits, labels, pad_id),
    }
    if greedy_tokens is not None and target_tokens is not None:
        f1_metrics = note_f1(greedy_tokens, target_tokens)
        metrics.update(f1_metrics)
        metrics["beat_accuracy"] = beat_accuracy(greedy_tokens, target_tokens)
    return metrics


# ── Internal helpers ───────────────────────────────────────────────────────────

def _decode_to_note_set(tokens: list[int], decode_fn) -> frozenset[tuple]:
    """Decode tokens and return a frozenset of (tick, pitches_frozenset) pairs."""
    try:
        chart = decode_fn(tokens)
        result = set()
        for track_notes in chart.tracks.values():
            for note in track_notes:
                result.add((note.tick, frozenset(note.pitches)))
        return frozenset(result)
    except Exception:
        return frozenset()


def _split_into_beats(tokens: list[int]) -> list[list[int]]:
    """Split token sequence into per-beat spans at BEAT_BOUNDARY tokens."""
    beats = []
    current: list[int] = []
    for tok in tokens:
        if tok == Vocab.BEAT_BOUNDARY and current:
            beats.append(current)
            current = []
        current.append(tok)
    if current:
        beats.append(current)
    return beats


def _beat_notes(tokens: list[int]) -> frozenset[int]:
    """Extract note token IDs from a single beat span."""
    notes = set()
    for tok in tokens:
        if Vocab.GUITAR_NOTE_START <= tok <= Vocab.GUITAR_NOTE_END:
            notes.add(tok)
        elif Vocab.DRUM_NOTE_START <= tok <= Vocab.DRUM_NOTE_END:
            notes.add(tok)
    return frozenset(notes)
