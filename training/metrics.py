from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import Tensor


# ── Token-level helpers ───────────────────────────────────────────────────────

def compute_token_accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Fraction of non-PAD tokens predicted correctly.

    Args:
        logits: [B, S, vocab_size]
        labels: [B, S]  (-100 for ignored positions)
    """
    preds = logits.argmax(dim=-1)           # [B, S]
    mask = labels != -100                   # [B, S]
    correct = (preds == labels) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from mean cross-entropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_sequence_accuracy(
    pred_ids: List[List[int]],
    target_ids: List[List[int]],
    eos_id: int = 1,
) -> float:
    """
    Fraction of sequences that are an exact match (token-for-token).
    Sequences are truncated at the first EOS before comparison.
    """
    if not pred_ids:
        return 0.0
    matches = 0
    for pred, tgt in zip(pred_ids, target_ids):
        pred_trunc = _truncate_at_eos(pred, eos_id)
        tgt_trunc = _truncate_at_eos(tgt, eos_id)
        if pred_trunc == tgt_trunc:
            matches += 1
    return matches / len(pred_ids)


# ── Note-level helpers ────────────────────────────────────────────────────────

def decode_to_note_blocks(token_ids: List[int], bos_id: int = 0, eos_id: int = 1) -> List[Tuple]:
    """
    Parse a flat token ID sequence into note tuples.

    Expected sequence structure after BOS + Instrument + Difficulty:
        [Beatshift_X, NoteType, Pitch, Minute, Beat, Beatshift_D]*

    Returns a list of (beatshift, note_type_id, pitch_id, minute, beat, duration_beatshift)
    tuples.  Tokens that do not fit the 6-token block structure are skipped.
    """
    # Strip BOS, EOS, instrument, difficulty tokens (first 3 meaningful tokens)
    try:
        start = token_ids.index(bos_id) + 1
    except ValueError:
        start = 0
    try:
        end = token_ids.index(eos_id, start)
    except ValueError:
        end = len(token_ids)

    body = token_ids[start:end]
    # Skip the instrument and difficulty tokens at the beginning (2 tokens)
    if len(body) >= 2:
        body = body[2:]

    notes = []
    i = 0
    while i + 5 < len(body):
        block = body[i : i + 6]
        notes.append(tuple(block))
        i += 6
    return notes


def compute_note_f1(
    pred_ids: List[int],
    target_ids: List[int],
    bos_id: int = 0,
    eos_id: int = 1,
) -> dict:
    """
    Note-level Precision, Recall, F1.

    A "note" is identified by (note_type_id, pitch_id) — ignoring timing so
    that the metric measures whether the model predicts the right buttons/pads.
    """
    pred_notes = decode_to_note_blocks(pred_ids, bos_id, eos_id)
    tgt_notes = decode_to_note_blocks(target_ids, bos_id, eos_id)

    # Use (note_type, pitch) as the note identity
    pred_set = _note_multiset(pred_notes)
    tgt_set = _note_multiset(tgt_notes)

    tp = sum(min(pred_set.get(k, 0), tgt_set.get(k, 0)) for k in tgt_set)
    precision = tp / (sum(pred_set.values()) + 1e-8)
    recall = tp / (sum(tgt_set.values()) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}


def compute_timing_accuracy(
    pred_ids: List[int],
    target_ids: List[int],
    tolerance_beats: float = 0.5,
    bos_id: int = 0,
    eos_id: int = 1,
) -> float:
    """
    Fraction of target notes that are matched by a predicted note within
    ``tolerance_beats`` beats (same pitch, timing within tolerance).

    Timing is reconstructed from (minute, beat) token values.
    """
    pred_notes = decode_to_note_blocks(pred_ids, bos_id, eos_id)
    tgt_notes = decode_to_note_blocks(target_ids, bos_id, eos_id)

    if not tgt_notes:
        return 1.0

    matched = 0
    for tgt in tgt_notes:
        _, _, tgt_pitch, tgt_minute, tgt_beat, _ = tgt
        tgt_time = tgt_minute * 60 + tgt_beat
        for pred in pred_notes:
            _, _, pred_pitch, pred_minute, pred_beat, _ = pred
            if pred_pitch == tgt_pitch:
                pred_time = pred_minute * 60 + pred_beat
                if abs(pred_time - tgt_time) <= tolerance_beats:
                    matched += 1
                    break

    return matched / len(tgt_notes)


# ── Aggregate validation loop ─────────────────────────────────────────────────

def compute_validation_metrics(
    all_logits: List[Tensor],
    all_labels: List[Tensor],
    total_loss: float,
    n_batches: int,
    bos_id: int = 0,
    eos_id: int = 1,
) -> dict:
    """
    Aggregate all metrics over the full validation set.

    Args:
        all_logits:  List of [B, S, vocab] tensors
        all_labels:  List of [B, S] label tensors (-100 at pads)
        total_loss:  Accumulated CE loss (sum, not mean)
        n_batches:   Number of batches accumulated
    """
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = compute_perplexity(avg_loss)

    # Token accuracy over all batches
    all_correct = 0
    all_total = 0
    for logits, labels in zip(all_logits, all_labels):
        mask = labels != -100
        preds = logits.argmax(dim=-1)
        all_correct += ((preds == labels) & mask).sum().item()
        all_total += mask.sum().item()
    token_acc = all_correct / max(all_total, 1)

    # Note-level metrics — compute on greedy predictions, first sample of each batch
    note_f1_scores, timing_scores = [], []
    seq_matches, seq_total = 0, 0

    for logits, labels in zip(all_logits, all_labels):
        pred_ids_batch = logits.argmax(dim=-1).tolist()   # [B, S]
        label_ids_batch = labels.tolist()                  # [B, S]

        for pred_ids, label_ids in zip(pred_ids_batch, label_ids_batch):
            # Reconstruct label without -100
            tgt = [t for t in label_ids if t != -100]
            f1 = compute_note_f1(pred_ids, tgt, bos_id, eos_id)
            note_f1_scores.append(f1["f1"])
            timing_scores.append(
                compute_timing_accuracy(pred_ids, tgt, bos_id=bos_id, eos_id=eos_id)
            )
            seq_matches += int(
                _truncate_at_eos(pred_ids, eos_id) == _truncate_at_eos(tgt, eos_id)
            )
            seq_total += 1

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": token_acc,
        "sequence_accuracy": seq_matches / max(seq_total, 1),
        "note_f1": sum(note_f1_scores) / max(len(note_f1_scores), 1),
        "timing_accuracy": sum(timing_scores) / max(len(timing_scores), 1),
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _truncate_at_eos(ids: List[int], eos_id: int) -> List[int]:
    try:
        return ids[: ids.index(eos_id) + 1]
    except ValueError:
        return ids


def _note_multiset(notes: List[Tuple]) -> dict:
    counts: dict = {}
    for note in notes:
        if len(note) >= 3:
            key = (note[1], note[2])  # (note_type_id, pitch_id)
            counts[key] = counts.get(key, 0) + 1
    return counts
