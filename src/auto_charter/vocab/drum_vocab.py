"""Drum chord bitmask ↔ token ID conversion.

Lanes (bit positions):
  bit 0 = Kick   (pitch 0)
  bit 1 = Snare  (pitch 1)
  bit 2 = Hi-hat (pitch 2)
  bit 3 = Tom    (pitch 3)
  bit 4 = Cymbal (pitch 4)

Bitmask range: 1..31 (0 = no drum hit, never emitted)
Token IDs: 92..122  (DRUM_NOTE_START + bitmask - 1)

Note: drums have no sustain in Clone Hero — the sustain field is always 0 and ignored.
"""

from .tokens import Vocab


def drum_pitches_to_bitmask(pitches: set[int]) -> int:
    """Convert a set of drum pitches (0-4) to a bitmask."""
    mask = 0
    for p in pitches:
        if 0 <= p <= 4:
            mask |= 1 << p
    return mask


def drum_bitmask_to_pitches(bitmask: int) -> frozenset[int]:
    """Convert a bitmask back to a frozenset of drum pitches."""
    return frozenset(i for i in range(5) if bitmask & (1 << i))


def drum_bitmask_to_id(bitmask: int) -> int:
    """Return the drum token ID for a given drum bitmask (1..31)."""
    assert 1 <= bitmask <= 31, f"Invalid drum bitmask: {bitmask}"
    return Vocab.DRUM_NOTE_START + bitmask - 1


def id_to_drum_bitmask(token_id: int) -> int:
    """Return the drum bitmask (1..31) for a given drum token ID."""
    assert Vocab.is_drum_note(token_id), f"Not a drum note token: {token_id}"
    return token_id - Vocab.DRUM_NOTE_START + 1
