"""Guitar/Bass chord bitmask ↔ token ID conversion.

Lanes (bit positions):
  bit 0 = Green  (pitch 0)
  bit 1 = Red    (pitch 1)
  bit 2 = Yellow (pitch 2)
  bit 3 = Blue   (pitch 3)
  bit 4 = Orange (pitch 4)

Bitmask range: 1..31 (0 = no note, never emitted)
Token IDs: 57..87  (GUITAR_NOTE_START + bitmask - 1)
"""

from .tokens import Vocab

LANE_PITCHES = [0, 1, 2, 3, 4]  # Green, Red, Yellow, Blue, Orange


def pitches_to_bitmask(pitches: set[int]) -> int:
    """Convert a set of lane pitches (0-4) to a bitmask."""
    mask = 0
    for p in pitches:
        if 0 <= p <= 4:
            mask |= 1 << p
    return mask


def bitmask_to_pitches(bitmask: int) -> frozenset[int]:
    """Convert a bitmask back to a frozenset of lane pitches."""
    return frozenset(i for i in range(5) if bitmask & (1 << i))


def chord_bitmask_to_id(bitmask: int) -> int:
    """Return the guitar/bass token ID for a given chord bitmask (1..31)."""
    assert 1 <= bitmask <= 31, f"Invalid chord bitmask: {bitmask}"
    return Vocab.GUITAR_NOTE_START + bitmask - 1


def id_to_chord_bitmask(token_id: int) -> int:
    """Return the chord bitmask (1..31) for a given guitar/bass token ID."""
    assert Vocab.is_guitar_note(token_id), f"Not a guitar note token: {token_id}"
    return token_id - Vocab.GUITAR_NOTE_START + 1
