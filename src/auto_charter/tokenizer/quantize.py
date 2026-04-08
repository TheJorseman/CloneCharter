"""Tick quantization utilities.

GRID = 16 ticks per step.

With resolution=192 ticks/quarter note:
  1 step   =  16 ticks  =  1/12 quarter note
  2 steps  =  32 ticks  =  16th triplet (critical for sierreño/banda)
  3 steps  =  48 ticks  =  16th note
  6 steps  =  96 ticks  =  8th note
  12 steps = 192 ticks  =  quarter note
  48 steps = 768 ticks  =  full 4/4 measure
"""

from __future__ import annotations

from auto_charter.vocab.tokens import Vocab

GRID = 16  # ticks per quantization step


def snap_to_grid(tick: int, grid: int = GRID) -> int:
    """Round a tick to the nearest grid position."""
    return round(tick / grid) * grid


def ticks_to_steps(ticks: int, grid: int = GRID) -> int:
    """Convert ticks to the nearest integer number of grid steps."""
    return round(ticks / grid)


def steps_to_ticks(steps: int, grid: int = GRID) -> int:
    """Convert grid steps back to ticks."""
    return steps * grid


def quantize_sustain(sustain_ticks: int) -> int:
    """Return the SUS token step index (0..59) closest to the given sustain in ticks."""
    steps = sustain_ticks / GRID
    all_steps = Vocab.SUS_STEPS
    best_idx = min(range(len(all_steps)), key=lambda i: abs(all_steps[i] - steps))
    return best_idx


def sustain_from_sus_index(sus_index: int) -> int:
    """Reconstruct the approximate sustain ticks from a SUS step index."""
    return Vocab.SUS_STEPS[sus_index] * GRID
