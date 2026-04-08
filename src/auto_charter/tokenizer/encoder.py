"""Encode a chart track into a token sequence.

Token sequence grammar for guitar/bass:
    BOS INSTR [BEAT_BOUNDARY | SP_ON | SP_OFF | SOLO_ON | SOLO_OFF |
               WAIT_k+ | (NOTE_bitmask SUS_n MOD_*)] * EOS

For drums (no sustain):
    BOS INSTR [BEAT_BOUNDARY | SP_ON | SP_OFF | SOLO_ON | SOLO_OFF |
               WAIT_k+ | DRUM_bitmask] * EOS

Ordering at equal tick:
    BEAT_BOUNDARY → special events (SP/SOLO) → NOTE events

This ensures the audio conditioning anchor (BEAT_BOUNDARY) always precedes
the notes it conditions, which is critical for causal cross-attention.
"""

from __future__ import annotations

from auto_charter.parsers.chart_parser import ChartData, NoteEvent, SpecialEvent
from auto_charter.parsers.sync_track import BPMMap
from auto_charter.vocab.guitar_vocab import chord_bitmask_to_id, pitches_to_bitmask
from auto_charter.vocab.drum_vocab import drum_bitmask_to_id, drum_pitches_to_bitmask
from auto_charter.vocab.tokens import Vocab
from .quantize import GRID, ticks_to_steps, quantize_sustain


def encode_track(
    chart: ChartData,
    instrument: str,
    bpm_map: BPMMap | None = None,
    include_beat_boundaries: bool = True,
) -> list[int]:
    """Encode a single instrument track to a token ID list.

    Args:
        chart: Parsed ChartData (from parse_chart or parse_midi).
        instrument: One of "guitar", "bass", "drums", "rhythm", etc.
        bpm_map: BPMMap for beat grid generation. Uses chart.bpm_map if None.
        include_beat_boundaries: Whether to insert BEAT_BOUNDARY tokens.

    Returns:
        List of integer token IDs.
    """
    if bpm_map is None:
        bpm_map = chart.bpm_map

    notes = chart.tracks.get(instrument, [])
    specials = chart.specials.get(instrument, [])
    is_drums = "drum" in instrument

    instr_id = Vocab.INSTRUMENT_TO_ID.get(
        instrument.split("_")[0],  # strip _hard, _medium suffixes
        Vocab.INSTR_GUITAR,
    )

    # Build beat grid if needed
    beat_ticks: list[int] = []
    if include_beat_boundaries and bpm_map.bpm_events:
        beat_ticks = bpm_map.build_beat_grid(chart.end_tick)

    # Build unified timeline: (tick, priority, event_type, payload)
    # Priority ordering (lower = emitted first at same tick):
    #   0 = BEAT_BOUNDARY
    #   1 = SP_ON / SP_OFF  (star power events)
    #   2 = SOLO_ON / SOLO_OFF
    #   3 = NOTE
    # This ordering is deterministic and survives encode→decode→re-encode.
    _PRIO = {"SP_ON": 1, "SP_OFF": 1, "SOLO_ON": 2, "SOLO_OFF": 2}
    timeline: list[tuple[int, int, str, object]] = []

    for tick in beat_ticks:
        timeline.append((tick, 0, "BEAT", None))

    for sp in specials:
        if sp.kind == "star_power":
            timeline.append((sp.tick, _PRIO["SP_ON"], "SP_ON", None))
            timeline.append((sp.tick + sp.length, _PRIO["SP_OFF"], "SP_OFF", None))
        elif sp.kind == "solo":
            timeline.append((sp.tick, _PRIO["SOLO_ON"], "SOLO_ON", None))
        elif sp.kind == "soloend":
            timeline.append((sp.tick, _PRIO["SOLO_OFF"], "SOLO_OFF", None))

    # Group simultaneous notes into single chord events
    for note in notes:
        timeline.append((note.tick, 3, "NOTE", note))

    # Sort by (tick, priority)
    timeline.sort(key=lambda x: (x[0], x[1]))

    tokens: list[int] = [Vocab.BOS, instr_id]
    cursor_tick = 0

    for tick, _priority, event_type, payload in timeline:
        # Advance time cursor
        if tick > cursor_tick:
            gap_steps = ticks_to_steps(tick - cursor_tick)
            gap_steps = max(1, gap_steps)
            while gap_steps > 0:
                chunk = min(gap_steps, Vocab.WAIT_MAX_K)
                tokens.append(Vocab.wait_id(chunk))
                gap_steps -= chunk
            cursor_tick = tick

        if event_type == "BEAT":
            tokens.append(Vocab.BEAT_BOUNDARY)

        elif event_type == "SP_ON":
            tokens.append(Vocab.STAR_POWER_ON)

        elif event_type == "SP_OFF":
            tokens.append(Vocab.STAR_POWER_OFF)

        elif event_type == "SOLO_ON":
            tokens.append(Vocab.SOLO_ON)

        elif event_type == "SOLO_OFF":
            tokens.append(Vocab.SOLO_OFF)

        elif event_type == "NOTE":
            note: NoteEvent = payload  # type: ignore[assignment]

            if is_drums:
                bitmask = drum_pitches_to_bitmask(note.pitches)
                if bitmask > 0:
                    tokens.append(drum_bitmask_to_id(bitmask))
            else:
                bitmask = pitches_to_bitmask(note.pitches)
                if bitmask > 0:
                    tokens.append(chord_bitmask_to_id(bitmask))
                    sus_idx = quantize_sustain(note.sustain)
                    tokens.append(Vocab.sus_id(sus_idx))
                    # Modifiers (optional, emitted after SUS)
                    if note.is_hopo:
                        tokens.append(Vocab.MOD_HOPO)
                    if note.is_tap:
                        tokens.append(Vocab.MOD_TAP)
                    if note.is_force_strum:
                        tokens.append(Vocab.MOD_FORCE_STRUM)

    tokens.append(Vocab.EOS)
    return tokens
