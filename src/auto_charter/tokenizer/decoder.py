"""Decode a token sequence back into a ChartData object.

Inverse of encoder.encode_track(). Reconstructs NoteEvents and SpecialEvents
from the token stream. Beat boundary positions are also preserved for inspection.

Grammar expected:
    BOS INSTR_* [token]* EOS

Where each [token] is one of:
    WAIT_k              → advance cursor by k×16 ticks
    BEAT_BOUNDARY       → record beat tick (no cursor change)
    STAR_POWER_ON/OFF   → open/close star power phrase
    SOLO_ON/SOLO_OFF    → record solo event
    NOTE_bitmask SUS_n [MOD_*]    → guitar/bass note
    DRUM_bitmask        → drum note (no SUS)
"""

from __future__ import annotations

from auto_charter.parsers.chart_parser import ChartData, NoteEvent, SpecialEvent
from auto_charter.parsers.sync_track import BPMMap
from auto_charter.vocab.guitar_vocab import id_to_chord_bitmask, bitmask_to_pitches
from auto_charter.vocab.drum_vocab import id_to_drum_bitmask, drum_bitmask_to_pitches
from auto_charter.vocab.tokens import Vocab
from .quantize import GRID, steps_to_ticks, sustain_from_sus_index


def decode_tokens(
    tokens: list[int],
    resolution: int = 192,
    bpm_map: BPMMap | None = None,
) -> ChartData:
    """Decode a token sequence into a ChartData object.

    Args:
        tokens: List of token IDs produced by encode_track().
        resolution: Ticks per quarter note (default 192).
        bpm_map: Optional BPMMap to attach to the result.

    Returns:
        ChartData with one instrument track.
    """
    data = ChartData(resolution=resolution)
    if bpm_map is not None:
        data.bpm_map = bpm_map

    notes: list[NoteEvent] = []
    specials: list[SpecialEvent] = []
    beat_ticks: list[int] = []

    cursor_tick = 0
    instrument: str | None = None
    is_drums = False
    sp_open_tick: int | None = None

    i = 0
    n = len(tokens)

    # Skip BOS
    if i < n and tokens[i] == Vocab.BOS:
        i += 1

    # Read instrument token
    if i < n and tokens[i] in Vocab.ID_TO_INSTRUMENT:
        instrument = Vocab.ID_TO_INSTRUMENT[tokens[i]]
        is_drums = "drum" in instrument
        i += 1
    else:
        instrument = "guitar"

    while i < n:
        tok = tokens[i]

        if tok == Vocab.EOS or tok == Vocab.PAD:
            break

        elif Vocab.is_wait(tok):
            k = Vocab.wait_k(tok)
            cursor_tick += k * GRID
            i += 1

        elif tok == Vocab.BEAT_BOUNDARY:
            beat_ticks.append(cursor_tick)
            i += 1

        elif tok == Vocab.MEASURE_START:
            i += 1

        elif tok == Vocab.STAR_POWER_ON:
            # Record placeholder; length filled in when SP_OFF is seen
            sp_open_tick = cursor_tick
            specials.append(SpecialEvent(tick=cursor_tick, kind="star_power", length=0))
            i += 1

        elif tok == Vocab.STAR_POWER_OFF:
            if sp_open_tick is not None:
                length = cursor_tick - sp_open_tick
                # Update the placeholder in-place (replace the last star_power entry)
                for j in range(len(specials) - 1, -1, -1):
                    if specials[j].kind == "star_power" and specials[j].tick == sp_open_tick:
                        specials[j] = SpecialEvent(tick=sp_open_tick, kind="star_power",
                                                   length=length)
                        break
                sp_open_tick = None
            i += 1

        elif tok == Vocab.SOLO_ON:
            specials.append(SpecialEvent(tick=cursor_tick, kind="solo", length=0))
            i += 1

        elif tok == Vocab.SOLO_OFF:
            specials.append(SpecialEvent(tick=cursor_tick, kind="soloend", length=0))
            i += 1

        elif Vocab.is_guitar_note(tok) and not is_drums:
            bitmask = id_to_chord_bitmask(tok)
            pitches = bitmask_to_pitches(bitmask)
            i += 1

            # Read SUS token (mandatory after guitar/bass note)
            sustain = 0
            if i < n and Vocab.is_sus(tokens[i]):
                sus_idx = tokens[i] - Vocab.SUS_START
                sustain = sustain_from_sus_index(sus_idx)
                i += 1

            # Read optional modifiers
            is_hopo = is_tap = is_force_strum = False
            while i < n and Vocab.is_modifier(tokens[i]):
                if tokens[i] == Vocab.MOD_HOPO:
                    is_hopo = True
                elif tokens[i] == Vocab.MOD_TAP:
                    is_tap = True
                elif tokens[i] == Vocab.MOD_FORCE_STRUM:
                    is_force_strum = True
                elif tokens[i] == Vocab.MOD_OPEN:
                    pitches = pitches | {7}
                i += 1

            notes.append(NoteEvent(
                tick=cursor_tick,
                pitches=pitches,
                sustain=sustain,
                is_hopo=is_hopo,
                is_tap=is_tap,
                is_force_strum=is_force_strum,
            ))

        elif Vocab.is_drum_note(tok):
            bitmask = id_to_drum_bitmask(tok)
            pitches = drum_bitmask_to_pitches(bitmask)
            notes.append(NoteEvent(tick=cursor_tick, pitches=pitches, sustain=0))
            i += 1

        else:
            # Unknown/unexpected token — skip gracefully
            i += 1

    # Close any unclosed star power at end of sequence
    if sp_open_tick is not None:
        specials.append(SpecialEvent(tick=sp_open_tick, kind="star_power",
                                     length=cursor_tick - sp_open_tick))

    if instrument:
        data.tracks[instrument] = sorted(notes, key=lambda n: n.tick)
        data.specials[instrument] = sorted(specials, key=lambda s: s.tick)
        data.section_events = [(t, "beat") for t in beat_ticks]

    return data
