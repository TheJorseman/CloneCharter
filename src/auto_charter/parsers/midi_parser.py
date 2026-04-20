"""Parser for Clone Hero .mid files.

Normalises to the same ChartData format as chart_parser.py with resolution=192.

Clone Hero MIDI track names (case-insensitive):
  PART GUITAR        → guitar
  PART BASS          → bass
  PART DRUMS         → drums
  PART RHYTHM        → rhythm

Pitch mapping (standard Clone Hero MIDI convention, Expert difficulty):
  Guitar/Bass:
    96 → pitch 0 (Green)
    97 → pitch 1 (Red)
    98 → pitch 2 (Yellow)
    99 → pitch 3 (Blue)
    100 → pitch 4 (Orange)
    101 → pitch 5 (Force HOPO)
    102 → pitch 6 (Force Strum)
    103 → Star Power phrase marker

  Drums (Expert):
    96  → pitch 0 (Kick)
    97  → pitch 1 (Red/Snare)
    98  → pitch 2 (Yellow/Hi-hat)
    99  → pitch 3 (Blue/Tom)
    100 → pitch 4 (Orange/Cymbal)

Star Power: track name "PART GUITAR" etc., note 103 on/off → SpecialEvent star_power

Tick normalisation:
    chart_tick = round(midi_tick * 192 / ticks_per_beat)

Dependencies: mido
"""

from __future__ import annotations

import logging
from pathlib import Path

from .chart_parser import ChartData, NoteEvent, SpecialEvent
from .sync_track import BPMEvent, BPMMap, TimeSigEvent

logger = logging.getLogger(__name__)

try:
    import mido
    _MIDO_AVAILABLE = True
except ImportError:
    _MIDO_AVAILABLE = False

TARGET_RESOLUTION = 192

# Guitar/Bass: note pitch offset from 60 (Expert = 60..66, Hard = 48..54, etc.)
# Clone Hero uses 60-based offsets for Expert
_GUITAR_EXPERT_BASE = 60
_DRUM_EXPERT_BASE = 60

# Standard pitch maps for Expert difficulty
_GUITAR_MIDI_TO_CHART = {
    96: 0, 97: 1, 98: 2, 99: 3, 100: 4,
    101: 5,  # Force HOPO
    102: 6,  # Force Strum
}

_DRUM_MIDI_TO_CHART = {
    96: 0,   # Kick
    97: 1,   # Red / Snare
    98: 2,   # Yellow / Hi-hat
    99: 3,   # Blue / Tom
    100: 4,  # Orange / Cymbal
    # Some charts use kick2 at 95
    95: 0,
}

_STAR_POWER_PITCH = 103

_TRACK_NAME_MAP: dict[str, str] = {
    "part guitar": "guitar",
    "part bass": "bass",
    "part drums": "drums",
    "part rhythm": "rhythm",
    "part keys": "keys",
    "t1 gems": "guitar",  # old GH format
}


def _norm_tick(midi_tick: int, ticks_per_beat: int) -> int:
    return round(midi_tick * TARGET_RESOLUTION / ticks_per_beat)


def parse_midi(path: str | Path) -> ChartData:
    """Parse a .mid file and return a ChartData normalised to resolution=192."""
    if not _MIDO_AVAILABLE:
        raise ImportError("mido is required for MIDI parsing: pip install mido")

    path = Path(path)
    try:
        mid = mido.MidiFile(str(path), clip=True)
    except Exception:
        # clip=True not available in older mido versions — fall back without it
        mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat

    data = ChartData(resolution=TARGET_RESOLUTION)
    data.song_meta["source"] = "midi"

    bpm_events: list[BPMEvent] = []
    ts_events: list[TimeSigEvent] = []

    # Accumulate note-on/off by (track_name, pitch)
    # key = (instrument, pitch), value = list of open note_on ticks
    open_notes: dict[tuple[str, int], list[int]] = {}

    # Per-instrument: tick → list of (pitch, sustain)
    track_raw: dict[str, dict[int, list[tuple[int, int]]]] = {}
    # Per-instrument: list of SpecialEvent
    track_specials: dict[str, list[SpecialEvent]] = {}
    # star power open ticks per instrument
    sp_open: dict[str, int] = {}

    for track in mid.tracks:
        track_name_raw = (track.name or "").strip().lower()
        instrument = _TRACK_NAME_MAP.get(track_name_raw)

        is_tempo_track = instrument is None and track_name_raw in ("", "tempo track", "sync track")

        abs_tick = 0
        for msg in track:
            abs_tick += msg.time

            # Tempo and time signature from any track (typically track 0)
            if msg.type == "set_tempo":
                chart_tick = _norm_tick(abs_tick, tpb)
                bpm = round(60_000_000 / msg.tempo, 6)
                bpm_events.append(BPMEvent(tick=chart_tick, bpm=bpm))

            elif msg.type == "time_signature":
                chart_tick = _norm_tick(abs_tick, tpb)
                ts_events.append(TimeSigEvent(
                    tick=chart_tick,
                    numerator=msg.numerator,
                    denominator=msg.denominator,
                ))

            if instrument is None:
                continue

            # Initialise per-instrument stores
            if instrument not in track_raw:
                track_raw[instrument] = {}
                track_specials[instrument] = []

            if msg.type in ("note_on", "note_off"):
                pitch = msg.note
                velocity = getattr(msg, "velocity", 0)
                is_on = msg.type == "note_on" and velocity > 0
                chart_tick = _norm_tick(abs_tick, tpb)

                # Clamp out-of-range MIDI pitches (malformed files)
                if not (0 <= pitch <= 127):
                    logger.debug(
                        "Clamping out-of-range MIDI pitch %d in track '%s'",
                        pitch, track_name_raw,
                    )
                    pitch = max(0, min(127, pitch))

                # Star power note
                if pitch == _STAR_POWER_PITCH:
                    if is_on:
                        sp_open[instrument] = chart_tick
                    else:
                        if instrument in sp_open:
                            length = chart_tick - sp_open.pop(instrument)
                            track_specials[instrument].append(
                                SpecialEvent(tick=sp_open.get(instrument, chart_tick - length),
                                             kind="star_power", length=length)
                            )
                    continue

                # Guitar/bass pitch mapping
                if "drum" not in instrument:
                    chart_pitch = _GUITAR_MIDI_TO_CHART.get(pitch)
                else:
                    chart_pitch = _DRUM_MIDI_TO_CHART.get(pitch)

                if chart_pitch is None:
                    # Log unexpected pitches at debug level to help diagnose non-standard charts
                    logger.debug(
                        "Ignoring unmapped pitch %d in track '%s' (instrument='%s')",
                        pitch, track_name_raw, instrument,
                    )
                    continue

                key = (instrument, pitch)
                if is_on:
                    if key not in open_notes:
                        open_notes[key] = []
                    open_notes[key].append(chart_tick)
                else:
                    if open_notes.get(key):
                        on_tick = open_notes[key].pop()
                        sustain = chart_tick - on_tick
                        if on_tick not in track_raw[instrument]:
                            track_raw[instrument][on_tick] = []
                        track_raw[instrument][on_tick].append((chart_pitch, sustain))

    # Close any notes that never got a note_off
    # (treat end of track as note_off — sustain = 0)
    for (instrument, pitch), ticks in open_notes.items():
        for on_tick in ticks:
            chart_pitch = (_GUITAR_MIDI_TO_CHART if "drum" not in instrument
                           else _DRUM_MIDI_TO_CHART).get(pitch)
            if chart_pitch is not None:
                if on_tick not in track_raw.get(instrument, {}):
                    track_raw.setdefault(instrument, {})[on_tick] = []
                track_raw[instrument][on_tick].append((chart_pitch, 0))

    # Fallback BPM if none found
    if not bpm_events:
        bpm_events.append(BPMEvent(tick=0, bpm=120.0))
    if not ts_events:
        ts_events.append(TimeSigEvent(tick=0, numerator=4, denominator=4))

    data.bpm_map = BPMMap(
        resolution=TARGET_RESOLUTION,
        bpm_events=sorted(bpm_events, key=lambda e: e.tick),
        time_sig_events=sorted(ts_events, key=lambda e: e.tick),
    )

    # Convert raw per-tick note groups into NoteEvent lists
    for instrument, tick_map in track_raw.items():
        events: list[NoteEvent] = []
        for tick in sorted(tick_map):
            entries = tick_map[tick]
            pitches_raw = {p for p, _ in entries}

            is_hopo = 5 in pitches_raw
            is_force_strum = 6 in pitches_raw
            lane_pitches = frozenset(p for p in pitches_raw if 0 <= p <= 4)
            sustain = max((s for p, s in entries if 0 <= p <= 4), default=0)

            if lane_pitches:
                events.append(NoteEvent(
                    tick=tick,
                    pitches=lane_pitches,
                    sustain=sustain,
                    is_hopo=is_hopo,
                    is_force_strum=is_force_strum,
                ))

        data.tracks[instrument] = events
        data.specials[instrument] = sorted(track_specials.get(instrument, []),
                                           key=lambda e: e.tick)

    return data
