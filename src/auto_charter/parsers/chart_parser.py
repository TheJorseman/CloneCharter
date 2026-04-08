"""Parser for Clone Hero .chart files.

Produces a ChartData object containing:
- resolution (int): ticks per quarter note
- bpm_map (BPMMap): timing information
- tracks (dict[instrument, list[NoteEvent]]): note data per instrument
- specials (dict[instrument, list[SpecialEvent]]): star power + solos
- song_meta (dict): raw [Song] block fields

Supported track names mapped to canonical instrument names:
  ExpertSingle       → "guitar"
  HardSingle         → "guitar_hard"
  MediumSingle       → "guitar_medium"
  EasySingle         → "guitar_easy"
  ExpertDoubleBass   → "bass"
  HardDoubleBass     → "bass_hard"
  MediumDoubleBass   → "bass_medium"
  EasyDoubleBass     → "bass_easy"
  ExpertDrums        → "drums"
  HardDrums          → "drums_hard"
  MediumDrums        → "drums_medium"
  EasyDrums          → "drums_easy"
  ExpertDoubleRhythm → "rhythm"
  Events             → special events (section markers)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from .sync_track import BPMEvent, BPMMap, TimeSigEvent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NoteEvent(NamedTuple):
    """A single note event within a chart track."""
    tick: int
    pitches: frozenset[int]  # set of simultaneous pitch values (can be chord)
    sustain: int             # sustain length in ticks (0 = staccato)
    is_hopo: bool = False    # True if pitch 5 present
    is_tap: bool = False     # True if pitch 7 present (newer format)
    is_force_strum: bool = False  # True if pitch 6 present


class SpecialEvent(NamedTuple):
    """A special phrase event (star power, solo markers)."""
    tick: int
    kind: str    # "star_power", "solo", "soloend"
    length: int  # phrase length in ticks (0 for instant events)


# ---------------------------------------------------------------------------
# Section name → canonical instrument mapping
# ---------------------------------------------------------------------------

_TRACK_MAP: dict[str, str] = {
    "expertsingle": "guitar",
    "hardsingle": "guitar_hard",
    "mediumsingle": "guitar_medium",
    "easysingle": "guitar_easy",
    "expertdoublebass": "bass",
    "harddoublebass": "bass_hard",
    "mediumdoublebass": "bass_medium",
    "easydoublebass": "bass_easy",
    "expertdrums": "drums",
    "harddrums": "drums_hard",
    "mediumdrums": "drums_medium",
    "easydrums": "drums_easy",
    "expertdoublerhythm": "rhythm",
    "harddoublerhythm": "rhythm_hard",
    "mediumdoublerhythm": "rhythm_medium",
    "easydoublerhythm": "rhythm_easy",
    "expertghl": "ghl",
    "expertbassghl": "bass_ghl",
}


# ---------------------------------------------------------------------------
# ChartData
# ---------------------------------------------------------------------------

@dataclass
class ChartData:
    resolution: int = 192
    song_meta: dict[str, str] = field(default_factory=dict)
    bpm_map: BPMMap = field(default_factory=lambda: BPMMap(resolution=192))
    tracks: dict[str, list[NoteEvent]] = field(default_factory=dict)
    specials: dict[str, list[SpecialEvent]] = field(default_factory=dict)
    section_events: list[tuple[int, str]] = field(default_factory=list)

    @property
    def end_tick(self) -> int:
        """Estimated song end tick based on the last note across all tracks."""
        last = 0
        for notes in self.tracks.values():
            if notes:
                n = notes[-1]
                last = max(last, n.tick + n.sustain)
        return last

    def instruments(self) -> list[str]:
        """Return list of charted instrument names (tracks with at least 1 note)."""
        return [k for k, v in self.tracks.items() if v]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_NOTE_RE = re.compile(r"^\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
_SPECIAL_RE = re.compile(r"^\s*(\d+)\s*=\s*S\s+(\d+)\s+(\d+)")
_EVENT_RE = re.compile(r'^\s*(\d+)\s*=\s*E\s+"?([^"]+)"?')
_BPM_RE = re.compile(r"^\s*(\d+)\s*=\s*B\s+(\d+)")
_TS_RE = re.compile(r"^\s*(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?")
_SONG_KV_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+)")
_SECTION_RE = re.compile(r"^\s*\[(\w+)\]")


def parse_chart(path: str | Path) -> ChartData:
    """Parse a .chart file and return a ChartData instance."""
    path = Path(path)
    data = ChartData()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_section: str | None = None
    bpm_events: list[BPMEvent] = []
    ts_events: list[TimeSigEvent] = []

    # Per-track accumulators: tick → list of (pitch, sustain)
    track_raw: dict[str, dict[int, list[tuple[int, int]]]] = {}
    track_specials: dict[str, list[SpecialEvent]] = {}

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        m = _SECTION_RE.match(line)
        if m:
            current_section = m.group(1).lower()
            i += 1
            continue

        if current_section == "song":
            m = _SONG_KV_RE.match(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip().strip('"')
                data.song_meta[key.lower()] = val
                if key.lower() == "resolution":
                    try:
                        data.resolution = int(val)
                    except ValueError:
                        pass
            i += 1
            continue

        if current_section == "synctrack":
            m = _BPM_RE.match(line)
            if m:
                tick = int(m.group(1))
                bpm = int(m.group(2)) / 1000.0
                bpm_events.append(BPMEvent(tick=tick, bpm=bpm))
                i += 1
                continue

            m = _TS_RE.match(line)
            if m:
                tick = int(m.group(1))
                num = int(m.group(2))
                den_exp = int(m.group(3)) if m.group(3) else 2  # 2^2 = 4
                den = 2 ** den_exp
                ts_events.append(TimeSigEvent(tick=tick, numerator=num, denominator=den))
                i += 1
                continue
            i += 1
            continue

        if current_section == "events":
            m = _EVENT_RE.match(line)
            if m:
                tick = int(m.group(1))
                text = m.group(2).strip()
                data.section_events.append((tick, text))
                # Solo markers go into all instrument specials
                if text in ("solo", "soloend"):
                    for instr in track_specials:
                        track_specials[instr].append(SpecialEvent(tick=tick, kind=text, length=0))
            i += 1
            continue

        # Check if this is a known note track
        if current_section in _TRACK_MAP:
            instrument = _TRACK_MAP[current_section]
            if instrument not in track_raw:
                track_raw[instrument] = {}
                track_specials[instrument] = []

            m = _NOTE_RE.match(line)
            if m:
                tick = int(m.group(1))
                pitch = int(m.group(2))
                sustain = int(m.group(3))
                if tick not in track_raw[instrument]:
                    track_raw[instrument][tick] = []
                track_raw[instrument][tick].append((pitch, sustain))
                i += 1
                continue

            m = _SPECIAL_RE.match(line)
            if m:
                tick = int(m.group(1))
                stype = int(m.group(2))
                length = int(m.group(3))
                if stype == 2:  # star power
                    track_specials[instrument].append(
                        SpecialEvent(tick=tick, kind="star_power", length=length)
                    )
                i += 1
                continue

            m = _EVENT_RE.match(line)
            if m:
                tick = int(m.group(1))
                text = m.group(2).strip()
                if text in ("solo", "soloend"):
                    track_specials[instrument].append(
                        SpecialEvent(tick=tick, kind=text, length=0)
                    )
            i += 1
            continue

        i += 1

    # Build BPMMap
    data.bpm_map = BPMMap(
        resolution=data.resolution,
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
            is_tap = 7 in pitches_raw  # newer format uses pitch 7 for tap in some charts
            is_force_strum = 6 in pitches_raw

            # Lane pitches only (0-4); sustain = max sustain among lane pitches
            lane_pitches = frozenset(p for p in pitches_raw if 0 <= p <= 4)
            sustain = max((s for p, s in entries if 0 <= p <= 4), default=0)

            if lane_pitches:
                events.append(NoteEvent(
                    tick=tick,
                    pitches=lane_pitches,
                    sustain=sustain,
                    is_hopo=is_hopo,
                    is_tap=is_tap,
                    is_force_strum=is_force_strum,
                ))

        data.tracks[instrument] = events
        data.specials[instrument] = sorted(track_specials.get(instrument, []),
                                           key=lambda e: e.tick)

    return data
