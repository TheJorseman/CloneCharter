"""chart_renderer — convert ChartData back to .chart text format.

This is the inverse of chart_parser.parse_chart(). Used by the Gradio demo
to produce a playable notes.chart file from model-generated ChartData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from auto_charter.parsers.chart_parser import ChartData, NoteEvent, SpecialEvent


# Canonical instrument name → ExpertXxx section name
_INSTR_TO_SECTION: dict[str, str] = {
    "guitar": "ExpertSingle",
    "bass": "ExpertDoubleBass",
    "drums": "ExpertDrums",
    "rhythm": "ExpertDoubleRhythm",
}


def render_chart(
    chart_data: ChartData,
    bpm: float,
    resolution: int = 192,
    song_name: str = "Unknown",
    artist: str = "Unknown",
    album: str = "",
    year: int = 0,
    charter: str = "auto-charter",
) -> str:
    """Convert a ChartData object to .chart file text.

    Args:
        chart_data: Decoded chart data (from decode_tokens()).
        bpm:        Song BPM (used for SyncTrack if chart_data.bpm_map is empty).
        resolution: Ticks per quarter note (always 192 after normalization).
        song_name:  For the [Song] header.
        artist:     For the [Song] header.
        album:      For the [Song] header.
        year:       For the [Song] header.
        charter:    For the [Song] header.

    Returns:
        Complete .chart file text.
    """
    parts: list[str] = []

    # ── [Song] section ─────────────────────────────────────────────────────────
    parts.append("[Song]")
    parts.append("{")
    parts.append(f'  Name = "{song_name}"')
    parts.append(f'  Artist = "{artist}"')
    if album:
        parts.append(f'  Album = "{album}"')
    if year:
        parts.append(f'  Year = ", {year}"')
    parts.append(f'  Charter = "{charter}"')
    parts.append(f"  Resolution = {resolution}")
    parts.append("  MusicStream = song.ogg")
    parts.append("}")
    parts.append("")

    # ── [SyncTrack] ────────────────────────────────────────────────────────────
    parts.append("[SyncTrack]")
    parts.append("{")

    # Use BPM events from bpm_map if available, otherwise single constant BPM
    bpm_events_written = False
    if chart_data.bpm_map and chart_data.bpm_map.bpm_events:
        for ev in chart_data.bpm_map.bpm_events:
            bpm_millis = int(round(ev.bpm * 1000))
            parts.append(f"  {ev.tick} = B {bpm_millis}")
        bpm_events_written = True

    if not bpm_events_written:
        # Constant BPM from librosa estimate
        bpm_millis = int(round(bpm * 1000))
        parts.append(f"  0 = B {bpm_millis}")

    # Add 4/4 time signature at tick 0 if not present
    ts_written = False
    if chart_data.bpm_map and chart_data.bpm_map.time_sig_events:
        for ts in chart_data.bpm_map.time_sig_events:
            parts.append(f"  {ts.tick} = TS {ts.numerator}")
        ts_written = True
    if not ts_written:
        parts.append("  0 = TS 4")

    parts.append("}")
    parts.append("")

    # ── Instrument tracks ──────────────────────────────────────────────────────
    for instr, notes in chart_data.tracks.items():
        if not notes:
            continue

        section = _INSTR_TO_SECTION.get(instr)
        if section is None:
            continue

        is_drums = instr == "drums"

        parts.append(f"[{section}]")
        parts.append("{")

        # Collect note lines: (tick, line_str)
        events: list[tuple[int, str]] = []

        for note in notes:
            tick = note.tick
            # Write one line per pitch in the chord
            for pitch in sorted(note.pitches):
                sustain = 0 if is_drums else note.sustain
                events.append((tick, f"  {tick} = N {pitch} {sustain}"))

            # Modifiers (guitar/bass only)
            if not is_drums:
                if note.is_hopo:
                    events.append((tick, f"  {tick} = N 5 0"))
                if note.is_tap:
                    events.append((tick, f"  {tick} = N 7 0"))
                if note.is_force_strum:
                    events.append((tick, f"  {tick} = N 6 0"))

        # Star power specials
        specials_for_instr = chart_data.specials.get(instr, [])
        for sp in specials_for_instr:
            if sp.kind == "star_power":
                events.append((sp.tick, f"  {sp.tick} = S 2 {sp.length}"))
            elif sp.kind == "solo":
                events.append((sp.tick, f"  {sp.tick} = E solo"))
            elif sp.kind == "soloend":
                events.append((sp.tick, f"  {sp.tick} = E soloend"))

        # Sort by tick (stable), then write
        events.sort(key=lambda x: x[0])
        for _, line in events:
            parts.append(line)

        parts.append("}")
        parts.append("")

    return "\n".join(parts)


def render_ini(
    song_name: str,
    artist: str,
    album: str = "",
    genre: str = "",
    year: int = 0,
    instrument: str = "guitar",
    difficulty: int = 2,
    charter: str = "auto-charter",
    song_length_ms: int = 0,
) -> str:
    """Generate a song.ini file for a Clone Hero song package.

    Args:
        song_name:      Song title.
        artist:         Artist name.
        album:          Album name (optional).
        genre:          Genre (optional).
        year:           Release year (optional).
        instrument:     "guitar", "bass", or "drums".
        difficulty:     Difficulty level 0–6 for this instrument.
        charter:        Charter credit string.
        song_length_ms: Song duration in milliseconds (0 if unknown).

    Returns:
        song.ini text string.
    """
    diff_fields = {
        "guitar": -1,
        "bass": -1,
        "drums": -1,
    }
    if instrument in diff_fields:
        diff_fields[instrument] = difficulty

    lines = [
        "[Song]",
        f"name = {song_name}",
        f"artist = {artist}",
    ]
    if album:
        lines.append(f"album = {album}")
    if genre:
        lines.append(f"genre = {genre}")
    if year:
        lines.append(f"year = {year}")
    if song_length_ms:
        lines.append(f"song_length = {song_length_ms}")
    lines += [
        f"charter = {charter}",
        f"diff_guitar = {diff_fields['guitar']}",
        f"diff_bass = {diff_fields['bass']}",
        f"diff_drums = {diff_fields['drums']}",
        "diff_keys = -1",
        "pro_drums = 0",
        "five_lane_drums = 0",
    ]
    return "\n".join(lines) + "\n"
