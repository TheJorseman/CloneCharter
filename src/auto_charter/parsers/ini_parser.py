"""Parser for Clone Hero song.ini metadata files.

Handles both [Song] and [song] section headers (case-insensitive).
All keys are normalised to lowercase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SongMetadata:
    name: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    charter: str = ""
    year: int = 0
    song_length_ms: int = 0
    preview_start_time_ms: int = 0
    delay_ms: int = 0

    # Per-instrument difficulties (-1 = not charted, 0-6 = difficulty)
    diff_guitar: int = -1
    diff_bass: int = -1
    diff_drums: int = -1
    diff_keys: int = -1
    diff_rhythm: int = -1

    # Extended flags
    pro_drums: bool = False
    five_lane_drums: bool = False

    # Raw key→value store for any unknown fields
    extra: dict[str, str] = field(default_factory=dict)


def parse_ini(path: str | Path) -> SongMetadata:
    """Parse a song.ini file and return a SongMetadata instance."""
    path = Path(path)
    raw: dict[str, str] = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                raw[key.strip().lower()] = value.strip()

    meta = SongMetadata()
    meta.name = raw.get("name", "")
    meta.artist = raw.get("artist", "")
    meta.album = raw.get("album", "")
    meta.genre = raw.get("genre", "")
    meta.charter = raw.get("charter", raw.get("frets", ""))
    meta.extra = raw

    try:
        meta.year = int(raw.get("year", "0"))
    except ValueError:
        meta.year = 0

    for key in ("song_length", "song_length_ms"):
        if key in raw:
            try:
                meta.song_length_ms = int(raw[key])
            except ValueError:
                pass
            break

    try:
        meta.preview_start_time_ms = int(raw.get("preview_start_time", "0"))
    except ValueError:
        pass

    try:
        meta.delay_ms = int(raw.get("delay", "0"))
    except ValueError:
        pass

    for attr in ("diff_guitar", "diff_bass", "diff_drums", "diff_keys", "diff_rhythm"):
        if attr in raw:
            try:
                setattr(meta, attr, int(raw[attr]))
            except ValueError:
                pass

    meta.pro_drums = raw.get("pro_drums", "0") not in ("0", "", "false", "False")
    meta.five_lane_drums = raw.get("five_lane_drums", "0") not in ("0", "", "false", "False")

    return meta
