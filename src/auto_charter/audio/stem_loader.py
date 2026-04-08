"""Resolve audio stem paths for each instrument.

Clone Hero audio conventions:
  Single-mix songs: song.ogg
  Separated stems:  guitar.ogg, bass.ogg, drums.ogg, vocals.ogg, keys.ogg, rhythm.ogg

Strategy: use the instrument-specific stem if it exists; fall back to song.ogg
or any available mix file.
"""

from __future__ import annotations

from pathlib import Path

# Candidate stem file names per instrument, in preference order
_STEM_CANDIDATES: dict[str, list[str]] = {
    "guitar":  ["guitar.ogg", "song.ogg"],
    "bass":    ["bass.ogg", "song.ogg"],
    "drums":   ["drums.ogg", "song.ogg"],
    "vocals":  ["vocals.ogg", "song.ogg"],
    "keys":    ["keys.ogg", "song.ogg"],
    "rhythm":  ["rhythm.ogg", "guitar.ogg", "song.ogg"],
}

_MIX_NAMES = ["song.ogg", "music.ogg", "preview.ogg"]
_ALL_STEM_NAMES = {
    "guitar.ogg", "bass.ogg", "drums.ogg", "vocals.ogg",
    "keys.ogg", "rhythm.ogg",
}


def resolve_stem_path(song_dir: str | Path, instrument: str) -> Path | None:
    """Return the Path to the best audio file for an instrument, or None.

    Prefers the instrument-specific stem; falls back to song.ogg (full mix).
    """
    song_dir = Path(song_dir)
    candidates = _STEM_CANDIDATES.get(instrument.split("_")[0], ["song.ogg"])
    for name in candidates:
        p = song_dir / name
        if p.exists():
            return p
    return None


def has_dedicated_stem(song_dir: str | Path, instrument: str) -> bool:
    """Return True if an instrument-specific (non-mix) stem file exists."""
    song_dir = Path(song_dir)
    stem_name = f"{instrument.split('_')[0]}.ogg"
    return (song_dir / stem_name).exists()


def find_all_audio(song_dir: str | Path) -> list[Path]:
    """Return all .ogg audio files in a song directory."""
    return sorted(Path(song_dir).glob("*.ogg"))
