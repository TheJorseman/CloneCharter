"""Resolve audio stem paths for each instrument.

Clone Hero audio conventions:
  Single-mix songs: song.ogg / song.mp3 / song.opus
  Separated stems:  guitar.ogg, bass.ogg, drums.ogg, vocals.ogg, keys.ogg, rhythm.ogg
                    (also .mp3, .opus, .wav, .flac variants)

Strategy: use the instrument-specific stem if it exists; fall back to song.ogg
or any available mix file.
"""

from __future__ import annotations

from pathlib import Path

# Supported audio extensions, in preference order.
# .ogg and .opus are most common in Clone Hero packs.
_AUDIO_EXTS = [".ogg", ".opus", ".mp3", ".wav", ".flac"]

# Candidate stem base names per instrument, in preference order
_STEM_BASES: dict[str, list[str]] = {
    "guitar":  ["guitar", "song"],
    "bass":    ["bass", "song"],
    "drums":   ["drums", "song"],
    "vocals":  ["vocals", "song"],
    "keys":    ["keys", "song"],
    "rhythm":  ["rhythm", "guitar", "song"],
}

_MIX_BASES = ["song", "music", "preview"]
_DEDICATED_STEM_BASES = {"guitar", "bass", "drums", "vocals", "keys", "rhythm"}


def _find_audio(song_dir: Path, base: str) -> Path | None:
    """Return first existing file matching base + any supported extension."""
    for ext in _AUDIO_EXTS:
        p = song_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def resolve_stem_path(song_dir: str | Path, instrument: str) -> Path | None:
    """Return the Path to the best audio file for an instrument, or None.

    Prefers the instrument-specific stem; falls back to song.ogg/opus/mp3 (full mix).
    Supports .ogg, .opus, .mp3, .wav, and .flac stem files.
    """
    song_dir = Path(song_dir)
    bases = _STEM_BASES.get(instrument.split("_")[0], ["song"])
    for base in bases:
        p = _find_audio(song_dir, base)
        if p is not None:
            return p
    return None


def has_dedicated_stem(song_dir: str | Path, instrument: str) -> bool:
    """Return True if an instrument-specific (non-mix) stem file exists.

    Checks .ogg, .opus, .mp3, .wav, and .flac variants.
    """
    song_dir = Path(song_dir)
    base = instrument.split("_")[0]
    if base not in _DEDICATED_STEM_BASES:
        return False
    return _find_audio(song_dir, base) is not None


def find_all_audio(song_dir: str | Path) -> list[Path]:
    """Return all audio files (.ogg, .opus, .mp3, .wav, .flac) in a song directory."""
    song_dir = Path(song_dir)
    files: list[Path] = []
    for ext in _AUDIO_EXTS:
        files.extend(song_dir.glob(f"*{ext}"))
    return sorted(set(files))
