"""Demucs-based audio source separation.

Separates a stereo mix (song.ogg) into individual stems using the
`htdemucs` model from Facebook Research. The four output stems are:

    drums   →  drums track conditioning
    bass    →  bass track conditioning
    other   →  guitar / lead instrument conditioning
    vocals  →  vocals (discarded for chart generation)

The separator only runs when a song is missing its dedicated stem file
(guitar.ogg, bass.ogg, drums.ogg). If the file already exists, the
existing file is used directly.

Demucs separates at its native sample rate (44 100 Hz) and the results
are written as 16-bit WAV files next to the original audio or in a
configurable stems cache directory.

Requirements:
    pip install demucs torchaudio

Usage:
    sep = StemSeparator(device="cuda")  # or "cpu"
    stems = sep.separate(
        audio_path=Path("song.ogg"),
        output_dir=Path("separated/"),
        instruments=["guitar", "bass", "drums"],
    )
    # stems = {"guitar": Path("separated/other.wav"),
    #          "bass":   Path("separated/bass.wav"),
    #          "drums":  Path("separated/drums.wav")}
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Demucs stem name → canonical instrument name
_DEMUCS_TO_INSTR: dict[str, str] = {
    "other":  "guitar",   # lead instrument / guitar
    "bass":   "bass",
    "drums":  "drums",
    "vocals": "vocals",
}

# Canonical instrument name → Demucs stem name
_INSTR_TO_DEMUCS: dict[str, str] = {v: k for k, v in _DEMUCS_TO_INSTR.items()}
_INSTR_TO_DEMUCS["guitar"] = "other"  # explicit override


def _load_audio_robust(audio_path: Path):
    """Load audio file as a torchaudio tensor, falling back to ffmpeg if needed.

    torchaudio's default backends (soundfile/sox) don't reliably decode .opus
    files even when the file is valid. ffmpeg handles virtually all formats.

    Returns:
        (wav, sr): wav is [C, T] float32 tensor, sr is the sample rate.
    """
    import subprocess
    import tempfile
    import torchaudio

    # First try torchaudio directly (fast path for .ogg/.mp3/.wav)
    try:
        wav, sr = torchaudio.load(str(audio_path))
        return wav, sr
    except Exception as direct_err:
        logger.debug(
            "torchaudio direct load failed for '%s' (%s) — trying ffmpeg fallback.",
            audio_path.name, direct_err,
        )

    # ffmpeg fallback: decode to a temporary WAV file
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-ar", "44100",   # standard sample rate
            "-ac", "2",       # stereo
            "-f", "wav", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr.decode(errors='replace')}"
            )

        wav, sr = torchaudio.load(tmp_path)
        return wav, sr
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class StemSeparator:
    """Separate a stereo mix into instrument stems using Demucs htdemucs.

    Args:
        model_name: Demucs model ID (default "htdemucs").
        device: Torch device string ("cpu", "cuda", "mps").
        segment: Chunk duration in seconds for memory-constrained GPUs.
            Use None for full-song processing (requires more VRAM).
        shifts: Number of random shifts for ensemble (1 = no ensemble,
            faster; 4+ = better quality but slower).
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
        segment: float | None = None,
        shifts: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.segment = segment
        self.shifts = shifts
        self._model = None

    def _load_model(self):
        """Lazy-load the Demucs model."""
        if self._model is not None:
            return self._model
        try:
            from demucs.pretrained import get_model
        except ImportError:
            raise ImportError(
                "demucs is required for stem separation: pip install demucs"
            )
        logger.info("Loading Demucs model '%s' on %s ...", self.model_name, self.device)
        self._model = get_model(self.model_name)
        self._model.eval()
        return self._model

    def separate(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        instruments: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        """Separate audio into stems and write WAV files to output_dir.

        Args:
            audio_path: Path to the source audio file (OGG, WAV, MP3, …).
            output_dir: Directory where separated WAV files will be written.
            instruments: Which instruments to extract. Default: all three
                (guitar, bass, drums). Vocals are never written.
            force: Re-run separation even if output files already exist.

        Returns:
            Dict mapping instrument name → Path of the output WAV file.
            Only includes instruments that were successfully separated.
        """
        try:
            import torch
            import torchaudio
            import soundfile as sf
        except ImportError as e:
            raise ImportError(f"Missing audio dependency: {e}") from e

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if instruments is None:
            instruments = ["guitar", "bass", "drums"]

        # Check which stems are already done
        result: dict[str, Path] = {}
        needed: list[str] = []
        for instr in instruments:
            stem_name = _INSTR_TO_DEMUCS.get(instr, instr)
            out_path = output_dir / f"{stem_name}.wav"
            if out_path.exists() and not force:
                result[instr] = out_path
            else:
                needed.append(instr)

        if not needed:
            logger.debug("All stems already exist in %s, skipping Demucs.", output_dir)
            return result

        # Run Demucs
        model = self._load_model()

        # Load audio with torchaudio.
        # Some formats (e.g. .opus) are not reliably decoded by torchaudio's
        # default backend. If loading fails, fall back to ffmpeg → WAV conversion.
        wav, sr = _load_audio_robust(audio_path)  # [C, T]
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        wav = wav.unsqueeze(0).to(self.device)  # [1, 2, T]

        logger.info(
            "Separating '%s' with Demucs (%s, %s) ...",
            audio_path.name, self.model_name, self.device
        )

        try:
            from demucs.apply import apply_model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav,
                    device=self.device,
                    shifts=self.shifts,
                    split=True,
                    segment=self.segment,
                    overlap=0.25,
                    progress=False,
                )
        except Exception as e:
            logger.error("Demucs separation failed: %s", e)
            raise

        # sources: [1, n_stems, 2, T]
        sources = sources.squeeze(0).cpu()  # [n_stems, 2, T]
        stem_names: list[str] = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']

        # Write requested stems
        for idx, demucs_name in enumerate(stem_names):
            instr = _DEMUCS_TO_INSTR.get(demucs_name)
            if instr not in needed:
                continue

            out_path = output_dir / f"{demucs_name}.wav"
            stem_wav = sources[idx]  # [2, T]
            # Write as 32-bit float WAV via soundfile
            sf.write(
                str(out_path),
                stem_wav.numpy().T,  # [T, 2]
                samplerate=model.samplerate,
                subtype="FLOAT",
            )
            result[instr] = out_path
            logger.info("  Wrote %s stem → %s", instr, out_path.name)

        return result


def resolve_or_separate(
    song_dir: Path,
    instruments: list[str],
    stems_cache: Path | None,
    separator: StemSeparator | None,
) -> dict[str, Path]:
    """Return audio paths for requested instruments.

    Uses existing stem files when available; falls back to Demucs separation
    of song.ogg if a separator is provided.

    Returns:
        Dict mapping instrument name → Path to audio file.
        An instrument may be missing from the dict if no audio is available.
    """
    from auto_charter.audio.stem_loader import resolve_stem_path, has_dedicated_stem

    result: dict[str, Path] = {}
    missing: list[str] = []

    for instr in instruments:
        if has_dedicated_stem(song_dir, instr):
            p = resolve_stem_path(song_dir, instr)
            if p:
                result[instr] = p
        else:
            missing.append(instr)

    # If nothing is missing or we have no separator, fill remaining from mix
    if not missing or separator is None:
        for instr in missing:
            p = resolve_stem_path(song_dir, instr)  # returns song.ogg fallback
            if p:
                result[instr] = p
        return result

    # Source audio for Demucs (prefer song.ogg/mp3/wav, then any audio file)
    from auto_charter.audio.stem_loader import _find_audio, find_all_audio
    mix_path = _find_audio(song_dir, "song") or _find_audio(song_dir, "music")
    if mix_path is None:
        all_audio = find_all_audio(song_dir)
        mix_path = all_audio[0] if all_audio else None

    if mix_path is None:
        logger.warning("No audio file found in %s — skipping Demucs.", song_dir)
        return result

    # Output directory for separated stems
    out_dir = (stems_cache / song_dir.name) if stems_cache else (song_dir / ".stems")

    try:
        separated = separator.separate(
            audio_path=mix_path,
            output_dir=out_dir,
            instruments=missing,
        )
        result.update(separated)
    except Exception as e:
        logger.warning("Stem separation failed for %s: %s — using mix.", song_dir.name, e)
        for instr in missing:
            p = resolve_stem_path(song_dir, instr)
            if p:
                result[instr] = p

    return result
