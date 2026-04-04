"""
End-to-end inference pipeline: audio file → notes.chart

Usage:
    python inference/pipeline.py \\
        --audio el_rescate.mp3 \\
        --instrument Single \\
        --difficulty Expert \\
        --checkpoint model_checkpoints/step_0005000 \\
        --output nueva_cancion.chart
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from models.chart_transformer import ChartTransformer
from models.tokenizer import CloneHeroTokenizer
from models.demucs import DemucsAudioSeparator
from models.mert import MERT
from data.audio_loaders import AudioProcessor
from models.conditioning import INSTRUMENT_MAP, DIFFICULTY_MAP
from inference.generate import beam_search, greedy_decode


# Mapping from CloneHero instrument name to Demucs stem name
INSTRUMENT_TO_STEM = {
    "Single": "guitar_other",
    "DoubleRhythm": "guitar_other",
    "GuitarCoop": "guitar_other",
    "DoubleBass": "bass",
    "Drums": "drums",
}

# Mapping from instrument name to tokenizer token string
INSTRUMENT_TO_TOKEN = {
    "Single": "<Guitar>",
    "DoubleRhythm": "<Guitar>",
    "GuitarCoop": "<Guitar>",
    "DoubleBass": "<Bass>",
    "Drums": "<Drums>",
}

DIFFICULTY_TO_TOKEN = {
    "Expert": "<Expert>",
    "Hard": "<Hard>",
    "Medium": "<Medium>",
    "Easy": "<Easy>",
}

# Clone Hero section names for each instrument × difficulty
SECTION_NAMES = {
    ("Single", "Expert"): "ExpertSingle",
    ("Single", "Hard"): "HardSingle",
    ("Single", "Medium"): "MediumSingle",
    ("Single", "Easy"): "EasySingle",
    ("DoubleBass", "Expert"): "ExpertDoubleBass",
    ("DoubleBass", "Hard"): "HardDoubleBass",
    ("DoubleBass", "Medium"): "MediumDoubleBass",
    ("DoubleBass", "Easy"): "EasyDoubleBass",
    ("Drums", "Expert"): "ExpertDrums",
    ("Drums", "Hard"): "HardDrums",
    ("Drums", "Medium"): "MediumDrums",
    ("Drums", "Easy"): "EasyDrums",
}


class ChartGenerationPipeline:
    """
    Full pipeline from an audio file to a Clone Hero notes.chart string.

    Steps:
        1. Separate audio into stems via Demucs
        2. Extract log-mel spectrogram from the relevant stem
        3. Extract MERT embedding from the original audio
        4. Detect BPM (or use provided value)
        5. Encode with ChartTransformer encoder
        6. Autoregressively decode token sequence (beam search)
        7. Decode tokens → beat_sequence → notes.chart text
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {checkpoint_path}...")
        self.model = ChartTransformer.load(checkpoint_path, map_location=self.device)
        self.model.eval().to(self.device)

        self.tokenizer = CloneHeroTokenizer()
        self.mert = MERT()

    def generate(
        self,
        audio_path: str,
        instrument: str = "Single",
        difficulty: str = "Expert",
        bpm: Optional[float] = None,
        time_signature: int = 4,
        resolution: int = 192,
        offset: float = 0.0,
        beam_size: int = 4,
        max_new_tokens: int = 2044,
    ) -> str:
        """
        Generate a notes.chart section for the given audio and parameters.

        Args:
            audio_path:     Path to the input audio file
            instrument:     "Single" | "DoubleBass" | "Drums"
            difficulty:     "Expert" | "Hard" | "Medium" | "Easy"
            bpm:            Song BPM; if None, estimated from audio
            time_signature: Numerator of time signature (default 4)
            resolution:     Ticks per quarter note (default 192)
            offset:         Audio sync offset in seconds

        Returns:
            Complete notes.chart file content as a string
        """
        audio_path = str(audio_path)

        # Step 1: Stem separation
        print("Separating audio stems...")
        separator = DemucsAudioSeparator(audio_path)
        stem_name = INSTRUMENT_TO_STEM.get(instrument, "guitar_other")
        stem_array = separator.get_stem(stem_name, as_tensor=False)

        # Step 2: Log-mel spectrogram from stem
        print("Extracting log-mel spectrogram...")
        processor = AudioProcessor(stem_array, sample_rate=44100)
        log_mel = processor.calculate_logmel(
            n_mels=512, n_fft=4096, hop_length=1024, as_tensor=True
        )  # [512, T]

        # Step 3: MERT embedding from original audio
        print("Extracting MERT embeddings...")
        mert_emb = torch.tensor(self.mert.forward(audio_path), dtype=torch.float32)  # [768]

        # Step 4: BPM estimation if not provided
        if bpm is None:
            bpm = self._estimate_bpm(audio_path)
            print(f"Estimated BPM: {bpm:.1f}")

        # Step 5: Build encoder input and encode
        print("Encoding...")
        enc_out, enc_mask = self._encode(
            log_mel, mert_emb, bpm, time_signature, resolution, offset,
            instrument, difficulty,
        )

        # Step 6: Decode token sequence
        print("Generating chart tokens...")
        if beam_size > 1:
            token_ids = beam_search(
                self.model, enc_out, enc_mask,
                bos_id=self.tokenizer.vocab["<BOS>"],
                eos_id=self.tokenizer.vocab["<EOS>"],
                pad_id=self.tokenizer.vocab["<PAD>"],
                beam_size=beam_size,
                max_new_tokens=max_new_tokens,
            )
        else:
            token_ids = greedy_decode(
                self.model, enc_out, enc_mask,
                bos_id=self.tokenizer.vocab["<BOS>"],
                eos_id=self.tokenizer.vocab["<EOS>"],
                max_new_tokens=max_new_tokens,
            )

        # Step 7: Convert tokens to .chart format
        print("Writing chart...")
        chart_str = self._tokens_to_chart(
            token_ids, bpm, time_signature, resolution, offset,
            instrument, difficulty, audio_path,
        )
        return chart_str

    def _encode(
        self,
        log_mel: torch.Tensor,
        mert_emb: torch.Tensor,
        bpm: float,
        ts: int,
        resolution: int,
        offset: float,
        instrument: str,
        difficulty: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.model.config
        # Pad log_mel to enc_max_frames
        T = log_mel.shape[1]
        enc_max_frames = 8192
        padded = torch.zeros(1, 512, enc_max_frames)
        padded[0, :, : min(T, enc_max_frames)] = log_mel[:, : enc_max_frames]
        padded = padded.to(self.device)

        # Encoding padding mask
        n_audio = min(T // 16, 512 - 7)
        enc_mask = torch.zeros(1, 512, dtype=torch.bool, device=self.device)
        if 7 + n_audio < 512:
            enc_mask[0, 7 + n_audio :] = True

        inst_idx = torch.tensor([INSTRUMENT_MAP.get(instrument, 0)], device=self.device)
        diff_idx = torch.tensor([DIFFICULTY_MAP.get(difficulty, 0)], device=self.device)

        with torch.no_grad():
            enc_out = self.model.encode(
                log_mel=padded,
                mert_emb=mert_emb.unsqueeze(0).to(self.device),
                bpm=torch.tensor([[bpm]], device=self.device),
                ts=torch.tensor([[float(ts)]], device=self.device),
                resolution=torch.tensor([[float(resolution)]], device=self.device),
                offset=torch.tensor([[offset]], device=self.device),
                instrument_idx=inst_idx,
                difficulty_idx=diff_idx,
                enc_padding_mask=enc_mask,
            )

        return enc_out, enc_mask

    def _tokens_to_chart(
        self,
        token_ids: List[int],
        bpm: float,
        ts: int,
        resolution: int,
        offset: float,
        instrument: str,
        difficulty: str,
        audio_path: str,
    ) -> str:
        """
        Decode the generated token IDs into a full notes.chart text.
        """
        id_to_token = self.tokenizer.ids_to_tokens

        # Strip BOS, instrument, difficulty tokens from the start, and EOS at the end
        bos_id = self.tokenizer.vocab["<BOS>"]
        eos_id = self.tokenizer.vocab["<EOS>"]

        try:
            start = token_ids.index(bos_id) + 3  # skip BOS + instrument + difficulty
        except ValueError:
            start = 0
        try:
            end = token_ids.index(eos_id, start)
        except ValueError:
            end = len(token_ids)

        body = token_ids[start:end]

        # Parse 6-token blocks into note events
        note_events = []  # list of (tick, button, duration)
        seconds_per_beat = 60.0 / bpm

        i = 0
        while i + 5 < len(body):
            beatshift_id, notetype_id, pitch_id, minute_id, beat_id, dur_bs_id = body[i : i + 6]

            beatshift_tok = id_to_token.get(beatshift_id, "")
            pitch_tok = id_to_token.get(pitch_id, "")
            minute_tok = id_to_token.get(minute_id, "")
            beat_tok = id_to_token.get(beat_id, "")
            dur_bs_tok = id_to_token.get(dur_bs_id, "")

            if not all([beatshift_tok, pitch_tok, minute_tok, beat_tok]):
                i += 6
                continue

            # Extract numeric values from token strings
            try:
                init_bs = int(beatshift_tok.replace("<Beatshift_", "").replace(">", ""))
                minute = int(minute_tok.replace("<Minute_", "").replace(">", ""))
                beat = int(beat_tok.replace("<Beat_", "").replace(">", ""))
                dur_bs = int(dur_bs_tok.replace("<Beatshift_", "").replace(">", ""))

                if "Pitch_" in pitch_tok:
                    button = int(pitch_tok.replace("<Pitch_", "").replace(">", ""))
                elif "DrumsPitch_" in pitch_tok:
                    button = int(pitch_tok.replace("<DrumsPitch_", "").replace(">", ""))
                else:
                    i += 6
                    continue
            except (ValueError, AttributeError):
                i += 6
                continue

            # Convert beat position to ticks
            # total_beats = minute * 60 / seconds_per_beat + beat + init_bs/32
            total_beats = (minute * 60.0 / seconds_per_beat) + beat + (init_bs / 32.0)
            tick = int(total_beats * resolution)

            # Duration in ticks (dur_bs subdivisions × 1 beat / 32)
            dur_ticks = int((dur_bs / 32.0) * resolution)

            # Note type: 0=Normal, 1=Special (StarPower)
            notetype_tok = id_to_token.get(notetype_id, "")
            is_special = "<Special>" in notetype_tok

            note_events.append((tick, button, dur_ticks, is_special))
            i += 6

        # Sort by tick position
        note_events.sort(key=lambda x: x[0])

        # Build the .chart text
        audio_filename = Path(audio_path).name
        section_name = SECTION_NAMES.get((instrument, difficulty), f"{difficulty}{instrument}")

        lines = [
            "[Song]",
            "{",
            f'  Name = "Generated Chart"',
            f'  Artist = "Unknown"',
            f'  Charter = "CloneCharter AI"',
            f"  Resolution = {resolution}",
            f"  Offset = {offset}",
            f'  MusicStream = "{audio_filename}"',
            "}",
            "",
            "[SyncTrack]",
            "{",
            f"  0 = TS {ts}",
            f"  0 = B {int(bpm * 1000)}",
            "}",
            "",
            "[Events]",
            "{",
            "}",
            "",
            f"[{section_name}]",
            "{",
        ]

        for tick, button, duration, is_special in note_events:
            if is_special:
                lines.append(f"  {tick} = S 2 {duration}")
            else:
                lines.append(f"  {tick} = N {button} {duration}")

        lines.append("}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _estimate_bpm(audio_path: str) -> float:
        """Estimate BPM from audio using librosa."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None, mono=True, duration=60)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except Exception:
            return 120.0


def main():
    parser = argparse.ArgumentParser(description="Generate a Clone Hero chart from audio")
    parser.add_argument("--audio", required=True, help="Input audio file path")
    parser.add_argument("--instrument", default="Single",
                        choices=["Single", "DoubleBass", "Drums", "DoubleRhythm", "GuitarCoop"])
    parser.add_argument("--difficulty", default="Expert",
                        choices=["Expert", "Hard", "Medium", "Easy"])
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output", default="notes.chart", help="Output .chart file path")
    parser.add_argument("--bpm", type=float, default=None, help="Override BPM detection")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipeline = ChartGenerationPipeline(args.checkpoint, device=args.device)

    chart_str = pipeline.generate(
        audio_path=args.audio,
        instrument=args.instrument,
        difficulty=args.difficulty,
        bpm=args.bpm,
        beam_size=args.beam_size,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(chart_str)

    print(f"Chart written to: {args.output}")


if __name__ == "__main__":
    main()
