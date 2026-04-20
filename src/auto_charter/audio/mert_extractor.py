"""Beat-synchronous MERT embedding extraction.

MERT (Music Encoder Representations from Transformers) is a self-supervised
music audio model that produces 768-dimensional frame-level embeddings at ~75Hz.

Model: m-a-p/MERT-v1-95M (HuggingFace hub)

This module:
1. Loads the MERT model and processor (cached after first use)
2. Runs inference on an audio file (in chunks for long songs)
3. Mean-pools the frame embeddings over each beat window
4. Returns [num_beats, 768] float32 array

Dependencies: transformers, torch, librosa/soundfile

Note: MERT requires 24kHz mono audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

from .beat_aligner import mean_pool_beats

MERT_MODEL_NAME = "m-a-p/MERT-v1-330M"
MERT_SR = 24000       # MERT expects 24kHz
MERT_HZ = 75.0        # approximate frame rate of MERT hidden states
MERT_DIM = 1024       # hidden state dim for MERT-v1-330M (95M variant is 768)
CHUNK_DURATION_S = 30 # process audio in 30-second chunks


class MERTExtractor:
    """Extract beat-synchronous MERT embeddings.

    Args:
        model_name: HuggingFace model ID (default "m-a-p/MERT-v1-330M").
        device: Torch device string (default "cpu"; use "cuda" if available).
        chunk_duration_s: Audio chunk size for long songs (default 30s).
    """

    def __init__(
        self,
        model_name: str = MERT_MODEL_NAME,
        device: str | None = None,
        chunk_duration_s: float = CHUNK_DURATION_S,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for MERT extraction: pip install torch")
        if not _LIBROSA_AVAILABLE:
            raise ImportError("librosa is required: pip install librosa")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_duration_s = chunk_duration_s
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy-load MERT model and processor."""
        if self._model is not None:
            return
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    def _run_chunk(self, waveform: np.ndarray) -> np.ndarray:
        """Run MERT on a single audio chunk. Returns [T, 768] float32."""
        import torch
        self._load_model()
        inputs = self._processor(
            waveform, sampling_rate=MERT_SR, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=False)
        # Use last hidden state: [1, T, 768]
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hidden.astype(np.float32)

    def extract(self, audio_path: str | Path) -> np.ndarray:
        """Load audio and extract full MERT hidden state sequence.

        Returns:
            Float32 array [T_frames, 768].
        """
        from auto_charter.audio.audio_io import load_audio
        y = load_audio(audio_path, sr=MERT_SR, mono=True)
        chunk_size = int(self.chunk_duration_s * MERT_SR)

        if len(y) <= chunk_size:
            return self._run_chunk(y)

        # Process in chunks and concatenate
        chunks = []
        start = 0
        while start < len(y):
            end = min(start + chunk_size, len(y))
            chunk_hidden = self._run_chunk(y[start:end])
            chunks.append(chunk_hidden)
            start = end

        return np.concatenate(chunks, axis=0)

    def extract_per_beat(
        self,
        audio_path: str | Path,
        beat_times_s: list[float],
        beat_durations_s: list[float],
    ) -> np.ndarray:
        """Extract MERT embeddings mean-pooled over each beat.

        Streams through audio in chunks without accumulating all hidden states,
        so VRAM and RAM stay bounded even for very long songs.

        Returns:
            Float32 array [num_beats, 768].
        """
        from auto_charter.audio.audio_io import load_audio
        self._load_model()

        y = load_audio(audio_path, sr=MERT_SR, mono=True)
        chunk_size = int(self.chunk_duration_s * MERT_SR)
        num_beats = len(beat_times_s)

        counts = np.zeros(num_beats, dtype=np.int64)
        accum: np.ndarray | None = None  # allocated after first chunk (dim inferred)

        start = 0
        while start < len(y):
            end = min(start + chunk_size, len(y))
            chunk_start_s = start / MERT_SR
            chunk_end_s = end / MERT_SR

            hidden = self._run_chunk(y[start:end])  # [T_chunk, D]
            T_chunk = len(hidden)
            actual_dim = hidden.shape[1]
            if accum is None:
                accum = np.zeros((num_beats, actual_dim), dtype=np.float64)

            for beat_idx in range(num_beats):
                bt = float(beat_times_s[beat_idx])
                bd = float(beat_durations_s[beat_idx])
                beat_end_s = bt + bd

                # Skip beats that don't overlap this chunk at all
                if beat_end_s <= chunk_start_s or bt >= chunk_end_s:
                    continue

                local_start_s = max(0.0, bt - chunk_start_s)
                local_end_s = min(beat_end_s - chunk_start_s, chunk_end_s - chunk_start_s)

                f_start = int(local_start_s * MERT_HZ)
                f_end = max(f_start + 1, int(local_end_s * MERT_HZ))
                f_end = min(f_end, T_chunk)
                if f_start >= T_chunk or f_start >= f_end:
                    continue

                accum[beat_idx] += hidden[f_start:f_end].sum(axis=0)
                counts[beat_idx] += f_end - f_start

            del hidden  # free immediately, don't wait for GC
            start = end

        del y

        # Average pooling — beats with no frames get zero embedding
        if accum is None:
            # Audio was empty; return zeros with the constant fallback dim
            return np.zeros((num_beats, MERT_DIM), dtype=np.float32)
        result = np.zeros_like(accum, dtype=np.float32)
        valid = counts > 0
        result[valid] = (accum[valid] / counts[valid, np.newaxis]).astype(np.float32)
        return result
