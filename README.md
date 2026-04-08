# auto-charter — Clone Hero Auto-Charting Pipeline

A complete pipeline for building HuggingFace datasets from Clone Hero charts **and training a Transformer model** to generate new charts from audio.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thejorseman/auto-charter
cd auto-charter

# Install all dependencies (CUDA 12.8 PyTorch + Demucs + MERT + training stack)
uv sync

# Activate the virtual environment
source .venv/bin/activate          # Linux/macOS
# or: .venv\Scripts\activate       # Windows

# Verify installation
uv run pytest tests/ -v
```

**GPU support:** The project targets CUDA 12.8 by default (H100 / RTX 40/50 series). If you use a different CUDA version, edit the `[tool.uv.index]` in `pyproject.toml` before running `uv sync`.

---

## Quick Start

### 1. Build a dataset

```bash
uv run process-dataset \
  -i /path/to/clone_hero_songs/ \
  -o ./my_dataset/ \
  --extract-logmel \
  --extract-mert \
  --separate-stems \
  --device cuda
```

### 2. Train the model

```bash
# Configure accelerate once (choose bf16 for H100, fp16 for RTX)
accelerate config

# Train (single GPU)
accelerate launch src/auto_charter/scripts/train.py \
  --dataset ./my_dataset/ \
  --output-dir ./checkpoints/run1 \
  --batch-size 4 --grad-accum 4 --num-epochs 100

# Train (8× H100)
accelerate launch --num_processes 8 --multi_gpu \
  src/auto_charter/scripts/train.py \
  --dataset ./my_dataset/ --output-dir ./checkpoints/run1
```

### 3. Validate

```bash
uv run validate-charter \
  --checkpoint ./checkpoints/run1/best \
  --dataset ./my_dataset/ \
  --output-json metrics.json
```

### 4. Generate a chart (Gradio demo)

```bash
uv run demo-charter --checkpoint ./checkpoints/run1/best --port 7860
# Open http://localhost:7860
# Upload a song → select instrument + difficulty → download .zip
```

---

## Quick Example (Python API)

```bash
# Build a dataset with log-mel spectrograms
uv run process-dataset \
  -i /path/to/clone_hero_songs/ \
  -o ./my_dataset/ \
  --extract-logmel

# Push to HuggingFace Hub
uv run process-dataset \
  -i /path/to/songs/ \
  --push-to-hub username/clone-hero-charts \
  --extract-logmel --device cuda
```

Load and train:

```python
from datasets import load_dataset
from auto_charter.dataset.collator import AutoCharterCollator
from torch.utils.data import DataLoader

ds = load_dataset("username/clone-hero-charts")
collator = AutoCharterCollator(max_tokens=8192, return_tensors="pt")
loader = DataLoader(ds["train"], batch_size=4, collate_fn=collator)

# Ready to feed into a transformer!
for batch in loader:
    # batch["input_ids"]: [B, T] token IDs
    # batch["mert_embeddings"]: [B, num_beats, 768]
    # batch["logmel_frames"]: [B, num_beats, 32, 128]
    pass
```

---

## Features

✅ **Chart Parsing** — `.chart` and `.mid` formats, variable BPM + time signatures
✅ **Event-Based Tokenization** — 187-token vocabulary, handles chords/sustains/modifiers
✅ **Beat-Level Audio Conditioning** — MERT embeddings + log-mel spectrograms
✅ **Automatic Stem Separation** — Demucs htdemucs for instrument isolation from mixes
✅ **HuggingFace Integration** — Arrow datasets, Hub push/pull, standard schema
✅ **Round-Trip Reversibility** — encode → decode → re-encode produces identical tokens
✅ **Comprehensive Testing** — 39 unit tests, all test songs verified

---

## Core Modules

### Vocabulary (`auto_charter.vocab`)
- **tokens.py** — 187 token IDs: WAIT, NOTE, DRUM, SUS, BEAT_BOUNDARY, SP_ON/OFF, etc.
- **guitar_vocab.py** — chord bitmask encoding (31 lane combinations)
- **drum_vocab.py** — drum hit bitmasks (5 lanes, no sustain)

### Parsing (`auto_charter.parsers`)
- **chart_parser.py** — `.chart` file parsing with [SyncTrack] (BPM/time-sig) support
- **midi_parser.py** — `.mid` parsing with tick normalization to resolution=192
- **ini_parser.py** — song.ini metadata extraction
- **sync_track.py** — BPM mapping, tick↔seconds conversion, beat grid generation

### Tokenization (`auto_charter.tokenizer`)
- **encoder.py** — chart → token sequence (autoregressive, deterministic)
- **decoder.py** — token sequence → chart (reversible)
- **quantize.py** — 16-tick grid snapping, sustain quantization

### Audio (`auto_charter.audio`)
- **stem_loader.py** — resolve dedicated stems or fall back to mix
- **separator.py** — Demucs htdemucs wrapper for automatic stem separation
- **mert_extractor.py** — MERT hidden states → [num_beats, 768]
- **logmel.py** — log-mel spectrogram → [num_beats, 32, 128] (resampled per beat)
- **beat_aligner.py** — slice/resample audio features to beat boundaries
- **beat_estimator.py** — librosa-based beat detection for inference-time use

### Model (`auto_charter.model`)
- **config.py** — `AutoCharterConfig` dataclass (hyperparameters, save/load)
- **audio_encoder.py** — `AudioEncoder`: MERT + LogMel + BPM + time-sig → Transformer encoder
- **token_decoder.py** — `AutoregressiveDecoder`: causal decoder with beat-distance cross-attention bias
- **charter_model.py** — `AutoCharterModel`: full encoder-decoder, `.generate()`, `.save_pretrained()`

### Training (`auto_charter.training`)
- **dataset.py** — `AutoCharterDataset`: filters missing audio, stratified 80/20 split by instrument
- **collator.py** — `AutoCharterTrainCollator`: computes `beat_ids`, instrument/difficulty tensors, beat timing
- **metrics.py** — token accuracy, perplexity, note-level F1, beat accuracy
- **trainer.py** — `AutoCharterTrainer`: Accelerate loop, early stopping, TensorBoard/W&B logging

### Dataset (`auto_charter.dataset`)
- **schema.py** — HuggingFace Features definition (30 fields)
- **builder.py** — SongProcessor: song folder → dataset rows (with stats computation)
- **collator.py** — AutoCharterCollator: batching + padding for training

### Scripts (`auto_charter.scripts`)
- **process-dataset** — main CLI: multiple input folders → HuggingFace dataset
- **inspect-song** — debug: print human-readable token traces
- **validate-roundtrip** — verify encode↔decode reversibility
- **train-charter** — train `AutoCharterModel` with Accelerate (single/multi-GPU)
- **validate-charter** — evaluate on test split, print per-instrument metrics table
- **demo-charter** — Gradio demo: upload audio → generate chart → download .zip

---

## Quantization & Vocabulary

**16-tick grid** handles all observed note spacings:

| Subdivision    | Ticks | Appears in dataset |
|---|---|---|
| Quarter note   | 192   | ✓ (base beat) |
| 8th note       | 96    | ✓ |
| 16th note      | 48    | ✓ |
| 16th triplet   | 32    | ✓ **Critical** (sierreño) |
| 32nd note      | 24    | ✓ |

**Token vocabulary** (187 IDs):

- **0–5** — Special: PAD, BOS, EOS, UNK, BEAT_BOUNDARY, MEASURE_START
- **6–8** — Instrument: GUITAR, BASS, DRUMS
- **9–56** — WAIT_k (k=1..48, advances by k×16 ticks)
- **57–87** — Guitar/bass chords (31 lane bitmasks: 0b00000–0b11111)
- **88–91** — Modifiers: HOPO, TAP, OPEN, FORCE_STRUM
- **92–122** — Drum chords (31 bitmasks, no sustain)
- **123–182** — Sustain durations (60 steps: 0–752 ticks linear, 864–2640 coarse)
- **183–186** — Events: STAR_POWER_ON/OFF, SOLO_ON/OFF

---

## Example Dataset Structure

One row per instrument per song:

```python
{
  "song_id": "a3f9b1c2d0e4...",
  "instrument": "guitar",
  "source_format": "chart",

  "tokens": [1, 6, 4, 17, 183, 11, 60, 123, 91, ...],  # [T]
  "num_tokens": 3231,
  "num_beats": 396,

  "mert_embeddings": [...],     # [396, 768] float32
  "logmel_frames": [...],       # [396, 32, 128] float32

  "beat_times_s": [0.0, 0.462, 0.923, ...],
  "beat_durations_s": [0.462, ...],
  "bpm_at_beat": [130.0, ...],

  "song_name": "El Precio de la Soledad",
  "artist": "Alfredo Olivas",
  "genre": "banda",
  "charter": "Spidey_3089",
  "year": 2012,
  "song_length_ms": 184668,
  "difficulty": 2,

  "has_star_power": true,
  "has_solo": true,
  "has_dedicated_stem": false,

  "num_notes": 640,
  "notes_per_beat_mean": 1.62,
  "chord_ratio": 0.0,
  "bpm_mean": 130.0,
  "bpm_std": 0.0,
}
```

---

## Commands

```bash
# Build dataset from song folders
uv run process-dataset -i songs1/ -i songs2/ -o ./dataset/ \
    --extract-logmel --extract-mert --device cuda

# Train model (configure accelerate first: accelerate config)
accelerate launch src/auto_charter/scripts/train.py \
    --dataset ./dataset/ --output-dir ./runs/run1

# Evaluate on test split
uv run validate-charter --checkpoint ./runs/run1/best --dataset ./dataset/

# Launch Gradio demo
uv run demo-charter --checkpoint ./runs/run1/best

# Debug: print token traces for inspection
uv run inspect-song "path/to/song/" --instrument guitar

# Verify: encode→decode→re-encode roundtrip test
uv run validate-roundtrip test_dataset/
```

---

## Documentation

- **[TOKENIZATION.md](TOKENIZATION.md)** — Complete tokenization guide with examples from real songs
- **[PROCESS_DATASET.md](PROCESS_DATASET.md)** — Command reference, workflows, troubleshooting
- **README.MD** (in parent) — Clone Hero chart format (.chart, .mid, song.ini) specification

---

## Testing

```bash
# Run all tests (39 passed)
uv run pytest tests/ -v

# Specific test
uv run pytest tests/test_tokenizer.py::test_roundtrip_guitar -v

# With coverage
uv run pytest tests/ --cov=auto_charter
```

**Test coverage:**
- Chart parsing (format, BPM/time-sig, events)
- MIDI normalization (tick conversion, pitch maps)
- Tokenization (vocab, encoding, decoding, round-trips)
- Audio extraction shapes (MERT, log-mel, beat alignment)

---

## Example Workflows

### Scenario 1: Build a dataset with dedicated stems (no Demucs)

```bash
uv run process-dataset \
  -i ~/clone_hero_library/ \
  -o ./clone_hero.hf/ \
  --extract-logmel \
  --extract-mert \
  --device cuda
```

**Time:** ~5s per song (with GPU)

### Scenario 2: Separate stems first, then extract features

```bash
uv run process-dataset \
  -i ~/clone_hero_library/ \
  -o ./clone_hero.hf/ \
  --separate-stems \
  --stems-cache /tmp/separated_stems/ \
  --extract-logmel \
  --device cuda
```

**Time:** ~20s per song first run (Demucs), ~2s subsequent runs (cached stems)

### Scenario 3: Fast tokenization only (no audio)

```bash
uv run process-dataset \
  -i ~/songs/ \
  -o ./tokens_only.hf/ \
  --no-logmel \
  --max-songs 100  # test with 100 songs
```

**Time:** ~1s per song

### Scenario 4: Push to HuggingFace Hub

```bash
huggingface-cli login  # once

uv run process-dataset \
  -i ~/clone_hero_library/ \
  --push-to-hub thejorseman/clone-hero-charts \
  --extract-logmel \
  --split train  # use "eval" or "test" for other splits
```

Then load anywhere:

```python
from datasets import load_dataset
ds = load_dataset("thejorseman/clone-hero-charts", "train")
```

---

## Transformer Architecture

AutoCharterModel is a **conditioned encoder-decoder Transformer** (~10–12M parameters, designed for small datasets):

```
AudioEncoder                      TokenDecoder
────────────                      ────────────
MERT [B,N,768]    ──┐             Token Emb + Pos Enc
LogMel [B,N,32,128] ──→ Sum ──→   + instrument/difficulty bias
BPM [B,N]         ──┘   ↓         (added to every position)
Time Sig [B,N]          Transformer Encoder    ↓
Beat Duration [B,N]     (4 layers, d=256)  Causal Self-Attn (Flash Attn 2)
                              ↓                  ↓
                    audio_context [B,N,256] ──→ Cross-Attn
                                              (beat distance bias)
                                                  ↓
                                           Logits [B,T,187]
```

**Beat alignment:** Each token position maps to a beat via `beat_ids` (incremented at each `BEAT_BOUNDARY` token). A learnable ALiBi-like bias biases cross-attention toward the current beat.

**Conditioning:** Instrument (guitar/bass/drums) and difficulty (0–6) are added as fixed offsets to every decoder position — simple and effective for small models.

See **TOKENIZATION.md § 11** for full architecture details.

---

## Performance

| Task | Time/Song | Notes |
|---|---|---|
| Parse + tokenize | 1–3s | CPU |
| Log-mel extraction | 2–3s | CPU, depends on song length |
| MERT extraction | 5–10s | GPU (cuda), 30–60s CPU |
| Demucs separation | 15–30s | GPU (first run), cached after |
| **Full pipeline (MERT+Demucs)** | **30–40s** | GPU recommended |
| **Minimal (tokens only)** | **1–2s** | CPU sufficient |

For 150 songs: ~5 hours full pipeline (GPU), ~5 minutes tokenization only (CPU).

---

## Roadmap

- [x] Encoder-Decoder Transformer model with beat-aligned cross-attention
- [x] Accelerate training (single GPU / 8× H100 / RTX 5070 Ti)
- [x] Gradio demo with full pipeline (audio → .zip)
- [ ] KV-cache for faster autoregressive generation
- [ ] Curriculum learning (sort by BPM variance, chord density)
- [ ] Support for other difficulty levels (Hard/Medium/Easy chart generation)
- [ ] Benchmarks against professional charters

---

## Citation

If you use this in research, cite:

```bibtex
@software{auto_charter_2025,
  title={Auto-Charter: Clone Hero Song Tokenization & Dataset Pipeline},
  author={[thejorseman]},
  year={2025},
  url={https://github.com/thejorseman/auto-charter}
}
```

---

## License

MIT — see LICENSE file

---

## Contributing

Pull requests welcome! Please:
1. Run tests: `uv run pytest tests/`
2. Format: `uv run ruff check --fix src/ tests/`
3. Include test coverage for new features

---

## Contact

Questions? Open an issue on GitHub or check the documentation above.
