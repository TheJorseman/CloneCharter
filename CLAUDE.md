# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CloneCharter is an auto-charter for Clone Hero — it tokenizes `.chart`/`.mid` files into sequences and trains a Transformer to generate charts from audio. The pipeline extracts MERT audio embeddings and log-mel spectrograms per beat, then trains an encoder-decoder model conditioned on instrument+difficulty.

## Commands

```bash
# Install dependencies (requires uv, CUDA 12.8)
uv sync

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_tokenizer.py::test_roundtrip_guitar -v  # single test

# Format
uv run ruff check --fix src/ tests/

# Build dataset from song folders
uv run process-dataset -i /path/to/songs/ -o ./dataset/ \
    --extract-logmel --extract-mert --device cuda

# Train (configure accelerate first: accelerate config)
accelerate launch src/auto_charter/scripts/train.py \
    --dataset ./dataset/ --output-dir ./checkpoints/run1 \
    --batch-size 4 --grad-accum 4 --num-epochs 100

# Train streaming from HuggingFace Hub
accelerate launch src/auto_charter/scripts/train.py \
    --hub-dataset thejorseman/CloneHeroDatasetCharts \
    --streaming --steps-per-epoch 2000 --output-dir ./checkpoints/run1

# Evaluate
uv run validate-charter --checkpoint ./checkpoints/run1/best --dataset ./dataset/

# Gradio demo
uv run demo-charter --checkpoint ./checkpoints/run1/best

# Debug token traces
uv run inspect-song "path/to/song/" --instrument guitar

# Verify roundtrip (encode→decode→re-encode)
uv run validate-roundtrip test_dataset/
```

## Architecture

**AutoCharterModel** — encoder-decoder Transformer (~10–12M params):

```
AudioEncoder                          TokenDecoder
MERT [B,N,768] ────────────────────→  Token Embedding + Pos Encoding
LogMel [B,N,32,128] ──→ concat ──→    + instrument/difficulty bias (added to every position)
BPM [B,N]                            (4-layer Transformer encoder, d=256)
TimeSig [B,N]                              ↓
Beat Duration [B,N]              audio_context [B,N,256] ──→ Cross-Attention
                                        (beat-distance ALiBi bias)
                                               ↓
                                        Logits [B,T,187]
```

**Beat alignment**: Each token position maps to a beat via `beat_ids` (incremented at each `BEAT_BOUNDARY` token). Cross-attention uses a learnable ALiBi-like bias based on distance between current beat and target beat.

**Conditioning**: Instrument (guitar/bass/drums) and difficulty (0–6) are added as fixed offset embeddings to every decoder position — simple and effective for small models.

## Quantization & Vocabulary

**16-tick grid** handles all observed note spacings:

| Subdivision    | Ticks | Notes |
|---|---|---|
| Quarter note   | 192   | base beat |
| 8th note       | 96    | |
| 16th note      | 48    | |
| 16th triplet   | 32    | **Critical** — appears in siereño patterns |
| 32nd note      | 24    | |

**Token vocabulary** (187 IDs):
- **0–5**: Special tokens (PAD, BOS, EOS, UNK, BEAT_BOUNDARY, MEASURE_START)
- **6–8**: Instrument tokens (GUITAR, BASS, DRUMS)
- **9–56**: WAIT tokens (48 durations, k×16 ticks each, k=1..48)
- **57–87**: Guitar/bass chords (31 lane bitmasks)
- **88–91**: Modifiers (HOPO, TAP, OPEN, FORCE_STRUM)
- **92–122**: Drum chords (31 bitmasks)
- **123–182**: Sustain durations (60 steps, 0–752 ticks linear + 864–2640 coarse)
- **183–186**: Events (STAR_POWER_ON/OFF, SOLO_ON/OFF)

## Dataset Schema

One row per instrument per song, with these key fields:
- `tokens [T]`: encoded chart tokens
- `num_beats`: number of beats in the song
- `mert_embeddings [num_beats, 768]`: MERT hidden states per beat
- `logmel_frames [num_beats, 32, 128]`: log-mel spectrogram per beat
- `beat_times_s`, `beat_durations_s`, `bpm_at_beat`: beat timing arrays
- `instrument`, `difficulty`, `song_name`, `artist`, `genre`, `charter`, `year`

## File Structure

```
src/auto_charter/
├── vocab/           # 187-token vocabulary (tokens.py, guitar_vocab.py, drum_vocab.py)
├── parsers/         # .chart/.mid parsing (chart_parser.py, midi_parser.py, sync_track.py)
├── tokenizer/       # encode/decode (encoder.py, decoder.py, quantize.py)
├── audio/           # MERT, log-mel, Demucs stem separation, beat alignment
├── model/           # AutoCharterModel (audio_encoder.py, token_decoder.py, charter_model.py)
├── training/        # dataset, collator, trainer, metrics
├── dataset/         # schema, builder, collator for HuggingFace datasets
└── scripts/         # CLI entry points (train.py, process_dataset.py, validate.py, etc.)
tests/               # unit tests
```

## Key Implementation Notes

- **Roundtrip**: encoder → decoder → re-encoder produces identical tokens — test with `validate-roundtrip`
- **Beat alignment**: `beat_aligner.py` resamples audio features to beat boundaries; `beat_estimator.py` handles inference
- **Streaming dataset**: `StreamingAutoCharterDataset` is an IterableDataset; trainer auto-detects this (disables shuffle, uses `steps_per_epoch` for LR schedule). Use `materialize_val()` to snapshot a fixed validation set
- **Stem loading**: `stem_loader.py` resolves dedicated stems; falls back to mix if not available
- **Tick normalization**: MIDI parser normalizes to resolution=192 ticks per quarter note

## Dependencies

PyTorch 2.7.0 (CUDA 12.8), MERT audio encoder, Demucs stem separation, HuggingFace datasets/transformers stack. Edit `[tool.uv.index]` in `pyproject.toml` to change CUDA version.