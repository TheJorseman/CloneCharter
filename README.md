# CloneCharter

A transformer-based automatic charter for Clone Hero. This project processes Clone Hero song charts (`.chart` files) and their associated audio to build a large-scale machine learning dataset for training a model that can automatically generate rhythm game charts from audio.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset Structure](#dataset-structure)
  - [clone\_hero\_dataset\_final/](#clone_hero_dataset_final)
  - [test\_dataset/](#test_dataset)
  - [Song Folder Format](#song-folder-format)
- [File Formats](#file-formats)
  - [.chart Format](#chart-format)
  - [.mid Format](#mid-format)
- [Tokenizer](#tokenizer)
- [Processing Pipeline](#processing-pipeline)
- [Key Scripts](#key-scripts)
- [Models & Utilities](#models--utilities)
- [Technologies](#technologies)

---

## Project Overview

The goal is to create an AI system that takes raw audio as input and outputs a valid Clone Hero chart. The pipeline:

1. **Parse** Clone Hero song folders (`.chart` + audio)
2. **Tokenize** the chart data into a structured sequence
3. **Embed** the audio using MERT (music transformer)
4. **Separate** audio into stems (drums, bass, vocals, other) using Demucs
5. **Store** everything as a Hugging Face Arrow dataset (~74 GB)
6. **Train** a transformer model on the resulting dataset

The dataset is published to the Hugging Face Hub at `thejorseman/clone_hero_dataset`.

---

## Repository Structure

```
CloneCharter/
├── data/                           # Data loading modules
│   ├── chart_loader.py             # CloneHeroChartParser — parses .chart files
│   ├── audio_loaders.py            # AudioProcessor — loads and processes audio
│   └── midi_loader.py              # MIDIProcessor — loads and creates MIDI files
│
├── models/                         # ML model wrappers
│   ├── tokenizer.py                # CloneHeroTokenizer (HuggingFace PreTrainedTokenizer)
│   ├── demucs.py                   # DemucsAudioSeparator — stem separation
│   └── mert.py                     # MERT — music audio embeddings
│
├── utils/                          # Shared utilities
│   ├── time_utils.py               # BPM / tick / second conversion functions
│   └── audio_utils.py              # Musical data interpolation for NN input
│
├── training/                       # Transformer training code (see plan)
│   ├── config.py                   # TrainingConfig + ModelConfig dataclasses
│   ├── dataset.py                  # CloneHeroDataset — song-level train/val split
│   ├── collator.py                 # ChartCollator — dynamic padding + masks
│   ├── train.py                    # Main loop (HuggingFace Accelerate, 8-GPU DDP)
│   └── metrics.py                  # Perplexity, Token Acc, Note F1, Timing Acc
│
├── inference/                      # Inference pipeline
│   ├── generate.py                 # Beam search + greedy decode
│   └── pipeline.py                 # ChartGenerationPipeline: audio → notes.chart
│
├── scripts/                        # Data pipeline & analysis scripts
│   ├── tokenize_chart_dataset.py   # Main pipeline — processes charts → Arrow dataset
│   ├── parse_all_songs.py          # Scans folders, builds chart_analysis_results.json
│   ├── merge_dataset.py            # Merges dataset shards and uploads to HF Hub
│   ├── migrate_pickle.py           # Converts legacy pickle files to Arrow format
│   ├── data_anal.py                # Chart analysis and Plotly visualizations
│   └── duration_anal.py            # Note duration histogram analysis
│
├── tests/                          # Unit & integration tests
│   ├── test_chart_loader.py
│   ├── test_tokenizer.py
│   ├── test_bpm.py
│   ├── test_separator.py
│   └── ...
│
├── samples/                        # Example chart files for reference
│   ├── notes.chart                 # Full multi-difficulty guitar chart
│   ├── notes_drums.chart           # Complex drums chart
│   ├── nueva_cancion.chart         # Minimal chart skeleton
│   └── song.ini                    # Song metadata example
│
├── docs/
│   └── tokenizer.txt               # Token vocabulary reference
│
└── test_dataset/                   # 3 complete songs for local testing
    ├── Caos - La Planta/
    ├── El Precio de la Soledad/
    └── Grupo Marca Registrada - El Rescate/
```

> **Not tracked in git** (see `.gitignore`): the Arrow dataset (`clone_hero_dataset_final/`, ~74 GB), audio files, processing checkpoints, generated analysis outputs, and MIDI outputs.

---

## Dataset Structure

### `clone_hero_dataset_final/`

The final processed dataset in [Apache Arrow](https://arrow.apache.org/) format (Hugging Face Datasets).

```
clone_hero_dataset_final/
├── dataset_info.json               # Schema and feature definitions
├── state.json                      # Dataset state metadata
├── data-00000-of-00158.arrow       # Shard 0
├── data-00001-of-00158.arrow       # Shard 1
│   ...
└── data-00157-of-00158.arrow       # Shard 157 (total ~74 GB)
```

| Property | Value |
|---|---|
| Format | Apache Arrow (Hugging Face Datasets) |
| Shards | 158 files |
| Total Size | ~74 GB |
| HF Hub | `thejorseman/clone_hero_dataset` |

Each record in the dataset represents a tokenized chart track and includes:
- Tokenized chart sequence (instrument, difficulty, notes, timing)
- Audio embeddings extracted by MERT
- Song metadata (BPM, time signature, resolution, offset)

Processing progress is saved to `checkpoints/checkpoint_metadata.json` so runs can be resumed.

---

### `test_dataset/`

Three complete Clone Hero song folders used for development and testing:

```
test_dataset/
├── Caos - La Planta/
│   ├── notes.chart
│   ├── song.ini
│   ├── song.ogg
│   └── album.png
│
├── El Precio de la Soledad/
│   ├── notes.chart
│   ├── song.ini
│   ├── song.ogg
│   └── album.png
│
└── Grupo Marca Registrada - El Rescate/
    ├── notes.chart
    ├── song.ini
    ├── guitar.ogg
    ├── bass.ogg
    ├── vocals.ogg
    └── album.png
```

| Song | Genre | Length |
|---|---|---|
| Caos - La Planta | Rock | ~249 s |
| El Precio de la Soledad | Banda | ~185 s |
| Grupo Marca Registrada - El Rescate | Sierreño | ~161 s |

---

### Song Folder Format

Each song in the source dataset follows the standard Clone Hero layout:

```
Song Name - Artist/
├── notes.chart       # Required — the rhythm chart
├── song.ini          # Required — metadata
├── song.ogg          # Audio (single mix) OR separated stems:
├── guitar.ogg        # Guitar stem
├── bass.ogg          # Bass stem
├── vocals.ogg        # Vocals stem
├── drums.ogg         # Drums stem
└── album.png         # Optional — cover art
```

**`song.ini` fields:**

```ini
[song]
name = Song Title
artist = Artist Name
charter = Charter Name
album = Album Name
year = 2022
genre = Rock
song_length = 249195    # milliseconds
diff_guitar = 3         # 0–6 difficulty (-1 = not charted)
diff_bass = 2
diff_drums = 4
diff_keys = -1
```

---

## File Formats

### `.chart` Format

Text-based format with INI-like sections. The Clone Hero chart format stores all timing and note data.

**Top-level sections:**

| Section | Description |
|---|---|
| `[Song]` | Metadata: name, artist, BPM, resolution, audio file paths |
| `[SyncTrack]` | Tempo map (BPM changes) and time signature changes |
| `[Events]` | Section markers and lyrics |
| `[ExpertSingle]` | Expert guitar/keys notes |
| `[HardSingle]` | Hard guitar/keys notes |
| `[MediumSingle]` | Medium guitar/keys notes |
| `[EasySingle]` | Easy guitar/keys notes |
| `[ExpertDoubleBass]` | Expert bass notes |
| `[ExpertDrums]` | Expert drum notes |

**`[Song]` block:**

```
[Song]
{
  Name = "Song Title"
  Artist = "Artist Name"
  Charter = "Charter Name"
  Resolution = 192        # Ticks per quarter note
  Offset = 0              # Audio sync offset (seconds)
  GuitarStream = "guitar.ogg"
  BassStream = "bass.ogg"
  MusicStream = "song.ogg"
}
```

**`[SyncTrack]` block — timing events at tick positions:**

```
[SyncTrack]
{
  0 = TS 4          # Time Signature: 4/4 (numerator only; denominator = 4)
  0 = B 120000      # Tempo: BPM × 1000 → 120.000 BPM
  768 = B 116753    # Tempo change at tick 768 → 116.753 BPM
  768 = TS 3        # Time signature change to 3/4
}
```

**Tick ↔ Time conversion:**

```
time_seconds = (tick / Resolution) × (60 / BPM) + Offset
tick         = ((time_seconds − Offset) / (60 / BPM)) × Resolution
```

**Note tracks — `[ExpertSingle]`, `[ExpertDrums]`, etc.:**

```
[ExpertSingle]
{
  1536 = N 2 0      # Note at tick 1536, pitch 2, sustain 0 ticks
  1920 = N 0 192    # Note at tick 1920, pitch 0, held for 192 ticks
  3456 = S 2 0      # Special note (Star Power) at tick 3456
}
```

**Guitar/Bass pitch values:**

| Value | Color |
|---|---|
| 0 | Green |
| 1 | Red |
| 2 | Yellow |
| 3 | Blue |
| 4 | Orange |
| 5–7 | Extended colors |

**Drums pitch values:**

| Value | Drum |
|---|---|
| 0 | Kick |
| 1 | Red (snare) |
| 2 | Yellow (hi-hat) |
| 3 | Blue (tom) |
| 4 | Orange (cymbal) |

---

### `.mid` Format

Standard MIDI binary format, used as an alternative chart representation. Handled by `data/midi_loader.py` (`MIDIProcessor` class).

| Parameter | Default |
|---|---|
| `ticks_per_beat` | 480 |
| MIDI type | 0, 1, or 2 |
| Pitch range | 0–127 (MIDI standard) |

---

## Tokenizer

The `CloneHeroTokenizer` (in `models/tokenizer.py`) converts a parsed chart into a sequence of tokens suitable for a language model. It extends HuggingFace's `PreTrainedTokenizer`.

**Vocabulary:**

| Token group | Tokens |
|---|---|
| Special | `<BOS>`, `<EOS>`, `<UNK>`, `<PAD>` |
| Instruments | `<Guitar>`, `<Bass>`, `<Drums>` |
| Difficulties | `<Expert>`, `<Hard>`, `<Medium>`, `<Easy>` |
| Timing — minute | `<Minute_0>` … `<Minute_120>` |
| Timing — beat | `<Beat_0>` … `<Beat_512>` |
| Timing — beatshift | `<Beatshift_0>` … `<Beatshift_32>` |
| Guitar/Bass pitches | `<Pitch_0>` … `<Pitch_7>` |
| Drums pitches | `<DrumsPitch_0>` … `<DrumsPitch_4>` |

**Configuration:**

```python
max_duration_minutes = 120   # Maximum song duration
max_beats            = 512   # Maximum beats per segment
max_beatshifts       = 32    # Sub-beat timing granularity
```

The vocabulary and formulas are documented in [`tokenizer.txt`](tokenizer.txt).

---

## Processing Pipeline

```
Clone Hero song folders
        │
        ▼
parse_all_songs.py          → chart_analysis_results.json
        │
        ▼
tokenize_chart_dataset.py
  ├── CloneHeroChartParser   → parse .chart file
  ├── AudioProcessor         → load audio stems
  ├── DemucsAudioSeparator   → separate audio (drums/bass/vocals/other)
  ├── MERT                   → extract audio embeddings
  └── CloneHeroTokenizer     → tokenize chart sequence
        │
        ▼
Arrow shards (clone_hero_dataset_final/)
        │
        ▼
merge_dataset.py             → merged dataset → HF Hub
```

Processing is **resumable**: `checkpoints/checkpoint_metadata.json` tracks which files have been processed, the current index, total samples, and a version number.

---

## Key Scripts

| Script | Purpose |
|---|---|
| `tokenize_chart_dataset.py` | Main pipeline — builds the Arrow dataset |
| `parse_all_songs.py` | Scans source folders, outputs `chart_analysis_results.json` |
| `merge_dataset.py` | Merges multiple dataset shards; uploads to HF Hub |
| `migrate_pickle.py` | Converts legacy pickle files to Arrow format |
| `data_anal.py` | Analyzes dataset statistics; generates HTML visualizations |
| `duration_anal.py` | Plots note duration histogram |

---

## Models & Utilities

| File | Class | Description |
|---|---|---|
| `models/tokenizer.py` | `CloneHeroTokenizer` | Chart tokenizer |
| `models/demucs.py` | `DemucsAudioSeparator` | Stem separation via Facebook Demucs |
| `models/mert.py` | `MERT` | Audio embeddings via MERT-v1-95M |
| `data/chart_loader.py` | `CloneHeroChartParser` | `.chart` file parser |
| `data/audio_loaders.py` | `AudioProcessor` | Audio loading and resampling |
| `data/midi_loader.py` | `MIDIProcessor` | MIDI file creation and loading |
| `utils/time_utils.py` | — | Tick/BPM/second conversions |
| `utils/audio_utils.py` | — | Audio data interpolation for NN input |

---

## Technologies

- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — Arrow dataset storage
- [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) — Music audio representations
- [Demucs](https://github.com/facebookresearch/demucs) — Audio stem separation
- [PyTorch](https://pytorch.org/) & [Transformers](https://huggingface.co/docs/transformers) — Deep learning
- [LibROSA](https://librosa.org/) / [PyDub](https://github.com/jiaaro/pydub) / [SoundFile](https://python-soundfile.readthedocs.io/) — Audio processing
- [Mido](https://mido.readthedocs.io/) — MIDI file handling
- [Plotly](https://plotly.com/python/) — Data visualization
