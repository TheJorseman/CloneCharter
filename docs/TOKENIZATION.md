# Tokenization — Auto-Charter

This document explains how Clone Hero chart data is converted into discrete token sequences
that can be used to train an autoregressive transformer. All examples are drawn from the real
test-dataset songs.

---

## Table of Contents

1. [Why Tokenize?](#1-why-tokenize)
2. [Quantization Grid](#2-quantization-grid)
3. [Vocabulary Layout](#3-vocabulary-layout)
4. [Encoding a Track](#4-encoding-a-track)
   - 4.1 [Guitar / Bass — Step by Step](#41-guitar--bass--step-by-step)
   - 4.2 [Drums — No Sustain](#42-drums--no-sustain)
   - 4.3 [Chords as Bitmasks](#43-chords-as-bitmasks)
   - 4.4 [Star Power and Solo Events](#44-star-power-and-solo-events)
5. [Beat-Level Audio Conditioning](#5-beat-level-audio-conditioning)
   - 5.1 [Beat Boundaries in the Token Stream](#51-beat-boundaries-in-the-token-stream)
   - 5.2 [MERT Embeddings](#52-mert-embeddings)
   - 5.3 [Log-Mel Spectrogram per Beat](#53-log-mel-spectrogram-per-beat)
6. [Full Example: El Precio de la Soledad (Guitar)](#6-full-example-el-precio-de-la-soledad-guitar)
7. [Full Example: Caos La Planta (Drums)](#7-full-example-caos-la-planta-drums)
8. [Full Example: El Rescate (Bass)](#8-full-example-el-rescate-bass)
9. [Dataset Row Structure](#9-dataset-row-structure)
10. [Round-Trip Verification](#10-round-trip-verification)
11. [TODO — Transformer Architecture](#11-todo--transformer-architecture)

---

## 1. Why Tokenize?

A transformer requires a **discrete, fixed-vocabulary** input sequence. Chart data is stored
as tick-timestamped events:

```
480 = N 2 224        ← tick 480, Yellow note, 224-tick sustain
512 = N 3 128        ← tick 512, Blue note, 128-tick sustain
```

This format is ideal for human inspection but cannot feed a neural network directly.
We need to convert it into a flat integer sequence like:

```
[1, 6, 11, 183, 11, 60, 123, 91, 60, 123, 91, ...]
```

Each integer is a **token ID** with a specific meaning. The model learns to predict the
next token given all previous tokens and the audio context.

**Why not a piano-roll (grid) representation?**

A fixed-resolution grid at 16 ticks/cell for a 249-second song at ~115 BPM = ~22,900 frames.
That far exceeds typical transformer context windows (4 096 – 8 192 tokens). Our event-based
representation produces 2 000 – 12 000 tokens per song, stays within context, and handles
chords compactly as a single token.

---

## 2. Quantization Grid

All tick positions are rounded to the nearest **16-tick grid step**.

With `resolution = 192` ticks per quarter note:

```
  Subdivision        Ticks    Steps   Musical meaning
  ─────────────────────────────────────────────────────
  Quarter note        192      12     1 beat
  Eighth note          96       6     1/2 beat
  16th note            48       3     1/4 beat
  8th triplet          64       4     1/3 beat
  16th triplet  ▶▶▶    32       2     1/6 beat  ← critical for sierreño
  32nd note            24    1.5→2    fast runs
  Grid step            16       1     minimum unit
```

The **16th triplet (32 ticks = 2 steps)** is the tightest non-trivial subdivision
that appears in the dataset (El Precio de la Soledad, El Rescate). A coarser grid
of 48 ticks (1/16 note) would merge two different notes at 32 and 64 ticks into the
same position.

```
Tick timeline (resolution = 192):

  |←—————————— 1 beat (192 ticks) ——————————→|

  t=0    48    96   128   144  192   240   288
  ┼──────┼─────┼─────┼─────┼───┼─────┼─────┼  (real ticks)
  
  On 16-tick grid:
  ┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼  (12 steps per beat)
  0  1  2  3  4  5  6  7  8  9  10 11 12

  Common note positions:
  step 0  = tick 0   (beat 1)
  step 2  = tick 32  (16th triplet)
  step 3  = tick 48  (16th note)
  step 4  = tick 64  (8th triplet)
  step 6  = tick 96  (8th note)
  step 12 = tick 192 (quarter note / next beat)
```

---

## 3. Vocabulary Layout

The full vocabulary has **187 tokens** (IDs 0–186):

```
 Token ID   Token               Description
 ─────────────────────────────────────────────────────────────────────
   0        PAD                 Batch padding
   1        BOS                 Beginning of sequence
   2        EOS                 End of sequence
   3        UNK                 Unknown / reserved
   4        BEAT_BOUNDARY       Beat onset — audio cross-attention anchor
   5        MEASURE_START       Downbeat marker (optional)

   6        INSTR:guitar        Guitar track
   7        INSTR:bass          Bass track
   8        INSTR:drums         Drums track

   9–56     WAIT_k (k=1..48)   Advance cursor by k × 16 ticks
              WAIT_1  =  16t    (1/12 beat)
              WAIT_3  =  48t    (16th note)
              WAIT_6  =  96t    (8th note)
              WAIT_12 = 192t    (quarter note)
              WAIT_48 = 768t    (full 4/4 measure)

  57–87     NOTE:*  (31 IDs)   Guitar/bass chord bitmask over lanes G R Y B O
              57 = NOTE:G       Green only     (bitmask 00001)
              58 = NOTE:R       Red only       (bitmask 00010)
              60 = NOTE:Y       Yellow only    (bitmask 00100)
              64 = NOTE:B       Blue only      (bitmask 01000)
              72 = NOTE:O       Orange only    (bitmask 10000)
              87 = NOTE:GRYBO  All five lanes (bitmask 11111)

  88        MOD:HOPO            Hammer-on / pull-off modifier
  89        MOD:TAP             Tap note modifier
  90        MOD:OPEN            Open bass note (no fret)
  91        MOD:STRUM           Force strum modifier

  92–122    DRUM:*  (31 IDs)   Drum chord bitmask over K S H T C
              92 = DRUM:K       Kick only
              93 = DRUM:S       Snare only
              94 = DRUM:KS      Kick + Snare
             107 = DRUM:C       Cymbal only    (bitmask 10000)

 123–182    SUS_n  (60 IDs)   Sustain duration (guitar/bass only)
            Linear  [123–170]: step 0–47  → 0–752 ticks  (× 16)
              123 = SUS_0(0t)   staccato
              125 = SUS_2(32t)  16th triplet sustain
              129 = SUS_6(96t)  8th note sustain
              135 = SUS_12(192t) quarter note sustain
              137 = SUS_14(224t) dotted 8th sustain
            Coarse  [171–182]: steps for long sustains → 864–2640 ticks

 183        SP_ON               Star Power phrase begins
 184        SP_OFF              Star Power phrase ends
 185        SOLO_ON             Solo section begins
 186        SOLO_OFF            Solo section ends
```

**Bitmask encoding** for guitar/bass chords:

```
Lane:   O  B  Y  R  G
Bit:    4  3  2  1  0

Examples:
  Yellow only       → 0b00100 = 4  → token ID 57 + 4 - 1 = 60  (NOTE:Y)
  Blue only         → 0b01000 = 8  → token ID 57 + 8 - 1 = 64  (NOTE:B)
  Green + Red       → 0b00011 = 3  → token ID 57 + 3 - 1 = 59  (NOTE:GR)
  Red + Yellow      → 0b00110 = 6  → token ID 57 + 6 - 1 = 62  (NOTE:RY)
  All five (chord)  → 0b11111 = 31 → token ID 57 + 31 - 1 = 87 (NOTE:GRYBO)
```

---

## 4. Encoding a Track

### 4.1 Guitar / Bass — Step by Step

The encoder walks through a unified event timeline sorted by **(tick, priority)**:

```
Priority order at equal tick:
  0 = BEAT_BOUNDARY   ← always first (audio anchor)
  1 = SP_ON / SP_OFF
  2 = SOLO_ON / SOLO_OFF
  3 = NOTE events
```

For each note the emitted token sub-sequence is:

```
┌──────────────────────────────────────────────────────────────┐
│  [WAIT_k ...]  [BEAT?]  [NOTE_bitmask]  [SUS_n]  [MOD_* ...] │
└──────────────────────────────────────────────────────────────┘
     advance       anchor   which lanes    how long   modifier?
```

**Concrete example** — Yellow note at tick 240, sustain 0, force strum:

```
chart line:    240 = N 2 0
               240 = N 6 0       ← pitch 6 = force strum modifier

Step 1 — cursor is at tick 192 (just after a beat boundary)
          gap = 240 - 192 = 48 ticks = 3 steps → emit WAIT_3

Step 2 — emit chord token
          pitches = {2} (Yellow lane, bit 2) → bitmask = 0b00100 = 4
          chord_id = 57 + 4 - 1 = 60 → NOTE:Y

Step 3 — emit sustain token
          sustain = 0 ticks = 0 steps → SUS_0 (ID 123)

Step 4 — emit modifier (pitch 6 present)
          → MOD:STRUM (ID 91)

Token output:  11  60  123  91
               ↑   ↑    ↑    ↑
            WAIT_3 Y   SUS0 STRUM
```

---

### 4.2 Drums — No Sustain

Drums in Clone Hero always have `sustain = 0`. The `SUS` token is **never emitted** for drums,
making drum sequences ~30% shorter than equivalent guitar sequences.

```
chart line:    1408 = N 4 0      ← Cymbal hit

Drum encoding:

  [WAIT_k ...]  [DRUM_bitmask]
       ↑               ↑
   advance time    which drums
   (no SUS, no MOD)

Cymbal only → bit 4 set → bitmask = 0b10000 = 16 → drum_id = 92 + 16 - 1 = 107 (DRUM:C)
```

Simultaneous kick + snare (common in rock):

```
  Kick  = bit 0 (value 1)
  Snare = bit 1 (value 2)
  bitmask = 0b00011 = 3 → drum_id = 92 + 3 - 1 = 94 (DRUM:KS)
```

**Drum hit pattern diagram:**

```
Time →       beat 1        beat 2        beat 3        beat 4
             ↓             ↓             ↓             ↓
Kick    ──── K ──────────── K ────────────────── K ─────── K
Snare   ──────────── S ──────────── S ──────────── S ──────
HiHat   ── H ── H ── H ── H ── H ── H ── H ── H ── H ── H

Token stream (simplified):
  BEAT  DRUM:KH  WAIT_3  DRUM:H  WAIT_3  DRUM:SH  WAIT_3  DRUM:H  BEAT  ...
```

---

### 4.3 Chords as Bitmasks

When multiple notes share the same tick, they are merged into **one chord token**:

```
chart lines:   1536 = N 1 0    ← Red
               1536 = N 2 0    ← Yellow
               1536 = N 3 0    ← Blue

Lane bitmask:
  R = bit 1 → 0b00010
  Y = bit 2 → 0b00100
  B = bit 3 → 0b01000
  combined   → 0b01110 = 14

chord_id = 57 + 14 - 1 = 70  (NOTE:RYB)

Without bitmask encoding: 3 tokens + 3 sustains = 6 tokens
With bitmask encoding:    1 token  + 1 sustain  = 2 tokens  ← 3× more compact
```

All 31 non-empty lane combinations are pre-assigned token IDs:

```
 Single notes:  G(57)  R(58)  Y(60)  B(64)  O(72)
 Two-note:     GR(59) GY(61) GB(65) GO(73) RY(62) RB(66) RO(74) YB(68) YO(76) BO(80)
 Three-note:   GRY(63) GRB(67) ...  (10 combinations)
 Four-note:    GRYB(79) GRYO(87-related) ...  (5 combinations)
 Full chord:   GRYBO(87)
```

---

### 4.4 Star Power and Solo Events

Star power phrases (`S 2 length`) and solos (`E solo` / `E soloend`) produce paired tokens:

```
chart:
  144 = S 2 1680     ← star power starts at tick 144, lasts 1680 ticks
  192 = E solo       ← solo section starts at tick 192
  ...
  4320 = E soloend   ← solo ends

Token stream (beat boundaries omitted for clarity):

  tick  144: SP_ON  (183)
  tick  192: SOLO_ON (185)
  tick  240: NOTE:Y  SUS_0  MOD:STRUM
  ...
  tick 1824: SP_OFF (184)     ← tick 144 + 1680
  ...
  tick 4320: SOLO_OFF (186)

Visual timeline:
  144        192                                1824      4320
  ↓          ↓                                 ↓         ↓
  SP_ON    SOLO_ON  [notes ...]               SP_OFF   SOLO_OFF
  ├──────────────────────────────────────────┤
  │          Star Power phrase               │
  │          ├─────────────────────────────────────────────────┤
  │          Solo section                                      │
```

---

## 5. Beat-Level Audio Conditioning

The model is conditioned on audio features aligned to **quarter-note beats**.
This is why `BEAT_BOUNDARY` tokens appear in the sequence — they are the synchronization
points where the decoder attends to the audio of that beat.

### 5.1 Beat Boundaries in the Token Stream

```
Beat grid for El Precio de la Soledad (BPM = 130, resolution = 192):

  Beat  Tick   Time (s)   Token stream position
  ─────────────────────────────────────────────────────────────
    0      0   0.000 s    [BOS] [INSTR:guitar] [BEAT] ...
    1    192   0.231 s    ... [BEAT] [WAIT_9] [SP_ON] [WAIT_3] [BEAT] ...
    2    384   0.462 s    ...
    ...

Token stream with beats included:

  [BOS] [INSTR:guitar]
  [BEAT]                          ← beat 0 (tick 0, t=0.000s)
    [WAIT_9] [SP_ON]
  [BEAT]                          ← beat 1 (tick 192, t=0.231s)
    [SOLO_ON]
    [WAIT_3] [NOTE:Y] [SUS_0] [MOD:STRUM]
    [WAIT_3] [NOTE:Y] [SUS_0] [MOD:STRUM]
    [WAIT_6]
  [BEAT]                          ← beat 2 (tick 384, t=0.462s)
    [WAIT_2] [NOTE:B] [SUS_0] [MOD:STRUM]
    ...
  [EOS]
```

Each `BEAT_BOUNDARY` token is an **implicit pointer** to the audio feature vector for that beat.
During training, the model cross-attends to `audio_context[beat_index]` when generating
tokens after each `BEAT_BOUNDARY`.

### 5.2 MERT Embeddings

MERT (Music Encoder Representations from Transformers, `m-a-p/MERT-v1-95M`) outputs
hidden states at ~75 frames/second. These are **mean-pooled over each beat window**:

```
Audio waveform (guitar.ogg):
  ──────────────────────────────────────────────────────────→ time

MERT hidden states [T_frames × 768]:
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │f0│f1│f2│f3│f4│f5│f6│f7│f8│f9│..│  │  │  │  │  │  │  │
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
  each frame: 768-dim vector, ~13ms apart

Beat windows (at 130 BPM, 1 beat = 0.462s ≈ 35 frames):
  ┌───────────────────────┐ ┌─────────────────────┐ ┌──────
  │  beat 0 (35 frames)   │ │  beat 1 (35 frames) │ │ ...
  └───────────────────────┘ └─────────────────────┘ └──────
           ↓ mean pool             ↓ mean pool
  ┌────────────────────────┐ ┌────────────────────┐
  │  mert[0]  [768-dim]   │ │  mert[1] [768-dim] │ ...
  └────────────────────────┘ └────────────────────┘

Output shape: [num_beats, 768]
For El Precio (396 beats): shape = [396, 768]
```

### 5.3 Log-Mel Spectrogram per Beat

Each beat window is extracted from the log-mel spectrogram and **resampled to 32 frames**,
regardless of BPM. This normalizes the shape across variable-tempo songs.

```
Log-mel spectrogram [T_frames × 128]:
  Freq │████████████████████████████████████████████████│
  (mel)│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
       │                                                │
       └────────────────────────────────────────────────→ time

Beat window extraction (variable width due to BPM):

  At 80 BPM (0.75s/beat):           At 200 BPM (0.30s/beat):
  ┌──────────────────────────┐       ┌──────────┐
  │  ~75 mel frames          │       │ ~30 frames│
  └──────────────────────────┘       └──────────┘
           ↓ resample to 32                ↓ resample to 32
  ┌──────────────────────────┐       ┌──────────────────────────┐
  │  32 frames × 128 mels   │       │  32 frames × 128 mels   │
  └──────────────────────────┘       └──────────────────────────┘

Output shape: [num_beats, 32, 128]
For El Rescate (180 beats): shape = [180, 32, 128]

Combined audio context per beat:
  mert[b]   = [768]        (global timbral/harmonic embedding)
  logmel[b] = [32, 128]    (fine temporal structure)
  concat    → [768 + 4096] → project to d_model
```

---

## 6. Full Example: El Precio de la Soledad (Guitar)

**Song metadata:** Alfredo Olivas · Banda · 130 BPM (constant) · 184.7s · 640 notes · 396 beats

```
Full notes.chart snippet:
  [SyncTrack] { 0 = TS 4 · 0 = B 130000 }

  [ExpertSingle]
  {
    144  = S 2 1680     ← star power phrase
    192  = E solo       ← solo starts
    240  = N 2 0        ← Yellow, staccato, force strum (N 6 0 same tick)
    240  = N 6 0
    288  = N 2 0
    288  = N 6 0
    416  = N 3 0        ← Blue  (tick 416 = step 26 = 2 steps past 8th note)
    480  = N 2 224      ← Yellow, 224-tick sustain
    480  = N 6 0
    ...
  }
```

**Token stream (first 8 beats, beat boundaries shown):**

```
 Pos   ID   Token              Tick  Action
 ───────────────────────────────────────────────────────────────────────
   0    1   BOS
   1    6   INSTR:guitar
   2    4   BEAT               0     beat 0 begins
   3   17   WAIT_9(144t)       144   advance to tick 144
   4  183   SP_ON              144   star power phrase opens
   5   11   WAIT_3(48t)        192   advance to tick 192
   6    4   BEAT               192   beat 1 begins  ← 130 BPM, 0.462s in
   7  185   SOLO_ON            192   solo opens
   8   11   WAIT_3(48t)        240   advance to tick 240
   9   60   NOTE:Y             240   Yellow note
  10  123   SUS_0(0t)          240   staccato
  11   91   MOD:STRUM          240   force strum
  12   11   WAIT_3(48t)        288   advance to tick 288
  13   60   NOTE:Y             288   Yellow note
  14  123   SUS_0(0t)          288   staccato
  15   91   MOD:STRUM          288
  16   14   WAIT_6(96t)        384   advance past 8th note gap
  17    4   BEAT               384   beat 2 begins
  18   10   WAIT_2(32t)        416   advance to tick 416  (32-tick = 16th triplet!)
  19   64   NOTE:B             416   Blue note
  20  123   SUS_0(0t)          416   staccato
  21   91   MOD:STRUM
  22   12   WAIT_4(64t)        480   advance 64 ticks (8th triplet)
  23   60   NOTE:Y             480   Yellow note
  24  137   SUS_14(224t)       480   224-tick sustain (held note)
  25   91   MOD:STRUM
  ...

Stats: 3 231 tokens · avg 5.0 tokens/note · vocab utilization = 37 distinct token IDs
```

**Token distribution breakdown:**

```
Category         Count    % of sequence
─────────────────────────────────────────
WAIT tokens       909       28.1%
NOTE tokens       640       19.8%   ← 1:1 with original notes
SUS tokens        640       19.8%   ← always paired with NOTE
MOD:STRUM         594       18.4%   ← most notes have force-strum flag
BEAT_BOUNDARY     396       12.3%
SP_ON/OFF          24        0.7%
SOLO_ON/OFF         8        0.2%
BOS + EOS + INSTR   3        0.1%
                 ─────      ─────
Total            3231      100%
```

---

## 7. Full Example: Caos La Planta (Drums)

**Song metadata:** Caos · Rock · variable BPM 82–153 · 249.2s · 987 drum hits · 473 beats

The drums track has no sustains, making sequences lean:

```
[ExpertDrums] snippet:
  1408 = N 4 0     ← Cymbal (C)
  1440 = N 4 0     ← Cymbal
  1472 = N 4 0     ← Cymbal
  1504 = N 4 0
  1536 = N 4 0
  1584 = N 4 0
  ...
  2976 = N 3 0     ← Tom (T)
  3024 = N 3 0
```

**Token stream (first 20 hits, beats omitted):**

```
 Pos  ID   Token           Ticks advanced   Description
 ─────────────────────────────────────────────────────────────
   0   1   BOS
   1   8   INSTR:drums
   2   4   BEAT                              beat 0
   3–16    [WAIT_12 × 7]   1344t            7 empty beats (intro)
  17   12  WAIT_4(64t)     +64t = 1408t     advance to first hit
  18  107  DRUM:C          1408t            Cymbal hit
  19   10  WAIT_2(32t)     +32t = 1440t     32-tick gap (16th triplet)
  20  107  DRUM:C          1440t
  21   10  WAIT_2(32t)
  22  107  DRUM:C          1472t
  23   10  WAIT_2(32t)
  24  107  DRUM:C          1504t
  25   10  WAIT_2(32t)
  26    4  BEAT                              beat in middle of run
  27  107  DRUM:C          1536t
  28   11  WAIT_3(48t)     +48t = 1584t     48-tick gap (16th note)
  29  107  DRUM:C          1584t
  ...
```

Drum pattern diagram (first 4 beats shown, tick 0 = song start):

```
Tick:     0                  192                384               576
          ↓                   ↓                  ↓                 ↓
Beat:   [beat 0]           [beat 1]           [beat 2]          [beat 3]
          │                   │                  │                 │
Cymbal:   .    .    .    .    .    .    .    .   C  C  C  C  C  C  C
Tom:      .    .    .    .    .    .    .    .   .  .  .  .  T  T  T

Encoded:  [BEAT][WAIT×7][BEAT][WAIT_4][C][WAIT_2][C][WAIT_2][C]...
```

**Stats:** 2 571 tokens · avg 2.6 tokens/hit (WAIT + DRUM, no SUS/MOD) · 38% shorter than guitar

---

## 8. Full Example: El Rescate (Bass)

**Song metadata:** Grupo Marca Registrada · Sierreño · variable BPM 55–117 · 160.6s · 231 notes · 180 beats

Bass has dedicated `bass.ogg` stem → the model gets a clean, separated signal.

```
[ExpertDoubleBass] snippet:
  1536 = N 3 96     ← Blue, 96-tick sustain
  1728 = N 1 75     ← Red,  75-tick sustain  (75t ≈ 80t = step 5)
  1824 = N 3 75     ← Blue
  1920 = N 2 75     ← Yellow
  2016 = N 1 75     ← Red
  2112 = N 0 75     ← Green
  2208 = N 3 96     ← Blue,  96-tick sustain
  2496 = N 1 144    ← Red,  144-tick sustain (9-beat rest before next phrase)
```

**Token stream (first 12 notes, beats included):**

```
 Pos  ID   Token            Tick   Note
 ──────────────────────────────────────────────────────────────────
   0   1   BOS
   1   7   INSTR:bass
   2   4   BEAT             0      beat 0
  [WAIT_12 × 8]             1536   8 empty beats (sierreño intro)
  17    4  BEAT             1536   beat 8  (8×192 = 1536)
  18   64  NOTE:B           1536   Blue note
  19  129  SUS_6(96t)       ─      sustain = 96t (1/2 beat)
  20    4  BEAT             1728   beat 9
  21   20  WAIT_12(192t)           (note starts on beat — no extra wait)
  22   58  NOTE:R           1728   Red note
  23  128  SUS_5(80t)       ─      sustain ≈ 75t → quantized to 80t (step 5)
  24   14  WAIT_6(96t)      1824   advance 96t
  25   64  NOTE:B           1824   Blue
  26  128  SUS_5(80t)
  27   14  WAIT_6(96t)      1920
  28    4  BEAT             1920   beat 10
  29   60  NOTE:Y           1920   Yellow
  30  128  SUS_5(80t)
  31   14  WAIT_6(96t)      2016
  32   58  NOTE:R           2016   Red
  33  128  SUS_5(80t)
  34   14  WAIT_6(96t)      2112
  35    4  BEAT             2112   beat 11
  36   57  NOTE:G           2112   Green
  37  128  SUS_5(80t)
  38   14  WAIT_6(96t)      2208
  39   64  NOTE:B           2208   Blue, 96-tick sustain
  40  129  SUS_6(96t)
  41   14  WAIT_6(96t)
  42    4  BEAT             2304   beat 12
  43   20  WAIT_12(192t)    2496   1 quarter-note rest
  44   58  NOTE:R           2496   Red, 144-tick sustain
  45  132  SUS_9(144t)
  ...
```

**Pattern visualization** — the walking bass line across 4 beats:

```
Beat:    8         9         10        11        12
Tick:   1536      1728      1920      2112      2304
         ↓         ↓         ↓         ↓         ↓
Note:    B ─96─   R─80─ B─80─ Y─80─ R─80─ G─80─ B─96─   R──144──

Lane:    ●         ●    ●    ●    ●    ●    ●         ●
         B    .    R    B    Y    R    G    B    .    R
         ───────────────────────────── sierreño walking pattern ───
```

---

## 9. Dataset Row Structure

Each row in the HuggingFace dataset represents one instrument from one song:

```
{
  # Identity
  "song_id":               "a3f9b1c2d0e4...",     # MD5 hash of "artist|title"
  "instrument":            "guitar",               # "guitar" | "bass" | "drums"
  "source_format":         "chart",                # "chart" | "midi"

  # Target sequence
  "tokens":                [1, 6, 4, 17, 183, ...],   # token IDs, len ~3000
  "num_tokens":            3231,
  "num_beats":             396,

  # Audio conditioning (per beat)
  "mert_embeddings":       [[0.12, -0.04, ...], ...],  # [396, 768]  float32
  "logmel_frames":         [[[...], ...], ...],         # [396, 32, 128]  float32

  # Timing (for position encoding)
  "beat_times_s":          [0.000, 0.462, 0.923, ...], # [396] float32
  "beat_durations_s":      [0.462, 0.462, 0.462, ...], # [396] float32  (constant at 130 BPM)
  "bpm_at_beat":           [130.0, 130.0, ...],         # [396] float32
  "time_sig_num_at_beat":  [4, 4, 4, ...],              # [396] int32

  # Metadata
  "song_name":             "El Precio de la Soledad",
  "artist":                "Alfredo Olivas",
  "genre":                 "banda",
  "year":                  2012,
  "song_length_ms":        184668,
  "difficulty":            2,

  # Flags
  "has_star_power":        true,
  "has_solo":              true,
  "has_dedicated_stem":    false,   # no guitar.ogg, uses song.ogg

  # Stats (for curriculum learning / filtering)
  "num_notes":             640,
  "notes_per_beat_mean":   1.62,
  "chord_ratio":           0.0,     # no chords in this track
  "bpm_mean":              130.0,
  "bpm_std":               0.0,     # constant BPM
}
```

**Batch collation** pads variable-length sequences per batch:

```
Batch of 3 songs with different beat counts:
  song A: 396 beats  →  mert_A: [396, 768]
  song B: 180 beats  →  mert_B: [180, 768]
  song C: 473 beats  →  mert_C: [473, 768]

After collation (pad to max_beats = 473):
  mert_batch:       [3, 473, 768]    float32   (zeros for padding)
  beat_attn_mask:   [3, 473]         int32     (1=valid, 0=padded)
  input_ids:        [3, max_tokens]  int32     (right-padded with 0=PAD)
  attention_mask:   [3, max_tokens]  int32
```

---

## 10. Round-Trip Verification

The tokenizer is reversible: `encode → decode → re-encode` produces an identical sequence.
This guarantees that no information is lost during tokenization (beyond the intentional
quantization of sustain durations to the 16-tick grid).

```
Original chart note:
  480 = N 2 224     (Yellow, 224-tick sustain)

Encode:
  NOTE:Y (60)  SUS_14(224t) (137)  MOD:STRUM (91)

Decode:
  NoteEvent(tick=480, pitches={2}, sustain=224, is_force_strum=True)

Re-encode:
  NOTE:Y (60)  SUS_14(224t) (137)  MOD:STRUM (91)   ✓ identical

Run the verifier on all test songs:
  $ uv run validate-roundtrip test_dataset/
  PASS  Caos - La Planta / guitar
  PASS  Caos - La Planta / drums
  PASS  El Precio de la Soledad / guitar
  PASS  Grupo Marca Registrada - El Rescate / guitar
  PASS  Grupo Marca Registrada - El Rescate / bass
  PASS  ZOE - Asteroide / guitar         ← MIDI source, normalized to chart format
  Results: 6/6 passed, 0 failed
```

**Quantization error** (the only lossy step):

```
Original sustain:  75 ticks  (MIDI-converted, non-standard value)
Nearest step:       5 steps = 80 ticks
Error:              5 ticks  (< 1 grid step = < 3ms at 130 BPM)

Original sustain: 224 ticks
Nearest step:      14 steps = 224 ticks
Error:              0 ticks  (exact match)

Max possible error: 7 ticks (half a grid step = ~2ms at 130 BPM)
```

---

## 11. TODO — Transformer Architecture

> **Status:** Dataset pipeline is complete. The following describes the planned model
> architecture to train on the generated dataset.

### 11.1 Overall Architecture: Conditioned Encoder-Decoder

```
                    ┌───────────────────────────────────────────────────┐
                    │                  AUDIO ENCODER                    │
                    │                                                   │
  guitar.ogg ──────►│  MERT extractor  ──►  [num_beats, 768]          │
                    │       +                                           │
                    │  Log-Mel extractor ──► [num_beats, 32, 128]      │
                    │                                  │                │
                    │  Linear projection: concat → d_model             │
                    │                                  │                │
                    │  Beat Transformer (optional):  self-attention     │
                    │  over beat sequence                               │
                    │                                  │                │
                    │            audio_context: [num_beats, d_model]   │
                    └───────────────────────────────┬───────────────────┘
                                                    │ cross-attention
                    ┌───────────────────────────────▼───────────────────┐
                    │                 TOKEN DECODER                     │
                    │                                                   │
  [BOS][INSTR][...] ─►  Token Embedding [vocab=187, d_model]          │
                    │              +                                    │
                    │  Rotary / Sinusoidal Positional Encoding          │
                    │              │                                    │
                    │  N × Transformer Decoder Block:                   │
                    │    ├─ Causal Self-Attention                       │
                    │    ├─ Cross-Attention → audio_context             │
                    │    │    (each token attends to its beat index)    │
                    │    └─ Feed-Forward                                │
                    │              │                                    │
                    │  LM Head: Linear [d_model → 187]                 │
                    │              │                                    │
                    │  Softmax → next token probabilities               │
                    └───────────────────────────────────────────────────┘
```

### 11.2 Beat Index Alignment

Each token in the sequence must know **which beat it belongs to**, so the cross-attention
can retrieve the correct audio vector. This is computed from `BEAT_BOUNDARY` positions:

```
Token sequence:
  [BOS][INSTR][BEAT][WAIT_9][SP_ON][BEAT][SOLO_ON][NOTE:Y][SUS_0]...
                 ↑                    ↑
               beat=0               beat=1

Beat index mapping:
  pos  0: BOS       → beat_idx = 0
  pos  1: INSTR     → beat_idx = 0
  pos  2: BEAT      → beat_idx = 0   ← BEAT_BOUNDARY updates index to 0
  pos  3: WAIT_9    → beat_idx = 0
  pos  4: SP_ON     → beat_idx = 0
  pos  5: BEAT      → beat_idx = 1   ← BEAT_BOUNDARY updates index to 1
  pos  6: SOLO_ON   → beat_idx = 1
  pos  7: NOTE:Y    → beat_idx = 1
  pos  8: SUS_0     → beat_idx = 1
  ...

Cross-attention query: token_hidden[pos]
Cross-attention key/value: audio_context[beat_idx[pos]]
```

### 11.3 Recommended Hyperparameters (Starting Point)

```
Model dimension        d_model = 512
Decoder layers         N = 8
Attention heads        H = 8
FFN dimension          d_ff = 2048
Vocab size             187
Max context length     8 192 tokens   (covers ~2 full songs)
Audio encoder dim      768 + 4096 → 512  (linear projection per beat)
Beat transformer       2 layers, d=512  (optional, improves global harmony)

Training:
  Optimizer:           AdamW  (lr=1e-4, β=(0.9, 0.95), weight_decay=0.1)
  LR schedule:         Cosine with warmup (1 000 steps)
  Batch size:          8–16 songs (collated, padded)
  Loss:                Cross-entropy on next-token prediction
                       (mask PAD positions, optionally weight BEAT tokens lower)
  Curriculum:          Start with constant-BPM songs (bpm_std = 0),
                       then add variable-BPM as loss drops

Inference (generation):
  Strategy:            Autoregressive sampling with temperature = 0.8
  Constraint:          Grammar mask — after NOTE:* always force SUS_*;
                       after DRUM:* never allow SUS_*
  Stop condition:      EOS token or max_tokens reached
```

### 11.4 Simpler Alternative: GPT-style with Injected Audio

If a full encoder-decoder is too heavy, inject audio as **virtual tokens** at each beat:

```
Modified token stream:

  [BOS][INSTR][AUDIO_EMBED_0][BEAT][WAIT_9][SP_ON]
              [AUDIO_EMBED_1][BEAT][SOLO_ON][NOTE:Y]...

Where AUDIO_EMBED_b is the projected audio vector for beat b, inserted directly
into the sequence before each BEAT_BOUNDARY token. The model treats it as a
special "soft token" with a learned value but no vocabulary ID.

Pro:  Works with any standard GPT-2 / LLaMA decoder-only architecture.
Con:  Doubles the effective sequence length (one audio token per beat).
```

### 11.5 Data Pipeline for Training

```
Disk (HuggingFace Dataset)
    │
    ▼
datasets.load_from_disk("./out/train")
    │
    ▼
.map(tokenize)              ← tokens already stored, just load
    │
    ▼
.filter(lambda r:           ← curriculum: easy songs first
    r["bpm_std"] < 5 and
    r["num_tokens"] < 4096)
    │
    ▼
DataLoader(
    batch_size=8,
    collate_fn=AutoCharterCollator(max_tokens=8192),
    num_workers=4,
)
    │
    ▼
model.forward(
    input_ids,              # [B, T]        token IDs
    attention_mask,         # [B, T]        causal mask + padding
    mert_embeddings,        # [B, num_beats, 768]
    logmel_frames,          # [B, num_beats, 32, 128]
    beat_attention_mask,    # [B, num_beats] valid beats
)
    │
    ▼
loss = cross_entropy(logits, labels)   # next-token prediction
```

### 11.6 Evaluation Metrics

```
Metric                  Description
────────────────────────────────────────────────────────────────────────
Token accuracy          % of tokens predicted exactly (upper bound)
Note F1                 Compare decoded notes to ground truth at tick level
Pitch accuracy          % of correct pitch/lane assignments per beat
Sustain error (ms)      Mean absolute error on sustain durations
Star power overlap      IoU between predicted and true SP phrases
BPM-aligned accuracy    Note F1 weighted by BPM difficulty (inverse bpm_std)
Human play rate         % of generated charts that are physically playable
                        (no impossible chord stretches, no superhuman tempo)
```

### 11.7 Audio Separation Preprocessing

Before extracting MERT/log-mel features, songs without dedicated stems should be
source-separated using [Demucs](https://github.com/facebookresearch/demucs):

```
TODO: Add a demucs preprocessing step in audio/separator.py

  from demucs.pretrained import get_model
  from demucs.apply import apply_model

  model = get_model("htdemucs")   # 4-stem: drums, bass, other, vocals
  # "other" = guitar in most rock/banda songs

  Mapping:
    Demucs "drums" → use for drums track conditioning
    Demucs "bass"  → use for bass track conditioning
    Demucs "other" → use for guitar track conditioning (approximate)

  This enables extracting instrument-specific features even for songs
  that only ship a single stereo mix (song.ogg), which is the majority
  of the Clone Hero library.
```
