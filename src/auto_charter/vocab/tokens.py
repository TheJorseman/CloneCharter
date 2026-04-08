"""Token vocabulary constants for the auto-charter.

Vocabulary layout (187 tokens total):
  [0]       PAD
  [1]       BOS
  [2]       EOS
  [3]       UNK
  [4]       BEAT_BOUNDARY   — beat onset anchor for audio cross-attention
  [5]       MEASURE_START   — downbeat marker

  [6]       INSTR_GUITAR
  [7]       INSTR_BASS
  [8]       INSTR_DRUMS

  [9–56]    WAIT_k  (k = 1..48)  →  k × 16 ticks of time advance
  [57–87]   GUITAR/BASS chord bitmask (1..31 over Green/Red/Yellow/Blue/Orange lanes)
  [88]      MOD_HOPO
  [89]      MOD_TAP
  [90]      MOD_OPEN          — open bass (pitch 7 in .chart)
  [91]      MOD_FORCE_STRUM   — force strum (pitch 6 in .chart)

  [92–122]  DRUMS chord bitmask (1..31 over Kick/Snare/HiHat/Tom/Cymbal lanes)

  [123–182] SUS_n sustain tokens (60 entries)
            Linear region  [123–170]: steps 0..47  → 0..752 ticks  (step × 16)
            Coarse region  [171–182]: steps [54,60,66,72,78,84,90,96,110,120,144,165]
                                      → 864..2640 ticks

  [183]     STAR_POWER_ON
  [184]     STAR_POWER_OFF
  [185]     SOLO_ON
  [186]     SOLO_OFF
"""


class Vocab:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    BEAT_BOUNDARY = 4
    MEASURE_START = 5

    INSTR_GUITAR = 6
    INSTR_BASS = 7
    INSTR_DRUMS = 8

    # WAIT tokens: WAIT_START + (k-1) for k in 1..48
    WAIT_START = 9   # WAIT_1 = 9, WAIT_48 = 56
    WAIT_END = 56
    WAIT_MIN_K = 1
    WAIT_MAX_K = 48  # 48 × 16 = 768 ticks = 1 full 4/4 measure

    # Guitar/Bass chord tokens: bitmask 1..31 → IDs 57..87
    GUITAR_NOTE_START = 57
    GUITAR_NOTE_END = 87

    # Modifier tokens
    MOD_HOPO = 88
    MOD_TAP = 89
    MOD_OPEN = 90
    MOD_FORCE_STRUM = 91

    # Drum chord tokens: bitmask 1..31 → IDs 92..122
    DRUM_NOTE_START = 92
    DRUM_NOTE_END = 122

    # Sustain tokens
    SUS_START = 123
    SUS_END = 182

    # Event tokens
    STAR_POWER_ON = 183
    STAR_POWER_OFF = 184
    SOLO_ON = 185
    SOLO_OFF = 186

    SIZE = 187  # total vocabulary size

    # Sustain step values corresponding to SUS_START..SUS_END
    SUS_LINEAR_STEPS = list(range(48))                               # 0..47 → 0..752 ticks
    SUS_COARSE_STEPS = [54, 60, 66, 72, 78, 84, 90, 96, 110, 120, 144, 165]  # → 864..2640
    SUS_STEPS = SUS_LINEAR_STEPS + SUS_COARSE_STEPS  # 60 entries

    INSTRUMENT_TO_ID = {
        "guitar": INSTR_GUITAR,
        "bass": INSTR_BASS,
        "drums": INSTR_DRUMS,
    }
    ID_TO_INSTRUMENT = {v: k for k, v in INSTRUMENT_TO_ID.items()}

    @classmethod
    def wait_id(cls, k: int) -> int:
        """Token ID for WAIT_k (k must be in 1..48)."""
        assert cls.WAIT_MIN_K <= k <= cls.WAIT_MAX_K, f"WAIT_k out of range: {k}"
        return cls.WAIT_START + k - 1

    @classmethod
    def wait_k(cls, token_id: int) -> int:
        """Number of 16-tick steps encoded in a WAIT token."""
        return token_id - cls.WAIT_START + 1

    @classmethod
    def sus_id(cls, step_index: int) -> int:
        """Token ID for a sustain step index (0..59)."""
        return cls.SUS_START + step_index

    @classmethod
    def is_wait(cls, token_id: int) -> bool:
        return cls.WAIT_START <= token_id <= cls.WAIT_END

    @classmethod
    def is_guitar_note(cls, token_id: int) -> bool:
        return cls.GUITAR_NOTE_START <= token_id <= cls.GUITAR_NOTE_END

    @classmethod
    def is_drum_note(cls, token_id: int) -> bool:
        return cls.DRUM_NOTE_START <= token_id <= cls.DRUM_NOTE_END

    @classmethod
    def is_sus(cls, token_id: int) -> bool:
        return cls.SUS_START <= token_id <= cls.SUS_END

    @classmethod
    def is_modifier(cls, token_id: int) -> bool:
        return token_id in (cls.MOD_HOPO, cls.MOD_TAP, cls.MOD_OPEN, cls.MOD_FORCE_STRUM)

    @classmethod
    def token_name(cls, token_id: int) -> str:
        """Human-readable name for a token ID (for inspect_song script)."""
        if token_id == cls.PAD: return "PAD"
        if token_id == cls.BOS: return "BOS"
        if token_id == cls.EOS: return "EOS"
        if token_id == cls.UNK: return "UNK"
        if token_id == cls.BEAT_BOUNDARY: return "BEAT"
        if token_id == cls.MEASURE_START: return "MEASURE"
        if token_id == cls.INSTR_GUITAR: return "INSTR:guitar"
        if token_id == cls.INSTR_BASS: return "INSTR:bass"
        if token_id == cls.INSTR_DRUMS: return "INSTR:drums"
        if cls.is_wait(token_id):
            k = cls.wait_k(token_id)
            return f"WAIT_{k}({k * 16}t)"
        if cls.is_guitar_note(token_id):
            bitmask = token_id - cls.GUITAR_NOTE_START + 1
            lanes = ["G", "R", "Y", "B", "O"]
            active = "".join(l for i, l in enumerate(lanes) if bitmask & (1 << i))
            return f"NOTE:{active}"
        if token_id == cls.MOD_HOPO: return "MOD:HOPO"
        if token_id == cls.MOD_TAP: return "MOD:TAP"
        if token_id == cls.MOD_OPEN: return "MOD:OPEN"
        if token_id == cls.MOD_FORCE_STRUM: return "MOD:STRUM"
        if cls.is_drum_note(token_id):
            bitmask = token_id - cls.DRUM_NOTE_START + 1
            lanes = ["K", "S", "H", "T", "C"]
            active = "".join(l for i, l in enumerate(lanes) if bitmask & (1 << i))
            return f"DRUM:{active}"
        if cls.is_sus(token_id):
            idx = token_id - cls.SUS_START
            steps = cls.SUS_STEPS[idx]
            return f"SUS_{steps}({steps * 16}t)"
        if token_id == cls.STAR_POWER_ON: return "SP_ON"
        if token_id == cls.STAR_POWER_OFF: return "SP_OFF"
        if token_id == cls.SOLO_ON: return "SOLO_ON"
        if token_id == cls.SOLO_OFF: return "SOLO_OFF"
        return f"UNKNOWN({token_id})"
