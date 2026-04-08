"""Tests for the tokenizer: vocab, encoder, decoder, round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
EL_PRECIO = REPO_ROOT / "El Precio de la Soledad"
CAOS = REPO_ROOT / "Caos - La Planta"
EL_RESCATE = REPO_ROOT / "Grupo Marca Registrada - El Rescate"


def requires_fixture(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"Fixture not found: {path}")


# ─── Vocabulary tests ──────────────────────────────────────────────────────────

def test_vocab_size():
    from auto_charter.vocab.tokens import Vocab
    assert Vocab.SIZE == 187


def test_wait_ids():
    from auto_charter.vocab.tokens import Vocab
    assert Vocab.wait_id(1) == 9
    assert Vocab.wait_id(48) == 56
    assert Vocab.wait_k(9) == 1
    assert Vocab.wait_k(56) == 48


def test_guitar_note_ids():
    from auto_charter.vocab.guitar_vocab import chord_bitmask_to_id, id_to_chord_bitmask
    from auto_charter.vocab.tokens import Vocab

    # bitmask 1 (Green only) → ID 57
    assert chord_bitmask_to_id(1) == 57
    # bitmask 31 (all lanes) → ID 87
    assert chord_bitmask_to_id(31) == 87
    # round-trip
    for bitmask in range(1, 32):
        tid = chord_bitmask_to_id(bitmask)
        assert id_to_chord_bitmask(tid) == bitmask
        assert Vocab.is_guitar_note(tid)


def test_drum_note_ids():
    from auto_charter.vocab.drum_vocab import drum_bitmask_to_id, id_to_drum_bitmask
    from auto_charter.vocab.tokens import Vocab

    assert drum_bitmask_to_id(1) == 92
    assert drum_bitmask_to_id(31) == 122
    for bitmask in range(1, 32):
        tid = drum_bitmask_to_id(bitmask)
        assert id_to_drum_bitmask(tid) == bitmask
        assert Vocab.is_drum_note(tid)


def test_sus_steps_count():
    from auto_charter.vocab.tokens import Vocab
    assert len(Vocab.SUS_STEPS) == 60


def test_sus_ids():
    from auto_charter.vocab.tokens import Vocab
    assert Vocab.sus_id(0) == 123          # SUS_0 (staccato)
    assert Vocab.sus_id(59) == 182         # SUS_165 (max)


def test_quantize_staccato():
    from auto_charter.tokenizer.quantize import quantize_sustain, sustain_from_sus_index
    idx = quantize_sustain(0)
    assert idx == 0
    assert sustain_from_sus_index(0) == 0


def test_quantize_quarter_note():
    from auto_charter.tokenizer.quantize import quantize_sustain, sustain_from_sus_index
    # 192 ticks = quarter note = 12 steps of 16
    idx = quantize_sustain(192)
    reconstructed = sustain_from_sus_index(idx)
    assert abs(reconstructed - 192) < 16  # within 1 grid step


def test_token_names():
    from auto_charter.vocab.tokens import Vocab
    assert Vocab.token_name(0) == "PAD"
    assert Vocab.token_name(1) == "BOS"
    assert Vocab.token_name(2) == "EOS"
    assert Vocab.token_name(4) == "BEAT"
    assert "WAIT" in Vocab.token_name(9)
    assert "NOTE" in Vocab.token_name(57)
    assert "DRUM" in Vocab.token_name(92)
    assert "SUS" in Vocab.token_name(123)


# ─── Encoder / Decoder tests ──────────────────────────────────────────────────

@requires_fixture(EL_PRECIO)
def test_encoder_starts_with_bos_instr():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    chart = parse_chart(EL_PRECIO / "notes.chart")
    tokens = encode_track(chart, "guitar")
    assert tokens[0] == Vocab.BOS
    assert tokens[1] == Vocab.INSTR_GUITAR
    assert tokens[-1] == Vocab.EOS


@requires_fixture(EL_PRECIO)
def test_encoder_all_tokens_in_range():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    chart = parse_chart(EL_PRECIO / "notes.chart")
    tokens = encode_track(chart, "guitar")
    for tok in tokens:
        assert 0 <= tok < Vocab.SIZE, f"Token {tok} out of range"


@requires_fixture(CAOS)
def test_encoder_drums_no_sustain():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    chart = parse_chart(CAOS / "notes.chart")
    tokens = encode_track(chart, "drums")

    # After a drum note token, the next token should NOT be a SUS token
    for i in range(len(tokens) - 1):
        if Vocab.is_drum_note(tokens[i]):
            assert not Vocab.is_sus(tokens[i + 1]), \
                f"Drum note at {i} followed by SUS token"


@requires_fixture(EL_PRECIO)
def test_roundtrip_guitar(tmp_path):
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.tokenizer.decoder import decode_tokens

    chart = parse_chart(EL_PRECIO / "notes.chart")
    tokens1 = encode_track(chart, "guitar", include_beat_boundaries=False)
    decoded = decode_tokens(tokens1, resolution=chart.resolution)
    tokens2 = encode_track(decoded, "guitar", include_beat_boundaries=False)

    assert tokens1 == tokens2, (
        f"Round-trip failed: {len(tokens1)} original tokens vs {len(tokens2)} re-encoded. "
        f"First mismatch: {next((i for i, (a, b) in enumerate(zip(tokens1, tokens2)) if a != b), -1)}"
    )


@requires_fixture(EL_RESCATE)
def test_roundtrip_bass(tmp_path):
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.tokenizer.decoder import decode_tokens

    chart = parse_chart(EL_RESCATE / "notes.chart")
    tokens1 = encode_track(chart, "bass", include_beat_boundaries=False)
    decoded = decode_tokens(tokens1, resolution=chart.resolution)
    tokens2 = encode_track(decoded, "bass", include_beat_boundaries=False)

    assert tokens1 == tokens2


@requires_fixture(CAOS)
def test_roundtrip_drums(tmp_path):
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.tokenizer.decoder import decode_tokens

    chart = parse_chart(CAOS / "notes.chart")
    tokens1 = encode_track(chart, "drums", include_beat_boundaries=False)
    decoded = decode_tokens(tokens1, resolution=chart.resolution)
    tokens2 = encode_track(decoded, "drums", include_beat_boundaries=False)

    assert tokens1 == tokens2


@requires_fixture(EL_PRECIO)
def test_decoder_note_count():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.tokenizer.decoder import decode_tokens

    chart = parse_chart(EL_PRECIO / "notes.chart")
    original_count = len(chart.tracks["guitar"])
    tokens = encode_track(chart, "guitar", include_beat_boundaries=False)
    decoded = decode_tokens(tokens)
    decoded_count = len(decoded.tracks.get("guitar", []))
    assert original_count == decoded_count


@requires_fixture(EL_PRECIO)
def test_beat_boundaries_present():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    chart = parse_chart(EL_PRECIO / "notes.chart")
    tokens = encode_track(chart, "guitar", include_beat_boundaries=True)
    beat_count = tokens.count(Vocab.BEAT_BOUNDARY)
    assert beat_count > 0


@requires_fixture(EL_PRECIO)
def test_star_power_tokens():
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    chart = parse_chart(EL_PRECIO / "notes.chart")
    tokens = encode_track(chart, "guitar")
    sp_on = tokens.count(Vocab.STAR_POWER_ON)
    sp_off = tokens.count(Vocab.STAR_POWER_OFF)
    assert sp_on > 0
    assert sp_on == sp_off  # must be paired
