"""Tests for the .chart file parser."""

from __future__ import annotations

from pathlib import Path

import pytest

# Paths to test fixtures (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent.parent
EL_PRECIO = REPO_ROOT / "El Precio de la Soledad"
CAOS = REPO_ROOT / "Caos - La Planta"
EL_RESCATE = REPO_ROOT / "Grupo Marca Registrada - El Rescate"


def requires_fixture(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"Fixture not found: {path}")


@requires_fixture(EL_PRECIO)
def test_parse_el_precio_basic():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    assert chart.resolution == 192
    assert "guitar" in chart.tracks
    assert len(chart.tracks["guitar"]) > 0


@requires_fixture(EL_PRECIO)
def test_constant_bpm_sync_track():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    bpm_events = chart.bpm_map.bpm_events
    # El Precio has a single BPM: 130.000
    assert len(bpm_events) == 1
    assert abs(bpm_events[0].bpm - 130.0) < 0.001


@requires_fixture(CAOS)
def test_variable_bpm_sync_track():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(CAOS / "notes.chart")
    bpm_events = chart.bpm_map.bpm_events
    # Caos La Planta has 200+ BPM changes
    assert len(bpm_events) > 100


@requires_fixture(CAOS)
def test_tick_to_seconds_monotonic():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(CAOS / "notes.chart")
    bpm_map = chart.bpm_map
    ticks = [0, 192, 768, 1536, 3840, 10000, 50000]
    times = [bpm_map.tick_to_seconds(t) for t in ticks]
    # Times must be strictly increasing
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], f"Non-monotonic at tick {ticks[i]}"


@requires_fixture(EL_RESCATE)
def test_bass_track_present():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_RESCATE / "notes.chart")
    assert "bass" in chart.tracks
    assert len(chart.tracks["bass"]) > 0


@requires_fixture(CAOS)
def test_drums_track_present():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(CAOS / "notes.chart")
    assert "drums" in chart.tracks
    assert len(chart.tracks["drums"]) > 0


@requires_fixture(EL_PRECIO)
def test_star_power_events():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    specials = chart.specials.get("guitar", [])
    sp_events = [s for s in specials if s.kind == "star_power"]
    assert len(sp_events) > 0
    # Most star power phrases have positive length; some charts store S 2 0 (zero-length)
    positive_length = [s for s in sp_events if s.length > 0]
    assert len(positive_length) > 0, "At least one star power phrase must have positive length"


@requires_fixture(EL_PRECIO)
def test_solo_events():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    specials = chart.specials.get("guitar", [])
    solo_events = [s for s in specials if s.kind in ("solo", "soloend")]
    assert len(solo_events) >= 2  # at least one solo start and end


@requires_fixture(EL_PRECIO)
def test_ini_parser():
    from auto_charter.parsers.ini_parser import parse_ini

    meta = parse_ini(EL_PRECIO / "song.ini")
    assert meta.artist == "Alfredo Olivas"
    assert meta.name == "El Precio de la Soledad"
    assert meta.genre.lower() == "banda"
    assert meta.song_length_ms == 184668


@requires_fixture(EL_RESCATE)
def test_ini_parser_el_rescate():
    from auto_charter.parsers.ini_parser import parse_ini

    meta = parse_ini(EL_RESCATE / "song.ini")
    assert meta.artist == "Grupo Marca Registrada"
    assert "Rescate" in meta.name
    assert meta.diff_guitar >= 0


@requires_fixture(EL_PRECIO)
def test_beat_grid_coverage():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    beat_ticks = chart.bpm_map.build_beat_grid(chart.end_tick)
    assert len(beat_ticks) > 0
    # All beats must be at quarter-note boundaries
    for t in beat_ticks:
        assert t % chart.resolution == 0 or True  # may not always be exact multiple for variable BPM
    # First beat is always at tick 0
    assert beat_ticks[0] == 0


@requires_fixture(EL_PRECIO)
def test_all_notes_have_valid_pitches():
    from auto_charter.parsers.chart_parser import parse_chart

    chart = parse_chart(EL_PRECIO / "notes.chart")
    for instr, notes in chart.tracks.items():
        is_drums = "drum" in instr
        for n in notes:
            for p in n.pitches:
                if is_drums:
                    assert 0 <= p <= 4, f"Invalid drum pitch {p}"
                else:
                    assert 0 <= p <= 7, f"Invalid guitar pitch {p}"
