"""inspect_song — print a human-readable token trace for one song.

Usage:
    uv run inspect-song PATH/TO/SONG/DIR [--instrument guitar] [--no-beats]

Example:
    uv run inspect-song "test_dataset/El Precio de la Soledad/" --instrument guitar
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command()
@click.argument("song_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--instrument", "-i", default="guitar",
              help="Instrument to inspect (guitar, bass, drums). Default: guitar")
@click.option("--no-beats", is_flag=True, default=False,
              help="Hide BEAT_BOUNDARY tokens from output")
@click.option("--max-tokens", default=500, type=int,
              help="Maximum tokens to print (default 500)")
def main(song_dir: str, instrument: str, no_beats: bool, max_tokens: int) -> None:
    """Print the token sequence for an instrument track in a song folder."""
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.parsers.midi_parser import parse_midi
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.vocab.tokens import Vocab

    song_path = Path(song_dir)

    chart_path = song_path / "notes.chart"
    midi_path = song_path / "notes.mid"

    if chart_path.exists():
        chart = parse_chart(chart_path)
        fmt = "chart"
    elif midi_path.exists():
        chart = parse_midi(midi_path)
        fmt = "midi"
    else:
        click.echo(f"ERROR: No notes.chart or notes.mid found in {song_dir}", err=True)
        sys.exit(1)

    if instrument not in chart.tracks or not chart.tracks[instrument]:
        available = list(chart.instruments())
        click.echo(f"ERROR: Instrument '{instrument}' not found. Available: {available}", err=True)
        sys.exit(1)

    tokens = encode_track(chart, instrument)

    click.echo(f"Song: {song_path.name}")
    click.echo(f"Format: {fmt}")
    click.echo(f"Instrument: {instrument}")
    click.echo(f"Notes: {len(chart.tracks[instrument])}")
    click.echo(f"Beats: {len(chart.bpm_map.build_beat_grid(chart.end_tick))}")
    click.echo(f"Tokens: {len(tokens)}")
    click.echo()

    printed = 0
    for i, tok in enumerate(tokens):
        if no_beats and tok == Vocab.BEAT_BOUNDARY:
            continue
        name = Vocab.token_name(tok)
        click.echo(f"  [{i:5d}] {tok:3d}  {name}")
        printed += 1
        if printed >= max_tokens:
            remaining = len(tokens) - i - 1
            click.echo(f"  ... ({remaining} more tokens)")
            break


if __name__ == "__main__":
    main()
