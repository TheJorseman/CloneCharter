"""validate_roundtrip — verify encode→decode→re-encode produces identical tokens.

For each song in a directory:
  1. Parse the chart
  2. Encode each instrument track → tokens
  3. Decode tokens → ChartData'
  4. Re-encode ChartData' → tokens'
  5. Assert tokens == tokens' (beat boundaries excluded, as they depend on BPMMap)

Usage:
    uv run validate-roundtrip PATH/TO/TEST_DATASET/
    uv run validate-roundtrip test_dataset/ --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(dataset_dir: str, verbose: bool) -> None:
    """Run encode→decode→re-encode roundtrip test for all songs."""
    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.parsers.midi_parser import parse_midi
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.tokenizer.decoder import decode_tokens
    from auto_charter.vocab.tokens import Vocab

    dataset_path = Path(dataset_dir)
    song_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if not song_dirs:
        click.echo("No song directories found.", err=True)
        sys.exit(1)

    total_tracks = 0
    passed = 0
    failed = 0

    for song_dir in sorted(song_dirs):
        chart_path = song_dir / "notes.chart"
        midi_path = song_dir / "notes.mid"

        if chart_path.exists():
            try:
                chart = parse_chart(chart_path)
            except Exception as e:
                click.echo(f"SKIP  {song_dir.name}: parse error: {e}")
                continue
        elif midi_path.exists():
            try:
                chart = parse_midi(midi_path)
            except Exception as e:
                click.echo(f"SKIP  {song_dir.name}: parse error: {e}")
                continue
        else:
            continue

        for instrument in chart.instruments():
            total_tracks += 1

            # Step 1: encode (no beat boundaries for round-trip — decoder can't reproduce them
            # without the original BPMMap)
            tokens1 = encode_track(chart, instrument, include_beat_boundaries=False)

            # Step 2: decode
            decoded = decode_tokens(tokens1, resolution=chart.resolution)

            # Step 3: re-encode the decoded chart
            tokens2 = encode_track(decoded, instrument, include_beat_boundaries=False)

            if tokens1 == tokens2:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            if verbose or status == "FAIL":
                click.echo(f"{status}  {song_dir.name} / {instrument}  "
                           f"({len(tokens1)} tokens -> {len(tokens2)} tokens)")
                if status == "FAIL":
                    # Find first mismatch
                    for j, (t1, t2) in enumerate(zip(tokens1, tokens2)):
                        if t1 != t2:
                            click.echo(f"      First mismatch at pos {j}: "
                                       f"{Vocab.token_name(t1)} vs {Vocab.token_name(t2)}")
                            break
                    if len(tokens1) != len(tokens2):
                        click.echo(f"      Length mismatch: {len(tokens1)} vs {len(tokens2)}")
            elif not verbose:
                click.echo(f"PASS  {song_dir.name} / {instrument}")

    click.echo()
    click.echo(f"Results: {passed}/{total_tracks} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
