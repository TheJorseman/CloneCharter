"""inspect-shards — diagnose schema / data problems across Parquet shards.

Loads shards one by one (streaming, low RAM) and reports:
  - Column mismatches vs the first shard (reference schema)
  - Column mismatches vs the HF Features schema (if available)
  - Data-level issues: null counts, array shape variance for logmel/mert columns
  - Which specific shards are incompatible

Usage:
    python -m auto_charter.scripts.inspect_shards F:/CloneCharter_converted/shards
    python -m auto_charter.scripts.inspect_shards F:/CloneCharter_converted/shards --split train --sample-rows 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def _fmt_type(t) -> str:
    return str(t)


def compare_schemas(ref_schema, schema, shard_name: str) -> list[str]:
    """Return list of human-readable differences between ref_schema and schema.

    Uses string comparison for types because PyArrow considers list<element:x>
    equal to list<item:x> at the Python level even though they differ in naming.
    """
    diffs = []
    ref = {f.name: str(f.type) for f in ref_schema}
    cur = {f.name: str(f.type) for f in schema}
    for col in sorted(set(ref) | set(cur)):
        if col not in cur:
            diffs.append(f"  MISSING col '{col}' in {shard_name}  (expected {ref[col]})")
        elif col not in ref:
            diffs.append(f"  EXTRA   col '{col}' in {shard_name}  (type {cur[col]})")
        elif ref[col] != cur[col]:
            diffs.append(f"  TYPE    '{col}' in {shard_name}: got {cur[col]!r}, expected {ref[col]!r}")
    return diffs


def check_data_issues(table, shard_name: str, sample_rows: int) -> list[str]:
    """Scan a pyarrow Table for data-level problems (nulls, shape variance)."""
    import pyarrow as pa

    issues = []
    nrows = len(table)

    # --- Null checks on required columns ---
    required_not_null = ["song_id", "instrument", "tokens", "logmel_frames"]
    for col in required_not_null:
        if col not in table.schema.names:
            continue
        arr = table.column(col)
        null_count = arr.null_count
        if null_count:
            issues.append(f"  NULL    '{col}' has {null_count}/{nrows} nulls in {shard_name}")

    # --- Array shape consistency for logmel_frames: must be [B, 32, 128] ---
    if "logmel_frames" in table.schema.names:
        col = table.column("logmel_frames")
        bad_beats = []
        for row_idx, cell in enumerate(col):
            if cell is None or cell.as_py() is None:
                continue
            beats = cell.as_py()        # list of (list of list of float)
            for beat_idx, frame in enumerate(beats):
                if frame is None:
                    bad_beats.append((row_idx, beat_idx, "None"))
                    continue
                rows_in_frame = len(frame)
                if rows_in_frame != 32:
                    bad_beats.append((row_idx, beat_idx, f"time_frames={rows_in_frame} (expected 32)"))
                    continue
                for t, mel_row in enumerate(frame):
                    if mel_row is None:
                        bad_beats.append((row_idx, beat_idx, f"None at t={t}"))
                        break
                    if len(mel_row) != 128:
                        bad_beats.append((row_idx, beat_idx, f"mels={len(mel_row)} at t={t} (expected 128)"))
                        break
                if len(bad_beats) >= 5:
                    break  # don't scan entire shard, just flag early
            if len(bad_beats) >= 5:
                break
        if bad_beats:
            for row_idx, beat_idx, reason in bad_beats[:5]:
                issues.append(
                    f"  SHAPE   logmel_frames[row={row_idx}, beat={beat_idx}] {reason} in {shard_name}"
                )

    # --- mert_embeddings: [B, 768] ---
    if "mert_embeddings" in table.schema.names:
        col = table.column("mert_embeddings")
        bad = []
        for row_idx, cell in enumerate(col):
            if cell is None or cell.as_py() is None:
                continue
            beats = cell.as_py()
            for beat_idx, vec in enumerate(beats):
                if vec is None:
                    bad.append((row_idx, beat_idx, "None"))
                    break
                if len(vec) != 768:
                    bad.append((row_idx, beat_idx, f"dim={len(vec)} (expected 768)"))
                    break
            if len(bad) >= 3:
                break
        if bad:
            for row_idx, beat_idx, reason in bad[:3]:
                issues.append(
                    f"  SHAPE   mert_embeddings[row={row_idx}, beat={beat_idx}] {reason} in {shard_name}"
                )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Parquet shards for schema/data problems.")
    parser.add_argument("shards_dir", help="Directory containing Parquet shards")
    parser.add_argument("--split", default="train", help="Shard prefix (default: train)")
    parser.add_argument(
        "--sample-rows", type=int, default=0,
        help="Rows to data-check per shard (0 = schema-only, fast). Use 10-50 for data checks.",
    )
    parser.add_argument(
        "--check-hf-schema", action="store_true", default=True,
        help="Compare each shard against the HuggingFace Features schema.",
    )
    parser.add_argument(
        "--no-check-hf-schema", dest="check_hf_schema", action="store_false",
    )
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow is required. pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    shards_dir = Path(args.shards_dir)
    shards = sorted(shards_dir.glob(f"{args.split}-*.parquet"))
    if not shards:
        print(f"ERROR: No shards matching '{args.split}-*.parquet' in {shards_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shards)} shards in {shards_dir}")
    print(f"Data check: {'first ' + str(args.sample_rows) + ' rows per shard' if args.sample_rows else 'schema only (fast)'}")
    print()

    # --- Load HF schema for comparison ---
    hf_schema = None
    if args.check_hf_schema:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parents[3]))
            from auto_charter.dataset.schema import get_features
            hf_schema = get_features().arrow_schema
            print(f"HF schema loaded: {len(hf_schema)} columns")
        except Exception as e:
            print(f"WARNING: Could not load HF schema ({e}) — skipping HF comparison.")
        print()

    # --- Per-shard inspection (streaming: load one at a time) ---
    ref_schema = None
    ref_name = None
    all_issues: list[tuple[str, list[str]]] = []   # (shard_name, [issue_lines])
    schema_summary: dict[str, int] = defaultdict(int)   # schema_fingerprint -> count

    for i, shard in enumerate(shards):
        name = shard.name
        issues: list[str] = []

        # Read schema only first (very cheap)
        try:
            schema = pq.read_schema(str(shard))
        except Exception as exc:
            issues.append(f"  CORRUPT cannot read schema: {exc}")
            all_issues.append((name, issues))
            continue

        # Fingerprint for grouping
        fingerprint = "|".join(f"{f.name}:{f.type}" for f in schema)
        schema_summary[fingerprint] += 1

        # Set reference schema from first shard
        if ref_schema is None:
            ref_schema = schema
            ref_name = name

        # Compare against reference
        diffs = compare_schemas(ref_schema, schema, name)
        issues.extend(diffs)

        # Compare against HF schema
        if hf_schema is not None:
            hf_diffs = compare_schemas(hf_schema, schema, name)
            # Only report HF diffs not already caught by ref comparison
            for d in hf_diffs:
                if d not in diffs:
                    issues.append(d.replace("  ", "  [vs HF] "))

        # Data-level checks (only when requested — reads row data, costs RAM per shard)
        if args.sample_rows > 0 and not issues:
            try:
                table = pq.read_table(str(shard))
                if args.sample_rows < len(table):
                    table = table.slice(0, args.sample_rows)
                data_issues = check_data_issues(table, name, args.sample_rows)
                issues.extend(data_issues)
                del table  # release RAM immediately
            except Exception as exc:
                issues.append(f"  ERROR   reading table data: {exc}")

        if issues:
            all_issues.append((name, issues))

        # Progress every 50 shards
        if (i + 1) % 50 == 0 or (i + 1) == len(shards):
            bad_so_far = len(all_issues)
            print(f"  [{i+1:>4}/{len(shards)}] scanned — {bad_so_far} shard(s) with issues so far", flush=True)

    # --- Summary ---
    print()
    print("=" * 70)
    print(f"SCHEMA VARIANTS FOUND: {len(schema_summary)}")
    for idx, (fp, count) in enumerate(sorted(schema_summary.items(), key=lambda x: -x[1])):
        cols = [c.split(":")[0] for c in fp.split("|")]
        print(f"  Variant {idx+1} ({count} shards): {len(cols)} columns — {cols[:5]}{'...' if len(cols) > 5 else ''}")
    print()

    if not all_issues:
        print("OK — No schema or data issues found across all shards.")
        print()
        print("The failure in Dataset.from_parquet is likely caused by incompatible")
        print("Arrow list types (e.g. logmel_frames inner array size varies between rows).")
        print("Re-run with --sample-rows 10 to check actual data shapes.")
    else:
        print(f"ISSUES FOUND IN {len(all_issues)} SHARD(S):")
        print()
        for shard_name, issues in all_issues:
            print(f"[{shard_name}]")
            for line in issues:
                print(line)
        print()
        print(f"Total: {len(all_issues)} / {len(shards)} shards have problems.")

        # Suggest fix
        print()
        print("SUGGESTED FIX:")
        print("  Delete the bad shards and re-run process-dataset with --resume:")
        for shard_name, _ in all_issues:
            print(f"    del \"{shards_dir / shard_name}\"")


if __name__ == "__main__":
    main()
