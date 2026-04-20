"""repair-shards — rewrite Parquet shards to use HuggingFace list field naming.

PyArrow names list inner fields 'element' by default.
HuggingFace datasets expects 'item'.  This mismatch causes Dataset.from_parquet()
to fail with "An error occurred while generating the dataset".

This script rewrites each shard in-place (with a temp file swap) so that
all list columns use 'item' naming, making them compatible with HF datasets.

Usage:
    python -m auto_charter.scripts.repair_shards F:/CloneCharter_converted/shards
    python -m auto_charter.scripts.repair_shards F:/CloneCharter_converted/shards --dry-run
    python -m auto_charter.scripts.repair_shards F:/CloneCharter_converted/shards --verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _make_item_schema(schema) -> "pa.Schema":
    """Recursively convert a PyArrow schema so all list types use field name 'item'."""
    import pyarrow as pa

    def fix_type(t):
        if pa.types.is_list(t):
            return pa.list_(fix_type(t.value_type))
        if pa.types.is_large_list(t):
            return pa.large_list(fix_type(t.value_type))
        if pa.types.is_struct(t):
            return pa.struct({f.name: fix_type(f.type) for f in t})
        return t

    new_fields = [pa.field(f.name, fix_type(f.type), nullable=f.nullable) for f in schema]
    return pa.schema(new_fields)


def _needs_repair(schema) -> bool:
    """Check if any list column uses 'element' naming (PyArrow default vs HF 'item')."""
    # PyArrow considers list<element:x> == list<item:x> as equal types,
    # so we must use string comparison to detect the naming difference.
    return any("<element:" in str(f.type) for f in schema)


def repair_shard(shard: Path, dry_run: bool, compression: str) -> tuple[bool, str]:
    """Repair a single shard. Returns (changed, message)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(str(shard))
    except Exception as e:
        return False, f"ERROR reading schema: {e}"

    if not _needs_repair(schema):
        return False, "already OK (no element->item rename needed)"

    new_schema = _make_item_schema(schema)

    if dry_run:
        diffs = []
        for f_old, f_new in zip(schema, new_schema):
            if str(f_old.type) != str(f_new.type):
                diffs.append(f"{f_old.name}: {f_old.type} -> {f_new.type}")
        return True, f"would rename {len(diffs)} column(s): {', '.join(diffs[:3])}{'...' if len(diffs) > 3 else ''}"

    # Load, cast, write to temp, then replace
    tmp = shard.with_suffix(".tmp.parquet")
    try:
        table = pq.read_table(str(shard))
        new_table = table.cast(new_schema)
        pq.write_table(new_table, str(tmp), compression=compression)
        del table, new_table  # release RAM before swap
        tmp.replace(shard)
        return True, "repaired"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return False, f"ERROR during repair: {e}"


def verify_shard(shard: Path) -> tuple[bool, str]:
    """Try loading a shard with HF datasets and report success/failure."""
    try:
        from datasets import Dataset
        Dataset.from_parquet(str(shard))
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Parquet shards: element->item list naming.")
    parser.add_argument("shards_dir", help="Directory containing Parquet shards")
    parser.add_argument("--split", default="train", help="Shard prefix (default: train)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After repair, verify each shard loads with datasets.Dataset.from_parquet.",
    )
    parser.add_argument(
        "--compression", default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
        help="Compression for repaired shards (default: zstd).",
    )
    parser.add_argument(
        "--first-n", type=int, default=0,
        help="Only process first N shards (0 = all, useful for testing).",
    )
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow is required. pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    shards_dir = Path(args.shards_dir)
    shards = sorted(shards_dir.glob(f"{args.split}-*.parquet"))
    if not shards:
        print(f"ERROR: No shards matching '{args.split}-*.parquet' in {shards_dir}", file=sys.stderr)
        sys.exit(1)

    if args.first_n > 0:
        shards = shards[: args.first_n]

    compression = None if args.compression == "none" else args.compression

    print(f"{'DRY RUN — ' if args.dry_run else ''}Repairing {len(shards)} shard(s) in {shards_dir}")
    print()

    n_changed = 0
    n_ok = 0
    n_error = 0

    for i, shard in enumerate(shards):
        changed, msg = repair_shard(shard, dry_run=args.dry_run, compression=compression)
        if "ERROR" in msg:
            n_error += 1
            status = "ERR"
        elif changed:
            n_changed += 1
            status = "FIX" if not args.dry_run else "DRY"
        else:
            n_ok += 1
            status = "OK "

        # Only print non-OK lines unless verbose
        if status != "OK ":
            print(f"  [{status}] {shard.name}: {msg}")

        # Progress every 100 shards
        if (i + 1) % 100 == 0 or (i + 1) == len(shards):
            print(
                f"  [{i+1:>4}/{len(shards)}]  "
                f"fixed={n_changed}  already-ok={n_ok}  errors={n_error}",
                flush=True,
            )

    print()
    if args.dry_run:
        print(f"DRY RUN complete: {n_changed} shard(s) would be repaired, {n_ok} already OK, {n_error} errors.")
        if n_changed > 0:
            print("Re-run without --dry-run to apply the fix.")
    else:
        print(f"Repair complete: {n_changed} shard(s) repaired, {n_ok} already OK, {n_error} errors.")

    if args.verify and not args.dry_run:
        print()
        print("Verifying repaired shards with datasets.Dataset.from_parquet ...")
        n_pass = 0
        n_fail = 0
        for shard in shards:
            ok, msg = verify_shard(shard)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                print(f"  FAIL {shard.name}: {msg}")
        print(f"Verification: {n_pass} passed, {n_fail} failed.")
        if n_fail > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
