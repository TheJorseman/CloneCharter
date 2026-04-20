"""push-dataset — validate shard schemas and push to HuggingFace Hub without cache.

Reads shards ONE AT A TIME (low RAM). Validates schema consistency across all
shards. Uploads parquet files directly via huggingface_hub.HfApi — the datasets
library cache (~/.cache/huggingface) is never written.

Optionally generates a README.md dataset card and uploads it alongside the data.

Usage:
    python -m auto_charter.scripts.push_dataset \\
        --shards-dir F:/CloneCharter_converted/shards \\
        --repo-id thejorseman/CloneHeroCharts

    # validate only, no upload:
    python -m auto_charter.scripts.push_dataset \\
        --shards-dir F:/CloneCharter_converted/shards \\
        --validate-only

    # skip validation, just upload:
    python -m auto_charter.scripts.push_dataset \\
        --shards-dir F:/CloneCharter_converted/shards \\
        --repo-id thejorseman/CloneHeroCharts \\
        --no-validate
"""

from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path
from collections import defaultdict

import click

logger = logging.getLogger(__name__)


# ─── Schema helpers ────────────────────────────────────────────────────────────


def _load_hf_arrow_schema():
    """Load the canonical HuggingFace Features schema as an Arrow schema."""
    try:
        from auto_charter.dataset.schema import get_features
        return get_features().arrow_schema
    except Exception as e:
        logger.warning("Could not load HF Features schema (%s) — skipping HF comparison.", e)
        return None


def _normalize_list_naming(t: str) -> str:
    """Normalize PyArrow list field naming: 'element' and 'item' are semantically identical.

    PyArrow uses 'element' by default; HuggingFace datasets expects 'item'.
    Both represent the same Arrow list type — treat them as equivalent.
    """
    return t.replace("element: ", "item: ").replace("<element:", "<item:")


def _compare_schemas(ref_schema, schema, shard_name: str) -> list[str]:
    """Return list of human-readable schema differences (ignores element vs item naming)."""
    diffs = []
    ref = {f.name: _normalize_list_naming(str(f.type)) for f in ref_schema}
    cur = {f.name: _normalize_list_naming(str(f.type)) for f in schema}
    for col in sorted(set(ref) | set(cur)):
        if col not in cur:
            diffs.append(f"  MISSING '{col}' in {shard_name}  (expected {ref[col]})")
        elif col not in ref:
            diffs.append(f"  EXTRA   '{col}' in {shard_name}  (type {cur[col]})")
        elif ref[col] != cur[col]:
            diffs.append(f"  TYPE    '{col}' in {shard_name}: got {cur[col]!r}, expected {ref[col]!r}")
    return diffs


def validate_shards(shards: list[Path], split: str) -> tuple[bool, dict]:
    """Validate schema consistency across all shards. Returns (ok, stats)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        click.echo("ERROR: pyarrow is required. pip install pyarrow", err=True)
        sys.exit(1)

    hf_schema = _load_hf_arrow_schema()
    if hf_schema is not None:
        click.echo(f"  HF schema loaded: {len(hf_schema)} columns")

    ref_schema = None
    ref_name = None
    all_issues: list[tuple[str, list[str]]] = []
    schema_variants: dict[str, int] = defaultdict(int)

    click.echo(f"\nValidating {len(shards)} shard(s) (schema-only, no full data load)...")

    for i, shard in enumerate(shards):
        name = shard.name
        issues: list[str] = []

        try:
            schema = pq.read_schema(str(shard))
        except Exception as exc:
            issues.append(f"  CORRUPT: cannot read schema — {exc}")
            all_issues.append((name, issues))
            continue

        fingerprint = "|".join(f"{f.name}:{f.type}" for f in schema)
        schema_variants[fingerprint] += 1

        if ref_schema is None:
            ref_schema = schema
            ref_name = name

        diffs = _compare_schemas(ref_schema, schema, name)
        issues.extend(diffs)

        if hf_schema is not None:
            hf_diffs = _compare_schemas(hf_schema, schema, name)
            for d in hf_diffs:
                if d not in diffs:
                    issues.append(d.replace("  ", "  [vs HF] "))

        if issues:
            all_issues.append((name, issues))

        if (i + 1) % 100 == 0 or (i + 1) == len(shards):
            print(
                f"  [{i+1:>5}/{len(shards)}] scanned — {len(all_issues)} shard(s) with issues",
                flush=True,
            )

    click.echo()

    # Report schema variants
    n_variants = len(schema_variants)
    if n_variants == 1:
        click.echo(f"Schema: OK — all {len(shards)} shards share the same schema.")
    else:
        click.echo(f"Schema: {n_variants} VARIANTS FOUND across {len(shards)} shards:")
        for idx, (fp, count) in enumerate(sorted(schema_variants.items(), key=lambda x: -x[1])):
            cols = [c.split(":")[0] for c in fp.split("|")]
            click.echo(f"  Variant {idx+1} ({count} shards): {cols[:6]}{'...' if len(cols) > 6 else ''}")

    # Detect element→item naming issue separately (cosmetic, not a real inconsistency)
    needs_repair = False
    if ref_schema is not None:
        needs_repair = any("<element:" in str(f.type) for f in ref_schema)
    if needs_repair:
        shards_dir = shards[0].parent if shards else Path(".")
        click.echo(
            "WARNING: Shards use PyArrow list naming ('element') instead of "
            "HuggingFace naming ('item').\n"
            "         Run repair_shards.py to fix before HF load_dataset() will work:\n"
            f"         uv run python -m auto_charter.scripts.repair_shards {shards_dir}"
        )

    if all_issues:
        click.echo(f"\nISSUES IN {len(all_issues)} SHARD(S):")
        for shard_name, issues in all_issues[:20]:  # cap output
            click.echo(f"  [{shard_name}]")
            for line in issues[:5]:
                click.echo(f"    {line}")
        if len(all_issues) > 20:
            click.echo(f"  ... and {len(all_issues) - 20} more")
        click.echo()

    stats = {
        "total": len(shards),
        "ok": len(shards) - len(all_issues),
        "issues": len(all_issues),
        "schema_variants": n_variants,
        "needs_repair": needs_repair,
    }
    ok = len(all_issues) == 0 and n_variants == 1
    return ok, stats


# ─── README generation ─────────────────────────────────────────────────────────


def generate_readme(shards: list[Path], repo_id: str, split: str) -> str:
    """Generate a HuggingFace dataset card README.md as a string."""
    import pyarrow.parquet as pq

    # Gather basic stats without loading full data (metadata only)
    total_rows = 0
    schema = None
    for shard in shards:
        try:
            meta = pq.read_metadata(str(shard))
            total_rows += meta.num_rows
            if schema is None:
                schema = pq.read_schema(str(shard))
        except Exception:
            pass

    col_names = [f.name for f in schema] if schema else []
    has_mert = "mert_embeddings" in col_names
    has_logmel = "logmel_frames" in col_names

    readme = f"""---
language:
- en
license: mit
task_categories:
- audio-to-audio
- token-classification
task_ids:
- music-generation
tags:
- music
- gaming
- clone-hero
- rhythm-game
- guitar-hero
- audio-generation
- chart-generation
pretty_name: Clone Hero Charts Dataset
size_categories:
- {_size_category(total_rows)}
---

# Clone Hero Charts Dataset

## Dataset Description

Tokenized [Clone Hero](https://clonehero.net/) charts with beat-level audio conditioning.
Each row is one instrument track (guitar / bass / drums) from one song.

| Feature | Value |
|---------|-------|
| Total rows (`{split}`) | {total_rows:,} |
| Parquet shards | {len(shards)} |
| Audio: MERT embeddings | {"Yes [num_beats, 768]" if has_mert else "No"} |
| Audio: log-mel frames | {"Yes [num_beats, 32, 128]" if has_logmel else "No"} |

## Dataset Structure

### Data Fields

| Column | Type | Description |
|--------|------|-------------|
| `song_id` | string | MD5 hash of artist + title |
| `instrument` | string | `"guitar"` / `"bass"` / `"drums"` |
| `source_format` | string | `"chart"` or `"midi"` |
| `tokens` | list[int32] | Token sequence (model target) |
| `num_tokens` | int32 | Token count |
| `num_beats` | int32 | Beat count |
| `mert_embeddings` | list[list[float32]] | MERT embeddings per beat [B, 768] |
| `logmel_frames` | list[list[list[float32]]] | Log-mel spectrogram [B, 32, 128] |
| `beat_times_s` | list[float32] | Beat start times (seconds) |
| `beat_durations_s` | list[float32] | Beat durations (seconds) |
| `bpm_at_beat` | list[float32] | BPM at each beat |
| `time_sig_num_at_beat` | list[int32] | Time signature numerator |
| `time_sig_den_at_beat` | list[int32] | Time signature denominator |
| `song_name` | string | Song title |
| `artist` | string | Artist name |
| `genre` | string | Genre tag |
| `charter` | string | Charter name |
| `year` | int32 | Release year |
| `song_length_ms` | int32 | Song length in milliseconds |
| `difficulty` | int32 | Difficulty 0–6 (−1 = unset) |
| `resolution` | int32 | Tick resolution (normalised to 192) |
| `has_star_power` | bool | Track contains star-power sections |
| `has_solo` | bool | Track contains solo sections |
| `has_dedicated_stem` | bool | Instrument-specific audio stem available |
| `num_notes` | int32 | Total note count |
| `notes_per_beat_mean` | float32 | Average notes per beat |
| `chord_ratio` | float32 | Fraction of notes that are chords |
| `sustain_mean_ticks` | float32 | Mean sustain length in ticks |
| `bpm_mean` | float32 | Mean BPM |
| `bpm_std` | float32 | BPM standard deviation |

### Data Splits

| Split | Rows |
|-------|------|
| `{split}` | {total_rows:,} |

## Usage

```python
from datasets import load_dataset

# Stream individual shards (low RAM):
ds = load_dataset("{repo_id}", split="{split}", streaming=True)
for row in ds:
    print(row["song_name"], row["instrument"], len(row["tokens"]))
```

## Source Data

Charts scraped from public Clone Hero repositories in `.chart` and `.mid` format.
Audio conditioning extracted with:
- **MERT** (`m-a-p/MERT-v1-330M`) — 768-dimensional embeddings, mean-pooled per beat
- **Log-mel** — 128-bin spectrograms resampled to 32 time frames per beat

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{{clonecharter2025,
  title  = {{Clone Hero Charts Dataset}},
  author = {{The Jorseman}},
  year   = {{2025}},
  url    = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
    return readme


def _size_category(n: int) -> str:
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    return "1M<n<10M"


# ─── Upload helpers ────────────────────────────────────────────────────────────


def _get_uploaded_filenames(api, repo_id: str, split: str) -> set[str]:
    """Return the set of shard filenames already present in data/{split}/ on the Hub."""
    try:
        from huggingface_hub import list_repo_tree
        uploaded = set()
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=f"data/{split}",
            recursive=False,
        ):
            if hasattr(item, "path"):
                uploaded.add(Path(item.path).name)
        return uploaded
    except Exception as e:
        logger.warning("Could not list existing files in repo (%s) — will upload all shards.", e)
        return set()


def push_to_hub(
    shards: list[Path],
    repo_id: str,
    split: str,
    readme_text: str | None,
    token: str | None,
) -> None:
    """Upload parquet shards + README directly via huggingface_hub (no datasets cache).

    Resumes automatically: shards already present in the Hub repo are skipped.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        click.echo("ERROR: huggingface_hub is required. pip install huggingface_hub", err=True)
        sys.exit(1)

    api = HfApi(token=token)

    # Ensure repo exists
    click.echo(f"Creating/verifying repo: {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload README first (so it's visible immediately)
    if readme_text is not None:
        click.echo("Uploading README.md ...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(readme_text)
            readme_tmp = Path(f.name)
        try:
            api.upload_file(
                path_or_fileobj=str(readme_tmp),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset card README.md",
            )
        finally:
            readme_tmp.unlink(missing_ok=True)
        click.echo("  README.md uploaded.")

    # Check which shards are already on the Hub
    click.echo(f"\nChecking already-uploaded shards in data/{split}/ ...")
    already_uploaded = _get_uploaded_filenames(api, repo_id, split)
    pending = [s for s in shards if s.name not in already_uploaded]
    n_total = len(shards)
    n_skip = len(already_uploaded)
    n_pending = len(pending)

    if n_skip:
        click.echo(f"  {n_skip}/{n_total} already uploaded — skipping.")
    click.echo(f"  {n_pending} shard(s) remaining to upload.")

    if not pending:
        click.echo("All shards already uploaded.")
        click.echo(f"\nDone. Dataset available at: https://huggingface.co/datasets/{repo_id}")
        return

    # Upload in batches to stay well under the 128 commits/hour rate limit.
    # 50 shards per commit → ceil(1753/50) = 36 commits total.
    _upload_batched(api, pending, repo_id, split, n_skip, n_total, batch_size=50, token=token)

    click.echo(f"\nDone. Dataset available at: https://huggingface.co/datasets/{repo_id}")


def _parse_retry_after(error: Exception) -> int:
    """Extract seconds to wait from a 429 error message. Returns 120 if not found."""
    import re
    msg = str(error)
    m = re.search(r"[Rr]etry after (\d+) second", msg)
    if m:
        return int(m.group(1)) + 5  # +5s safety margin
    # Fallback: look for 'about N hour'
    m = re.search(r"about (\d+) hour", msg)
    if m:
        return int(m.group(1)) * 3600 + 30
    return 120


def _upload_batched(
    api,
    pending: list[Path],
    repo_id: str,
    split: str,
    n_already: int,
    n_total: int,
    batch_size: int = 50,
    max_retries: int = 10,
    token: str | None = None,
) -> None:
    """Upload shards in batches, auto-waiting on 429 rate-limit errors."""
    import time
    from huggingface_hub import CommitOperationAdd, HfApi as _HfApi

    n_pending = len(pending)
    n_batches = (n_pending + batch_size - 1) // batch_size
    click.echo(
        f"\nUploading {n_pending} shard(s) in {n_batches} batch(es) "
        f"of up to {batch_size} files each ..."
    )

    n_uploaded = 0
    batch_idx = 0
    while batch_idx < n_batches:
        batch = pending[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        operations = [
            CommitOperationAdd(
                path_in_repo=f"data/{split}/{s.name}",
                path_or_fileobj=str(s),
            )
            for s in batch
        ]
        first = batch[0].name
        last = batch[-1].name

        for attempt in range(1, max_retries + 1):
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Add shards {first}…{last} (batch {batch_idx+1}/{n_batches})",
                )
                break  # success
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                is_network = any(
                    kw in err_str.lower()
                    for kw in ("getaddrinfo", "connecterror", "connectionerror",
                               "timeout", "remotedisconnected", "connectionreset",
                               "errno 11002", "errno 10054", "errno 104",
                               "client has been closed")
                )

                if is_rate_limit:
                    wait = _parse_retry_after(e)
                    print(
                        f"\n  [rate limit] batch {batch_idx+1}/{n_batches} — "
                        f"waiting {wait}s (attempt {attempt}/{max_retries}) ...",
                        flush=True,
                    )
                elif is_network:
                    wait = min(30 * attempt, 300)  # 30s, 60s, 90s … cap at 5 min
                    print(
                        f"\n  [network error] {e.__class__.__name__}: {err_str[:120]}\n"
                        f"  batch {batch_idx+1}/{n_batches} — "
                        f"waiting {wait}s before retry (attempt {attempt}/{max_retries}) ...",
                        flush=True,
                    )
                else:
                    raise

                time.sleep(wait)

                # Recreate the HfApi client — the internal httpx client may be
                # closed/corrupted after a network error.
                api = _HfApi(token=token)

                # Re-check which files are now uploaded to avoid double-uploading
                # (the commit may have partially landed before the error)
                try:
                    uploaded_now = _get_uploaded_filenames(api, repo_id, split)
                    batch = [s for s in batch if s.name not in uploaded_now]
                    if not batch:
                        print(f"  batch {batch_idx+1} already committed — skipping.", flush=True)
                        break
                    operations = [
                        CommitOperationAdd(
                            path_in_repo=f"data/{split}/{s.name}",
                            path_or_fileobj=str(s),
                        )
                        for s in batch
                    ]
                except Exception:
                    pass  # keep original operations if listing fails
        else:
            click.echo(
                f"\nERROR: batch {batch_idx+1} failed after {max_retries} retries. "
                "Re-run the script to resume from this point.",
                err=True,
            )
            raise RuntimeError(f"Batch {batch_idx+1} upload failed after {max_retries} retries.")

        n_uploaded += len(pending[batch_idx * batch_size : (batch_idx + 1) * batch_size])
        batch_idx += 1
        print(
            f"  batch {batch_idx:>4}/{n_batches}  "
            f"[{n_uploaded:>5}/{n_pending}] uploaded  "
            f"(total {n_already + n_uploaded}/{n_total})",
            flush=True,
        )


# ─── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--shards-dir", "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing Parquet shards (e.g. F:/CloneCharter_converted/shards).",
)
@click.option(
    "--split",
    default="train",
    show_default=True,
    help="Shard prefix / dataset split name.",
)
@click.option(
    "--repo-id",
    default=None,
    help="HuggingFace repo ID to push to (e.g. thejorseman/CloneHeroCharts). "
         "Required unless --validate-only.",
)
@click.option(
    "--token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace write token (or set HF_TOKEN env var).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help="Validate schema consistency across all shards before uploading.",
)
@click.option(
    "--validate-only",
    is_flag=True,
    default=False,
    help="Only validate; do not upload anything.",
)
@click.option(
    "--readme/--no-readme",
    default=True,
    show_default=True,
    help="Generate and upload a dataset card README.md.",
)
@click.option(
    "--fail-on-schema-mismatch/--no-fail-on-schema-mismatch",
    default=True,
    show_default=True,
    help="Abort upload if schema mismatches are found.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    show_default=True,
)
def main(
    shards_dir: str,
    split: str,
    repo_id: str | None,
    token: str | None,
    validate: bool,
    validate_only: bool,
    readme: bool,
    fail_on_schema_mismatch: bool,
    log_level: str,
) -> None:
    """Validate shard schemas and push parquet dataset to HuggingFace Hub.

    Processes shards one at a time — does NOT load the full dataset into RAM.
    Does NOT write to the HuggingFace cache (~/.cache/huggingface/datasets).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not validate_only and not repo_id:
        click.echo(
            "ERROR: --repo-id is required unless --validate-only is set.", err=True
        )
        sys.exit(1)

    shards_path = Path(shards_dir)
    shards = sorted(shards_path.glob(f"{split}-*.parquet"))

    if not shards:
        click.echo(
            f"ERROR: No shards matching '{split}-*.parquet' found in {shards_path}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Found {len(shards)} shard(s) in {shards_path}")

    # ── Validation ─────────────────────────────────────────────────────────────
    schema_ok = True
    if validate:
        schema_ok, stats = validate_shards(shards, split)
        click.echo(
            f"Validation: {stats['ok']}/{stats['total']} shards OK  "
            f"| {stats['schema_variants']} schema variant(s)  "
            f"| {stats['issues']} issue(s)"
        )
        if schema_ok:
            click.echo("Validation PASSED.")
        else:
            click.echo("Validation FAILED — schema mismatches detected.")
            if fail_on_schema_mismatch and not validate_only:
                click.echo(
                    "Aborting upload. Fix the issues above or pass "
                    "--no-fail-on-schema-mismatch to upload anyway.",
                    err=True,
                )
                sys.exit(1)
    else:
        click.echo("Validation skipped (--no-validate).")

    if validate_only:
        sys.exit(0 if schema_ok else 1)

    # ── README ─────────────────────────────────────────────────────────────────
    readme_text: str | None = None
    if readme and repo_id:
        click.echo("\nGenerating README.md ...")
        try:
            readme_text = generate_readme(shards, repo_id, split)
            click.echo(f"  README generated ({len(readme_text):,} chars).")
        except Exception as e:
            logger.warning("README generation failed: %s", e)
            click.echo(f"WARNING: README generation failed ({e}). Continuing without it.")

    # ── Upload ─────────────────────────────────────────────────────────────────
    click.echo()
    push_to_hub(
        shards=shards,
        repo_id=repo_id,
        split=split,
        readme_text=readme_text,
        token=token,
    )


if __name__ == "__main__":
    main()
