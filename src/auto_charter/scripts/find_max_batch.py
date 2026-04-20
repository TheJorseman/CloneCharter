"""Find maximum batch size for AutoCharterModel using synthetic data (no disk I/O).

Binary-searches batch sizes with a full forward+backward pass on random tensors.
Run before training to pick the right --batch-size for your GPU.

Usage:
    uv run find-max-batch
    uv run find-max-batch --max-tokens 4096 --max-beats 512 --mixed-precision bf16
"""

from __future__ import annotations

import gc

import click
import torch


def _make_batch(batch_size: int, max_tokens: int, max_beats: int, vocab_size: int, device: torch.device) -> dict:
    B, T, N = batch_size, max_tokens, max_beats
    return {
        "mert_embeddings":    torch.randn(B, N, 1024, device=device, dtype=torch.float16),
        "logmel_frames":      torch.randn(B, N, 32, 128, device=device, dtype=torch.float16),
        "bpm_at_beat":        torch.full((B, N), 120.0, device=device),
        "time_sig_num":       torch.full((B, N), 4, device=device, dtype=torch.long),
        "time_sig_den":       torch.full((B, N), 4, device=device, dtype=torch.long),
        "beat_duration_s":    torch.full((B, N), 0.5, device=device),
        "beat_attention_mask": torch.ones(B, N, device=device, dtype=torch.bool),
        "input_ids":          torch.randint(0, vocab_size, (B, T), device=device),
        "beat_ids":           torch.arange(T, device=device).unsqueeze(0).expand(B, -1) % N,
        "instrument_ids":     torch.zeros(B, device=device, dtype=torch.long),
        "difficulty_ids":     torch.full((B,), 3, device=device, dtype=torch.long),
        "labels":             torch.randint(0, vocab_size, (B, T), device=device),
    }


def _try_batch(model, batch, amp_dtype, device) -> bool:
    """Returns True if forward+backward fits in memory."""
    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**batch)
            out.loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return False
    finally:
        torch.cuda.empty_cache()
        gc.collect()


@click.command()
@click.option("--max-tokens", default=2048, type=int, show_default=True, help="Token sequence length per sample")
@click.option("--max-beats", default=256, type=int, show_default=True, help="Beat sequence length per sample")
@click.option("--mixed-precision", default="bf16", type=click.Choice(["bf16", "fp16", "no"]), show_default=True)
@click.option("--d-model", default=256, type=int, show_default=True)
@click.option("--n-enc-layers", default=4, type=int, show_default=True)
@click.option("--n-dec-layers", default=4, type=int, show_default=True)
@click.option("--n-heads", default=8, type=int, show_default=True)
@click.option("--d-ff", default=512, type=int, show_default=True)
@click.option("--max-search", default=256, type=int, show_default=True, help="Upper bound for binary search")
def main(max_tokens, max_beats, mixed_precision, d_model, n_enc_layers, n_dec_layers, n_heads, d_ff, max_search):
    """Binary-search the maximum batch size that fits in GPU memory."""
    from auto_charter.model.charter_model import AutoCharterModel
    from auto_charter.model.config import AutoCharterConfig

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU:  {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"Sequence: {max_tokens} tokens × {max_beats} beats  |  precision: {mixed_precision}")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[mixed_precision]

    config = AutoCharterConfig(
        d_model=d_model, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_tokens, max_beats=max_beats,
    )
    model = AutoCharterModel(config).to(device)
    if amp_dtype is not None:
        model = model.to(amp_dtype)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters\n")

    # ── Binary search ──────────────────────────────────────────────────────────
    lo, hi = 1, max_search
    last_ok = 0

    # First check if batch=1 even fits
    batch = _make_batch(1, max_tokens, max_beats, config.vocab_size, device)
    if not _try_batch(model, batch, amp_dtype, device):
        print("ERROR: batch_size=1 already OOMs. Reduce --max-tokens or --max-beats.")
        return

    print("Searching", end="", flush=True)
    while lo <= hi:
        mid = (lo + hi) // 2
        batch = _make_batch(mid, max_tokens, max_beats, config.vocab_size, device)
        ok = _try_batch(model, batch, amp_dtype, device)
        print(f"  bs={mid} {'OK' if ok else 'OOM'}", end="", flush=True)
        if ok:
            last_ok = mid
            lo = mid + 1
        else:
            hi = mid - 1

    print(f"\n\nMax batch size: {last_ok}")
    print(f"Recommended --batch-size for training (with some headroom): {max(1, int(last_ok * 0.8))}")


if __name__ == "__main__":
    main()
