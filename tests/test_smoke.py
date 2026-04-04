"""
Smoke tests — verify the model, collator, metrics and a training step
work correctly WITHOUT needing the full 74GB dataset.

Run from the project root:
    python tests/test_smoke.py
    python tests/test_smoke.py --device cuda   # explicit GPU
    python tests/test_smoke.py --fast           # skip the training step test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from models.chart_transformer import ChartTransformer, ModelConfig
from models.tokenizer import CloneHeroTokenizer
from training.collator import ChartCollator
from training.metrics import (
    compute_token_accuracy,
    compute_perplexity,
    compute_note_f1,
    compute_timing_accuracy,
    compute_sequence_accuracy,
    decode_to_note_blocks,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def section(name: str):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print('─' * 60)


def ok(msg: str):
    print(f"  {PASS}  {msg}")


def check(condition: bool, msg: str):
    if condition:
        ok(msg)
    else:
        print(f"  {FAIL}  {msg}")
        raise AssertionError(msg)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def make_fake_batch(B: int = 2, T: int = 256, seq_len: int = 50, device="cpu"):
    """Build a fake collated batch with the shapes the model expects.

    decoder_input_ids  = token_ids[:, :-1]  → shape (B, seq_len)
    decoder_labels     = token_ids[:, 1:]   → shape (B, seq_len)
    logits output      → shape (B, seq_len, vocab_size)
    """
    tokenizer = CloneHeroTokenizer()
    bos = tokenizer.vocab["<BOS>"]
    eos = tokenizer.vocab["<EOS>"]
    # seq_len + 1 tokens so that input/labels are both exactly seq_len
    token_ids = torch.randint(4, tokenizer.vocab_size, (B, seq_len + 1))
    token_ids[:, 0] = bos
    token_ids[:, -1] = eos

    return {
        "log_mel": torch.randn(B, 512, T),
        "enc_padding_mask": torch.zeros(B, 512, dtype=torch.bool),
        "mert_emb": torch.randn(B, 768),
        "bpm": torch.full((B, 1), 120.0),
        "ts": torch.full((B, 1), 4.0),
        "resolution": torch.full((B, 1), 192.0),
        "offset": torch.zeros(B, 1),
        "instrument_idx": torch.zeros(B, dtype=torch.long),
        "difficulty_idx": torch.zeros(B, dtype=torch.long),
        "decoder_input_ids": token_ids[:, :-1].to(device),
        "decoder_labels": token_ids[:, 1:].to(device),
        "decoder_attention_mask": torch.ones(B, seq_len, dtype=torch.long),
    }


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tokenizer():
    section("Tokenizer")
    tok = CloneHeroTokenizer()

    check(tok.vocab_size == 693, f"vocab_size == 693  (got {tok.vocab_size})")

    # Encode a short chart
    beat_seq = [
        (0, "normal", 2, 0, 1, 0),
        (16, "normal", 0, 0, 0, 16),
        (0, "special", 3, 0, 2, 8),
    ]
    ids = tok.encode_complete_chart("<Guitar>", "<Expert>", beat_seq)
    check(len(ids) > 0, f"encode_complete_chart returns {len(ids)} tokens")
    check(ids[0] == tok.vocab["<BOS>"], "first token is BOS")
    check(ids[-1] == tok.vocab["<EOS>"], "last token is EOS")

    # Round-trip decode
    blocks = decode_to_note_blocks(ids)
    check(len(blocks) == len(beat_seq), f"decoded {len(blocks)} note blocks (expected {len(beat_seq)})")


def test_cnn_frontend(device: str):
    section("AudioCNNFrontEnd")
    from models.cnn_frontend import AudioCNNFrontEnd

    model = AudioCNNFrontEnd(d_model=768).to(device)
    x = torch.randn(2, 512, 256).to(device)  # [B, n_mels, T]
    t0 = time.time()
    out = model(x)
    elapsed = time.time() - t0

    check(out.shape == (2, 256 // 16, 768), f"output shape {out.shape} == (2, 16, 768)")
    check(not out.isnan().any(), "no NaN in output")
    ok(f"forward in {elapsed*1000:.1f} ms")


def test_conditioning(device: str):
    section("ConditioningEncoder")
    from models.conditioning import ConditioningEncoder

    enc = ConditioningEncoder(d_model=768).to(device)
    B = 3
    out = enc(
        mert_emb=torch.randn(B, 768).to(device),
        bpm=torch.full((B, 1), 120.0).to(device),
        ts=torch.full((B, 1), 4.0).to(device),
        resolution=torch.full((B, 1), 192.0).to(device),
        offset=torch.zeros(B, 1).to(device),
        instrument_idx=torch.tensor([0, 1, 2]).to(device),
        difficulty_idx=torch.tensor([0, 1, 2]).to(device),
    )
    check(out.shape == (B, 7, 768), f"prefix shape {out.shape} == ({B}, 7, 768)")
    check(not out.isnan().any(), "no NaN in prefix tokens")


def test_model_forward(device: str):
    section("ChartTransformer — forward pass")
    cfg = ModelConfig(enc_layers=2, dec_layers=2, enc_ckpt_every=0, dec_ckpt_every=0)
    model = ChartTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"Model parameters: {n_params/1e6:.1f}M")

    batch = move_batch(make_fake_batch(B=2, T=128, seq_len=30), device)
    t0 = time.time()
    out = model(batch)
    elapsed = time.time() - t0

    check("logits" in out and "loss" in out, "output has logits and loss keys")
    check(out["logits"].shape == (2, 30, cfg.vocab_size),
          f"logits shape {out['logits'].shape} == (2, 30, {cfg.vocab_size})")
    check(not out["logits"].isnan().any(), "no NaN in logits")
    check(not out["loss"].isnan(), f"loss is not NaN  (loss={out['loss'].item():.4f})")

    # Sanity: initial loss should be close to log(vocab_size) ≈ 6.54
    import math
    expected_loss = math.log(cfg.vocab_size)
    check(
        abs(out["loss"].item() - expected_loss) < 2.0,
        f"initial loss {out['loss'].item():.4f} ≈ log({cfg.vocab_size})={expected_loss:.4f}  (±2.0)"
    )
    ok(f"forward in {elapsed*1000:.1f} ms")


def test_model_encode_decode(device: str):
    section("ChartTransformer — encode / decode separately")
    cfg = ModelConfig(enc_layers=2, dec_layers=2, enc_ckpt_every=0, dec_ckpt_every=0)
    model = ChartTransformer(cfg).eval().to(device)

    B, T, S = 1, 128, 20
    log_mel = torch.randn(B, 512, T).to(device)
    mert_emb = torch.randn(B, 768).to(device)
    enc_mask = torch.zeros(B, 512, dtype=torch.bool).to(device)

    with torch.no_grad():
        enc_out = model.encode(
            log_mel=log_mel, mert_emb=mert_emb,
            bpm=torch.full((B, 1), 120.0).to(device),
            ts=torch.full((B, 1), 4.0).to(device),
            resolution=torch.full((B, 1), 192.0).to(device),
            offset=torch.zeros(B, 1).to(device),
            instrument_idx=torch.zeros(B, dtype=torch.long).to(device),
            difficulty_idx=torch.zeros(B, dtype=torch.long).to(device),
            enc_padding_mask=enc_mask,
        )
        check(enc_out.shape == (B, 512, cfg.d_model),
              f"enc_out shape {enc_out.shape} == ({B}, 512, {cfg.d_model})")

        tgt = torch.randint(4, cfg.vocab_size, (B, S)).to(device)
        logits = model.decode(tgt, enc_out, enc_mask)
        check(logits.shape == (B, S, cfg.vocab_size),
              f"logits shape {logits.shape} == ({B}, {S}, {cfg.vocab_size})")


def test_collator():
    section("ChartCollator")
    tok = CloneHeroTokenizer()
    bos = tok.vocab["<BOS>"]
    eos = tok.vocab["<EOS>"]

    beat_seq = [(i % 32, "normal", i % 5, 0, i % 10, 0) for i in range(20)]
    token_ids = torch.tensor(tok.encode_complete_chart("<Guitar>", "<Expert>", beat_seq))

    items = [
        {
            "log_mel": torch.randn(512, 300),
            "token_ids": token_ids,
            "mert_emb": torch.randn(768),
            "bpm": torch.tensor([120.0]),
            "ts": torch.tensor([4.0]),
            "resolution": torch.tensor([192.0]),
            "offset": torch.tensor([0.0]),
            "instrument": "Single",
            "difficulty": "Expert",
        },
        {
            "log_mel": torch.randn(512, 150),   # shorter clip
            "token_ids": token_ids[:30],          # shorter sequence
            "mert_emb": torch.randn(768),
            "bpm": torch.tensor([90.0]),
            "ts": torch.tensor([3.0]),
            "resolution": torch.tensor([192.0]),
            "offset": torch.tensor([0.5]),
            "instrument": "Drums",
            "difficulty": "Hard",
        },
    ]

    collator = ChartCollator()
    batch = collator(items)

    check(batch["log_mel"].shape == (2, 512, 8192), f"log_mel shape {batch['log_mel'].shape}")
    check(batch["enc_padding_mask"].shape == (2, 512), f"enc_padding_mask shape {batch['enc_padding_mask'].shape}")
    check(batch["enc_padding_mask"].dtype == torch.bool, "enc_padding_mask is bool")
    check(batch["mert_emb"].shape == (2, 768), f"mert_emb shape {batch['mert_emb'].shape}")
    check(batch["decoder_input_ids"].shape[0] == 2, "decoder_input_ids batch dim == 2")
    check(batch["decoder_labels"].shape == batch["decoder_input_ids"].shape, "labels same shape as input_ids")
    check((batch["decoder_labels"] == -100).any(), "labels contain -100 for PAD positions")
    check(batch["instrument_idx"].tolist() == [0, 2], f"instrument_idx [0,2] (got {batch['instrument_idx'].tolist()})")
    check(batch["difficulty_idx"].tolist() == [0, 1], f"difficulty_idx [0,1] (got {batch['difficulty_idx'].tolist()})")

    # Short clip should have masked positions
    check(batch["enc_padding_mask"][1].any(), "short clip has masked encoder positions")
    # Full-length clip (300 frames → 18 audio tokens, < 505) should also have masked tail
    check(batch["enc_padding_mask"][0].any(), "first clip has masked encoder tail")


def test_metrics():
    section("Metrics")

    # Token accuracy
    logits = torch.zeros(2, 10, 693)
    labels = torch.zeros(2, 10, dtype=torch.long)
    logits[:, :, 0] = 10.0  # always predict token 0
    acc = compute_token_accuracy(logits, labels)
    check(abs(acc - 1.0) < 1e-6, f"perfect prediction → token_acc=1.0 (got {acc:.4f})")

    # Perplexity
    ppl = compute_perplexity(0.0)
    check(abs(ppl - 1.0) < 1e-6, "loss=0 → ppl=1.0")

    # Note F1 — perfect prediction
    tok = CloneHeroTokenizer()
    beat_seq = [(0, "normal", 2, 0, 1, 0), (0, "normal", 3, 0, 2, 0)]
    ids = tok.encode_complete_chart("<Guitar>", "<Expert>", beat_seq)
    result = compute_note_f1(ids, ids)
    check(abs(result["f1"] - 1.0) < 1e-4, f"identical sequences → Note F1=1.0 (got {result['f1']:.4f})")

    # Sequence accuracy
    acc = compute_sequence_accuracy([ids], [ids])
    check(abs(acc - 1.0) < 1e-6, "identical sequences → seq_acc=1.0")

    # Timing accuracy
    t_acc = compute_timing_accuracy(ids, ids)
    check(abs(t_acc - 1.0) < 1e-4, f"identical sequences → timing_acc=1.0 (got {t_acc:.4f})")


def test_training_step(device: str):
    section("Training step (backward + optimizer)")
    cfg = ModelConfig(enc_layers=2, dec_layers=2, enc_ckpt_every=0, dec_ckpt_every=0)
    model = ChartTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = move_batch(make_fake_batch(B=2, T=64, seq_len=20), device)
    if device == "cuda":
        # Use autocast for bf16/fp16
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
    else:
        out = model(batch)

    loss_before = out["loss"].item()
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Second forward pass — loss should change (parameters updated)
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out2 = model(batch)
    else:
        out2 = model(batch)
    loss_after = out2["loss"].item()

    check(not out["loss"].isnan(), f"loss before step is not NaN ({loss_before:.4f})")
    check(loss_before != loss_after, f"loss changed after optimizer step ({loss_before:.4f} → {loss_after:.4f})")
    ok(f"backward + optimizer step OK  ({loss_before:.4f} → {loss_after:.4f})")


def test_save_load(device: str, tmp_path: str = None):
    section("Model save / load")
    import tempfile, os
    if tmp_path is None:
        tmp_path = os.path.join(tempfile.gettempdir(), "charter_test.pt")

    cfg = ModelConfig(enc_layers=1, dec_layers=1, enc_ckpt_every=0, dec_ckpt_every=0)
    model = ChartTransformer(cfg).to(device)
    model.save(tmp_path)
    ok(f"saved to {tmp_path}")

    loaded = ChartTransformer.load(tmp_path, map_location=device)
    loaded.eval().to(device)

    batch = move_batch(make_fake_batch(B=1, T=64, seq_len=10), device)
    with torch.no_grad():
        out1 = model(batch)
        out2 = loaded(batch)

    diff = (out1["logits"] - out2["logits"]).abs().max().item()
    check(diff < 1e-5, f"loaded model produces identical output (max diff={diff:.2e})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu | cuda (auto-detect if omitted)")
    parser.add_argument("--fast", action="store_true", help="Skip training step test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning smoke tests on device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tests = [
        ("Tokenizer", lambda: test_tokenizer()),
        ("CNN Frontend", lambda: test_cnn_frontend(device)),
        ("Conditioning", lambda: test_conditioning(device)),
        ("Model Forward", lambda: test_model_forward(device)),
        ("Model Encode/Decode", lambda: test_model_encode_decode(device)),
        ("Collator", lambda: test_collator()),
        ("Metrics", lambda: test_metrics()),
        ("Save/Load", lambda: test_save_load(device)),
    ]

    if not args.fast:
        tests.append(("Training Step", lambda: test_training_step(device)))

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  {FAIL}  TEST FAILED: {name}")
            print(f"       {e}")
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print('═' * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
