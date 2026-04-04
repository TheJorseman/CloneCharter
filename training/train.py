"""
Main training loop for CloneCharter transformer.

Launch command (8 GPUs):
    accelerate launch --num_processes 8 --mixed_precision bf16 training/train.py

Single-GPU debug:
    python training/train.py --debug
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from models.chart_transformer import ChartTransformer
from training.config import TrainingConfig
from training.collator import ChartCollator
from training.dataset import build_splits
from training.metrics import compute_token_accuracy, compute_validation_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--debug", action="store_true", help="Single-GPU debug run")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def save_checkpoint(accelerator: Accelerator, step: int, checkpoint_dir: str, keep: int = 3):
    path = Path(checkpoint_dir) / f"step_{step:07d}"
    accelerator.save_state(str(path))
    accelerator.print(f"Checkpoint saved: {path}")

    # Remove old checkpoints
    ckpts = sorted(Path(checkpoint_dir).glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in ckpts[:-keep]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)


def run_validation(model, val_loader, accelerator, global_step: int, config: TrainingConfig):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            loss = outputs["loss"]
            total_loss += accelerator.gather(loss).mean().item()
            n_batches += 1

            # Gather logits and labels for metric computation (limit to avoid OOM)
            if len(all_logits) < 50:
                all_logits.append(outputs["logits"].detach().cpu())
                all_labels.append(batch["decoder_labels"].detach().cpu())

    metrics = compute_validation_metrics(all_logits, all_labels, total_loss, n_batches)

    if accelerator.is_main_process:
        accelerator.print(
            f"[val step={global_step}] "
            f"loss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.2f}  "
            f"tok_acc={metrics['token_accuracy']:.4f}  note_f1={metrics['note_f1']:.4f}  "
            f"timing_acc={metrics['timing_accuracy']:.4f}  "
            f"seq_acc={metrics['sequence_accuracy']:.4f}"
        )
        if accelerator.log_with:
            accelerator.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

    model.train()
    return metrics


def main():
    args = parse_args()
    cfg = TrainingConfig()

    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.resume:
        cfg.resume_from_checkpoint = args.resume
    if args.debug:
        cfg.debug = True

    log_with = None if args.no_wandb or not cfg.use_wandb else "wandb"

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision if not cfg.debug else "no",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=cfg.checkpoint_dir,
    )
    set_seed(cfg.train_val_seed)

    if accelerator.is_main_process:
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if log_with == "wandb":
            accelerator.init_trackers(
                project_name=cfg.wandb_project,
                config=vars(cfg),
            )

    # ── Dataset & DataLoaders ───────────────────────────────────────────────
    accelerator.print("Loading dataset...")
    train_ds, val_ds = build_splits(cfg.dataset_path, cfg.val_fraction, cfg.train_val_seed)
    accelerator.print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    collator = ChartCollator(
        enc_max_frames=cfg.enc_max_frames,
        dec_max_len=cfg.dec_max_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # ── Model & Optimizer ───────────────────────────────────────────────────
    accelerator.print("Building model...")
    model = ChartTransformer(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Parameters: {n_params / 1e6:.1f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    total_steps = steps_per_epoch * (cfg.debug_steps if cfg.debug else cfg.num_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ──────────────────────────────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Resume from checkpoint
    global_step = 0
    if cfg.resume_from_checkpoint:
        accelerator.load_state(cfg.resume_from_checkpoint)
        global_step = int(Path(cfg.resume_from_checkpoint).name.split("_")[1])
        accelerator.print(f"Resumed from step {global_step}")

    # ── Training loop ───────────────────────────────────────────────────────
    accelerator.print("Starting training...")
    model.train()

    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(batch)
                loss = outputs["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
                        gathered_loss = accelerator.gather(loss).mean().item()
                        tok_acc = compute_token_accuracy(
                            outputs["logits"].detach(), batch["decoder_labels"]
                        )
                        lr = scheduler.get_last_lr()[0]
                        accelerator.print(
                            f"epoch={epoch+1} step={global_step}  "
                            f"loss={gathered_loss:.4f}  tok_acc={tok_acc:.4f}  lr={lr:.2e}"
                        )
                        if log_with == "wandb":
                            accelerator.log(
                                {"train/loss": gathered_loss, "train/token_accuracy": tok_acc, "train/lr": lr},
                                step=global_step,
                            )

                    # Validation
                    if global_step % cfg.eval_every_n_steps == 0:
                        run_validation(model, val_loader, accelerator, global_step, cfg)

                    # Checkpoint
                    if global_step % cfg.save_every_n_steps == 0:
                        save_checkpoint(accelerator, global_step, cfg.checkpoint_dir, cfg.keep_last_n_checkpoints)

                    # Debug: stop after N steps
                    if cfg.debug and global_step >= cfg.debug_steps:
                        accelerator.print("Debug run complete.")
                        accelerator.end_training()
                        return

    # Final checkpoint
    save_checkpoint(accelerator, global_step, cfg.checkpoint_dir, cfg.keep_last_n_checkpoints)
    accelerator.print(f"Training complete. Total steps: {global_step}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
