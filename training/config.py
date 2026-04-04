from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models.chart_transformer import ModelConfig


@dataclass
class TrainingConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    model: ModelConfig = field(default_factory=ModelConfig)

    # ── Hardware ─────────────────────────────────────────────────────────────
    n_gpus: int = 8
    mixed_precision: str = "bf16"   # "bf16" | "fp16" | "no"

    # ── Optimization ─────────────────────────────────────────────────────────
    per_gpu_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    # effective_batch = per_gpu_batch_size × gradient_accumulation_steps × n_gpus = 256

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-8
    clip_grad_norm: float = 1.0

    # ── Schedule ─────────────────────────────────────────────────────────────
    num_epochs: int = 150
    warmup_steps: int = 500         # steps (not epochs)
    lr_scheduler: str = "cosine"    # "cosine" | "linear"

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_path: str = "clone_hero_dataset_final"
    val_fraction: float = 0.05      # fraction of unique songs held out
    train_val_seed: int = 42
    dataloader_num_workers: int = 4

    # Collator
    enc_max_frames: int = 8192      # log-mel time frames padded to this (→ 512 audio tokens)
    dec_max_len: int = 2048

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_dir: str = "model_checkpoints"
    save_every_n_steps: int = 500
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "clone-hero-charter"
    wandb_entity: Optional[str] = None

    # ── Debug ────────────────────────────────────────────────────────────────
    debug: bool = False             # single-GPU, limited steps
    debug_steps: int = 10
