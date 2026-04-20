"""AutoCharterTrainer — Accelerate-based training loop with early stopping.

Supports:
  - Single GPU (RTX 5070 Ti, H100)
  - Multi-GPU (8× H100 with DDP via accelerate)
  - Mixed precision: bf16 (H100) or fp16 (RTX)
  - TensorBoard logging (default) or W&B (--use-wandb flag)
  - Early stopping on validation loss
  - Cosine LR schedule with warmup
  - Gradient accumulation

Usage:
    accelerate launch scripts/train.py --dataset ./data --output-dir ./runs/run1
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from auto_charter.model.charter_model import AutoCharterModel
from auto_charter.model.config import AutoCharterConfig
from auto_charter.training import metrics as M
from auto_charter.training.collator import AutoCharterTrainCollator
from auto_charter.training.dataset import AutoCharterDataset, ShardGroupedSampler, ShardIndexedDataset


@dataclass
class EarlyStoppingState:
    patience: int = 10
    best_val_loss: float = float("inf")
    no_improve_count: int = 0
    best_step: int = 0

    @property
    def should_stop(self) -> bool:
        return self.no_improve_count >= self.patience

    def update(self, val_loss: float, step: int) -> bool:
        """Returns True if this is a new best."""
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            self.best_step = step
            return True
        self.no_improve_count += 1
        return False


class AutoCharterTrainer:
    def __init__(
        self,
        model: AutoCharterModel,
        train_dataset: AutoCharterDataset,
        val_dataset: AutoCharterDataset,
        output_dir: Path | str,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        use_wandb: bool = False,
        eval_every_n_steps: int = 0,   # 0 = evaluate every epoch
        save_every_n_steps: int = 0,   # 0 = save only on best
        log_every_n_steps: int = 10,
        num_workers: int = 0,
        mixed_precision: str = "bf16",  # "bf16", "fp16", or "no"
        resume_from: Optional[Path] = None,
        steps_per_epoch: int = 0,      # required when train_dataset is IterableDataset
        seed: int = 42,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import ProjectConfiguration

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_with = "wandb" if use_wandb else "tensorboard"
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with=log_with,
            project_config=ProjectConfiguration(project_dir=str(self.output_dir)),
        )

        self.model = model
        self.config = model.config
        self.num_epochs = num_epochs
        self.log_every = log_every_n_steps
        self.eval_every = eval_every_n_steps
        self.save_every = save_every_n_steps
        self.early_stopping = EarlyStoppingState(patience=early_stopping_patience)

        collator = AutoCharterTrainCollator(
            max_tokens=self.config.max_seq_len,
            max_beats=self.config.max_beats,
            mert_dim=self.config.mert_dim,
        )

        self._raw_train_dataset = train_dataset  # keep ref for evict_all / set_epoch

        _train_is_streaming = isinstance(train_dataset, IterableDataset)
        _is_shard_indexed = isinstance(train_dataset, ShardIndexedDataset)

        if _is_shard_indexed:
            self._shard_sampler = ShardGroupedSampler(
                train_dataset, shuffle=True, seed=seed if seed is not None else 42
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=self._shard_sampler,
                collate_fn=collator,
                num_workers=0,       # must be 0: shard cache is not fork-safe
                pin_memory=False,
            )
        else:
            self._shard_sampler = None
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=not _train_is_streaming,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=False,
            )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )

        # Optimizer (no weight decay on bias / LayerNorm)
        no_decay = {"bias", "layer_norm.weight", "LayerNorm.weight"}
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.config.learning_rate)

        if steps_per_epoch > 0:
            total_steps = steps_per_epoch * num_epochs // grad_accum_steps
        else:
            total_steps = len(self.train_loader) * num_epochs // grad_accum_steps
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max(total_steps, self.config.warmup_steps + 1),
        )

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

        self.global_step = 0
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

        # Init trackers (only on main process)
        if self.accelerator.is_main_process:
            run_name = self.output_dir.name
            self.accelerator.init_trackers(
                project_name="auto-charter",
                config=self.config.to_dict(),
                init_kwargs={"wandb": {"name": run_name}} if use_wandb else {},
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        if self.accelerator.is_main_process:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Training AutoCharterModel ({n_params:,} parameters)")
            try:
                print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
            except TypeError:
                print("Train batches: (streaming — unknown), Val batches: (streaming — unknown)")

        for epoch in range(self.num_epochs):
            if self._shard_sampler is not None:
                self._shard_sampler.set_epoch(epoch)

            epoch_loss = self._train_epoch(epoch)

            # Release shard cache between epochs so next epoch starts fresh
            if isinstance(self._raw_train_dataset, ShardIndexedDataset):
                self._raw_train_dataset.evict_all()

            # Validate every epoch (or at specified step interval)
            val_metrics = self._validate()
            val_loss = val_metrics["val/loss"]

            if self.accelerator.is_main_process:
                self.accelerator.log(
                    {**val_metrics, "epoch": epoch},
                    step=self.global_step,
                )
                improved = self.early_stopping.update(val_loss, self.global_step)
                status = "✓ improved" if improved else f"no improvement ({self.early_stopping.no_improve_count}/{self.early_stopping.patience})"
                print(
                    f"Epoch {epoch:3d} | train_loss={epoch_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | {status}"
                )
                if improved:
                    self._save_checkpoint("best")

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

            # Check early stopping (broadcast from main to all processes)
            should_stop = torch.tensor(
                int(self.early_stopping.should_stop),
                device=self.accelerator.device,
            )
            if self.accelerator.num_processes > 1:
                torch.distributed.broadcast(should_stop, src=0)
            if should_stop.item():
                if self.accelerator.is_main_process:
                    print(f"Early stopping at epoch {epoch} (best step: {self.early_stopping.best_step})")
                break

        if self.accelerator.is_main_process:
            self.accelerator.end_training()

    # ── Private methods ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        from tqdm import tqdm

        self.model.train()
        total_loss = 0.0
        n_steps = 0

        try:
            total_batches = len(self.train_loader)
        except TypeError:
            total_batches = None

        pbar = tqdm(
            self.train_loader,
            total=total_batches,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
        )

        for batch in pbar:
            with self.accelerator.accumulate(self.model):
                output = self.model(
                    mert_embeddings=batch["mert_embeddings"],
                    logmel_frames=batch["logmel_frames"],
                    bpm_at_beat=batch["bpm_at_beat"],
                    time_sig_num=batch["time_sig_num"],
                    time_sig_den=batch["time_sig_den"],
                    beat_duration_s=batch["beat_duration_s"],
                    beat_padding_mask=batch["beat_attention_mask"],
                    input_ids=batch["input_ids"],
                    beat_ids=batch["beat_ids"],
                    instrument_ids=batch["instrument_ids"],
                    difficulty_ids=batch["difficulty_ids"],
                    labels=batch["labels"],
                )
                loss = output.loss
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if (
                        self.accelerator.is_main_process
                        and self.global_step % self.log_every == 0
                    ):
                        tok_acc = M.token_accuracy(
                            output.logits.detach().float(),
                            batch["labels"],
                        )
                        ppl = M.perplexity(
                            output.logits.detach().float(),
                            batch["labels"],
                        )
                        lr_now = self.scheduler.get_last_lr()[0]
                        self.accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/token_accuracy": tok_acc,
                                "train/perplexity": ppl,
                                "train/lr": lr_now,
                            },
                            step=self.global_step,
                        )

            total_loss += loss.detach().float().item()
            n_steps += 1
            if self.accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss.detach().float().item():.4f}", step=self.global_step)

        pbar.close()

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tok_acc = 0.0
        total_ppl = 0.0
        n_steps = 0

        for batch in self.val_loader:
            output = self.model(
                mert_embeddings=batch["mert_embeddings"],
                logmel_frames=batch["logmel_frames"],
                bpm_at_beat=batch["bpm_at_beat"],
                time_sig_num=batch["time_sig_num"],
                time_sig_den=batch["time_sig_den"],
                beat_duration_s=batch["beat_duration_s"],
                beat_padding_mask=batch["beat_attention_mask"],
                input_ids=batch["input_ids"],
                beat_ids=batch["beat_ids"],
                instrument_ids=batch["instrument_ids"],
                difficulty_ids=batch["difficulty_ids"],
                labels=batch["labels"],
            )
            total_loss += output.loss.float().item()
            total_tok_acc += M.token_accuracy(output.logits.float(), batch["labels"])
            total_ppl += M.perplexity(output.logits.float(), batch["labels"])
            n_steps += 1

        n = max(n_steps, 1)
        return {
            "val/loss": total_loss / n,
            "val/token_accuracy": total_tok_acc / n,
            "val/perplexity": total_ppl / n,
        }

    def _save_checkpoint(self, tag: str) -> None:
        save_dir = self.output_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(save_dir)
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.early_stopping.best_val_loss,
        }
        with open(save_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        print(f"  Saved checkpoint → {save_dir}")

    def _load_checkpoint(self, path: Path) -> None:
        state_path = path / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.early_stopping.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"Resumed from {path} at step {self.global_step}")
