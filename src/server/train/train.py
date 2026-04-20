"""
train.py — Training engine for Scout (modular architecture)

Handles:
• dataset loading
• model initialization
• training loop
• checkpointing
• validation
"""

import csv
import datetime
import logging
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

import config
from .data import build_dataloader
from ai_clients.tokenizer import load_tokenizer
from corpus.dataset_repository import DatasetRepository
from corpus.models.packed_chunk_dataset import PackedChunkDataset
from model.loader import (
    load_checkpoint,
    load_fresh_model,
)
from model.model import count_parameters


logger = logging.getLogger(config.LOGGER_NAME)


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps, min_lr):
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(out_dir, step, model, optimizer, scheduler, cfg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / f"model_{step}.pt"
    latest_path = out_dir / "latest.pt"

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
    }

    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)

    logger.info("Saved checkpoint: %s", ckpt_path)


def try_resume_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    latest = Path(checkpoint_dir) / "latest.pt"

    if not latest.exists():
        logger.info("No checkpoint found — starting fresh.")
        return 0

    logger.info("Resuming from checkpoint: %s", latest)

    checkpoint, state = load_checkpoint(latest, model, device)

    step = checkpoint.get("step", 0)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            logger.warning("Scheduler state incompatible — resetting.")

    logger.info("Resumed from step %d", step)

    return step


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validation_loss(model, loader, device, num_batches: int = 20):
    """
    Compute validation loss using a DataLoader built from the
    dataset's validation split.
    """

    model.eval()

    total_loss = 0.0
    count = 0

    for batch in loader:
        if count >= num_batches:
            break

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        total_loss += loss.item()
        count += 1

    model.train()

    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────

def get_now():
    return datetime.datetime.now(datetime.UTC)


def build_log_path(prefix="training"):
    today = get_now().strftime("%Y-%m-%d")
    idx = 1

    while True:
        path = config.TRAINING_LOG_DIR / f"{prefix}_{today}_{idx}.csv"

        if not path.exists():
            return path

        idx += 1


# ─────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────

def run_training(
    dataset_name: str,
    model_config: dict,
    batch_size: int,
    max_steps: int,
    lr: float = None,
    min_lr: float = None,
    warmup_steps: int = None,
    reset_optimizer: bool = False,
    stop_flag: list = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_dir = Path(config.CHECKPOINT_DIR)

    tokenizer = load_tokenizer()
    logger.info("Tokenizer loaded.")

    # --------------------------------------------------
    # CSV logging
    # --------------------------------------------------

    metrics_path = build_log_path()

    csv_file = open(metrics_path, "a", newline="")

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "step",
            "loss",
            "avg_loss",
            "lr",
            "val_loss",
            "elapsed",
            "tokens_per_sec",
            "eta",
        ],
    )

    if csv_file.tell() == 0:
        csv_writer.writeheader()
    logger.info("CSV log file ready.")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    repo = DatasetRepository()

    dataset_model = repo.get_dataset(dataset_name)
    if not dataset_model.is_tokenized():
        dataset_model.tokenize()
    
    train_dataset = dataset_model.get_tokenized("train")

    val_dataset = None
    if "validation" in dataset_model.get_split_names():
        val_dataset = dataset_model.get_tokenized("validation")

    logger.info("Begin packing dataset.")
    train_dataset = PackedChunkDataset(
        train_dataset,
        block_size=model_config["block_size"],
    )
    logger.info("Done packing dataset.")

    logger.info("Setting up data loader.")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    logger.info("Done setting up data loader.")

    val_loader = None
    if val_dataset is not None:

        val_dataset = PackedChunkDataset(
            val_dataset,
            block_size=model_config["block_size"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

    data_iter = iter(loader)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = load_fresh_model(device, model_config)

    param_stats = count_parameters(model)

    logger.info(
        "Model parameters: %d (%fM)",
        param_stats["total"],
        param_stats["total"] / 1e6,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    effective_lr = lr if lr is not None else config.LEARNING_RATE
    effective_min_lr = min_lr if min_lr is not None else config.MIN_LR
    effective_warmup = warmup_steps if warmup_steps is not None else config.WARMUP_STEPS

    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = build_scheduler(
        optimizer,
        total_steps=max_steps,
        warmup_steps=effective_warmup,
        min_lr=effective_min_lr,
    )

    # --------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------

    if reset_optimizer:
        # Load model weights only — discard optimizer/scheduler state.
        # Use this when fine-tuning from a checkpoint trained on a different
        # corpus so stale momentum doesn't interfere with the new signal.
        latest = Path(checkpoint_dir) / "latest.pt"
        if latest.exists():
            logger.info("Loading model weights only (--reset-optimizer): %s", latest)
            load_checkpoint(latest, model, device)
            start_step = 0
        else:
            logger.info("No checkpoint found — starting fresh.")
            start_step = 0
    else:
        start_step = try_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_dir,
            device,
        )

    model = torch.compile(model)
    model.train()

    step = start_step

    start_time = time.time()
    last_log_time = start_time
    accum_loss = 0.0

    logger.info("Training from step %d → %d", start_step, max_steps)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    while step < max_steps and not (stop_flag and stop_flag[0]):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        accum_loss += loss.item()

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if step % config.LOG_INTERVAL == 0:
            now = time.time()

            elapsed = now - start_time
            interval = now - last_log_time
            last_log_time = now

            step_time = interval / max(config.LOG_INTERVAL, 1)

            tokens_per_step = model_config["block_size"] * batch_size

            tokens_per_sec = (
                tokens_per_step * config.LOG_INTERVAL
            ) / max(interval, 1e-6)

            remaining_steps = max_steps - step
            eta = remaining_steps * step_time

            current_lr = scheduler.get_last_lr()[0]

            avg_loss = accum_loss / max(step - start_step + 1, 1)

            val_loss = None

            if (
                val_loader is not None
                and step > 0
                and step % (config.LOG_INTERVAL * 10) == 0
            ):
                val_loss = validation_loss(model, val_loader, device)

            metrics = {
                "step": step,
                "loss": loss.item(),
                "avg_loss": avg_loss,
                "lr": current_lr,
                "val_loss": val_loss,
                "elapsed": elapsed,
                "tokens_per_sec": tokens_per_sec,
                "eta": eta,
            }

            csv_writer.writerow(metrics)
            csv_file.flush()

            logger.info(
                "step %d | loss %.4f | lr %.6f | %.0f tok/s",
                step,
                loss.item(),
                current_lr,
                tokens_per_sec,
            )

            yield metrics

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        if step % config.SAVE_INTERVAL == 0 and step > start_step:

            save_checkpoint(
                checkpoint_dir,
                step,
                model,
                optimizer,
                scheduler,
                {
                    "vocab_size": tokenizer.vocab_size,
                    "block_size": model_config["block_size"],
                },
            )

        step += 1

    logger.info("Training complete.")

    csv_file.close()

    save_checkpoint(
        checkpoint_dir,
        step,
        model,
        optimizer,
        scheduler,
        {},
    )