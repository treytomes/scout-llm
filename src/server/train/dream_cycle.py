"""
train/dream_cycle.py

Dream cycle: trains Scout on a single conversation, then locks it permanently.

Two passes:
  1. SFT (supervised fine-tune) on the full post-edit conversation transcript.
  2. DPO on any assistant messages that have edit history (edits[-1] = rejected,
     current content = chosen).

After training completes, sets conversation status → locked.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(config.LOGGER_NAME)


# ─────────────────────────────────────────────────────────────
# Helpers — conversation → training data
# ─────────────────────────────────────────────────────────────

def _build_sft_text(messages: list[dict]) -> str:
    """Format conversation as the same [Speaker] prompt the model was trained on."""
    parts = []
    for msg in messages:
        speaker = msg.get("user_name", "Trey") if msg["role"] == "user" else "Scout"
        parts.append(f"[{speaker}] {msg['content']}")
    return "\n\n".join(parts)


def _extract_dpo_pairs(messages: list[dict]) -> list[dict]:
    """
    Return list of {prompt, chosen, rejected} dicts.

    prompt  — conversation up to (but not including) the assistant turn
    chosen  — current content of the assistant message
    rejected — the content before the most recent edit (edits[-1])
    """
    pairs = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        edits = msg.get("edits", [])
        if not edits:
            continue
        prompt_parts = []
        for prev in messages[:i]:
            speaker = prev.get("user_name", "Trey") if prev["role"] == "user" else "Scout"
            prompt_parts.append(f"[{speaker}] {prev['content']}")
        prompt_parts.append("[Scout]")
        pairs.append({
            "prompt": "\n\n".join(prompt_parts),
            "chosen": msg["content"],
            "rejected": edits[-1]["content"],
        })
    return pairs


# ─────────────────────────────────────────────────────────────
# Inline dataset
# ─────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset


class _SFTDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int):
        ids = tokenizer.encode(text)
        # Build overlapping blocks so a short conversation still produces samples
        self.blocks = []
        stride = max(1, block_size // 2)
        for start in range(0, max(1, len(ids) - block_size + 1), stride):
            chunk = ids[start:start + block_size]
            if len(chunk) < 16:
                continue
            if len(chunk) < block_size:
                chunk = chunk + [tokenizer.eos_token_id] * (block_size - len(chunk))
            self.blocks.append(chunk)
        if not self.blocks:
            # Conversation shorter than block_size — use the whole thing once
            padded = ids + [tokenizer.eos_token_id] * max(0, block_size - len(ids))
            self.blocks.append(padded[:block_size])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


class _DPODataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, block_size: int):
        self.samples = []
        for p in pairs:
            def _enc(text):
                ids = tokenizer.encode(text)
                if len(ids) < block_size:
                    ids = ids + [tokenizer.eos_token_id] * (block_size - len(ids))
                return ids[:block_size]
            self.samples.append({
                "prompt_ids":   torch.tensor(_enc(p["prompt"]),   dtype=torch.long),
                "chosen_ids":   torch.tensor(_enc(p["prompt"] + " " + p["chosen"]), dtype=torch.long),
                "rejected_ids": torch.tensor(_enc(p["prompt"] + " " + p["rejected"]), dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────

import torch.nn.functional as F


def _log_probs(model, ids: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs for the token sequence under the model."""
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
        logits = model(x)
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(2, y.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def _dpo_loss(model, chosen_ids, rejected_ids, beta=0.1):
    lp_chosen   = _log_probs(model, chosen_ids)
    lp_rejected = _log_probs(model, rejected_ids)
    return -F.logsigmoid(beta * (lp_chosen - lp_rejected)).mean()


# ─────────────────────────────────────────────────────────────
# Dream cycle job
# ─────────────────────────────────────────────────────────────

class DreamCycleJob(threading.Thread):
    """
    Runs SFT + optional DPO on a single conversation, then locks it.

    Steps are derived from dataset size × epoch targets so training
    intensity is consistent regardless of conversation length.

    LR rationale:
      sft_lr=1e-5  — below the min_lr used in main training (3e-5); safe for
                     a small nudge without disturbing the base weights.
      dpo_lr=1e-7  — matches v1 DPO conservatism; with only a handful of pairs
                     the gradient is perfectly correlated, so lr matters more.
      beta=0.05    — v1 recommendation; lower beta = softer preference push.
    """

    def __init__(
        self,
        conversation_id: str,
        sft_epochs: int = 8,
        dpo_epochs: int = 3,
        sft_lr: float = 1e-5,
        dpo_lr: float = 1e-7,
        beta: float = 0.05,
    ):
        super().__init__(daemon=True)
        self.conversation_id = conversation_id
        self.sft_epochs = sft_epochs
        self.dpo_epochs = dpo_epochs
        self.sft_lr = sft_lr
        self.dpo_lr = dpo_lr
        self.beta = beta

        self.running = False
        self.completed = False
        self.error: Optional[str] = None
        self.phase = "pending"   # pending → sft → dpo → locking → done
        self.progress = 0        # 0–100
        self.sft_steps = 0       # computed at runtime from dataset size
        self.dpo_steps = 0
        self.start_time: Optional[float] = None

    def run(self):
        from ai_clients.tokenizer import load_tokenizer
        from model.loader import load_model
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from chat.conversation_store import get_conversation, set_conversation_status

        self.running = True
        self.start_time = time.time()

        try:
            # ── Load conversation ──────────────────────────────────────────────
            conv = get_conversation(self.conversation_id)
            if conv is None:
                raise RuntimeError(f"Conversation not found: {self.conversation_id}")

            messages = conv.get("messages", [])
            if not messages:
                raise RuntimeError("Conversation has no messages.")

            sft_text = _build_sft_text(messages)
            dpo_pairs = _extract_dpo_pairs(messages)

            # ── Load model ─────────────────────────────────────────────────────
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = load_tokenizer()
            ckpt_path = Path(config.CHECKPOINT_DIR) / "latest.pt"
            model = load_model(ckpt_path, device)
            model.train()

            block_size = config.BLOCK_SIZE

            # ── Build datasets and derive step counts ──────────────────────────
            sft_ds = _SFTDataset(sft_text, tokenizer, block_size)
            self.sft_steps = len(sft_ds) * self.sft_epochs

            dpo_ds = _DPODataset(dpo_pairs, tokenizer, block_size) if dpo_pairs else None
            self.dpo_steps = len(dpo_ds) * self.dpo_epochs if dpo_ds else 0

            total_steps = self.sft_steps + self.dpo_steps
            logger.info(
                "Dream cycle: %d SFT blocks × %d epochs = %d steps; "
                "%d DPO pairs × %d epochs = %d steps",
                len(sft_ds), self.sft_epochs, self.sft_steps,
                len(dpo_pairs), self.dpo_epochs, self.dpo_steps,
            )

            # ── SFT pass ───────────────────────────────────────────────────────
            self.phase = "sft"

            sft_loader = DataLoader(sft_ds, batch_size=1, shuffle=True)
            sft_iter = iter(sft_loader)

            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.sft_lr,
            )

            for step in range(self.sft_steps):
                try:
                    batch = next(sft_iter)
                except StopIteration:
                    sft_iter = iter(sft_loader)
                    batch = next(sft_iter)

                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                self.progress = int((step + 1) / max(total_steps, 1) * 100)
                logger.debug("Dream SFT step %d/%d loss %.4f", step + 1, self.sft_steps, loss.item())

            # ── DPO pass ───────────────────────────────────────────────────────
            if dpo_ds and self.dpo_steps > 0:
                self.phase = "dpo"
                logger.info("Dream cycle DPO: %d pairs × %d epochs = %d steps",
                            len(dpo_pairs), self.dpo_epochs, self.dpo_steps)

                dpo_loader = DataLoader(dpo_ds, batch_size=1, shuffle=True)
                dpo_iter = iter(dpo_loader)

                dpo_optimizer = AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.dpo_lr,
                )

                for step in range(self.dpo_steps):
                    try:
                        batch = next(dpo_iter)
                    except StopIteration:
                        dpo_iter = iter(dpo_loader)
                        batch = next(dpo_iter)

                    chosen   = batch["chosen_ids"].to(device)
                    rejected = batch["rejected_ids"].to(device)

                    loss = _dpo_loss(model, chosen, rejected, beta=self.beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    dpo_optimizer.step()
                    dpo_optimizer.zero_grad()

                    self.progress = int(
                        (self.sft_steps + step + 1) / max(total_steps, 1) * 100
                    )
                    logger.debug("Dream DPO step %d/%d loss %.4f", step + 1, self.dpo_steps, loss.item())
            else:
                logger.info("Dream cycle: no DPO pairs, skipping DPO pass.")

            # ── Save checkpoint ────────────────────────────────────────────────
            self.phase = "locking"
            logger.info("Dream cycle: saving checkpoint")

            import json as _json
            meta_path = Path(config.CHECKPOINT_DIR) / "metadata.json"
            meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
            ckpt_phase = meta.get("latest.pt", {}).get("phase", 0)

            from train.train import save_checkpoint, _frozen_state
            save_checkpoint(
                config.CHECKPOINT_DIR,
                step=meta.get("latest.pt", {}).get("step", 0),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                cfg={
                    "vocab_size": tokenizer.vocab_size,
                    "block_size": block_size,
                    "phase": ckpt_phase,
                    **_frozen_state(model),
                },
            )

            # ── Lock conversation ──────────────────────────────────────────────
            set_conversation_status(self.conversation_id, "locked")
            logger.info("Dream cycle complete. Conversation %s locked.", self.conversation_id)

            self.phase = "done"
            self.progress = 100

        except Exception as e:
            logger.exception("Dream cycle failed")
            self.error = str(e)
            # Revert conversation to full so user can retry
            try:
                from chat.conversation_store import set_conversation_status
                set_conversation_status(self.conversation_id, "full")
            except Exception:
                pass
        finally:
            self.running = False
            self.completed = True

    def status(self) -> dict:
        return {
            "running": self.running,
            "completed": self.completed,
            "error": self.error,
            "phase": self.phase,
            "progress": self.progress,
            "sft_steps": self.sft_steps,
            "dpo_steps": self.dpo_steps,
            "elapsed": time.time() - self.start_time if self.start_time else None,
        }
