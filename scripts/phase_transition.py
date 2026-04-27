"""
phase_transition.py — Advance Scout to the next developmental phase.

Usage:
    python scripts/phase_transition.py

What this does:
    1. Loads latest.pt (Module 0, TinyStories + dialogue foundation)
    2. Freezes Module 0 and the language core (embedding + output head)
    3. Appends Module 1 (conversational layer, randomly initialised)
    4. Saves the result as data/checkpoints/phase1_start.pt
       (does NOT overwrite latest.pt)

After running this script:
    - Copy phase1_start.pt → latest.pt to begin Phase 1 training
    - Train on SODA + DailyDialog with reset_optimizer=True
    - Only Module 1 parameters are trainable; frozen layers don't move
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "server"))

import torch
import config
from model.loader import load_checkpoint, init_model
from model.model import count_parameters
from ai_clients.tokenizer import load_tokenizer


def main():
    device = torch.device("cpu")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size

    checkpoint_path = config.CHECKPOINT_DIR / "latest.pt"
    out_path = config.CHECKPOINT_DIR / "phase1_start.pt"

    print(f"Loading checkpoint: {checkpoint_path}")

    # Build Module 0 model matching the checkpoint's block_size
    cfg_module0 = dict(config.MODEL_TINYSTORIES)
    cfg_module0["block_size"] = config.BLOCK_SIZE

    model = init_model(vocab_size, device, config_dict=cfg_module0)
    ckpt, _ = load_checkpoint(checkpoint_path, model, device)

    print(f"  Loaded step {ckpt['step']}")
    stats_before = count_parameters(model)
    print(f"  Parameters: {stats_before['total_millions']}M total, "
          f"{stats_before['trainable_millions']}M trainable")

    # Freeze Module 0 and the language core
    model.freeze_module(0)
    model.freeze_language_core()

    # Add Module 1 — randomly initialised, trains freely on conversational corpus
    model.add_module(config.MODEL_CONVERSATIONAL)

    stats_after = count_parameters(model)
    print(f"\nAfter adding Module 1:")
    print(f"  Total:     {stats_after['total_millions']}M parameters")
    print(f"  Trainable: {stats_after['trainable_millions']}M parameters (Module 1 only)")
    print(f"  Frozen:    {round(stats_after['frozen'] / 1e6, 2)}M parameters (Module 0 + language core)")

    for name, info in stats_after["per_module"].items():
        status = "frozen" if info["trainable"] == 0 else "trainable"
        print(f"  {name}: {info['total_millions']}M ({status})")

    # Save — step carried forward so training dashboard shows continuity.
    # frozen_modules and language_core_frozen let the loader re-apply
    # requires_grad=False after loading (not stored in state_dict itself).
    checkpoint = {
        "step": ckpt["step"],
        "model": model.state_dict(),
        "optimizer": None,
        "scheduler": None,
        "config": {
            "vocab_size": vocab_size,
            "block_size": config.BLOCK_SIZE,
            "phase": 1,
            "frozen_modules": [0],
            "language_core_frozen": True,
        },
    }

    torch.save(checkpoint, out_path)

    import json
    meta_path = config.CHECKPOINT_DIR / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:
        meta = {}
    meta["phase1_start.pt"] = {"step": ckpt["step"], "phase": 1}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved: {out_path}")
    print("\nNext steps:")
    print("  cp data/checkpoints/phase1_start.pt data/checkpoints/latest.pt")
    print("  Then start training with reset_optimizer=True on SODA+DailyDialog")


if __name__ == "__main__":
    main()