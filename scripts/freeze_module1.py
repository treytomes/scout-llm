"""
scripts/freeze_module1.py

Freeze Module 1 in the current checkpoint and save a new checkpoint.
This is required before LoRA dream cycles can engage — the dream cycle
uses LoRA mode only when all base modules are frozen.

Run from project root:
    python scripts/freeze_module1.py

Output: data/checkpoints/latest.pt (updated with frozen_modules=[0, 1])
        data/checkpoints/scout_dialogue_200.pt (same — updated in place)
"""

import sys
from pathlib import Path

# Allow importing from src/server
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "server"))

import torch
import config
from model.loader import init_model_for_checkpoint, load_checkpoint
from model.model import count_parameters
from ai_clients.tokenizer import load_tokenizer


CHECKPOINT_PATH = config.CHECKPOINT_PATH
DEVICE = torch.device("cpu")


def audit(model, label=""):
    params = count_parameters(model)
    print(f"\n{'─' * 50}")
    if label:
        print(f"  {label}")
    print(f"  Total:     {params['total_millions']}M params")
    print(f"  Trainable: {params['trainable_millions']}M params")
    for name, p in params["per_module"].items():
        frozen = p["total"] == p["frozen"]
        print(f"  {name}: {p['total_millions']}M  frozen={frozen}")
    lc_trainable = sum(p.numel() for p in model.language.parameters() if p.requires_grad)
    print(f"  LanguageCore: frozen={lc_trainable == 0}")
    lora_mode = (
        len(model.expert_modules) >= 2
        and not any(p.requires_grad for p in model.expert_modules[1].parameters())
    )
    print(f"\n  Dream cycle LoRA mode: {lora_mode}")
    print(f"{'─' * 50}\n")
    return params


def main():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Load model with existing freeze state from checkpoint
    model = init_model_for_checkpoint(CHECKPOINT_PATH, vocab_size, DEVICE)
    checkpoint, _ = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)

    print("\nBefore freeze:")
    audit(model, "Current state")

    # Freeze Module 1
    print("Freezing Module 1...")
    model.freeze_module(1)

    print("\nAfter freeze:")
    audit(model, "After Model 1 freeze")

    # Build updated checkpoint with frozen_modules=[0, 1]
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    ckpt_config["frozen_modules"] = [0, 1]
    ckpt_config["language_core_frozen"] = ckpt_config.get("language_core_frozen", False)

    new_checkpoint = {
        "step": checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0,
        "model": model.state_dict(),
        "optimizer": checkpoint.get("optimizer", {}) if isinstance(checkpoint, dict) else {},
        "scheduler": checkpoint.get("scheduler", {}) if isinstance(checkpoint, dict) else {},
        "config": ckpt_config,
    }

    # Save
    print(f"Saving updated checkpoint to {CHECKPOINT_PATH} ...")
    torch.save(new_checkpoint, CHECKPOINT_PATH)

    # Also update the named copy
    named_copy = config.CHECKPOINT_DIR / "scout_dialogue_200.pt"
    if named_copy.exists():
        print(f"Updating named copy: {named_copy} ...")
        torch.save(new_checkpoint, named_copy)

    print("\nDone. Module 1 is now frozen.")
    print("LoRA dream cycles will engage on the next full conversation.")


if __name__ == "__main__":
    main()
