"""
model/loader.py
Shared model loading for the modular Scout architecture.
"""

import torch

import config
from ai_clients.tokenizer import load_tokenizer
from .model import ScoutModel


def load_checkpoint(checkpoint_path, model, device):
    """
    Load a checkpoint into a ScoutModel.

    Handles:
    - compiled model checkpoints
    - tensors dependent on BLOCK_SIZE (RoPE tables)
    - router/module architecture differences
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Fix compiled-model checkpoints
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Remove tensors dependent on BLOCK_SIZE
    # These are regenerated when the model is initialized
    state = {
        k: v
        for k, v in state.items()
        if "attn.mask" not in k
        and "rope_cos" not in k
        and "rope_sin" not in k
    }

    # Load state dict (allow missing keys for future modular expansion)
    model.load_state_dict(state, strict=False)

    return checkpoint, state


def init_model(vocab_size, device, config_dict=None):
    """
    Initialize a ScoutModel.

    If no config is provided, the TinyStories configuration is used.
    """

    if config_dict is None:
        config_dict = config.MODEL_TINYSTORIES

    model = ScoutModel(
        vocab_size=vocab_size,
        cfg=config_dict,
    ).to(device)

    return model


def count_modules_in_state(state: dict) -> int:
    """
    Infer the number of expert modules from a checkpoint state dict.
    Looks for the highest expert_modules.N prefix.
    """
    import re
    indices = set()
    for k in state:
        m = re.match(r"expert_modules\.(\d+)\.", k)
        if m:
            indices.add(int(m.group(1)))
    return len(indices) if indices else 1


def init_model_for_checkpoint(checkpoint_path, vocab_size, device, base_config=None):
    """
    Initialize a ScoutModel with the correct number of modules to match a checkpoint,
    and re-apply any frozen state recorded in the checkpoint config.

    For Phase 0 checkpoints (1 module), returns a standard single-module model.
    For Phase 1+ checkpoints, builds the base module then appends additional modules
    using MODULE_CONVERSATIONAL config, then re-freezes as recorded.
    """
    import torch as _torch
    from .model import ScoutModel

    if base_config is None:
        base_config = config.MODEL_TINYSTORIES

    # Peek at the checkpoint to count modules and read phase config
    raw = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    state_clean = {k: v for k, v in state.items()
                   if "rope_cos" not in k and "rope_sin" not in k and "attn.mask" not in k}

    ckpt_config = raw.get("config", {}) if isinstance(raw, dict) else {}
    num_modules = count_modules_in_state(state_clean)

    model = ScoutModel(vocab_size=vocab_size, cfg=base_config).to(device)

    for _ in range(num_modules - 1):
        model.add_module(config.MODEL_CONVERSATIONAL)

    # Re-apply frozen state — requires_grad is not persisted in state_dict
    frozen_modules = ckpt_config.get("frozen_modules", [])
    for idx in frozen_modules:
        model.freeze_module(idx)

    if ckpt_config.get("language_core_frozen", False):
        model.freeze_language_core()

    return model


def load_model(checkpoint_path, device):
    """
    Load tokenizer + model from checkpoint.
    """

    tokenizer = load_tokenizer()

    vocab_size = tokenizer.vocab_size

    model = init_model_for_checkpoint(checkpoint_path, vocab_size, device)

    checkpoint, state = load_checkpoint(checkpoint_path, model, device)

    model.eval()

    return model


def load_fresh_model(device, config_dict=None):
    """
    Initialize a brand new model without loading a checkpoint.

    Useful for starting a new training run.
    """

    tokenizer = load_tokenizer()

    vocab_size = tokenizer.vocab_size

    model = init_model(
        vocab_size=vocab_size,
        device=device,
        config_dict=config_dict,
    )

    return model