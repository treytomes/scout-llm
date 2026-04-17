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


def load_model(checkpoint_path, device):
    """
    Load tokenizer + model from checkpoint.
    """

    tokenizer = load_tokenizer()

    vocab_size = tokenizer.vocab_size

    model = init_model(vocab_size, device)

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