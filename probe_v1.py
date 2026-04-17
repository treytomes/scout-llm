"""
Probe v1 Scout checkpoints from custom-llm project.
"""

import sys
import torch
from pathlib import Path

# Add custom-llm to path for imports
v1_src = Path("/home/trey/projects/custom-llm/src")
sys.path.insert(0, str(v1_src))

from model.model import GPT
from transformers import AutoTokenizer


def load_v1_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a v1 Scout checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    vocab_size = config.get("vocab_size", 32000)

    # Infer block_size from RoPE buffer shape in the checkpoint
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Get block_size from rope_cos shape
    rope_key = "blocks.0.attn.rope_cos"
    if rope_key in state_dict:
        block_size = state_dict[rope_key].shape[0]
    else:
        block_size = config.get("block_size", 768)

    model = GPT(
        vocab_size=vocab_size,
        dim=512,
        layers=12,
        heads=8,
        max_seq=block_size,
        dropout=0.1
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, block_size


def format_prompt(text: str) -> str:
    """Format text as a Scout prompt (with trailing space removed)."""
    return f"[Trey] {text}\n[Scout]"


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    rep_penalty: float = 1.3,
):
    """Generate text from a v1 Scout checkpoint."""

    formatted = format_prompt(prompt)
    tokens = tokenizer.encode(formatted, return_tensors="pt").to(device)

    generated_ids = []  # Track token IDs for repetition penalty
    printed_length = 0  # Track how much text we've printed

    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :] / temperature

        # Apply repetition penalty
        if generated_ids:
            for token_id in set(generated_ids[-50:]):
                logits[0, token_id] /= rep_penalty

        # Top-k sampling
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token[0].item()

        generated_ids.append(next_token_id)

        # Decode entire generated sequence to preserve spacing
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop at newline followed by bracket (end of Scout's turn)
        if '\n[' in full_text:
            # Print up to the stopping point
            stop_idx = full_text.index('\n[')
            new_text = full_text[printed_length:stop_idx]
            print(new_text, end='', flush=True)
            break

        # Print only the new portion
        new_text = full_text[printed_length:]
        print(new_text, end='', flush=True)
        printed_length = len(full_text)

        tokens = torch.cat([tokens, next_token], dim=1)

        # Truncate if exceeding max_seq
        if tokens.shape[1] > model.max_seq:
            tokens = tokens[:, -model.max_seq:]

    print()  # Final newline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt text to send to Scout")
    parser.add_argument("--checkpoint", "-c", default="model_50000_block_size_256.pt",
                       help="Checkpoint filename in custom-llm/data/checkpoints/")
    parser.add_argument("--max-tokens", "-m", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = f"/home/trey/projects/custom-llm/data/checkpoints/{args.checkpoint}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, block_size = load_v1_checkpoint(checkpoint_path, device)

    print(f"Loading tokenizer: mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    print(f"\nPrompt: {args.prompt}")
    print(f"[Scout] ", end='', flush=True)

    generate(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_tokens,
        temperature=0.7,
        top_k=40,
        rep_penalty=1.3
    )
