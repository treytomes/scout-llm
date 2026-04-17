# repl.py
import torch
from rich.console import Console

import config
from ai_clients.tokenizer import load_tokenizer
from model.loader import load_model


console = Console()


# ── Sampling ──────────────────────────────────────────────────────────────
def sample_next(logits):
    """
    Temperature + top‑k sampling.
    """
    logits = logits / config.TEMPERATURE
    if config.TOP_K:
        v, _ = torch.topk(logits, config.TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ── Generation ─────────────────────────────────────────────────────────────
def stream_generate(model, tokenizer, prompt, device):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    last_text = ""

    for _ in range(config.MAX_NEW_TOKENS):
        tokens = tokens[:, -config.BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(tokens)

        logits = logits[:, -1, :]
        next_token = sample_next(logits)

        tokens = torch.cat([tokens, next_token], dim=1)
        tok_id = next_token.item()

        if eos_id is not None and tok_id == eos_id:
            break

        generated_ids.append(tok_id)

        # Decode the full generated sequence
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stream only the newly added portion
        delta = text[len(last_text):]
        last_text = text

        if delta:
            yield delta


# ── REPL ───────────────────────────────────────────────────────────────────
def run_repl(checkpoint_path):
    """
    Simple CLI REPL for testing inference.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print("[bold green]Loading model...[/bold green]")
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer()
    console.print("[bold green]Model ready.[/bold green]\n")

    try:
        while True:
            prompt = console.input("[bold cyan][You][/bold cyan] ")
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            console.print("[bold yellow][Scout] [/bold yellow]", end="")
            for piece in stream_generate(model, tokenizer, prompt, device):
                console.print(piece, end="")
            console.print()

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting.[/bold red]")