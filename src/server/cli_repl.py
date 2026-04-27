# repl.py
import torch
from rich.console import Console

import config
from ai_clients.tokenizer import load_tokenizer
from model.loader import load_model


console = Console()


# ── Sampling ──────────────────────────────────────────────────────────────
def sample_next(logits, temperature=None, top_k=None):
    """
    Temperature + top‑k sampling.
    """
    t = temperature if temperature is not None else config.TEMPERATURE
    k = top_k if top_k is not None else config.TOP_K

    logits = logits / t
    if k:
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ── Generation ─────────────────────────────────────────────────────────────
def stream_generate(model, tokenizer, prompt, device,
                    temperature=None, top_k=None, rep_penalty=None, max_new_tokens=None,
                    skip_modules=None):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    effective_rep_penalty = rep_penalty if rep_penalty is not None else config.REP_PENALTY
    effective_max = max_new_tokens if max_new_tokens is not None else config.MAX_NEW_TOKENS

    generated_ids = []
    last_text = ""

    for _ in range(effective_max):
        tokens = tokens[:, -config.BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(tokens, skip_modules=skip_modules)

        logits = logits[:, -1, :]

        # Repetition penalty: down-weight tokens already generated
        if effective_rep_penalty != 1.0 and generated_ids:
            for tok_id in set(generated_ids):
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= effective_rep_penalty
                else:
                    logits[0, tok_id] *= effective_rep_penalty

        next_token = sample_next(logits, temperature=temperature, top_k=top_k)

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