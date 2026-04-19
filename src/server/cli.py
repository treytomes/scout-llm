# cli.py

import logging
import time
import typer
from dotenv import load_dotenv
from rich.logging import RichHandler

import config
from cli_repl import run_repl
from train.data import (
    corpus_needs_tokenization,
    load_token_tensor,
    tokenize_corpus,
)
from train.train import run_training


load_dotenv()

app = typer.Typer()


# ---------------------------------------------------------
# logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(config.LOGGER_NAME)


@app.command()
def chat(
    model: str = typer.Argument("latest")
):
    """
    Simple inference REPL.
    Example:
        cli.py chat model_1000
    """
    checkpoint_dir = config.CHECKPOINT_PATH.parent
    checkpoint_path = checkpoint_dir / f"{model}.pt"
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint_path}")
    run_repl(checkpoint_path)


@app.command()
def probe(
    prompt: str,
    model: str = typer.Option("latest", "--model", "-m"),
):
    """
    Send a single prompt to a checkpoint and print the response.
    Example:
        cli.py probe "Once upon a time" --model latest
    """
    import torch
    from cli_repl import stream_generate
    from model.loader import load_model
    from ai_clients.tokenizer import load_tokenizer

    checkpoint_dir = config.CHECKPOINT_PATH.parent
    checkpoint_path = checkpoint_dir / f"{model}.pt"
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer()

    logger.info("Prompt: %s", prompt)
    print()  # blank line before output

    for piece in stream_generate(model, tokenizer, prompt, device):
        print(piece, end="", flush=True)

    print()  # newline after completion
    

@app.command()
def train(
    dataset: str = typer.Option("TinyStories", "--dataset", "-d", help="Dataset name"),
    max_steps: int = typer.Option(1000, "--max-steps", "-s", help="Total training steps"),
    lr: float = typer.Option(config.LEARNING_RATE, "--lr", help="Peak learning rate"),
    min_lr: float = typer.Option(config.MIN_LR, "--min-lr", help="Minimum learning rate"),
    warmup_steps: int = typer.Option(config.WARMUP_STEPS, "--warmup-steps", help="LR warmup steps"),
    batch_size: int = typer.Option(config.BATCH_SIZE, "--batch-size", "-b", help="Batch size"),
    reset_optimizer: bool = typer.Option(False, "--reset-optimizer", help="Load weights only, reset optimizer and scheduler"),
):
    """
    Launch model training.

    Fine-tuning example:
        cli.py train --dataset tinystories_dialogue --max-steps 1000 --lr 5e-5 --min-lr 5e-6 --warmup-steps 50 --reset-optimizer
    """

    model_config = config.MODEL_TINYSTORIES

    logger.info("Training job starting for dataset: %s", dataset)
    logger.info("max_steps=%d  lr=%.2e  min_lr=%.2e  warmup=%d  reset_optimizer=%s",
                max_steps, lr, min_lr, warmup_steps, reset_optimizer)

    try:
        for metrics in run_training(
            dataset_name=dataset,
            model_config=model_config,
            batch_size=batch_size,
            max_steps=max_steps,
            lr=lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            reset_optimizer=reset_optimizer,
        ):
            pass

    except Exception:
        logger.exception("Training job failed")

    else:
        logger.info("Training job finished")


# @app.command()
# def chat():
#     """
#     Start interactive chat with the trained model.
#     """
#     run_chat_repl()


# ---------------------------------------------------------

if __name__ == "__main__":
    app()