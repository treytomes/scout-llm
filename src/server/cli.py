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
def train():
    """
    Launch model training.
    """

    dataset_name = "TinyStories"
    model_config = config.MODEL_TINYSTORIES
    batch_size = 8
    max_steps = 40000

    latest_metrics = None
    metrics_history = []

    logger.info("Training job starting for dataset: %s", dataset_name)

    running = True
    start_time = time.time()

    try:
        for metrics in run_training(
            dataset_name=dataset_name,
            model_config=model_config,
            batch_size=batch_size,
            max_steps=max_steps,
        ):
            latest_metrics = metrics
            metrics_history.append(metrics)

    except Exception as e:
        logger.exception("Training job failed")
        error = str(e)

    finally:
        running = False
        completed = True

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