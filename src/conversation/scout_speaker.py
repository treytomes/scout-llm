"""
scout_speaker.py

Watches the conversation log. When a signal for Scout's turn appears,
generates her response from the checkpoint and writes it to the log.

Usage:
    python scout_speaker.py --log data/conversations/session.jsonl
    python scout_speaker.py --log data/conversations/session.jsonl --model latest
"""

import argparse
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

import config
from ai_clients.tokenizer import load_tokenizer
from model.loader import load_model
from cli_repl import stream_generate
from conversation_log import (
    read_turns,
    append_turn,
    format_as_dialogue,
    get_latest_signal,
)

STOP_SEQUENCES = ["[Trey]", "[Claude]"]
POLL_INTERVAL = 2.0


def generate_scout_turn(model, tokenizer, device, dialogue_so_far: str) -> str:
    """Generate Scout's response to the current conversation."""
    prompt = dialogue_so_far + "\n\n[Scout]"

    response_pieces = []
    for piece in stream_generate(model, tokenizer, prompt, device):
        # Stop at the next speaker marker
        accumulated = "".join(response_pieces) + piece
        stop_found = False
        for stop in STOP_SEQUENCES:
            if stop in accumulated:
                # Trim everything from the stop sequence onward
                idx = accumulated.index(stop)
                response_pieces = [accumulated[:idx]]
                stop_found = True
                break
        if stop_found:
            break
        response_pieces.append(piece)

    return "".join(response_pieces).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to conversation log file")
    parser.add_argument("--model", default="latest", help="Checkpoint name (default: latest)")
    parser.add_argument("--poll", type=float, default=POLL_INTERVAL, help="Poll interval in seconds")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.CHECKPOINT_DIR / f"{args.model}.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading Scout from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer()
    print(f"Scout ready. Watching {log_path}...")

    last_seen_count = 0

    while True:
        turns = read_turns(log_path)

        if len(turns) != last_seen_count:
            last_seen_count = len(turns)
            signal = get_latest_signal(turns)

            if signal == "Scout":
                dialogue = format_as_dialogue(turns)
                print("\n[Scout is thinking...]")
                response = generate_scout_turn(model, tokenizer, device, dialogue)
                print(f"[Scout] {response}")
                append_turn(log_path, "Scout", response)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()