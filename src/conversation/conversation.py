"""
conversation.py

Round-robin conversation orchestrator for Trey, Scout, and Claude.

Manages turn order, prompts Trey for input, signals automated speakers,
and waits for their responses before advancing.

Usage:
    python conversation.py --log data/conversations/session.jsonl
    python conversation.py --log data/conversations/session.jsonl --order Trey,Scout,Claude
    python conversation.py --log data/conversations/session.jsonl --resume
"""

import argparse
import sys
import time
from pathlib import Path

from conversation_log import (
    SPEAKERS,
    read_turns,
    append_turn,
    append_signal,
    format_as_dialogue,
    get_latest_signal,
)

AUTOMATED_SPEAKERS = {"Scout", "Claude"}
RESPONSE_TIMEOUT = 120   # seconds to wait for automated speaker
POLL_INTERVAL = 1.0


def wait_for_response(log_path: Path, speaker: str, turns_before: int) -> str | None:
    """Wait until the automated speaker has written their turn."""
    deadline = time.time() + RESPONSE_TIMEOUT
    while time.time() < deadline:
        turns = read_turns(log_path)
        spoken = [t for t in turns if t.get("speaker") == speaker]
        if len(spoken) > sum(1 for t in turns[:turns_before] if t.get("speaker") == speaker):
            return spoken[-1]["text"]
        time.sleep(POLL_INTERVAL)
    return None


def display_conversation(turns: list[dict]):
    """Print the full conversation so far."""
    print("\n" + "=" * 60)
    for turn in turns:
        if "speaker" in turn:
            print(f"\n[{turn['speaker']}] {turn['text']}")
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to conversation log file")
    parser.add_argument("--order", default="Trey,Scout,Claude",
                        help="Turn order as comma-separated names (default: Trey,Scout,Claude)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing conversation from the log")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    order = [s.strip() for s in args.order.split(",")]
    for speaker in order:
        if speaker not in SPEAKERS:
            print(f"Unknown speaker: {speaker}. Must be one of: {SPEAKERS}")
            sys.exit(1)

    print(f"Starting conversation: {' → '.join(order)}")
    print(f"Log: {log_path}")
    print(f"Automated speakers ({', '.join(AUTOMATED_SPEAKERS & set(order))}) must be running separately.")
    print("\nType 'pass' to skip your turn. Type 'quit' or Ctrl+C to end.\n")

    if args.resume and log_path.exists():
        turns = read_turns(log_path)
        if turns:
            display_conversation(turns)
            print("Resuming conversation...\n")

    turn_index = 0

    try:
        while True:
            speaker = order[turn_index % len(order)]
            turns_before = len(read_turns(log_path))

            if speaker in AUTOMATED_SPEAKERS:
                print(f"[Signaling {speaker}...]")
                append_signal(log_path, speaker)

                response = wait_for_response(log_path, speaker, turns_before)
                if response is None:
                    print(f"[{speaker} did not respond within {RESPONSE_TIMEOUT}s — skipping turn]")
                else:
                    print(f"\n[{speaker}] {response}\n")

            else:
                # Trey's turn
                try:
                    text = input(f"[{speaker}] ").strip()
                except EOFError:
                    break

                if text.lower() in {"quit", "exit"}:
                    break
                elif text.lower() == "pass":
                    print(f"[{speaker} passed]")
                    turn_index += 1
                    continue
                elif text:
                    append_turn(log_path, speaker, text)

            turn_index += 1

    except KeyboardInterrupt:
        print("\n\nConversation ended.")

    # Print final transcript
    turns = read_turns(log_path)
    print("\n" + "=" * 60)
    print("Final transcript:")
    print("=" * 60)
    print(format_as_dialogue(turns))
    print("=" * 60)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()