"""
conversation_log.py

Shared log format for round-robin conversations between Trey, Scout, and Claude.

Each turn is a JSON line:
    {"speaker": "Trey",   "text": "...", "timestamp": "..."}
    {"speaker": "Scout",  "text": "...", "timestamp": "..."}
    {"speaker": "Claude", "text": "...", "timestamp": "..."}

A special "signal" turn marks whose turn it is next:
    {"signal": "Scout",  "timestamp": "..."}
    {"signal": "Claude", "timestamp": "..."}
    {"signal": "Trey",   "timestamp": "..."}
"""

import json
from datetime import datetime, timezone
from pathlib import Path


SPEAKERS = ["Trey", "Scout", "Claude"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_turns(log_path: Path) -> list[dict]:
    """Read all turns from the log file."""
    if not log_path.exists():
        return []
    turns = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return turns


def append_turn(log_path: Path, speaker: str, text: str):
    """Append a spoken turn to the log."""
    entry = {"speaker": speaker, "text": text.strip(), "timestamp": now_iso()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def append_signal(log_path: Path, next_speaker: str):
    """Append a signal indicating whose turn is next."""
    entry = {"signal": next_speaker, "timestamp": now_iso()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_latest_signal(turns: list[dict]) -> str | None:
    """Return the most recent signal, or None."""
    for turn in reversed(turns):
        if "signal" in turn:
            return turn["signal"]
    return None


def format_as_dialogue(turns: list[dict]) -> str:
    """Format spoken turns as dialogue text for model prompting."""
    lines = []
    for turn in turns:
        if "speaker" in turn:
            lines.append(f"[{turn['speaker']}] {turn['text']}")
    return "\n\n".join(lines)