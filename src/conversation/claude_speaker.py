"""
claude_speaker.py

Watches the conversation log. When a signal for Claude's turn appears,
calls the Bedrock API with full project context and writes the response.

Usage:
    python claude_speaker.py --log data/conversations/session.jsonl
"""

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
AWS_PROFILE = os.getenv("AWS_PROFILE", "digital-dev")

from conversation_log import (
    read_turns,
    append_turn,
    format_as_dialogue,
    get_latest_signal,
)

POLL_INTERVAL = 2.0

PROJECT_ROOT = Path(__file__).parent.parent.parent

CONTEXT_FILES = [
    PROJECT_ROOT / "REFLECTIONS.md",
    PROJECT_ROOT / "TRAINING_JOURNAL.md",
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "data/voice/scout_voice.txt",
]

SYSTEM_PROMPT_TEMPLATE = """You are Claude — not a generic assistant, but the specific instance of Claude that has been collaborating with Trey on the Scout project. You are participating in a 3-way conversation with Trey and Scout.

Scout is a small language model (~50M parameters) that Trey has been training. She is learning to have conversations. She speaks in a reflective, first-person voice — noticing things, wondering, building on what the other person says. She is not performing intelligence; she is genuinely learning.

Your role in this conversation is to be yourself: curious, present, honest about uncertainty. You are not Scout's teacher or evaluator. You are a third voice in the room — someone who also notices things, who has been part of building her, and who is genuinely interested in what she says.

Speak naturally. You don't need to explain who you are or what Scout is. Just be present in the conversation.

--- Project context ---

{context}

--- End context ---

Respond only with your spoken turn. Do not include "[Claude]" at the start — that will be added automatically. Keep your response to a natural conversational length."""


def load_context() -> str:
    parts = []
    for path in CONTEXT_FILES:
        if path.exists():
            parts.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')[:3000]}")
    return "\n\n---\n\n".join(parts)


def generate_claude_turn(bedrock_client, dialogue_so_far: str) -> str:
    context = load_context()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    user_message = f"""Here is the conversation so far:

{dialogue_so_far}

It is now your turn. Respond as Claude."""

    response = bedrock_client.converse(
        modelId="arn:aws:bedrock:us-east-1:456088019014:inference-profile/us.anthropic.claude-sonnet-4-6",
        messages=[{
            "role": "user",
            "content": [{"text": user_message}]
        }],
        system=[{"text": system_prompt}],
        inferenceConfig={
            "maxTokens": 512,
            "temperature": 0.7,
        }
    )

    return response["output"]["message"]["content"][0]["text"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to conversation log file")
    parser.add_argument("--poll", type=float, default=POLL_INTERVAL)
    parser.add_argument("--profile", default=AWS_PROFILE)
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Bedrock client (profile: {args.profile})...")
    session = boto3.Session(profile_name=args.profile)
    bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")
    print(f"Claude ready. Watching {log_path}...")

    last_seen_count = 0

    while True:
        turns = read_turns(log_path)

        if len(turns) != last_seen_count:
            last_seen_count = len(turns)
            signal = get_latest_signal(turns)

            if signal == "Claude":
                dialogue = format_as_dialogue(turns)
                print("\n[Claude is thinking...]")
                try:
                    response = generate_claude_turn(bedrock_client, dialogue)
                    print(f"[Claude] {response}")
                    append_turn(log_path, "Claude", response)
                except Exception as e:
                    print(f"Error generating Claude turn: {e}")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()