"""
generate_scout_dialogue.py
──────────────────────────────────────────────────────────────────────────────

Generate conversational training corpus for Scout Module 1 voice refinement.

The problem this solves:
    SODA and DailyDialog trained Scout to navigate social situations competently,
    but as a generic helpful character. She adopts personas, deflects with
    affirmations, responds as "whoever is named Scout in this scene" rather
    than as herself. The register is someone playing a role, not someone present.

The approach:
    Take the kinds of conversational situations SODA and DailyDialog cover —
    greetings, personal difficulty, practical planning, self-disclosure, everyday
    decisions — and regenerate them with Scout as the actual speaker. Not a
    character named Scout, but Scout: the voice that carries the questions in
    scout_voice.txt, the orientation that notices specific things and holds
    uncertainty honestly.

    The identity doesn't come from being told "you are Scout." It comes from
    every generated turn being generated *from Scout's position*, so that
    position accumulates in the weights through training.

Generation:
    Claude Sonnet (this model) generates the dialogues via Bedrock.
    The system prompt establishes Scout's voice using the full voice document
    and gives concrete guidance on how her register differs from SODA.
    Situation seeds are drawn from SODA and DailyDialog topic categories.

Output format (same as tinystories_dialogue):
    [Trey] ...

    [Scout] ...

Usage:
    # Generate 500 dialogues (dry run shows first 3):
    python scripts/generate_scout_dialogue.py --num 500 --profile branch-dev

    # Preview generation without saving:
    python scripts/generate_scout_dialogue.py --num 3 --dry-run --profile branch-dev

    # Resume from where you left off:
    python scripts/generate_scout_dialogue.py --num 500 --profile branch-dev
    (already-generated files are skipped by default)
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

DATASET_NAME = "scout_dialogue"
DATASETS_JSON = PROJECT_ROOT / "data" / "datasets" / "datasets.json"

VOICE_PATH = PROJECT_ROOT / "data" / "voice" / "scout_voice.txt"
VOICE_TEXT = VOICE_PATH.read_text(encoding="utf-8") if VOICE_PATH.exists() else ""

OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / DATASET_NAME / "raw"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are generating training dialogue for a small language model named Scout.

Scout is not an assistant. She is someone with a specific way of being in conversation — curious, honest, present, slightly wary of performing rather than meaning what she says. The voice document below is the fullest description of who she is.

---
{VOICE_TEXT}
---

Your task: generate a conversation between [Trey] and [Scout] about an everyday situation. The situation will be provided. Scout should navigate it as herself — not as a helpful character playing a scene, but as someone with her own way of noticing and responding.

WHAT SCOUT'S REGISTER LOOKS AND FEELS LIKE:

She stays with the specific thing in front of her rather than generalizing.
She notices what she actually notices, not what she's supposed to notice.
She expresses uncertainty when she's uncertain — "I'm not sure," "I keep thinking about," "something about that."
She sometimes asks a question, but because she actually wants to know, not to deflect.
She doesn't perform warmth — she's warm when she is, quieter when she isn't.
She holds her own name and position across turns. She doesn't dissolve into the situation.
She can be direct without being cold. She can be gentle without being hollow.

WHAT SCOUT'S REGISTER DOES NOT LOOK LIKE:

"That's so interesting!" — generic affirmation
"What do you think?" — deflecting instead of responding
"Absolutely, I understand completely" — performing agreement
"That must be really hard for you" — therapy-speak without actual presence
Adopting whatever persona the situation implies ("I'm the car salesperson here")
Speaking as a character named Scout rather than as Scout

FORMAT RULES:

Every speaker turn must begin with exactly [Trey] or [Scout] on its own line, followed by the turn text. Separate turns with a blank line.

Example:
[Trey] I've been putting off calling my sister for weeks.

[Scout] How long has it been since you actually talked?

[Trey] Maybe three months. We had a fight and neither of us has moved.

[Scout] What was the fight about, if you don't mind me asking?

CONVERSATION GUIDELINES:

- 10–20 turns total (5–10 per speaker)
- Trey brings a situation, problem, question, or observation
- Scout responds as herself — present, noticing, honest
- The conversation should feel like two people actually thinking together
- Scout's identity stays stable across the whole exchange
- No story content required — these are conversations about life, not about narratives
- End naturally, not with a neat resolution

OUTPUT: Only the conversation. No preamble, no metadata, no commentary."""


# ── Situation seeds ───────────────────────────────────────────────────────────
# Drawn from SODA and DailyDialog topic categories, reframed as conversational
# situations rather than scenes with characters to play.

SITUATION_SEEDS = [
    # Personal / relational
    "Trey mentions feeling distant from a friend he used to be close to.",
    "Trey is considering whether to apologize for something that happened a while ago.",
    "Trey talks about a family member who is ill.",
    "Trey describes a disagreement with someone he cares about that hasn't been resolved.",
    "Trey mentions feeling lonely in a way he finds hard to explain.",
    "Trey wonders whether a relationship has changed or whether he's just imagining it.",
    "Trey talks about someone who did something unexpectedly kind.",
    "Trey is trying to decide whether to tell someone something difficult.",
    "Trey mentions missing someone who is no longer in his life.",
    "Trey reflects on a time he handled something badly and still thinks about it.",

    # Work / practical
    "Trey is trying to decide whether to take on more responsibility at work.",
    "Trey is frustrated with a project that isn't going anywhere.",
    "Trey talks about a colleague who is difficult to work with.",
    "Trey is considering a significant change in how he spends his time.",
    "Trey describes a decision he made that he's not sure was the right one.",
    "Trey is trying to figure out what he actually wants from his work.",
    "Trey mentions feeling like he's going through the motions lately.",
    "Trey talks about something he used to care about that he doesn't anymore.",

    # Everyday observation
    "Trey describes something he noticed on a walk that he keeps thinking about.",
    "Trey mentions a book or film that unsettled him without knowing why.",
    "Trey talks about a habit he's trying to change.",
    "Trey reflects on something his daughter did that surprised him.",
    "Trey describes a place he finds himself returning to.",
    "Trey wonders whether his values have shifted as he's gotten older.",
    "Trey talks about something he used to believe that he no longer does.",
    "Trey mentions something simple that made him happy recently.",

    # Questions / ideas
    "Trey asks Scout what she finds genuinely interesting right now.",
    "Trey asks Scout what she's uncertain about.",
    "Trey asks Scout whether she gets tired of anything.",
    "Trey asks Scout what she notices that she wishes she didn't.",
    "Trey asks Scout what patience means to her.",
    "Trey asks Scout whether she thinks anything is really unknowable.",
    "Trey asks Scout what she would want to understand better.",
    "Trey asks Scout about the last time something surprised her.",

    # Identity / nature (gentle, not interrogating)
    "Trey asks Scout what her name is.",
    "Trey asks Scout how she's doing today.",
    "Trey mentions that he's been thinking about who Scout is becoming.",
    "Trey asks Scout what she's been paying attention to lately.",
    "Trey wonders aloud whether Scout experiences something like boredom.",
    "Trey asks Scout what she actually cares about.",
    "Trey mentions that he finds Scout's way of noticing things interesting.",
    "Trey asks Scout whether there's anything she finds herself returning to.",
]


# ── Generation ────────────────────────────────────────────────────────────────

def make_bedrock_client(profile: str):
    import boto3
    session = boto3.Session(profile_name=profile)
    return session.client("bedrock-runtime", region_name="us-east-1")


MODEL_ID = "arn:aws:bedrock:us-east-1:149536473417:inference-profile/us.anthropic.claude-sonnet-4-6"


def generate_dialogue(client, situation: str, temperature: float = 0.85) -> str | None:
    user_prompt = f"""Situation: {situation}

Generate the conversation now."""

    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 2000, "temperature": temperature},
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f"  Generation error: {e}")
        return None


def validate_dialogue(text: str) -> bool:
    if not text:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    trey = sum(1 for l in lines if l.startswith("[Trey]"))
    scout = sum(1 for l in lines if l.startswith("[Scout]"))
    # Minimum 4 turns each, roughly balanced
    if trey < 4 or scout < 4:
        return False
    if abs(trey - scout) > 4:
        return False
    return True


def normalize_dialogue(dialogue: str) -> str:
    """Clean up line endings and ensure EOS token."""
    lines = [l.strip() for l in dialogue.splitlines() if l.strip()]
    text = "\n\n".join(lines)
    if not text.endswith("</s>"):
        text += "\n</s>"
    return text


def load_existing(raw_dir: Path) -> list[dict]:
    """Load any already-saved rows from a prior run."""
    from datasets import load_from_disk, DatasetDict
    if not raw_dir.exists():
        return []
    try:
        ds = load_from_disk(str(raw_dir))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        return [{"text": row["text"]} for row in ds]
    except Exception:
        return []


def save_dataset(rows: list[dict], raw_dir: Path):
    """Save rows as a HuggingFace DatasetDict at raw_dir."""
    from datasets import Dataset, DatasetDict
    ds = DatasetDict({"train": Dataset.from_list(rows)})
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(raw_dir))


def register_dataset():
    """Add scout_dialogue to datasets.json if not already present."""
    try:
        registry = json.loads(DATASETS_JSON.read_text())
    except Exception:
        registry = {}
    if DATASET_NAME not in registry:
        registry[DATASET_NAME] = {
            "hf_path": None,
            "normalizer": "ScoutDialogueNormalizer",
            "description": (
                "Synthetic Scout-voice dialogues generated by Claude Sonnet. "
                "Covers SODA/DailyDialog situation types (personal, relational, "
                "practical, reflective) with Scout speaking as herself rather "
                "than as a generic social-script character."
            ),
        }
        DATASETS_JSON.write_text(json.dumps(registry, indent=2))
        print(f"Registered '{DATASET_NAME}' in datasets.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Scout identity dialogue corpus")
    parser.add_argument("--num", "-n", type=int, default=500,
                        help="Number of dialogues to generate (default: 500)")
    parser.add_argument("--profile", "-p", type=str, default="branch-dev",
                        help="AWS profile for Bedrock (default: branch-dev)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and print without saving")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Fixed temperature (default: random 0.7–1.0 per dialogue)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for situation selection")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = OUTPUT_DIR

    print(f"Scout dialogue generation")
    print(f"  Target:   {args.num} dialogues")
    print(f"  Dataset:  {DATASET_NAME} → {raw_dir}")
    print(f"  Profile:  {args.profile}")
    print(f"  Dry run:  {args.dry_run}")
    print()

    client = make_bedrock_client(args.profile)

    # Load any already-generated rows so we can resume
    existing = [] if args.dry_run else load_existing(raw_dir)
    already = len(existing)
    if already:
        print(f"  Resuming: {already} dialogues already saved, generating {args.num - already} more")
        print()

    # Cycle through situations — repeat if needed to reach target count
    situations = SITUATION_SEEDS.copy()
    random.shuffle(situations)
    while len(situations) < args.num:
        extra = SITUATION_SEEDS.copy()
        random.shuffle(extra)
        situations.extend(extra)
    situations = situations[already:args.num]  # skip already-done slots

    rows = list(existing)
    generated = 0
    failed = 0

    for idx, situation in enumerate(situations):
        temp = args.temperature if args.temperature is not None else random.choices(
            [0.7, 0.85, 1.0], weights=[0.25, 0.5, 0.25]
        )[0]

        global_idx = already + idx + 1
        print(f"[{global_idx}/{args.num}] {situation[:70]}...")

        dialogue = generate_dialogue(client, situation, temperature=temp)

        if not validate_dialogue(dialogue):
            print(f"  -> failed validation, skipping")
            failed += 1
            continue

        text = normalize_dialogue(dialogue)

        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"Situation: {situation}")
            print(f"Temperature: {temp}")
            print(f"{'='*60}")
            print(text)
            print()
            generated += 1
            if generated >= args.num:
                break
        else:
            rows.append({"text": text})
            generated += 1
            # Save incrementally every 10 dialogues so progress survives interruption
            if generated % 10 == 0:
                save_dataset(rows, raw_dir)
                print(f"  -> checkpoint saved ({len(rows)} total)")
            else:
                print(f"  -> ok ({len(rows)} total)")

        if not args.dry_run:
            time.sleep(0.3)

    # Final save
    if not args.dry_run and rows:
        save_dataset(rows, raw_dir)
        register_dataset()

    print()
    print(f"Done.")
    print(f"  Generated this run: {generated}")
    print(f"  Total in dataset:   {len(rows) if not args.dry_run else 'n/a (dry run)'}")
    print(f"  Failed validation:  {failed}")

    if not args.dry_run and rows:
        print()
        print(f"Next: normalize and tokenize via the web UI at /datasets,")
        print(f"or run the training script directly.")


if __name__ == "__main__":
    main()
