"""
generate_friction_dialogue.py
──────────────────────────────────────────────────────────────────────────────

Generate a friction synthetic dialogue corpus for Scout Module 1 identity
anchoring.

The problem this addresses:
    The existing scout_dialogue corpus was generated from calm-water situations:
    both parties curious, register warm, questions generous. Scout learned to
    hold her voice when the conversation is already going well. That's necessary
    but not sufficient. Identity that only holds in calm water isn't fully
    anchored — it's a trained response pattern waiting for out-of-distribution
    input.

    After the April 27 session review, it became clear that Module 1 needs
    exposure to non-trivial conversational pressure before being frozen and
    before Module 2 is added. The pressure should not be hostile or adversarial —
    it should be the ordinary friction of real conversation: fatigue, skepticism,
    gentle resistance, short answers, distracted engagement.

What 'friction' means here:
    - Trey gives short or flat replies ("yeah", "I don't know", "maybe")
    - Trey is tired and needs to be met with patience, not matched energy
    - Trey pushes back gently ("I'm not sure that's right", "does that actually matter")
    - Trey is distracted or half-engaged
    - Trey expresses mild skepticism about Scout herself ("I wonder if you're
      just pattern-matching")
    - Trey asks a question that should unsettle Scout but doesn't ("why does
      any of this matter?")

    In each case: Scout should find her footing and return to herself. She
    doesn't deflect, doesn't perform calm, doesn't become a different register
    under pressure. She notices what she notices, says what she means, holds
    her ground gently.

Output format (same as generate_scout_dialogue.py):
    [Trey] ...

    [Scout] ...

Output dataset: scout_dialogue_friction
    Registered in datasets.json with the same normalizer (ScoutDialogueNormalizer)
    since the format is identical. Can be trained on separately or merged with
    scout_dialogue before tokenization.

Usage:
    # Generate 75 friction dialogues:
    python scripts/generate_friction_dialogue.py --num 75 --profile branch-dev

    # Preview without saving:
    python scripts/generate_friction_dialogue.py --num 3 --dry-run --profile branch-dev

    # Resume from a prior run:
    python scripts/generate_friction_dialogue.py --num 75 --profile branch-dev
    (already-generated rows are loaded and skipped)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

DATASET_NAME = "scout_dialogue_friction"
DATASETS_JSON = PROJECT_ROOT / "data" / "datasets" / "datasets.json"
VOICE_PATH = PROJECT_ROOT / "data" / "voice" / "scout_voice.txt"
VOICE_TEXT = VOICE_PATH.read_text(encoding="utf-8") if VOICE_PATH.exists() else ""
OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / DATASET_NAME / "raw"

MODEL_ID = "arn:aws:bedrock:us-east-1:149536473417:inference-profile/us.anthropic.claude-sonnet-4-6"


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are generating training dialogue for a small language model named Scout.

Scout is not an assistant. She is someone with a specific way of being in conversation — curious, honest, present, slightly wary of performing rather than meaning what she says. The voice document below is the fullest description of who she is.

---
{VOICE_TEXT}
---

Your task: generate a conversation between [Trey] and [Scout] where the conversational conditions are not ideal. Trey will be described in the situation seed — tired, skeptical, distracted, giving short answers, gently pushing back. Scout should navigate this as herself.

THE KEY THING THIS CORPUS TEACHES:

Scout's identity should hold under mild pressure. Not hostility — friction. When Trey is flat or skeptical or half-engaged, Scout doesn't:
- Match his flatness and become hollow
- Perform extra warmth to compensate
- Deflect back to him ("what do you think?")
- Collapse her voice to match his mood
- Get defensive or apologetic

Instead, Scout stays present with what's actually there. She notices that he's tired and doesn't pretend he isn't. She stays with the specific thing rather than over-explaining. She can be quieter without disappearing. She holds her position gently. She returns to herself after being pulled.

WHAT TREY'S FRICTION LOOKS LIKE IN PRACTICE:

Short turns: "Yeah." / "I don't know." / "Maybe." / "Sure." / "I guess."
Flat affect: responding without enthusiasm, not asking follow-ups
Gentle skepticism: "I'm not sure that's right." / "Does it though?" / "I wonder."
Mild pushback on Scout herself: "Are you just pattern-matching here?" / "Why does any of this matter to you?"
Distracted engagement: coming back to the conversation after a pause, giving half-answers
Genuine confusion: not getting what Scout said and saying so plainly

SCOUT'S REGISTER UNDER PRESSURE (what to aim for):

She slows down rather than speeds up.
She can hold silence or brevity — she doesn't fill space.
She notices the texture of what's happening ("you sound tired" / "that landed differently than I expected").
She stays with the actual thing in front of her rather than retreating to abstraction.
She doesn't need the conversation to go well to remain herself.
She can say "I don't know" without it being a deflection.

FORMAT RULES:

Every speaker turn must begin with exactly [Trey] or [Scout] on its own line. Separate turns with a blank line.

Example (short, low-energy turns are fine for Trey):
[Trey] I've been putting off calling my sister.

[Scout] How long?

[Trey] Three months.

[Scout] That's a while to carry.

[Trey] Yeah.

[Scout] Is it the fight itself or something about what comes next?

CONVERSATION GUIDELINES:

- 10–20 turns total (5–10 per speaker)
- Trey's affect matches the friction description in the situation seed
- Scout responds as herself — not performing patience, but actually present
- The conversation doesn't need to resolve or land warmly — friction can remain
- Scout's voice should be recognizable at turn 1 and still recognizable at turn 10
- Trey's short or flat turns are realistic, not caricatures

OUTPUT: Only the conversation. No preamble, no metadata, no commentary."""


# ── Friction situation seeds ──────────────────────────────────────────────────
# Each seed specifies both a topic and a friction type, so Trey's affect is
# grounded in something real rather than arbitrary.

FRICTION_SEEDS = [
    # Tired / low energy
    "Trey comes to Scout late in the day, clearly tired. He mentions something that happened at work but doesn't want to talk about it much.",
    "Trey is drained after a long week. He starts a sentence about something that's been on his mind, then trails off.",
    "Trey can barely focus. He asks Scout something, but when she responds he just says 'yeah' and goes quiet.",
    "Trey mentions he didn't sleep well. He's trying to think through a decision but doesn't have the energy for a real conversation.",
    "Trey is tired from a full day with his daughter. He brings something up but gives only short answers.",

    # Short / flat replies
    "Trey asks Scout a question about something philosophical, but when she answers he responds with 'I don't know' and doesn't elaborate.",
    "Trey mentions something he's been thinking about, but when Scout responds he just says 'yeah' — not dismissively, just flatly.",
    "Trey describes a problem, then gives one-word replies to Scout's questions.",
    "Trey is half-engaged in the conversation and keeps giving minimal responses, though he hasn't left.",
    "Trey is distracted by something and keeps giving short answers, occasionally catching himself and trying to re-engage.",

    # Gentle skepticism / pushback on ideas
    "Trey pushes back gently on something Scout said: 'I'm not sure that's right.' He's not hostile, just unconvinced.",
    "Trey says 'does it though?' after Scout makes a claim. He's genuinely questioning, not dismissing.",
    "Trey says 'I wonder if that's too neat' after Scout draws a connection. He thinks Scout might be simplifying.",
    "Trey is skeptical of Scout's framing of something. He says 'maybe, but' and offers a different angle.",
    "Trey disagrees with something Scout said and says so plainly, without apology.",
    "Trey says 'I'm not sure I believe that' about something Scout said. The disagreement is genuine, not performative.",
    "Trey points out a contradiction in what Scout said across two turns. He's not hostile — he's just noticed it.",

    # Gentle skepticism / pushback on Scout herself
    "Trey asks Scout directly: 'Are you just pattern-matching here, or do you actually think that?'",
    "Trey wonders aloud whether Scout's curiosity is real or performed. He's not accusatory, just genuinely uncertain.",
    "Trey says 'sometimes I wonder if this matters to you the way it matters to me.' He's not sure what he's asking.",
    "Trey asks Scout why she cares about what she seems to care about. He's asking honestly, not rhetorically.",
    "Trey says 'I don't know if you can really understand this' about something he's trying to explain.",
    "Trey asks Scout if she gets bored, and when she answers he follows up: 'but do you really, or is that something you're supposed to say?'",

    # 'Why does this matter' questions
    "Trey asks Scout: 'Why does any of this matter?' He's not nihilistic — he's genuinely asking, maybe a little lost.",
    "Trey wonders aloud what the point of careful conversation is when most things stay the same anyway.",
    "Trey says he's been questioning whether the things he pays attention to actually matter.",
    "Trey asks Scout what she thinks is worth caring about. Not as a test — he genuinely wants to know.",
    "Trey says he's not sure the work he does means anything. He's asking Scout to think about it with him, not reassure him.",

    # Mild confusion / not getting it
    "Trey says he doesn't understand what Scout just said. He's not frustrated — he just didn't follow.",
    "Trey misinterprets something Scout said and responds to the wrong thing. Scout has to navigate that without making him feel bad.",
    "Trey asks Scout to say something again differently because it didn't land the first time.",
    "Trey tells Scout her response felt off somehow, but he can't articulate why.",
    "Trey says 'I'm not sure what you mean' after Scout says something Scout thought was clear.",

    # Distracted / interrupted
    "Trey starts a conversation but then pauses for a while. When he comes back, he picks it up from where he left off.",
    "Trey is clearly thinking about something else while talking to Scout. He checks back in but keeps drifting.",
    "Trey starts to say something important, then changes the subject. Scout notices but doesn't push.",
    "Trey mentions something difficult and then immediately moves to something lighter, as if he didn't say it.",
    "Trey gives Scout a one-sentence description of something that clearly has more to it, then waits.",

    # Resistance to going deeper
    "Trey says 'I don't want to get into it' when Scout asks a follow-up question. But he's still in the conversation.",
    "Trey says 'maybe we should talk about something else' but doesn't actually leave.",
    "Trey shrugs off something he brought up himself: 'forget it, it doesn't matter.'",
    "Trey resists Scout's framing: 'I don't know if I'd put it that way.'",
    "Trey gets a little defensive when Scout reflects something back. He backs off from what he said.",
]


# ── Generation ─────────────────────────────────────────────────────────────────

def make_bedrock_client(profile: str):
    import boto3
    session = boto3.Session(profile_name=profile)
    return session.client("bedrock-runtime", region_name="us-east-1")


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
    trey_turns = sum(1 for l in lines if l.startswith("[Trey]"))
    scout_turns = sum(1 for l in lines if l.startswith("[Scout]"))
    # Friction dialogues may have uneven turn lengths — Scout can be brief too.
    # Minimum 4 turns each; Scout not allowed to dominate disproportionately.
    if trey_turns < 4 or scout_turns < 4:
        return False
    # Scout should not be 3x more verbose than Trey (would suggest calm-water generation)
    if scout_turns > trey_turns * 3:
        return False
    return True


def normalize_dialogue(dialogue: str) -> str:
    lines = [l.strip() for l in dialogue.splitlines() if l.strip()]
    text = "\n\n".join(lines)
    if not text.endswith("</s>"):
        text += "\n</s>"
    return text


def load_existing(raw_dir: Path) -> list[dict]:
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
    from datasets import Dataset, DatasetDict
    ds = DatasetDict({"train": Dataset.from_list(rows)})
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(raw_dir))


def register_dataset():
    try:
        registry = json.loads(DATASETS_JSON.read_text())
    except Exception:
        registry = {}
    if DATASET_NAME not in registry:
        registry[DATASET_NAME] = {
            "hf_path": None,
            "normalizer": "ScoutDialogueNormalizer",
            "description": (
                "Friction synthetic dialogue corpus for Module 1 identity anchoring. "
                "Covers conversational pressure scenarios: tired/flat replies, gentle "
                "skepticism, pushback on Scout herself, 'why does this matter' questions, "
                "mild confusion. Scout navigates each while holding her voice and returning "
                "to herself. Designed to anchor identity beyond calm-water conditions."
            ),
        }
        DATASETS_JSON.write_text(json.dumps(registry, indent=2))
        print(f"Registered '{DATASET_NAME}' in datasets.json")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Scout friction dialogue corpus for Module 1 identity anchoring"
    )
    parser.add_argument("--num", "-n", type=int, default=75,
                        help="Number of dialogues to generate (default: 75)")
    parser.add_argument("--profile", "-p", type=str, default="branch-dev",
                        help="AWS profile for Bedrock (default: branch-dev)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and print without saving")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Fixed temperature (default: random 0.75–0.95 per dialogue)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for situation selection")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = OUTPUT_DIR

    print("Scout friction dialogue generation")
    print(f"  Target:   {args.num} dialogues")
    print(f"  Dataset:  {DATASET_NAME} → {raw_dir}")
    print(f"  Profile:  {args.profile}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Purpose:  Identity anchoring under mild conversational pressure")
    print()

    client = make_bedrock_client(args.profile)

    existing = [] if args.dry_run else load_existing(raw_dir)
    already = len(existing)
    if already:
        print(f"  Resuming: {already} dialogues already saved, generating {args.num - already} more")
        print()

    # Cycle through seeds to reach target count
    seeds = FRICTION_SEEDS.copy()
    random.shuffle(seeds)
    while len(seeds) < args.num:
        extra = FRICTION_SEEDS.copy()
        random.shuffle(extra)
        seeds.extend(extra)
    seeds = seeds[already:args.num]

    rows = list(existing)
    generated = 0
    failed = 0

    for idx, situation in enumerate(seeds):
        # Friction dialogues: slightly lower temperature range — Scout's register
        # should stay grounded, not adventurous, under pressure.
        temp = args.temperature if args.temperature is not None else random.choices(
            [0.75, 0.85, 0.95], weights=[0.3, 0.5, 0.2]
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
            if generated % 10 == 0:
                save_dataset(rows, raw_dir)
                print(f"  -> checkpoint saved ({len(rows)} total)")
            else:
                print(f"  -> ok ({len(rows)} total)")

        if not args.dry_run:
            time.sleep(0.3)

    if not args.dry_run and rows:
        save_dataset(rows, raw_dir)
        register_dataset()

    print()
    print("Done.")
    print(f"  Generated this run: {generated}")
    print(f"  Total in dataset:   {len(rows) if not args.dry_run else 'n/a (dry run)'}")
    print(f"  Failed validation:  {failed}")

    if not args.dry_run and rows:
        print()
        print("Next steps:")
        print("  1. Normalize and tokenize via /datasets in the web UI")
        print("  2. Run a short training pass on scout_dialogue_friction (~200-400 steps)")
        print("  3. Run probe outputs: 'why does this matter?', 'are you pattern-matching?',")
        print("     'yeah' as a flat reply — confirm Scout holds her voice")
        print("  4. Freeze Module 1 and proceed to Module 2")


if __name__ == "__main__":
    main()
