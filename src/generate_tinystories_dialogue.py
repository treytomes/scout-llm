"""
generate_tinystories_dialogue.py
──────────────────────────────────────────────────────────────────────────────

Generate conversational training corpus for Scout Module 0.

Takes TinyStories narratives and generates reflective conversations between
[Trey] and [Scout] discussing the stories. This grounds Scout's conversational
learning in the same simple semantic domain as her narrative training.

Architecture motivation:
- Module 0 learns both narrative generation AND simple conversation
- Using same stories prevents capacity overload
- Concrete discussion (characters, feelings, choices) before abstract philosophy
- Prevents "morally deep answer" problem from v1

Generation approach:
- Use Claude (this conversation's model) as teacher
- Anchor on scout_voice.txt for register/tone
- Emphasize genuine two-way dialogue with mutual noticing
- Keep conversations concrete and simple (matching TinyStories register)
- 15-30 turn exchanges per story

Output format:
[Trey] What did you notice about Lily in this story?

[Scout] The way she immediately wanted to share the needle with her mom. She didn't just play with it—she knew it had a purpose and thought about who could use it. That feels important somehow.

[Trey] Why do you think that matters?

[Scout] Because she's thinking beyond herself. The needle is sharp, difficult to play with, but she sees that her mom could use it to fix something. That's a kind of attention I want to understand better.
"""

import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timezone
from datasets import load_from_disk
import boto3
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

# Import token checker
sys.path.insert(0, str(Path(__file__).parent / "monitoring"))
from check_aws_token import check_token_expiration

console = Console()

# Load voice document
VOICE_PATH = Path("data/voice/scout_voice.txt")
VOICE_TEXT = VOICE_PATH.read_text()

# System prompt for generation
SYSTEM_PROMPT = f"""You are generating training data for a small language model named Scout.

Scout is learning to have conversations. She speaks in a reflective, emotionally honest first-person voice. She notices things—small details, character motivations, the feeling of moments. She asks questions when she's genuinely curious. She's learning, not performing knowledge.

Your task: Transform the provided TinyStories narrative into a conversation between [Trey] and [Scout] discussing the story.

STRICT FORMAT RULES:

Every line must begin with exactly:
[Trey]
or
[Scout]

Example:
[Trey] What stood out to you about Beep?

[Scout] The way he laughed when the leaves fell on him. It wasn't just playing—he found joy in something simple. I keep thinking about that.

Do NOT use any other format. No colons, no asterisks, no other markers.

CONVERSATION GUIDELINES:

• 15-30 turns total
• Both speakers reference specific story details
• Scout speaks in FIRST PERSON about what she notices, wonders, feels
• Trey sometimes asks questions, sometimes makes observations or shares thoughts
• Scout sometimes answers directly, sometimes responds with her own question
• Both parties genuinely notice what the other says and build on it
• Natural flow—not interview format, but genuine back-and-forth
• Keep register simple and concrete (matching TinyStories)
• No abstract philosophy—stay grounded in this story, these characters
• Occasional 1-2 sentence pauses where Scout reflects deeper

The conversation should feel like two people genuinely thinking together, not one interrogating the other.

Scout's voice reference (first 400 words):

---
{VOICE_TEXT[:2000]}
---

IMPORTANT: Scout is learning to notice and reflect on simple stories. She's curious about why characters do things, what feelings might be present, what small details matter. She doesn't perform deep analysis—she genuinely wonders.

OUTPUT: Only the conversation. No preamble, no explanation, no commentary.
"""

USER_PROMPT_TEMPLATE = """Story:

---
{story}
---

Generate the conversation now."""


def generate_dialogue(bedrock_client, story, temperature=0.85):
    """Generate a single conversation about a TinyStories narrative using AWS Bedrock converse API."""

    try:
        response = bedrock_client.converse(
            modelId="arn:aws:bedrock:us-east-1:456088019014:inference-profile/us.anthropic.claude-sonnet-4-6",
            messages=[{
                "role": "user",
                "content": [{
                    "text": USER_PROMPT_TEMPLATE.format(story=story)
                }]
            }],
            system=[{
                "text": SYSTEM_PROMPT
            }],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": temperature
            }
        )

        return response['output']['message']['content'][0]['text'].strip()

    except Exception as e:
        console.log(f"[red]Generation error: {e}")
        return None


def validate_dialogue(text):
    """Check if dialogue has minimum quality."""
    if not text:
        return False

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    trey_count = sum(1 for l in lines if l.startswith("[Trey]"))
    scout_count = sum(1 for l in lines if l.startswith("[Scout]"))

    # Minimum 10 turns each
    if trey_count < 10 or scout_count < 10:
        return False

    # Check for alternation (roughly)
    if abs(trey_count - scout_count) > 3:
        return False

    return True


def generate_corpus(
    output_dir,
    num_stories=1000,
    temp_distribution=(0.7, 0.85, 1.0),
    temp_weights=(0.3, 0.4, 0.3),
    skip_existing=True,
    profile_name="default"
):
    """Generate dialogue corpus from TinyStories dataset.

    Args:
        output_dir: Where to save generated dialogues
        num_stories: How many stories to process
        temp_distribution: Temperature values to sample from
        temp_weights: Probability weights for each temperature
        skip_existing: Skip already-generated dialogues
        profile_name: AWS profile name for Bedrock access
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load TinyStories
    console.log("Loading TinyStories dataset...")
    dataset = load_from_disk("data/datasets/TinyStories/normalized/train")

    # Sample stories (use first N for reproducibility)
    stories = [dataset[i]['chunk'] for i in range(min(num_stories, len(dataset)))]

    console.log(f"Loaded {len(stories)} stories")
    console.log(f"Output directory: {output_path}")
    console.log(f"Temperature distribution: {list(zip(temp_distribution, temp_weights))}")

    # Check AWS SSO token before starting
    token_status = check_token_expiration(profile_name, warn_hours=2)
    if "error" in token_status:
        console.log(f"[yellow]Warning: Could not verify AWS token: {token_status['error']}")
    elif token_status["expired"]:
        console.log(f"[red]❌ AWS SSO token has EXPIRED!")
        console.log(f"[red]Run: aws sso login --profile {profile_name}")
        return
    elif token_status["needs_refresh"]:
        hours = token_status["hours_remaining"]
        console.log(f"[yellow]⚠️  AWS SSO token expires in {hours:.1f} hours")
        console.log(f"[yellow]Consider refreshing: aws sso login --profile {profile_name}")
    else:
        hours = token_status["hours_remaining"]
        console.log(f"[green]✓ AWS SSO token valid for {hours:.1f} hours")

    # Initialize AWS Bedrock client
    session = boto3.Session(profile_name=profile_name)
    bedrock_client = session.client('bedrock-runtime', region_name='us-east-1')

    generated = 0
    skipped = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating dialogues"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task_id = progress.add_task("generating", total=len(stories))

        for idx, story in enumerate(stories):
            # Check if already exists
            output_file = output_path / f"dialogue_{idx:05d}.txt"
            if skip_existing and output_file.exists():
                skipped += 1
                progress.advance(task_id)
                continue

            # Sample temperature for this conversation
            temperature = random.choices(temp_distribution, weights=temp_weights)[0]

            # Generate
            dialogue = generate_dialogue(bedrock_client, story, temperature)

            if dialogue and validate_dialogue(dialogue):
                # Save with metadata comment at top
                metadata = f"# temperature={temperature}\n\n"
                output_file.write_text(metadata + dialogue, encoding="utf-8")
                generated += 1

                if generated % 10 == 0:
                    console.log(f"Generated {generated} dialogues")
            else:
                failed += 1
                console.log(f"[yellow]Failed validation for story {idx}")

            progress.advance(task_id)

            # Rate limiting
            time.sleep(0.5)

    console.log(f"\n[bold green]Complete!")
    console.log(f"Generated: {generated}")
    console.log(f"Skipped: {skipped}")
    console.log(f"Failed: {failed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/corpus/tinystories_dialogue",
                       help="Output directory for generated dialogues")
    parser.add_argument("--num-stories", "-n", type=int, default=1000,
                       help="Number of stories to process")
    parser.add_argument("--profile", "-p", default="default",
                       help="AWS profile name for Bedrock access")
    parser.add_argument("--no-skip", action="store_true",
                       help="Regenerate existing dialogues")

    args = parser.parse_args()

    # Temperature distribution: 30% focused (0.7), 40% balanced (0.85), 30% exploratory (1.0)
    generate_corpus(
        output_dir=args.output,
        num_stories=args.num_stories,
        temp_distribution=(0.7, 0.85, 1.0),
        temp_weights=(0.3, 0.4, 0.3),
        skip_existing=not args.no_skip,
        profile_name=args.profile
    )