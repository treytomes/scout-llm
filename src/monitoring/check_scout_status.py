"""
Unified Scout status checker - called by hourly cron job.

Checks:
- Training progress and loss metrics
- AWS SSO token expiration
- Dialogue generation progress (if running)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")
AWS_PROFILE = os.getenv("AWS_PROFILE", "digital-dev")

from check_aws_token import check_token_expiration
from check_aws_costs import get_costs


def check_training_status():
    """Get latest training metrics."""
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / "data/training_log/training_2026-04-17_2.csv"

    if not log_file.exists():
        return None

    # Read last line
    lines = log_file.read_text().strip().split('\n')
    if len(lines) < 2:
        return None

    header = lines[0].split(',')
    latest = lines[-1].split(',')

    return dict(zip(header, latest))


def check_dialogue_generation():
    """Check dialogue generation progress."""
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data/corpus/tinystories_dialogue"

    if not corpus_dir.exists():
        return None

    generated = len(list(corpus_dir.glob("dialogue_*.txt")))

    if generated == 0:
        return None

    return {
        "generated": generated,
        "target": 1000,
        "percent": (generated / 1000) * 100
    }


def main():
    print("\n" + "="*70)
    print("Scout Status Check")
    print("="*70)

    # Training status
    training = check_training_status()
    if training:
        print(f"\n📊 Training Progress:")
        print(f"   Step:     {training.get('step', 'N/A')}")
        print(f"   Loss:     {float(training.get('loss', 0)):.4f}")
        print(f"   Avg Loss: {float(training.get('avg_loss', 0)):.4f}")
        if training.get('val_loss') and training['val_loss']:
            print(f"   Val Loss: {float(training['val_loss']):.4f}")
    else:
        print("\n📊 Training: No active log found")

    # Dialogue generation
    dialogue = check_dialogue_generation()
    if dialogue:
        print(f"\n💬 Dialogue Generation:")
        print(f"   Progress: {dialogue['generated']}/{dialogue['target']} ({dialogue['percent']:.1f}%)")

    # Token status
    token = check_token_expiration(AWS_PROFILE, warn_hours=2)
    print(f"\n🔑 AWS SSO Token ({AWS_PROFILE}):")

    if "error" in token:
        print(f"   ⚠️  Could not verify: {token['error']}")
    elif token["expired"]:
        print(f"   ❌ EXPIRED - Run: aws sso login --profile {AWS_PROFILE}")
    elif token["needs_refresh"]:
        hours = token["hours_remaining"]
        print(f"   ⚠️  Expires in {hours:.1f} hours")
        print(f"   Consider refreshing: aws sso login --profile {AWS_PROFILE}")
    else:
        hours = token["hours_remaining"]
        print(f"   ✓ Valid for {hours:.1f} hours")

    # AWS costs
    from datetime import date, timedelta
    try:
        today = date.today()
        yesterday = today - timedelta(days=1)
        costs = get_costs(AWS_PROFILE, yesterday, today + timedelta(days=1))

        print(f"\n💰 AWS Costs ({AWS_PROFILE}):")
        for day in sorted(costs.keys()):
            day_total = sum(costs[day].values())
            label = "today    " if day == today.isoformat() else "yesterday"
            details = "  ".join(
                f"{s.replace('Amazon ', '')}: ${v:.2f}"
                for s, v in costs[day].items() if v > 0
            )
            print(f"   {label} ({day}): ${day_total:.2f}" + (f"  ({details})" if details else ""))
    except Exception as e:
        print(f"\n💰 AWS Costs: Could not fetch ({e})")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()