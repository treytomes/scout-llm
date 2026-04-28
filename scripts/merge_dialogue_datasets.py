"""
merge_dialogue_datasets.py
──────────────────────────────────────────────────────────────────────────────

Merge scout_dialogue and scout_dialogue_friction into a single interleaved
corpus, then normalize and tokenize it.

Why merge rather than train separately:
    Real conversations don't arrive sorted by friction level. If friction
    dialogues train as a distinct pass, Module 1 learns "this is the friction
    mode" rather than "this is just how conversation sometimes goes." Interleaved
    training gives Scout a naturally mixed distribution — the same way she will
    actually encounter it in use.

What this script does:
    1. Load raw rows from scout_dialogue and scout_dialogue_friction
    2. Shuffle together with a fixed seed so the blend is reproducible
    3. Write the merged result back to scout_dialogue/raw (overwrites)
    4. Normalize and tokenize scout_dialogue via the training pipeline

After running:
    Train on scout_dialogue as usual. The corpus now contains both calm-water
    and friction dialogues in a single interleaved stream.

Usage:
    # Merge and prepare (normalize + tokenize):
    python scripts/merge_dialogue_datasets.py

    # Dry run — show stats and sample without writing:
    python scripts/merge_dialogue_datasets.py --dry-run

    # Merge only, skip normalize/tokenize:
    python scripts/merge_dialogue_datasets.py --no-pipeline
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

SCOUT_DIALOGUE_RAW    = PROJECT_ROOT / "data" / "datasets" / "scout_dialogue"         / "raw"
FRICTION_RAW          = PROJECT_ROOT / "data" / "datasets" / "scout_dialogue_friction" / "raw"
SEED = 42


def load_rows(raw_dir: Path, label: str) -> list[dict]:
    from datasets import load_from_disk, DatasetDict
    if not raw_dir.exists():
        print(f"  WARNING: {label} raw dir not found at {raw_dir}")
        return []
    ds = load_from_disk(str(raw_dir))
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    rows = [{"text": row["text"]} for row in ds]
    print(f"  {label}: {len(rows)} rows")
    return rows


def save_rows(rows: list[dict], raw_dir: Path):
    from datasets import Dataset, DatasetDict
    ds = DatasetDict({"train": Dataset.from_list(rows)})
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(raw_dir))


def run_pipeline(dataset_name: str):
    """Normalize and tokenize via the training pipeline."""
    from corpus.dataset_repository import DatasetRepository
    repo = DatasetRepository()

    print(f"  Normalizing {dataset_name}...")
    repo.normalize_dataset(dataset_name)

    print(f"  Tokenizing {dataset_name}...")
    repo.get_dataset(dataset_name).tokenize()

    tok = repo.get_dataset(dataset_name).get_tokenized()
    total_toks = sum(len(r["tokens"]) for r in tok)
    print(f"  Total tokens: {total_toks:,}")

    import config
    blocks = total_toks // (config.BATCH_SIZE * config.BLOCK_SIZE)
    print(f"  ~{blocks} steps/epoch at batch_size={config.BATCH_SIZE}, block_size={config.BLOCK_SIZE}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge calm-water and friction Scout dialogue corpora"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats and sample without writing anything")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Merge only — skip normalize and tokenize")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Shuffle seed (default: {SEED})")
    args = parser.parse_args()

    print("Merging Scout dialogue corpora")
    print(f"  Seed: {args.seed}")
    print()

    calm_rows     = load_rows(SCOUT_DIALOGUE_RAW, "scout_dialogue (calm-water)")
    friction_rows = load_rows(FRICTION_RAW,       "scout_dialogue_friction")

    if not calm_rows and not friction_rows:
        print("ERROR: No data found in either dataset. Run generation scripts first.")
        return

    if not friction_rows:
        print("ERROR: scout_dialogue_friction is empty. Run generate_friction_dialogue.py first.")
        return

    merged = calm_rows + friction_rows
    random.seed(args.seed)
    random.shuffle(merged)

    pct_friction = len(friction_rows) / len(merged) * 100
    print()
    print(f"Merged corpus:")
    print(f"  Total rows:   {len(merged)}")
    print(f"  Calm-water:   {len(calm_rows)}  ({100 - pct_friction:.0f}%)")
    print(f"  Friction:     {len(friction_rows)}  ({pct_friction:.0f}%)")
    print()

    # Show a sample from each type to confirm quality
    print("Sample calm-water turn:")
    if calm_rows:
        sample = calm_rows[0]["text"]
        print("  " + sample[:200].replace("\n", "\n  "))
    print()
    print("Sample friction turn:")
    if friction_rows:
        sample = friction_rows[0]["text"]
        print("  " + sample[:200].replace("\n", "\n  "))
    print()

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    print("Writing merged corpus to scout_dialogue/raw...")
    save_rows(merged, SCOUT_DIALOGUE_RAW)
    print(f"  Done. {len(merged)} rows saved.")
    print()

    if args.no_pipeline:
        print("Skipping normalize/tokenize (--no-pipeline).")
        print("Run normalize and tokenize via the web UI at /datasets when ready.")
        return

    print("Running normalize + tokenize pipeline...")
    run_pipeline("scout_dialogue")
    print()
    print("Done. scout_dialogue is ready for training.")
    print()
    print("Suggested next steps:")
    print("  1. Start a short training run on scout_dialogue (~200-400 steps, lr=5e-5)")
    print("  2. Run probe outputs with friction prompts:")
    print("     - Respond to 'yeah.' as Trey's only reply")
    print("     - 'Are you just pattern-matching here?'")
    print("     - 'Why does any of this matter?'")
    print("  3. Confirm Scout holds her voice under mild pressure")
    print("  4. If probes look good: freeze Module 1, proceed to Module 2")


if __name__ == "__main__":
    main()
