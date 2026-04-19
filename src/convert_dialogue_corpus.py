"""
convert_dialogue_corpus.py

One-off script: converts tinystories_dialogue/ text files into a
HuggingFace Dataset saved at data/datasets/tinystories_dialogue/raw.

Run once before normalizing and tokenizing via the web UI or pipeline.
"""

import sys
from pathlib import Path

import datasets as hf_datasets

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "data/corpus/tinystories_dialogue"
OUTPUT_DIR = PROJECT_ROOT / "data/datasets/tinystories_dialogue/raw"


def load_dialogues(corpus_dir: Path) -> list[dict]:
    records = []
    for path in sorted(corpus_dir.glob("dialogue_*.txt")):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        # Strip leading metadata lines (# comments) and blank lines after them
        body_lines = []
        past_header = False
        for line in lines:
            if not past_header:
                if line.startswith("#"):
                    continue
                past_header = True
            body_lines.append(line)
        text = "\n".join(body_lines).strip()
        if text:
            records.append({"text": text})
    return records


def main():
    if not CORPUS_DIR.exists():
        print(f"Corpus directory not found: {CORPUS_DIR}")
        sys.exit(1)

    print(f"Loading dialogues from {CORPUS_DIR}...")
    records = load_dialogues(CORPUS_DIR)
    print(f"Loaded {len(records)} dialogues")

    dataset = hf_datasets.Dataset.from_list(records)
    dataset_dict = hf_datasets.DatasetDict({"train": dataset})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}...")
    dataset_dict.save_to_disk(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()