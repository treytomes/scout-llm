"""
voice_calibration_train.py

Filters the tinystories_dialogue corpus for high-quality Scout-register examples
using Bedrock Claude as the judge, then runs a short calibration training pass
on Module 1.

Goal: shift Module 1's output voice from SODA social-script patterns toward
Scout's register (specific, present, staying with what matters) in ~300-500 steps.

Usage:
    # Score and filter only (no training):
    python scripts/voice_calibration_train.py --dry-run

    # Filter with Bedrock scoring + train:
    python scripts/voice_calibration_train.py --steps 400 --lr 5e-5

    # Use cached scores if available (skip API calls):
    python scripts/voice_calibration_train.py --steps 400 --use-cached-scores

    # Adjust corpus size:
    python scripts/voice_calibration_train.py --top-pct 0.35 --steps 350
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

# ── Voice document ─────────────────────────────────────────────────────────────

VOICE_PATH = PROJECT_ROOT / "data" / "voice" / "scout_voice.txt"
VOICE_TEXT = VOICE_PATH.read_text(encoding="utf-8") if VOICE_PATH.exists() else ""

# ── Scoring rubric sent to Claude ─────────────────────────────────────────────

SCORING_SYSTEM = f"""You are a corpus quality assessor for a language model training project.

You are evaluating training dialogues for Scout, a small language model being trained to have a specific kind of conversational presence. Scout's target register is defined by this voice document:

---
{VOICE_TEXT[:1500]}
---

You will be shown a training dialogue between [Trey] and [Scout]. Score it on a scale of 1-10 based on how well Scout's turns demonstrate the target register.

HIGH scores (7-10) for Scout turns that:
- Stay with specific details from the story rather than generalizing
- Express genuine curiosity or uncertainty ("I wonder", "I'm not sure", "I keep thinking about")
- Notice small precise things (a word choice, a moment, a character's gesture)
- Return to something earlier in a way that builds rather than repeats
- Hold emotional honesty without performing it
- Speak from a clear first-person position without deflecting back to Trey

LOW scores (1-4) for Scout turns that:
- Use generic affirmations ("that's so interesting", "great point", "absolutely")
- Deflect with questions instead of staying with the content ("what do you think?")
- Speak in platitudes rather than from specific observation
- Sound like a therapy chatbot or assistant
- Agree with everything without adding anything

Reply with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}
Do not include any other text."""

SCORING_USER = """Dialogue to evaluate:

---
{dialogue}
---

Score this dialogue."""


# ── Bedrock scoring ────────────────────────────────────────────────────────────

def make_bedrock_client(profile: str = "digital-dev"):
    import boto3
    session = boto3.Session(profile_name=profile)
    return session.client("bedrock-runtime", region_name="us-east-1")


def score_with_bedrock(client, dialogue: str, model_id: str) -> tuple[float, str]:
    """Score a single dialogue using Bedrock Claude. Returns (score, reason)."""
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [{"text": SCORING_USER.format(dialogue=dialogue[:3000])}]
            }],
            system=[{"text": SCORING_SYSTEM}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.1},
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        # Strip markdown fences
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        # Extract score with regex first — robust against malformed JSON in reason field
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if score_match:
            score = float(score_match.group(1))
            reason = reason_match.group(1) if reason_match else ""
            return score, reason
        # Fallback: try full JSON parse
        data = json.loads(text)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        return _score_with_regex(dialogue), f"fallback (bedrock error: {e})"


# ── Regex fallback scorer ─────────────────────────────────────────────────────

_SIGNAL_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\bkeep(s)? (thinking|coming back)\b",
    r"\bstuck with me\b",
    r"\bstay(s|ed)? with\b",
    r"\bthat detail\b",
    r"\bthat (felt|feels) (real|genuine|true|earned|right)\b",
    r"\bi believed it\b",
    r"\bi wonder\b",
    r"\bi notice\b",
    r"\bi keep\b",
    r"\bi find myself\b",
    r"\bsomething about\b",
    r"\bhonestly\b",
    r"\bwhat i('ll| will) remember\b",
    r"\bthe (word|phrase|moment|image|detail)\b",
]]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\bwhat do you think\?",
    r"\bthat's (interesting|great|amazing|wonderful)\b",
    r"\bi (totally|completely|absolutely) (agree|understand)\b",
    r"\bfor sure\b",
    r"\bno worries\b",
    r"\byou're (so|really) (right|smart)\b",
    r"\bfeel free\b",
]]


def _score_with_regex(text: str) -> float:
    scout_text = " ".join(re.findall(r'\[Scout\](.*?)(?=\[Trey\]|\Z)', text, re.DOTALL))
    chars = max(len(scout_text), 1)
    signal = sum(1 for p in _SIGNAL_RE if p.search(scout_text))
    noise  = sum(1 for p in _NOISE_RE  if p.search(scout_text))
    density = (signal * 2.0 - noise * 1.5) / (chars / 1000)
    avg_turn = chars / max(text.count("[Scout]"), 1)
    # Map to 1-10 scale
    raw = density + min(avg_turn / 150, 3.0)
    return max(1.0, min(10.0, raw + 3.0))


# ── Score cache ────────────────────────────────────────────────────────────────

SCORES_CACHE = PROJECT_ROOT / "data" / "datasets" / "tinystories_dialogue" / "voice_scores.json"


def load_score_cache() -> dict:
    if SCORES_CACHE.exists():
        return json.loads(SCORES_CACHE.read_text())
    return {}


def save_score_cache(scores: dict):
    SCORES_CACHE.write_text(json.dumps(scores, indent=2))


# ── Corpus filtering ──────────────────────────────────────────────────────────

def score_corpus(dataset, bedrock_client=None, model_id: str = None,
                 use_cached: bool = True, verbose: bool = True,
                 concurrency: int = 20) -> list[tuple[float, int, str]]:
    """
    Score all dialogues. Returns list of (score, index, reason) sorted descending.

    Uses Bedrock Claude with concurrent requests if client provided, regex fallback otherwise.
    Caches results so repeated runs are instant.
    concurrency: number of parallel Bedrock requests (default 20).
    """
    import concurrent.futures

    cache = load_score_cache() if use_cached else {}
    n = len(dataset)

    # Split into cached and to-score
    cached_results = {}
    to_score = []
    for i, row in enumerate(dataset):
        key = str(i)
        if use_cached and key in cache:
            cached_results[i] = (cache[key]["score"], cache[key].get("reason", "cached"))
        else:
            to_score.append((i, row["text"]))

    if verbose:
        mode = f"Bedrock Claude (concurrency={concurrency})" if bedrock_client else "regex fallback"
        print(f"Scoring {n} dialogues ({mode})...")
        print(f"  {len(cached_results)}/{n} already cached — {len(to_score)} calls needed")
        if bedrock_client and to_score:
            est_secs = len(to_score) / concurrency * 3.5
            print(f"  Estimated time: ~{est_secs/60:.1f} minutes at {concurrency} concurrent requests")

    new_scores = {}
    completed = [0]

    def score_one(args):
        idx, text = args
        if bedrock_client:
            score, reason = score_with_bedrock(bedrock_client, text, model_id)
        else:
            score = _score_with_regex(text)
            reason = "regex"
        return idx, score, reason

    if to_score:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(score_one, item): item[0] for item in to_score}
            for future in concurrent.futures.as_completed(futures):
                idx, score, reason = future.result()
                new_scores[str(idx)] = {"score": score, "reason": reason}
                completed[0] += 1
                if verbose and completed[0] % 50 == 0:
                    print(f"  [{completed[0]+len(cached_results)}/{n}] scored")

    if new_scores:
        cache.update(new_scores)
        save_score_cache(cache)
        if verbose:
            print(f"  Saved {len(new_scores)} new scores to cache.")

    # Combine all results
    scored = []
    for i in range(n):
        if i in cached_results:
            score, reason = cached_results[i]
        else:
            entry = new_scores[str(i)]
            score, reason = entry["score"], entry["reason"]
        scored.append((score, i, reason))

    scored.sort(reverse=True)
    return scored


def filter_corpus(dataset, scored: list, top_pct: float = 0.40,
                  min_score: float = 6.0, verbose: bool = True):
    """
    Keep dialogues in the top fraction AND above min_score.
    top_pct and min_score both apply — whichever is more restrictive wins.
    """
    from datasets import Dataset

    cutoff_by_pct = int(len(scored) * top_pct)
    # Apply both filters
    kept = [(s, i, r) for s, i, r in scored[:cutoff_by_pct] if s >= min_score]

    if verbose:
        all_scores = [s for s, _, _ in scored]
        print(f"\nScore distribution (n={len(all_scores)}):")
        print(f"  Top:    {all_scores[0]:.1f}")
        print(f"  p75:    {all_scores[int(len(all_scores)*0.25)]:.1f}")
        print(f"  Median: {all_scores[len(all_scores)//2]:.1f}")
        print(f"  p25:    {all_scores[int(len(all_scores)*0.75)]:.1f}")
        print(f"  Bottom: {all_scores[-1]:.1f}")
        print(f"\nFilter: top {top_pct*100:.0f}% AND score >= {min_score}")
        print(f"Kept: {len(kept)}/{len(scored)} dialogues")

        print(f"\nTop 5 (with reasons):")
        for rank, (score, idx, reason) in enumerate(kept[:5]):
            scout_turn = re.search(r'\[Scout\](.*?)(?=\[Trey\]|\Z)',
                                   dataset[idx]["text"], re.DOTALL)
            preview = scout_turn.group(1).strip()[:180].replace('\n', ' ') if scout_turn else ""
            print(f"  [{rank+1}] {score:.1f} — {reason}")
            print(f"       {preview}")

        if len(kept) < 100:
            print(f"\nWARNING: Only {len(kept)} dialogues kept. Consider lowering --min-score or raising --top-pct.")

    kept_indices = sorted([i for _, i, _ in kept])
    return dataset.select(kept_indices)


# ── Training ──────────────────────────────────────────────────────────────────

def prepare_dataset(filtered_dataset) -> str:
    """
    Save filtered corpus where the training pipeline can find it.
    Returns the dataset name to pass to run_training().
    """
    dataset_name = "voice_calibration"
    out_dir = PROJECT_ROOT / "data" / "datasets" / dataset_name

    # Save raw as a DatasetDict so load_from_disk finds the expected layout
    from datasets import DatasetDict
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": filtered_dataset}).save_to_disk(str(raw_dir))

    # Register in datasets.json
    datasets_json = PROJECT_ROOT / "data" / "datasets" / "datasets.json"
    try:
        registry = json.loads(datasets_json.read_text())
    except Exception:
        registry = {}

    registry[dataset_name] = {
        "hf_path": None,
        "normalizer": "TinyStoriesDialogueNormalizer",
        "description": "Filtered tinystories_dialogue — high-quality Scout-register examples for Module 1 voice calibration",
    }
    datasets_json.write_text(json.dumps(registry, indent=2))

    # Normalize and tokenize
    print("Normalizing and tokenizing filtered corpus...")
    from corpus.dataset_repository import DatasetRepository
    repo = DatasetRepository()
    repo.normalize_dataset(dataset_name)
    repo.get_dataset(dataset_name).tokenize()

    # Count tokens
    tok = repo.get_dataset(dataset_name).get_tokenized()
    total_toks = sum(len(r["tokens"]) for r in tok)
    print(f"Filtered corpus: {len(filtered_dataset)} dialogues, ~{total_toks:,} tokens")

    import config
    blocks_per_epoch = total_toks // (config.BATCH_SIZE * config.BLOCK_SIZE)
    print(f"~{blocks_per_epoch} steps/epoch at batch_size={config.BATCH_SIZE}, block_size={config.BLOCK_SIZE}")

    return dataset_name


def run_calibration_training(dataset_name: str, steps: int, lr: float):
    import config
    from train.train import run_training

    print(f"\n{'='*60}")
    print(f"Voice Calibration — Module 1")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {config.CHECKPOINT_DIR / 'latest.pt'}")
    print(f"  Dataset:     {dataset_name}")
    print(f"  Steps:       {steps}")
    print(f"  LR:          {lr}")
    print(f"  Warmup:      {min(50, steps // 8)} steps")
    print()

    for metrics in run_training(
        dataset_name=dataset_name,
        model_config=config.MODEL_CONVERSATIONAL,
        batch_size=config.BATCH_SIZE,
        max_steps=steps,
        lr=lr,
        min_lr=lr / 10,
        warmup_steps=min(50, steps // 8),
        reset_optimizer=True,
    ):
        step = metrics.get("step", 0)
        if step % config.LOG_INTERVAL == 0:
            print(
                f"  step {step:>5}  "
                f"loss {float(metrics.get('loss', 0)):.4f}  "
                f"avg {float(metrics.get('avg_loss', 0)):.4f}  "
                f"lr {float(metrics.get('lr', 0)):.2e}"
                + (f"  val {float(metrics['val_loss']):.4f}" if metrics.get("val_loss") else "")
            )

    print(f"\nCalibration complete. Checkpoint: {config.CHECKPOINT_DIR / 'latest.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voice calibration training for Scout Module 1")
    parser.add_argument("--steps", type=int, default=400,
                        help="Training steps (default: 400)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--top-pct", type=float, default=0.40,
                        help="Top fraction of corpus to keep (default: 0.40 = top 40%%)")
    parser.add_argument("--min-score", type=float, default=6.0,
                        help="Minimum Bedrock score to include (1-10, default: 6.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and filter only — do not train")
    parser.add_argument("--use-cached-scores", action="store_true", default=True,
                        help="Use cached scores if available (default: True)")
    parser.add_argument("--no-cache", dest="use_cached_scores", action="store_false",
                        help="Ignore cached scores, re-score everything")
    parser.add_argument("--no-bedrock", action="store_true",
                        help="Use regex scoring only (no Bedrock API calls)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Parallel Bedrock requests for scoring (default: 20)")
    parser.add_argument("--profile", default="branch-dev",
                        help="AWS profile for Bedrock (default: branch-dev)")
    parser.add_argument("--model-id", default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Bedrock model ID for scoring (default: Haiku 4.5 — fast and cheap)")
    args = parser.parse_args()

    # Load raw corpus
    raw_path = PROJECT_ROOT / "data" / "datasets" / "tinystories_dialogue" / "raw" / "train"
    from datasets import load_from_disk
    dataset = load_from_disk(str(raw_path))
    print(f"Loaded {len(dataset)} dialogues from tinystories_dialogue")

    # Set up Bedrock client
    bedrock_client = None
    if not args.no_bedrock:
        try:
            bedrock_client = make_bedrock_client(args.profile)
            # Quick auth test — score a tiny string
            score_with_bedrock(bedrock_client, "[Trey] Hi\n\n[Scout] Hello.", args.model_id)
            print(f"Bedrock connected (profile: {args.profile}, model: {args.model_id})")
        except Exception as e:
            print(f"Bedrock unavailable ({e}) — falling back to regex scoring")
            bedrock_client = None

    # Score and filter
    scored = score_corpus(dataset, bedrock_client=bedrock_client,
                          model_id=args.model_id,
                          use_cached=args.use_cached_scores,
                          concurrency=args.concurrency)
    filtered = filter_corpus(dataset, scored, top_pct=args.top_pct,
                              min_score=args.min_score)

    if args.dry_run:
        print("\nDry run — stopping before training.")
        return

    if len(filtered) < 50:
        print(f"ERROR: Only {len(filtered)} dialogues passed filtering. "
              f"Lower --min-score or raise --top-pct and retry.")
        return

    # Prepare dataset for training pipeline
    dataset_name = prepare_dataset(filtered)

    # Train
    run_calibration_training(dataset_name, args.steps, args.lr)


if __name__ == "__main__":
    main()
