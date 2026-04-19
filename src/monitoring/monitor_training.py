"""
Automated training monitor for Scout v2 Module 0 (TinyStories).

Watches for new checkpoints, runs probe prompts, evaluates quality and loss metrics.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "server"))

from model.loader import load_model
from cli_repl import stream_generate


# Test prompts designed for Phase 0b dialogue evaluation.
# Each gives Scout a [Trey] opener and expects a [Scout] response
# in her characteristic voice: first-person, noticing, wondering.
TEST_PROMPTS = [
    "[Trey] What did you notice about the story?\n\n[Scout]",
    "[Trey] That part where she decided to share — what did you make of that?\n\n[Scout]",
    "[Trey] Is there anything in the story that stayed with you?\n\n[Scout]",
    "[Trey] Do you think she made the right choice?\n\n[Scout]",
    "[Trey] What do you think she was feeling at the end?\n\n[Scout]",
]


def get_latest_checkpoint():
    """Get the most recent checkpoint file."""
    checkpoint_dir = Path("data/checkpoints")
    checkpoints = list(checkpoint_dir.glob("model_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def get_checkpoint_step(checkpoint_path):
    """Extract step number from checkpoint filename."""
    stem = checkpoint_path.stem  # e.g., "model_12500"
    if stem == "latest":
        # Load checkpoint to get step
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt.get("step", 0)
    parts = stem.split("_")
    return int(parts[1]) if len(parts) > 1 else 0


def get_latest_loss_metrics():
    """Read the most recent training log for loss metrics."""
    log_dir = Path("data/training_log")
    log_files = list(log_dir.glob("training_*.csv"))
    if not log_files:
        return None

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_log)

    if len(df) == 0:
        return None

    latest = df.iloc[-1]

    # Calculate loss trend (last 10 steps)
    if len(df) >= 10:
        recent = df.tail(10)
        loss_trend = recent['loss'].diff().mean()
        loss_std = recent['loss'].std()
    else:
        loss_trend = 0.0
        loss_std = 0.0

    return {
        "step": int(latest["step"]),
        "loss": float(latest["loss"]),
        "avg_loss": float(latest["avg_loss"]),
        "val_loss": float(latest["val_loss"]) if pd.notna(latest.get("val_loss")) else None,
        "lr": float(latest["lr"]),
        "loss_trend": float(loss_trend),
        "loss_std": float(loss_std),
    }


@torch.no_grad()
def probe_checkpoint(checkpoint_path, prompt, max_tokens=100):
    """Run a single probe on a checkpoint and return the response."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Get max sequence length from model
    max_seq = getattr(model, '_max_seq', 512)

    # Generate response
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = []

    for _ in range(max_tokens):
        logits = model.forward(tokens)
        logits = logits[:, -1, :] / 0.8  # Lower temp for evaluation

        # Top-k sampling
        top_k = 40
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token[0].item()

        generated_ids.append(next_token_id)

        # Stop at period followed by space (end of sentence)
        if len(generated_ids) >= 10:
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if '. ' in decoded or '\n' in decoded:
                break

        tokens = torch.cat([tokens, next_token], dim=1)

        # Truncate if exceeding max sequence length
        if tokens.shape[1] > max_seq:
            tokens = tokens[:, -max_seq:]

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def evaluate_response(prompt, response):
    """
    Evaluate response quality for Phase 0b dialogue probes.
    Checks for first-person voice, Scout register, and dialogue coherence.
    """
    issues = []

    if len(response) < 10:
        issues.append("too_short")

    words = response.lower().split()

    # Repetition: same word 3+ times consecutively
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            issues.append("repetitive")
            break

    response_lower = response.lower()

    # Narrative drift: TinyStories prior reasserting itself
    narrative_phrases = ["once upon a time", "the end", "lived happily", "he ran to", "she ran to"]
    if any(p in response_lower for p in narrative_phrases):
        issues.append("narrative_drift")

    # First-person voice: Scout should speak as "I"
    first_person = any(w in words for w in ["i", "i'm", "i've", "i'd", "i'll", "me", "my", "myself"])
    if not first_person:
        issues.append("no_first_person")

    # Speaker marker bleed: model should not reproduce [Trey] in the response
    if "[trey]" in response_lower:
        issues.append("speaker_bleed")

    # Noticing language: words that signal Scout's characteristic attention
    noticing_words = ["notice", "noticed", "feel", "felt", "think", "thought", "wonder",
                      "wondering", "something", "keep", "stays", "stayed", "interesting"]
    has_noticing = any(w in response_lower for w in noticing_words)

    return {
        "length": len(response),
        "word_count": len(words),
        "issues": issues,
        "first_person": first_person,
        "has_noticing_language": has_noticing,
        "response": response,
    }


def check_loss_concerns(metrics):
    """Check loss metrics for concerning patterns."""
    concerns = []

    if metrics["loss"] > 5.0:
        concerns.append(f"high_loss: {metrics['loss']:.3f}")

    if metrics["loss"] < 0.5:
        concerns.append(f"very_low_loss: {metrics['loss']:.3f} (possible overfitting)")

    # Plateau detection: very low trend and very low std
    if abs(metrics["loss_trend"]) < 0.001 and metrics["loss_std"] < 0.01:
        concerns.append(f"plateau: trend={metrics['loss_trend']:.4f}, std={metrics['loss_std']:.4f}")

    # Explosion detection: large positive trend
    if metrics["loss_trend"] > 0.1:
        concerns.append(f"loss_increasing: trend={metrics['loss_trend']:.4f}")

    # Val loss divergence
    if metrics["val_loss"] is not None:
        gap = metrics["val_loss"] - metrics["avg_loss"]
        if gap > 0.5:
            concerns.append(f"val_gap: {gap:.3f} (possible overfitting)")

    return concerns


def run_evaluation(checkpoint_path):
    """Run full evaluation on a checkpoint."""
    step = get_checkpoint_step(checkpoint_path)

    print(f"\n{'='*70}")
    print(f"Evaluating checkpoint: {checkpoint_path.name} (step {step})")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Get loss metrics
    loss_metrics = get_latest_loss_metrics()
    if loss_metrics:
        print(f"\nLoss Metrics:")
        print(f"  Step:     {loss_metrics['step']}")
        print(f"  Loss:     {loss_metrics['loss']:.4f}")
        print(f"  Avg Loss: {loss_metrics['avg_loss']:.4f}")
        if loss_metrics['val_loss']:
            print(f"  Val Loss: {loss_metrics['val_loss']:.4f}")
        print(f"  LR:       {loss_metrics['lr']:.2e}")
        print(f"  Trend:    {loss_metrics['loss_trend']:.4f}")
        print(f"  Std:      {loss_metrics['loss_std']:.4f}")

        loss_concerns = check_loss_concerns(loss_metrics)
        if loss_concerns:
            print(f"\n  ⚠️  CONCERNS: {', '.join(loss_concerns)}")

    # Run probes
    print(f"\nRunning {len(TEST_PROMPTS)} test prompts...")
    results = []

    for prompt in TEST_PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        try:
            response = probe_checkpoint(checkpoint_path, prompt, max_tokens=100)
            eval_result = evaluate_response(prompt, response)
            results.append({
                "prompt": prompt,
                "evaluation": eval_result,
            })

            print(f"  Response: \"{response}\"")
            if eval_result["issues"]:
                print(f"  ⚠️  Issues: {', '.join(eval_result['issues'])}")
        except Exception as e:
            import traceback
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
            results.append({
                "prompt": prompt,
                "error": str(e),
            })

    # Aggregate quality assessment
    print(f"\n{'='*70}")
    print("Quality Assessment:")

    all_issues = []
    for r in results:
        if "evaluation" in r:
            all_issues.extend(r["evaluation"]["issues"])

    if all_issues:
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        print(f"  Issues found: {issue_counts}")
    else:
        print(f"  ✓ No major issues detected")

    avg_length = sum(r["evaluation"]["length"] for r in results if "evaluation" in r) / len(TEST_PROMPTS)
    print(f"  Average response length: {avg_length:.1f} chars")

    print(f"{'='*70}\n")

    return {
        "step": step,
        "checkpoint": str(checkpoint_path),
        "timestamp": datetime.now().isoformat(),
        "loss_metrics": loss_metrics,
        "loss_concerns": loss_concerns if loss_metrics else [],
        "probe_results": results,
        "issues": all_issues,
    }


def save_report(report, report_dir="data/training_monitor"):
    """Save evaluation report to JSON."""
    report_path = Path(report_dir)
    report_path.mkdir(exist_ok=True)

    filename = f"eval_step_{report['step']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = report_path / filename

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", help="Specific checkpoint to evaluate (default: latest)")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=360, help="Check interval in seconds (default: 360 = 6 min)")
    args = parser.parse_args()

    if args.watch:
        print("Starting continuous monitoring...")
        print(f"Check interval: {args.interval}s ({args.interval/60:.1f} minutes)")
        print("Press Ctrl+C to stop\n")

        last_checked_step = -1

        try:
            while True:
                checkpoint = get_latest_checkpoint()
                if checkpoint:
                    step = get_checkpoint_step(checkpoint)
                    if step > last_checked_step:
                        report = run_evaluation(checkpoint)
                        save_report(report)
                        last_checked_step = step
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] No new checkpoint (last: step {last_checked_step})")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No checkpoints found")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    else:
        # Single evaluation
        if args.checkpoint:
            checkpoint = Path("data/checkpoints") / args.checkpoint
        else:
            checkpoint = get_latest_checkpoint()

        if not checkpoint or not checkpoint.exists():
            print(f"Checkpoint not found: {checkpoint}")
            exit(1)

        report = run_evaluation(checkpoint)
        save_report(report)