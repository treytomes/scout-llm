"""
train_sagemaker.py

Entry point for SageMaker training jobs.

Simplified approach: downloads TinyStories directly from HuggingFace,
tokenizes it on the fly, and trains. No S3 dataset caching for now.
"""

import argparse
import sys
import os
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train.train import run_training
from model.model import ScoutModel
from corpus.models.packed_chunk_dataset import PackedChunkDataset
from ai_clients.tokenizer import load_tokenizer
import config
import torch
import datasets


def check_s3_dataset(bucket, dataset_name="TinyStories"):
    """Check if tokenized dataset exists in S3."""
    import boto3
    s3 = boto3.client('s3')
    prefix = f"datasets/{dataset_name}/tokenized/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return 'Contents' in response and len(response['Contents']) > 0
    except Exception as e:
        print(f"   Error checking S3: {e}")
        return False


def download_dataset_from_s3_to_path(bucket, local_path, dataset_name="TinyStories"):
    """Download tokenized dataset from S3 to specific path."""
    import boto3
    s3 = boto3.client('s3')

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    prefix = f"datasets/{dataset_name}/tokenized/"
    paginator = s3.get_paginator('list_objects_v2')

    print("   Downloading from S3...")
    file_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue

            # Remove prefix to get relative path within tokenized/
            rel_path = key.replace(prefix, "")
            local_file = local_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(local_file))
            file_count += 1

    print(f"   Downloaded {file_count} files to {local_path}")

    # If dataset_dict.json is missing (S3 data predates DatasetDict save), create it
    dict_json = local_path / "dataset_dict.json"
    if not dict_json.exists() and (local_path / "train").exists():
        import json as _json
        dict_json.write_text(_json.dumps({"splits": ["train"]}))
        print("   Created dataset_dict.json (missing from S3 cache)")


def upload_dataset_to_s3(local_path, bucket, dataset_name="TinyStories"):
    """Upload tokenized dataset to S3 for future runs."""
    import boto3
    s3 = boto3.client('s3')

    local_dir = Path(local_path)
    prefix = f"datasets/{dataset_name}/tokenized/"

    print("   Uploading to S3...")
    file_count = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_dir)
            s3_key = prefix + str(rel_path)

            s3.upload_file(str(file_path), bucket, s3_key)
            file_count += 1

    print(f"   Uploaded {file_count} files")


def prepare_tiny_stories(s3_bucket):
    """Download and tokenize TinyStories, with S3 caching."""

    print("=" * 70)
    print("Preparing TinyStories Dataset")
    print("=" * 70)

    # Set output to expected location (tokenized folder, not tokenized/train)
    tokenized_path = Path(config.DATASETS_PATH) / "TinyStories" / "tokenized"
    tokenized_path.mkdir(parents=True, exist_ok=True)

    # Check if dataset is cached in S3
    print("\n1. Checking S3 cache...")
    if check_s3_dataset(s3_bucket):
        print("   Found cached dataset in S3!")
        download_dataset_from_s3_to_path(s3_bucket, tokenized_path)
        print("=" * 70)
        return str(tokenized_path)

    print("   No cache found, preparing from scratch...")

    # Download from HuggingFace
    print("\n2. Downloading from HuggingFace...")
    raw_dataset = datasets.load_dataset(
        "roneneldan/TinyStories",
        split="train",
        cache_dir="/opt/ml/processing/hf_cache"
    )

    print(f"   Downloaded {len(raw_dataset)} stories")

    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Tokenize (matching our normalizer/tokenizer pipeline)
    print("\n4. Tokenizing...")

    eos_id = tokenizer.eos_token_id

    def tokenize_function(example):
        # Encode without special tokens, then add EOS
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        if eos_id is not None:
            tokens.append(eos_id)
        return {"tokens": tokens}

    tokenized = raw_dataset.map(
        tokenize_function,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    print(f"   Created {len(tokenized)} tokenized samples")

    # Wrap in DatasetDict with "train" split (matching local structure)
    from datasets import DatasetDict
    dataset_dict = DatasetDict({"train": tokenized})

    print(f"\n5. Saving to {tokenized_path}...")
    dataset_dict.save_to_disk(str(tokenized_path))

    # Upload to S3 for future runs
    print(f"\n6. Caching to S3 (s3://{s3_bucket}/datasets/TinyStories/)...")
    upload_dataset_to_s3(tokenized_path, s3_bucket)

    print("=" * 70)

    return str(tokenized_path)


def main():
    parser = argparse.ArgumentParser()

    # SageMaker passes these automatically
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")

    # Training hyperparameters
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--s3-bucket", type=str, default="scout-llm-data")

    args = parser.parse_args()

    print("=" * 70)
    print("Scout LLM SageMaker Training")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Model output directory: {args.model_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Block size: {config.BLOCK_SIZE}")
    print("=" * 70)

    # Set checkpoint directory
    config.CHECKPOINT_DIR = Path(args.checkpoint_dir)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Override training config
    config.MAX_STEPS = args.max_steps
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.SAVE_INTERVAL = args.save_interval
    config.LOG_INTERVAL = args.log_interval

    # Prepare dataset (with S3 caching)
    # This downloads/creates the dataset directly in the expected location
    dataset_path = prepare_tiny_stories(args.s3_bucket)
    print(f"Dataset ready at: {dataset_path}")

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    # Call the standard training function
    for metrics in run_training(
        dataset_name="TinyStories",
        model_config=config.MODEL_TINYSTORIES,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    ):
        # Training loop yields metrics periodically
        pass

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Copy final checkpoint to model_dir for SageMaker upload
    latest_checkpoint = config.CHECKPOINT_DIR / "latest.pt"
    if latest_checkpoint.exists():
        final_model_path = Path(args.model_dir) / "final_model.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(latest_checkpoint, final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()