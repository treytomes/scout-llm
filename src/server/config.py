import os
from pathlib import Path

#
# Training parameters
#

# This should be based on the corpus size.
MAX_STEPS        = 1000

# Increased warmup from 100 to 500 to give Scout time to adjust to the increased block size.
WARMUP_STEPS     = 500

LEARNING_RATE    = 3e-4
MIN_LR           = 3e-5

LOG_INTERVAL     = 50
SAVE_INTERVAL    = 100

# Maximizing the CPU worker count.
# In "Power Saver" mode at half the CPUs, we can process ~40 tokens / second at the 256 block size.
# In "Performance" mode at the same block size we can process ~60 tokens / second.
# I'm hoping to get to 120 tokens per second with using all of the CPUs.
# ...
# In reality, the gains are almost inconsequential,
# even with maximizing the priority of the Python processes.
NUM_WORKERS    = os.cpu_count()
# NUM_WORKERS    = os.cpu_count() // 2


WEB_DIR = Path(__file__).parent.parent / "web"


# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).parent.parent.parent

# Repository root (adjust if config.py is not at root)
PROJECT_ROOT = BASE_DIR

DATA_PATH = PROJECT_ROOT / "data" #Path("./data")
DATASETS_PATH = DATA_PATH / "datasets"
DATASET_FILE = DATASETS_PATH / "datasets.json"

CHECKPOINT_DIR      = DATA_PATH / "checkpoints"
CHECKPOINT_PATH     = Path(CHECKPOINT_DIR) / "latest.pt"

TRAINING_LOG_DIR    = DATA_PATH / "training_log"
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER_NAME = "scout-llm"


MODEL_NAME = "Scout"
USER_NAME = "Trey"

# The tokenizer determines Scout's inner vocabulary.
# Changing it post-training would likely be catastrophic.
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

# Assets downloaded from HuggingFace should get dumped here.
# Not datasets.
HUGGINGFACE_CACHE_PATH = Path("./hf_cache")

# This is how many training sequences are processed per optimization step.
# i.e. the tokens / step throughput
BATCH_SIZE = 8


#
# Inference parameters
#

# The expected length of the context window before the model begins hallucinating.
BLOCK_SIZE = 512


# ──────────────────────────────────────────────────────────────────────────────
# Model Configurations
# ──────────────────────────────────────────────────────────────────────────────

MODEL_TINYSTORIES = {
    # This is the size of the embedding vector for each token.
    # Larger dimensions allow richer representations but increase compute.
    # | Model size | Dim |
    # |---|---|
    # | Small | 256–512 |
    # | Medium | 768–1024 |
    # | Large | 2048+ |
    "dim": 512,

    # This is the number of transformer blocks stacked together.
    # 
    # Each layer performs:
    # * attention
    # * feedforward transformation
    # * residual mixing
    # 
    # More layers → deeper reasoning.
    "layer": 12,

    # This is the number of attention heads per layer.
    # 
    # Each head attends to different relationships.
    # 
    # Example:
    # * head 1 → grammar
    # * head 2 → topic continuity
    # * head 3 → punctuation
    # * etc.
    "heads": 8,

    # Roughly defines the richness of the model's inner world.
    "mlp_ratio": 3.5,

    "dropout": 0.15,

    "block_size": BLOCK_SIZE,
}

# Future module configs — sized to their respective corpora and task complexity.
# Conversational layer: richer structure, more varied vocabulary than TinyStories.
# Reflective layer: highest representational density — first-person interiority.
#
# MODEL_CONVERSATIONAL = {
#     "dim": 512,
#     "layer": 8,
#     "heads": 8,
#     "mlp_ratio": 3.5,
#     "block_size": 512,
#     "dropout": 0.1,
# }
#
# MODEL_REFLECTIVE = {
#     "dim": 512,
#     "layer": 6,
#     "heads": 8,
#     "mlp_ratio": 4.0,
#     "block_size": 512,
#     "dropout": 0.1,
# }

#
# Generation parameters (how the model speaks)
#

# This limits how many tokens the model is allowed to generate in a single response.
# 
# Kept short intentionally — Scout's coherent window at 50M is ~3 sentences.
# Increase as coherence improves with scale.
# MAX_NEW_TOKENS = BLOCK_SIZE // 4
MAX_NEW_TOKENS = BLOCK_SIZE

# Temperature controls randomness in sampling.
# Lower values → safer, repetitive, predictable.
# Higher values → more creative but more errors.
# Typical range:
# | Temperature | Behavior |
# |---|---|
# | 0.2–0.4 | deterministic |
# | 0.5–0.7 | balanced |
# | 0.8–1.0 | creative |
# | >1.0 | chaotic |
TEMPERATURE    = 0.7   # Try raising this later in the training cycle.

# Top‑K sampling restricts token choices to the K most probable tokens.
# Example:
# 
# If the vocabulary has 32k tokens but TOP_K = 40, the model only samples from the 40 most likely next tokens.
# 
# This reduces:
# * nonsense outputs
# * rare-token glitches
# * degenerate sampling loops
TOP_K          = 40

# This penalizes tokens that already appeared earlier in the response.
# Typical values:
# | Value | Effect |
# |---|---|
# | 1.0 | off |
# | 1.1 | mild |
# | 1.2–1.5 | strong |
REP_PENALTY    = 1.3
