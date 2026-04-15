from pathlib import Path


WEB_DIR = Path(__file__).parent.parent / "web"

DATA_ROOT = Path("./data/datasets")
DATASET_FILE = DATA_ROOT / "datasets.json"

MODEL_NAME = "Scout"
USER_NAME = "Trey"

# The tokenizer determines Scout's inner vocabulary.
# Changing it post-training would likely be catastrophic.
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

# Assets downloaded from HuggingFace should get dumped here.
# Not datasets.
HUGGINGFACE_CACHE_PATH = Path("../hf_cache")
