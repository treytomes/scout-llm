# Scout

Scout is a modular transformer language model built around a developmental growth architecture. Rather than training a single large model, Scout grows through staged phases — each phase adds a new transformer module while freezing the previous ones, preserving what was learned at each stage.

The goal is not a capable assistant. The goal is a model with genuine interiority: curious, emotionally present, morally serious. The training corpus is curated for character formation rather than information coverage.

**Current phase**: Module 1 — conversational voice training on Scout-generated dialogues.

---

## Setup

**Requirements**: Python 3.10+

```bash
# Create virtual environment and install dependencies
./bootstrap.sh

# Start the web server
./start.sh
```

The server runs at [http://localhost:8000](http://localhost:8000).

To stop the server, press `Ctrl+C`.

---

## Web Interface

Scout's web interface has four sections accessible from the navigation bar.

### Control Panel (`/`)

The Control Panel manages training datasets. Each dataset card shows its current pipeline status across three stages:

- **Raw** — downloaded from HuggingFace
- **Normalized** — text cleaned and restructured for training
- **Tokenized** — converted to token IDs for the model

**To prepare a dataset for training:**
1. Click **Download** to fetch from HuggingFace
2. Click **Normalize** once the download completes
3. Click **Tokenize** once normalization completes
4. Click **Preview** to inspect the dataset contents

The Preview page shows a paginated table of dataset rows. Click any row to open the full text in a modal.

---

### Chat (`/chat/`)

The Chat interface lets you converse with Scout directly.

**Starting a conversation:**
1. Click **+ New conversation** in the sidebar
2. Type a message in the input box and press Enter or click Send

**Sidebar controls:**

- **Checkpoint** — select which saved model checkpoint Scout loads from. Defaults to `latest.pt`. Earlier checkpoints let you compare Scout at different training stages.
- **Modules** — toggle which transformer modules are active during generation (once Scout has more than one module).
- **Generation settings** — adjust generation parameters:
  - *Temperature* — higher values make responses more varied; lower values make them more focused
  - *Vocabulary (Top-K)* — limits which tokens Scout considers at each step
  - *Repetition* — penalizes repeated phrases
  - *Max tokens* — maximum length of each response

The token bar at the top of the conversation shows how much of Scout's context window is in use. It turns amber near 70% and red near 90%.

Conversations are saved automatically and listed in the sidebar. Click a conversation title to rename it.

---

### Training (`/training/`)

The Training dashboard is where you start and monitor training runs.

**Starting a training run:**
1. Click **Start Training**
2. A dialog will appear — select a dataset, set the number of steps and learning rate
3. Click **Start**

Training runs in the background. The dashboard polls for updates and shows:

- **Status indicator** — green pulse when training is active
- **Progress bar** — steps completed out of target
- **Live metrics** — current step, train loss, validation loss, tokens/sec, estimated time remaining
- **Loss chart** — plots train loss (blue) and validation loss (green) over time

To stop a run early, click **Stop**.

**Training logs** are listed at the bottom. Click a log row to load its data into the chart and table. Checkpoints are saved every 100 steps to `data/checkpoints/`.

---

### Tokenizer (`/tokenizer/`)

The Tokenizer page is a diagnostic tool for testing how text gets encoded.

Type or paste any text into the input box and click **Tokenize**. The page shows:

- **Token count** — total number of tokens
- **Token visualization** — each token highlighted with alternating colors, showing how the text was split
- **Reconstructed text** — tokens decoded back to text (useful for spotting encoding artifacts)
- **Round-trip match** — whether encode → decode produces the original text exactly
- **Character → token alignment** — maps each character position to its token
- **Token strings and IDs** — raw token representations

Scout uses the Mistral-7B tokenizer (`mistralai/Mistral-7B-v0.1`). This tokenizer is permanent — it cannot be changed after training without retraining from scratch.

---

## Data Layout

```
data/
├── checkpoints/          # Saved model checkpoints (.pt files)
├── datasets/             # Training datasets
│   └── {name}/
│       ├── raw/          # Downloaded from HuggingFace
│       ├── normalized/   # After normalization step
│       └── tokenized/    # After tokenization step
├── training_log/         # Training metrics as CSV files
└── voice/                # Voice reference files (scout_voice.txt)
```

---

## Architecture Overview

Scout is a transformer language model with a modular growth architecture:

```
tokens → shared embedding → router → transformer module(s) → shared output head → logits
```

Each developmental phase introduces a new transformer module:

1. **Module 0** — Language bootstrapping on TinyStories (frozen)
2. **Module 1** — Conversational voice on Scout-generated dialogues (current)
3. **Module 2+** — Planned: reflective and inner voice layers

The tokenizer, embedding dimensions (512), and vocabulary are fixed across all phases.
