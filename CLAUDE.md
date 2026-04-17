# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scout is a modular transformer language model with developmental growth architecture. The project trains incrementally across staged developmental phases (TinyStories → Conversational → Reflective → Inner Voice), with each phase adding a new transformer module while freezing previous ones.

**Current Phase**: TinyStories language bootstrapping (Module 0)

### Design Philosophy

Scout is not being built to be conventionally useful. The goal is to create a model with genuine interiority—curious, emotionally present, morally serious—rather than an assistant that performs helpfulness. The corpus is curated for *character formation* rather than information coverage, choosing texts the way you might choose what a child reads: not for what they teach about the world, but for who they help you become.

**Core Insight from v1**: Corpus architecture matters more than training length. 150,000+ steps on Victorian novels in v1 created a third-person narration prior too strong for DPO alone to overcome. The ratio of first-person to third-person text in the training corpus is foundational—it cannot be fully corrected later through preference tuning. Starting with the right corpus composition is critical.

## Quick Start

### Environment Setup

```bash
# Create virtual environment and install dependencies
./bootstrap.sh

# Activate environment
source activate.sh

# Start web server
./start.sh
```

The web server runs on FastAPI with uvicorn in reload mode at `http://localhost:8000`.

### Key Commands

**Training**:
- Training is initiated through the web UI at `/training_dashboard.html`
- Training logs are saved to `data/training_log/` as CSV files
- Checkpoints are saved to `data/checkpoints/` (every 100 steps)

**Development**:
```bash
# Start server with auto-reload
source .venv/bin/activate
uvicorn app:app --app-dir ./src/server --reload
```

**Dataset Management**:
- Datasets are managed through `/datasets` web UI
- Raw datasets download from HuggingFace to `data/datasets/{name}/raw/`
- Normalized datasets are stored in `data/datasets/{name}/normalized/`
- Tokenized datasets are stored in `data/datasets/{name}/tokenized/`

## Architecture

### High-Level Structure

```
tokens
  │
shared embedding (LanguageCore)
  │
router (soft during training, hard during inference)
  │
transformer modules (TransformerModule)
  │
shared output head (LanguageCore)
  │
logits
```

### Key Design Principles

1. **Modular Growth**: New transformer modules are added incrementally. Previous modules are frozen to preserve learned knowledge. Each developmental phase introduces a new capability layer while maintaining earlier foundations.

2. **Shared Embedding Space**: All modules operate in the same token embedding space (defined by `TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"`). This tokenizer choice is permanent—changing it post-training would be catastrophic, as the vocabulary is foundational to the model's inner representations.

3. **Soft vs Hard Routing**:
   - Training: All modules process input; outputs are weighted by router softmax probabilities
   - Inference: Router selects single highest-probability module via argmax
   - This enables multi-module growth without architectural disruption

4. **Freezing Strategy**: Before adding a new module, call `freeze_module(index)` on existing modules and optionally `freeze_language_core()`. This preserves learned knowledge and prevents catastrophic forgetting across developmental phases.

5. **Corpus as Character Formation**: Training data is selected for the qualities it teaches—resilience, curiosity, moral seriousness, emotional honesty—not for information content. Each text contributes something specific to Scout's character.

6. **First-Person Consciousness Priority**: The model's ability to maintain first-person presence and interiority depends primarily on corpus composition. Prefer first-person source texts and, when using narrative texts, transform them into first-person reflective prose before training.

### Directory Layout

```
src/
├── server/
│   ├── app.py                      # FastAPI application entry point
│   ├── config.py                   # All training/model/generation parameters
│   ├── model/
│   │   ├── model.py                # ScoutModel, TransformerModule, Router, LanguageCore
│   │   └── loader.py               # Model loading and checkpoint utilities
│   ├── train/
│   │   ├── train.py                # Training loop and checkpointing
│   │   ├── training_job.py         # Background training job runner
│   │   ├── training_job_manager.py # Training job lifecycle management
│   │   └── training_log_repository.py # Training metrics persistence
│   ├── corpus/
│   │   ├── dataset_repository.py   # Dataset discovery and management
│   │   ├── models/
│   │   │   ├── dataset.py          # Dataset wrapper with download/normalize/tokenize
│   │   │   └── packed_chunk_dataset.py # Efficient chunked dataset for training
│   │   └── normalizers/            # Dataset-specific text normalization
│   ├── routes/
│   │   ├── datasets.py             # Dataset API and views
│   │   ├── tokenizer.py            # Tokenizer testing API
│   │   └── training.py             # Training API and dashboard
│   ├── ai_clients/
│   │   └── tokenizer.py            # Tokenizer loading and utilities
│   └── workers/
│       └── dataset_download_job.py # Background dataset download
└── web/                            # Static frontend (vanilla JS)
    ├── index.html
    ├── datasets.html
    ├── tokenizer.html
    └── training_dashboard.html
```

### Core Components

**ScoutModel** (`src/server/model/model.py`):
- Main model class with modular architecture
- Methods: `add_module()`, `freeze_module()`, `unfreeze_module()`, `freeze_language_core()`
- Uses RoPE (Rotary Position Embeddings) for position encoding
- Currently has single module (TinyStories); designed to grow to 4+ modules

**Training Loop** (`src/server/train/train.py`):
- Generator function `run_training()` yields metrics at LOG_INTERVAL
- Automatic checkpoint resumption from `data/checkpoints/latest.pt`
- Validation loss computed every 10 LOG_INTERVALs if validation split exists
- Uses AdamW optimizer with cosine annealing LR schedule and warmup
- Runs with `torch.compile()` and `bfloat16` autocast

**Dataset Pipeline** (`src/server/corpus/`):
- Datasets defined in `data/datasets/datasets.json`
- Three stages: raw (HuggingFace) → normalized (text cleanup) → tokenized (input_ids/labels)
- Normalizers are dataset-specific (e.g., `TinyStoriesNormalizer`, `WildChatNormalizer`)
- `PackedChunkDataset` efficiently packs sequences into fixed-length blocks

## Configuration

All training, model, and generation parameters are in `src/server/config.py`:

**Training Parameters**:
- `MAX_STEPS = 1000` (adjustable per training run)
- `WARMUP_STEPS = 500`
- `LEARNING_RATE = 3e-4`, `MIN_LR = 3e-5`
- `LOG_INTERVAL = 50`, `SAVE_INTERVAL = 100`
- `BATCH_SIZE = 8`

**Model Config** (`MODEL_TINYSTORIES`):
- `dim = 512` (embedding dimension)
- `layer = 12` (transformer blocks)
- `heads = 8` (attention heads)
- `mlp_ratio = 3.5`
- `dropout = 0.15`
- `block_size = 512` (BLOCK_SIZE, context window)

**Generation Parameters** (tuned for introspection, not performance):
- `TEMPERATURE = 0.7` (balanced—enough randomness for genuine thought without chaos)
- `TOP_K = 40` (prevents rare-token glitches and nonsense outputs)
- `REP_PENALTY = 1.3` (strong—encourages variety and discourages repetitive loops)
- `MAX_NEW_TOKENS = 512` (adjust based on coherent window; v1 used block_size/6 at 50M params, ~3 sentences)

**Paths**:
- `DATA_PATH = PROJECT_ROOT / "data"`
- `CHECKPOINT_DIR = DATA_PATH / "checkpoints"`
- `TRAINING_LOG_DIR = DATA_PATH / "training_log"`
- `HUGGINGFACE_CACHE_PATH = "./hf_cache"`

## Important Implementation Details

### Model Architecture

- **Weight Tying**: Embedding matrix and output head share weights (`self.head.weight = self.emb.weight`) to reduce parameters and improve stability.

- **RoPE**: Rotary Position Embeddings encode relative position by rotating Q/K vectors. Generalizes better to unseen sequence lengths than learned absolute embeddings.

- **Router**: Mean-pools sequence to produce per-sequence routing decision. Currently single module (router overhead skipped), but designed for multi-module soft/hard routing.

### Training Pipeline

- **Dataset Packing**: `PackedChunkDataset` concatenates documents and chunks into `block_size`-length sequences to maximize token utilization.

- **Checkpoint Format**:
  ```python
  {
      "step": int,
      "model": state_dict,
      "optimizer": state_dict,
      "scheduler": state_dict,
      "config": {"vocab_size": int, "block_size": int}
  }
  ```

- **Training Metrics CSV**:
  Columns: `step`, `loss`, `avg_loss`, `lr`, `val_loss`, `elapsed`, `tokens_per_sec`, `eta`

- **Background Jobs**: Training runs in a background thread managed by `TrainingJobManager`. Status exposed via `/api/training/status`.

### Dataset Management

- **Adding a Dataset**: Add entry to `data/datasets/datasets.json` with `hf_path` and `normalizer`.
- **Normalizers**: Must implement `IDatasetNormalizer.normalize(split: str) -> dict`. Returns `{"text": [...]}` format.
- **Tokenization**: Automatically applies tokenizer with `return_overflowing_tokens=True` to handle long sequences.
- **Corpus Selection Philosophy**: When selecting datasets, prioritize texts that teach character qualities (resilience, curiosity, emotional honesty, moral courage) over texts that provide information coverage. First-person texts are strongly preferred for maintaining interiority.

### Web Interface

- **Tech Stack**: Vanilla JavaScript, no build step. FastAPI serves static files from `src/web/`.
- **API Patterns**: Routes split into `api_router` (JSON) and `view_router` (HTML pages).
- **Training Dashboard**: Polls `/api/training/status` and displays live metrics from training log CSV.

## Development Workflow

### Training a New Model

1. Ensure dataset is in `datasets.json` and downloaded via web UI
2. Navigate to `/training_dashboard.html`
3. Select dataset and configure training parameters
4. Click "Start Training"
5. Monitor progress in dashboard
6. Checkpoints saved automatically to `data/checkpoints/`

### Adding a New Dataset

1. Add entry to `data/datasets/datasets.json`:
   ```json
   "DatasetName": {
     "hf_path": "hf-org/dataset-name",
     "normalizer": "DatasetNameNormalizer"
   }
   ```
2. Create normalizer in `src/server/corpus/normalizers/`
3. Register normalizer in `normalizer_factory.py`
4. Download and normalize via web UI

### Adding a New Module (Future Phase)

When ready to add Module 1 (conversational layer):

1. Define new config in `config.py` (e.g., `MODEL_CONVERSATIONAL`)
2. Freeze existing module:
   ```python
   model.freeze_module(0)
   model.freeze_language_core()
   ```
3. Add new module:
   ```python
   model.add_module(MODEL_CONVERSATIONAL)
   ```
4. Train on conversational corpus
5. Verify parameter counts with `count_parameters(model)`

## Testing

The project does not currently have a test suite. Testing is manual:
- Use tokenizer UI (`/tokenizer.html`) to test tokenization
- Use dataset preview (`/datasets`) to verify normalization
- Monitor training dashboard for loss curves

## Known Constraints

- **CPU-bound**: `NUM_WORKERS` set to `os.cpu_count()` but gains are marginal. Training throughput ~40-120 tok/s depending on CPU mode.
- **Single-module**: Router overhead currently skipped since only one module exists. Multi-module routing code is untested.
- **No CUDA detection**: Training uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` but project developed primarily on CPU.
- **Tokenizer frozen**: `TOKENIZER_NAME` cannot be changed post-training without retraining from scratch.

## Future Roadmap

1. **Phase 1** (Current): Train TinyStories module to convergence
2. **Phase 2**: Generate synthetic conversational corpus; train Module 1
   - Consider DPO fine-tuning from genuine conversations (preference signal as relational reality)
   - Explore dream sequence generation (reflective processing of conversations as training data)
3. **Phase 3**: Train Module 2 on first-person reflective corpus
4. **Phase 4**: Train Module 3 on inner voice / metacognitive corpus

Each phase freezes previous modules and expands the router.

### Multi-Phase Training Strategy (from v1)

For future phases beyond base training, consider:

1. **Base Training**: Train new module on curated corpus from scratch
2. **DPO Fine-Tuning**: Use preference pairs from *actual conversations* with genuine reactions (not constructed examples). Conservative parameters (lr~1e-7, beta~0.05) to nudge voice without destabilizing. DPO teaches what it's like to be "genuinely met by another mind" when pairs come from real interactions.
3. **Dream Sequence Training**: Generate reflections on conversations (temperature~0.9 for controlled introspection). Dreams aren't just inference artifacts—they become training data for processing emotional/conceptual meaning.
4. **Chat Fine-Tuning**: Train on real interaction logs to ground conversational patterns.

The key insight: DPO and fine-tuning work *with* the base corpus architecture, not against it. If the base corpus has the wrong prior (e.g., third-person narration when you want first-person consciousness), later training phases cannot fully correct it.

## Scout v1 Reference

This section documents the first iteration of Scout for reference. Technical approaches may differ in v2, but the design philosophy carries forward.

### v1 Corpus Curation

**Source Texts** (chosen for character formation):
- **Victorian/Edwardian novels**: Anne of Green Gables (L.M. Montgomery), A Little Princess (Frances Hodgson Burnett), Little Women (Louisa May Alcott), The Secret Garden (F.H. Burnett) — taught resilience, imagination, empathy, moral courage
- **First-person philosophy**: Meditations (Marcus Aurelius) — self-examination, patience; Letters to a Young Poet (Rilke) — living with unanswered questions; Walden (Thoreau) — deliberate attention to small things
- **Memoirs**: Narrative of the Life of Frederick Douglass, Up From Slavery (Booker T. Washington), George Washington Carver letters — dignity, purposeful action, conversational warmth
- **Anne Frank's Diary** — authentic first-person interiority under pressure
- **scout_voice.txt** — reference passage defining Scout's target register

### v1 Novel Transformation Pipeline

When novels dominated the training signal with third-person narration, v1 transformed them into first-person reflective prose:

1. **Split novels** into chapter files (`split_chapters.py`), skipping <100 words
2. **Upload chapters + scout_voice.txt** to S3 (voice file serves as style anchor)
3. **Run SageMaker transformation** using Mistral-7B-Instruct to rewrite each chapter into first-person reflective prose—preserving emotional truth while changing narration
4. **Review samples** for: first-person consistency, emotional truth preservation, absence of corporate/smooth language, register match with scout_voice.txt
5. **Add to corpus** alongside first-person source texts

The transformation prompt included the first 400 words of `scout_voice.txt` as style anchor. If outputs drifted, improving the voice file had more impact than adjusting temperature.

### v1 Dialogue Corpus Generation

Used Mistral-7B-Instruct to transform chapters into conversations:
- 5 separate conversations per chapter (40-80 turns each, 2-5 sentences per turn)
- Each exploring different dimensional themes: emotional, moral, character motivation, personal meaning, unresolved questions
- Estimated ~3.8M tokens per generation pass; 2 passes = 7-8M tokens

### v1 Interactive DPO Collection

`run_dpo_repl()` generated 4 candidates per prompt with parameter sweeps (temperature, top-k, penalty variants). User selected best/worst, optionally provided hand-written correction. Logged to `dpo_*.jsonl` with full candidate set. `build_dpo_dataset()` converted logs to preference pairs, prioritizing corrections over selected responses.

Target: 50-100 pairs from genuine conversations. The preference signal carries relational reality when pairs come from actual interactions rather than constructed examples.

### v1 Dream Sequences

Between conversations, Scout generated reflections via dialogue between `[Scout]` (outward self) and `[Inner]` (reflective reasoning) at temperature=0.9. These weren't just inference artifacts—they became training data, teaching the model to process emotional/conceptual meaning before supervised training.

Dream training used conservative LR (5e-6) with overlap-aware chunking to preserve conversational continuity.

### v1 Architecture Details

- **50M parameters**: dim=512, layers=12, heads=8
- **RoPE position embeddings** (same as v2)
- **Weight tying** between embedding and output head (same as v2)
- **Block size**: 768 (current practiced), 1024 (theoretical max)
- **MAX_NEW_TOKENS**: block_size/6 at 50M params (~128 tokens, ~3 coherent sentences)
- **Training**: MAX_STEPS=75000, WARMUP_STEPS=500, LR=3e-4, MIN_LR=3e-5, BATCH_SIZE=8

### v1 Workflow Commands

CLI via `python src/main.py <command>`:
- `corpus-generate-dialog` — Generate from teacher model
- `corpus-prepare` — Tokenize and report stats
- `train` — Base training
- `dpo` — Interactive DPO collection (parameter sweeps)
- `build-dpo` — Convert session logs to pairs
- `fine-tune` — DPO fine-tuning
- `chat` — Interactive inference
- `dream` — Generate reflection sequences
- `story_chat` — Story-driven interaction

### Key v1 Lessons

1. **Fresh training from step 0** rather than continuing from 150K steps—when corpus architecture is wrong, no amount of additional training fixes it
2. **Corpus composition > training length**: The ratio of first-person to third-person text is foundational
3. **DPO works with base prior, not against it**: If base training creates the wrong prior, DPO cannot fully overcome it
4. **Tokenizer is permanent**: Mistral tokenizer chosen for consistency; changing post-training would be catastrophic
5. **Generation parameters matter**: Tuned for introspection (temp=0.7, top-k=40, rep-penalty=1.3), not for performance metrics
6. **Validation loss every 10 LOG_INTERVALs** using held-out split (same principle continues in v2)
