#!/usr/bin/env python3
"""
Sample normalization script — runs the full normalizer pipeline on a small
slice of each dataset and writes results to data/sample_normalize/.

Usage:
    source .venv/bin/activate
    python scripts/sample_normalize.py

Logs to console and to data/sample_normalize/run.log.
"""

import logging
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

OUT_DIR = PROJECT_ROOT / "data" / "sample_normalize"
LOG_FILE = OUT_DIR / "run.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── logging: console + file ────────────────────────────────────────────────
fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=fmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger("sample_normalize")

# ── imports (after sys.path) ───────────────────────────────────────────────
import datasets as hf_datasets
from corpus.normalizers.soda_normalizer import SodaNormalizer
from corpus.normalizers.daily_dialog_normalizer import DailyDialogNormalizer

SODA_RAW       = PROJECT_ROOT / "data" / "datasets" / "SODA"       / "raw"
DAILYDIALOG_RAW = PROJECT_ROOT / "data" / "datasets" / "DailyDialog" / "raw"

SAMPLE_ROWS = 50   # conversations to attempt from each dataset


def sample_soda():
    log.info("═══ SODA sample (%d rows) ═══", SAMPLE_ROWS)
    data = hf_datasets.load_from_disk(SODA_RAW)
    train = data["train"]
    normalizer = SodaNormalizer()

    kept = []
    scanned = 0
    for i in range(len(train)):
        scanned += 1
        row = train[i]
        if not normalizer.filter(row):
            continue
        result = normalizer.map(row)
        kept.append((i, result))
        if len(kept) >= SAMPLE_ROWS:
            break

    log.info("SODA: scanned %d rows, kept %d", scanned, len(kept))
    out_path = OUT_DIR / "soda_sample.txt"
    with open(out_path, "w") as f:
        for idx, result in kept:
            f.write(f"# row {idx}\n")
            f.write(result["chunk"])
            f.write("\n\n")
    log.info("SODA sample written to %s", out_path)
    return kept


def sample_daily_dialog():
    log.info("═══ DailyDialog sample (%d rows) ═══", SAMPLE_ROWS)
    data = hf_datasets.load_from_disk(DAILYDIALOG_RAW)
    if isinstance(data, hf_datasets.DatasetDict):
        train = data["train"]
    else:
        train = data

    normalizer = DailyDialogNormalizer()
    # DailyDialog uses normalize_dataset — run it and take first N results
    result_ds = normalizer.normalize_dataset(train)

    kept = []
    for i in range(min(SAMPLE_ROWS, len(result_ds))):
        kept.append((i, result_ds[i]))

    log.info("DailyDialog: kept %d conversations", len(kept))
    out_path = OUT_DIR / "daily_dialog_sample.txt"
    with open(out_path, "w") as f:
        for idx, result in kept:
            f.write(f"# row {idx}\n")
            f.write(result["chunk"])
            f.write("\n\n")
    log.info("DailyDialog sample written to %s", out_path)
    return kept


if __name__ == "__main__":
    log.info("Starting sample normalization — output dir: %s", OUT_DIR)

    soda_results    = sample_soda()
    dialog_results  = sample_daily_dialog()

    log.info("═══ Done ═══")
    log.info("SODA:        %d conversations → %s", len(soda_results),    OUT_DIR / "soda_sample.txt")
    log.info("DailyDialog: %d conversations → %s", len(dialog_results), OUT_DIR / "daily_dialog_sample.txt")
    log.info("Full log:    %s", LOG_FILE)