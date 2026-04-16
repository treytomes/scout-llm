import importlib
import threading
import requests
from huggingface_hub import HfApi
from datasets import load_dataset

import config
from corpus.normalizers.dataset_normalizer import IDatasetNormalizer


class DatasetDownloadJob:
    def __init__(self, name, hf_dataset, normalizer_name):
        self.name = name
        self.hf_dataset = hf_dataset
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.running = False
        self.complete = False
        self.thread = None
        self.normalizer = DatasetDownloadJob.load_normalizer(normalizer_name)


    def start(self):
        if self.running:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()


    def _download_file(self, url, dest_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    f.write(chunk)
                    self.downloaded_bytes += len(chunk)


    def load_normalizer(class_name: str) -> IDatasetNormalizer:
        module = importlib.import_module("corpus.normalizers")
        cls = getattr(module, class_name)
        return cls()


    def _run(self):
        dataset_dir = config.DATASETS_PATH / self.name
        raw_dir = dataset_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Loading HuggingFace dataset: {self.hf_dataset}")

            # Download dataset through HF
            ds = load_dataset(self.hf_dataset)

            # Estimate dataset size for progress reporting
            try:
                info = HfApi().dataset_info(self.hf_dataset)
                self.total_bytes = sum(s.size for s in info.siblings if s.size)
            except Exception:
                self.total_bytes = 0

            # --------------------------------------------------
            # Save RAW dataset (all splits preserved)
            # --------------------------------------------------
            print(f"Saving raw dataset to {raw_dir}")
            ds.save_to_disk(raw_dir)

            # --------------------------------------------------
            # Apply normalization
            # --------------------------------------------------
            self.normalize()

            self.complete = True

        finally:
            self.running = False