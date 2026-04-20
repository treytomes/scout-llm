import importlib
import threading
import requests
from huggingface_hub import HfApi
from datasets import load_dataset

import config
from corpus.normalizers.dataset_normalizer import IDatasetNormalizer


class DatasetDownloadJob:
    def __init__(self, name: str, hf_dataset: str, normalizer_name: str):
        self.name = name
        self.hf_dataset = hf_dataset
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.running = False
        self.complete = False
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.normalizer = DatasetDownloadJob._load_normalizer(normalizer_name)

    def start(self) -> None:
        if self.running:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()

    @staticmethod
    def _load_normalizer(class_name: str) -> IDatasetNormalizer:
        module = importlib.import_module("corpus.normalizers")
        cls = getattr(module, class_name)
        return cls()

    def _run(self) -> None:
        dataset_dir = config.DATASETS_PATH / self.name
        raw_dir = dataset_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Loading HuggingFace dataset: {self.hf_dataset}")
            ds = load_dataset(self.hf_dataset)

            try:
                info = HfApi().dataset_info(self.hf_dataset)
                self.total_bytes = sum(
                    s.size for s in info.siblings if s.size
                )
            except Exception:
                self.total_bytes = 0

            print(f"Saving raw dataset to {raw_dir}")
            ds.save_to_disk(raw_dir)

            self._normalize()

            self.complete = True

        except Exception as e:
            self.error = str(e)
            print(f"Download job failed for {self.name!r}: {e}")

        finally:
            self.running = False

    def _normalize(self) -> None:
        from corpus.dataset_repository import DatasetRepository
        repo = DatasetRepository()
        repo.normalize_dataset(self.name)