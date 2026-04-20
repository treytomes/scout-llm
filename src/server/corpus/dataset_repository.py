# corpus.dataset_repository.py

import json
from pathlib import Path

import config
from .models.dataset import Dataset
from .models.dataset_info import DatasetInfo
from .models.dataset_status import DatasetStatus
from .normalizers.normalizer_factory import NormalizerFactory


class DatasetRepository:
    _DATASETS_PATH: Path = config.DATASETS_PATH
    _dataset_file: Path = config.DATASET_FILE

    def load_datasets(self) -> list[DatasetInfo]:
        if not self._dataset_file.exists():
            return []

        with open(self._dataset_file, "r") as f:
            data = json.load(f)

        result = []
        for key, entry in data.items():
            normalizer = entry.get("normalizer")
            if not normalizer:
                continue
            result.append(DatasetInfo(key, entry.get("hf_path"), normalizer))
        return result

    def load_dataset_info(self, name: str) -> DatasetInfo:
        for info in self.load_datasets():
            if info.name == name:
                return info
        raise KeyError(f"Unknown dataset: {name!r}")

    def dataset_path(self, name: str) -> Path:
        return self._DATASETS_PATH / name / "transformed"

    def status(self, name: str) -> DatasetStatus:
        return Dataset(name).status()

    def list_datasets(self) -> list[DatasetStatus]:
        return [self.status(info.name) for info in self.load_datasets()]

    def get_dataset(self, name: str) -> Dataset:
        return Dataset(name)

    def normalize_dataset(self, name: str) -> None:
        info = self.load_dataset_info(name)
        normalizer = NormalizerFactory().get_normalizer(info.normalizer)
        self.get_dataset(name).normalize(normalizer)