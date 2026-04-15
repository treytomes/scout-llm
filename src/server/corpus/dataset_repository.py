import json
from pathlib import Path
from typing import Self

import config
from .models.dataset import Dataset
from .models.dataset_info import DatasetInfo
from .models.dataset_status import DatasetStatus
from .normalizers.normalizer_factory import NormalizerFactory
from .normalizers.dataset_normalizer import IDatasetNormalizer

class DatasetRepository:
    _data_root: Path = config.DATA_ROOT
    _dataset_file: Path = config.DATASET_FILE


    def __init__(self) -> Self:
        pass


    def load_datasets(self) -> list[DatasetInfo]:
        if not self._dataset_file.exists():
            return {}

        with open(self._dataset_file, "r") as f:
            data = json.load(f)
            result = []
            for key in data.keys():
                result.append(DatasetInfo(key, data[key]["hf_path"], data[key]["normalizer"]))
            return result
            

    def dataset_path(self, name: str) -> Path:
        return self._data_root / name / "transformed"
    

    def status(self, name: str) -> DatasetStatus:
        model = Dataset(name)
        return model.status()


    def list_datasets(self) -> list[DatasetStatus]:
        datasets = self.load_datasets()
        return [self.status(data.name) for data in datasets]


    def get_dataset(self, name: str) -> Dataset:
        return Dataset(name)


    def normalize_dataset(self, name: str) -> None:
        datasets = self.load_datasets()
        normalizer_name = [data.normalizer for data in datasets if data.name == name][0]
        factory = NormalizerFactory()
        normalizer = factory.get_normalizer(normalizer_name)
        data = self.get_dataset(name)
        data.normalize(normalizer)
