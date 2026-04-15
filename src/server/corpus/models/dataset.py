import datasets
import shutil
from pathlib import Path
from typing import Self

import config
from .dataset_status import DatasetStatus
from ..normalizers.dataset_normalizer import IDatasetNormalizer


class Dataset:
    _root: Path = config.DATA_ROOT
    _path_raw: Path
    _path_normalized: Path
    path: Path


    def __init__(self, name: str) -> Self:
        self.name = name
        self.path = self._root / name
        self._path_raw = self.path / "raw"
        self._path_normalized = self.path / "normalized"

    
    def delete(self) -> None:
        if self.exists():
            shutil.rmtree(self.path)
        

    def exists(self) -> bool:
        return self.path.exists()
    

    def is_downloaded(self) -> bool:
        return self._path_raw.exists()
    

    def is_normalized(self) -> bool:
        return self._path_normalized.exists()
    

    def status(self) -> DatasetStatus:
        return DatasetStatus(self.name, self.path)
    
    
    def get_split_names(self) -> list[str]:
        data = datasets.load_from_disk(self._path_raw)
        if data is  hasattr(data, "keys"):
            return data.keys
        return []
    
    
    def get_total_rows(self, is_raw: bool = False, split_name: str = "train"):
        ds = self.get_raw(split_name) if is_raw else self.get_normalized(split_name)
        total_rows = len(ds)
        return total_rows
    

    def _normalize_dataset(self, data: datasets.Dataset, normalizer: IDatasetNormalizer) -> datasets.Dataset:
        data = data.filter(normalizer.filter)
        data = data.map(
            normalizer.map,
            remove_columns=data.column_names
        )
        return data
    
    
    def normalize(self, normalizer: IDatasetNormalizer) -> None:
        """
        Launch the normalization process.
        This might take a minute.
        """
        if not self._path_raw.exists():
            raise FileExistsError("Raw dataset has not been downloaded.")
        if self._path_normalized.exists():
            shutil.rmtree(self._path_normalized)
        print("Applying dataset normalization")

        data = datasets.load_from_disk(self._path_raw)
        if isinstance(data, datasets.DatasetDict):
            for split_name in data.keys():
                data[split_name] = self._normalize_dataset(data[split_name], normalizer)
        else:
            data = self._normalize_dataset(data, normalizer)
        
        self._path_normalized.mkdir(parents=True, exist_ok=True)
        print(f"Saving transformed dataset to `{self._path_normalized}`.")
        data.save_to_disk(self._path_normalized)
        print("Normalization complete.")


    def get_raw(self, split_name: str = "train") -> datasets.Dataset:
        if not self._path_raw.exists():
            raise FileExistsError("Raw dataset has not been downloaded.")
        data = datasets.load_from_disk(self._path_raw)
        if isinstance(data, datasets.Dataset):
            return data
        
        # `data` must be a a `datasets.DatasetDict`.
        return data[split_name]
 
    
    def get_normalized(self, split_name: str = "train") -> datasets.Dataset:
        if not self._path_normalized.exists():
            raise FileExistsError("Dataset has not been normalized.")
        data = datasets.load_from_disk(self._path_normalized)
        if isinstance(data, datasets.Dataset):
            return data
        
        # `data` must be a a `datasets.DatasetDict`.
        return data[split_name]
   

    def get_rows(self, is_raw: bool, split_name: str, limit: int, page: int) -> list[dict]:
        data = self.get_raw(split_name) if is_raw else self.get_normalized(split_name)
        start = page * limit
        end = start + limit
        total = len(data)

        rows = []
        for i in range(start, min(end, total)):
            rows.append(data[i])
        return rows