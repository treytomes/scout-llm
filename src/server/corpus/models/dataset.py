# corpus.models.dataset.py

import datasets
import shutil
from pathlib import Path
from typing import Self

import config
from ai_clients.tokenizer import load_tokenizer
from .dataset_status import DatasetStatus
from ..normalizers.dataset_normalizer import IDatasetNormalizer


class Dataset:
    _root: Path = config.DATASETS_PATH
    _path_raw: Path
    _path_normalized: Path
    _path_tokenized: Path
    path: Path


    def __init__(self, name: str) -> None:
        self.name = name
        self.path = self._root / name
        self._path_raw = self.path / "raw"
        self._path_normalized = self.path / "normalized"
        self._path_tokenized = self.path / "tokenized"

    
    def delete(self) -> None:
        if self.exists():
            shutil.rmtree(self.path)
        

    def exists(self) -> bool:
        return self.path.exists()
    

    def is_downloaded(self) -> bool:
        return self._path_raw.exists()
    

    def is_normalized(self) -> bool:
        return self._path_normalized.exists()
    

    def is_tokenized(self) -> bool:
        return self._path_tokenized.exists()
    

    def status(self) -> DatasetStatus:
        return DatasetStatus(self.name, self.path)
    
    
    def get_split_names(self) -> list[str]:
        data = datasets.load_from_disk(self._path_raw)
        if hasattr(data, "keys"):
            return data.keys()
        return []
    
    
    def get_total_rows(self, is_raw: bool = False, split_name: str = "train"):
        ds = self.get_raw(split_name) if is_raw else self.get_normalized(split_name)
        total_rows = len(ds)
        return total_rows
    

    def _normalize_dataset(self, data: datasets.Dataset, normalizer: IDatasetNormalizer) -> datasets.Dataset:
        # Allow normalizers that need full-dataset access to override the pipeline
        result = normalizer.normalize_dataset(data)
        if result is not None:
            return result

        data = data.filter(normalizer.filter, load_from_cache_file=False)
        data = data.map(
            normalizer.map,
            remove_columns=data.column_names,
            load_from_cache_file=False,
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
        if self._path_tokenized.exists():
            shutil.rmtree(self._path_tokenized)
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


    def tokenize(self) -> None:
        """
        Tokenize all normalized dataset splits and persist the tokenized dataset.
        EOS is embedded in the chunk text by normalizers as '</s>', which encodes
        to the correct token ID — no separate append needed.
        """
        if not self._path_normalized.exists():
            raise FileExistsError("Dataset has not been normalized.")

        tokenizer = load_tokenizer()

        print("Loading normalized dataset")
        data = datasets.load_from_disk(self._path_normalized)

        def tokenize_row(row):
            text = row.get("chunk")
            if not text:
                return {"tokens": []}

            tokens = tokenizer.encode(text, add_special_tokens=False)
            return {"tokens": tokens}

        print("Tokenizing dataset")

        if isinstance(data, datasets.DatasetDict):
            for split_name in data.keys():
                data[split_name] = data[split_name].map(
                    tokenize_row,
                    remove_columns=data[split_name].column_names,
                )
        else:
            data = data.map(
                tokenize_row,
                remove_columns=data.column_names,
            )

        print(f"Saving tokenized dataset to `{self._path_tokenized}`.")
        data.save_to_disk(self._path_tokenized)

        print("Tokenization complete.")


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
    

    def get_tokenized(self, split_name: str = "train") -> datasets.Dataset:
        if not self._path_tokenized.exists():
            raise FileNotFoundError("Dataset has not been tokenized.")
        data = datasets.load_from_disk(self._path_tokenized)
        if isinstance(data, datasets.Dataset):
            return data
        
        # `data` must be a a `datasets.DatasetDict`.
        return data[split_name]
   

    def get_rows(self, is_raw: bool, split_name: str, limit: int, page: int) -> list[dict]:
        data = self.get_raw(split_name) if is_raw else self.get_normalized(split_name)
        start = page * limit
        end = min(start + limit, len(data))
        if start >= len(data):
            return []
        return [dict(data[i]) for i in range(start, end)]