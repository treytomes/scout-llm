import importlib
from typing import Self

from .dataset_normalizer import IDatasetNormalizer

class NormalizerFactory:
    def __init__(self) -> Self:
        pass

    
    def get_normalizer(self, class_name: str) -> IDatasetNormalizer:
        module = importlib.import_module("corpus.normalizers")
        cls = getattr(module, class_name)
        return cls()