from typing import Self


class DatasetInfo:
    name: str
    hf_path: str
    normalizer: str

    def __init__(self, name: str, hf_path: str, normalizer: str) -> Self:
        self.name = name
        self.hf_path = hf_path
        self.normalizer = normalizer
