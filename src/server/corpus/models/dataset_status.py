from pathlib import Path
from typing import Self


class DatasetStatus:
    name: str
    path: Path
    exists: bool
    downloaded: bool
    normalized: bool
    tokenized: bool

    def __init__(self, name, path) -> Self:
        self.name = name
        self.path = path
        self.exists = path.exists()
        self.downloaded = (path / "raw").exists()
        self.normalized = (path / "normalized").exists()
        self.tokenized = (path / "tokenized").exists()
