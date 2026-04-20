from pathlib import Path


class DatasetStatus:
    name: str
    path: Path
    exists: bool
    downloaded: bool
    normalized: bool
    tokenized: bool

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.exists = path.exists()
        self.downloaded = (path / "raw").exists()
        self.normalized = (path / "normalized").exists()
        self.tokenized = (path / "tokenized").exists()