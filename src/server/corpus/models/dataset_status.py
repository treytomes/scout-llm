from pathlib import Path
from pydantic import BaseModel


class DatasetStatus(BaseModel):
    name: str
    exists: bool
    downloaded: bool
    normalized: bool
    tokenized: bool

    def __init__(self, name: str, path: Path) -> None:
        super().__init__(
            name=name,
            exists=path.exists(),
            downloaded=(path / "raw").exists(),
            normalized=(path / "normalized").exists(),
            tokenized=(path / "tokenized").exists(),
        )