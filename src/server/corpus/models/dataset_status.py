from pathlib import Path
from typing import Self


class DatasetStatus:
    name: str
    path: Path
    exists: bool

    
    def __init__(self, name, path) -> Self:
        self.name = name
        self.path = path
        self.exists = path.exists()
