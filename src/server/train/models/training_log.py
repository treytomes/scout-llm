from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List

from .training_log_entry import TrainingLogEntry


@dataclass
class TrainingLogModel:
    path: Path
    log_date: date
    index: int
    entries: List[TrainingLogEntry] = field(default_factory=list)


    @property
    def filename(self) -> str:
        return self.path.name


    def add_entry(self, entry: TrainingLogEntry):
        self.entries.append(entry)


    @staticmethod
    def parse_filename(path: Path):
        """
        Parse filename training_YYYY-MM-DD_N.csv
        """
        name = path.stem  # remove .csv
        parts = name.split("_")

        if len(parts) != 3:
            raise ValueError(f"Invalid training log filename: {path.name}")

        _, date_str, idx_str = parts

        y, m, d = map(int, date_str.split("-"))

        return date(y, m, d), int(idx_str)