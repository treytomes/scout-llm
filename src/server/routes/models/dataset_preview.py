from typing import Self


class DatasetPreview:
    dataset: str
    split_name: str
    page: int
    limit: int
    total_rows: int
    rows: list[dict]

    
    def __init__(self, name: str, split_name: str, page: int, limit: int, total_rows: int, rows: list[dict]) -> Self:
        self.name = name
        self.split_name = split_name
        self.page = page
        self.limit = limit
        self.total_rows = total_rows
        self.rows = rows
