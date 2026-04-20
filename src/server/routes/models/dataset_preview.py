from pydantic import BaseModel


class DatasetPreview(BaseModel):
    name: str
    split_name: str
    page: int
    limit: int
    total_rows: int
    rows: list[dict]