from __future__ import annotations
from pydantic import BaseModel, computed_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset_download_job import DatasetDownloadJob


class DatasetJobStatus(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    downloaded: bool
    normalized: bool
    tokenized: bool
    downloading: bool
    downloaded_bytes: int
    total_bytes: int
    complete: bool
    error: str | None

    @classmethod
    def from_job(
        cls,
        name: str,
        downloaded: bool,
        normalized: bool,
        tokenized: bool,
        job: "DatasetDownloadJob | None",
    ) -> "DatasetJobStatus":
        return cls(
            name=name,
            downloaded=downloaded,
            normalized=normalized,
            tokenized=tokenized,
            downloading=job.running if job else False,
            downloaded_bytes=job.downloaded_bytes if job else 0,
            total_bytes=job.total_bytes if job else 0,
            complete=job.complete if job else False,
            error=job.error if job else None,
        )