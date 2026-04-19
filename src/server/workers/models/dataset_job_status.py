from typing import Self
from ..dataset_download_job import DatasetDownloadJob


class DatasetJobStatus:
    name: str
    downloaded: bool
    normalized: bool
    tokenized: bool
    downloading: bool
    downloaded_bytes: int
    total_bytes: int
    complete: bool


    def __init__(self, name: str, downloaded: bool, normalized: bool, tokenized: bool, job: DatasetDownloadJob) -> Self:
        self.name = name
        self.downloaded = downloaded
        self.normalized = normalized
        self.tokenized = tokenized
        self.downloading = job.running if job else False
        self.downloaded_bytes = job.downloaded_bytes if job else 0
        self.total_bytes = job.total_bytes if job else 0
        self.complete = job.complete if job else False