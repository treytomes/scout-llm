from pathlib import Path

import config
from corpus.dataset_repository import DatasetRepository
from .dataset_download_job import DatasetDownloadJob
from .models.dataset_job_status import DatasetJobStatus


class DatasetDownloadJobManager:
    """Manages dataset download and normalization jobs."""

    def __init__(self):
        self.jobs: dict[str, DatasetDownloadJob] = {}
        self.repo = DatasetRepository()

    def dataset_path(self, name: str) -> Path:
        return config.DATASETS_PATH / name / "transformed"

    def job_status(self, name: str) -> DatasetJobStatus:
        data = self.repo.get_dataset(name)
        job = self.jobs.get(name)
        return DatasetJobStatus.from_job(
            name,
            data.is_downloaded(),
            data.is_normalized(),
            data.is_tokenized(),
            job,
        )

    def start_download(self, name: str) -> None:
        info = self.repo.load_dataset_info(name)  # raises KeyError if unknown

        if not info.hf_path:
            raise ValueError(f"Dataset {name!r} has no hf_path and cannot be downloaded")

        if name in self.jobs and self.jobs[name].running:
            return

        job = DatasetDownloadJob(
            name=name,
            hf_dataset=info.hf_path,
            normalizer_name=info.normalizer,
        )
        self.jobs[name] = job
        job.start()

    def delete(self, name: str) -> None:
        """Delete both the job and the underlying dataset files."""
        self.repo.get_dataset(name).delete()
        self.jobs.pop(name, None)