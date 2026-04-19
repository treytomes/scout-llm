from pathlib import Path

import config
from corpus.dataset_repository import DatasetRepository
from .dataset_download_job import DatasetDownloadJob
from .models.dataset_job_status import DatasetJobStatus


class DatasetDownloadJobManager:
    """
    Manages dataset download jobs.
    """

    jobs: list[DatasetDownloadJob]
    repo: DatasetRepository = DatasetRepository()

    def __init__(self):
        self.jobs = {}
            

    def dataset_path(self, name) -> Path:
        return config.DATASETS_PATH / name / "transformed"
    

    def job_status(self, name: str) -> DatasetJobStatus:
        data = self.repo.get_dataset(name)
        job = self.jobs.get(name)

        return DatasetJobStatus(name, data.is_downloaded(), data.is_normalized(), data.is_tokenized(), job)


    def start_download(self, name: str) -> None:
        datasets = self.load_datasets()

        if name not in datasets:
            raise ValueError("Unknown dataset")

        if name in self.jobs and self.jobs[name].running:
            return

        job = DatasetDownloadJob(
            name=name,
            hf_dataset=datasets[name]["hf_path"],
            normalizer_name=datasets[name]["normalizer"]
        )

        self.jobs[name] = job
        job.start()


    def delete(self, name: str) -> None:
        """
        Delete both the job and the underlying dataset.
        """

        data = self.repo.get_dataset(name)
        data.delete()

        if name in self.jobs:
            del self.jobs[name]
