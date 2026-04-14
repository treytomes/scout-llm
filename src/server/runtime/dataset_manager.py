import json
from pathlib import Path
from server.workers.dataset_downloader import DatasetDownloadJob


DATA_ROOT = Path("./datasets")
DATASET_FILE = DATA_ROOT / "datasets.json"


class DatasetManager:
    def __init__(self):
        self.jobs = {}


    def load_datasets(self):
        if not DATASET_FILE.exists():
            return {}

        with open(DATASET_FILE, "r") as f:
            return json.load(f)
        

    def dataset_path(self, name):
        return DATA_ROOT / name


    def list_datasets(self):
        datasets = self.load_datasets()
        return [self.status(name) for name in datasets]
    

    def status(self, name):
        path = self.dataset_path(name)

        job = self.jobs.get(name)

        return {
            "name": name,
            "downloaded": path.exists(),
            "downloading": job.running if job else False,
            "downloaded_bytes": job.downloaded_bytes if job else 0,
            "total_bytes": job.total_bytes if job else 0,
            "complete": job.complete if job else False
        }


    def start_download(self, name):
        datasets = self.load_datasets()

        if name not in datasets:
            raise ValueError("Unknown dataset")

        if name in self.jobs and self.jobs[name].running:
            return

        job = DatasetDownloadJob(
            name=name,
            hf_dataset=datasets[name]["hf_path"]
        )

        self.jobs[name] = job
        job.start()


    def delete(self, name):
        path = self.dataset_path(name)

        if path.exists():
            import shutil
            shutil.rmtree(path)

        if name in self.jobs:
            del self.jobs[name]