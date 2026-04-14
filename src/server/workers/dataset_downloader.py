import threading
import requests
from pathlib import Path
from huggingface_hub import HfApi


DATA_ROOT = Path("./datasets")


class DatasetDownloadJob:
    def __init__(self, name, hf_dataset):
        self.name = name
        self.hf_dataset = hf_dataset

        self.downloaded_bytes = 0
        self.total_bytes = 0

        self.running = False
        self.complete = False
        self.thread = None


    def start(self):
        if self.running:
            return

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()


    def _download_file(self, url, dest_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue

                    f.write(chunk)

                    size = len(chunk)
                    self.downloaded_bytes += size


    def _run(self):

        dataset_dir = DATA_ROOT / self.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        api = HfApi()

        info = api.dataset_info(self.hf_dataset)

        files = [
            s for s in info.siblings
            if s.rfilename.endswith(".json") or s.rfilename.endswith(".parquet")
        ]

        # compute total dataset size
        self.total_bytes = sum(s.size for s in files if s.size)

        try:
            for file in files:
                url = f"https://huggingface.co/datasets/{self.hf_dataset}/resolve/main/{file.rfilename}"
                print(f"Downloading: {url}")

                local_path = dataset_dir / file.rfilename
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if local_path.exists():
                    self.downloaded_bytes += local_path.stat().st_size
                    continue

                self._download_file(url, local_path)

            self.complete = True

        finally:
            self.running = False