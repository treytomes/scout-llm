from fastapi import APIRouter
from server.runtime.dataset_manager import DatasetManager

router = APIRouter()

manager = DatasetManager()


@router.get("/api/datasets")
def list_datasets():
    return manager.list_datasets()


@router.get("/api/datasets/{name}")
def dataset_status(name: str):
    return manager.status(name)


@router.post("/api/datasets/{name}/download")
def start_download(name: str):
    manager.start_download(name)
    return {"status": "started"}


@router.get("/api/datasets/{name}/progress")
def dataset_progress(name: str):
    return manager.status(name)


@router.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    manager.delete(name)
    return {"status": "deleted"}
