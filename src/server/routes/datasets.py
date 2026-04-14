from fastapi import APIRouter, HTTPException, Query

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


@router.get("/api/datasets/{name}/preview")
def preview_dataset(
    name: str,
    limit: int = Query(20, ge=1, le=200),
    page: int = Query(0, ge=0),
    split: str = Query("train"),
):
    try:
        return manager.get_preview(name, split, limit, page)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
