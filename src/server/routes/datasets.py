from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import config as config
from workers.dataset_download_job_manager import DatasetDownloadJobManager
from corpus.dataset_repository import DatasetRepository
from .models.dataset_preview import DatasetPreview


router = APIRouter()
job_manager = DatasetDownloadJobManager()
repo = DatasetRepository();


@router.get("/datasets")
def index():
    return FileResponse(config.WEB_DIR / "preview.html")


@router.get("/api/datasets")
def list_datasets():
    return repo.list_datasets()


@router.get("/api/datasets/{name}/status")
def dataset_status(name: str):
    return job_manager.job_status(name)


@router.post("/api/datasets/{name}/download")
def start_download(name: str):
    job_manager.start_download(name)
    return {"status": "started"}


@router.get("/api/datasets/{name}/progress")
def dataset_progress(name: str):
    return job_manager.job_status(name)


@router.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    job_manager.delete(name)
    return {"status": "ok"}


@router.post("/api/datasets/{name}/transform")
def transform_dataset(dataset_name: str):
    try:
        data = repo.list_datasets()
        data = repo.get_dataset(dataset_name)
        if not data.is_normalized():
            repo.normalize_dataset(dataset_name)
            data = repo.get_dataset(dataset_name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/datasets/{name}/preview")
def preview_dataset(
    name: str,
    limit: int = Query(20, ge=1, le=200),
    page: int = Query(0, ge=0),
    is_raw: bool = Query(False),
    split_name: str = Query("train"),
):
    try:
        model = repo.get_dataset(name)
        if not model.exists():
            raise Exception("Dataset not downloaded.")
        total_rows = model.get_total_rows(is_raw, split_name)
        rows = model.get_rows(is_raw, split_name, limit, page)

        return DatasetPreview(name, split_name, page, limit, total_rows, rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
