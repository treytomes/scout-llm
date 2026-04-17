# routes/training.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import config
from train.models.training_log_entry import TrainingLogEntry
from train.training_job_manager import TrainingJobManager
from train.training_log_repository import TrainingLogRepository


api_router = APIRouter(prefix="/api/training")
view_router = APIRouter(prefix="/training")

training_manager = TrainingJobManager()
log_repo = TrainingLogRepository()


@view_router.get("/")
def index():
    return FileResponse(config.WEB_DIR / "training_dashboard.html")


@api_router.get("/start")
def start_training():
    training_manager.start_training(
        dataset_name="TinyStories",
        model_config=config.MODEL_TINYSTORIES,
        batch_size=8,
        max_steps=1000,
    )

    return {"status": "started"}


@api_router.get("/status")
def training_status():
    return training_manager.status()


# --------------------------------------------------
# Training job control
# --------------------------------------------------

@api_router.get("/start")
def start_training():
    training_manager.start_training(
        dataset_name="TinyStories",
        model_config=config.MODEL_TINYSTORIES,
        batch_size=8,
        max_steps=1000,
    )
    return {"status": "started"}


@api_router.get("/status")
def training_status():
    return training_manager.status()


# --------------------------------------------------
# Training logs
# --------------------------------------------------

@api_router.get("/logs")
def list_training_logs():
    logs = log_repo.list_logs()

    return [
        {
            "filename": log.filename,
            "date": log.log_date.isoformat(),
            "index": log.index,
        }
        for log in logs
    ]


@api_router.get("/logs/{filename}")
def get_training_log(filename: str):
    path = config.TRAINING_LOG_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Training log not found")

    log = log_repo.load(path)

    return {
        "filename": log.filename,
        "date": log.log_date.isoformat(),
        "index": log.index,
        "entries": [e.to_dict() for e in log.entries],
    }


@api_router.get("/logs/{filename}/curve")
def get_training_curve(filename: str):
    path = config.TRAINING_LOG_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Training log not found")

    return log_repo.get_training_curve(path)


@api_router.delete("/logs/{filename}")
def delete_training_log(filename: str):
    path = config.TRAINING_LOG_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Training log not found")

    log = log_repo.load(path)

    log_repo.delete(log)

    return {"status": "deleted"}