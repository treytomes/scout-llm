# routes/training.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from train.training_job_manager import TrainingJobManager
from train.training_log_repository import TrainingLogRepository


api_router = APIRouter(prefix="/api/training")
view_router = APIRouter(prefix="/training")

training_manager = TrainingJobManager()
log_repo = TrainingLogRepository()


_MODEL_CONFIGS = {
    "tinystories": config.MODEL_TINYSTORIES,
    "conversational": config.MODEL_CONVERSATIONAL,
}


class StartTrainingRequest(BaseModel):
    dataset_name: "str | list[str]" = "TinyStories"
    batch_size: int = 8
    max_steps: int = 1000
    module_config: str = "tinystories"
    lr: float = None
    warmup_steps: int = None
    reset_optimizer: bool = False
    freeze_modules: list = None
    freeze_language_core: bool = False


@view_router.get("/")
def index():
    return FileResponse(config.WEB_DIR / "training_dashboard.html")


@api_router.post("/start")
def start_training(req: StartTrainingRequest):
    if training_manager.job and training_manager.job.running:
        raise HTTPException(status_code=409, detail="Training already running")

    model_cfg = _MODEL_CONFIGS.get(req.module_config)
    if model_cfg is None:
        raise HTTPException(status_code=400, detail=f"Unknown module_config: {req.module_config}")

    training_manager.start_training(
        dataset_name=req.dataset_name,
        model_config=model_cfg,
        batch_size=req.batch_size,
        max_steps=req.max_steps,
        lr=req.lr,
        warmup_steps=req.warmup_steps,
        reset_optimizer=req.reset_optimizer,
        freeze_modules=req.freeze_modules,
        freeze_language_core=req.freeze_language_core,
    )
    return {"status": "started"}


@api_router.post("/stop")
def stop_training():
    if not training_manager.job or not training_manager.job.running:
        raise HTTPException(status_code=409, detail="No training job running")
    training_manager.stop()
    return {"status": "stopping"}


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