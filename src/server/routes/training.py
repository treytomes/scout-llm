from fastapi import APIRouter, HTTPException, Query

import config
from train.training_job_manager import TrainingJobManager


api_router = APIRouter(prefix="/api/training")
training_manager = TrainingJobManager()


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