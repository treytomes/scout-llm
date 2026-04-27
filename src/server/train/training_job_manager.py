import logging
from typing import Optional

import config
from .training_job import TrainingJob


logger = logging.getLogger(config.LOGGER_NAME)


class TrainingJobManager:
    """
    Manages the lifecycle of the training job.

    Only one training job may run at a time.
    """


    def __init__(self):
        self.job: Optional[TrainingJob] = None


    def start_training(
        self,
        dataset_name: str,
        model_config: dict,
        batch_size: int,
        max_steps: int,
        lr: float = None,
        warmup_steps: int = None,
        reset_optimizer: bool = False,
        freeze_modules: list = None,
        freeze_language_core: bool = False,
    ):
        logger.info("TrainingJobManager.start_training")
        if self.job and self.job.running:
            logger.warning("Training already running")
            return

        self.job = TrainingJob(
            dataset_name=dataset_name,
            model_config=model_config,
            batch_size=batch_size,
            max_steps=max_steps,
            lr=lr,
            warmup_steps=warmup_steps,
            reset_optimizer=reset_optimizer,
            freeze_modules=freeze_modules,
            freeze_language_core=freeze_language_core,
        )
        self.job.start()
        logger.info("Training job launched.")


    def status(self):
        if not self.job:
            return {
                "running": False,
                "message": "No training job started"
            }
        return self.job.status()


    def latest_metrics(self):
        if not self.job:
            return None
        return self.job.latest_metrics


    def stop(self):
        if not self.job:
            return
        logger.info("Stopping training job.")
        self.job.stop()

