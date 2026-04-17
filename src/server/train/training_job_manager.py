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
        """
        Soft stop, specifically.

        Safe stop would require run_training() to periodically check a stop flag (cancellation token?).
        """

        if not self.job:
            return
        logger.warning("Stopping training job not yet implemented safely.")

