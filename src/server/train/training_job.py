import logging
import threading
import time
from typing import Optional

import config
from train.train import run_training


logger = logging.getLogger(config.LOGGER_NAME)


class TrainingJob(threading.Thread):
    """
    Background training job.

    Executes run_training() and captures metrics for external monitoring.
    """

    def __init__(
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
        super().__init__(daemon=True)
        logger.info("Creating training job.")

        self.dataset_name = dataset_name
        self.model_config = model_config
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.reset_optimizer = reset_optimizer
        self.freeze_modules = freeze_modules
        self.freeze_language_core = freeze_language_core

        self.running = False
        self.completed = False
        self.error: Optional[str] = None

        self.latest_metrics = None
        self.metrics_history = []

        self.start_time = None
        self._stop_flag = [False]


    def run(self):
        """
        Thread entrypoint.
        """

        logger.info("Training job starting for dataset: %s", self.dataset_name)

        self.running = True
        self.start_time = time.time()

        try:
            for metrics in run_training(
                dataset_name=self.dataset_name,
                model_config=self.model_config,
                batch_size=self.batch_size,
                max_steps=self.max_steps,
                lr=self.lr,
                warmup_steps=self.warmup_steps,
                reset_optimizer=self.reset_optimizer,
                freeze_modules=self.freeze_modules,
                freeze_language_core=self.freeze_language_core,
                stop_flag=self._stop_flag,
            ):
                self.latest_metrics = metrics
                self.metrics_history.append(metrics)

        except Exception as e:
            logger.exception("Training job failed")
            self.error = str(e)

        finally:
            self.running = False
            self.completed = True

            logger.info("Training job finished")


    def stop(self):
        self._stop_flag[0] = True

    def status(self):
        """
        Return status snapshot suitable for API responses.
        """

        return {
            "running": self.running,
            "completed": self.completed,
            "error": self.error,
            "dataset": self.dataset_name,
            "max_steps": self.max_steps,
            "latest_metrics": self.latest_metrics,
            "elapsed": (
                time.time() - self.start_time
                if self.start_time
                else None
            ),
        }