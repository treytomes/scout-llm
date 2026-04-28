import csv
from pathlib import Path
from typing import List

import config
from .models.training_log import TrainingLogModel
from .models.training_log_entry import TrainingLogEntry


class TrainingLogRepository:
    def __init__(self):
        self.root: Path = config.TRAINING_LOG_DIR
        self.root.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _step_range(path: Path) -> tuple[int | None, int | None]:
        """Return (first_step, last_step) by reading only header + data rows."""
        first = last = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row["step"])
                except (KeyError, ValueError):
                    continue
                if first is None:
                    first = step
                last = step
        return first, last

    def list_logs(self) -> List[TrainingLogModel]:
        logs = []
        for path in sorted(self.root.glob("training_*.csv")):
            log_date, idx = TrainingLogModel.parse_filename(path)
            step_start, step_end = self._step_range(path)
            logs.append(
                TrainingLogModel(
                    path=path,
                    log_date=log_date,
                    index=idx,
                    step_start=step_start,
                    step_end=step_end,
                )
            )
        return logs


    def load(self, path: Path) -> TrainingLogModel:
        log_date, idx = TrainingLogModel.parse_filename(path)
        model = TrainingLogModel(
            path=path,
            log_date=log_date,
            index=idx,
        )

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model.entries.append(
                    TrainingLogEntry.from_dict(row)
                )

        return model

    # --------------------------------------------------
    # Create a new log file
    # --------------------------------------------------

    def create(self, log_date, index) -> TrainingLogModel:
        filename = f"training_{log_date}_{index}.csv"
        path = self.root / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "avg_loss",
                    "lr",
                    "val_loss",
                    "elapsed",
                    "tokens_per_sec",
                    "eta",
                ],
            )
            writer.writeheader()

        return TrainingLogModel(
            path=path,
            log_date=log_date,
            index=index,
        )


    def append(self, log: TrainingLogModel, entry: TrainingLogEntry):
        with open(log.path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "avg_loss",
                    "lr",
                    "val_loss",
                    "elapsed",
                    "tokens_per_sec",
                    "eta",
                ],
            )
            writer.writerow(entry.to_dict())


    def delete(self, log: TrainingLogModel):
        if log.path.exists():
            log.path.unlink()


    def get_training_curve(self, path: Path):
        """
        Stream training metrics from a CSV log without loading the entire file.
        Returns arrays suitable for plotting.
        """

        steps = []
        loss = []
        val_loss = []
        lr = []
        tokens_per_sec = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                steps.append(int(row["step"]))
                loss.append(float(row["loss"]))

                if row["val_loss"]:
                    val_loss.append(float(row["val_loss"]))
                else:
                    val_loss.append(None)

                lr.append(float(row["lr"]))
                tokens_per_sec.append(float(row["tokens_per_sec"]))

        return {
            "step": steps,
            "loss": loss,
            "val_loss": val_loss,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
        }