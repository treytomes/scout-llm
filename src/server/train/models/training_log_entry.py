from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingLogEntry:
    step: int
    loss: float
    avg_loss: float
    lr: float
    val_loss: Optional[float]
    elapsed: float
    tokens_per_sec: float
    eta: float


    @staticmethod
    def from_dict(row: dict) -> "TrainingLogEntry":
        return TrainingLogEntry(
            step=int(row["step"]),
            loss=float(row["loss"]),
            avg_loss=float(row["avg_loss"]),
            lr=float(row["lr"]),
            val_loss=float(row["val_loss"]) if row["val_loss"] not in (None, "", "None") else None,
            elapsed=float(row["elapsed"]),
            tokens_per_sec=float(row["tokens_per_sec"]),
            eta=float(row["eta"]),
        )


    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "loss": self.loss,
            "avg_loss": self.avg_loss,
            "lr": self.lr,
            "val_loss": self.val_loss,
            "elapsed": self.elapsed,
            "tokens_per_sec": self.tokens_per_sec,
            "eta": self.eta,
        }