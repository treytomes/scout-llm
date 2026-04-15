from pydantic import BaseModel


class TrainingPlanRequest(BaseModel):
    block_size: int
    batch_size: int
    split_name: str = "train"