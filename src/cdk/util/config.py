import os


class Config:
    def __init__(self):
        self.project = os.getenv("PROJECT")
        self.owner = os.getenv("OWNER")
        self.environment = os.getenv("ENVIRONMENT")

