class DatasetInfo:
    name: str
    hf_path: str | None
    normalizer: str

    def __init__(self, name: str, hf_path: str | None, normalizer: str) -> None:
        self.name = name
        self.hf_path = hf_path
        self.normalizer = normalizer