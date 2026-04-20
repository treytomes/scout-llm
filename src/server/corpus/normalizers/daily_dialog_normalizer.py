import re
from collections import defaultdict
import datasets as hf_datasets
import config
from .dataset_normalizer import IDatasetNormalizer

MIN_TURNS = 4
MAX_WORDS_PER_TURN = 120

_GARBAGE_RE = re.compile(r"[<>]{2,}")


class DailyDialogNormalizer(IDatasetNormalizer):

    def normalize_dataset(self, data: hf_datasets.Dataset) -> hf_datasets.Dataset:
        convs: dict[int, list[str]] = defaultdict(list)

        for row in data:
            utterance = row.get("utterance", "").strip()
            words = len(utterance.split())
            if words < 3 or words > MAX_WORDS_PER_TURN:
                continue
            if _GARBAGE_RE.search(utterance):
                continue
            convs[row["dialog_id"]].append(utterance)

        chunks = []
        for turns in convs.values():
            if len(turns) < MIN_TURNS:
                continue
            lines = []
            for i, turn in enumerate(turns):
                speaker = config.USER_NAME if i % 2 == 0 else config.MODEL_NAME
                lines.append(f"[{speaker}] {turn}")
            chunks.append({
                "source": "DailyDialog",
                "chunk": "\n".join(lines) + "\n<|endoftext|>",
            })

        return hf_datasets.Dataset.from_list(chunks)