import re
import config
from .dataset_normalizer import IDatasetNormalizer

MIN_TURNS = 4
MIN_WORDS_PER_TURN = 5
MAX_WORDS_PER_TURN = 120
MAX_TURN_RATIO = 4.0

_GARBAGE_RE = re.compile(r"[<>]{2,}|\[.*?\]")


class SodaNormalizer(IDatasetNormalizer):

    def filter(self, row) -> bool:
        dialogue = row.get("dialogue", [])
        if len(dialogue) < MIN_TURNS:
            return False

        lengths = []
        for turn in dialogue:
            words = len(turn.split())
            if words < MIN_WORDS_PER_TURN or words > MAX_WORDS_PER_TURN:
                return False
            if _GARBAGE_RE.search(turn):
                return False
            lengths.append(words)

        if max(lengths) / max(min(lengths), 1) > MAX_TURN_RATIO:
            return False

        return True

    def map(self, row) -> dict:
        dialogue = row["dialogue"]
        lines = []
        for i, turn in enumerate(dialogue):
            speaker = config.USER_NAME if i % 2 == 0 else config.MODEL_NAME
            lines.append(f"[{speaker}] {turn.strip()}")

        return {
            "source": "SODA",
            "chunk": "\n".join(lines) + "\n<|endoftext|>",
        }