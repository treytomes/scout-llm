import re
import config
from .dataset_normalizer import IDatasetNormalizer, EOS, extract_vocative_names

MIN_TURNS = 4
MIN_WORDS_PER_TURN = 5
MAX_WORDS_PER_TURN = 120
MAX_TURN_RATIO = 4.0

_GARBAGE_RE = re.compile(r"[<>]{2,}|\[.*?\]")


def _build_name_map(speakers: list[str]) -> dict[str, str]:
    """
    Map each unique character name to a corpus tag name.
    Even-position speakers → USER_NAME, odd-position → MODEL_NAME.
    Generic role words (too short or all-lowercase common nouns) are excluded.
    """
    _GENERIC = {
        "man", "woman", "boy", "girl", "kid", "child", "friend", "boss",
        "client", "customer", "doctor", "nurse", "teacher", "student",
        "officer", "agent", "host", "guest", "caller", "person",
    }
    mapping: dict[str, str] = {}
    for i, name in enumerate(speakers):
        if not name or name.lower() in _GENERIC or len(name) < 3:
            continue
        if name not in mapping:
            tag = config.USER_NAME if i % 2 == 0 else config.MODEL_NAME
            mapping[name] = tag
    return mapping


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
        speakers = row.get("speakers", [])

        tag_names = [
            config.USER_NAME if i % 2 == 0 else config.MODEL_NAME
            for i in range(len(dialogue))
        ]

        # Build name map from metadata speakers
        name_map = _build_name_map(speakers)

        # Scan original dialogue for vocative names missing from metadata
        original_tagged = [(tag_names[i], turn.strip()) for i, turn in enumerate(dialogue)]
        known = set(name_map.keys()) | set(name_map.values())
        extra = {
            k: v for k, v in extract_vocative_names(original_tagged).items()
            if k not in known
        }

        # Merge and apply in a single pass over original text
        full_map = {**name_map, **extra}
        lines = []
        for i, turn in enumerate(dialogue):
            text = turn.strip()
            for char_name, tag_name in full_map.items():
                text = re.sub(rf'\b{re.escape(char_name)}\b', tag_name, text)
            lines.append(f"[{tag_names[i]}] {text}")

        return {
            "source": "SODA",
            "chunk": "\n".join(lines) + f"\n{EOS}",
        }