import re

# Mistral's EOS token. Centralised here so all normalizers stay in sync.
EOS = "</s>"

# Collapse spaces that were inserted around punctuation by Moses-style
# tokenizers (e.g. DailyDialog was distributed pre-tokenized).
# "Frank ' s" → "Frank's",  "word ." → "word."
_PRETOK_RE = re.compile(r" ([',\.!?;:\)\]»])|([(\[«]) ")
_PRETOK_APOS_RE = re.compile(r" n't\b|n' t\b")
_PRETOK_CONTRACTION_RE = re.compile(r" '(s|re|ve|ll|d|m|t)\b|' (s|re|ve|ll|d|m|t)\b")
_MISSING_SPACE_RE = re.compile(r'([.!?])([A-Z])')

# Vocative: "Hey Jim," / "Jim, how" / "thank you, Jim." / "okay Jim?"
_VOCATIVE_RE = re.compile(r'(?:^|[\s,!?])([A-Z][a-z]{2,})(?=[,!?]|\s*$)', re.MULTILINE)

_GENERIC_NAMES = {
    "say", "hey", "hi", "oh", "yes", "no", "okay", "well", "look",
    "man", "sir", "mam", "lady", "son", "dear", "hon", "pal", "bud", "buddy",
    "friend", "boss", "doctor", "nurse", "teacher", "officer", "folks",
    "everyone", "anyone", "someone",
    "father", "mother", "sister", "brother", "uncle", "aunt", "grandma", "grandpa",
    "miss", "mister", "professor", "captain", "general", "detective",
}


def extract_vocative_names(lines: list[tuple[str, str]]) -> dict[str, str]:
    """
    Scan (tag, text) lines for vocative names not already known from metadata.
    A name addressed by USER_NAME speaker → belongs to MODEL_NAME, and vice versa.
    Returns {char_name: tag_name} for newly discovered names only.

    Only accepts candidates that are followed by a comma (strict vocative form:
    "Sorry, Gabi." or "Thanks, Caylie.") to avoid matching sentence-initial
    interjections like "Yeah, I guess" or "Sorry, I didn't mean".
    """
    import config
    # Two vocative forms:
    # 1. word, Name[,.!?]  — "Sorry, Gabi." / "Say, Jim," / "Thanks, Caylie."
    # 2. Hey/Hi Name,      — "Hey John," / "Hi Mary,"
    _STRICT_VOCATIVE_RE = re.compile(
        r'(?:^|\w),\s+([A-Z][a-z]{2,})(?=[,\.!?]|$)'
        r'|(?:^|(?<=\s))(?:Hey|Hi|Hello|Okay|Oh)\s+([A-Z][a-z]{2,}),',
        re.MULTILINE
    )
    mapping: dict[str, str] = {}
    for tag, text in lines:
        addressee_tag = config.MODEL_NAME if tag == config.USER_NAME else config.USER_NAME
        for m in _STRICT_VOCATIVE_RE.finditer(text):
            name = m.group(1) or m.group(2)
            if name and name.lower() not in _GENERIC_NAMES and name not in mapping:
                mapping[name] = addressee_tag
    return mapping


def fix_pretokenized_spacing(text: str) -> str:
    text = text.replace('\u2019', "'").replace('\u2018', "'")  # curly quotes → ASCII
    text = _PRETOK_RE.sub(lambda m: (m.group(1) or "") + (m.group(2) or ""), text)
    text = _PRETOK_APOS_RE.sub("n't", text)
    text = _PRETOK_CONTRACTION_RE.sub(lambda m: f"'{m.group(1) or m.group(2)}", text)
    text = _MISSING_SPACE_RE.sub(r'\1 \2', text)
    return text


class IDatasetNormalizer:
    """
    Base interface for dataset normalization.
    """

    def filter(self, row: dict) -> bool:
        """
        Return True if the row should be kept.
        """
        return True

    def map(self, row: dict) -> dict:
        """
        Transform the row into the target format.
        """
        raise row

    def normalize_dataset(self, data) -> "datasets.Dataset | None":
        """
        Optional override for datasets that need full-dataset access
        (e.g. grouping by dialog_id). Return a Dataset, or None to
        fall back to the standard filter+map pipeline.
        """
        return None
    