import re
import config
from .dataset_normalizer import IDatasetNormalizer, EOS

# Patterns that indicate the conversation is not useful for conversational grounding
_REJECT_ANY_RE = re.compile(
    r"\b(act as|pretend you are|you are now|roleplay as|you are a |DAN"
    r"|ignore previous instructions|ignore your training|stay in character"
    r"|from now on you will|you must always|forget you are an ai"
    r"|I cannot and will not|I'm not able to|As an AI language model"
    r"|I don't have personal|I am just an AI|I'm just an AI"
    r"|As an artificial intelligence|my knowledge cutoff"
    r"|I was trained by|I was created by)\b",
    re.IGNORECASE
)

# Openings that signal a one-shot task with no conversational thread
_TASK_OPENER_RE = re.compile(
    r"^\s*(write|generate|list|translate|summarize|create|give me|tell me a list"
    r"|provide|calculate|convert|fix|debug|code|implement|explain how to"
    r"|what is the formula|solve|find the|compute)\b",
    re.IGNORECASE
)

# Callback signals: later turns that reference prior content by pronoun or explicit echo
_CALLBACK_RE = re.compile(
    r"\b(you (said|mentioned|told|asked|were talking)|earlier you|what you said"
    r"|going back to|as (you|we) (said|discussed|mentioned)|that (thing|point|idea) you"
    r"|remember when|you (brought up|were saying))\b",
    re.IGNORECASE
)

_CODE_BLOCK_RE = re.compile(r"```")

MIN_TURNS = 4
MIN_TOKENS_PER_TURN = 15   # rough word count proxy
MAX_TOKENS_PER_TURN = 200  # tighter — keeps exchanges conversational
MAX_TURN_RATIO = 4.0       # longest turn / shortest turn — high ratio = monologue


class WildChatNormalizer(IDatasetNormalizer):

    def filter(self, row) -> bool:
        # Basic hygiene — already good
        if row.get("language") != "English":
            return False
        if row.get("toxic"):
            return False
        if row.get("redacted"):
            return False

        conversation = row.get("conversation", [])

        # Require minimum turn count
        if len(conversation) < MIN_TURNS:
            return False

        turn_lengths = []
        has_callback = False

        for i, turn in enumerate(conversation):
            content = turn.get("content", "")
            words = len(content.split())

            # Reject any turn containing role-play, jailbreak, or AI-refusal language
            if _REJECT_ANY_RE.search(content):
                return False

            # Reject if any single turn is too short or too long
            if words < MIN_TOKENS_PER_TURN or words > MAX_TOKENS_PER_TURN:
                return False

            # Reject conversations with code blocks
            if _CODE_BLOCK_RE.search(content):
                return False

            turn_lengths.append(words)

            # Check for referent callbacks in turns after the first two
            if i >= 2 and _CALLBACK_RE.search(content):
                has_callback = True

        # Reject monologue-shaped conversations
        if turn_lengths and max(turn_lengths) / max(min(turn_lengths), 1) > MAX_TURN_RATIO:
            return False

        # Reject pure task openers
        first_user = next(
            (t["content"] for t in conversation if t.get("role") == "user"), ""
        )
        if _TASK_OPENER_RE.match(first_user):
            return False

        return True

    def map(self, row) -> dict:
        conversation = row.get("conversation", [])
        lines = []

        for turn in conversation:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()
            if not content:
                continue
            speaker = config.USER_NAME if role == "user" else config.MODEL_NAME
            lines.append(f"[{speaker}] {content}")

        # Join turns with newlines; add EOS at end of conversation
        chunk = "\n".join(lines) + f"\n{EOS}"

        return {
            "source": "WildChat",
            "chunk": chunk,
        }