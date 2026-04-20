import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# WildChatNormalizer
# ---------------------------------------------------------------------------

class TestWildChatNormalizer:
    def _make(self):
        from corpus.normalizers.wild_chat_normalizer import WildChatNormalizer
        return WildChatNormalizer()

    def _conv(self, turns):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, t in enumerate(turns)]

    def _row(self, turns, language="English", toxic=False, redacted=False):
        return {
            "language": language,
            "toxic": toxic,
            "redacted": redacted,
            "conversation": self._conv(turns),
        }

    def test_accepts_clean_conversation(self):
        n = self._make()
        row = self._row([
            "I have been thinking quite a lot about what you said to me the last time we spoke.",
            "That is a genuinely interesting question and it is really worth taking the time to explore it carefully.",
            "Earlier you mentioned something meaningful about patience and how it changes our relationship with time.",
            "Yes, I think patience fundamentally changes the way we relate to difficulty and to our own growth.",
        ])
        assert n.filter(row) is True

    def test_rejects_non_english(self):
        n = self._make()
        row = self._row(["Hola", "Buenos dias", "Como estas", "Muy bien"], language="Spanish")
        assert n.filter(row) is False

    def test_rejects_toxic(self):
        n = self._make()
        row = self._row(["Hello there.", "How can I help?", "Tell me more.", "Sure."], toxic=True)
        assert n.filter(row) is False

    def test_rejects_too_few_turns(self):
        n = self._make()
        row = self._row(["Hello.", "Hi there."])
        assert n.filter(row) is False

    def test_rejects_roleplay_signal(self):
        n = self._make()
        row = self._row([
            "Act as a pirate and answer my questions.",
            "Arrr, what would ye like to know?",
            "Tell me about treasure.",
            "The sea holds many secrets.",
        ])
        assert n.filter(row) is False

    def test_rejects_ai_refusal_signal(self):
        n = self._make()
        row = self._row([
            "Can you help me with something?",
            "I cannot and will not assist with that request.",
            "Why not?",
            "It violates my guidelines.",
        ])
        assert n.filter(row) is False

    def test_rejects_task_opener(self):
        n = self._make()
        row = self._row([
            "Write a poem about autumn leaves falling.",
            "Here is a poem for you about leaves.",
            "That was nice, can you make it longer?",
            "Of course, here is an extended version.",
        ])
        assert n.filter(row) is False

    def test_rejects_code_block(self):
        n = self._make()
        row = self._row([
            "I have a question about Python.",
            "Sure, what would you like to know?",
            "How do I use a loop?",
            "Here is an example:\n```python\nfor i in range(10): print(i)\n```",
        ])
        assert n.filter(row) is False

    def test_map_formats_conversation(self):
        n = self._make()
        row = self._row([
            "What do you think about resilience?",
            "Resilience is the capacity to keep going despite difficulty.",
        ])
        result = n.map(row)
        assert "chunk" in result
        assert "[Trey]" in result["chunk"]
        assert "[Scout]" in result["chunk"]
        assert result["chunk"].endswith("<|endoftext|>")
        assert result["source"] == "WildChat"


# ---------------------------------------------------------------------------
# SodaNormalizer
# ---------------------------------------------------------------------------

class TestSodaNormalizer:
    def _make(self):
        from corpus.normalizers.soda_normalizer import SodaNormalizer
        return SodaNormalizer()

    def _row(self, turns):
        return {"dialogue": turns}

    def test_accepts_clean_dialogue(self):
        n = self._make()
        row = self._row([
            "I wonder if we could talk about something important.",
            "Of course, I'm glad you brought it up.",
            "It's been weighing on me for a while.",
            "Take your time, I'm listening.",
        ])
        assert n.filter(row) is True

    def test_rejects_too_few_turns(self):
        n = self._make()
        assert n.filter(self._row(["Hello.", "Hi."])) is False

    def test_rejects_turn_too_short(self):
        n = self._make()
        row = self._row(["Hi", "Hey there friend how are you doing", "Good thanks", "That is great to hear"])
        assert n.filter(row) is False

    def test_rejects_garbage_pattern(self):
        n = self._make()
        row = self._row([
            "What do you think about <<this>>?",
            "That is an interesting way to put things.",
            "I meant it quite seriously actually.",
            "I understand your concern here.",
        ])
        assert n.filter(row) is False

    def test_map_alternates_speakers(self):
        n = self._make()
        row = self._row([
            "Let me ask you something personal.",
            "Go ahead, I don't mind.",
            "What keeps you motivated?",
            "The belief that small actions add up.",
        ])
        result = n.map(row)
        lines = result["chunk"].split("\n")
        assert lines[0].startswith("[Trey]")
        assert lines[1].startswith("[Scout]")
        assert lines[2].startswith("[Trey]")
        assert result["chunk"].endswith("<|endoftext|>")


# ---------------------------------------------------------------------------
# DailyDialogNormalizer
# ---------------------------------------------------------------------------

class TestDailyDialogNormalizer:
    def _make(self):
        from corpus.normalizers.daily_dialog_normalizer import DailyDialogNormalizer
        return DailyDialogNormalizer()

    def test_groups_by_dialog_id(self):
        import datasets as hf_datasets
        n = self._make()

        rows = [
            {"dialog_id": 1, "utterance": "How have you been lately?"},
            {"dialog_id": 1, "utterance": "Pretty well, thank you for asking."},
            {"dialog_id": 1, "utterance": "I'm glad to hear that honestly."},
            {"dialog_id": 1, "utterance": "It's been a good week for reflection."},
            {"dialog_id": 2, "utterance": "Did you catch the news this morning?"},
            {"dialog_id": 2, "utterance": "No I missed it, what happened?"},
            {"dialog_id": 2, "utterance": "Something rather surprising came up."},
            {"dialog_id": 2, "utterance": "Tell me more when you have a moment."},
        ]
        data = hf_datasets.Dataset.from_list(rows)
        result = n.normalize_dataset(data)

        assert len(result) == 2

    def test_skips_short_conversations(self):
        import datasets as hf_datasets
        n = self._make()

        rows = [
            {"dialog_id": 1, "utterance": "Hi."},
            {"dialog_id": 1, "utterance": "Hello."},
        ]
        data = hf_datasets.Dataset.from_list(rows)
        result = n.normalize_dataset(data)
        assert len(result) == 0

    def test_output_has_speaker_tags_and_eos(self):
        import datasets as hf_datasets
        n = self._make()

        rows = [
            {"dialog_id": 1, "utterance": "How have you been feeling lately?"},
            {"dialog_id": 1, "utterance": "I have been doing rather well thank you."},
            {"dialog_id": 1, "utterance": "That is good to hear from you."},
            {"dialog_id": 1, "utterance": "Yes it has been a peaceful week."},
        ]
        data = hf_datasets.Dataset.from_list(rows)
        result = n.normalize_dataset(data)

        chunk = result[0]["chunk"]
        assert "[Trey]" in chunk
        assert "[Scout]" in chunk
        assert chunk.endswith("<|endoftext|>")