from .dataset_normalizer import IDatasetNormalizer


class ScoutDialogueNormalizer(IDatasetNormalizer):
    def filter(self, row):
        return bool(row.get("text", "").strip())

    def map(self, row):
        return {
            "source": "scout_dialogue",
            "chunk": row["text"],
        }
