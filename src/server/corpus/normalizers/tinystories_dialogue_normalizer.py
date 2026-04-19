from .dataset_normalizer import IDatasetNormalizer


class TinyStoriesDialogueNormalizer(IDatasetNormalizer):
    def filter(self, row):
        return bool(row.get("text", "").strip())

    def map(self, row):
        return {
            "source": "tinystories_dialogue",
            "chunk": row["text"],
        }