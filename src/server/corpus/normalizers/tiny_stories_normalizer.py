from .dataset_normalizer import IDatasetNormalizer


class TinyStoriesNormalizer(IDatasetNormalizer):
    def filter(self, row):
        return True
    
    
    def map(self, row):
        return {
            "source": "TinyStories",
            "chunk": row["text"],
        }
