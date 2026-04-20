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
    