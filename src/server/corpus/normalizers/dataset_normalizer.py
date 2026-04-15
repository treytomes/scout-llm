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
    