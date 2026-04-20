import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_repo(tmp_path, datasets_json=None):
    from corpus.dataset_repository import DatasetRepository

    dataset_file = tmp_path / "datasets.json"
    if datasets_json is not None:
        dataset_file.write_text(json.dumps(datasets_json))

    repo = DatasetRepository()
    repo._DATASETS_PATH = tmp_path
    repo._dataset_file = dataset_file
    return repo


class TestLoadDatasets:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        repo = _make_repo(tmp_path, datasets_json=None)
        assert repo.load_datasets() == []

    def test_parses_entries_with_hf_path(self, tmp_path):
        repo = _make_repo(tmp_path, {
            "TinyStories": {"hf_path": "roneneldan/TinyStories", "normalizer": "TinyStoriesNormalizer"}
        })
        infos = repo.load_datasets()
        assert len(infos) == 1
        assert infos[0].name == "TinyStories"
        assert infos[0].hf_path == "roneneldan/TinyStories"
        assert infos[0].normalizer == "TinyStoriesNormalizer"

    def test_parses_entry_without_hf_path(self, tmp_path):
        repo = _make_repo(tmp_path, {
            "tinystories_dialogue": {"normalizer": "TinyStoriesDialogueNormalizer"}
        })
        infos = repo.load_datasets()
        assert len(infos) == 1
        assert infos[0].hf_path is None

    def test_skips_entries_missing_normalizer(self, tmp_path):
        repo = _make_repo(tmp_path, {
            "bad": {"hf_path": "foo/bar"},
            "good": {"hf_path": "a/b", "normalizer": "SomeNormalizer"},
        })
        infos = repo.load_datasets()
        assert len(infos) == 1
        assert infos[0].name == "good"


class TestLoadDatasetInfo:
    def test_returns_info_for_known_dataset(self, tmp_path):
        repo = _make_repo(tmp_path, {
            "SODA": {"hf_path": "allenai/soda", "normalizer": "SodaNormalizer"}
        })
        info = repo.load_dataset_info("SODA")
        assert info.name == "SODA"

    def test_raises_key_error_for_unknown_dataset(self, tmp_path):
        repo = _make_repo(tmp_path, {})
        with pytest.raises(KeyError, match="unknown-dataset"):
            repo.load_dataset_info("unknown-dataset")


class TestNormalizeDataset:
    def test_delegates_to_normalizer(self, tmp_path):
        repo = _make_repo(tmp_path, {
            "SODA": {"hf_path": "allenai/soda", "normalizer": "SodaNormalizer"}
        })
        mock_normalizer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.normalize = MagicMock()

        with patch.object(repo, "load_dataset_info") as mock_info, \
             patch("corpus.dataset_repository.NormalizerFactory") as MockFactory, \
             patch.object(repo, "get_dataset", return_value=mock_dataset):
            mock_info.return_value = MagicMock(normalizer="SodaNormalizer")
            MockFactory.return_value.get_normalizer.return_value = mock_normalizer

            repo.normalize_dataset("SODA")

        mock_dataset.normalize.assert_called_once_with(mock_normalizer)