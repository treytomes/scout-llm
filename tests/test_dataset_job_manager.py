import pytest
from unittest.mock import patch, MagicMock


def _make_manager():
    from workers.dataset_download_job_manager import DatasetDownloadJobManager
    mgr = DatasetDownloadJobManager()
    return mgr


class TestStartDownload:
    def test_raises_key_error_for_unknown_dataset(self):
        mgr = _make_manager()
        with patch.object(mgr.repo, "load_dataset_info", side_effect=KeyError("nope")):
            with pytest.raises(KeyError):
                mgr.start_download("nope")

    def test_raises_value_error_for_missing_hf_path(self):
        mgr = _make_manager()
        info = MagicMock()
        info.hf_path = None
        with patch.object(mgr.repo, "load_dataset_info", return_value=info):
            with pytest.raises(ValueError, match="no hf_path"):
                mgr.start_download("local-only")

    def test_starts_new_job(self):
        mgr = _make_manager()
        info = MagicMock()
        info.hf_path = "org/dataset"
        info.normalizer = "SodaNormalizer"
        with patch.object(mgr.repo, "load_dataset_info", return_value=info), \
             patch("workers.dataset_download_job_manager.DatasetDownloadJob") as MockJob:
            mock_job = MagicMock()
            mock_job.running = False
            MockJob.return_value = mock_job

            mgr.start_download("SODA")

            MockJob.assert_called_once_with(
                name="SODA",
                hf_dataset="org/dataset",
                normalizer_name="SodaNormalizer",
            )
            mock_job.start.assert_called_once()

    def test_does_not_restart_running_job(self):
        mgr = _make_manager()
        info = MagicMock()
        info.hf_path = "org/dataset"
        info.normalizer = "SodaNormalizer"

        existing_job = MagicMock()
        existing_job.running = True
        mgr.jobs["SODA"] = existing_job

        with patch.object(mgr.repo, "load_dataset_info", return_value=info):
            mgr.start_download("SODA")

        existing_job.start.assert_not_called()


class TestJobStatus:
    def test_status_no_job(self):
        mgr = _make_manager()
        mock_dataset = MagicMock()
        mock_dataset.is_downloaded.return_value = True
        mock_dataset.is_normalized.return_value = False
        mock_dataset.is_tokenized.return_value = False

        with patch.object(mgr.repo, "get_dataset", return_value=mock_dataset):
            status = mgr.job_status("SODA")

        assert status.downloaded is True
        assert status.normalized is False
        assert status.downloading is False
        assert status.error is None

    def test_status_with_running_job(self):
        mgr = _make_manager()
        mock_dataset = MagicMock()
        mock_dataset.is_downloaded.return_value = False
        mock_dataset.is_normalized.return_value = False
        mock_dataset.is_tokenized.return_value = False

        job = MagicMock()
        job.running = True
        job.downloaded_bytes = 1024
        job.total_bytes = 4096
        job.complete = False
        job.error = None
        mgr.jobs["SODA"] = job

        with patch.object(mgr.repo, "get_dataset", return_value=mock_dataset):
            status = mgr.job_status("SODA")

        assert status.downloading is True
        assert status.downloaded_bytes == 1024


class TestDelete:
    def test_deletes_dataset_and_job(self):
        mgr = _make_manager()
        mock_dataset = MagicMock()
        mgr.jobs["SODA"] = MagicMock()

        with patch.object(mgr.repo, "get_dataset", return_value=mock_dataset):
            mgr.delete("SODA")

        mock_dataset.delete.assert_called_once()
        assert "SODA" not in mgr.jobs