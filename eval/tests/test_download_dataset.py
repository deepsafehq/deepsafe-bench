"""Tests for the dataset download pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def dataset_root(tmp_path):
    """Create a temporary dataset root with structure."""
    for d in [
        "images/real",
        "images/fake",
        "audio/real",
        "audio/fake",
        "video/real",
        "video/fake",
        "metadata",
    ]:
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


class TestDownloadLog:
    def test_log_starts_empty(self, dataset_root):
        from scripts.download_dataset import DownloadLog

        log = DownloadLog(dataset_root / "metadata" / "download_log.json")
        assert log.is_complete("synthbuster") is False

    def test_mark_complete(self, dataset_root):
        from scripts.download_dataset import DownloadLog

        log_path = dataset_root / "metadata" / "download_log.json"
        log = DownloadLog(log_path)
        log.mark_complete("synthbuster", file_count=9000)
        assert log.is_complete("synthbuster") is True

    def test_log_persists_to_disk(self, dataset_root):
        from scripts.download_dataset import DownloadLog

        log_path = dataset_root / "metadata" / "download_log.json"
        log = DownloadLog(log_path)
        log.mark_complete("synthbuster", file_count=9000)
        log2 = DownloadLog(log_path)
        assert log2.is_complete("synthbuster") is True


class TestDiskSpaceCheck:
    def test_check_disk_space_passes_with_enough(self, dataset_root):
        from scripts.download_dataset import check_disk_space

        assert check_disk_space(dataset_root, required_gb=0.001) is True

    def test_check_disk_space_fails_with_too_much(self, dataset_root):
        from scripts.download_dataset import check_disk_space

        assert check_disk_space(dataset_root, required_gb=999_000_000) is False


class TestFileValidation:
    def test_validate_image_file(self, tmp_path):
        from scripts.download_dataset import validate_file

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert validate_file(img) is True

    def test_validate_empty_file_fails(self, tmp_path):
        from scripts.download_dataset import validate_file

        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        assert validate_file(f) is False

    def test_validate_tiny_file_fails(self, tmp_path):
        from scripts.download_dataset import validate_file

        f = tmp_path / "tiny.wav"
        f.write_bytes(b"\x00" * 10)
        assert validate_file(f) is False


class TestDownloadWget:
    """Tests for the download_wget function."""

    def test_creates_dest_dir(self, tmp_path):
        """Destination directory is created if it does not exist."""
        from scripts.download_dataset import download_wget

        dest = tmp_path / "new_dir"
        url = "https://example.com/file.tar.gz"
        with patch("scripts.download_dataset.subprocess.run"):
            download_wget(url, dest)
        assert dest.exists()

    def test_calls_wget_with_correct_args(self, tmp_path):
        """wget is invoked with -c (continue) and -O (output) flags."""
        from scripts.download_dataset import download_wget

        url = "https://example.com/data/train-clean-100.tar.gz"
        with patch("scripts.download_dataset.subprocess.run") as mock_run:
            download_wget(url, tmp_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "wget"
        assert "-c" in cmd
        assert "-O" in cmd
        assert cmd[-1] == url
        expected_dest = str(tmp_path / "train-clean-100.tar.gz")
        assert expected_dest in cmd

    def test_skips_when_file_exists(self, tmp_path):
        """Existing file is not re-downloaded."""
        from scripts.download_dataset import download_wget

        url = "https://example.com/file.tar.gz"
        existing = tmp_path / "file.tar.gz"
        existing.write_bytes(b"\x00" * 100)

        with patch("scripts.download_dataset.subprocess.run") as mock_run:
            result = download_wget(url, tmp_path)

        mock_run.assert_not_called()
        assert result == [existing]

    def test_returns_dest_file_path(self, tmp_path):
        """Returns a list containing the destination file path."""
        from scripts.download_dataset import download_wget

        url = "https://example.com/val2017.zip"
        with patch("scripts.download_dataset.subprocess.run"):
            result = download_wget(url, tmp_path)

        assert len(result) == 1
        assert result[0] == tmp_path / "val2017.zip"

    def test_extracts_filename_from_url(self, tmp_path):
        """Filename is extracted from the last segment of the URL."""
        from scripts.download_dataset import download_wget

        url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
        with patch("scripts.download_dataset.subprocess.run"):
            result = download_wget(url, tmp_path)

        assert result[0].name == "train-clean-100.tar.gz"
