import hashlib
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.download_dataset import (
    verify_md5,
    find_exact_match,
    find_sample_match,
    download_full_file,
    download_sample_stream,
    main,
)


def test_verify_md5(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    content = b"Malicious-HDG Zenodo Downloader Test Payload"
    test_file.write_bytes(content)
    expected_md5 = hashlib.md5(content).hexdigest()

    assert verify_md5(test_file, expected_md5) is True
    assert verify_md5(test_file, "00000000000000000000000000000000") is False
    assert verify_md5(tmp_path / "non_existent.txt", expected_md5) is False


def test_download_full_file_cache_hit(tmp_path: Path):
    dest = tmp_path / "data_schema.json"
    content = b'{"schema_version": 2, "records": []}'
    dest.write_bytes(content)
    expected_size = len(content)
    expected_md5 = hashlib.md5(content).hexdigest()

    # Cache hit when force=False
    ok = download_full_file(
        url="https://example.com/fake.json",
        dest_path=dest,
        expected_md5=expected_md5,
        expected_size=expected_size,
        force=False,
    )
    assert ok is True


def test_download_sample_stream_float_serialization(tmp_path: Path, monkeypatch):
    """
    Regression test ensuring ijson parses numbers as floats (use_float=True)
    so standard json.dump never crashes with TypeError: Object of type Decimal is not JSON serializable.
    """
    dest = tmp_path / "stream_sample.json"
    mock_payload = b'[{"domain_name": "test.com", "score": 3.14159, "ttl": 300}]'

    # Mock subprocess.Popen to emit bytes from a pipe
    class MockProc:
        def __init__(self):
            self.stdout = io.BytesIO(mock_payload)
            self.stderr = io.BytesIO(b"")

        def terminate(self):
            pass

        def communicate(self, timeout=None):
            return b"", b""

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProc())

    ok = download_sample_stream(
        url="https://example.com/stream.json",
        dest_path=dest,
        target_count=1,
        force=True,
    )
    assert ok is True
    assert dest.exists()
    with open(dest, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["domain_name"] == "test.com"
    assert isinstance(loaded[0]["score"], float)


def test_exact_match_reuses_existing_without_download(tmp_path: Path):
    """
    Ensures that if candidate folder contains exact match (same size and hash),
    it is reused without downloading.
    """
    source_dir = tmp_path / "existing_repo_data"
    target_dir = tmp_path / "target_downloads"
    source_dir.mkdir()
    target_dir.mkdir()

    content = b"Exact Dataset Content for Zenodo DomainRadar"
    expected_size = len(content)
    expected_md5 = hashlib.md5(content).hexdigest()

    existing_file = source_dir / "malware.json"
    existing_file.write_bytes(content)

    dest_file = target_dir / "malware.json"
    assert not dest_file.exists()

    ok = download_full_file(
        url="https://invalid-non-existent-domain.com/malware.json",
        dest_path=dest_file,
        expected_md5=expected_md5,
        expected_size=expected_size,
        force=False,
        search_dirs=[source_dir]
    )
    assert ok is True
    assert dest_file.exists()
    assert dest_file.read_bytes() == content


def test_exact_match_similar_filename_in_folder(tmp_path: Path):
    """
    Ensures that if the folder contains a similar/renamed file with identical size and hash,
    it is detected as an exact match and reused without downloading.
    """
    target_dir = tmp_path / "target_dir"
    target_dir.mkdir()

    content = b"Renamed exact payload content"
    expected_size = len(content)
    expected_md5 = hashlib.md5(content).hexdigest()

    # Pre-existing file with alternative/similar name
    alt_file = target_dir / "malware_backup.json"
    alt_file.write_bytes(content)

    dest_file = target_dir / "malware.json"
    assert not dest_file.exists()

    ok = download_full_file(
        url="https://invalid-non-existent-domain.com/malware.json",
        dest_path=dest_file,
        expected_md5=expected_md5,
        expected_size=expected_size,
        force=False,
        search_dirs=[target_dir]
    )
    assert ok is True
    assert dest_file.exists()
    assert dest_file.read_bytes() == content


