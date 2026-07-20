#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from acquire_public_dataverse_candidate import AcquisitionError, download_dataverse_file, download_file, select_file, validate_config


def config() -> dict:
    return {
        "schema": "blindassist_public_dataverse_unlabeled_candidate_source_v1",
        "source_id": "sample",
        "dataset_persistent_id": "doi:example",
        "dataset_metadata_url": "https://example.test/meta",
        "expected_license": "CC0 1.0",
        "candidate_file_name": "sample.7z",
        "maximum_file_bytes": 100,
        "source_role": "unlabeled_candidate",
        "privacy_processing_required": True,
        "source_constraints": ["candidate only"],
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
    }


def metadata(*, restricted: bool = False, license_name: str = "CC0 1.0") -> dict:
    return {
        "data": {
            "latestVersion": {
                "license": {"name": license_name},
                "files": [{"dataFile": {"id": 7, "filename": "sample.7z", "filesize": 5, "md5": hashlib.md5(b"hello").hexdigest(), "restricted": restricted}}],
            }
        }
    }


class PublicDataverseCandidateTests(unittest.TestCase):
    def test_accepts_bounded_cc0_file(self) -> None:
        source = config()
        validate_config(source)
        self.assertEqual(select_file(source, metadata())["id"], 7)

    def test_rejects_license_or_restricted_file(self) -> None:
        source = config()
        with self.assertRaises(AcquisitionError):
            select_file(source, metadata(license_name="CC-BY 4.0"))
        with self.assertRaises(AcquisitionError):
            select_file(source, metadata(restricted=True))

    def test_download_hashes_local_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.7z"
            sha256, md5 = download_file(
                lambda _url: BytesIO(b"hello"),
                "https://example.test/file",
                destination,
                expected_md5=hashlib.md5(b"hello").hexdigest(),
            )
            self.assertEqual(sha256, hashlib.sha256(b"hello").hexdigest())
            self.assertEqual(md5, hashlib.md5(b"hello").hexdigest())

    def test_resumes_partial_download_only_when_server_honors_range(self) -> None:
        case = self

        class FakeResponse:
            status_code = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                case.assertEqual(chunk_size, 1024 * 1024)
                yield b"llo"

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.7z"
            destination.with_suffix(".7z.part").write_bytes(b"he")
            with patch("acquire_public_dataverse_candidate.requests.get", return_value=FakeResponse()) as get:
                sha256, md5 = download_dataverse_file(
                    "https://example.test/file",
                    destination,
                    expected_md5=hashlib.md5(b"hello").hexdigest(),
                )
            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), b"hello")
            self.assertEqual(sha256, hashlib.sha256(b"hello").hexdigest())
            self.assertEqual(md5, hashlib.md5(b"hello").hexdigest())
            self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=2-")


if __name__ == "__main__":
    unittest.main()
