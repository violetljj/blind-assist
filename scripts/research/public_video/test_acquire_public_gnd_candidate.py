from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gnd_acquire", SCRIPTS / "acquire_public_gnd_candidate.py")
assert SPEC and SPEC.loader
gnd_acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gnd_acquire)


class PublicGndCandidateTest(unittest.TestCase):
    def test_accepts_valid_source_config(self) -> None:
        gnd_acquire.validate_config(self._config())

    def test_selects_only_expected_public_bounded_file(self) -> None:
        config = self._config()
        selected = gnd_acquire.select_file(config, self._metadata())
        self.assertEqual(42, selected["id"])
        item = gnd_acquire.receipt(config, selected, local_name=None, local_sha256=None)
        self.assertFalse(item["training_execution_authorized"])
        self.assertFalse(item["production_model_replacement_authorized"])

    def test_rejects_restricted_or_oversized_file(self) -> None:
        config = self._config()
        metadata = self._metadata()
        metadata["data"]["latestVersion"]["files"][0]["dataFile"]["restricted"] = True
        with self.assertRaisesRegex(gnd_acquire.AcquisitionError, "restricted"):
            gnd_acquire.select_file(config, metadata)
        metadata = self._metadata()
        metadata["data"]["latestVersion"]["files"][0]["dataFile"]["filesize"] = 101
        with self.assertRaisesRegex(gnd_acquire.AcquisitionError, "exceeds"):
            gnd_acquire.select_file(config, metadata)

    @staticmethod
    def _config() -> dict:
        return {"schema": "blindassist_public_video_candidate_source_v1", "source_id": "gnd", "dataset_persistent_id": "doi:test", "dataset_metadata_url": "https://example.invalid", "expected_license": "CC0-1.0", "candidate_file_name": "AU.zip", "maximum_file_bytes": 100, "source_role": "auxiliary", "source_constraints": ["candidate only"], "production_model_replacement_authorized": False}

    @staticmethod
    def _metadata() -> dict:
        return {"data": {"latestVersion": {"files": [{"dataFile": {"id": 42, "filename": "AU.zip", "filesize": 100, "restricted": False, "md5": "abc"}}]}}}


if __name__ == "__main__":
    unittest.main()
