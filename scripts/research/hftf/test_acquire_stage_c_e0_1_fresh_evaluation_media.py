from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_stage_c_e0_1_fresh_evaluation_media as subject


class E01FreshEvaluationAcquisitionTest(unittest.TestCase):
    def test_compatible_protocol_uses_only_new_evaluation_sources(self) -> None:
        protocol = {
            "fresh_evaluation_selection": {
                "dataset_repo": "repo",
                "dataset_revision": "rev",
            },
            "fresh_evaluation_sources": [{"trajectory": "new"}],
        }
        e0 = {
            "dataset_binding": {
                "metadata_files": {"meta/a": "hash"}
            }
        }
        result = subject._e0_compatible_protocol(protocol, e0)
        self.assertEqual(result["frozen_sources"], [{"trajectory": "new"}])
        self.assertEqual(
            result["dataset_binding"]["metadata_files"],
            {"meta/a": "hash"},
        )

    def test_unvalidated_lock_fails_closed(self) -> None:
        lock = {"schema": subject.LOCK_SCHEMA, "terminal": "NO"}
        with self.assertRaisesRegex(ValueError, "not validated"):
            subject._validate_lock({}, Path(__file__), lock)


if __name__ == "__main__":
    unittest.main()
