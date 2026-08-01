from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_stage_c_e0_fresh_media as subject


class FreshMediaAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "dataset_binding": {
                "metadata_files": {
                    "meta/a.json": "a" * 64,
                    "meta/b.json": "b" * 64,
                }
            },
            "frozen_sources": [
                {
                    "role": "train",
                    "trajectory": "one",
                    "files": {
                        "pose": {"path": "data/one.parquet"},
                        "rgb": {"path": "video/rgb/one.mp4"},
                        "depth": {"path": "video/depth/one.mkv"},
                    },
                },
                {
                    "role": "heldout",
                    "trajectory": "two",
                    "files": {
                        "pose": {"path": "data/two.parquet"},
                        "rgb": {"path": "video/rgb/two.mp4"},
                        "depth": {"path": "video/depth/two.mkv"},
                    },
                },
            ],
        }

    def test_allow_patterns_are_exact_sorted_and_unique(self) -> None:
        self.assertEqual(
            subject._allow_patterns(self.protocol),
            [
                "data/one.parquet",
                "data/two.parquet",
                "meta/a.json",
                "meta/b.json",
                "video/depth/one.mkv",
                "video/depth/two.mkv",
                "video/rgb/one.mp4",
                "video/rgb/two.mp4",
            ],
        )

    def test_lock_rejects_unvalidated_terminal(self) -> None:
        lock = {
            "schema": subject.LOCK_SCHEMA,
            "terminal": "NOT_EVALUABLE",
        }
        with self.assertRaisesRegex(ValueError, "not validated"):
            subject._validate_lock(
                self.protocol, Path(__file__), lock
            )

    def test_lock_rejects_prior_fresh_media_read(self) -> None:
        lock = {
            "schema": subject.LOCK_SCHEMA,
            "terminal": "E0_FRESH_SOURCE_LOCK_VALIDATED",
            "protocol_sha256": subject._sha256(Path(__file__)),
            "selected_sources": self.protocol["frozen_sources"],
            "rgb_or_depth_media_content_read": True,
            "exact_selected_media_acquisition_authorized": True,
        }
        with self.assertRaisesRegex(ValueError, "unexpectedly read"):
            subject._validate_lock(
                self.protocol, Path(__file__), lock
            )


if __name__ == "__main__":
    unittest.main()
