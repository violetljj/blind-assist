#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("plan_p3_r0_2_1_arkit_validation_extension.py")
SPEC = importlib.util.spec_from_file_location("plan_extension", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidationExtensionPlannerTest(unittest.TestCase):
    def test_metadata_excludes_na_and_cross_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["video_id", "visit_id", "fold"])
                writer.writeheader()
                writer.writerow({"video_id": "3", "visit_id": "v1", "fold": "Validation"})
                writer.writerow({"video_id": "2", "visit_id": "v1", "fold": "Validation"})
                writer.writerow({"video_id": "4", "visit_id": "cross", "fold": "Validation"})
                writer.writerow({"video_id": "5", "visit_id": "cross", "fold": "Training"})
                writer.writerow({"video_id": "6", "visit_id": "NA", "fold": "Validation"})
            visits, cross = MODULE.metadata_visits(path)
            self.assertEqual({"v1": ["2", "3"]}, visits)
            self.assertEqual({"cross"}, cross)

    def test_rank_is_deterministic_and_protocol_specific(self) -> None:
        self.assertEqual(MODULE.selection_rank("p", "v"), MODULE.selection_rank("p", "v"))
        self.assertNotEqual(MODULE.selection_rank("p", "v"), MODULE.selection_rank("q", "v"))

    def test_bound_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bound SHA mismatch"):
                MODULE.bound_file(root, {"path": "x", "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
