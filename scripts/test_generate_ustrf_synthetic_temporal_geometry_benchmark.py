#!/usr/bin/env python3
"""Deterministic checks for the analytic USTRF temporal geometry benchmark generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import generate_ustrf_synthetic_temporal_geometry_benchmark as subject


class SyntheticTemporalGeometryBenchmarkTest(unittest.TestCase):
    def test_generation_and_cpu_audit_preserve_known_geometry_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            specification = subject.generate(root)
            report = subject.audit(root, require_cuda=False)

            self.assertEqual(specification["format"], subject.SCHEMA)
            self.assertEqual(specification["sequence_count"], 14)
            self.assertEqual(specification["frame_count"], 28)
            self.assertTrue(report["ok"])
            self.assertEqual(report["static_target_pair_count"], 8)
            self.assertEqual(report["visibility_gap_pair_count"], 2)
            self.assertEqual(report["static_reprojection_rmse_meters"], 0.0)
            self.assertEqual(report["target_classification_count"], 16)
            self.assertEqual(report["target_classification_accuracy"], 1.0)
            self.assertEqual(report["visibility_gap_missing_depth_count"], 4)
            self.assertEqual(report["visibility_gap_false_drop_count"], 0)
            self.assertEqual(report["drop_expected_sample_count"], 4)
            self.assertEqual(report["drop_detected_sample_count"], 4)
            self.assertEqual(report["drop_recall"], 1.0)
            self.assertGreater(report["valid_depth_samples"], 0)
            self.assertTrue((root / "kotlin_replay.tsv").is_file())
            self.assertEqual(len((root / "kotlin_replay.tsv").read_text(encoding="utf-8").splitlines()), 15)
            self.assertEqual(len(report["kotlin_replay_tsv_sha256"]), 64)
            self.assertFalse(report["production_authority"])

    def test_generator_refuses_to_overwrite_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            subject.generate(root)
            with self.assertRaises(FileExistsError):
                subject.generate(root)


if __name__ == "__main__":
    unittest.main()
