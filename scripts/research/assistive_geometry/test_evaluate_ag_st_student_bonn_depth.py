#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from evaluate_ag_st_student_bonn_depth import (
    BAD_DEPTH_THRESHOLD_M,
    BONN_DEPTH_SCALE,
    FIXED_FRAME_INDICES_BY_SEQUENCE,
    RESULT_SCHEMA,
    depth_error_sums,
    depth_uint16_to_metres,
    finalize_depth_metrics,
    merge_depth_error_sums,
    pair_rgb_depth_unique,
    parent_macro_metrics,
    read_tum_index,
)


class BonnDepthEvaluatorTest(unittest.TestCase):
    def test_fixed_cohort_is_eight_parents_with_three_unique_indices(self) -> None:
        self.assertEqual(len(FIXED_FRAME_INDICES_BY_SEQUENCE), 8)
        for indices in FIXED_FRAME_INDICES_BY_SEQUENCE.values():
            self.assertEqual(len(indices), 3)
            self.assertEqual(len(set(indices)), 3)
            self.assertTrue(all(index >= 0 for index in indices))
        self.assertIn("DEVELOPMENT", RESULT_SCHEMA.upper())

    def test_pairing_uses_each_materialized_depth_at_most_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            for name in ("0.png", "1.png", "2.png"):
                (root / "rgb" / name).write_bytes(b"rgb")
            for name in ("a.png", "b.png"):
                (root / "depth" / name).write_bytes(b"depth")
            (root / "rgb.txt").write_text(
                "0.000 rgb/0.png\n0.010 rgb/1.png\n0.020 rgb/2.png\n",
                encoding="utf-8",
            )
            (root / "depth.txt").write_text(
                "0.005 depth/a.png\n0.025 depth/b.png\n",
                encoding="utf-8",
            )
            rgb = read_tum_index(root, "rgb.txt")
            depth = read_tum_index(root, "depth.txt")
            pairs = pair_rgb_depth_unique(
                "fixture",
                rgb,
                depth,
                maximum_delta_seconds=0.02,
            )
            self.assertEqual(len(pairs), 2)
            selected = [pair.depth.absolute_path for pair in pairs.values()]
            self.assertEqual(len(selected), len(set(selected)))

    def test_pairing_ignores_missing_depth_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            (root / "rgb" / "0.png").write_bytes(b"rgb")
            (root / "depth" / "present.png").write_bytes(b"depth")
            (root / "rgb.txt").write_text(
                "1.000 rgb/0.png\n",
                encoding="utf-8",
            )
            (root / "depth.txt").write_text(
                "0.999 depth/missing.png\n1.001 depth/present.png\n",
                encoding="utf-8",
            )
            pairs = pair_rgb_depth_unique(
                "fixture",
                read_tum_index(root, "rgb.txt"),
                read_tum_index(root, "depth.txt"),
            )
            self.assertEqual(pairs[0].depth.relative_path.as_posix(), "depth/present.png")

    def test_depth_uint16_scale_and_validity_are_source_native(self) -> None:
        raw = np.asarray([[0, 5000], [7500, 1]], dtype=np.uint16)
        depth_m, valid = depth_uint16_to_metres(raw)
        np.testing.assert_allclose(
            depth_m,
            np.asarray([[0.0, 1.0], [1.5, 1.0 / BONN_DEPTH_SCALE]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            valid,
            np.asarray([[False, True], [True, True]]),
        )

    def test_parent_macro_pools_pixels_within_parent_then_means_parents(self) -> None:
        truth = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        valid = np.asarray([[True, True], [True, False]])
        baseline_a = depth_error_sums(
            truth,
            np.asarray([[1.2, 1.2], [1.0, 99.0]], dtype=np.float32),
            valid,
        )
        student_a = depth_error_sums(
            truth,
            np.asarray([[1.05, 1.05], [1.0, 99.0]], dtype=np.float32),
            valid,
        )
        baseline_b = depth_error_sums(
            truth,
            np.full_like(truth, 1.1),
            np.ones_like(valid),
        )
        student_b = depth_error_sums(
            truth,
            np.full_like(truth, 1.0),
            np.ones_like(valid),
        )
        rows = [
            {
                "parent_id": "a",
                "baseline": finalize_depth_metrics(
                    merge_depth_error_sums([baseline_a])
                ),
                "student": finalize_depth_metrics(merge_depth_error_sums([student_a])),
            },
            {
                "parent_id": "b",
                "baseline": finalize_depth_metrics(
                    merge_depth_error_sums([baseline_b])
                ),
                "student": finalize_depth_metrics(merge_depth_error_sums([student_b])),
            },
        ]
        macro = parent_macro_metrics(rows)
        expected_baseline_mae = ((0.2 + 0.2 + 0.0) / 3.0 + 0.1) / 2.0
        self.assertAlmostEqual(
            macro["mae_m"]["initialized_baseline"],
            expected_baseline_mae,
            places=6,
        )
        self.assertLess(
            macro["mae_m"]["student"],
            macro["mae_m"]["initialized_baseline"],
        )
        self.assertEqual(BAD_DEPTH_THRESHOLD_M, 0.10)

    def test_non_monotonic_index_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb.txt").write_text(
                "2.0 rgb/a.png\n1.0 rgb/b.png\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_tum_index(root, "rgb.txt")


if __name__ == "__main__":
    unittest.main()
