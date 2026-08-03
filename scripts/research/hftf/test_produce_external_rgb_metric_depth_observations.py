#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from produce_external_rgb_metric_depth_observations import (
    intrinsics_matrix,
    load_manifest,
    robust_roi_median,
    validate_roi,
)


class ExternalRgbMetricDepthProducerTest(unittest.TestCase):
    def test_robust_roi_median_excludes_invalid_and_extreme_values(self) -> None:
        depth = np.full((10, 10), 2.0, dtype=np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = 200.0
        value, valid_pixels, valid_fraction = robust_roi_median(
            depth, (0, 0, 10, 10)
        )
        self.assertEqual(value, 2.0)
        self.assertGreater(valid_pixels, 0)
        self.assertEqual(valid_fraction, 0.99)

    def test_roi_and_intrinsics_validation(self) -> None:
        self.assertEqual(
            validate_roi([1, 2, 9, 8], (10, 10, 3)),
            (1, 2, 9, 8),
        )
        matrix = intrinsics_matrix(
            {"intrinsics_fx_fy_cx_cy": [500, 501, 320, 240]}
        )
        np.testing.assert_allclose(
            matrix,
            [[500, 0, 320], [0, 501, 240], [0, 0, 1]],
        )
        with self.assertRaises(ValueError):
            validate_roi([-1, 0, 5, 5], (10, 10, 3))

    def test_manifest_resolves_relative_frame_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            row = {
                "sequence_id": "static-1m",
                "frame_index": 0,
                "timestamp_ns": 0,
                "frame_path": "frames/000000.png",
                "scenario": "static",
                "camera_motion": "static",
                "truth_depth_m": 1.0,
                "torso_roi_xyxy_px": [1, 1, 2, 2],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_manifest(manifest)
            self.assertEqual(
                Path(loaded[0]["frame_path"]),
                (root / "frames/000000.png").resolve(),
            )

    def test_manifest_accepts_direction_only_dynamic_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            row = {
                "sequence_id": "approach",
                "frame_index": 0,
                "timestamp_ns": 0,
                "frame_path": "frame.png",
                "scenario": "approach",
                "camera_motion": "static",
                "truth_depth_m": None,
                "truth_direction": "approach",
                "torso_roi_xyxy_px": [1, 1, 2, 2],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(
                load_manifest(manifest)[0]["truth_direction"], "approach"
            )


if __name__ == "__main__":
    unittest.main()
