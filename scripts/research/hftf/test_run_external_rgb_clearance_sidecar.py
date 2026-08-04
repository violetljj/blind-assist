#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from metric_scale_anchor import MetricScaleAnchor, MetricScaleTracker
from run_external_rgb_clearance_sidecar import (
    assess_image_quality,
    calibrated_field,
    write_research_depth_artifact,
    write_visualization_assets,
)


class RunExternalRgbClearanceSidecarTest(unittest.TestCase):
    def test_field_is_unknown_until_anchor_exists(self) -> None:
        tracker = MetricScaleTracker(100)
        raw = {
            "status": "VALID",
            "bands": {
                band: {"clearance_m": 2.0}
                for band in ("left", "center", "right")
            },
        }
        self.assertEqual(
            calibrated_field(raw, tracker, 10)["status"],
            "UNKNOWN_NO_METRIC_SCALE_ANCHOR",
        )

    def test_field_applies_scale_and_recomputes_horizons(self) -> None:
        tracker = MetricScaleTracker(100)
        tracker.update(MetricScaleAnchor(10, 0.5, 3, 0.0, "tof"))
        raw = {
            "status": "VALID",
            "bands": {
                "left": {"clearance_m": 2.0},
                "center": {"clearance_m": 3.2},
                "right": {"clearance_m": None},
            },
        }
        result = calibrated_field(raw, tracker, 20)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["bands"]["left"]["clearance_m"], 1.0)
        self.assertTrue(result["bands"]["center"]["occupied_by_horizon"]["2.0"])
        self.assertFalse(result["bands"]["right"]["occupied_by_horizon"]["2.0"])

    def test_visualization_assets_distinguish_metric_depth_from_unknown(self) -> None:
        bgr = np.full((24, 32, 3), 120, dtype=np.uint8)
        depth = np.linspace(0.5, 3.0, 24 * 32, dtype=np.float64).reshape(24, 32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            valid = write_visualization_assets(bgr, depth, output, "seq/a", 1)
            unknown = write_visualization_assets(bgr, None, output, "seq/a", 2)
            self.assertEqual(
                valid["metric_depth_heatmap_status"], "VALID_DEVELOPMENT_DISPLAY"
            )
            self.assertEqual(unknown["metric_depth_heatmap_status"], "UNKNOWN")
            self.assertIsNotNone(cv2.imread(valid["rgb_path"]))
            self.assertIsNotNone(cv2.imread(unknown["metric_depth_heatmap_path"]))

    def test_image_quality_and_replayable_depth_artifact_fail_closed(self) -> None:
        flat = np.full((240, 320, 3), 100, dtype=np.uint8)
        checker = np.indices((240, 320)).sum(axis=0) % 2
        textured = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.assertFalse(assess_image_quality(flat)["pass"])
        self.assertTrue(assess_image_quality(textured)["pass"])
        raw = np.full((12, 16), 2.0, dtype=np.float32)
        calibrated = raw * 0.5
        with tempfile.TemporaryDirectory() as directory:
            receipt = write_research_depth_artifact(
                raw, calibrated, Path(directory), "seq", 3
            )
            self.assertEqual(len(receipt["sha256"]), 64)
            with np.load(receipt["path"]) as payload:
                np.testing.assert_allclose(payload["raw_depth"], raw)
                np.testing.assert_allclose(payload["calibrated_depth_m"], calibrated)
                self.assertTrue(bool(payload["calibrated_available"][0]))


if __name__ == "__main__":
    unittest.main()
