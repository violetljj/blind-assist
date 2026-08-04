#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from metric_traversability_field import AlertMapper, build_metric_traversability_field
from render_metric_traversability_field_demo import HEIGHT, WIDTH, render
from test_metric_traversability_field import synthetic_depth


class RenderMetricTraversabilityFieldDemoTest(unittest.TestCase):
    def test_renders_valid_and_unknown_four_panel_frames(self) -> None:
        depth, intrinsics = synthetic_depth()
        scale = {
            "status": "VALID",
            "scale": 1.0,
            "anchor_age_ns": 10,
            "anchor_source": "synthetic-test-anchor",
        }
        valid = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale=scale,
            source_model="synthetic-test-depth",
        )
        unknown = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale={"status": "UNKNOWN_NO_METRIC_SCALE_ANCHOR"},
            source_model="synthetic-test-depth",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.png"
            cv2.imwrite(str(rgb_path), np.full((80, 120, 3), 150, dtype=np.uint8))
            cv2.imwrite(str(depth_path), np.full((80, 120, 3), (20, 140, 220), dtype=np.uint8))
            records = [
                {
                    "frame_index": 1,
                    "metric_traversability_field": valid,
                    "shadow_demo_alert_projection": AlertMapper().map(valid),
                    "visualization_assets": {
                        "rgb_path": str(rgb_path),
                        "metric_depth_heatmap_path": str(depth_path),
                    },
                },
                {
                    "frame_index": 2,
                    "metric_traversability_field": unknown,
                    "shadow_demo_alert_projection": AlertMapper().map(unknown),
                },
            ]
            frames = root / "frames"
            summary = render(records, frames_dir=frames, video_path=None, fps=10.0)

            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["valid_fields"], 1)
            image = cv2.imread(str(frames / "frame_000000.png"))
            self.assertEqual(image.shape[:2], (HEIGHT, WIDTH))
            self.assertTrue((frames / "frame_000001.png").is_file())
            self.assertIn("not algorithm evidence", summary["claim_ceiling"])


if __name__ == "__main__":
    unittest.main()
