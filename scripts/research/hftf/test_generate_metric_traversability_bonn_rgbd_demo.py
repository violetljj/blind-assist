import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from generate_metric_traversability_bonn_rgbd_demo import _load_sequence
from generate_metric_traversability_bonn_rgbd_demo import (
    _write_fixed_metric_depth_preview,
    observe_center_image_near_surface,
)


class GenerateMetricTraversabilityBonnRgbdDemoTest(unittest.TestCase):
    def test_loads_a_bounded_segment_from_complete_tum_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rgb_rows = []
            depth_rows = []
            for index in range(20):
                timestamp = 1000.0 + index * 0.1
                rgb_rows.append(f"{timestamp:.3f} rgb/{index:06d}.png")
                depth_rows.append(f"{timestamp + 0.005:.3f} depth/{index:06d}.png")
            (root / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
            (root / "depth.txt").write_text("\n".join(depth_rows) + "\n", encoding="utf-8")

            rows = _load_sequence(
                root,
                sequence_id="bounded-real-rgbd",
                start_s=0.5,
                duration_s=1.0,
                target_fps=5.0,
                maximum_frames=None,
            )

            self.assertEqual(5, len(rows))
            self.assertEqual("bounded-real-rgbd", rows[0]["sequence_id"])
            self.assertEqual(0, rows[0]["timestamp_ns"])
            self.assertEqual(800_000_000, rows[-1]["timestamp_ns"])
            self.assertTrue(rows[0]["frame_path"].endswith("rgb\\000005.png"))

    def test_writes_a_fixed_metric_depth_display_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "depth.png"
            assets = {"metric_depth_heatmap_path": str(output)}
            depth_m = np.asarray([[0.5, 1.0, 4.0, np.nan]], dtype=np.float32)

            _write_fixed_metric_depth_preview(depth_m, assets)

            self.assertEqual([0.5, 4.0], assets["metric_depth_display_range_m"])
            preview = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertEqual((1, 4, 3), preview.shape)
            self.assertTrue(np.array_equal((74, 78, 84), preview[0, 3]))

    def test_observes_near_surface_without_a_ground_plane(self) -> None:
        depth_m = np.full((120, 160), 3.0, dtype=np.float32)
        depth_m[40:80, 65:95] = 1.1

        observation = observe_center_image_near_surface(depth_m)

        self.assertEqual("OBSERVED_NEAR_SURFACE", observation["status"])
        self.assertAlmostEqual(1.1, observation["robust_nearest_surface_m"], places=4)
        self.assertIn("no object, ground", observation["claim_ceiling"])


if __name__ == "__main__":
    unittest.main()
