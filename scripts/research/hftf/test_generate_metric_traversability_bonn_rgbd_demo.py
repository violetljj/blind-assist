import tempfile
import unittest
from pathlib import Path

from generate_metric_traversability_bonn_rgbd_demo import _load_sequence


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


if __name__ == "__main__":
    unittest.main()
