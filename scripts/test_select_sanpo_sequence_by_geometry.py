from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("geometry", SCRIPTS / "select_sanpo_sequence_by_geometry.py")
assert spec and spec.loader
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class SanpoSequenceGeometryTest(unittest.TestCase):
    def blank_mask(self) -> np.ndarray:
        # Other walkable surface covers the full image, providing a usable path.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 17
        return image

    def test_center_obstacle_intrudes_conservative_corridor(self) -> None:
        image = self.blank_mask()
        image[60:90, 44:56, 0] = 20
        components, path = geometry.components_for_mask(image)
        target = components[20][0]
        self.assertGreaterEqual(target["corridor_target_ratio"], 0.12)
        self.assertGreaterEqual(target["bottom_ratio"], 0.45)
        self.assertGreaterEqual(path["walkable_corridor_ratio"], 0.18)

    def test_lateral_pedestrian_is_clean_only_without_center_hazard(self) -> None:
        image = self.blank_mask()
        image[50:90, 3:18, 0] = 12
        components, path = geometry.components_for_mask(image)
        target = components[12][0]
        self.assertLessEqual(target["corridor_target_ratio"], 0.01)
        self.assertLessEqual(target["center_x_ratio"], 0.35)
        image[60:90, 45:55, 0] = 20
        components, path = geometry.components_for_mask(image)
        self.assertTrue(any(item["corridor_target_ratio"] >= 0.12 for item in components[20]))

    def test_lateral_sequence_rejects_a_center_target(self) -> None:
        clean_frame = {
            "path_geometry": {"walkable_corridor_ratio": 1.0},
            "target_clean_lateral": True,
            "target_center_intrusion": False,
            "any_center_hazard": False,
            "best_target": {"corridor_blocking_ratio": 0.0},
        }
        frames = [{**clean_frame, "frame_index": index} for index in range(50)]
        accepted = geometry.summarize_frame_evidence(frames, "lateral_pedestrian_or_ebike", "clean")
        self.assertEqual("accept_for_model_review", accepted["decision"])
        contaminated = [dict(item) for item in frames]
        contaminated[25]["target_center_intrusion"] = True
        rejected = geometry.summarize_frame_evidence(contaminated, "lateral_pedestrian_or_ebike", "contaminated")
        self.assertIn("center_target_contaminates_lateral_negative", rejected["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
