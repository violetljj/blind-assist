from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.renderer import (
    camera_intrinsics,
    render_frame,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.truth import (
    evaluate_truth_pair,
)


MODULE = Path(__file__).resolve().parents[1] / "rcle_synthetic_scene_r0"


class TruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(
            (MODULE / "dataset_spec.json").read_text(encoding="utf-8")
        )
        cls.scene = cls.spec["splits"]["development"]["scene_families"][0]

    def _pair(self, root: Path, motion_id: str) -> dict:
        motion = next(
            item
            for item in self.spec["motions"]
            if item["motion_id"] == motion_id
        )
        first = render_frame(self.spec, self.scene, motion, 0)
        second = render_frame(self.spec, self.scene, motion, 1)
        first_path = root / f"{motion_id}_0.npy"
        second_path = root / f"{motion_id}_1.npy"
        np.save(first_path, first.depth_m.astype(np.float32), allow_pickle=False)
        np.save(second_path, second.depth_m.astype(np.float32), allow_pickle=False)
        intrinsics = camera_intrinsics(self.spec).tolist()
        previous = {
            "depth_npy_path": first_path.name,
            "intrinsics": intrinsics,
            "t_world_camera": first.t_world_camera.tolist(),
            "timestamp": {"seconds": 0.0},
        }
        current = {
            "depth_npy_path": second_path.name,
            "intrinsics": intrinsics,
            "t_world_camera": second.t_world_camera.tolist(),
            "timestamp": {"seconds": 0.1},
        }
        return evaluate_truth_pair(root, previous, current)

    def test_static_and_pure_yaw_compensate_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for motion_id in ("below_static", "below_pure_yaw"):
                result = self._pair(root, motion_id)
                self.assertTrue(result["evaluable"])
                self.assertLess(
                    abs(result["compensated_expansion_median_per_s"]),
                    1.0e-8,
                )

    def test_forward_approach_has_positive_truth_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self._pair(Path(temporary), "positive_approach")
            self.assertTrue(result["evaluable"])
            self.assertGreater(
                result["compensated_expansion_median_per_s"], 0.02
            )


if __name__ == "__main__":
    unittest.main()
