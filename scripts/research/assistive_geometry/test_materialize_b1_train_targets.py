import unittest

import numpy as np

from scripts.research.assistive_geometry.materialize_b1_train_targets import (
    build_band_targets,
    frozen_train_videos,
    resize_cached_dense,
    tensor_intrinsics,
)


class MaterializeB1TrainTargetsTest(unittest.TestCase):
    def test_only_exact_train_identities_are_admitted(self) -> None:
        videos = [{"role": "TRAIN", "visit_id": str(index), "video_id": str(index)} for index in range(16)]
        videos.append({"role": "DEVELOPMENT", "visit_id": "99", "video_id": "99"})
        expected = [{"visit_id": str(index), "video_id": str(index)} for index in range(16)]
        self.assertEqual(16, len(frozen_train_videos({"videos": videos}, expected)))
        expected[0] = {"visit_id": "99", "video_id": "99"}
        with self.assertRaisesRegex(ValueError, "missing"):
            frozen_train_videos({"videos": videos}, expected)

    def test_tensor_intrinsics_uses_independent_scales(self) -> None:
        matrix = np.asarray([[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])
        result = tensor_intrinsics(matrix, (192, 256), (448, 608))
        self.assertAlmostEqual(237.5, float(result[0, 0]))
        self.assertAlmostEqual(280.0, float(result[1, 1]))

    def test_censored_clear_has_occupancy_but_no_clearance(self) -> None:
        clear = {
            "bands": {
                name: {"clearance_m": None, "occupied_by_horizon": {"1.0": False, "1.5": False, "2.0": False}}
                for name in ("left", "center", "right")
            }
        }
        targets = build_band_targets(clear)
        self.assertFalse(np.any(targets["clearance_valid"]))
        self.assertTrue(np.all(targets["occupancy_valid"]))
        self.assertFalse(np.any(targets["occupancy"]))
        self.assertTrue(np.all(targets["band_confidence_valid"]))

    def test_nearest_resize_preserves_mask_domain(self) -> None:
        mask = np.asarray([[True, False], [False, True]])
        resized = resize_cached_dense(mask, (8, 4), mask=True)
        self.assertEqual((8, 4), resized.shape)
        self.assertEqual(np.bool_, resized.dtype)


if __name__ == "__main__":
    unittest.main()
