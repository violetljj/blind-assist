import unittest

import numpy as np

from scripts.research.assistive_geometry.validate_b1_train_targets import validate_target_arrays


def valid_arrays() -> tuple[dict, dict[str, np.ndarray]]:
    frame = {
        "frame_stem": "frame-0",
        "source_hw": [192, 256],
        "target_hw": [448, 608],
        "orientation_index": 0,
        "orientation_family": "landscape",
        "ground_plane_valid": True,
    }
    depth = np.ones((192, 256), dtype=np.float32)
    k = np.asarray([[100.0, 0.0, 128.0], [0.0, 100.0, 96.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    arrays = {
        "depth_m_source": depth,
        "depth_valid_source": np.ones_like(depth, dtype=np.bool_),
        "ground_probability_source": np.zeros_like(depth),
        "ground_label_valid_source": np.ones_like(depth, dtype=np.bool_),
        "intrinsics_source": k,
        "intrinsics_tensor": np.asarray([[237.5, 0.0, 304.0], [0.0, 233.33333, 224.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "up_camera": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "camera_height_m": np.asarray(1.5, dtype=np.float32),
        "ground_plane_valid": np.asarray(True, dtype=np.bool_),
        "clearance_m": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "clearance_valid": np.asarray([False, True, False], dtype=np.bool_),
        "occupancy": np.zeros((3, 3), dtype=np.float32),
        "occupancy_valid": np.ones((3, 3), dtype=np.bool_),
        "band_confidence_valid": np.ones(3, dtype=np.bool_),
        "target_hw": np.asarray([448, 608], dtype=np.int32),
        "orientation_index": np.asarray(0, dtype=np.int8),
    }
    return frame, arrays


class ValidateB1TrainTargetsTest(unittest.TestCase):
    def test_valid_target_passes(self) -> None:
        frame, arrays = valid_arrays()
        result = validate_target_arrays(frame, arrays)
        self.assertEqual(1, result["ground_plane_valid"])
        self.assertEqual(1, result["clearance_known_bands"])

    def test_unknown_clearance_cannot_be_filled_as_clear(self) -> None:
        frame, arrays = valid_arrays()
        arrays["clearance_m"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "UNKNOWN clearance"):
            validate_target_arrays(frame, arrays)

    def test_unknown_plane_cannot_leak_ground_mask(self) -> None:
        frame, arrays = valid_arrays()
        frame["ground_plane_valid"] = False
        arrays["ground_plane_valid"] = np.asarray(False, dtype=np.bool_)
        arrays["camera_height_m"] = np.asarray(np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "ground mask leak"):
            validate_target_arrays(frame, arrays)


if __name__ == "__main__":
    unittest.main()
