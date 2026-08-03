import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_camera_calibration import (
    CALIBRATION_SCHEMA,
    CameraCalibration,
    FrameRectifier,
    finite_ratio,
    load_calibration,
    pinhole_calibration,
)


class ExternalCameraCalibrationTest(unittest.TestCase):
    def test_zero_distortion_is_identity(self) -> None:
        calibration = pinhole_calibration([100, 100, 10, 5], [20, 10])
        frame = np.arange(20 * 10 * 3, dtype=np.uint8).reshape(10, 20, 3)
        rectified, valid = FrameRectifier(calibration).rectify(frame)
        self.assertIs(rectified, frame)
        self.assertEqual(finite_ratio(valid), 1.0)
        self.assertEqual(calibration.intrinsics, [100.0, 100.0, 10.0, 5.0])

    def test_rectifier_rejects_wrong_resolution(self) -> None:
        calibration = pinhole_calibration([100, 100, 10, 5], [20, 10])
        with self.assertRaisesRegex(ValueError, "differs from calibration size"):
            FrameRectifier(calibration).rectify(np.zeros((11, 20, 3), dtype=np.uint8))

    def test_nonzero_distortion_builds_bounded_valid_mask(self) -> None:
        calibration = CameraCalibration(
            width=40,
            height=30,
            camera_matrix=np.asarray(
                [[35.0, 0.0, 20.0], [0.0, 35.0, 15.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.asarray([0.4, 0.1, 0.0, 0.0, 0.0]),
            source_id="test",
            rectification_required=True,
        )
        rectified, valid = FrameRectifier(calibration).rectify(
            np.zeros((30, 40, 3), dtype=np.uint8)
        )
        self.assertEqual(rectified.shape, (30, 40, 3))
        self.assertGreater(finite_ratio(valid), 0.0)
        self.assertLess(finite_ratio(valid), 1.0)

    def test_json_loader_requires_admitted_profile(self) -> None:
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "admitted": True,
            "image_size_px": [20, 10],
            "camera_matrix": [[100, 0, 10], [0, 100, 5], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_calibration(path)
            self.assertTrue(loaded.source_id.startswith("json:"))
            payload["admitted"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not admitted"):
                load_calibration(path)


if __name__ == "__main__":
    unittest.main()
