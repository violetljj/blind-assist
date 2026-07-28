from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
CONFIG = (
    REPO
    / "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r1"
    / "identity_config.v1.json"
)


class MvsecAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        relation = cls.config["camera_relation"]
        raw = relation["raw_intrinsics"]
        cls.raw_k = np.asarray(
            (
                (raw[0], 0.0, raw[2]),
                (0.0, raw[1], raw[3]),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        cls.distortion = np.asarray(
            relation["distortion_coefficients"],
            dtype=np.float64,
        )
        cls.rectification = np.asarray(
            (
                (
                    0.999877311526236,
                    0.015019439766575743,
                    -0.004447282784398257,
                ),
                (
                    -0.014996983873604017,
                    0.9998748347535599,
                    0.005040367172759556,
                ),
                (
                    0.004522429630305261,
                    -0.004973052949604937,
                    0.9999774079320989,
                ),
            ),
            dtype=np.float64,
        )
        rectified = relation["rectified_projection_intrinsic"]
        cls.rectified_k = np.asarray(
            (
                (rectified[0], 0.0, rectified[2]),
                (0.0, rectified[1], rectified[3]),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        cls.map_x, cls.map_y = cv2.fisheye.initUndistortRectifyMap(
            cls.raw_k,
            cls.distortion,
            cls.rectification,
            cls.rectified_k,
            (346, 260),
            cv2.CV_32FC1,
        )

    def test_mono8_frozen_gray_decode_is_exhaustively_exact(self) -> None:
        mono = np.arange(256, dtype=np.uint8).reshape(16, 16)
        rgb = np.asarray(Image.fromarray(mono, mode="L").convert("RGB"))
        decoded = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        np.testing.assert_array_equal(decoded, mono)

    def test_frozen_crop_contains_only_in_bounds_map_coordinates(self) -> None:
        x, y, width, height = self.config["adapter"]["valid_crop_xywh"]
        crop_x = self.map_x[y : y + height, x : x + width]
        crop_y = self.map_y[y : y + height, x : x + width]
        self.assertEqual(crop_x.shape, (255, 346))
        self.assertTrue(np.all(crop_x >= 0.0))
        self.assertTrue(np.all(crop_x <= 345.0))
        self.assertTrue(np.all(crop_y >= 0.0))
        self.assertTrue(np.all(crop_y <= 259.0))

    def test_rectification_is_deterministic_and_shape_locked(self) -> None:
        source = np.arange(346 * 260, dtype=np.uint32)
        source = (source % 256).astype(np.uint8).reshape(260, 346)
        first = cv2.remap(
            source,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[:255, :]
        second = cv2.remap(
            source,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[:255, :]
        self.assertEqual(first.shape, (255, 346))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
