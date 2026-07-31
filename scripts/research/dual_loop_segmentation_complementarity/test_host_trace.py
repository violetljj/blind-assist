from __future__ import annotations

import unittest

import numpy as np

from .produce_host_trace import decode, letterbox


class HostTraceDecoderTests(unittest.TestCase):
    def test_letterbox_matches_fixed_320_contract(self) -> None:
        from PIL import Image

        tensor, transform = letterbox(Image.new("RGB", (640, 480), (255, 0, 0)))
        self.assertEqual(tensor.shape, (1, 320, 320, 3))
        self.assertEqual(transform, (0.5, 0.0, 40.0))
        self.assertAlmostEqual(float(tensor[0, 40, 0, 0]), 1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 0.0)

    def test_decode_maps_boxes_and_applies_classwise_nms(self) -> None:
        labels = [f"class_{index}" for index in range(80)]
        output = np.zeros((1, 84, 2100), dtype=np.float32)
        output[0, 0:4, 0] = [160.0, 160.0, 100.0, 80.0]
        output[0, 4, 0] = 0.9
        output[0, 0:4, 1] = [160.0, 160.0, 90.0, 70.0]
        output[0, 4, 1] = 0.8
        output[0, 0:4, 2] = [160.0, 160.0, 90.0, 70.0]
        output[0, 5, 2] = 0.7
        detections = decode(output, (320, 320), (1.0, 0.0, 0.0), labels)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_id"], 0)
        self.assertEqual(detections[1]["class_id"], 1)
        self.assertEqual(detections[0]["left"], 110.0)
        self.assertEqual(detections[0]["right"], 210.0)
        self.assertEqual(detections[0]["source"], "OBJECT_DETECTOR")

    def test_decode_rejects_non_finite_output(self) -> None:
        labels = [f"class_{index}" for index in range(80)]
        output = np.zeros((1, 84, 2100), dtype=np.float32)
        output[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(output, (320, 320), (1.0, 0.0, 0.0), labels)


if __name__ == "__main__":
    unittest.main()
