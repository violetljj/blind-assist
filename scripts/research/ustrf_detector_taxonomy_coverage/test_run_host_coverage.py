from __future__ import annotations

import unittest

import numpy as np

from run_host_coverage import channels_by_prediction, decode


class HostCoverageDecoderTest(unittest.TestCase):
    def test_channels_first_layout(self) -> None:
        raw = np.zeros((1, 84, 3), dtype=np.float32)
        raw[0, 4, 1] = 0.9
        self.assertEqual(channels_by_prediction(raw, 80).shape, (84, 3))
        self.assertEqual(float(channels_by_prediction(raw, 80)[4, 1]), np.float32(0.9))

    def test_predictions_first_layout(self) -> None:
        raw = np.zeros((1, 3, 84), dtype=np.float32)
        raw[0, 1, 4] = 0.9
        normalized = channels_by_prediction(raw, 80)
        self.assertEqual(normalized.shape, (84, 3))
        self.assertEqual(float(normalized[4, 1]), np.float32(0.9))

    def test_rejects_incompatible_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            channels_by_prediction(np.zeros((1, 85, 3), dtype=np.float32), 80)

    def test_person_mapping_and_classwise_nms(self) -> None:
        labels = ["person", "chair"]
        raw = np.zeros((1, 6, 3), dtype=np.float32)
        raw[0, :4, 0] = [50, 50, 20, 20]
        raw[0, :4, 1] = [50, 50, 20, 20]
        raw[0, :4, 2] = [50, 50, 20, 20]
        raw[0, 4, 0] = 0.9
        raw[0, 4, 1] = 0.8
        raw[0, 5, 2] = 0.7
        detections, diagnostics = decode(raw, (100, 100), (1.0, 0.0, 0.0), labels, 0.35, 0.45, 100)
        self.assertEqual([row["label"] for row in detections], ["person", "chair"])
        self.assertAlmostEqual(detections[0]["confidence"], 0.9)
        self.assertAlmostEqual(detections[1]["confidence"], 0.7)
        self.assertAlmostEqual(diagnostics["raw_person_max_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
