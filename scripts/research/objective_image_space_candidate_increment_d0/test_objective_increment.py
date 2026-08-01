from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_objective_view, summarize_masks, trapezoid_roi
from .prepare_objective_view import prepare


class ObjectiveIncrementTest(unittest.TestCase):
    def test_trapezoid_is_mechanical_and_widens_downward(self) -> None:
        roi = trapezoid_roi()
        self.assertEqual(roi.shape, (256, 256))
        self.assertFalse(roi[0].any())
        self.assertGreater(int(roi[-1].sum()), int(roi[128].sum()))

    def test_combined_arm_recovers_detector_uncovered_truth(self) -> None:
        truth = np.zeros((1, 256, 256), dtype=np.uint8)
        predicted = np.zeros_like(truth)
        detector = np.zeros_like(truth, dtype=bool)
        truth[0, 100:120, 100:120] = 1
        predicted[0, 100:120, 100:120] = 1
        result = summarize_masks(truth, predicted, detector)
        self.assertEqual(result["arms"]["A_YOLO_ONLY"]["recall"], 0.0)
        self.assertEqual(result["arms"]["C_YOLO_PLUS_PIDNET"]["recall"], 1.0)
        self.assertEqual(result["components"]["component_recall"], 1.0)
        self.assertEqual(
            result["components"]["false_activation_components_per_frame"], 0.0
        )

    def test_unknown_pixels_are_not_false_positive_denominator(self) -> None:
        truth = np.full((1, 256, 256), 3, dtype=np.uint8)
        predicted = np.ones_like(truth)
        detector = np.ones_like(truth, dtype=bool)
        result = summarize_masks(truth, predicted, detector)
        self.assertEqual(result["arms"]["C_YOLO_PLUS_PIDNET"]["fp"], 0)

    def test_prepare_strips_event_and_action_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "oracle_masks").mkdir()
            image = root / "images" / "00.png"
            mask = root / "oracle_masks" / "00.png"
            Image.new("RGB", (512, 288), (1, 2, 3)).save(image)
            Image.new("L", (256, 256), 0).save(mask)
            from .common import sha256_file

            source = root / "manifest.json"
            source.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "source_session_id": "session-a",
                                "positive": True,
                                "bucket": "forbidden",
                                "alertable_interval_frames": [0, 1],
                                "frames": [
                                    {
                                        "frame_index": 0,
                                        "source_frame_index": 10,
                                        "timestamp_ms": 0,
                                        "image_path": "images/00.png",
                                        "image_sha256": sha256_file(image),
                                        "oracle_mask_path": "oracle_masks/00.png",
                                        "oracle_mask_sha256": sha256_file(mask),
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "objective.jsonl"
            receipt = root / "receipt.json"
            prepare(source, output, receipt)
            rows = load_objective_view(output)
            self.assertEqual(len(rows), 1)
            self.assertNotIn("positive", rows[0])
            self.assertNotIn("bucket", rows[0])
            self.assertNotIn("alertable_interval_frames", rows[0])


if __name__ == "__main__":
    unittest.main()
