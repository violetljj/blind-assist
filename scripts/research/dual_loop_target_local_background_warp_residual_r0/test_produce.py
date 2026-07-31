from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .common import ABSTENTION_REASONS
from .produce import _reject_forbidden_inputs, process_pair, ransac_similarity, ring_mask


def _row(**overrides):
    row = {
        "source_id": "SYNTH",
        "session_id": "SESSION",
        "sequence_id": "SEQ",
        "parent_event_id": "EVENT",
        "previous_source_frame_id": "f0",
        "current_source_frame_id": "f1",
        "previous_frame_index": 0,
        "current_frame_index": 1,
        "previous_frame_shape": [160, 160],
        "current_frame_shape": [160, 160],
        "captured_at_ns_previous": 0,
        "captured_at_ns_current": 50_000_000,
        "target_id": "target-1",
        "track_epoch": 0,
        "previous_bbox": [50, 45, 90, 105],
        "current_bbox": [50, 45, 90, 105],
        "previous_dynamic_bboxes": [],
        "current_dynamic_bboxes": [],
    }
    row.update(overrides)
    return row


def _texture() -> np.ndarray:
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)
    image[45:105, 50:90] = 128
    return image


class ProduceTests(unittest.TestCase):
    def test_known_zero_motion_is_deterministic(self):
        image = _texture()
        row = _row()
        first = process_pair(row, image, image.copy(), "R2")
        second = process_pair(row, image, image.copy(), "R2")
        self.assertEqual(first, second)
        self.assertIn(first["abstention_reason"], {None, "RING_EMPTY_OR_LOW_AREA", "FEATURE_COUNT_LOW", "LK_TRACK_COUNT_LOW"})
        if first["quality"] == "PASS":
            self.assertAlmostEqual(first["residual_rate_per_s"], 0.0, delta=0.25)
            self.assertIsNone(first["abstention_reason"])

    def test_shape_mismatch_precedes_geometry(self):
        row = _row(previous_frame_shape=[159, 160])
        result = process_pair(row, np.zeros((160, 160), dtype=np.uint8), np.zeros((160, 160), dtype=np.uint8), "R1")
        self.assertEqual(result["quality"], "ABSTAIN")
        self.assertEqual(result["abstention_reason"], "IMAGE_SHAPE_MISMATCH")

    def test_timestamp_priority(self):
        row = _row(captured_at_ns_current=0)
        result = process_pair(row, None, None, "R1")
        self.assertEqual(result["abstention_reason"], "INPUT_TIMESTAMP_INVALID")

    def test_forbidden_truth_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            _reject_forbidden_inputs([_row(truth_state="approach")])

    def test_abstention_reason_is_contract_enum(self):
        image = _texture()
        result = process_pair(_row(current_frame_index=3), image, image, "R4")
        self.assertIn(result["abstention_reason"], set(ABSTENTION_REASONS) | {None})

    def test_fixture_images_use_native_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            cv2.imwrite(str(path), _texture())
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(tuple(decoded.shape), (160, 160))

    def test_ransac_similarity_is_deterministic_and_recovers_scale(self):
        source = np.asarray([[10.0, 10.0], [40.0, 10.0], [10.0, 40.0], [40.0, 40.0], [25.0, 25.0]])
        matrix = 1.02 * np.asarray([[0.999, -0.045], [0.045, 0.999]])
        target = source @ matrix.T + np.asarray([2.0, -1.0])
        first = ransac_similarity(source, target)
        second = ransac_similarity(source, target)
        self.assertTrue(np.allclose(first[0].matrix, second[0].matrix))
        self.assertTrue(np.allclose(first[0].translation, second[0].translation))
        self.assertAlmostEqual(first[0].scale, 1.02, places=2)
        self.assertEqual(first[1].tolist(), second[1].tolist())

    def test_dynamic_mask_removes_only_masked_region(self):
        shape = (160, 160)
        mask, area = ring_mask(shape, [50, 45, 90, 105], [{"bbox": [5, 5, 30, 30], "dynamic": True}], "R1")
        self.assertGreater(area, 0)
        self.assertEqual(int(mask[10, 10]), 0)
        self.assertEqual(int(mask[30, 80]), 255)

    def test_gate_priority_matrix(self):
        image = _texture()
        cases = [
            ({"track_reset": True}, "TRACK_ID_MISMATCH"),
            ({"previous_bbox": [0, 45, 90, 105]}, "BOX_BOUNDARY_TRUNCATED"),
            ({"previous_dynamic_bboxes": [{"bbox": [1, 1, 2, 2], "dynamic": False}]}, "DYNAMIC_MASK_INVALID"),
        ]
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                result = process_pair(_row(**overrides), image, image, "R1")
                self.assertEqual(result["abstention_reason"], expected)


if __name__ == "__main__":
    unittest.main()
