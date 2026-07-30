from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("radial_geometry.py")
SPEC = importlib.util.spec_from_file_location("radial_geometry", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def textured_fixture(scale: float = 1.0, translation: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    image = np.zeros((220, 220), dtype=np.uint8)
    for y in range(70, 151, 16):
        for x in range(70, 151, 16):
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), 255, -1)
    matrix = cv2.getRotationMatrix2D((110.0, 110.0), 0.0, scale)
    matrix[:, 2] += np.asarray(translation)
    return cv2.warpAffine(image, matrix, (220, 220), flags=cv2.INTER_LINEAR)


def observation(
    frame_id: str,
    timestamp_ns: int,
    gray: np.ndarray,
    box_scale: float,
    center_shift: tuple[float, float] = (0.0, 0.0),
    epoch: str = "track-000:epoch-0001",
    reset: bool = False,
) -> object:
    return MODULE.FrameObservation(
        source_frame_id=frame_id,
        captured_at_ns=timestamp_ns,
        target_id="track-000",
        track_epoch=epoch,
        region="CENTER",
        roi_xywh_normalized=(
            (110.0 + center_shift[0]) / 220.0,
            (110.0 + center_shift[1]) / 220.0,
            110.0 * box_scale / 220.0,
            110.0 * box_scale / 220.0,
        ),
        gray=gray,
        history_reset=reset,
    )


class RadialGeometryFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = observation("f0", 0, textured_fixture(), 1.0, reset=True)

    def _scores(self, current: object) -> dict[str, float]:
        rows = MODULE.evaluate_pair(self.previous, current)
        self.assertTrue(all(row["abstention_reason"] is None for row in rows))
        return {row["arm_id"]: row["signed_approach_rate_per_s"] for row in rows}

    def test_static_is_near_zero(self) -> None:
        scores = self._scores(observation("f1", 50_000_000, textured_fixture(), 1.0))
        self.assertAlmostEqual(scores[MODULE.ARM_BBOX], 0.0, places=9)
        self.assertLess(abs(scores[MODULE.ARM_FLOW]), 0.01)

    def test_enlargement_is_positive(self) -> None:
        scores = self._scores(observation("f1", 50_000_000, textured_fixture(1.10), 1.10))
        self.assertGreater(scores[MODULE.ARM_BBOX], 0.0)
        self.assertGreater(scores[MODULE.ARM_FLOW], 0.0)
        self.assertAlmostEqual(scores[MODULE.ARM_BBOX], math.log(1.10) / 0.05, places=8)

    def test_shrinkage_is_negative(self) -> None:
        scores = self._scores(observation("f1", 50_000_000, textured_fixture(0.90), 0.90))
        self.assertLess(scores[MODULE.ARM_BBOX], 0.0)
        self.assertLess(scores[MODULE.ARM_FLOW], 0.0)

    def test_translation_about_shifted_roi_is_near_zero(self) -> None:
        shift = (8.0, -5.0)
        scores = self._scores(observation("f1", 50_000_000, textured_fixture(1.0, shift), 1.0, shift))
        self.assertAlmostEqual(scores[MODULE.ARM_BBOX], 0.0, places=9)
        self.assertLess(abs(scores[MODULE.ARM_FLOW]), 0.03)

    def test_gap_and_epoch_switch_abstain(self) -> None:
        gap = observation("f1", 150_000_000, textured_fixture(), 1.0)
        self.assertEqual({row["abstention_reason"] for row in MODULE.evaluate_pair(self.previous, gap)}, {"HISTORY_GAP"})
        switched = observation(
            "f1",
            50_000_000,
            textured_fixture(),
            1.0,
            epoch="track-000:epoch-0002",
            reset=True,
        )
        self.assertEqual({row["abstention_reason"] for row in MODULE.evaluate_pair(self.previous, switched)}, {"INSUFFICIENT_HISTORY"})

    def test_ttl_is_capture_anchored(self) -> None:
        current = observation("f1", 50_000_000, textured_fixture(), 1.0)
        for row in MODULE.evaluate_pair(self.previous, current):
            self.assertEqual(row["ttl_ns"], 100_000_000)
            self.assertEqual(row["valid_until_ns"], row["captured_at_ns"] + row["ttl_ns"])
            self.assertEqual(row["available_at_ns"], row["captured_at_ns"])

    def test_ttl_expiry_becomes_stale_abstention_without_renewal(self) -> None:
        current = observation("f1", 50_000_000, textured_fixture(), 1.0)
        row = MODULE.evaluate_pair(self.previous, current)[0]
        valid_until = row["valid_until_ns"]
        self.assertIsNone(MODULE.apply_consumer_time(row, valid_until)["abstention_reason"])
        stale = MODULE.apply_consumer_time(row, valid_until + 1)
        self.assertEqual(stale["abstention_reason"], "STALE_RESULT")
        self.assertIsNone(stale["signed_approach_rate_per_s"])
        self.assertEqual(stale["valid_until_ns"], valid_until)

    def test_current_roi_expansion_is_ten_percent_total_dimension(self) -> None:
        roi = (50.0, 50.0, 20.0, 20.0)
        points = np.asarray([[60.9, 50.0], [61.1, 50.0]], dtype=np.float64)
        inside = MODULE._inside_expanded_roi(points, roi, (100, 100), 0.10)
        self.assertEqual(inside.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
