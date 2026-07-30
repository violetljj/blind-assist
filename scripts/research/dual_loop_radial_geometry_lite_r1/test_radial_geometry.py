from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("radial_geometry.py")
SPEC = importlib.util.spec_from_file_location("radial_geometry_r1_test", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def textured(height: int, width: int, scale: float = 1.0) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.uint8)
    center = (width / 2.0, height / 2.0)
    for y in range(max(10, height // 3), min(height - 10, 2 * height // 3), 12):
        for x in range(max(10, width // 3), min(width - 10, 2 * width // 3), 12):
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), 255, -1)
    matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
    return cv2.warpAffine(image, matrix, (width, height))


def observation(
    frame_id: str,
    timestamp_ns: int,
    gray: np.ndarray,
    *,
    box_scale: float = 1.0,
    epoch: str = "track-000:epoch-1",
    reset: bool = False,
) -> object:
    return MODULE.FrameObservation(
        source_frame_id=frame_id,
        captured_at_ns=timestamp_ns,
        target_id="track-000",
        track_epoch=epoch,
        region="CENTER",
        roi_xywh_normalized=(0.5, 0.5, 0.6 * box_scale, 0.6 * box_scale),
        gray=gray,
        history_reset=reset,
    )


class ShapeGuardTest(unittest.TestCase):
    def test_shape_change_common_abstention_precedes_arm_geometry(self) -> None:
        previous = observation("f0", 0, textured(260, 346), reset=True)
        current = observation("f1", 40_000_000, textured(258, 346))
        with (
            mock.patch.object(
                MODULE._R0,
                "bbox_log_area_growth",
                side_effect=AssertionError("bbox core called"),
            ),
            mock.patch.object(
                MODULE._R0,
                "roi_sparse_radial_flow",
                side_effect=AssertionError("flow core called"),
            ),
        ):
            rows = MODULE.evaluate_pair(previous, current)
        self.assertEqual(
            [row["abstention_reason"] for row in rows],
            ["FRAME_SHAPE_CHANGE", "FRAME_SHAPE_CHANGE"],
        )
        self.assertTrue(all(row["signed_approach_rate_per_s"] is None for row in rows))
        self.assertTrue(all(row["quality"]["score"] == 0.0 for row in rows))
        self.assertEqual(
            rows[0]["quality"]["components"]["previous_frame_shape_hw"],
            [260, 346],
        )
        self.assertEqual(
            rows[0]["quality"]["components"]["current_frame_shape_hw"],
            [258, 346],
        )

    def test_current_frame_is_next_pair_only_history(self) -> None:
        previous = observation("f0", 0, textured(260, 346), reset=True)
        changed = observation("f1", 40_000_000, textured(258, 346))
        resumed = observation("f2", 80_000_000, textured(258, 346))
        self.assertEqual(
            {row["abstention_reason"] for row in MODULE.evaluate_pair(previous, changed)},
            {"FRAME_SHAPE_CHANGE"},
        )
        self.assertTrue(
            all(
                row["abstention_reason"] is None
                for row in MODULE.evaluate_pair(changed, resumed)
            )
        )

    def test_reason_precedence_history_then_gap_then_shape(self) -> None:
        previous = observation("f0", 0, textured(260, 346), reset=True)
        reset = observation(
            "f1",
            40_000_000,
            textured(258, 346),
            epoch="track-000:epoch-2",
            reset=True,
        )
        self.assertEqual(
            {row["abstention_reason"] for row in MODULE.evaluate_pair(previous, reset)},
            {"INSUFFICIENT_HISTORY"},
        )
        gap = observation("f1", 150_000_000, textured(258, 346))
        self.assertEqual(
            {row["abstention_reason"] for row in MODULE.evaluate_pair(previous, gap)},
            {"HISTORY_GAP"},
        )

    def test_same_shape_is_numerically_identical_to_r0_core(self) -> None:
        previous = observation("f0", 0, textured(220, 220), reset=True)
        current = observation("f1", 50_000_000, textured(220, 220, 1.05), box_scale=1.05)
        r1_rows = MODULE.evaluate_pair(previous, current)
        r0_rows = MODULE._R0.evaluate_pair(previous, current)
        for r1, r0 in zip(r1_rows, r0_rows, strict=True):
            self.assertEqual(
                r1["signed_approach_rate_per_s"],
                r0["signed_approach_rate_per_s"],
            )
            self.assertEqual(r1["quality"], r0["quality"])
            self.assertEqual(r1["abstention_reason"], r0["abstention_reason"])
            self.assertEqual(r1["protocol_id"], MODULE.PROTOCOL_ID)
            self.assertEqual(r1["implementation_id"], MODULE.IMPLEMENTATION_ID)
            self.assertEqual(r1["parameter_sha256"], MODULE.PARAMETER_SHA256)


if __name__ == "__main__":
    unittest.main()
