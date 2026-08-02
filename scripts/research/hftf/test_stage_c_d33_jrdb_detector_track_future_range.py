#!/usr/bin/env python3
"""Tests for D33 tiled detection and future-range evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_stage_c_d33_jrdb_detector_track_future_range import (
    MINIMUM_CURRENT_MATCHES,
    MINIMUM_DIRECTION_EVIDENCE,
    MINIMUM_DISTINCT_IDENTITIES,
    MINIMUM_OPPORTUNITIES,
    MINIMUM_SEQUENCES_WITH_EVIDENCE,
    MINIMUM_SOURCE_FRAMES,
    MINIMUM_TOTAL_EVIDENCE,
    associate,
    determine_terminal,
    iou_matrix,
)
from produce_stage_c_d33_jrdb_detector_tracks import nms


class D33DetectorTrackFutureRangeTests(unittest.TestCase):
    def test_global_nms_removes_overlap_duplicate(self) -> None:
        boxes = np.asarray(
            [
                [100, 100, 200, 300, 0.9],
                [105, 105, 198, 298, 0.8],
                [400, 100, 450, 200, 0.7],
            ],
            dtype=np.float32,
        )
        kept = nms(boxes, 0.5)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(float(kept[0, 4]), 0.9)

    def test_iou_and_hungarian_association(self) -> None:
        source = [
            {"bbox_xyxy": [0, 0, 10, 10]},
            {"bbox_xyxy": [20, 0, 30, 10]},
        ]
        truth = [
            {"bbox_xyxy": [20, 0, 30, 10]},
            {"bbox_xyxy": [0, 0, 10, 10]},
        ]
        matrix = iou_matrix(
            np.asarray([row["bbox_xyxy"] for row in source]),
            np.asarray([row["bbox_xyxy"] for row in truth]),
        )
        self.assertEqual(matrix.tolist(), [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(
            associate(source, truth),
            [(0, 1, 1.0), (1, 0, 1.0)],
        )

    def test_insufficient_source_is_not_evaluable(self) -> None:
        terminal, evaluable, supported = determine_terminal(
            source_frames=MINIMUM_SOURCE_FRAMES - 1,
            current_matches=MINIMUM_CURRENT_MATCHES,
            opportunities=MINIMUM_OPPORTUNITIES,
            evidence_rows=MINIMUM_TOTAL_EVIDENCE,
            distinct_identities=MINIMUM_DISTINCT_IDENTITIES,
            sequences_with_evidence=MINIMUM_SEQUENCES_WITH_EVIDENCE,
            confirm_rows=MINIMUM_DIRECTION_EVIDENCE,
            contradict_rows=MINIMUM_DIRECTION_EVIDENCE,
            effect_gates=[True],
        )
        self.assertFalse(evaluable)
        self.assertFalse(supported)
        self.assertTrue(terminal.endswith("NOT_EVALUABLE"))

    def test_evaluable_effect_failure_is_not_supported(self) -> None:
        terminal, evaluable, supported = determine_terminal(
            source_frames=MINIMUM_SOURCE_FRAMES,
            current_matches=MINIMUM_CURRENT_MATCHES,
            opportunities=MINIMUM_OPPORTUNITIES,
            evidence_rows=MINIMUM_TOTAL_EVIDENCE,
            distinct_identities=MINIMUM_DISTINCT_IDENTITIES,
            sequences_with_evidence=MINIMUM_SEQUENCES_WITH_EVIDENCE,
            confirm_rows=MINIMUM_DIRECTION_EVIDENCE,
            contradict_rows=MINIMUM_DIRECTION_EVIDENCE,
            effect_gates=[True, False],
        )
        self.assertTrue(evaluable)
        self.assertFalse(supported)
        self.assertTrue(terminal.endswith("NOT_SUPPORTED"))


if __name__ == "__main__":
    unittest.main()
