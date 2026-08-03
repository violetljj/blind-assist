from __future__ import annotations

import argparse
import unittest

from materialize_review_bundle import (
    _assert_model_blind,
    _sample_egowalk_times,
    _sample_times,
    _select_candidates,
)
from pipeline import ContractError


class MaterializeReviewBundleTest(unittest.TestCase):
    def test_review_input_rejects_discovery_and_label_fields(self) -> None:
        with self.assertRaises(ContractError):
            _assert_model_blind({"candidate_id": "x", "model_hint": "front_obstacle"})
        with self.assertRaises(ContractError):
            _assert_model_blind({"candidate_id": "x", "event_bucket": "NORMAL_WALKABLE_NEGATIVE"})
        _assert_model_blind({"candidate_id": "x", "model_output_visible": False})


    def test_sample_times_are_monotone_and_clamped_to_video_duration(self) -> None:
        candidate = {
            "start_timestamp_ns": 10_000_000_000,
            "end_timestamp_ns": 14_000_000_000,
        }
        values = _sample_times(candidate, 10_000_000_000, 14.2)
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0.0)
        self.assertLessEqual(values[-1], 14.18)

    def test_egowalk_uses_pose_row_ordinals_not_container_timestamps(self) -> None:
        candidate = {
            "candidate_id": "c",
            "start_frame_index": 100,
            "end_frame_index": 119,
        }
        times, indices = _sample_egowalk_times(
            candidate,
            video_rate_hz=100.0,
            video_frame_count=530,
            pose_rate_hz=5.0,
        )
        self.assertEqual(indices, [96, 100, 110, 119, 123])
        self.assertEqual(times, [0.96, 1.0, 1.1, 1.19, 1.23])

    def test_egowalk_ordinal_samples_clamp_at_video_edges(self) -> None:
        candidate = {
            "candidate_id": "c",
            "start_frame_index": 0,
            "end_frame_index": 19,
        }
        _, indices = _sample_egowalk_times(
            candidate,
            video_rate_hz=100.0,
            video_frame_count=20,
            pose_rate_hz=5.0,
        )
        self.assertEqual(indices, [0, 10, 19])


    def test_candidate_selection_is_deterministic_and_explicit(self) -> None:
        rows = [
            {"candidate_id": "a", "dataset_id": "EgoWalk"},
            {"candidate_id": "b", "dataset_id": "EgoWalk"},
            {"candidate_id": "c", "dataset_id": "Other"},
        ]
        args = argparse.Namespace(candidate_id=[], dataset_id="EgoWalk", offset=1, count=1)
        self.assertEqual([row["candidate_id"] for row in _select_candidates(rows, args)], ["b"])
        args = argparse.Namespace(candidate_id=["c,a"], dataset_id="EgoWalk", offset=0, count=None)
        self.assertEqual([row["candidate_id"] for row in _select_candidates(rows, args)], ["c", "a"])


if __name__ == "__main__":
    unittest.main()
