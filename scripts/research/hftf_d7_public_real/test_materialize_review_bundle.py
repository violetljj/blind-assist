from __future__ import annotations

import argparse
import unittest

from materialize_review_bundle import _assert_model_blind, _sample_times, _select_candidates
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
