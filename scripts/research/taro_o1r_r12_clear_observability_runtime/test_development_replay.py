from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import development_replay as replay


def feature(query_id: str, fractions: list[float], anchors: int = 6) -> dict:
    return {
        "query_id": query_id,
        "query_receipt": {"query_id": query_id},
        "r6_state": "UNKNOWN",
        "positive_obstacle_veto": False,
        "occupied_hits": [[[False]]],
        "far_valid_anchor_count": anchors,
        "far_fractions": fractions,
        "observed_support_points": 100,
    }


def frame(parent: str, frame_id: str, features: list[dict], states: list[str]) -> tuple[dict, dict]:
    source = {
        "parent_id": parent,
        "video_id": f"v-{parent}",
        "physical_frame_id": frame_id,
        "query_features": features,
    }
    labels = {
        "parent_id": parent,
        "video_id": f"v-{parent}",
        "physical_frame_id": frame_id,
        "query_labels": [
            {"query_id": item["query_id"], "state": state}
            for item, state in zip(features, states, strict=True)
        ],
    }
    return source, labels


class DevelopmentReplayTest(unittest.TestCase):
    def test_candidate_changes_only_fraction_index(self) -> None:
        replay._validate_single_axis()
        self.assertEqual(replay.BASELINE_RULE["far_fraction_index"], 0)
        self.assertEqual(replay.CANDIDATE_RULE["far_fraction_index"], 2)

    def test_candidate_recovers_clear_frame_that_baseline_misses(self) -> None:
        sources = []
        labels = []
        for index in range(10):
            features = [feature(f"q-{index}-{slot}", [0.1 if index >= 6 else 0.0, 0.0, 0.0]) for slot in range(9)]
            states = ["CLEAR_OBSERVED"] + ["UNKNOWN"] * 8
            source, label = frame(f"p{index}", f"f{index}", features, states)
            sources.append(source)
            labels.append(label)
        result = replay.evaluate(sources, labels)
        self.assertEqual(result["baseline"]["eligible_clear_frame_count"], 6)
        self.assertEqual(result["candidate"]["eligible_clear_frame_count"], 10)
        self.assertEqual(result["clear_frame_recall_gain"], 0.4)
        self.assertTrue(result["development_candidate_passed"])
        self.assertFalse(result["unknown_is_negative"])

    def test_candidate_stops_when_frame_budget_exceeds_two_x(self) -> None:
        sources = []
        labels = []
        for index in range(3):
            features = [feature(f"q-{index}-{slot}", [0.0 if index == 0 else 0.1, 0.0, 0.0]) for slot in range(9)]
            states = ["CLEAR_OBSERVED"] + ["UNKNOWN"] * 8
            source, label = frame(f"p{index}", f"f{index}", features, states)
            sources.append(source)
            labels.append(label)
        result = replay.evaluate(sources, labels)
        self.assertFalse(result["stop_checks"]["eligible_frame_budget"])
        self.assertFalse(result["development_candidate_passed"])


if __name__ == "__main__":
    unittest.main()
