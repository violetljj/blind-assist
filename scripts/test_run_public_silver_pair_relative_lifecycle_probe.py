import copy
import unittest

import run_public_silver_pair_relative_lifecycle_probe as probe


def episode(pair_id: str, episode_id: str, source_id: str, label: int, start: int, end: int) -> dict:
    return {
        "counterfactual_pair_id": pair_id,
        "episode_id": episode_id,
        "source_id": source_id,
        "label": label,
        "frames": [{"frame_index": start}, {"frame_index": end}],
    }


def fixture(include_close: bool = True) -> tuple[list[dict], dict, dict]:
    pairs = {"dynamic": ["open-pair"], "static": ["close-pair"]}
    episodes = [
        episode("open-pair", "open-clear", "source-a", 0, 1, 2),
        episode("open-pair", "open-risk", "source-a", 1, 3, 4),
    ]
    scores = {
        "dynamic": [{
            "counterfactual_pair_id": "open-pair", "source_id": "source-a",
            "no_alert_score": 0.1, "alert_score": 0.4,
        }],
        "static": [],
    }
    if include_close:
        episodes.extend([
            episode("close-pair", "close-risk", "source-b", 1, 10, 11),
            episode("close-pair", "close-clear", "source-b", 0, 12, 13),
        ])
        scores["static"].append({
            "counterfactual_pair_id": "close-pair", "source_id": "source-b",
            "no_alert_score": 0.2, "alert_score": 0.3,
        })
    else:
        pairs = {"dynamic": ["open-pair"]}
        scores.pop("static")
    mechanism = {
        "schema": probe.MECHANISM_SCHEMA,
        "required_mechanisms": list(pairs),
        "coverage": {key: {"counterfactual_pair_ids": value} for key, value in pairs.items()},
        "mechanism_coverage_gate": {"passed": True},
        "isolation_contract": {"independent_model_direction_data_used": False},
    }
    temporal = {
        "schema": probe.TEMPORAL_SCHEMA,
        "qualified_pair_contract": pairs,
        "pair_scores": scores,
        "isolation_contract": {
            "independent_model_direction_data_used": False,
            "independent_model_direction_code_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
    }
    return episodes, mechanism, temporal


def evaluate(value: tuple[list[dict], dict, dict]) -> dict:
    episodes, mechanism, temporal = value
    return probe.evaluate(episodes=episodes, mechanism_report=mechanism, temporal_report=temporal)


class PairRelativeLifecycleProbeTest(unittest.TestCase):
    def test_open_and_close_transitions_pass_without_absolute_threshold(self) -> None:
        result = evaluate(fixture())
        self.assertTrue(result["acceptance"]["passed"])
        self.assertEqual(result["metrics"]["transition_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["open_event_count"], 1)
        self.assertEqual(result["metrics"]["close_event_count"], 1)
        self.assertFalse(result["score_contract"]["absolute_threshold_used"])
        self.assertEqual(len(result["margin_sensitivity"]), 5)

    def test_low_margin_transition_is_visible_in_stress_diagnostic(self) -> None:
        value = fixture()
        value[2]["pair_scores"]["static"][0]["no_alert_score"] = 0.29
        value[2]["pair_scores"]["static"][0]["alert_score"] = 0.30
        result = evaluate(value)
        self.assertTrue(result["acceptance"]["passed"])
        self.assertFalse(result["margin_sensitivity"][3]["all_qualified_pairs_retained"])

    def test_wrong_score_direction_fails_gate(self) -> None:
        value = fixture()
        value[2]["pair_scores"]["dynamic"][0]["alert_score"] = 0.05
        result = evaluate(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertEqual(result["transitions"][0]["predicted_transition"], probe.CLOSE_EVENT)

    def test_zero_delta_abstains_and_fails_gate(self) -> None:
        value = fixture()
        value[2]["pair_scores"]["dynamic"][0]["alert_score"] = 0.1
        result = evaluate(value)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertEqual(result["transitions"][0]["predicted_transition"], probe.ABSTAIN)

    def test_incomplete_pair_is_rejected(self) -> None:
        value = fixture()
        value[0].pop()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            evaluate(value)

    def test_cross_source_pair_is_rejected(self) -> None:
        value = fixture()
        value[0][1]["source_id"] = "other-source"
        with self.assertRaisesRegex(ValueError, "within one source"):
            evaluate(value)

    def test_overlapping_pair_chronology_is_rejected(self) -> None:
        value = fixture()
        value[0][1]["frames"] = [{"frame_index": 2}, {"frame_index": 3}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(value)

    def test_temporal_contract_mismatch_is_rejected(self) -> None:
        value = fixture()
        value[2]["qualified_pair_contract"]["dynamic"] = ["other"]
        with self.assertRaisesRegex(ValueError, "contract differs"):
            evaluate(value)

    def test_no_close_transition_cannot_pass_full_gate(self) -> None:
        result = evaluate(fixture(include_close=False))
        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["acceptance"]["close_event_coverage_present"])


if __name__ == "__main__":
    unittest.main()
