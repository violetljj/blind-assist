#!/usr/bin/env python3
"""Pure unit tests for the public-silver training-readiness gate."""

from __future__ import annotations

import unittest

import audit_public_silver_training_readiness as audit


def row(identifier: str, source: str, verdict: str, pair: str | None = None) -> dict[str, object]:
    return {
        "episode_id": identifier,
        "source_id": source,
        "verdict": verdict,
        "counterfactual_pair_id": pair,
        "evidence_frame_count": 3,
    }


class PublicSilverTrainingReadinessTest(unittest.TestCase):
    def test_current_shape_fails_negative_sources_pairs_and_probe(self) -> None:
        episodes = [row(f"p{i}", f"ps{i}", "candidate_alert") for i in range(5)]
        episodes += [row(f"n{i}", f"ns{i}", "candidate_no_alert") for i in range(3)]
        result = audit.evaluate(
            {"episodes": episodes, "abstentions": [{"episode_id": "a", "source_id": "x"}], "cross_source_duplicate_frames": []},
            minimum_sources_per_class=5,
            minimum_counterfactual_pairs=2,
            probe_report={"linear_separability_gate": {"passed": False}},
        )
        self.assertFalse(result["gates"]["dataset_ready_for_linear_probe"])
        self.assertFalse(result["gates"]["head_short_runs_authorized"])
        self.assertEqual(3, result["independent_source_counts"]["candidate_no_alert"])
        self.assertEqual(1, result["excluded_abstain_count"])
        self.assertIn("insufficient_explicit_matched_counterfactual_pairs", result["gates"]["failure_reasons"])

    def test_ready_shape_requires_explicit_pairs_and_passing_probe(self) -> None:
        episodes = []
        for index in range(5):
            pair = f"pair-{index}" if index < 2 else None
            episodes.append(row(f"p{index}", f"ps{index}", "candidate_alert", pair))
            episodes.append(row(f"n{index}", f"ns{index}", "candidate_no_alert", pair))
        result = audit.evaluate(
            {"episodes": episodes, "abstentions": [], "cross_source_duplicate_frames": []},
            minimum_sources_per_class=5,
            minimum_counterfactual_pairs=2,
            probe_report={"linear_separability_gate": {"passed": True}},
        )
        self.assertTrue(result["gates"]["dataset_ready_for_linear_probe"])
        self.assertTrue(result["gates"]["head_short_runs_authorized"])
        self.assertEqual(2, len(result["matched_counterfactual_pairs"]))

    def test_duplicate_frame_across_sources_fails_closed(self) -> None:
        episodes = [row("p", "ps", "candidate_alert", "pair") , row("n", "ns", "candidate_no_alert", "pair")]
        result = audit.evaluate(
            {
                "episodes": episodes,
                "abstentions": [],
                "cross_source_duplicate_frames": [{"frame_sha256": "abc", "source_ids": ["ps", "ns"]}],
            },
            minimum_sources_per_class=2,
            minimum_counterfactual_pairs=1,
            probe_report={"linear_separability_gate": {"passed": True}},
        )
        self.assertFalse(result["gates"]["cross_source_frame_disjointness_passed"])
        self.assertFalse(result["gates"]["head_short_runs_authorized"])

    def test_thresholds_cannot_be_weakened_to_one_source_and_zero_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two sources"):
            audit.evaluate(
                {"episodes": [], "abstentions": [], "cross_source_duplicate_frames": []},
                minimum_sources_per_class=1,
                minimum_counterfactual_pairs=0,
                probe_report=None,
            )


if __name__ == "__main__":
    unittest.main()
