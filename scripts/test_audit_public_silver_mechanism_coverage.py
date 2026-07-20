#!/usr/bin/env python3
"""Pure tests for mechanism-stratified public-silver coverage."""

from __future__ import annotations

import unittest

import audit_public_silver_mechanism_coverage as audit


def row(
    pair: str,
    mechanism: str,
    source: str,
    verdict: str,
    confidence: float = 0.9,
) -> dict[str, str | float]:
    return {
        "episode_id": f"{pair}-{verdict}",
        "source_id": source,
        "verdict": verdict,
        "counterfactual_pair_id": pair,
        "mechanism": mechanism,
        "confidence": confidence,
    }


class PublicSilverMechanismCoverageTest(unittest.TestCase):
    def test_two_pairs_each_fail_three_pair_gate(self) -> None:
        rows = []
        for mechanism in audit.DEFAULT_MECHANISMS:
            for index in range(2):
                pair = f"{mechanism}-{index}"
                rows += [
                    row(pair, mechanism, f"{mechanism}-s{index}", "candidate_no_alert"),
                    row(pair, mechanism, f"{mechanism}-s{index}", "candidate_alert"),
                ]
        result = audit.evaluate(
            rows,
            required_mechanisms=audit.DEFAULT_MECHANISMS,
            minimum_pairs_per_mechanism=3,
            minimum_sources_per_mechanism=3,
            minimum_pair_confidence=0.65,
        )
        self.assertFalse(result["mechanism_coverage_gate"]["passed"])
        self.assertEqual(2, result["coverage"]["static_corridor_narrowing"]["matched_pair_count"])

    def test_three_pairs_each_pass(self) -> None:
        rows = []
        for mechanism in audit.DEFAULT_MECHANISMS:
            for index in range(3):
                pair = f"{mechanism}-{index}"
                rows += [
                    row(pair, mechanism, f"{mechanism}-s{index}", "candidate_no_alert"),
                    row(pair, mechanism, f"{mechanism}-s{index}", "candidate_alert"),
                ]
        result = audit.evaluate(
            rows,
            required_mechanisms=audit.DEFAULT_MECHANISMS,
            minimum_pairs_per_mechanism=3,
            minimum_sources_per_mechanism=3,
            minimum_pair_confidence=0.65,
        )
        self.assertTrue(result["mechanism_coverage_gate"]["passed"])

    def test_profile_inference_distinguishes_static_and_dynamic(self) -> None:
        static = {"risk_profile": {"primary_hazard_type": "static_trash_bins_and_standing_sign"}}
        dynamic = {"risk_profile": {"primary_hazard_type": "approaching_pedestrian"}}
        self.assertEqual("static_corridor_narrowing", audit.infer_mechanism(static))
        self.assertEqual("dynamic_agent_approach", audit.infer_mechanism(dynamic))

    def test_conflicting_mechanisms_fail_closed(self) -> None:
        rows = [
            row("pair", "static_corridor_narrowing", "s", "candidate_no_alert"),
            row("pair", "dynamic_agent_approach", "s", "candidate_alert"),
        ]
        result = audit.evaluate(
            rows,
            required_mechanisms=("static_corridor_narrowing",),
            minimum_pairs_per_mechanism=2,
            minimum_sources_per_mechanism=2,
            minimum_pair_confidence=0.65,
        )
        self.assertFalse(result["mechanism_coverage_gate"]["passed"])
        self.assertEqual(1, len(result["mechanism_conflicts"]))

    def test_low_confidence_pair_does_not_close_coverage(self) -> None:
        rows = []
        for index in range(3):
            confidence = 0.63 if index == 2 else 0.9
            rows += [
                row("dynamic-" + str(index), "dynamic_agent_approach", "s" + str(index), "candidate_no_alert", confidence),
                row("dynamic-" + str(index), "dynamic_agent_approach", "s" + str(index), "candidate_alert", confidence),
            ]
        result = audit.evaluate(
            rows,
            required_mechanisms=("dynamic_agent_approach",),
            minimum_pairs_per_mechanism=3,
            minimum_sources_per_mechanism=3,
            minimum_pair_confidence=0.65,
        )
        coverage = result["coverage"]["dynamic_agent_approach"]
        self.assertEqual(3, coverage["all_matched_pair_count"])
        self.assertEqual(2, coverage["matched_pair_count"])
        self.assertEqual(["dynamic-2"], coverage["excluded_low_confidence_pair_ids"])
        self.assertFalse(result["mechanism_coverage_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
