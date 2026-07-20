#!/usr/bin/env python3
"""Pure tests for the frozen multi-cone risk-evidence policy."""

from __future__ import annotations

import copy
import unittest

import public_video_multicone_policy as subject


def contract_fixture() -> dict[str, object]:
    return {
        "risk_evidence_policy": {
            "policy_id": subject.POLICY_ID,
            "class_name": subject.CLASS_NAME,
            "minimum_count_per_frame": 2,
            "other_semantic_classes_used": [],
            "geometry_gate_used": False,
        }
    }


class PublicVideoMulticonePolicyTest(unittest.TestCase):
    def test_exact_policy_passes(self) -> None:
        contract = contract_fixture()
        policy = contract["risk_evidence_policy"]
        self.assertIs(policy, subject.validate_policy(contract))

    def test_threshold_drift_fails_closed(self) -> None:
        contract = copy.deepcopy(contract_fixture())
        contract["risk_evidence_policy"]["minimum_count_per_frame"] = 1
        with self.assertRaisesRegex(ValueError, "count threshold"):
            subject.validate_policy(contract)

    def test_apply_policy_requires_two_cones(self) -> None:
        policy = subject.validate_policy(contract_fixture())
        rows = [
            {"timestamp_ms": 0, "semantic_class_counts": {"traffic cone": 1}},
            {"timestamp_ms": 1000, "semantic_class_counts": {"traffic cone": 2}},
        ]
        filtered = subject.apply_policy(rows, policy)
        self.assertEqual({}, filtered[0]["semantic_group_counts"])
        self.assertEqual(
            {"barrier_structure": 1}, filtered[1]["semantic_group_counts"]
        )


if __name__ == "__main__":
    unittest.main()
