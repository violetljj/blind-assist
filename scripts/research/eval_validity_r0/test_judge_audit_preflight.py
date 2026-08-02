from __future__ import annotations

import unittest

from .judge_audit_preflight import preflight


class JudgeAuditPreflightTest(unittest.TestCase):
    def test_mask_discovery_is_not_promoted_to_truth(self) -> None:
        discovery = {
            "candidates": [
                {"session_id": "s1", "selection_profile": "center_obstacle"},
                {"session_id": "s2", "selection_profile": "strict_normal_walkable_source_mask_only"},
            ]
        }
        result = preflight([("discovery.json", discovery)], {"excluded_source_sessions": []})
        self.assertEqual("HOLD_JUDGE_AUDIT_COHORT", result["status"])
        self.assertEqual(2, result["candidate_session_count"])
        self.assertTrue(all(item["status"] == "NOT_ESTABLISHED" for item in result["coverage"].values()))
        self.assertFalse(result["required_artifacts"]["event_ledger"])
        self.assertFalse(result["formal_review_access"])
        self.assertEqual("NOT_STARTED", result["calibration_pilot"]["status"])
        self.assertFalse(result["calibration_pilot"]["formal_denominator_inclusion"])
        self.assertFalse(result["burned_assets"][0]["benchmark_ready"])

    def test_duplicate_session_is_counted_once(self) -> None:
        first = {"candidates": [{"session_id": "s1", "selection_profile": "center_obstacle"}]}
        second = {"candidates": [{"session_id": "s1", "selection_profile": "strict_normal_walkable_source_mask_only"}, {"session_id": "s2", "selection_profile": "step_curb"}]}
        result = preflight([("a.json", first), ("b.json", second)], {"excluded_source_sessions": []})
        self.assertEqual(2, result["candidate_session_count"])
        self.assertEqual(1, result["duplicate_candidate_records_across_discoveries"])

    def test_discovery_arm_mix_is_reported_without_becoming_reviewer_metadata(self) -> None:
        mask = {"candidates": [{"session_id": "s1", "selection_profile": "center_obstacle"}]}
        random_rgb = {"candidates": [{"session_id": "s2", "selection_profile": "random_continuous_rgb", "discovery_arm": "random_continuous_rgb"}]}
        result = preflight([("mask.json", mask, "source_mask"), ("rgb.json", random_rgb, "random_continuous_rgb")], {"excluded_source_sessions": []})
        self.assertEqual("ESTABLISHED", result["formal_discovery_mix"]["status"])
        self.assertEqual(["random_continuous_rgb", "source_mask"], result["formal_discovery_mix"]["distinct_arms"])
        self.assertEqual(["source_mask"], result["candidate_sessions"][0]["discovery_arms"])

    def test_discovery_arm_conflict_is_rejected(self) -> None:
        discovery = {"candidates": [{"session_id": "s1", "selection_profile": "center_obstacle", "discovery_arm": "motion_temporal_change"}]}
        with self.assertRaisesRegex(ValueError, "conflicts with input arm"):
            preflight([("mask.json", discovery, "source_mask")], {"excluded_source_sessions": []})


if __name__ == "__main__":
    unittest.main()
