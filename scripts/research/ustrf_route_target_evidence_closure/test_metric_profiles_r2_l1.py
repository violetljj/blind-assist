#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from metric_profiles_r2_l1 import (
    ProfileContractError,
    _delivery_groups,
    build_terminal_receipt,
    validate_profile_contract,
    wilson_interval,
)


class MetricProfilesR2L1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo / "configs/ustrf_route_target_r2_l1_metric_profile_r1.json"
        )
        cls.receipt, cls.profiles = build_terminal_receipt(
            cls.repo, cls.config_path
        )
        cls.config = __import__("json").loads(
            cls.config_path.read_text(encoding="utf-8")
        )

    def test_real_frozen_trace_inventory_constructs_three_profiles(self) -> None:
        self.assertEqual("METRIC_PROFILES_COMPLETE", self.receipt["terminal_state"])
        self.assertEqual(3, len(self.profiles))
        self.assertEqual(0, self.receipt["verified_scope"]["candidate_reruns"])
        self.assertEqual(123, self.receipt["verified_scope"]["authoritative_traces"])

    def test_fixed_denominators_and_l0_authority_hold(self) -> None:
        for profile in self.profiles.values():
            metrics = profile["metrics"]
            self.assertEqual(8, metrics["critical_miss"]["denominator"])
            self.assertEqual(12, metrics["clearance"]["denominator"])
            self.assertEqual(
                62229, metrics["unknown_or_stale_alert"]["denominator"]
            )
            for name in (
                "event_recall",
                "regeneration",
                "false_alerts_per_minute",
            ):
                self.assertEqual("diagnostic_only", metrics[name]["support_status"])
                self.assertEqual("not_tested", metrics[name]["result_status"])
                self.assertEqual("not_applicable", metrics[name]["gate_result"])

    def test_missing_consume_timestamp_invalidates_whole_age_metric(self) -> None:
        for profile in self.profiles.values():
            metric = profile["metrics"]["evidence_age"]
            self.assertEqual("not_evaluable", metric["support_status"])
            self.assertEqual(0, metric["timestamp_frame_count"])
            self.assertIsNone(metric["denominator"])
            self.assertIsNone(metric["value"])

    def test_frozen_critical_interval_uses_active_attribution(self) -> None:
        for profile in self.profiles.values():
            metric = profile["metrics"]["critical_miss"]
            self.assertEqual(0, metric["numerator"])
            self.assertEqual("estimate_only", metric["result_status"])
            self.assertFalse(metric["bound_sufficient"])

    def test_delivery_mapping_is_per_track_or_single_episode_group(self) -> None:
        one_to_one = {
            "deliveries": [7, 8],
            "delivery_track_ids": [101, 102],
        }
        self.assertEqual([(7, [101]), (8, [102])], _delivery_groups(one_to_one))
        episode = {"deliveries": [3], "delivery_track_ids": [101, 102]}
        self.assertEqual([(3, [101, 102])], _delivery_groups(episode))

    def test_denominator_mutation_fails_closed(self) -> None:
        profile = copy.deepcopy(next(iter(self.profiles.values())))
        profile["metrics"]["clearance"]["denominator"] = 11
        with self.assertRaisesRegex(
            ProfileContractError, "clearance_denominator_drift"
        ):
            validate_profile_contract(profile, self.config)

    def test_wilson_empty_denominator_is_null(self) -> None:
        self.assertIsNone(wilson_interval(0, 0))
        interval = wilson_interval(0, 8)
        self.assertEqual(0.0, interval["lower"])
        self.assertGreater(interval["upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
