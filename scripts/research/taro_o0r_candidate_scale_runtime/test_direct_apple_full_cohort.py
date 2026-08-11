#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_full_cohort as r4
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_candidate_scale_runtime import test_direct_apple_support as fixture_module
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


class DirectAppleFullCohortTest(unittest.TestCase):
    @staticmethod
    def _different_hash(value: str) -> str:
        return ("A" if value != "A" * 64 else "B") * 64

    @classmethod
    def setUpClass(cls) -> None:
        fixture_module.DirectAppleSupportTest.setUpClass()
        fixture = fixture_module.DirectAppleSupportTest
        for name in (
            "source", "geometry", "parent_id", "raw_candidate", "apple", "confidence",
            "apple_receipt", "binding", "scale_record", "candidate_input", "direct_source",
            "prepared", "query", "base", "r1_lost",
        ):
            setattr(cls, name, getattr(fixture, name))
        cls.plane = r4.derive_full_cohort_plane(
            cls.prepared,
            cls.apple,
            cls.confidence,
            cls.direct_source,
            cls.scale_record,
        )
        cls.record = r4.evaluate_full_cohort_query(
            cls.prepared,
            cls.direct_source["intrinsics_highres"]["matrix_3x3"],
            cls.direct_source["gravity_up_camera_xyz"],
            cls.base,
            cls.plane,
            cls.r1_lost,
            current_faro_geometry_sha256=cls.r1_lost["current_faro_geometry_sha256"],
            compact_faro_geometry_sha256=cls.r1_lost["committed_faro_geometry_sha256"],
        )

    def test_plane_uses_fixed_r3_math_under_full_cohort_claim(self) -> None:
        record = r4.validate_full_cohort_plane_record(self.plane.record)
        self.assertEqual(record["claim_ceiling"], r4.CLAIM_CEILING)
        self.assertEqual(record["fixed_r3_method_id"], r3.METHOD_ID)
        self.assertFalse(record["cohort_selection_used_for_plane"])
        self.assertFalse(record["apple_support_mask"]["candidate_depth_used"])

    def test_query_compares_direct_support_to_both_r1_modes(self) -> None:
        record = r4.validate_full_cohort_query_record(self.record)
        self.assertEqual(set(record["comparators"]), {"r1_baseline", "r1_source_anchored"})
        self.assertTrue(record["source_support_available"])
        self.assertTrue(record["direct_apple_support"]["extraction_evaluable"])
        self.assertFalse(record["threshold_or_pass_fail_decision_applied"])

    def test_effect_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.record)
        tampered.pop("content_sha256")
        tampered["effects"]["support_no_regret_vs_baseline"] = not tampered["effects"]["support_no_regret_vs_baseline"]
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            r4.validate_full_cohort_query_record(r4._seal(tampered))
        self.assertEqual(caught.exception.code, "R4_EFFECT_DRIFT")

    def test_r1_current_geometry_drift_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.r1_lost)
        drifted.pop("content_sha256")
        drifted["current_faro_geometry_sha256"] = "A" * 64
        drifted["runtime_geometry_matches_r3_commitment"] = False
        drifted = source_factor._seal(drifted)
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            r4.evaluate_full_cohort_query(
                self.prepared,
                self.direct_source["intrinsics_highres"]["matrix_3x3"],
                self.direct_source["gravity_up_camera_xyz"],
                self.base,
                self.plane,
                drifted,
                current_faro_geometry_sha256=self.r1_lost["current_faro_geometry_sha256"],
                compact_faro_geometry_sha256=self.r1_lost["committed_faro_geometry_sha256"],
            )
        self.assertEqual(caught.exception.code, "R4_R1_GEOMETRY_DRIFT")

    def test_r1_committed_geometry_drift_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.r1_lost)
        drifted.pop("content_sha256")
        drifted["committed_faro_geometry_sha256"] = self._different_hash(
            drifted["committed_faro_geometry_sha256"]
        )
        drifted["runtime_geometry_matches_r3_commitment"] = False
        drifted = source_factor._seal(drifted)
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            r4.evaluate_full_cohort_query(
                self.prepared,
                self.direct_source["intrinsics_highres"]["matrix_3x3"],
                self.direct_source["gravity_up_camera_xyz"],
                self.base,
                self.plane,
                drifted,
                current_faro_geometry_sha256=self.r1_lost["current_faro_geometry_sha256"],
                compact_faro_geometry_sha256=self.r1_lost["committed_faro_geometry_sha256"],
            )
        self.assertEqual(caught.exception.code, "R4_R1_GEOMETRY_DRIFT")

    def test_r1_geometry_match_flag_drift_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.r1_lost)
        drifted.pop("content_sha256")
        drifted["runtime_geometry_matches_r3_commitment"] = not drifted["runtime_geometry_matches_r3_commitment"]
        drifted = source_factor._seal(drifted)
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            r4.evaluate_full_cohort_query(
                self.prepared,
                self.direct_source["intrinsics_highres"]["matrix_3x3"],
                self.direct_source["gravity_up_camera_xyz"],
                self.base,
                self.plane,
                drifted,
                current_faro_geometry_sha256=self.r1_lost["current_faro_geometry_sha256"],
                compact_faro_geometry_sha256=self.r1_lost["committed_faro_geometry_sha256"],
            )
        self.assertEqual(caught.exception.code, "R4_R1_GEOMETRY_DRIFT")

    def test_current_geometry_may_differ_from_bound_compact_geometry(self) -> None:
        accepted = copy.deepcopy(self.r1_lost)
        accepted.pop("content_sha256")
        compact_geometry = self._different_hash(accepted["current_faro_geometry_sha256"])
        accepted["committed_faro_geometry_sha256"] = compact_geometry
        accepted["runtime_geometry_matches_r3_commitment"] = False
        accepted = source_factor._seal(accepted)

        record = r4.evaluate_full_cohort_query(
            self.prepared,
            self.direct_source["intrinsics_highres"]["matrix_3x3"],
            self.direct_source["gravity_up_camera_xyz"],
            self.base,
            self.plane,
            accepted,
            current_faro_geometry_sha256=accepted["current_faro_geometry_sha256"],
            compact_faro_geometry_sha256=compact_geometry,
        )

        self.assertFalse(record["geometry_binding"]["runtime_geometry_matches_compact_commitment"])

    def test_source_failure_and_summary_preserve_unknown(self) -> None:
        failure = r4.build_source_failure_record(
            self.parent_id,
            self.direct_source,
            self.prepared,
            r3.DirectAppleSupportError("SUPPORT_SLOPE_EXCEEDED", "synthetic source failure"),
        )
        self.assertTrue(r4.validate_source_failure_record(failure)["unknown_preserved"])
        summary = r4.summarize_full_cohort(
            [self.record],
            [],
            expected_query_count=1,
            expected_frame_count=1,
            expected_parent_count=1,
        )
        stored = json.loads(adapter.canonical_json_bytes(self.record).decode("utf-8"))
        rebuilt = r4.summarize_full_cohort(
            [stored],
            [],
            expected_query_count=1,
            expected_frame_count=1,
            expected_parent_count=1,
        )
        self.assertEqual(summary["content_sha256"], rebuilt["content_sha256"])
        self.assertFalse(summary["threshold_or_pass_fail_decision_applied"])


if __name__ == "__main__":
    unittest.main()
