#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_full_cohort as r4
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_hybrid as hybrid
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_candidate_scale_runtime import test_direct_apple_full_cohort as fixture_module
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


class DirectAppleHybridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_module.DirectAppleFullCohortTest.setUpClass()
        cls.r4_record = fixture_module.DirectAppleFullCohortTest.record

    @staticmethod
    def _rebuild_r4(record: dict[str, object]) -> dict[str, object]:
        payload = copy.deepcopy(record)
        payload.pop("content_sha256")
        payload["effects"] = r4._canonical(
            r4._effects(
                payload["comparators"]["r1_baseline"],
                payload["comparators"]["r1_source_anchored"],
                payload["direct_apple_support"],
            )
        )
        return r4._seal(payload)

    def test_direct_is_selected_when_source_and_extraction_are_available(self) -> None:
        record = hybrid.build_hybrid_query_record(self.r4_record)
        self.assertEqual(record["selected_mode"], "DIRECT_APPLE_SUPPORT")
        self.assertEqual(record["selected_result"], self.r4_record["direct_apple_support"])
        self.assertEqual(record["selection_metric_fields_read"], [])
        self.assertEqual(record["free_parameter_count"], 0)

    def test_baseline_is_selected_when_direct_is_unavailable(self) -> None:
        unavailable = copy.deepcopy(self.r4_record)
        unavailable["source_support_available"] = False
        unavailable["full_cohort_plane_sha256"] = None
        unavailable["direct_apple_support"] = source_factor._failed_mode(
            unavailable["direct_apple_support"]["depth_array_sha256"],
            "SYNTHETIC_DIRECT_UNAVAILABLE",
        )
        unavailable = self._rebuild_r4(unavailable)

        record = hybrid.build_hybrid_query_record(unavailable)

        self.assertEqual(record["selected_mode"], "R1_BASELINE_FALLBACK")
        self.assertEqual(record["selected_result"], unavailable["comparators"]["r1_baseline"])

    def test_truth_metric_regression_cannot_change_selection(self) -> None:
        regressed = copy.deepcopy(self.r4_record)
        regressed["direct_apple_support"]["support"]["height_abs_error_m"] = 99.0
        regressed["direct_apple_support"]["support"]["normal_angular_error_rad"] = 3.0
        regressed = self._rebuild_r4(regressed)

        record = hybrid.build_hybrid_query_record(regressed)

        self.assertEqual(record["selected_mode"], "DIRECT_APPLE_SUPPORT")
        self.assertFalse(record["effects"]["support_no_regret_vs_baseline"])

    def test_evaluation_extraction_failure_cannot_change_selection(self) -> None:
        failed = copy.deepcopy(self.r4_record)
        failed["direct_apple_support"] = source_factor._failed_mode(
            failed["direct_apple_support"]["depth_array_sha256"],
            "SYNTHETIC_EVALUATION_EXTRACTION_FAILURE",
        )
        failed = self._rebuild_r4(failed)

        record = hybrid.build_hybrid_query_record(failed)

        self.assertEqual(record["selection_inputs"], {"source_support_available": True})
        self.assertEqual(record["selected_mode"], "DIRECT_APPLE_SUPPORT")
        self.assertFalse(record["selected_result"]["extraction_evaluable"])

    def test_selection_tamper_is_rejected(self) -> None:
        record = hybrid.build_hybrid_query_record(self.r4_record)
        tampered = copy.deepcopy(record)
        tampered.pop("content_sha256")
        tampered["selected_mode"] = "R1_BASELINE_FALLBACK"
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            hybrid.validate_hybrid_query_record(hybrid._seal(tampered), self.r4_record)
        self.assertEqual(caught.exception.code, "R4A_SELECTION_DRIFT")

    def test_external_r4_binding_rejects_resealed_selected_payload(self) -> None:
        record = hybrid.build_hybrid_query_record(self.r4_record)
        tampered = copy.deepcopy(record)
        tampered.pop("content_sha256")
        tampered["selected_result"]["support"]["height_abs_error_m"] = 77.0
        tampered["selected_result_sha256"] = adapter.canonical_sha256(tampered["selected_result"])
        tampered["effects"] = hybrid._canonical(
            hybrid._effects(tampered["baseline"], tampered["selected_result"])
        )
        tampered = hybrid._seal(tampered)
        with self.assertRaises(r3.DirectAppleSupportError) as caught:
            hybrid.validate_hybrid_query_record(tampered, self.r4_record)
        self.assertEqual(caught.exception.code, "R4A_R4_BINDING_DRIFT")

    def test_summary_round_trip_is_stable(self) -> None:
        record = hybrid.build_hybrid_query_record(self.r4_record)
        summary = hybrid.summarize_hybrid(
            [record],
            [self.r4_record],
            expected_query_count=1,
            expected_frame_count=1,
            expected_parent_count=1,
        )
        stored = json.loads(adapter.canonical_json_bytes(record).decode("utf-8"))
        rebuilt = hybrid.summarize_hybrid(
            [stored],
            [self.r4_record],
            expected_query_count=1,
            expected_frame_count=1,
            expected_parent_count=1,
        )
        self.assertEqual(summary, rebuilt)
        self.assertFalse(summary["training_applied"])


if __name__ == "__main__":
    unittest.main()
