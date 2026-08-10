#!/usr/bin/env python3
"""Focused synthetic tests for the post-hoc descriptive factor canary."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.factor_canary import (
    FactorCanaryError,
    build_factor_canary_record,
    summarize_factor_canary,
    validate_factor_canary_record,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import (
    eval_geometry,
    fitted_model,
    manual_reseal,
    source_receipt,
    synthetic_faro_depth,
)


class FactorCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = source_receipt()
        cls.geometry = eval_geometry()
        cls.query = adapter.build_query_receipts(cls.source, cls.geometry)[4]
        cls.model = fitted_model(1)
        cls.truth = adapter.build_truth_query_factor_frame(
            cls.geometry,
            cls.query,
            cls.model,
            confidence_value=2,
            range_m=1.5,
        )
        cls.faro_depth_mm = synthetic_faro_depth(True)
        cls.candidate_depth_full_m = cls.faro_depth_mm.astype(np.float64) / 1000.0
        cls.candidate_depth_half_m = cls.faro_depth_mm.astype(np.float64) / 2000.0
        cls.candidate_full = cls._candidate_frame(cls.candidate_depth_full_m, "B" * 64)
        cls.candidate_half = cls._candidate_frame(cls.candidate_depth_half_m, "C" * 64)

    @classmethod
    def _candidate_frame(cls, depth_m: np.ndarray, inference_sha256: str) -> dict[str, object]:
        output = adapter.build_candidate_depth_output_receipt(
            depth_m,
            cls.source,
            inference_receipt_sha256=inference_sha256,
        )
        return adapter.build_candidate_query_factor_frame(
            depth_m,
            np.asarray(cls.source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            cls.source["gravity_up_camera_xyz"],
            cls.source,
            cls.query,
            cls.truth["base_geometry"],
            cls.model,
            output,
            confidence_value=2,
            range_m=1.5,
        )

    @classmethod
    def _truth_with_invalid_boundary(cls) -> dict[str, object]:
        frame = copy.deepcopy(cls.truth)
        frame["blocks"]["BOUNDARY"]["validity"]["valid"] = False
        frame["blocks"]["BOUNDARY"]["validity"]["local_valid_fraction"] = 0.0
        frame = manual_reseal(frame)
        adapter.validate_query_factor_frame(frame)
        return frame

    def test_scale_two_reduces_support_height_error_without_threshold_claim(self) -> None:
        record = build_factor_canary_record(
            "PARENT_SCALE_TWO",
            self.truth,
            self.candidate_half,
            self.faro_depth_mm,
            self.candidate_depth_half_m,
        )
        validated = validate_factor_canary_record(record)
        scale = validated["factors"]["SCALE"]
        support = validated["factors"]["SUPPORT"]
        self.assertTrue(scale["evaluable"])
        self.assertAlmostEqual(scale["metric_scale"], 2.0, places=12)
        self.assertAlmostEqual(scale["abs_log_correction"], math.log(2.0), places=12)
        self.assertTrue(support["evaluable"])
        self.assertLessEqual(
            support["scale_corrected_height_abs_error_m"],
            support["raw_height_abs_error_m"],
        )
        self.assertFalse(validated["claim_ceiling"]["threshold_or_pass_fail_decision"])

    def test_invalid_truth_boundary_stays_unknown_without_erasing_scale_support(self) -> None:
        record = build_factor_canary_record(
            "PARENT_BOUNDARY_UNKNOWN",
            self._truth_with_invalid_boundary(),
            self.candidate_full,
            self.faro_depth_mm,
            self.candidate_depth_full_m,
        )
        self.assertTrue(record["factors"]["SCALE"]["evaluable"])
        self.assertTrue(record["factors"]["SUPPORT"]["evaluable"])
        self.assertFalse(record["factors"]["BOUNDARY"]["evaluable"])
        self.assertEqual(record["factors"]["BOUNDARY"]["reason_codes"], ["TRUTH_BOUNDARY_INVALID"])
        self.assertIsNone(record["factors"]["BOUNDARY"]["point_id_jaccard"])

    def test_summary_is_parent_macro_and_reports_no_data_as_none(self) -> None:
        valid = build_factor_canary_record(
            "PARENT_VALID",
            self.truth,
            self.candidate_full,
            self.faro_depth_mm,
            self.candidate_depth_full_m,
        )
        invalid = build_factor_canary_record(
            "PARENT_INVALID",
            self._truth_with_invalid_boundary(),
            self.candidate_full,
            self.faro_depth_mm,
            self.candidate_depth_full_m,
        )
        summary = summarize_factor_canary([valid, invalid])
        self.assertEqual(summary["parent_count"], 2)
        self.assertEqual(
            summary["factors"]["BOUNDARY"]["query_coverage_parent_macro"]["median_across_parents"],
            0.5,
        )
        boundary_metric = summary["factors"]["BOUNDARY"]["metrics_parent_macro"]["raw_xyz_median_error_m"]
        self.assertEqual(boundary_metric["parents_with_metric"], 1)
        self.assertIsNotNone(boundary_metric["median_of_parent_medians"])

        invalid_second_parent = build_factor_canary_record(
            "PARENT_INVALID_2",
            self._truth_with_invalid_boundary(),
            self.candidate_full,
            self.faro_depth_mm,
            self.candidate_depth_full_m,
        )
        no_boundary_data = summarize_factor_canary([invalid, invalid_second_parent])
        no_data_metric = no_boundary_data["factors"]["BOUNDARY"]["metrics_parent_macro"]["point_id_jaccard"]
        self.assertIsNone(no_data_metric["median_of_parent_medians"])
        self.assertEqual(no_data_metric["reason_codes"], ["NO_EVALUABLE_PARENT_METRIC"])
        self.assertFalse(no_boundary_data["threshold_or_pass_fail_decision_applied"])

    def test_outer_seal_rejects_tamper_in_validation_and_summary(self) -> None:
        record = build_factor_canary_record(
            "PARENT_TAMPER",
            self.truth,
            self.candidate_half,
            self.faro_depth_mm,
            self.candidate_depth_half_m,
        )
        tampered = copy.deepcopy(record)
        tampered["factors"]["SCALE"]["metric_scale"] = 3.0
        with self.assertRaises(FactorCanaryError) as caught:
            validate_factor_canary_record(tampered)
        self.assertEqual(caught.exception.code, "CANARY_RECORD_SEAL_MISMATCH")
        with self.assertRaises(FactorCanaryError):
            summarize_factor_canary([tampered])

        nested_tamper = copy.deepcopy(record)
        nested_tamper["scale_record"]["metric_scale"] = 3.0
        nested_tamper = manual_reseal(nested_tamper)
        with self.assertRaises(FactorCanaryError) as nested_caught:
            validate_factor_canary_record(nested_tamper)
        self.assertEqual(nested_caught.exception.code, "CANARY_SCALE_RECORD_INVALID")


if __name__ == "__main__":
    unittest.main()
