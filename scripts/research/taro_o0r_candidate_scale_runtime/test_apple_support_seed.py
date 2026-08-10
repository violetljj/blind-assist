#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, apple_support_seed, source_factor
from scripts.research.taro_o0r_candidate_scale_runtime.test_source_factor import _apple_receipt, _candidate_binding
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import eval_geometry, source_receipt, synthetic_faro_depth


class AppleSupportSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = source_receipt()
        cls.geometry = eval_geometry()
        faro_m = synthetic_faro_depth(True).astype(np.float64) / 1000.0
        cls.raw_candidate = np.ascontiguousarray(faro_m * 1.2, dtype=np.float32)
        cls.apple = adapter.sample_faro_at_apple_centers(synthetic_faro_depth(True)).astype(np.uint16)
        cls.confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
        cls.apple_receipt = _apple_receipt(cls.source, cls.apple, cls.confidence)
        cls.binding = _candidate_binding(cls.source, cls.raw_candidate)
        cls.scale_record = apple_scale.build_source_scale_record(cls.apple, cls.confidence, cls.raw_candidate, cls.apple_receipt, cls.binding)
        cls.prepared = source_factor.prepare_source_anchored_candidate(
            cls.raw_candidate,
            cls.apple,
            cls.confidence,
            cls.source,
            cls.apple_receipt,
            cls.binding,
            cls.scale_record,
        )
        cls.plane = apple_support_seed.derive_apple_seeded_candidate_plane(
            cls.prepared,
            cls.apple,
            cls.confidence,
            cls.source,
            cls.scale_record,
        )
        cls.query = adapter.build_query_receipts(cls.source, cls.geometry)[4]
        cls.base = source_factor.build_query_truth_base(cls.geometry, cls.query)
        truth_point = source_factor._point_clearance(
            cls.base.truth_normal_camera_xyz,
            cls.base.truth_camera_height_m,
            cls.base.truth_boundary_points_camera_xyz,
            cls.base.truth_query_support_points,
            cls.base.truth_observed_forward_m,
            cls.base.local_valid_fraction,
            cls.query,
        )
        ordinary = source_factor.evaluate_source_anchored_query(
            cls.prepared,
            np.asarray(cls.source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            cls.source["gravity_up_camera_xyz"],
            cls.base,
            current_faro_geometry_sha256=cls.geometry.content_sha256,
            compact_truth_record_sha256="4" * 64,
            committed_faro_geometry_sha256=cls.geometry.content_sha256,
            committed_factor_frame_sha256="5" * 64,
            committed_base_geometry_sha256="6" * 64,
            compact_query_result={"physical_frame_id": cls.source["physical_frame_id"], "query_id": cls.query["query_id"], "value_m": truth_point["value_m"]},
        )
        lost = copy.deepcopy(ordinary)
        lost.pop("content_sha256")
        lost["source_anchored"] = source_factor._failed_mode(cls.prepared.anchored_depth_sha256, "SUPPORT_SLOPE_EXCEEDED")
        lost["effects"]["extraction_recovered"] = False
        lost["effects"]["extraction_lost"] = True
        for name in (
            "support_normal_error_reduction_rad", "support_height_error_reduction_m", "boundary_jaccard_increase",
            "boundary_xyz_error_reduction_m", "query_point_error_reduction_m",
        ):
            lost["effects"][name] = None
        cls.r1_lost = source_factor._seal(lost)

    def test_phase_a_is_source_only_and_refits_candidate(self) -> None:
        record = apple_support_seed.validate_apple_seeded_candidate_plane_record(self.plane.record)
        self.assertFalse(record["faro_payload_read"])
        self.assertFalse(record["query_receipt_read"])
        self.assertTrue(record["computed_before_truth_join"])
        self.assertEqual(record["r0_mask"]["selected_pixel_ids_sha256"], self.scale_record["selected_pixel_ids_sha256"])
        self.assertEqual(record["candidate_refit"]["point_stride"], adapter.SUPPORT_POINT_STRIDE)
        self.assertAlmostEqual(self.plane.camera_height_m, self.geometry.camera_height_m, places=2)

    def test_phase_a_fails_closed_on_scale_lineage_drift(self) -> None:
        drifted = copy.deepcopy(self.scale_record)
        drifted.pop("content_sha256")
        drifted["candidate_binding_sha256"] = "F" * 64
        drifted = apple_scale._seal(drifted)
        with self.assertRaises(apple_support_seed.AppleSupportSeedError) as caught:
            apple_support_seed.derive_apple_seeded_candidate_plane(
                self.prepared,
                self.apple,
                self.confidence,
                self.source,
                drifted,
            )
        self.assertEqual(caught.exception.code, "APPLE_SUPPORT_SCALE_BINDING_DRIFT")

    def test_phase_b_scores_sealed_plane_without_changing_source_decision(self) -> None:
        record = apple_support_seed.evaluate_recovery_query(
            self.prepared,
            self.source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            self.plane,
            self.r1_lost,
        )
        self.assertTrue(record["source_frame_support_recovered"])
        self.assertTrue(record["posthoc_query_comparison_evaluable"])
        self.assertTrue(record["support_no_regret_vs_r1_baseline"])
        self.assertFalse(record["faro_used_for_source_recovery_decision"])
        self.assertAlmostEqual(record["posthoc_query_comparison"]["support"]["height_abs_error_m"], 0.0, places=2)

    def test_source_failure_remains_unknown(self) -> None:
        record = apple_support_seed.evaluate_recovery_query(
            self.prepared,
            self.source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            None,
            self.r1_lost,
            source_failure_code="APPLE_SEEDED_SUPPORT_POINTS_INSUFFICIENT",
        )
        self.assertFalse(record["source_frame_support_recovered"])
        self.assertFalse(record["posthoc_query_comparison_evaluable"])
        self.assertEqual(record["posthoc_query_comparison"]["reason_codes"], ["APPLE_SEEDED_SUPPORT_POINTS_INSUFFICIENT"])
        self.assertIsNone(record["apple_seeded_candidate_plane_sha256"])

    def test_summary_is_parent_first_round_trip_stable(self) -> None:
        record = apple_support_seed.evaluate_recovery_query(
            self.prepared,
            self.source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            self.plane,
            self.r1_lost,
        )
        summary = apple_support_seed.summarize_recovery([record], [], expected_query_count=1, expected_frame_count=1)
        stored = json.loads(adapter.canonical_json_bytes(record).decode("utf-8"))
        rebuilt = apple_support_seed.summarize_recovery([stored], [], expected_query_count=1, expected_frame_count=1)
        self.assertEqual(summary["content_sha256"], rebuilt["content_sha256"])
        self.assertFalse(summary["threshold_or_pass_fail_decision_applied"])


if __name__ == "__main__":
    unittest.main()
