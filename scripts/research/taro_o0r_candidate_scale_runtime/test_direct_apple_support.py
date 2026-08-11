#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from unittest import mock

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, direct_apple_support, source_factor
from scripts.research.taro_o0r_candidate_scale_runtime.test_source_factor import _apple_receipt, _candidate_binding
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import build_candidate_input_receipt
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import (
    eval_geometry,
    lowres_intrinsics,
    source_receipt,
    synthetic_faro_depth,
    trajectory_rows_for,
)


def _member(role: str, token: str, digest: str, crc32: str) -> dict[str, object]:
    return {
        "container_id": f"sha256:{'9' * 64}",
        "member_path": f"{role}/{token}.png" if role == "color" else f"{role}/{token}.pincam",
        "bytes": 16,
        "sha256": digest,
        "crc32": crc32,
    }


class DirectAppleSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = source_receipt()
        cls.geometry = eval_geometry()
        cls.parent_id = str(cls.source["parent_id"])
        cls.video_id = str(cls.source["session_id"])
        cls.token = str(cls.source["sensor_timestamp"]["decimal_token"])
        faro_m = synthetic_faro_depth(True).astype(np.float64) / 1000.0
        cls.raw_candidate = np.ascontiguousarray(faro_m * 1.2, dtype=np.float32)
        cls.apple = adapter.sample_faro_at_apple_centers(synthetic_faro_depth(True)).astype(np.uint16)
        cls.confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
        cls.apple_receipt = _apple_receipt(cls.source, cls.apple, cls.confidence)
        cls.binding = _candidate_binding(cls.source, cls.raw_candidate)
        cls.scale_record = apple_scale.build_source_scale_record(cls.apple, cls.confidence, cls.raw_candidate, cls.apple_receipt, cls.binding)

        color = np.zeros((*adapter.HIGHRES_SHAPE_HW, 3), dtype=np.uint8)
        cls.intrinsics_digest = hashlib.sha256(b"synthetic-intrinsics").hexdigest().upper()
        cls.intrinsics_crc32 = "1234ABCD"
        cls.lowres = json.loads(adapter.canonical_json_bytes(lowres_intrinsics()).decode("utf-8"))
        cls.candidate_input = build_candidate_input_receipt(
            visit_id=cls.parent_id,
            video_id=cls.video_id,
            timestamp_token=cls.token,
            color_member_binding=_member("color", cls.token, hashlib.sha256(b"synthetic-color").hexdigest().upper(), "ABCD1234"),
            intrinsics_member_binding=_member("intrinsics", cls.token, cls.intrinsics_digest, cls.intrinsics_crc32),
            color_rgb_u8=color,
            lowres_intrinsics=cls.lowres,
        )
        cls.direct_source = direct_apple_support.build_direct_apple_source_receipt(
            cls.candidate_input,
            cls.apple_receipt,
            cls.apple,
            cls.confidence,
            cls.lowres,
            trajectory_rows_for(cls.token),
            intrinsics_member_sha256=cls.intrinsics_digest,
            intrinsics_member_crc32=cls.intrinsics_crc32,
            trajectory_container_sha256="7" * 64,
            trajectory_payload_sha256="8" * 64,
        )
        cls.prepared = direct_apple_support.prepare_direct_source_candidate(
            cls.raw_candidate,
            cls.apple,
            cls.confidence,
            cls.direct_source,
            cls.candidate_input,
            cls.apple_receipt,
            cls.binding,
            cls.scale_record,
        )
        cls.plane = direct_apple_support.derive_direct_apple_support_plane(
            cls.prepared,
            cls.apple,
            cls.confidence,
            cls.direct_source,
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
            cls.direct_source["intrinsics_highres"]["matrix_3x3"],
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
            "support_normal_error_reduction_rad",
            "support_height_error_reduction_m",
            "boundary_jaccard_increase",
            "boundary_xyz_error_reduction_m",
            "query_point_error_reduction_m",
        ):
            lost["effects"][name] = None
        cls.r1_lost = source_factor._seal(lost)

    def test_narrow_source_receipt_rebuilds_camera_without_truth(self) -> None:
        receipt = direct_apple_support.validate_direct_apple_source_receipt(self.direct_source, self.apple, self.confidence)
        self.assertEqual(receipt["opened_source_roles"], ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"])
        self.assertFalse(receipt["compact_truth_read"])
        self.assertFalse(receipt["faro_payload_read"])
        self.assertEqual(adapter.canonical_sha256(receipt["intrinsics_highres"]), adapter.canonical_sha256(self.candidate_input["intrinsics_highres"]))
        self.assertEqual(adapter.canonical_sha256(receipt["camera_to_world_4x4"]), adapter.canonical_sha256(self.source["camera_to_world_4x4"]))

    def test_raw_intrinsics_must_match_candidate_input(self) -> None:
        with self.assertRaises(direct_apple_support.DirectAppleSupportError) as caught:
            direct_apple_support.build_direct_apple_source_receipt(
                self.candidate_input,
                self.apple_receipt,
                self.apple,
                self.confidence,
                self.lowres,
                trajectory_rows_for(self.token),
                intrinsics_member_sha256="F" * 64,
                intrinsics_member_crc32=self.intrinsics_crc32,
                trajectory_container_sha256="7" * 64,
                trajectory_payload_sha256="8" * 64,
            )
        self.assertEqual(caught.exception.code, "DIRECT_APPLE_INTRINSICS_DRIFT")

    def test_phase_a_uses_apple_support_without_candidate_refit(self) -> None:
        record = direct_apple_support.validate_direct_apple_support_plane_record(self.plane.record)
        self.assertFalse(record["candidate_refit_or_veto_applied"])
        self.assertEqual(record["support_factor_source"], "REGISTERED_APPLEDEPTH_CONFIDENCE_EQ_2_APPLE_RANGE_ONLY")
        self.assertEqual(record["direct_source_receipt_sha256"], self.direct_source["content_sha256"])
        self.assertEqual(record["r0_scale_pair_mask"]["selected_pixel_ids_sha256"], self.scale_record["selected_pixel_ids_sha256"])
        self.assertFalse(record["apple_support_mask"]["candidate_depth_used"])
        self.assertAlmostEqual(self.plane.camera_height_m, self.geometry.camera_height_m, places=2)

    def test_candidate_range_change_cannot_veto_apple_support_points(self) -> None:
        changed = self.raw_candidate.copy()
        apple_m = self.apple.astype(np.float64) / 1000.0
        row, column = np.argwhere((self.confidence == 2) & (apple_m >= 0.25) & (apple_m <= 5.0))[0]
        scale_x, scale_y = adapter.LOWRES_TO_HIGHRES_SCALE_XY
        x = int(np.rint((float(column) + 0.5) * scale_x - 0.5))
        y = int(np.rint((float(row) + 0.5) * scale_y - 0.5))
        changed[y, x] = 10.0
        changed_binding = _candidate_binding(self.source, changed)
        changed_scale = apple_scale.build_source_scale_record(self.apple, self.confidence, changed, self.apple_receipt, changed_binding)
        changed_prepared = direct_apple_support.prepare_direct_source_candidate(
            changed,
            self.apple,
            self.confidence,
            self.direct_source,
            self.candidate_input,
            self.apple_receipt,
            changed_binding,
            changed_scale,
        )
        changed_plane = direct_apple_support.derive_direct_apple_support_plane(
            changed_prepared,
            self.apple,
            self.confidence,
            self.direct_source,
            changed_scale,
        )
        self.assertNotEqual(self.plane.record["r0_scale_pair_mask"], changed_plane.record["r0_scale_pair_mask"])
        self.assertEqual(self.plane.record["apple_support_mask"], changed_plane.record["apple_support_mask"])
        self.assertEqual(self.plane.record["apple_support"], changed_plane.record["apple_support"])

    def test_phase_a_rejects_refined_height_outside_physical_range(self) -> None:
        support = self.plane.record["apple_support"]
        invalid_plane = {
            "normal_camera_xyz": np.asarray(support["normal_camera_xyz"], dtype=np.float64),
            "camera_height_m": -0.05,
            "support_count": int(support["support_count"]),
            "support_fraction": float(support["support_fraction"]),
            "slope_degrees": float(support["slope_degrees"]),
            "median_residual_m": float(support["median_residual_m"]),
        }
        with mock.patch.object(adapter, "_fit_support_plane", return_value=invalid_plane):
            with self.assertRaises(direct_apple_support.DirectAppleSupportError) as caught:
                direct_apple_support.derive_direct_apple_support_plane(
                    self.prepared,
                    self.apple,
                    self.confidence,
                    self.direct_source,
                    self.scale_record,
                )
        self.assertEqual(caught.exception.code, "DIRECT_APPLE_SUPPORT_HEIGHT_IMPLAUSIBLE")

    def test_phase_b_scores_plane_and_preserves_unknown(self) -> None:
        record = direct_apple_support.evaluate_direct_apple_query(
            self.prepared,
            self.direct_source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            self.plane,
            self.r1_lost,
        )
        self.assertTrue(record["source_support_available"])
        self.assertTrue(record["posthoc_query_comparison_evaluable"])
        self.assertIsInstance(record["support_no_regret_vs_r1_baseline"], bool)
        self.assertFalse(record["faro_used_for_source_support"])

        unknown = direct_apple_support.evaluate_direct_apple_query(
            self.prepared,
            self.direct_source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            None,
            self.r1_lost,
            source_failure_code="SUPPORT_SLOPE_EXCEEDED",
        )
        self.assertFalse(unknown["posthoc_query_comparison_evaluable"])
        self.assertEqual(unknown["posthoc_query_comparison"]["reason_codes"], ["SUPPORT_SLOPE_EXCEEDED"])

    def test_summary_is_descriptive_and_round_trip_stable(self) -> None:
        record = direct_apple_support.evaluate_direct_apple_query(
            self.prepared,
            self.direct_source["intrinsics_highres"]["matrix_3x3"],
            self.source["gravity_up_camera_xyz"],
            self.base,
            self.plane,
            self.r1_lost,
        )
        summary = direct_apple_support.summarize_direct_apple([record], [], expected_query_count=1, expected_frame_count=1)
        stored = json.loads(adapter.canonical_json_bytes(record).decode("utf-8"))
        rebuilt = direct_apple_support.summarize_direct_apple([stored], [], expected_query_count=1, expected_frame_count=1)
        self.assertEqual(summary["content_sha256"], rebuilt["content_sha256"])
        self.assertFalse(summary["threshold_or_pass_fail_decision_applied"])


if __name__ == "__main__":
    unittest.main()
