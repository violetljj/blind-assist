#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import (
    eval_geometry,
    source_receipt,
    synthetic_faro_depth,
)


def _apple_receipt(source: dict, apple: np.ndarray, confidence: np.ndarray) -> dict:
    parent_id = str(source["parent_id"])
    video_id = str(source["session_id"])
    token = str(source["sensor_timestamp"]["decimal_token"])
    members = {}
    for role, value, dtype_name in (
        ("lowres_depth", apple, "uint16"),
        ("confidence", confidence, "uint8"),
    ):
        members[role] = {
            "role": role,
            "source_member_path": f"{video_id}/{role}/{video_id}_{token}.png",
            "member_bytes": 100,
            "member_sha256": ("A" if role == "lowres_depth" else "B") * 64,
            "member_crc32": "1234ABCD",
            "decoded_dtype": dtype_name,
            "decoded_shape_hw": list(adapter.APPLE_SHAPE_HW),
            "decoded_content_sha256": adapter.canonical_sha256(value),
        }
    return apple_scale._seal(
        {
            "schema": apple_scale.APPLE_SCALE_SOURCE_RECEIPT_SCHEMA,
            "parent_id": parent_id,
            "video_id": video_id,
            "timestamp_token": token,
            "physical_frame_id": source["physical_frame_id"],
            "frame_plan_sha256": "C" * 64,
            "candidate_phase_completion_sha256": "D" * 64,
            "upsampling_container": {"relative_path": "source.zip", "bytes": 1, "sha256": "E" * 64},
            "members": members,
            "opened_member_roles": list(apple_scale.SOURCE_ROLES),
            "faro_member_opened": False,
            "rgb_member_opened": False,
            "truth_alignment_used": False,
        }
    )


def _candidate_binding(source: dict, candidate: np.ndarray) -> dict:
    return apple_scale._seal(
        {
            "schema": apple_scale.CANDIDATE_REPLAY_BINDING_SCHEMA,
            "parent_id": str(source["parent_id"]),
            "video_id": str(source["session_id"]),
            "timestamp_token": str(source["sensor_timestamp"]["decimal_token"]),
            "physical_frame_id": source["physical_frame_id"],
            "candidate_frame_record_sha256": "1" * 64,
            "inference_receipt_sha256": "2" * 64,
            "native_depth_array_sha256": "3" * 64,
            "highres_depth_array_sha256": adapter.canonical_sha256(candidate),
            "candidate_truth_payload_read": False,
            "candidate_truth_alignment_used": False,
        }
    )


class SourceFactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = source_receipt()
        cls.geometry = eval_geometry()
        cls.query = adapter.build_query_receipts(cls.source, cls.geometry)[4]
        cls.base = source_factor.build_query_truth_base(cls.geometry, cls.query)
        faro_m = synthetic_faro_depth(True).astype(np.float64) / 1000.0
        cls.raw_candidate = np.ascontiguousarray(faro_m * 1.2, dtype=np.float32)
        cls.apple = adapter.sample_faro_at_apple_centers(synthetic_faro_depth(True)).astype(np.uint16)
        cls.confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
        cls.apple_receipt = _apple_receipt(cls.source, cls.apple, cls.confidence)
        cls.binding = _candidate_binding(cls.source, cls.raw_candidate)
        cls.scale_record = apple_scale.build_source_scale_record(
            cls.apple,
            cls.confidence,
            cls.raw_candidate,
            cls.apple_receipt,
            cls.binding,
        )
        cls.prepared = source_factor.prepare_source_anchored_candidate(
            cls.raw_candidate,
            cls.apple,
            cls.confidence,
            cls.source,
            cls.apple_receipt,
            cls.binding,
            cls.scale_record,
        )

    def test_source_scale_is_rederived_and_applied_internally(self) -> None:
        self.assertAlmostEqual(self.prepared.metric_scale, 1.0 / 1.2, places=4)
        self.assertEqual(self.prepared.raw_depth_sha256, self.binding["highres_depth_array_sha256"])
        self.assertNotEqual(self.prepared.anchored_depth_sha256, self.prepared.raw_depth_sha256)
        reliability = source_factor.validate_reliability_record(self.prepared.reliability)
        self.assertEqual(reliability["valid_pair_count"], self.scale_record["valid_pair_count"])
        self.assertFalse(reliability["abstention_threshold_selected"])

        tampered = copy.deepcopy(self.scale_record)
        tampered["metric_scale"] *= 1.01
        with self.assertRaises(apple_scale.AppleScaleError):
            source_factor.prepare_source_anchored_candidate(
                self.raw_candidate,
                self.apple,
                self.confidence,
                self.source,
                self.apple_receipt,
                self.binding,
                tampered,
            )

    def test_pre_extraction_anchor_improves_support_boundary_and_query(self) -> None:
        truth_point = source_factor._point_clearance(
            self.base.truth_normal_camera_xyz,
            self.base.truth_camera_height_m,
            self.base.truth_boundary_points_camera_xyz,
            self.base.truth_query_support_points,
            self.base.truth_observed_forward_m,
            self.base.local_valid_fraction,
            self.query,
        )
        record = source_factor.evaluate_source_anchored_query(
            self.prepared,
            np.asarray(self.source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            self.source["gravity_up_camera_xyz"],
            self.base,
            current_faro_geometry_sha256=self.geometry.content_sha256,
            compact_truth_record_sha256="4" * 64,
            committed_faro_geometry_sha256=self.geometry.content_sha256,
            committed_factor_frame_sha256="5" * 64,
            committed_base_geometry_sha256="6" * 64,
            compact_query_result={"physical_frame_id": self.source["physical_frame_id"], "query_id": self.query["query_id"], "value_m": truth_point["value_m"]},
        )
        self.assertTrue(record["baseline"]["extraction_evaluable"])
        self.assertTrue(record["source_anchored"]["extraction_evaluable"])
        self.assertGreater(record["effects"]["support_height_error_reduction_m"], 0.0)
        self.assertGreaterEqual(record["effects"]["boundary_jaccard_increase"], 0.0)
        self.assertGreater(record["effects"]["boundary_xyz_error_reduction_m"], 0.0)
        self.assertGreaterEqual(record["effects"]["query_point_error_reduction_m"], 0.0)
        self.assertAlmostEqual(record["source_anchored"]["support"]["height_abs_error_m"], 0.0, places=3)
        self.assertAlmostEqual(record["source_anchored"]["boundary"]["point_id_jaccard"], 1.0, places=3)
        self.assertFalse(record["formal_reducer_executed"])

        broken = copy.deepcopy(record)
        broken["source_metric_scale"] *= 2.0
        with self.assertRaises(source_factor.SourceFactorError):
            source_factor.validate_query_record(broken)

    def test_summary_is_parent_first_and_threshold_free(self) -> None:
        truth_point = source_factor._point_clearance(
            self.base.truth_normal_camera_xyz,
            self.base.truth_camera_height_m,
            self.base.truth_boundary_points_camera_xyz,
            self.base.truth_query_support_points,
            self.base.truth_observed_forward_m,
            self.base.local_valid_fraction,
            self.query,
        )
        record = source_factor.evaluate_source_anchored_query(
            self.prepared,
            np.asarray(self.source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            self.source["gravity_up_camera_xyz"],
            self.base,
            current_faro_geometry_sha256=self.geometry.content_sha256,
            compact_truth_record_sha256="7" * 64,
            committed_faro_geometry_sha256=self.geometry.content_sha256,
            committed_factor_frame_sha256="8" * 64,
            committed_base_geometry_sha256="9" * 64,
            compact_query_result={"physical_frame_id": self.source["physical_frame_id"], "query_id": self.query["query_id"], "value_m": truth_point["value_m"]},
        )
        summary = source_factor.summarize_source_anchored_canary([record], [self.prepared.reliability])
        self.assertEqual(summary["query_record_count"], 1)
        self.assertEqual(summary["parent_count"], 1)
        self.assertFalse(summary["threshold_or_pass_fail_decision_applied"])
        self.assertIsNotNone(summary["effects_parent_macro"]["support_height_error_reduction_m"]["median_of_parent_medians"])
        stored_record = json.loads(adapter.canonical_json_bytes(record).decode("utf-8"))
        stored_reliability = json.loads(adapter.canonical_json_bytes(self.prepared.reliability).decode("utf-8"))
        recomputed = source_factor.summarize_source_anchored_canary([stored_record], [stored_reliability])
        self.assertEqual(recomputed["content_sha256"], summary["content_sha256"])


if __name__ == "__main__":
    unittest.main()
