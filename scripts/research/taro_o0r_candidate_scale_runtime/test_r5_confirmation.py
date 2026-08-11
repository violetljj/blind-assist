#!/usr/bin/env python3
"""Focused synthetic tests for the independent TARO R5 confirmation mechanics."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _faro_depth() -> np.ndarray:
    rows = np.arange(adapter.HIGHRES_SHAPE_HW[0], dtype=np.float64)[:, None]
    denominator = rows - 700.0
    floor_z = np.divide(1.5 * 80.0, denominator, out=np.zeros_like(denominator), where=denominator > 0.0)
    valid_rows = (floor_z >= 0.25) & (floor_z <= 6.0)
    depth = np.broadcast_to(
        np.where(valid_rows, np.rint(floor_z * 1000.0), 0.0),
        adapter.HIGHRES_SHAPE_HW,
    ).astype(np.uint16).copy()
    depth[710:811, 948:972] = 1000
    return depth


def _source_fixture() -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent, video = r5.R5_ROSTER[0]
    token = "1.100000000"
    trajectory = [
        {"timestamp_token": "1.090000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"timestamp_token": "1.110000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    ]
    lowres = {
        "width": 256,
        "height": 192,
        "fx": 800.0 / 7.5,
        "fy": 80.0 / 7.5,
        "cx": (959.5 + 0.5) / 7.5 - 0.5,
        "cy": (700.0 + 0.5) / 7.5 - 0.5,
    }
    color = np.zeros((*adapter.HIGHRES_SHAPE_HW, 3), dtype=np.uint8)
    color[..., 1] = 127
    faro = _faro_depth()
    apple = adapter.sample_faro_at_apple_centers(faro).astype(np.uint16)
    confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
    arrays: dict[str, object] = {
        "color": color,
        "highres_depth": faro,
        "lowres_depth": apple,
        "confidence": confidence,
        "intrinsics": lowres,
        "trajectory": trajectory,
    }
    assets: dict[str, dict[str, object]] = {}
    decoded: dict[str, dict[str, object]] = {}
    for index, role in enumerate(adapter.DECODED_PAYLOAD_KINDS):
        payload = f"r5:{parent}:{video}:{token}:{role}".encode("utf-8")
        assets[role] = {
            "container_id": "r5-synthetic-container",
            "member_path": f"{role}/{token}.bin" if role != "trajectory" else "trajectory/lowres_wide.traj",
            "exact_timestamp_stem": None if role == "trajectory" else token,
            "bytes": index + 1,
            "sha256": _sha(payload),
            "crc32": _sha(b"crc:" + payload)[:8],
        }
        decoded[role] = {
            "asset_role": role,
            "member_sha256": assets[role]["sha256"],
            "member_crc32": assets[role]["crc32"],
            "decoded_kind": adapter.DECODED_PAYLOAD_KINDS[role],
            "decoded_content_sha256": adapter.canonical_sha256(arrays[role]),
        }
    source = adapter.build_source_frame_receipt(
        source_role="ADAPTER_FIT",
        visit_id=parent,
        video_id=video,
        frame_timestamp_token=token,
        lowres_intrinsics=lowres,
        trajectory_rows=trajectory,
        asset_bindings=assets,
        decoded_payload_bindings=decoded,
    )
    return source, color, faro, apple, confidence


def _candidate_frame(source: dict[str, object], color: np.ndarray, depth_m: float) -> tuple[dict[str, object], np.ndarray]:
    candidate_input = r5.build_candidate_input(source, color)
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(color, matrix)
    native = np.full(depthart_runner.NATIVE_SHAPE_HW, depth_m, dtype=np.float32)
    highres = depthart_runner.upsample_native_depth(native)
    inference = r5.build_inference_receipt(
        candidate_input,
        color,
        tensor,
        resized_k,
        native,
        highres,
        {"backend": "synthetic-cpu-fp32"},
    )
    blob_bytes = depthart_runner.deterministic_npy_gzip_bytes(native)
    blob = {
        "path": "candidates/native-depth.npy.gz",
        "bytes": len(blob_bytes),
        "sha256": _sha(blob_bytes),
        "array_sha256": adapter.canonical_sha256(native),
        "shape_hw": list(depthart_runner.NATIVE_SHAPE_HW),
        "dtype": "float32",
        "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0",
    }
    return r5.build_candidate_frame_record(candidate_input, inference, blob), native


def _successful_mode(depth_hash: str) -> dict[str, object]:
    return {
        "extraction_evaluable": True,
        "reason_codes": [],
        "depth_array_sha256": depth_hash,
        "valid_common_point_count": 512,
        "support": {
            "evaluable": True,
            "reason_codes": [],
            "normal_angular_error_rad": 0.20,
            "height_abs_error_m": 0.30,
            "camera_height_m": 1.50,
            "support_point_count": 512,
            "support_fraction": 0.50,
            "slope_degrees": 2.0,
        },
        "boundary": {
            "evaluable": True,
            "reason_codes": [],
            "truth_point_count": 20,
            "candidate_point_count": 20,
            "point_id_intersection_count": 20,
            "point_id_union_count": 20,
            "point_id_jaccard": 1.0,
            "xyz_median_error_m": 0.10,
            "local_valid_fraction": 1.0,
        },
        "query_point_clearance": {
            "evaluable": True,
            "reason_codes": [],
            "value_m": 0.8,
            "truth_value_m": 0.7,
            "abs_error_m": 0.1,
            "query_support_points": 100,
            "observed_forward_m": 2.0,
            "local_valid_fraction": 1.0,
        },
    }


def _phase_a_completion() -> dict[str, object]:
    zero_roles = ["FARO", "QUERY_TRUTH", "COMPACT_TRUTH", "TASK_METRIC", "PRIOR_EVAL_OUTCOME"]
    return r5.validate_phase_a_completion(
        r5._seal(
            {
                "schema": r5.PHASE_A_COMPLETION_SCHEMA,
                "r5_role": r5.R5_ROLE,
                "policy_id": r5.POLICY_ID,
                "candidate_phase_completion_sha256": "4" * 64,
                "physical_frame_count": r5.EXPECTED_FRAME_COUNT,
                "source_decision_key_sequence_sha256": "5" * 64,
                "source_decision_hash_sequence_sha256": "6" * 64,
                "direct_selected_frame_count": 1,
                "baseline_fallback_frame_count": r5.EXPECTED_FRAME_COUNT - 1,
                "read_counts": {role: 0 for role in zero_roles},
                "forbidden_zero_read_roles": zero_roles,
                "all_candidates_before_source_decisions": True,
                "all_source_decisions_before_phase_b": True,
            }
        )
    )


def _positive_query_record(parent: str, video: str, frame_index: int, grid_index: int) -> dict[str, object]:
    token = f"{frame_index + 1}.000000000"
    physical = f"{video}:{token}"
    baseline = _successful_mode("A" * 64)
    direct = copy.deepcopy(baseline)
    direct["depth_array_sha256"] = "B" * 64
    direct["support"]["height_abs_error_m"] = 0.10
    direct["support"]["normal_angular_error_rad"] = 0.10
    return r5._seal(
        {
            "schema": r5.QUERY_RECORD_SCHEMA,
            "r5_role": r5.R5_ROLE,
            "policy_id": r5.POLICY_ID,
            "parent_id": parent,
            "physical_frame_id": physical,
            "query_id": f"{physical}:synthetic-{grid_index}",
            "grid_index": grid_index,
            "source_frame_receipt_sha256": "C" * 64,
            "candidate_frame_record_sha256": "D" * 64,
            "source_decision_sha256": "E" * 64,
            "phase_a_completion_sha256": "F" * 64,
            "faro_geometry_sha256": "1" * 64,
            "query_receipt_sha256": "2" * 64,
            "common_point_ids_sha256": "3" * 64,
            "phase_a_selected_branch": "DIRECT_APPLE_SUPPORT",
            "source_support_available": True,
            "baseline": baseline,
            "direct_apple_support": direct,
            "selected_hybrid": direct,
            "effects": r5._effects(baseline, direct),
            "branch_reselection_after_truth": False,
            "faro_used_for_scoring_only": True,
        }
    )


class R5ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.color, cls.faro, cls.apple, cls.confidence = _source_fixture()
        cls.candidate, cls.native = _candidate_frame(cls.source, cls.color, 2.0)
        cls.decision = r5.build_source_decision(cls.source, cls.candidate, cls.native, cls.apple, cls.confidence)

    def test_new_role_accepts_exact_fit_source_while_old_eval_api_rejects_it(self) -> None:
        self.assertEqual("ADAPTER_FIT", r5.validate_r5_source_receipt(self.source)["source_role"])
        candidate = r5.build_candidate_input(self.source, self.color)
        self.assertEqual(r5.R5_ROLE, candidate["r5_role"])
        with self.assertRaises(depthart_runner.DepthARTRuntimeError):
            depthart_runner.candidate_input_from_bound_source(self.source, self.color)

    def test_effective_transform_identity_and_native_lineage_are_enforced(self) -> None:
        receipt = self.candidate["inference_receipt"]
        self.assertEqual(depthart_runner.PREPROCESS_ID, receipt["preprocess_id"])
        self.assertEqual(depthart_runner.POSTPROCESS_ID, receipt["postprocess_id"])
        tampered = self.native.copy()
        tampered[0, 0] += np.float32(0.25)
        with self.assertRaisesRegex(r5.R5ConfirmationError, "query candidate native depth differs"):
            r5._prepared_and_plane(self.candidate, tampered, self.decision)

    def test_apple_support_membership_is_candidate_independent(self) -> None:
        other_candidate, other_native = _candidate_frame(self.source, self.color, 3.0)
        other = r5.build_source_decision(self.source, other_candidate, other_native, self.apple, self.confidence)
        self.assertTrue(self.decision["source_support_available"])
        self.assertTrue(other["source_support_available"])
        self.assertEqual(
            self.decision["direct_support_plane"]["apple_support_pixel_ids_sha256"],
            other["direct_support_plane"]["apple_support_pixel_ids_sha256"],
        )
        self.assertNotEqual(self.decision["scale_record"]["metric_scale"], other["scale_record"]["metric_scale"])

    def test_selected_direct_failure_remains_unknown_and_never_falls_back(self) -> None:
        baseline = _successful_mode("A" * 64)
        direct = source_factor._failed_mode("B" * 64, "FORCED_DIRECT_FAILURE")
        record = r5._seal(
            {
                "schema": r5.QUERY_RECORD_SCHEMA,
                "r5_role": r5.R5_ROLE,
                "policy_id": r5.POLICY_ID,
                "parent_id": r5.R5_ROSTER[0][0],
                "physical_frame_id": f"{r5.R5_ROSTER[0][1]}:1.100000000",
                "query_id": f"{r5.R5_ROSTER[0][1]}:1.100000000:lat_+0.00_yaw_+0.0",
                "grid_index": 0,
                "source_frame_receipt_sha256": "C" * 64,
                "candidate_frame_record_sha256": "D" * 64,
                "source_decision_sha256": "E" * 64,
                "phase_a_completion_sha256": "F" * 64,
                "faro_geometry_sha256": "1" * 64,
                "query_receipt_sha256": "2" * 64,
                "common_point_ids_sha256": "3" * 64,
                "phase_a_selected_branch": "DIRECT_APPLE_SUPPORT",
                "source_support_available": True,
                "baseline": baseline,
                "direct_apple_support": direct,
                "selected_hybrid": direct,
                "effects": r5._effects(baseline, direct),
                "branch_reselection_after_truth": False,
                "faro_used_for_scoring_only": True,
            }
        )
        validated = r5.validate_query_record(record)
        self.assertFalse(validated["selected_hybrid"]["extraction_evaluable"])
        self.assertTrue(validated["effects"]["extraction_lost_vs_baseline"])
        tampered = copy.deepcopy(record)
        tampered["selected_hybrid"] = baseline
        tampered.pop("content_sha256")
        tampered = r5._seal(tampered)
        with self.assertRaises(r5.R5ConfirmationError) as caught:
            r5.validate_query_record(tampered)
        self.assertEqual("R5_QUERY_OUTCOME_RESELECTION", caught.exception.code)

    def test_source_scale_and_plane_self_consistency_mutations_fail(self) -> None:
        mutated = copy.deepcopy(self.decision)
        mutated["scale_record"]["metric_scale"] = 9.0
        mutated["scale_record"].pop("content_sha256")
        mutated["scale_record"] = r5._seal(mutated["scale_record"])
        mutated.pop("content_sha256")
        mutated = r5._seal(mutated)
        with self.assertRaises(r5.R5ConfirmationError) as caught:
            r5.validate_source_decision(mutated)
        self.assertEqual("R5_SOURCE_SCALE_INVALID", caught.exception.code)

    def test_phase_b_frame_binds_completion_and_retains_nine_slots(self) -> None:
        phase_a = _phase_a_completion()
        geometry = r5.derive_faro_geometry(self.faro, self.source, self.decision, phase_a)
        records = r5.evaluate_frame(self.source, self.candidate, self.native, self.decision, geometry)
        self.assertEqual(list(range(9)), [row["grid_index"] for row in records])
        self.assertTrue(all(row["phase_a_completion_sha256"] == phase_a["content_sha256"] for row in records))
        self.assertTrue(all(row["phase_a_selected_branch"] == self.decision["selected_branch"] for row in records))

    def test_support_unobservable_faro_retains_nine_unknown_slots(self) -> None:
        phase_a = _phase_a_completion()
        faro = np.full(adapter.HIGHRES_SHAPE_HW, 250, dtype=np.uint16)
        source = copy.deepcopy(self.source)
        source["decoded_payload_bindings"]["highres_depth"]["decoded_content_sha256"] = adapter.canonical_sha256(faro)
        source.pop("content_sha256")
        source = adapter._seal(source)
        candidate, native = _candidate_frame(source, self.color, 2.0)
        decision = r5.build_source_decision(source, candidate, native, self.apple, self.confidence)
        with self.assertRaises(r5.R5ConfirmationError) as caught:
            r5.derive_faro_geometry(faro, source, decision, phase_a)
        self.assertIn(caught.exception.code, adapter._SUPPORT_UNOBSERVABLE_CODES)
        records = r5.evaluate_unobservable_faro_frame(
            source,
            candidate,
            native,
            decision,
            phase_a,
            faro,
            caught.exception.code,
        )
        self.assertEqual(list(range(9)), [row["grid_index"] for row in records])
        self.assertTrue(all(not row["baseline"]["extraction_evaluable"] for row in records))
        self.assertTrue(all(not row["selected_hybrid"]["extraction_evaluable"] for row in records))
        self.assertTrue(all(row["baseline"]["reason_codes"] == [caught.exception.code] for row in records))
        self.assertTrue(all(row["phase_a_completion_sha256"] == phase_a["content_sha256"] for row in records))

    def test_non_support_failure_cannot_be_mapped_to_unknown(self) -> None:
        with self.assertRaises(r5.R5ConfirmationError) as caught:
            r5.evaluate_unobservable_faro_frame(
                self.source,
                self.candidate,
                self.native,
                self.decision,
                _phase_a_completion(),
                self.faro,
                "R5_DECODED_PAYLOAD_HASH_DRIFT",
            )
        self.assertEqual("R5_FARO_UNKNOWN_CODE_INVALID", caught.exception.code)

    def test_exact_parent_macro_confirmation_gates(self) -> None:
        records = []
        for (parent, video), frame_count in zip(r5.R5_ROSTER, r5.EXPECTED_PARENT_FRAME_COUNTS):
            for frame_index in range(frame_count):
                records.extend(_positive_query_record(parent, video, frame_index, grid) for grid in range(9))
        summary = r5.summarize(records)
        self.assertEqual("TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_PASS", summary["terminal"])
        self.assertEqual(8, summary["parents_jointly_positive_height_and_normal"])
        self.assertAlmostEqual(0.2, summary["height_error_reduction_vs_baseline_parent_macro_m"]["median_of_parent_medians"])
        self.assertAlmostEqual(0.1, summary["normal_error_reduction_vs_baseline_parent_macro_rad"]["median_of_parent_medians"])


if __name__ == "__main__":
    unittest.main()
