#!/usr/bin/env python3
"""Focused tests for the TARO R6 two-stage untouched confirmation core."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _faro_depth() -> np.ndarray:
    rows = np.arange(adapter.HIGHRES_SHAPE_HW[0], dtype=np.float64)[:, None]
    denominator = rows - 700.0
    floor_z = np.divide(1.5 * 80.0, denominator, out=np.zeros_like(denominator), where=denominator > 0.0)
    valid_rows = (floor_z >= 0.25) & (floor_z <= 6.0)
    depth = np.broadcast_to(np.where(valid_rows, np.rint(floor_z * 1000.0), 0.0), adapter.HIGHRES_SHAPE_HW).astype(np.uint16).copy()
    depth[710:811, 948:972] = 1000
    return depth


def _source_fixture() -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent, video = r6.ROSTER[0]
    token = "1.100000000"
    trajectory = [
        {"timestamp_token": "1.090000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"timestamp_token": "1.110000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    ]
    lowres = {"width": 256, "height": 192, "fx": 800.0 / 7.5, "fy": 80.0 / 7.5, "cx": (959.5 + 0.5) / 7.5 - 0.5, "cy": (700.0 + 0.5) / 7.5 - 0.5}
    color = np.zeros((*adapter.HIGHRES_SHAPE_HW, 3), dtype=np.uint8)
    color[..., 1] = 127
    faro = _faro_depth()
    apple = adapter.sample_faro_at_apple_centers(faro).astype(np.uint16)
    confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
    decoded_values = {"color": color, "lowres_depth": apple, "confidence": confidence, "intrinsics": lowres, "trajectory": trajectory}
    archive_sha = _sha(b"upsampling")
    container_bindings = {
        "upsampling": {"path": "upsampling.zip", "bytes": 1, "sha256": archive_sha},
        "intrinsics": {"path": "intrinsics.zip", "bytes": 1, "sha256": _sha(b"intrinsics")},
        "trajectory": {"path": "lowres_wide.traj", "bytes": 1, "sha256": _sha(b"trajectory")},
    }
    assets = {}
    hashes = {}
    for role in r6.PHASE_A_ASSET_ROLES:
        payload = f"r6:{role}".encode()
        container = container_bindings["intrinsics"]["sha256"] if role == "intrinsics" else container_bindings["trajectory"]["sha256"] if role == "trajectory" else archive_sha
        assets[role] = {"container_sha256": container, "member_path": f"{role}/{token}", "bytes": len(payload), "sha256": _sha(payload), "crc32": _sha(b"crc" + payload)[:8]}
        hashes[role] = adapter.canonical_sha256(decoded_values[role])
    source = r6.build_phase_a_source_receipt(parent_id=parent, video_id=video, timestamp_token=token, lowres_intrinsics=lowres, trajectory_rows=trajectory, container_bindings=container_bindings, asset_bindings=assets, decoded_payload_hashes=hashes)
    return source, color, faro, apple, confidence


def _candidate(source: dict[str, object], color: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    candidate_input = r6.build_candidate_input(source, color)
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(color, matrix)
    native = np.full(depthart_runner.NATIVE_SHAPE_HW, 2.0, dtype=np.float32)
    highres = depthart_runner.upsample_native_depth(native)
    inference = r6.build_inference_receipt(candidate_input, color, tensor, resized_k, native, highres, {"backend": "synthetic-cpu-fp32"})
    payload = depthart_runner.deterministic_npy_gzip_bytes(native)
    blob = {"path": "candidates/native.npy.gz", "bytes": len(payload), "sha256": _sha(payload), "array_sha256": adapter.canonical_sha256(native), "shape_hw": list(depthart_runner.NATIVE_SHAPE_HW), "dtype": "float32", "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0"}
    return r6.build_candidate_frame(candidate_input, inference, blob), native


def _phase_a_completion() -> dict[str, object]:
    reads = {role: 0 for role in ("FARO", "QUERY_TRUTH", "TASK_METRIC", "PRIOR_OUTCOME")}
    return r6.validate_phase_a_completion(r6._seal({
        "schema": r6.PHASE_A_COMPLETION_SCHEMA,
        "analysis_role": r6.ANALYSIS_ROLE,
        "policy_id": r6.POLICY_ID,
        "candidate_completion_sha256": "1" * 64,
        "physical_frame_count": r6.EXPECTED_FRAME_COUNT,
        "decision_key_sequence_sha256": "2" * 64,
        "decision_hash_sequence_sha256": "3" * 64,
        "direct_selected_frame_count": 1,
        "baseline_fallback_frame_count": r6.EXPECTED_FRAME_COUNT - 1,
        "read_counts": reads,
        "forbidden_zero_read_roles": list(r6.FORBIDDEN_PHASE_A_READS),
        "all_candidates_before_decisions": True,
        "all_decisions_before_faro": True,
    }))


class R6ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.color, cls.faro, cls.apple, cls.confidence = _source_fixture()
        cls.candidate, cls.native = _candidate(cls.source, cls.color)
        cls.decision = r6.build_source_decision(cls.source, cls.candidate, cls.native, cls.apple, cls.confidence)
        cls.phase_a = _phase_a_completion()

    def test_phase_a_source_excludes_faro_and_rejects_truth_role(self) -> None:
        self.assertNotIn("highres_depth", self.source["asset_bindings"])
        self.assertFalse(self.source["highres_faro_member_bound"])
        with self.assertRaises(r6.R6ConfirmationError) as caught:
            r6.validate_bound_phase_a_payload(self.source, "highres_depth", self.faro)
        self.assertEqual("R6_PHASE_A_PAYLOAD_ROLE_INVALID", caught.exception.code)

    def test_source_decision_is_truth_blind_and_candidate_independent_membership(self) -> None:
        self.assertTrue(self.decision["source_support_available"])
        self.assertFalse(self.decision["faro_payload_read"])
        other_candidate, _ = _candidate(self.source, self.color)
        # Re-sealing another candidate is unnecessary for the membership claim;
        # the Apple support ids themselves are explicitly candidate-independent.
        self.assertFalse(self.decision["direct_support_plane"]["candidate_depth_used_for_support_mask"])
        apple_m = self.apple.astype(np.float64) / 1000.0
        support = (self.confidence == 2) & (apple_m >= 0.25) & (apple_m <= 6.0)
        self.assertEqual(adapter.canonical_sha256(np.flatnonzero(support).astype(np.int64)), self.decision["direct_support_plane"]["apple_support_pixel_ids_sha256"])
        self.assertEqual(other_candidate["physical_frame_id"], self.candidate["physical_frame_id"])

    def test_truth_binding_requires_same_source_container_and_phase_a_seal(self) -> None:
        binding = {"container_sha256": self.source["container_bindings"]["upsampling"]["sha256"], "member_path": "highres_depth/frame.png", "bytes": 1, "sha256": _sha(b"faro"), "crc32": "1234ABCD"}
        truth = r6.build_truth_binding(self.source, self.phase_a, member_binding=binding, highres_depth_mm=self.faro)
        self.assertTrue(truth["first_faro_read_after_phase_a_reload"])
        bad = copy.deepcopy(binding)
        bad["container_sha256"] = "F" * 64
        with self.assertRaises(r6.R6ConfirmationError) as caught:
            r6.build_truth_binding(self.source, self.phase_a, member_binding=bad, highres_depth_mm=self.faro)
        self.assertEqual("R6_FARO_CONTAINER_LINEAGE_DRIFT", caught.exception.code)

    def test_phase_b_emits_nine_exact_copy_composites(self) -> None:
        binding = {"container_sha256": self.source["container_bindings"]["upsampling"]["sha256"], "member_path": "highres_depth/frame.png", "bytes": 1, "sha256": _sha(b"faro"), "crc32": "1234ABCD"}
        truth = r6.build_truth_binding(self.source, self.phase_a, member_binding=binding, highres_depth_mm=self.faro)
        geometry = r6.derive_faro_geometry(self.faro, self.source, self.decision, self.phase_a, truth)
        records = r6.evaluate_frame(self.source, self.candidate, self.native, self.decision, geometry)
        self.assertEqual(9, len(records))
        self.assertEqual(list(range(9)), [composite["grid_index"] for _, _, composite in records])
        for truth_scoring, components, composite in records:
            self.assertEqual(truth_scoring["content_sha256"], components["truth_scoring_record_sha256"])
            self.assertEqual(components["baseline"]["query_point_clearance"], composite["query_clearance"])
            self.assertEqual(components["selected_support_boundary"]["support"], composite["support"])
            self.assertEqual("R1_BASELINE", composite["factor_owners"]["QUERY_CLEARANCE"])

    def test_support_unobservable_faro_retains_nine_unknown_slots(self) -> None:
        faro = np.full(adapter.HIGHRES_SHAPE_HW, 250, dtype=np.uint16)
        binding = {"container_sha256": self.source["container_bindings"]["upsampling"]["sha256"], "member_path": "highres_depth/frame.png", "bytes": 1, "sha256": _sha(b"flat-faro"), "crc32": "1234ABCD"}
        truth = r6.build_truth_binding(self.source, self.phase_a, member_binding=binding, highres_depth_mm=faro)
        with self.assertRaises(r6.R6ConfirmationError) as caught:
            r6.derive_faro_geometry(faro, self.source, self.decision, self.phase_a, truth)
        self.assertIn(caught.exception.code, adapter._SUPPORT_UNOBSERVABLE_CODES)
        records = r6.evaluate_unobservable_faro_frame(self.source, self.candidate, self.native, self.decision, self.phase_a, truth, faro, caught.exception.code)
        self.assertEqual(9, len(records))
        self.assertTrue(all(not component["baseline"]["extraction_evaluable"] for _, component, _ in records))
        self.assertTrue(all(composite["query_clearance"]["reason_codes"] == [caught.exception.code] for _, _, composite in records))


if __name__ == "__main__":
    unittest.main()
