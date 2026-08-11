#!/usr/bin/env python3
"""Synthetic mutation tests for the source-defined TARO factor runtime."""

from __future__ import annotations

import copy
import inspect
import json
import unittest
from unittest import mock

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as runtime
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _fixture() -> dict:
    high_k = np.asarray([[100.0, 0.0, 960.0], [0.0, 100.0, 720.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    low_k = np.asarray([[100.0 / 7.5, 0.0, 128.0], [0.0, 100.0 / 7.5, 96.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raw = np.full(adapter.HIGHRES_SHAPE_HW, 10.0, dtype=np.float32)
    rows = np.arange(adapter.HIGHRES_SHAPE_HW[0], dtype=np.float64)
    valid_rows = rows > high_k[1, 2]
    true_z = np.zeros_like(rows)
    true_z[valid_rows] = 1.2 * high_k[1, 1] / (rows[valid_rows] - high_k[1, 2])
    floor = valid_rows & (true_z >= 0.25) & (true_z <= 6.0)
    raw[floor, :] = (0.8 * true_z[floor])[:, None]

    apple = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint16)
    confidence = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
    low_rows = np.arange(adapter.APPLE_SHAPE_HW[0], dtype=np.float64)
    low_valid = low_rows > low_k[1, 2]
    low_z = np.zeros_like(low_rows)
    low_z[low_valid] = 1.2 * low_k[1, 1] / (low_rows[low_valid] - low_k[1, 2])
    low_floor = low_valid & (low_z >= 0.25) & (low_z <= 6.0)
    apple[low_floor, :] = np.rint(1000.0 * low_z[low_floor])[:, None].astype(np.uint16)
    confidence[low_floor, :] = 2
    return {
        "parent_id": "SYNTHETIC_PARENT",
        "video_id": "SYNTHETIC_VIDEO",
        "timestamp_token": "1.000",
        "source_frame_receipt_sha256": "A" * 64,
        "candidate_frame_record_sha256": "B" * 64,
        "max_source_timestamp_ns": 1_000_000_000,
        "candidate_highres_depth_m": raw,
        "apple_depth_mm": apple,
        "confidence": confidence,
        "intrinsics_highres_3x3": high_k,
        "intrinsics_apple_3x3": low_k,
        "gravity_up_camera_xyz": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    }


class ProspectiveFactorRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _fixture()
        cls.bundle = runtime.build_prospective_factor_bundle(**cls.fixture)

    def test_public_api_has_no_result_side_argument(self) -> None:
        parameters = inspect.signature(runtime.build_prospective_factor_bundle).parameters
        self.assertFalse(any(token in name.lower() for name in parameters for token in ("faro", "truth", "outcome", "task_metric")))
        self.assertEqual("DIRECT_APPLE_SUPPORT", self.bundle["selected_support_boundary_owner"])

    def test_source_pixel_ids_are_bound_for_every_evaluable_surface(self) -> None:
        for slot in self.bundle["query_slots"]:
            for name in ("BOUNDARY", "QUERY_CLEARANCE"):
                block = slot["factor_blocks"][name]
                if block["evaluable"]:
                    self.assertRegex(block["validity"]["source_surface_pixel_ids_sha256"], r"^[0-9A-F]{64}$")

    def test_wrong_factor_depth_lineage_is_rejected(self) -> None:
        changed = copy.deepcopy(self.bundle)
        slot = changed["query_slots"][0]
        block = slot["factor_blocks"]["SUPPORT"]
        self.assertTrue(block["evaluable"])
        block["depth_array_sha256"] = changed["input_bindings"]["candidate_highres_depth_sha256"]
        changed["query_slots"][0] = runtime._seal({key: value for key, value in slot.items() if key != "content_sha256"})
        changed = runtime._seal({key: value for key, value in changed.items() if key != "content_sha256"})
        with self.assertRaisesRegex(runtime.ProspectiveFactorRuntimeError, "support/boundary depth owner drift"):
            runtime.validate_prospective_factor_bundle(changed)

    def test_candidate_array_mutation_is_rejected(self) -> None:
        changed = self.fixture["candidate_highres_depth_m"].copy()
        changed[0, 0] = 9.0
        with self.assertRaisesRegex(runtime.ProspectiveFactorRuntimeError, "candidate differs from bundle binding"):
            runtime.validate_prospective_factor_bundle(self.bundle, candidate_highres_depth_m=changed)

    def test_exact_nine_query_slots_are_required(self) -> None:
        changed = copy.deepcopy(self.bundle)
        changed["query_slots"].pop()
        changed = runtime._seal({key: value for key, value in changed.items() if key != "content_sha256"})
        with self.assertRaisesRegex(runtime.ProspectiveFactorRuntimeError, "retain nine query slots"):
            runtime.validate_prospective_factor_bundle(changed)

    def test_failed_source_support_retains_nine_unknown_slots(self) -> None:
        fixture = _fixture()
        fixture["candidate_highres_depth_m"] = np.full(adapter.HIGHRES_SHAPE_HW, 10.0, dtype=np.float32)
        fixture["apple_depth_mm"] = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint16)
        fixture["confidence"] = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        bundle = runtime.build_prospective_factor_bundle(**fixture)
        self.assertEqual("UNAVAILABLE", bundle["query_frame_owner"])
        self.assertEqual(9, len(bundle["query_slots"]))
        for slot in bundle["query_slots"]:
            self.assertEqual("UNKNOWN", slot["final_state"])
            self.assertTrue(all(not block["evaluable"] for block in slot["factor_blocks"].values()))

    def test_deterministic_canonical_roundtrip(self) -> None:
        replay = runtime.build_prospective_factor_bundle(**self.fixture)
        self.assertEqual(self.bundle, replay)
        loaded = json.loads(adapter.canonical_json_bytes(self.bundle).decode("utf-8"))
        self.assertEqual(self.bundle, runtime.validate_prospective_factor_bundle(loaded))

    def test_r6_untouched_parent_is_rejected(self) -> None:
        fixture = dict(self.fixture)
        fixture["parent_id"] = "423306"
        with self.assertRaisesRegex(runtime.ProspectiveFactorRuntimeError, "cannot enter prospective"):
            runtime.build_prospective_factor_bundle(**fixture)

    def test_implausible_baseline_height_is_retained_as_unavailable(self) -> None:
        fitted = {
            "normal_camera_xyz": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
            "camera_height_m": -0.03,
            "support_count": 100,
            "sampled_valid_points": 1000,
            "support_fraction": 0.1,
            "slope_degrees": 0.0,
            "median_residual_m": 0.01,
        }
        with mock.patch.object(adapter, "_fit_support_plane", return_value=fitted):
            result = runtime._fit_depth_plane(
                np.ones(adapter.HIGHRES_SHAPE_HW, dtype=np.float64),
                np.asarray(self.fixture["intrinsics_highres_3x3"], dtype=np.float64),
                np.asarray(self.fixture["gravity_up_camera_xyz"], dtype=np.float64),
            )
        self.assertFalse(result["evaluable"])
        self.assertEqual(["R6_RUNTIME_BASELINE_HEIGHT_IMPLAUSIBLE"], result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
