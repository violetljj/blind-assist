#!/usr/bin/env python3
"""Focused mechanics and mutation tests for TARO O1R reducer integration."""

from __future__ import annotations

import copy
import inspect
import unittest

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as runtime
from scripts.research.taro_o1r_reducer_integration_runtime import locked_uncertainty
from scripts.research.taro_o1r_reducer_integration_runtime import validate_protocol_lock


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


def _factory_model(value: float, *, sparse_target: str | None = None) -> object:
    parents = tuple(parent for parent, _ in adapter.ADAPTER_FIT_ROSTER)
    cells = []
    counts = {target: 0 for target in adapter.UNCERTAINTY_TARGETS}
    for target in adapter.UNCERTAINTY_TARGETS:
        selected_parents = parents[:3] if target == sparse_target else parents
        for parent in selected_parents:
            values = np.full(16, float(value), dtype=np.float64)
            values.setflags(write=False)
            cells.append(adapter._UncertaintyCell(target, parent, 2, 2, values))
            counts[target] += int(values.size)
    target_rank = {target: index for index, target in enumerate(adapter.UNCERTAINTY_TARGETS)}
    parent_rank = {parent: index for index, parent in enumerate(parents)}
    cells.sort(key=lambda cell: (target_rank[cell.target], parent_rank[cell.parent_id], cell.confidence, cell.range_bin))
    support_frames = counts["support_height_abs_residual_m"]
    self_hashes = tuple(f"{index:064X}" for index in range(1, support_frames + 1))
    payload = adapter._uncertainty_model_payload(cells, parents, self_hashes, "C" * 64, support_frames, counts)
    content_sha256 = adapter.canonical_sha256(payload)
    model = adapter._UncertaintyModel(
        tuple(cells),
        parents,
        self_hashes,
        "C" * 64,
        support_frames,
        tuple((target, counts[target]) for target in adapter.UNCERTAINTY_TARGETS),
        content_sha256,
        adapter._UNCERTAINTY_FACTORY_TOKEN,
    )
    adapter._UNCERTAINTY_FACTORY_FINGERPRINTS[model] = content_sha256
    return adapter._validate_uncertainty_model(model)


class ReducerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _fixture()
        cls.bundle = prospective.build_prospective_factor_bundle(**cls.fixture)
        cls.zero_model = _factory_model(0.0)
        cls.high_model = _factory_model(1.0)

    def _integrate(self, model: object | None = None, **changes: object) -> dict:
        kwargs = {
            "prospective_bundle": self.bundle,
            "candidate_highres_depth_m": self.fixture["candidate_highres_depth_m"],
            "confidence": self.fixture["confidence"],
            "intrinsics_apple_3x3": self.fixture["intrinsics_apple_3x3"],
            "uncertainty_model": self.zero_model if model is None else model,
        }
        kwargs.update(changes)
        return runtime.integrate_prospective_factor_bundle(**kwargs)

    def test_protocol_lock_is_valid(self) -> None:
        result = validate_protocol_lock.validate(validate_protocol_lock.DEFAULT_LOCK)
        self.assertTrue(result["passed"], result)

    def test_real_locked_uncertainty_artifact_rehydrates(self) -> None:
        model = locked_uncertainty.load_locked_uncertainty_model()
        self.assertEqual(locked_uncertainty.MODEL_SHA256, model.content_sha256)
        self.assertEqual(211, len(model.source_receipt_sha256s))
        self.assertEqual(8, len(model.fit_parent_ids))

    def test_public_api_has_no_result_side_argument(self) -> None:
        parameters = inspect.signature(runtime.integrate_prospective_factor_bundle).parameters
        self.assertFalse(any(token in name.lower() for name in parameters for token in ("faro", "truth", "outcome", "task_metric")))

    def test_zero_uncertainty_executes_exact_nine_query_reducer(self) -> None:
        result = self._integrate()
        self.assertEqual(9, result["query_result_count"])
        self.assertTrue(result["complete_reducer_execution"])
        self.assertEqual(runtime.REDUCER_VERSION, result["final_state_producer"])
        self.assertGreater(result["state_counts"]["CLEAR_OBSERVED"] + result["state_counts"]["OCCUPIED_OBSERVED"], 0)
        self.assertTrue(all(row["final_state_authorized"] for row in result["query_results"]))

    def test_uncertainty_worsening_cannot_strengthen_state(self) -> None:
        low = self._integrate(self.zero_model)
        high = self._integrate(self.high_model)
        for left, right in zip(low["query_results"], high["query_results"], strict=True):
            if right["state"] != "UNKNOWN":
                self.assertEqual(left["state"], right["state"])
            if left["state"] != "UNKNOWN":
                self.assertGreaterEqual(right["uncertainty_m"] or 0.0, left["uncertainty_m"] or 0.0)

    def test_invalid_uncertainty_resolution_returns_unknown(self) -> None:
        sparse = _factory_model(0.0, sparse_target="boundary_localization_abs_residual_m")
        result = self._integrate(sparse)
        self.assertEqual(9, result["state_counts"]["UNKNOWN"])
        self.assertTrue(any("UNCERTAINTY_INVALID_QUERY_UNKNOWN" in row["reason_codes"] for row in result["query_results"]))

    def test_confidence_binding_mutation_is_rejected(self) -> None:
        changed = self.fixture["confidence"].copy()
        changed[0, 0] = 1
        with self.assertRaisesRegex(runtime.ReducerIntegrationError, "confidence differs"):
            self._integrate(confidence=changed)

    def test_intrinsics_binding_mutation_is_rejected(self) -> None:
        changed = np.asarray(self.fixture["intrinsics_apple_3x3"], dtype=np.float64).copy()
        changed[0, 0] += 1.0
        with self.assertRaisesRegex(runtime.ReducerIntegrationError, "intrinsics differ"):
            self._integrate(intrinsics_apple_3x3=changed)

    def test_query_clearance_owner_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self.bundle)
        changed["query_clearance_owner"] = "DIRECT_APPLE_SUPPORT"
        changed = prospective._seal({key: value for key, value in changed.items() if key != "content_sha256"})
        with self.assertRaises(prospective.ProspectiveFactorRuntimeError):
            self._integrate(prospective_bundle=changed)

    def test_missing_query_frame_retains_nine_unknown_results(self) -> None:
        fixture = _fixture()
        fixture["candidate_highres_depth_m"] = np.full(adapter.HIGHRES_SHAPE_HW, 10.0, dtype=np.float32)
        fixture["apple_depth_mm"] = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint16)
        fixture["confidence"] = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        bundle = prospective.build_prospective_factor_bundle(**fixture)
        result = runtime.integrate_prospective_factor_bundle(
            prospective_bundle=bundle,
            candidate_highres_depth_m=fixture["candidate_highres_depth_m"],
            confidence=fixture["confidence"],
            intrinsics_apple_3x3=fixture["intrinsics_apple_3x3"],
            uncertainty_model=self.zero_model,
        )
        self.assertEqual(9, result["state_counts"]["UNKNOWN"])
        self.assertTrue(all(row["reason_codes"] == ["SOURCE_QUERY_FRAME_UNAVAILABLE"] for row in result["query_results"]))

    def test_deterministic_roundtrip_does_not_mutate_r6_bundle(self) -> None:
        before = copy.deepcopy(self.bundle)
        left = self._integrate()
        right = self._integrate()
        self.assertEqual(left, right)
        self.assertEqual(before, self.bundle)
        self.assertEqual(left, runtime.validate_reducer_bundle(copy.deepcopy(left)))

    def test_result_state_mutation_is_rejected(self) -> None:
        changed = copy.deepcopy(self._integrate())
        query = changed["query_results"][0]
        query["state"] = "OCCUPIED_OBSERVED" if query["state"] != "OCCUPIED_OBSERVED" else "CLEAR_OBSERVED"
        changed["query_results"][0] = runtime._seal({key: value for key, value in query.items() if key != "content_sha256"})
        changed["state_counts"] = {state: sum(row["state"] == state for row in changed["query_results"]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}
        changed = runtime._seal({key: value for key, value in changed.items() if key != "content_sha256"})
        with self.assertRaisesRegex(runtime.ReducerIntegrationError, "not recomputable"):
            runtime.validate_reducer_bundle(changed)

    def test_nested_uncertainty_mutation_cannot_be_hidden_by_outer_reseal(self) -> None:
        changed = copy.deepcopy(self._integrate())
        query = next(row for row in changed["query_results"] if row["uncertainty"] is not None)
        query["uncertainty"]["total_m"] = float(query["uncertainty"].get("total_m") or 0.0) + 0.1
        changed["query_results"][query["grid_index"]] = runtime._seal({key: value for key, value in query.items() if key != "content_sha256"})
        changed = runtime._seal({key: value for key, value in changed.items() if key != "content_sha256"})
        with self.assertRaisesRegex(runtime.ReducerIntegrationError, "sealed record hash drift"):
            runtime.validate_reducer_bundle(changed)


if __name__ == "__main__":
    unittest.main()
