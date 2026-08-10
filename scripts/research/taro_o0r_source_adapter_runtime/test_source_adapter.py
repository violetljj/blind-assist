#!/usr/bin/env python3
"""Synthetic-only focused tests for the TARO O0R source-adapter runtime."""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import unittest
from functools import lru_cache

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as runtime_module
from scripts.research.taro_o0r_source_adapter_runtime.source_adapter import (
    ARMS,
    ORACLE_MODES,
    AdapterError,
    bootstrap_support_uncertainty,
    build_candidate_depth_output_receipt,
    build_candidate_query_factor_frame,
    build_query_receipts,
    build_source_frame_receipt,
    build_truth_query_factor_frame,
    canonical_json_bytes,
    canonical_sha256,
    decimal_timestamp_ns,
    derive_faro_geometry,
    fit_uncertainty_model,
    inject_factor_blocks,
    interpolate_camera_to_world_exact,
    reduce_complete_query_bundle,
    reduce_query_factor_frame,
    sample_faro_at_apple_centers,
    scale_lowres_intrinsics,
    validate_query_factor_frame,
)


FIT_ROSTER = runtime_module.ADAPTER_FIT_ROSTER
EVAL_ROSTER = runtime_module.O0R_EVAL_CANDIDATE_ROSTER


def timestamp_token(frame_index: int) -> str:
    nanoseconds = 1_100_000_000 + frame_index * 1_000_000
    return f"{nanoseconds // 1_000_000_000}.{nanoseconds % 1_000_000_000:09d}"


def trajectory_rows_for(token: str, *, gap_ns: int = 20_000_000, moving: bool = False) -> list[dict[str, object]]:
    center = decimal_timestamp_ns(token)

    def as_token(value: int) -> str:
        return f"{value // 1_000_000_000}.{value % 1_000_000_000:09d}"

    half = gap_ns // 2
    return [
        {
            "timestamp_token": as_token(center - half),
            "rotation_vector": [math.pi / 2.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
        },
        {
            "timestamp_token": as_token(center + half),
            "rotation_vector": [math.pi / 2.0, 0.0, 0.0],
            "translation": [-2.0 if moving else 0.0, 0.0, 0.0],
        },
    ]


def asset_bindings(token: str, identity: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for index, role in enumerate(("color", "highres_depth", "lowres_depth", "confidence", "intrinsics", "trajectory")):
        payload = f"impl_unit:{identity}:{token}:{role}".encode("utf-8")
        output[role] = {
            "container_id": f"synthetic-container-{identity}",
            "member_path": f"{role}/{token}.bin" if role != "trajectory" else f"trajectory/trajectory.{index}.bin",
            "exact_timestamp_stem": None if role == "trajectory" else token,
            "bytes": index + 1,
            "sha256": hashlib.sha256(payload).hexdigest().upper(),
            "crc32": hashlib.sha256(b"crc:" + payload).hexdigest()[:8].upper(),
        }
    return output


def lowres_intrinsics() -> dict[str, object]:
    return {
        "width": 256,
        "height": 192,
        "fx": 800.0 / 7.5,
        "fy": 80.0 / 7.5,
        "cx": (959.5 + 0.5) / 7.5 - 0.5,
        "cy": (700.0 + 0.5) / 7.5 - 0.5,
    }


def decoded_payload_bindings(
    assets: dict[str, dict[str, object]],
    trajectory: list[dict[str, object]],
    *,
    color_identity: str,
) -> dict[str, dict[str, object]]:
    faro = synthetic_faro_depth(True)
    apple = sample_faro_at_apple_centers(faro).astype(np.uint16)
    confidence = np.full(runtime_module.APPLE_SHAPE_HW, 2, dtype=np.uint8)
    decoded_values: dict[str, object] = {
        "color": {"synthetic_rgb_identity": color_identity},
        "highres_depth": faro,
        "lowres_depth": apple,
        "confidence": confidence,
        "intrinsics": lowres_intrinsics(),
        "trajectory": trajectory,
    }
    return {
        role: {
            "asset_role": role,
            "member_sha256": assets[role]["sha256"],
            "member_crc32": assets[role]["crc32"],
            "decoded_kind": runtime_module.DECODED_PAYLOAD_KINDS[role],
            "decoded_content_sha256": canonical_sha256(decoded_values[role]),
        }
        for role in runtime_module.DECODED_PAYLOAD_KINDS
    }


def source_receipt(
    *,
    role: str = "O0R_EVAL_CANDIDATE",
    roster_index: int = 0,
    frame_index: int = 0,
) -> dict[str, object]:
    roster = FIT_ROSTER if role == "ADAPTER_FIT" else EVAL_ROSTER
    visit_id, video_id = roster[roster_index]
    token = timestamp_token(frame_index)
    trajectory = trajectory_rows_for(token)
    assets = asset_bindings(token, f"{role}-{visit_id}-{video_id}-{frame_index}")
    return build_source_frame_receipt(
        source_role=role,
        visit_id=visit_id,
        video_id=video_id,
        frame_timestamp_token=token,
        lowres_intrinsics=lowres_intrinsics(),
        trajectory_rows=trajectory,
        asset_bindings=assets,
        decoded_payload_bindings=decoded_payload_bindings(assets, trajectory, color_identity=f"{role}-{visit_id}-{video_id}-{frame_index}"),
    )


@lru_cache(maxsize=2)
def synthetic_faro_depth(include_obstacle: bool = True) -> np.ndarray:
    height, width = 1440, 1920
    rows = np.arange(height, dtype=np.float64)[:, None]
    denominator = rows - 700.0
    floor_z = np.divide(1.5 * 80.0, denominator, out=np.zeros_like(denominator), where=denominator > 0.0)
    valid_rows = (floor_z >= 0.25) & (floor_z <= 6.0)
    depth = np.broadcast_to(np.where(valid_rows, np.rint(floor_z * 1000.0), 0.0), (height, width)).astype(np.uint16).copy()
    if include_obstacle:
        depth[710:811, 948:972] = 1000
    depth.setflags(write=False)
    return depth


@lru_cache(maxsize=3)
def fit_sources(frames_per_parent: int) -> tuple[dict[str, object], ...]:
    faro = synthetic_faro_depth(True)
    apple = sample_faro_at_apple_centers(faro).astype(np.uint16)
    apple.setflags(write=False)
    confidence = np.full(runtime_module.APPLE_SHAPE_HW, 2, dtype=np.uint8)
    confidence.setflags(write=False)
    frames: list[dict[str, object]] = []
    for parent_index in range(len(FIT_ROSTER)):
        for local_index in range(frames_per_parent):
            frames.append(
                {
                    "source_frame_receipt": source_receipt(
                        role="ADAPTER_FIT",
                        roster_index=parent_index,
                        frame_index=parent_index * 100 + local_index,
                    ),
                    "highres_faro_depth_mm": faro,
                    "apple_depth_mm": apple,
                    "confidence": confidence,
                }
            )
    return tuple(frames)


@lru_cache(maxsize=2)
def fitted_model(frames_per_parent: int = 16) -> object:
    return fit_uncertainty_model(fit_sources(frames_per_parent))


@lru_cache(maxsize=1)
def eval_geometry() -> runtime_module.FaroGeometry:
    receipt = source_receipt()
    matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    return derive_faro_geometry(synthetic_faro_depth(True), matrix, receipt["gravity_up_camera_xyz"], receipt)


def manual_reseal(value: dict[str, object]) -> dict[str, object]:
    output = copy.deepcopy(value)
    output.pop("content_sha256", None)
    output["content_sha256"] = canonical_sha256(output)
    return output


def with_constant_apple_depth(source: dict[str, object], depth_mm: int = 250) -> dict[str, object]:
    frame = dict(source)
    apple = np.full(runtime_module.APPLE_SHAPE_HW, depth_mm, dtype=np.uint16)
    receipt = copy.deepcopy(frame["source_frame_receipt"])
    receipt["decoded_payload_bindings"]["lowres_depth"]["decoded_content_sha256"] = canonical_sha256(apple)
    frame["source_frame_receipt"] = manual_reseal(receipt)
    frame["apple_depth_mm"] = apple
    return frame


class ExactReceiptTests(unittest.TestCase):
    def test_decimal_timestamp_is_exact_and_rejects_non_plain_tokens(self) -> None:
        self.assertEqual(decimal_timestamp_ns("1.000000001"), 1_000_000_001)
        self.assertEqual(decimal_timestamp_ns("169.123456789"), 169_123_456_789)
        with self.assertRaises(AdapterError) as subnano:
            decimal_timestamp_ns("1.0000000001")
        self.assertEqual(subnano.exception.code, "TIMESTAMP_SUBNANOSECOND")
        for token in ("1e0", "NaN", "-1.0"):
            with self.subTest(token=token), self.assertRaises(AdapterError):
                decimal_timestamp_ns(token)

    def test_pose_interpolation_binds_future_right_bracket_watermark(self) -> None:
        token = timestamp_token(0)
        transform, receipt = interpolate_camera_to_world_exact(trajectory_rows_for(token, moving=True), token)
        self.assertGreater(receipt["right_timestamp_ns"], receipt["frame_timestamp_ns"])
        self.assertEqual(receipt["max_source_timestamp_ns"], receipt["right_timestamp_ns"])
        self.assertAlmostEqual(float(transform[0, 3]), 1.0, places=12)

    def test_pose_gap_fails_closed(self) -> None:
        token = timestamp_token(0)
        with self.assertRaises(AdapterError) as caught:
            interpolate_camera_to_world_exact(trajectory_rows_for(token, gap_ns=300_000_000), token)
        self.assertEqual(caught.exception.code, "POSE_BRACKET_GAP_EXCEEDED")

    def test_intrinsics_use_frozen_pixel_center_algebra(self) -> None:
        scaled = scale_lowres_intrinsics(lowres_intrinsics())
        self.assertEqual((scaled["height"], scaled["width"]), (1440, 1920))
        self.assertAlmostEqual(scaled["fx"], 800.0, places=12)
        self.assertAlmostEqual(scaled["fy"], 80.0, places=12)
        self.assertAlmostEqual(scaled["cx"], 959.5, places=12)
        self.assertAlmostEqual(scaled["cy"], 700.0, places=12)

    def test_base_receipt_is_roster_bound_and_spawns_nine_bound_queries(self) -> None:
        base = source_receipt()
        self.assertEqual(base["source_role"], "O0R_EVAL_CANDIDATE")
        self.assertEqual(base["frozen_roster_sha256"], runtime_module.frozen_roster_sha256())
        self.assertEqual(base["max_source_timestamp_ns"], base["pose_bracket"]["right_timestamp_ns"])
        queries = build_query_receipts(base, eval_geometry())
        self.assertEqual(len(queries), 9)
        self.assertEqual([item["grid_index"] for item in queries], list(range(9)))
        self.assertTrue(all(item["source_frame_receipt_sha256"] == base["content_sha256"] for item in queries))

    def test_eval_parent_cannot_masquerade_as_adapter_fit(self) -> None:
        visit_id, video_id = EVAL_ROSTER[0]
        token = timestamp_token(0)
        trajectory = trajectory_rows_for(token)
        assets = asset_bindings(token, "role-leak")
        with self.assertRaises(AdapterError) as caught:
            build_source_frame_receipt(
                source_role="ADAPTER_FIT",
                visit_id=visit_id,
                video_id=video_id,
                frame_timestamp_token=token,
                lowres_intrinsics=lowres_intrinsics(),
                trajectory_rows=trajectory,
                asset_bindings=assets,
                decoded_payload_bindings=decoded_payload_bindings(assets, trajectory, color_identity="role-leak"),
            )
        self.assertEqual(caught.exception.code, "SOURCE_ROSTER_IDENTITY_MISMATCH")

    def test_asset_exact_stem_member_and_crc_are_mandatory(self) -> None:
        visit_id, video_id = EVAL_ROSTER[0]
        token = timestamp_token(0)
        for mutation, expected in (
            (("color", "exact_timestamp_stem", "wrong"), "ASSET_EXACT_STEM_MISMATCH"),
            (("color", "crc32", "bad"), "ASSET_BINDING_INVALID"),
            (("color", "member_path", "../escape"), "ASSET_BINDING_INVALID"),
            (("color", "member_path", "color/other.bin"), "ASSET_MEMBER_STEM_MISMATCH"),
        ):
            assets = asset_bindings(token, "asset-mutation")
            trajectory = trajectory_rows_for(token)
            role, field, value = mutation
            assets[role][field] = value
            with self.subTest(field=field), self.assertRaises(AdapterError) as caught:
                build_source_frame_receipt(
                    source_role="O0R_EVAL_CANDIDATE",
                    visit_id=visit_id,
                    video_id=video_id,
                    frame_timestamp_token=token,
                    lowres_intrinsics=lowres_intrinsics(),
                    trajectory_rows=trajectory,
                    asset_bindings=assets,
                    decoded_payload_bindings=decoded_payload_bindings(assets, trajectory, color_identity="asset-mutation"),
                )
            self.assertEqual(caught.exception.code, expected)

    def test_physical_frame_identity_is_recomputed_not_trusted(self) -> None:
        base = source_receipt()
        base["physical_frame_id"] = "forged"
        forged = manual_reseal(base)
        with self.assertRaises(AdapterError) as caught:
            build_query_receipts(forged, eval_geometry())
        self.assertEqual(caught.exception.code, "BASE_RECEIPT_IDENTITY")

    def test_decoded_payload_receipt_binds_member_and_actual_content(self) -> None:
        visit_id, video_id = EVAL_ROSTER[0]
        token = timestamp_token(0)
        trajectory = trajectory_rows_for(token)
        assets = asset_bindings(token, "decoded-binding")
        decoded = decoded_payload_bindings(assets, trajectory, color_identity="decoded-binding")
        decoded["highres_depth"]["member_sha256"] = "0" * 64
        with self.assertRaises(AdapterError) as caught:
            build_source_frame_receipt(
                source_role="O0R_EVAL_CANDIDATE",
                visit_id=visit_id,
                video_id=video_id,
                frame_timestamp_token=token,
                lowres_intrinsics=lowres_intrinsics(),
                trajectory_rows=trajectory,
                asset_bindings=assets,
                decoded_payload_bindings=decoded,
            )
        self.assertEqual(caught.exception.code, "DECODED_PAYLOAD_ASSET_MISMATCH")

        decoded = decoded_payload_bindings(assets, trajectory, color_identity="decoded-binding")
        decoded["intrinsics"]["decoded_content_sha256"] = "0" * 64
        with self.assertRaises(AdapterError) as caught:
            build_source_frame_receipt(
                source_role="O0R_EVAL_CANDIDATE",
                visit_id=visit_id,
                video_id=video_id,
                frame_timestamp_token=token,
                lowres_intrinsics=lowres_intrinsics(),
                trajectory_rows=trajectory,
                asset_bindings=assets,
                decoded_payload_bindings=decoded,
            )
        self.assertEqual(caught.exception.code, "DECODED_PAYLOAD_CONTENT_MISMATCH")

        receipt = source_receipt()
        faro = synthetic_faro_depth(True).copy()
        faro[800, 100] += 1
        with self.assertRaises(AdapterError) as caught:
            derive_faro_geometry(
                faro,
                np.asarray(receipt["intrinsics_highres"]["matrix_3x3"]),
                receipt["gravity_up_camera_xyz"],
                receipt,
            )
        self.assertEqual(caught.exception.code, "DECODED_PAYLOAD_CONTENT_MISMATCH")

    def test_faro_truth_geometry_rejects_adapter_fit_role(self) -> None:
        receipt = source_receipt(role="ADAPTER_FIT")
        with self.assertRaises(AdapterError) as caught:
            derive_faro_geometry(
                synthetic_faro_depth(True),
                np.asarray(receipt["intrinsics_highres"]["matrix_3x3"]),
                receipt["gravity_up_camera_xyz"],
                receipt,
            )
        self.assertEqual(caught.exception.code, "FARO_TRUTH_SOURCE_ROLE_INVALID")

    def test_canonicalization_normalizes_zero_rejects_nan_and_hashes_arrays(self) -> None:
        self.assertEqual(canonical_json_bytes({"x": -0.0}), canonical_json_bytes({"x": 0.0}))
        with self.assertRaises(AdapterError) as caught:
            canonical_json_bytes({"x": float("nan")})
        self.assertEqual(caught.exception.code, "NONFINITE_CANONICAL_VALUE")
        first = np.asarray([[1.0, 2.0]])
        second = first.copy()
        second[0, 1] = 2.1
        self.assertNotEqual(canonical_sha256({"a": first}), canonical_sha256({"a": second}))


class UncertaintyProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = fitted_model(16)

    def test_fit_requires_exact_eight_parent_roster_and_counts_support_once_per_frame(self) -> None:
        self.assertEqual(self.model.fit_parent_ids, tuple(visit_id for visit_id, _ in FIT_ROSTER))
        self.assertEqual(self.model.support_frame_observations, 128)
        self.assertEqual(self.model.observation_counts["support_height_abs_residual_m"], 128)
        self.assertEqual(self.model.observation_counts["support_normal_abs_residual_rad"], 128)
        self.assertGreater(self.model.observation_counts["scale_log_abs_residual"], 128)
        self.assertTrue(all(not cell.values.flags.writeable for cell in self.model.cells))

    def test_support_and_pixel_cells_resolve_with_parent_macro_q95(self) -> None:
        for target in runtime_module.UNCERTAINTY_TARGETS:
            with self.subTest(target=target):
                resolved = self.model.resolve(2, 1.5, target)
                self.assertTrue(resolved["valid"])
                self.assertEqual(resolved["independent_parents"], 8)
                self.assertGreaterEqual(resolved["samples"], 128)
                self.assertAlmostEqual(resolved["value"], 0.0, places=12)

    def test_support_gate_uses_physical_frames_not_repeated_pixels(self) -> None:
        sparse = fitted_model(1)
        support = sparse.resolve(2, 1.5, "support_height_abs_residual_m")
        pixel = sparse.resolve(2, 1.5, "scale_log_abs_residual")
        self.assertFalse(support["valid"])
        self.assertEqual(support["samples"], 8)
        self.assertTrue(pixel["valid"])

    def test_support_unobservable_frame_keeps_scale_and_marks_other_targets_missing(self) -> None:
        sources = list(fit_sources(1))
        frame = dict(sources[0])
        faro = np.full(runtime_module.HIGHRES_SHAPE_HW, 250, dtype=np.uint16)
        apple = sample_faro_at_apple_centers(faro).astype(np.uint16)
        receipt = copy.deepcopy(frame["source_frame_receipt"])
        receipt["decoded_payload_bindings"]["highres_depth"]["decoded_content_sha256"] = canonical_sha256(faro)
        receipt["decoded_payload_bindings"]["lowres_depth"]["decoded_content_sha256"] = canonical_sha256(apple)
        frame["source_frame_receipt"] = manual_reseal(receipt)
        frame["highres_faro_depth_mm"] = faro
        frame["apple_depth_mm"] = apple
        sources[0] = frame

        model = fit_uncertainty_model(sources)
        self.assertEqual(len(model.source_receipt_sha256s), 8)
        self.assertEqual(model.support_frame_observations, 7)
        self.assertEqual(model.observation_counts["support_height_abs_residual_m"], 7)
        self.assertEqual(model.observation_counts["support_normal_abs_residual_rad"], 7)
        self.assertGreater(model.observation_counts["scale_log_abs_residual"], 7)
        support = model.resolve(2, 1.5, "support_height_abs_residual_m")
        self.assertEqual(support["independent_parents"], 7)
        boundary = model.resolve(2, 1.5, "boundary_localization_abs_residual_m")
        self.assertEqual(boundary["independent_parents"], 7)
        self.assertEqual(len({cell.parent_id for cell in model.cells if cell.target == "scale_log_abs_residual"}), 8)

    def test_apple_only_support_failure_preserves_faro_boundary(self) -> None:
        sources = list(fit_sources(1))
        sources[0] = with_constant_apple_depth(sources[0])
        model = fit_uncertainty_model(sources)
        support = model.resolve(2, 1.5, "support_height_abs_residual_m")
        boundary = model.resolve(2, 1.5, "boundary_localization_abs_residual_m")
        self.assertEqual(model.support_frame_observations, 7)
        self.assertEqual(support["independent_parents"], 7)
        self.assertEqual(boundary["independent_parents"], 8)

    def test_all_apple_support_unobservable_seals_scale_and_boundary_model(self) -> None:
        sources = [with_constant_apple_depth(frame) for frame in fit_sources(1)]
        model = fit_uncertainty_model(sources)
        support = model.resolve(2, 1.5, "support_height_abs_residual_m")
        boundary = model.resolve(2, 1.5, "boundary_localization_abs_residual_m")
        scale = model.resolve(2, 1.5, "scale_log_abs_residual")
        self.assertEqual(model.support_frame_observations, 0)
        self.assertFalse(support["valid"])
        self.assertEqual(support["independent_parents"], 0)
        self.assertEqual(support["samples"], 0)
        self.assertEqual(boundary["independent_parents"], 8)
        self.assertEqual(scale["independent_parents"], 8)

    def test_eval_receipt_is_rejected_before_it_can_enter_fit(self) -> None:
        leaked = dict(fit_sources(1)[0])
        leaked["source_frame_receipt"] = source_receipt(role="O0R_EVAL_CANDIDATE")
        with self.assertRaises(AdapterError) as caught:
            fit_uncertainty_model([leaked])
        self.assertEqual(caught.exception.code, "UNCERTAINTY_ROLE_LEAKAGE")

    def test_duplicate_source_receipt_cannot_inflate_support_count(self) -> None:
        duplicated = list(fit_sources(1))
        duplicated.append(duplicated[0])
        with self.assertRaises(AdapterError) as caught:
            fit_uncertainty_model(duplicated)
        self.assertEqual(caught.exception.code, "UNCERTAINTY_DUPLICATE_SOURCE_RECEIPT")

    def test_incomplete_fit_roster_fails_closed(self) -> None:
        partial = [frame for frame in fit_sources(1) if frame["source_frame_receipt"]["parent_id"] != FIT_ROSTER[-1][0]]
        with self.assertRaises(AdapterError) as caught:
            fit_uncertainty_model(partial)
        self.assertEqual(caught.exception.code, "UNCERTAINTY_FIT_ROSTER_INCOMPLETE")

    def test_model_integrity_detects_post_fit_cell_mutation(self) -> None:
        mutated = fit_uncertainty_model(fit_sources(1))
        object.__setattr__(mutated.cells[0], "confidence", (mutated.cells[0].confidence + 1) % 3)
        with self.assertRaises(AdapterError) as caught:
            mutated.resolve(2, 1.5, "scale_log_abs_residual")
        self.assertEqual(caught.exception.code, "UNCERTAINTY_MODEL_HASH_MISMATCH")

    def test_factory_model_cannot_be_mutated_and_resealed(self) -> None:
        mutated = fit_uncertainty_model(fit_sources(1))
        cells = []
        for cell in mutated.cells:
            values = np.full(cell.values.shape, 123.0, dtype=np.float64)
            immutable_values = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)
            cells.append(type(cell)(cell.target, cell.parent_id, cell.confidence, cell.range_bin, immutable_values))
        object.__setattr__(mutated, "cells", tuple(cells))
        payload = runtime_module._uncertainty_model_payload(
            mutated.cells,
            mutated.fit_parent_ids,
            mutated.source_receipt_sha256s,
            mutated.source_evidence_sha256,
            mutated.support_frame_observations,
            mutated.observation_counts,
        )
        object.__setattr__(mutated, "content_sha256", canonical_sha256(payload))
        with self.assertRaises(AdapterError) as caught:
            mutated.resolve(2, 1.5, "scale_log_abs_residual")
        self.assertEqual(caught.exception.code, "UNCERTAINTY_MODEL_FACTORY_FINGERPRINT_MISMATCH")

    def test_uncertainty_model_is_private_factory_bound_and_rejects_duck_type(self) -> None:
        self.assertFalse(hasattr(runtime_module, "UncertaintyModel"))
        self.assertNotIn("UncertaintyModel", runtime_module.__all__)

        class CallerModel:
            content_sha256 = "A" * 64

            def resolve(self, confidence: int, range_m: float, target: str) -> dict[str, object]:
                return {
                    "valid": True,
                    "value": 123.0,
                    "scope": "GLOBAL",
                    "independent_parents": 8,
                    "samples": 128,
                    "model_sha256": self.content_sha256,
                }

        geometry = eval_geometry()
        query = build_query_receipts(source_receipt(), geometry)[4]
        with self.assertRaises(AdapterError) as caught:
            build_truth_query_factor_frame(
                geometry,
                query,
                CallerModel(),
                confidence_value=2,
                range_m=1.5,
            )
        self.assertEqual(caught.exception.code, "UNCERTAINTY_MODEL_NOT_FACTORY_BOUND")

    def test_raw_residual_injection_api_does_not_exist(self) -> None:
        self.assertFalse(hasattr(runtime_module, "make_residual_samples"))
        self.assertEqual(tuple(inspect.signature(fit_uncertainty_model).parameters), ("source_frames",))

    def test_support_bootstrap_is_frame_seeded_and_deterministic(self) -> None:
        x, z = np.meshgrid(np.linspace(-1.0, 1.0, 16), np.linspace(0.5, 2.5, 16))
        y = 1.5 + 0.001 * np.sin(x * 3.0) * np.cos(z * 2.0)
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        first = bootstrap_support_uncertainty(points, [0.0, -1.0, 0.0], 1.5, "impl_unit_frame")
        second = bootstrap_support_uncertainty(points, [0.0, -1.0, 0.0], 1.5, "impl_unit_frame")
        other = bootstrap_support_uncertainty(points, [0.0, -1.0, 0.0], 1.5, "impl_unit_other")
        self.assertEqual(first, second)
        self.assertNotEqual(first["seed_first_64_bits"], other["seed_first_64_bits"])

    def test_adapter_fit_arrays_must_match_decoded_payload_receipts(self) -> None:
        source = dict(fit_sources(1)[0])
        apple = np.asarray(source["apple_depth_mm"]).copy()
        apple[100, 100] += 1
        source["apple_depth_mm"] = apple
        with self.assertRaises(AdapterError) as caught:
            fit_uncertainty_model([source])
        self.assertEqual(caught.exception.code, "DECODED_PAYLOAD_CONTENT_MISMATCH")


class GeometryReducerAndInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_receipt = source_receipt()
        cls.matrix = np.asarray(cls.base_receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
        cls.geometry = eval_geometry()
        cls.query_receipts = build_query_receipts(cls.base_receipt, cls.geometry)
        cls.center_query = cls.query_receipts[4]
        cls.model = fitted_model(16)
        cls.truth_frame = build_truth_query_factor_frame(cls.geometry, cls.center_query, cls.model, confidence_value=2, range_m=1.5)
        candidate_depth = synthetic_faro_depth(False).astype(np.float64) / 1000.0
        cls.candidate_output_receipt = build_candidate_depth_output_receipt(
            candidate_depth,
            cls.base_receipt,
            inference_receipt_sha256="A" * 64,
        )
        cls.candidate_frame = build_candidate_query_factor_frame(
            candidate_depth,
            cls.matrix,
            cls.base_receipt["gravity_up_camera_xyz"],
            cls.base_receipt,
            cls.center_query,
            cls.truth_frame["base_geometry"],
            cls.model,
            cls.candidate_output_receipt,
            confidence_value=2,
            range_m=1.5,
        )

    def test_faro_geometry_has_metric_support_and_signed_boundary(self) -> None:
        self.assertGreaterEqual(self.geometry.support_count, 256)
        self.assertGreaterEqual(self.geometry.support_fraction, 0.02)
        self.assertLessEqual(self.geometry.support_slope_degrees, 20.0)
        self.assertTrue(self.geometry.obstacle_mask_hw[750, 960])
        self.assertLess(self.geometry.boundary_signed_distance_m_hw[750, 960], 0.0)

    def test_truth_frame_has_immutable_base_and_boundary_only_points(self) -> None:
        frame = validate_query_factor_frame(self.truth_frame)
        self.assertEqual(tuple(frame["blocks"]), ("SCALE", "SUPPORT", "BOUNDARY"))
        self.assertEqual(frame["blocks"]["SCALE"]["value"]["log_metric_scale"], 0.0)
        self.assertEqual(frame["factor_identity"]["input_depth_array_sha256"], frame["base_geometry"]["faro_depth_array_sha256"])
        boundary_value = frame["blocks"]["BOUNDARY"]["value"]
        self.assertNotIn("surface_points_shape_camera_xyz", boundary_value)
        self.assertEqual(len(boundary_value["point_ids_uv"]), len(boundary_value["boundary_points_shape_camera_xyz"]))
        self.assertLess(len(boundary_value["point_ids_uv"]), len(frame["base_geometry"]["common_point_ids_uv"]))
        self.assertEqual(frame["base_geometry"]["faro_geometry_sha256"], self.geometry.content_sha256)
        for name in runtime_module.FACTOR_NAMES:
            self.assertEqual(
                frame["base_geometry"]["faro_factor_value_sha256s"][name],
                canonical_sha256(frame["blocks"][name]["value"]),
            )

    def test_faro_geometry_is_deeply_read_only_and_whole_hash_bound(self) -> None:
        for name in (
            "intrinsics",
            "depth_m",
            "valid_depth",
            "points_camera_xyz",
            "pixels_uv",
            "support_normal_camera_xyz",
            "support_points_camera_xyz",
            "obstacle_mask_hw",
            "boundary_signed_distance_m_hw",
        ):
            with self.subTest(field=name):
                self.assertFalse(np.asarray(getattr(self.geometry, name)).flags.writeable)
        with self.assertRaises(ValueError):
            self.geometry.depth_m[800, 100] = 0.5
        self.assertEqual(
            self.geometry.content_sha256,
            canonical_sha256(runtime_module._faro_geometry_payload(self.geometry)),
        )

    def test_truth_builder_rejects_resealed_query_not_derived_from_faro_plane(self) -> None:
        forged = copy.deepcopy(self.center_query)
        forged["virtual_query_frame"]["origin_camera_xyz"][1] += 1.0
        forged = manual_reseal(forged)
        with self.assertRaises(AdapterError) as caught:
            build_truth_query_factor_frame(
                self.geometry,
                forged,
                self.model,
                confidence_value=2,
                range_m=1.5,
            )
        self.assertEqual(caught.exception.code, "QUERY_GEOMETRY_BINDING_INVALID")

    def test_candidate_frame_comes_from_real_extractor_not_truth_clone(self) -> None:
        frame = validate_query_factor_frame(self.candidate_frame)
        self.assertEqual(frame["factor_identity"]["origin"], "CANDIDATE_DEPTH_EXTRACTOR")
        self.assertNotEqual(frame["factor_identity"]["input_depth_array_sha256"], self.truth_frame["factor_identity"]["input_depth_array_sha256"])
        self.assertEqual(frame["base_geometry"]["content_sha256"], self.truth_frame["base_geometry"]["content_sha256"])
        self.assertNotEqual(canonical_sha256(frame["blocks"]["BOUNDARY"]["value"]), canonical_sha256(self.truth_frame["blocks"]["BOUNDARY"]["value"]))

    def test_candidate_output_receipt_freezes_model_checkpoint_and_zero_scale(self) -> None:
        identity = self.candidate_frame["factor_identity"]
        output = identity["candidate_output_receipt"]
        self.assertEqual(identity["candidate_output_receipt_sha256"], output["content_sha256"])
        self.assertEqual(output["model_id"], runtime_module.BASELINE_MODEL_ID)
        self.assertEqual(output["checkpoint_sha256"], runtime_module.BASELINE_CHECKPOINT_SHA256)
        self.assertEqual(output["baseline_log_metric_scale"], 0.0)
        self.assertFalse(output["truth_alignment_used"])

        forged = copy.deepcopy(self.candidate_frame)
        forged_output = forged["factor_identity"]["candidate_output_receipt"]
        forged_output["baseline_log_metric_scale"] = 0.1
        forged_output = manual_reseal(forged_output)
        forged["factor_identity"]["candidate_output_receipt"] = forged_output
        forged["factor_identity"]["candidate_output_receipt_sha256"] = forged_output["content_sha256"]
        forged = manual_reseal(forged)
        with self.assertRaises(AdapterError) as caught:
            validate_query_factor_frame(forged)
        self.assertEqual(caught.exception.code, "CANDIDATE_SCALE_INVALID")

        wrong_array = copy.deepcopy(self.candidate_output_receipt)
        wrong_array["output_array_sha256"] = "0" * 64
        wrong_array = manual_reseal(wrong_array)
        with self.assertRaises(AdapterError) as caught:
            build_candidate_query_factor_frame(
                synthetic_faro_depth(False).astype(np.float64) / 1000.0,
                self.matrix,
                self.base_receipt["gravity_up_camera_xyz"],
                self.base_receipt,
                self.center_query,
                self.truth_frame["base_geometry"],
                self.model,
                wrong_array,
                confidence_value=2,
                range_m=1.5,
            )
        self.assertEqual(caught.exception.code, "CANDIDATE_OUTPUT_ARRAY_MISMATCH")

    def test_extreme_candidate_scale_fails_closed_to_unknown(self) -> None:
        forged = copy.deepcopy(self.candidate_frame)
        forged["blocks"]["SCALE"]["value"]["log_metric_scale"] = 1000.0
        forged = manual_reseal(forged)
        result = reduce_query_factor_frame(forged, self.center_query)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertEqual(result["reason_codes"], ["FACTOR_IDENTITY_INVALID"])

    def test_truth_obstacle_is_occupied_and_candidate_without_obstacle_is_clear(self) -> None:
        truth = reduce_query_factor_frame(self.truth_frame, self.center_query)
        candidate = reduce_query_factor_frame(self.candidate_frame, self.center_query)
        self.assertTrue(truth["knownness"]["known"])
        self.assertTrue(candidate["knownness"]["known"])
        self.assertEqual(truth["state"], "OCCUPIED_OBSERVED")
        self.assertEqual(candidate["state"], "CLEAR_OBSERVED")

    def test_boundary_patch_cannot_change_support_knownness_or_base_geometry(self) -> None:
        baseline_result = reduce_query_factor_frame(self.candidate_frame, self.center_query)
        patched = inject_factor_blocks(self.candidate_frame, self.truth_frame, arm="BOUNDARY", mode="FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")
        without_parents = reduce_query_factor_frame(patched, self.center_query)
        self.assertEqual(without_parents["reason_codes"], ["FACTORIAL_PARENT_CONTEXT_REQUIRED"])
        patched_result = reduce_query_factor_frame(
            patched,
            self.center_query,
            factorial_parent_context=(self.candidate_frame, self.truth_frame),
        )
        self.assertEqual(patched["base_geometry"]["content_sha256"], self.candidate_frame["base_geometry"]["content_sha256"])
        self.assertEqual(canonical_sha256(patched["blocks"]["SUPPORT"]), canonical_sha256(self.candidate_frame["blocks"]["SUPPORT"]))
        self.assertEqual(patched_result["knownness"], baseline_result["knownness"])
        self.assertNotEqual(patched_result["state"], baseline_result["state"])

    def test_all_eight_arms_in_both_modes_patch_only_named_blocks(self) -> None:
        for mode in ORACLE_MODES:
            for arm in ARMS:
                with self.subTest(mode=mode, arm=arm):
                    output = inject_factor_blocks(self.candidate_frame, self.truth_frame, arm=arm, mode=mode)
                    validate_query_factor_frame(output)
                    selected = set(output["factor_identity"]["patched_factors"])
                    expected = set() if arm == "NONE" else set(arm.split("_"))
                    self.assertEqual(selected, expected)
                    for name in ("SCALE", "SUPPORT", "BOUNDARY"):
                        if name not in expected:
                            self.assertEqual(canonical_sha256(output["blocks"][name]), canonical_sha256(self.candidate_frame["blocks"][name]))
                        elif mode == "VALUE_ONLY_COMMON_SUPPORT":
                            self.assertEqual(output["blocks"][name]["validity"], self.candidate_frame["blocks"][name]["validity"])
                            self.assertEqual(output["blocks"][name]["uncertainty"], self.candidate_frame["blocks"][name]["uncertainty"])
                        else:
                            self.assertEqual(canonical_sha256(output["blocks"][name]), canonical_sha256(self.truth_frame["blocks"][name]))

    def test_injection_rejects_base_geometry_drift_even_when_self_consistent(self) -> None:
        oracle = copy.deepcopy(self.truth_frame)
        oracle["base_geometry"]["local_valid_fraction"] -= 0.01
        oracle["base_geometry"] = manual_reseal(oracle["base_geometry"])
        for name in runtime_module.FACTOR_NAMES:
            oracle["blocks"][name]["base_geometry_sha256"] = oracle["base_geometry"]["content_sha256"]
        oracle["blocks"]["BOUNDARY"]["validity"]["local_valid_fraction"] = oracle["base_geometry"]["local_valid_fraction"]
        oracle["blocks"]["BOUNDARY"]["validity"]["valid"] = oracle["base_geometry"]["local_valid_fraction"] >= runtime_module.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
        oracle = manual_reseal(oracle)
        validate_query_factor_frame(oracle)
        with self.assertRaises(AdapterError) as caught:
            inject_factor_blocks(self.candidate_frame, oracle, arm="BOUNDARY", mode="VALUE_ONLY_COMMON_SUPPORT")
        self.assertEqual(caught.exception.code, "BASE_GEOMETRY_MISMATCH")

    def test_hash_valid_nested_boundary_extension_is_rejected(self) -> None:
        frame = copy.deepcopy(self.truth_frame)
        frame["blocks"]["BOUNDARY"]["validity"]["oracle_marker"] = True
        frame = manual_reseal(frame)
        with self.assertRaises(AdapterError) as caught:
            validate_query_factor_frame(frame)
        self.assertEqual(caught.exception.code, "BOUNDARY_VALIDITY_INVALID")

    def test_factorial_component_lineage_tampering_is_rejected(self) -> None:
        frame = inject_factor_blocks(
            self.candidate_frame,
            self.truth_frame,
            arm="BOUNDARY",
            mode="VALUE_ONLY_COMMON_SUPPORT",
        )
        frame["factor_identity"]["block_parent_lineage"]["BOUNDARY"]["value"]["sha256"] = "0" * 64
        frame = manual_reseal(frame)
        with self.assertRaises(AdapterError) as caught:
            validate_query_factor_frame(frame)
        self.assertEqual(caught.exception.code, "FACTOR_LINEAGE_INVALID")

    def test_boundary_point_outside_base_geometry_is_rejected(self) -> None:
        frame = copy.deepcopy(self.truth_frame)
        ids = np.asarray(frame["blocks"]["BOUNDARY"]["value"]["point_ids_uv"]).copy()
        self.assertGreater(len(ids), 0)
        ids[0] = [0, 0]
        frame["blocks"]["BOUNDARY"]["value"]["point_ids_uv"] = ids
        frame = manual_reseal(frame)
        with self.assertRaises(AdapterError) as caught:
            validate_query_factor_frame(frame)
        self.assertEqual(caught.exception.code, "BOUNDARY_OUTSIDE_BASE_GEOMETRY")

    def test_factor_frame_tampering_is_detected(self) -> None:
        frame = copy.deepcopy(self.truth_frame)
        frame["blocks"]["SCALE"]["value"]["log_metric_scale"] = 0.1
        with self.assertRaises(AdapterError) as caught:
            validate_query_factor_frame(frame)
        self.assertEqual(caught.exception.code, "FACTOR_FRAME_HASH_MISMATCH")

    def test_candidate_extractor_rejects_adapter_fit_receipt(self) -> None:
        fit_receipt = source_receipt(role="ADAPTER_FIT")
        with self.assertRaises(AdapterError) as caught:
            build_candidate_query_factor_frame(
                synthetic_faro_depth(False).astype(np.float64) / 1000.0,
                np.asarray(fit_receipt["intrinsics_highres"]["matrix_3x3"]),
                fit_receipt["gravity_up_camera_xyz"],
                fit_receipt,
                self.center_query,
                self.truth_frame["base_geometry"],
                self.model,
                {},
                confidence_value=2,
                range_m=1.5,
            )
        self.assertEqual(caught.exception.code, "CANDIDATE_SOURCE_ROLE_INVALID")

    def test_complete_bundle_requires_exactly_nine_genuine_truth_frames(self) -> None:
        frames = [
            build_truth_query_factor_frame(self.geometry, receipt, self.model, confidence_value=2, range_m=1.5)
            for receipt in self.query_receipts
        ]
        bundle = reduce_complete_query_bundle(frames, self.query_receipts, self.base_receipt)
        self.assertEqual(bundle["query_receipts"], 9)
        self.assertEqual(bundle["factor_frames"], 9)
        self.assertEqual(sum(bundle["state_counts"].values()), 9)
        with self.assertRaises(AdapterError) as caught:
            reduce_complete_query_bundle(frames[:-1], self.query_receipts, self.base_receipt)
        self.assertEqual(caught.exception.code, "QUERY_BUNDLE_CARDINALITY")
        candidate_masquerade = list(frames)
        candidate_masquerade[4] = self.candidate_frame
        with self.assertRaises(AdapterError) as caught:
            reduce_complete_query_bundle(candidate_masquerade, self.query_receipts, self.base_receipt)
        self.assertEqual(caught.exception.code, "QUERY_BUNDLE_TRUTH_ORIGIN_REQUIRED")

    def test_replay_is_byte_canonical(self) -> None:
        first = reduce_query_factor_frame(self.truth_frame, self.center_query)
        second = reduce_query_factor_frame(copy.deepcopy(self.truth_frame), copy.deepcopy(self.center_query))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))

    def test_no_public_reseal_or_self_signed_truth_correction_capability(self) -> None:
        self.assertFalse(hasattr(runtime_module, "reseal_query_factor_frame"))
        self.assertFalse(hasattr(runtime_module, "candidate_relative_log_scale_correction"))
        self.assertNotIn("reseal_query_factor_frame", runtime_module.__all__)
        self.assertNotIn("candidate_relative_log_scale_correction", runtime_module.__all__)
        self.assertIn("build_candidate_depth_output_receipt", runtime_module.__all__)
        self.assertIn("CANDIDATE_OUTPUT_RECEIPT_SCHEMA", runtime_module.__all__)

    def test_runtime_has_no_source_io_model_runner_or_legacy_reducer_import(self) -> None:
        source = inspect.getsource(runtime_module)
        for forbidden in (
            "from pathlib",
            "import pathlib",
            "requests",
            "urllib",
            "load_manifest_frame",
            "derive_assistive_truth",
            "geometry_r2_reducer",
            "open(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_nearest_faro_sampling_uses_frozen_pixel_centers(self) -> None:
        faro = np.zeros((1440, 1920), dtype=np.uint16)
        faro[3, 3] = 111
        faro[1436, 1916] = 222
        sampled = sample_faro_at_apple_centers(faro)
        self.assertEqual(int(sampled[0, 0]), 111)
        self.assertEqual(int(sampled[-1, -1]), 222)


if __name__ == "__main__":
    unittest.main()
