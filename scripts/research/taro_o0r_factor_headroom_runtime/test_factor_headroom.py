#!/usr/bin/env python3
"""Synthetic focused tests for TARO candidate-gauge oracle mechanics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime import factor_headroom as runtime
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime import test_source_adapter as fixtures


def _blocks() -> dict[str, object]:
    common = {
        "physical_frame_id": "frame",
        "query_id": "query",
        "source_frame_receipt_sha256": "1" * 64,
        "query_receipt_sha256": "2" * 64,
        "base_geometry_sha256": "3" * 64,
    }
    return {
        "SCALE": {
            **common,
            "factor_name": "SCALE",
            "value": {"log_metric_scale": 0.0, "value_kind": "ABSOLUTE_FARO_METRIC_REFERENCE"},
            "validity": {"valid": True, "model_independent": True},
            "uncertainty": {"valid": True, "q95_log": 0.1, "resolution_scope": "GLOBAL", "fit_model_sha256": "4" * 64},
        },
        "SUPPORT": {
            **common,
            "factor_name": "SUPPORT",
            "value": {"normal_camera_xyz": [0.0, 1.0, 0.0], "camera_height_shape_m": 1.6},
            "validity": {"valid": True, "support_point_count": 300, "query_support_points": 200, "observed_forward_shape_m": 2.4, "support_fraction": 0.5, "slope_degrees": 0.0, "median_residual_m": 0.04},
            "uncertainty": {"valid": True, "height_q95_shape_m": 0.08, "normal_q95_rad": 0.02, "height_resolution_scope": "GLOBAL", "normal_resolution_scope": "GLOBAL", "bootstrap_seed_first_64_bits": 1, "fit_model_sha256": "4" * 64},
        },
        "BOUNDARY": {
            **common,
            "factor_name": "BOUNDARY",
            "value": {"point_ids_uv": [[10, 20], [11, 20]], "boundary_points_shape_camera_xyz": [[0.2, 0.3, 1.0], [0.4, 0.5, 1.2]]},
            "validity": {"valid": True, "common_support_point_count": 300, "local_valid_fraction": 1.0},
            "uncertainty": {"valid": True, "localization_q95_shape_m": 0.06, "resolution_scope": "GLOBAL", "fit_model_sha256": "4" * 64},
        },
    }


class FactorHeadroomTests(unittest.TestCase):
    def test_executor_owned_gauge_oracle_binds_real_extractor_parents(self) -> None:
        source = fixtures.source_receipt()
        faro = fixtures.synthetic_faro_depth(True)
        geometry = adapter.derive_faro_geometry(
            faro,
            np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            source["gravity_up_camera_xyz"],
            source,
        )
        query = adapter.build_query_receipts(source, geometry)[4]
        model = fixtures.fitted_model(1)
        truth = adapter.build_truth_query_factor_frame(geometry, query, model, confidence_value=2, range_m=1.0)
        candidate = np.ascontiguousarray(faro.astype(np.float32) / 2000.0)
        output = adapter.build_candidate_depth_output_receipt(candidate, source, inference_receipt_sha256="A" * 64)
        baseline = adapter.build_candidate_query_factor_frame(
            candidate,
            np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64),
            source["gravity_up_camera_xyz"],
            source,
            query,
            truth["base_geometry"],
            model,
            output,
            confidence_value=2,
            range_m=1.0,
        )
        gauge, scale = runtime.build_candidate_gauge_oracle_frame(faro, candidate, truth, baseline)
        self.assertEqual(gauge["factor_identity"]["origin"], runtime.CANDIDATE_GAUGE_ORIGIN)
        self.assertEqual(
            gauge["factor_identity"]["candidate_output_receipt"]["output_array_sha256"],
            gauge["factor_identity"]["candidate_depth_array_sha256"],
        )
        self.assertEqual(
            gauge["factor_identity"]["candidate_output_receipt_sha256"],
            gauge["factor_identity"]["candidate_output_receipt"]["content_sha256"],
        )
        self.assertEqual(gauge["factor_identity"]["faro_geometry_sha256"], truth["factor_identity"]["faro_geometry_sha256"])
        self.assertAlmostEqual(scale["metric_scale"], 2.0, places=6)
        metric_scale = scale["metric_scale"]
        self.assertAlmostEqual(
            metric_scale * gauge["blocks"]["SUPPORT"]["value"]["camera_height_shape_m"],
            truth["blocks"]["SUPPORT"]["value"]["camera_height_shape_m"],
            places=9,
        )
        np.testing.assert_allclose(
            metric_scale * gauge["blocks"]["BOUNDARY"]["value"]["boundary_points_shape_camera_xyz"],
            truth["blocks"]["BOUNDARY"]["value"]["boundary_points_shape_camera_xyz"],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_scale_is_robust_median_on_frozen_common_support(self) -> None:
        faro = np.full(adapter.HIGHRES_SHAPE_HW, 2000, dtype=np.uint16)
        candidate = np.full(adapter.HIGHRES_SHAPE_HW, 1.0, dtype=np.float32)
        ids = np.stack((np.arange(300, dtype=np.int32), np.full(300, 100, dtype=np.int32)), axis=1)
        candidate[100, 0] = 5.0
        record = runtime.derive_candidate_relative_scale(
            faro,
            candidate,
            ids,
            physical_frame_id="frame",
            query_id="query",
            faro_factor_frame_sha256="A" * 64,
            candidate_factor_frame_sha256="B" * 64,
        )
        self.assertAlmostEqual(math.log(2.0), record["log_metric_scale"], places=12)
        self.assertAlmostEqual(2.0, record["metric_scale"], places=12)
        self.assertEqual(300, record["valid_pair_count"])
        self.assertFalse(record["truth_alignment_used_for_candidate_generation"])

    def test_scale_arms_use_candidate_gauge_only(self) -> None:
        for arm in adapter.ARMS:
            expected = "CANDIDATE_GAUGE" if "SCALE" in (() if arm == "NONE" else arm.split("_")) else "ABSOLUTE_METRIC"
            self.assertEqual(expected, runtime.oracle_representation_for_arm(arm), arm)

    def test_gauge_reexpression_prevents_double_scaling(self) -> None:
        absolute = _blocks()
        scale = math.log(1.25)
        record = {
            "schema": runtime.CANDIDATE_RELATIVE_SCALE_SCHEMA,
            "value_kind": runtime.SCALE_VALUE_KIND,
            "log_metric_scale": scale,
            "metric_scale": math.exp(scale),
        }
        gauge = runtime.reexpress_faro_blocks_in_candidate_gauge(absolute, record)
        metric_scale = math.exp(gauge["SCALE"]["value"]["log_metric_scale"])
        self.assertAlmostEqual(1.6, metric_scale * gauge["SUPPORT"]["value"]["camera_height_shape_m"], places=12)
        np.testing.assert_allclose(
            np.asarray(absolute["BOUNDARY"]["value"]["boundary_points_shape_camera_xyz"]),
            metric_scale * np.asarray(gauge["BOUNDARY"]["value"]["boundary_points_shape_camera_xyz"]),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(0.08, metric_scale * gauge["SUPPORT"]["uncertainty"]["height_q95_shape_m"], places=12)
        self.assertAlmostEqual(0.06, metric_scale * gauge["BOUNDARY"]["uncertainty"]["localization_q95_shape_m"], places=12)
        self.assertFalse(gauge["SCALE"]["validity"]["model_independent"])

    def test_gauge_reexpression_preserves_dense_point_id_array(self) -> None:
        absolute = _blocks()
        absolute["BOUNDARY"]["value"]["point_ids_uv"] = np.asarray([[10, 20], [11, 20]], dtype=np.int32)
        scale = math.log(1.25)
        record = {
            "schema": runtime.CANDIDATE_RELATIVE_SCALE_SCHEMA,
            "value_kind": runtime.SCALE_VALUE_KIND,
            "log_metric_scale": scale,
            "metric_scale": math.exp(scale),
        }
        gauge = runtime.reexpress_faro_blocks_in_candidate_gauge(absolute, record)
        self.assertIsInstance(gauge["BOUNDARY"]["value"]["point_ids_uv"], np.ndarray)
        self.assertEqual(gauge["BOUNDARY"]["value"]["point_ids_uv"].dtype, np.int32)

    def test_insufficient_candidate_support_fails_closed(self) -> None:
        faro = np.full(adapter.HIGHRES_SHAPE_HW, 2000, dtype=np.uint16)
        candidate = np.zeros(adapter.HIGHRES_SHAPE_HW, dtype=np.float32)
        ids = np.stack((np.arange(300, dtype=np.int32), np.full(300, 100, dtype=np.int32)), axis=1)
        with self.assertRaises(runtime.FactorHeadroomError) as raised:
            runtime.derive_candidate_relative_scale(
                faro,
                candidate,
                ids,
                physical_frame_id="frame",
                query_id="query",
                faro_factor_frame_sha256="A" * 64,
                candidate_factor_frame_sha256="B" * 64,
            )
        self.assertEqual("SCALE_COMMON_SUPPORT_INSUFFICIENT", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
