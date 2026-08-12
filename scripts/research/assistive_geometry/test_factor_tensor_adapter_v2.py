#!/usr/bin/env python3
"""Focused invariants for separated scale and shape uncertainty."""

from __future__ import annotations

import copy
import math
import unittest

from scripts.research.assistive_geometry.factor_tensor_adapter_v2 import (
    CALIBRATION_SCHEMA,
    FACTOR_SCHEMA_SHA256,
    PREDICTION_SCHEMA,
    adapt_factor_tensor,
)
from scripts.research.assistive_geometry.test_factor_tensor_adapter import case_input


def v2_input() -> dict:
    value, _ = case_input("nominal_landscape_single_component")
    value = copy.deepcopy(value)
    value["prediction"]["schema"] = PREDICTION_SCHEMA
    value["prediction"]["factor_identity"][
        "factor_schema_sha256"
    ] = FACTOR_SCHEMA_SHA256
    depth = value["prediction"]["depth_scale"]
    depth["depth_shape_log_sigma_hw"] = depth.pop("depth_log_sigma_hw")
    depth["metric_scale_log_sigma_scalar"] = math.log(0.08)
    calibration = value["calibration_receipt"]
    calibration["schema"] = CALIBRATION_SCHEMA
    calibration["factor_schema_sha256"] = FACTOR_SCHEMA_SHA256
    calibration["scale_relative_sigma_floor"] = 0.02
    return value


class FactorTensorAdapterV2Tests(unittest.TestCase):
    def test_scale_sigma_is_independent_of_shape_sigma(self) -> None:
        nominal = v2_input()
        high_shape = copy.deepcopy(nominal)
        high_shape["prediction"]["depth_scale"]["depth_shape_log_sigma_hw"] = [
            [math.log(1.0)] * 4 for _ in range(3)
        ]
        first = adapt_factor_tensor(nominal)
        second = adapt_factor_tensor(high_shape)
        self.assertEqual(
            first["depth_scale"]["scale_sigma_m"],
            second["depth_scale"]["scale_sigma_m"],
        )
        self.assertGreater(
            second["boundary"]["obstacles"][0]["depth_shape_sigma"],
            first["boundary"]["obstacles"][0]["depth_shape_sigma"],
        )

    def test_metric_scale_sigma_changes_only_global_scale_interval(self) -> None:
        nominal = v2_input()
        high_scale = copy.deepcopy(nominal)
        high_scale["prediction"]["depth_scale"][
            "metric_scale_log_sigma_scalar"
        ] = math.log(0.40)
        first = adapt_factor_tensor(nominal)
        second = adapt_factor_tensor(high_scale)
        self.assertGreater(
            second["depth_scale"]["scale_sigma_m"],
            first["depth_scale"]["scale_sigma_m"],
        )
        self.assertEqual(
            first["boundary"]["obstacles"][0]["depth_shape_sigma"],
            second["boundary"]["obstacles"][0]["depth_shape_sigma"],
        )

    def test_scale_floor_remains_monotone(self) -> None:
        value = v2_input()
        value["prediction"]["depth_scale"][
            "metric_scale_log_sigma_scalar"
        ] = math.log(0.001)
        value["calibration_receipt"]["scale_relative_sigma_floor"] = 0.10
        frame = adapt_factor_tensor(value)
        self.assertAlmostEqual(
            frame["depth_scale"]["scale_sigma_m"]
            / frame["depth_scale"]["scale_m"],
            0.10,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
