#!/usr/bin/env python3
"""FactorTensor adapter v2 with separate metric-scale and local-shape sigma.

V1 used one per-pixel depth sigma both as global metric-scale uncertainty and
as local obstacle-shape uncertainty.  Real-source evidence showed that this
coupling can force otherwise valid frames to UNKNOWN.  V2 keeps the unchanged
F0 reducer output contract while separating the two factor uncertainties.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

try:
    from . import factor_tensor_adapter as v1
except ImportError:  # pragma: no cover - direct script/import compatibility
    import factor_tensor_adapter as v1


INPUT_SCHEMA = "blindassist_assistive_geometry_r2_factortensor_adapter_input_v2"
PREDICTION_SCHEMA = "blindassist_assistive_geometry_r2_factor_prediction_v2"
CALIBRATION_SCHEMA = "blindassist_assistive_geometry_r2_adapter_calibration_receipt_v2"
OUTPUT_SCHEMA = v1.OUTPUT_SCHEMA
GEOMETRY_SCHEMA = v1.GEOMETRY_SCHEMA
AdapterError = v1.AdapterError
canonical_json_bytes = v1.canonical_json_bytes
canonical_sha256 = v1.canonical_sha256

DEPTH_KEYS = {
    "depth_shape_positive_hw",
    "log_metric_scale_m_scalar",
    "metric_scale_log_sigma_scalar",
    "depth_shape_log_sigma_hw",
    "depth_valid_probability_hw",
    "metric_scale_valid",
}
SCHEMA_CONTRACT = {
    "input_schema": INPUT_SCHEMA,
    "prediction_schema": PREDICTION_SCHEMA,
    "calibration_schema": CALIBRATION_SCHEMA,
    "output_schema": OUTPUT_SCHEMA,
    "depth_keys": sorted(DEPTH_KEYS),
    "uncertainty_semantics": {
        "metric_scale_log_sigma_scalar": "log relative one-sigma for global metric scale",
        "depth_shape_log_sigma_hw": "log absolute one-sigma metres for local depth shape",
    },
    "reducer_unchanged": True,
}
FACTOR_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        SCHEMA_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest().upper()


def adapt_factor_tensor(adapter_input: dict[str, Any]) -> dict[str, Any]:
    """Adapt one v2 factor tensor into the unchanged F0 factor frame."""

    v1._scan_forbidden(adapter_input)
    root = v1._exact_keys(adapter_input, v1.TOP_LEVEL_KEYS, "INPUT_KEY_SET")
    prediction = v1._exact_keys(
        root["prediction"], v1.PREDICTION_KEYS, "PREDICTION_KEY_SET"
    )
    calibration = v1._exact_keys(
        root["calibration_receipt"], v1.CALIBRATION_KEYS, "CALIBRATION_KEY_SET"
    )
    depth = v1._exact_keys(prediction["depth_scale"], DEPTH_KEYS, "DEPTH_KEY_SET_V2")
    v1.require(
        prediction.get("schema") == PREDICTION_SCHEMA,
        "PREDICTION_SCHEMA_V2",
        "prediction schema drift",
    )
    v1.require(
        calibration.get("schema") == CALIBRATION_SCHEMA,
        "CALIBRATION_SCHEMA_V2",
        "calibration schema drift",
    )
    v1.require(
        calibration.get("factor_schema_sha256") == FACTOR_SCHEMA_SHA256,
        "FACTOR_SCHEMA_V2",
        "factor schema receipt drift",
    )
    identity = prediction.get("factor_identity")
    v1.require(
        isinstance(identity, dict)
        and identity.get("factor_schema_sha256") == FACTOR_SCHEMA_SHA256,
        "FACTOR_IDENTITY_V2",
        "factor identity schema drift",
    )
    scale_log_sigma = depth.get("metric_scale_log_sigma_scalar")
    v1.require(
        v1.finite(scale_log_sigma),
        "METRIC_SCALE_LOG_SIGMA",
        "finite metric scale log sigma required",
    )
    relative_scale_sigma = math.exp(float(scale_log_sigma))
    v1.require(
        math.isfinite(relative_scale_sigma) and relative_scale_sigma >= 0.0,
        "METRIC_SCALE_SIGMA",
        "metric scale sigma invalid",
    )

    mapped = copy.deepcopy(adapter_input)
    mapped_prediction = mapped["prediction"]
    mapped_prediction["schema"] = v1.PREDICTION_SCHEMA
    mapped_depth = mapped_prediction["depth_scale"]
    mapped_depth.pop("metric_scale_log_sigma_scalar")
    mapped_depth["depth_log_sigma_hw"] = mapped_depth.pop(
        "depth_shape_log_sigma_hw"
    )
    mapped_calibration = mapped["calibration_receipt"]
    mapped_calibration["schema"] = v1.CALIBRATION_SCHEMA
    mapped_calibration["factor_schema_sha256"] = v1.FACTOR_SCHEMA_SHA256
    frame = v1.adapt_factor_tensor(mapped)
    if frame["depth_scale"]["valid"]:
        floor = float(calibration["scale_relative_sigma_floor"])
        cap = float(calibration["scale_relative_sigma_cap"])
        relative_scale_sigma = max(floor, min(cap, relative_scale_sigma))
        frame["depth_scale"]["scale_sigma_m"] = (
            float(frame["depth_scale"]["scale_m"]) * relative_scale_sigma
        )
    return v1._round_canonical(frame)


__all__ = [
    "AdapterError",
    "CALIBRATION_SCHEMA",
    "FACTOR_SCHEMA_SHA256",
    "GEOMETRY_SCHEMA",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "PREDICTION_SCHEMA",
    "SCHEMA_CONTRACT",
    "adapt_factor_tensor",
    "canonical_json_bytes",
    "canonical_sha256",
]
