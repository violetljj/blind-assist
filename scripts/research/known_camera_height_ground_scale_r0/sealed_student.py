"""Immutable loader and predictor for the sealed camera-conditioned scale student."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "blindassist_camera_conditioned_scale_student_sealed_model_v1"
MODEL_ID = "CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P"
FEATURE_COUNT = 10


@dataclass(frozen=True)
class SealedScaleStudent:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    weights: np.ndarray
    scale_range: tuple[float, float]

    @classmethod
    def load(cls, receipt_path: Path) -> "SealedScaleStudent":
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("model_id") != MODEL_ID:
            raise ValueError("unexpected sealed student identity")
        if receipt.get("status") != "SEALED_NO_REFIT_ENTRYPOINT":
            raise ValueError("student receipt is not sealed")
        operator = receipt.get("operator", {})
        if operator.get("family") != "closed_form_log_scale_ridge":
            raise ValueError("unexpected student family")
        if float(operator.get("ridge_alpha")) != 1.0:
            raise ValueError("unexpected frozen ridge alpha")
        names = tuple(operator.get("feature_names", ()))
        mean = np.asarray(operator.get("feature_mean"), dtype=np.float64)
        std = np.asarray(operator.get("feature_standard_deviation"), dtype=np.float64)
        weights = np.asarray(operator.get("weights_intercept_then_features"), dtype=np.float64)
        scale_range = tuple(float(value) for value in operator.get("scale_admission_range", ()))
        if len(names) != FEATURE_COUNT or mean.shape != (FEATURE_COUNT,):
            raise ValueError("invalid sealed feature contract")
        if std.shape != mean.shape or weights.shape != (FEATURE_COUNT + 1,):
            raise ValueError("invalid sealed parameter shape")
        if len(set(names)) != FEATURE_COUNT:
            raise ValueError("sealed feature names must be unique")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("non-finite sealed feature statistics")
        if not np.all(np.isfinite(weights)) or np.any(std <= 0.0):
            raise ValueError("invalid sealed weights or standard deviations")
        if len(scale_range) != 2 or not 0.0 < scale_range[0] < scale_range[1]:
            raise ValueError("invalid sealed scale range")
        return cls(names, mean, std, weights, (scale_range[0], scale_range[1]))

    def predict(self, features: np.ndarray | list[float]) -> dict[str, Any]:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
            return {"status": "UNKNOWN", "reason": "INVALID_RUNTIME_FEATURES"}
        standardized = (values - self.feature_mean) / self.feature_standard_deviation
        log_scale = float(self.weights[0] + standardized @ self.weights[1:])
        scale = float(np.exp(log_scale))
        if not np.isfinite(scale) or not self.scale_range[0] <= scale <= self.scale_range[1]:
            return {
                "status": "UNKNOWN",
                "reason": "STUDENT_SCALE_OUT_OF_RANGE",
                "log_scale": log_scale,
                "scale": scale,
            }
        return {"status": "VALID", "log_scale": log_scale, "scale": scale}


def validate_golden(receipt_path: Path, golden_path: Path) -> None:
    student = SealedScaleStudent.load(receipt_path)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if golden.get("model_id") != MODEL_ID:
        raise ValueError("golden/model identity mismatch")
    tolerance = float(golden["absolute_tolerance"])
    for vector in golden["vectors"]:
        result = student.predict(vector["features"])
        if result.get("status") != "VALID":
            raise ValueError(f"golden vector rejected: {vector['id']}")
        if abs(result["log_scale"] - float(vector["expected_log_scale"])) > tolerance:
            raise ValueError(f"golden log scale mismatch: {vector['id']}")
        if abs(result["scale"] - float(vector["expected_scale"])) > tolerance:
            raise ValueError(f"golden scale mismatch: {vector['id']}")
