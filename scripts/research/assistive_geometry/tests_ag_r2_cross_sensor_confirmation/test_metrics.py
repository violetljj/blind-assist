from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    ContractError,
    load_frozen_contracts,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.metrics import (
    PARENT_IDS,
    score,
    score_or_not_evaluable,
)


def _ratios(size: int) -> np.ndarray:
    pattern = np.asarray([0.50] * 68 + [1.50] * 27 + [2.50] * 5, dtype=np.float64)
    return np.resize(pattern, size)


def synthetic_inputs() -> tuple[dict, dict, list[dict]]:
    protocol, _ = load_frozen_contracts()
    source = {
        "parents": [
            {
                "parent_id": parent,
                "eligible_pair_count": 24,
                "calibration_count": 12,
                "score_count": 12,
                "camera_height_m": 1.5,
                "camera_height_mad_m": 0.02,
                "source_depth_known_coverage": 1.0,
                "source_support_known_coverage": 1.0,
                "source_boundary_known_coverage": 1.0,
            }
            for parent in PARENT_IDS
        ]
    }
    frames = []
    shape = (10, 10)
    index = np.arange(100, dtype=np.float64).reshape(shape)
    ratio = _ratios(100).reshape(shape)
    for parent in PARENT_IDS:
        for frame_index in range(12):
            phase = (index + frame_index) % 100.0
            depth_sigma = 0.015 + 0.0005 * phase
            depth_residual = depth_sigma * ratio
            support_sigma = 0.010 + 0.0003 * phase
            support_residual = support_sigma * ratio
            boundary_sigma_rad = 0.001 + 0.00004 * phase
            boundary_residual_rad = boundary_sigma_rad * ratio
            focal = 500.0
            boundary_delta_px = np.tan(boundary_residual_rad) * focal
            known = np.ones(shape, dtype=bool)
            frames.append(
                {
                    "parent_id": parent,
                    "frame_id": f"{parent}-{frame_index:02d}",
                    "fx": float(focal),
                    "fy": float(focal),
                    "prediction": {
                        "depth_m": np.exp(depth_residual).astype(np.float64),
                        "depth_log_sigma": depth_sigma.astype(np.float64),
                        "depth_known": known.copy(),
                        "support_probability": np.full(shape, 0.75, dtype=np.float64),
                        "support_residual_sigma_m": support_sigma.astype(np.float64),
                        "support_known": known.copy(),
                        "obstacle_probability": np.full(shape, 0.25, dtype=np.float64),
                        "boundary_distance_px": (5.0 + boundary_delta_px).astype(np.float64),
                        "boundary_sigma_px": (np.tan(boundary_sigma_rad) * focal).astype(np.float64),
                        "evidence_known": known.copy(),
                    },
                    "truth": {
                        "depth_m": np.ones(shape, dtype=np.float64),
                        "depth_known": known.copy(),
                        "support_probability": np.full(shape, 0.75, dtype=np.float64),
                        "support_signed_residual_m": support_residual.astype(np.float64),
                        "support_known": known.copy(),
                        "obstacle_probability": np.full(shape, 0.25, dtype=np.float64),
                        "boundary_distance_px": np.full(shape, 5.0, dtype=np.float64),
                        "evidence_known": known.copy(),
                    },
                }
            )
    return protocol, source, frames


def test_frozen_metrics_pass_synthetic_absolute_gates() -> None:
    protocol, source, frames = synthetic_inputs()
    result = score(protocol, source, frames)
    assert result["terminal"] == "CONFIRM_PASS"
    assert len(result["gates"]) == 27
    assert all(row["passed"] for row in result["gates"])
    assert result["metrics"]["confirmation_parent_count"] == 3
    assert result["spearman_method"]["strata"] == 10
    assert result["uncertainty"]["families"]["support"]["parent_macro_one_sigma_coverage"] == pytest.approx(0.68)


def test_unknown_must_be_nan_and_boolean_integer_confusion_fails() -> None:
    protocol, source, frames = synthetic_inputs()
    frames[0]["prediction"]["depth_known"][0, 0] = False
    with pytest.raises(ContractError, match="F2_PREDICTION_DEPTH_M_UNKNOWN_NOT_NAN"):
        score(protocol, source, frames)
    protocol, source, frames = synthetic_inputs()
    frames[0]["truth"]["support_known"] = frames[0]["truth"]["support_known"].astype(np.uint8)
    with pytest.raises(ContractError, match="F2_TRUTH_SUPPORT_KNOWN_DTYPE"):
        score(protocol, source, frames)


def test_zero_required_truth_denominator_is_not_evaluable_not_negative() -> None:
    protocol, source, frames = synthetic_inputs()
    for frame in frames:
        frame["truth"]["support_known"][:] = False
        frame["truth"]["support_probability"][:] = np.nan
        frame["truth"]["support_signed_residual_m"][:] = np.nan
    result = score_or_not_evaluable(protocol, source, frames)
    assert result["terminal"] == "NOT_EVALUABLE"
    assert "DENOMINATOR" in result["reason_code"]


def test_source_gate_failure_maps_to_not_evaluable_and_model_gate_to_fail() -> None:
    protocol, source, frames = synthetic_inputs()
    source_failure = deepcopy(source)
    source_failure["parents"][0]["source_depth_known_coverage"] = 0.49
    assert score(protocol, source_failure, frames)["terminal"] == "NOT_EVALUABLE"
    model_failure = deepcopy(frames)
    for frame in model_failure:
        frame["prediction"]["support_probability"][:] = 0.0
    assert score(protocol, source, model_failure)["terminal"] == "CONFIRM_FAIL"


def test_support_uncertainty_uses_signed_plane_residual_not_probability_error() -> None:
    protocol, source, frames = synthetic_inputs()
    baseline = score(protocol, source, frames)
    changed = deepcopy(frames)
    for frame in changed:
        frame["truth"]["support_signed_residual_m"] *= 100.0
    result = score(protocol, source, changed)
    assert (
        result["uncertainty"]["families"]["support"]["parent_macro_one_sigma_coverage"]
        < baseline["uncertainty"]["families"]["support"]["parent_macro_one_sigma_coverage"]
    )
    assert result["parents"][PARENT_IDS[0]]["support_brier"] == 0.0


def test_undefined_uncertainty_ordering_is_not_evaluable() -> None:
    protocol, source, frames = synthetic_inputs()
    for frame in frames:
        known = frame["prediction"]["depth_known"]
        frame["prediction"]["depth_log_sigma"][known] = 0.1
    result = score_or_not_evaluable(protocol, source, frames)
    assert result["terminal"] == "NOT_EVALUABLE"
    assert "SPEARMAN_UNDEFINED" in result["reason_code"]
