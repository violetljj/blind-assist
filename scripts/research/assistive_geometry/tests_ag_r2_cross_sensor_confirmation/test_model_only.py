from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation import (
    model_only,
)


def _raw_prediction(height_m: float = 1.0) -> dict:
    shape = (64, 64)
    fx = fy = 100.0
    cx = cy = 31.5
    rows, _ = np.indices(shape, dtype=np.float64)
    depth = np.full(shape, np.nan, dtype=np.float64)
    known = rows >= 40
    depth[known] = height_m * fy / (rows[known] - cy)
    support = np.full(shape, np.nan, dtype=np.float64)
    support[known] = 0.9
    sigma = np.full(shape, np.nan, dtype=np.float64)
    sigma[known] = 0.1
    obstacle = np.full(shape, np.nan, dtype=np.float64)
    obstacle[known] = 0.0
    boundary = np.full(shape, np.nan, dtype=np.float64)
    boundary[known] = 32.0
    return {
        "parent_id": "plant_scene_2",
        "frame_id": "frame",
        "source_hw": [64, 64],
        "output_hw": [64, 64],
        "intrinsics": np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        "depth_m": depth,
        "depth_log_sigma": sigma.copy(),
        "depth_known": known.copy(),
        "support_probability": support,
        "support_residual_sigma_m": sigma.copy(),
        "support_known": known.copy(),
        "obstacle_probability": obstacle,
        "boundary_distance_px": boundary,
        "boundary_sigma_px": sigma.copy(),
        "evidence_known": known.copy(),
    }


def test_conditioner_uses_one_session_height_and_preserves_learned_sigmas() -> None:
    raw = [_raw_prediction() | {"frame_id": f"frame-{index:02d}"} for index in range(12)]
    context = {
        "parent_id": "plant_scene_2",
        "camera_height_m": 1.5,
        "camera_height_mad_m": 0.01,
        "gravity_up_camera_xyz": [0.0, -1.0, 0.0],
    }
    conditioned, receipt = model_only.condition_parent_predictions(raw, context)
    assert receipt["session_scale_factor"] == 1.5
    assert receipt["score_truth_used"] is False
    known = raw[0]["depth_known"]
    assert np.allclose(conditioned[0]["depth_m"][known], raw[0]["depth_m"][known] * 1.5)
    assert np.array_equal(conditioned[0]["depth_log_sigma"], raw[0]["depth_log_sigma"], equal_nan=True)
    assert np.array_equal(conditioned[0]["support_residual_sigma_m"], raw[0]["support_residual_sigma_m"], equal_nan=True)
    assert np.array_equal(conditioned[0]["boundary_sigma_px"], raw[0]["boundary_sigma_px"], equal_nan=True)


def test_rgbk_predictor_public_surface_has_no_truth_or_label_parameter() -> None:
    parameters = set(inspect.signature(model_only.RGBKFactorPredictor.predict).parameters)
    assert parameters == {"self", "rgb_u8_hwc", "intrinsics"}
    source = Path(model_only.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("reducer" in name.lower() for name in imports)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "predict"
    )
    argument_names = {argument.arg for argument in function.args.args}
    assert not argument_names & {"label_path", "truth", "depth", "pose", "imu", "task_state"}


def test_scale_intrinsics_is_pixel_center_consistent() -> None:
    value = np.asarray([[100.0, 0.0, 9.5], [0.0, 200.0, 19.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    result = model_only.scale_intrinsics(value, (40, 20), (20, 10))
    assert result.tolist() == [[50.0, 0.0, 4.5], [0.0, 100.0, 9.5], [0.0, 0.0, 1.0]]
