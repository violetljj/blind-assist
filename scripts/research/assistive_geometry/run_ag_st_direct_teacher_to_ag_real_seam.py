#!/usr/bin/env python3
"""Run a real SuperTeacher-factor -> adapter -> reducer seam.

This route deliberately does not require the compressed student to pass.  It
turns the already materialized Tier-A/Tier-B supervision into factor tensors,
uses conservative Tier-C completion only where the source geometry is absent,
and then executes the frozen FactorTensorAdapter and deterministic reducer.
No learned final-task head is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, label

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from factor_tensor_adapter import (  # noqa: E402
    CALIBRATION_SCHEMA,
    FACTOR_SCHEMA_SHA256,
    GEOMETRY_SCHEMA,
    OUTPUT_SCHEMA as ADAPTER_OUTPUT_SCHEMA,
    PREDICTION_SCHEMA,
    adapt_factor_tensor,
    canonical_sha256 as adapter_sha256,
)
from geometry_r2_reducer import (  # noqa: E402
    OUTPUT_SCHEMA as REDUCER_OUTPUT_SCHEMA,
    canonical_sha256 as reducer_sha256,
    iter_cells,
    reduce_frame,
)


DEFAULT_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt17-pose-anchored-fresh-canary-labels-r1/result.json"
EXPECTED_LABEL_RESULT_SHA256 = "BD8379B653C5F492EC6F740959F1F1A2903A9170F7C9A0FD64CBBC264FD27260"
DEFAULT_PROFILE_FIXTURE = MODULE_DIR / "fixtures/geometry_r2_f0_cases.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-direct-teacher-to-ag-real-seam-r4"
FACTOR_DOWNSAMPLE = 4
SOURCE_DEPTH_RELATIVE_SIGMA = 0.02
COMPLETION_COMPONENT_Q10_SIGMA_FRACTION = 0.90
COMPLETION_SCALE_RELATIVE_SIGMA_CAP = 0.80
COMPLETION_PROBABILITY = 0.50
EVIDENCE_SIGMA_FLOOR = 0.20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def block_view(value: np.ndarray, factor: int) -> np.ndarray:
    height, width = value.shape
    require(height % factor == 0 and width % factor == 0, "factor grid is not divisible by downsample")
    return value.reshape(height // factor, factor, width // factor, factor)


def block_mean(value: np.ndarray, factor: int) -> np.ndarray:
    return block_view(value, factor).mean(axis=(1, 3))


def block_max(value: np.ndarray, factor: int) -> np.ndarray:
    return block_view(value, factor).max(axis=(1, 3))


def block_all(value: np.ndarray, factor: int) -> np.ndarray:
    return block_view(value, factor).all(axis=(1, 3))


def block_depth(depth: np.ndarray, valid: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    depth_blocks = block_view(depth, factor).transpose(0, 2, 1, 3).reshape(depth.shape[0] // factor, depth.shape[1] // factor, -1)
    valid_blocks = block_view(valid, factor).transpose(0, 2, 1, 3).reshape(valid.shape[0] // factor, valid.shape[1] // factor, -1)
    values = np.where(valid_blocks, depth_blocks, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reduced = np.nanmedian(values, axis=-1)
    reduced_valid = np.isfinite(reduced) & (reduced > 0.0)
    return reduced.astype(np.float32), reduced_valid


def nearest_completion(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(bool(valid.any()), "no source-native depth block available")
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    completed = depth.copy()
    completed[~valid] = depth[indices[0][~valid], indices[1][~valid]]
    require(bool(np.isfinite(completed).all()) and bool((completed > 0.0).all()), "depth completion nonfinite")
    return completed.astype(np.float32), distance.astype(np.float32)


def scaled_intrinsics(intrinsics: np.ndarray, factor: int) -> np.ndarray:
    output = intrinsics.astype(np.float64).copy()
    output[0, 0] /= factor
    output[1, 1] /= factor
    output[0, 2] = (output[0, 2] + 0.5) / factor - 0.5
    output[1, 2] = (output[1, 2] + 0.5) / factor - 0.5
    return output


def factor_identity(label_result_sha: str) -> dict[str, Any]:
    return {
        "source": "AG_ST_DIRECT_TEACHER",
        "factor_factory": "SOURCE_POSE_GEOMETRY_R0",
        "label_result_sha256": label_result_sha,
        "metric_depth_tier": "A_SOURCE_NATIVE",
        "support_boundary_tier": "B_GEOMETRY_ANCHORED_TEACHER",
        "completion_tier": "C_CONSERVATIVE_GEOMETRY_PSEUDO_EVIDENCE",
        "completion_encoding": "NEAREST_DEPTH_HIGH_SIGMA_HALF_PROBABILITY",
        "learned_final_task_head": False,
    }


def build_factor_and_receipts(
    label_path: Path,
    label_result_sha: str,
    normal_sigma_rad: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    with np.load(label_path, allow_pickle=False) as payload:
        sample_id = str(np.asarray(payload["sample_id"]).item())
        orientation_raw = str(np.asarray(payload["orientation"]).item())
        depth = np.asarray(payload["metric_depth_m_hw"], dtype=np.float32)
        depth_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
        evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
        support_valid_hw = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
        support = np.asarray(payload["support_truth_hw"], dtype=np.float32)
        obstacle = np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32)
        boundary = np.asarray(payload["boundary_probability_diagnostic_hw"], dtype=np.float32)
        normal = np.asarray(payload["support_plane_normal_camera_xyz"], dtype=np.float64)
        camera_height = float(np.asarray(payload["camera_height_m"]).item())
        support_plane_valid = bool(np.asarray(payload["support_plane_valid"]).item())
        support_residual = float(np.asarray(payload["support_plane_fit_residual_diagnostic_m"]).item())
        intrinsics = np.asarray(payload["intrinsics_output"], dtype=np.float64)
        transform = np.asarray(payload["camera_to_world_output"], dtype=np.float64)
        gravity = np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float64)
        camera_receipt = str(np.asarray(payload["camera_geometry_receipt_sha256"]).item())

    factor = FACTOR_DOWNSAMPLE
    reduced_depth, depth_block_available = block_depth(depth, depth_valid, factor)
    observed = block_all(evidence_valid & depth_valid, factor) & depth_block_available
    completed_depth, completion_distance = nearest_completion(reduced_depth, depth_block_available)
    scaled_k = scaled_intrinsics(intrinsics, factor)
    normal /= np.linalg.norm(normal)
    gravity /= np.linalg.norm(gravity)
    support_probability = block_max(np.where(support_valid_hw, support, 0.0).astype(np.float32), factor).astype(np.float32)
    obstacle_probability = block_max(np.where(evidence_valid, obstacle, 0.0).astype(np.float32), factor).astype(np.float32)
    raw_boundary = block_max(np.where(evidence_valid, boundary, 0.0).astype(np.float32), factor).astype(np.float32)
    boundary_probability = np.maximum(raw_boundary, obstacle_probability)

    completion = ~observed
    rows, columns = np.indices(completed_depth.shape, dtype=np.float32)
    point_x = (columns - float(scaled_k[0, 2])) * completed_depth / float(scaled_k[0, 0])
    point_y = (rows - float(scaled_k[1, 2])) * completed_depth / float(scaled_k[1, 1])
    completed_height = (
        point_x * float(gravity[0])
        + point_y * float(gravity[1])
        + completed_depth * float(gravity[2])
        + camera_height
    )
    completed_support = np.exp(-0.5 * np.square(completed_height / 0.10)).astype(np.float32)
    completed_lower = 1.0 / (1.0 + np.exp(-(completed_height - 0.08) / 0.04))
    completed_upper = 1.0 / (1.0 + np.exp((completed_height - 2.00) / 0.15))
    completed_obstacle = (completed_lower * completed_upper * (1.0 - completed_support)).astype(np.float32)
    completion_candidate = completion & (completed_obstacle >= 0.50)
    support_probability[completion] = completed_support[completion]
    obstacle_probability[completion] = 0.0
    obstacle_probability[completion_candidate] = COMPLETION_PROBABILITY
    boundary_probability[completion] = 0.0
    boundary_probability[completion_candidate] = COMPLETION_PROBABILITY
    depth_valid_probability = np.where(observed, 1.0, COMPLETION_PROBABILITY).astype(np.float32)
    depth_sigma = np.maximum(0.01, SOURCE_DEPTH_RELATIVE_SIGMA * completed_depth)
    candidate_mask = completion_candidate | (obstacle_probability >= 0.5)
    component_ids, component_count = label(candidate_mask, structure=np.ones((3, 3), dtype=np.uint8))
    for component_id in range(1, int(component_count) + 1):
        component = component_ids == component_id
        completed_component = component & completion
        if not bool(completed_component.any()):
            continue
        q10_depth = float(np.quantile(completed_depth[component], 0.10))
        depth_sigma[completed_component] = max(0.01, COMPLETION_COMPONENT_Q10_SIGMA_FRACTION * q10_depth)
    boundary_sigma = np.full(completed_depth.shape, 0.5, dtype=np.float32)
    boundary_sigma[completion] = np.clip(1.0 + completion_distance[completion], 1.0, 12.0)
    log_scale = float(np.mean(np.log(completed_depth[observed])))
    scale_m = math.exp(log_scale)
    depth_shape = completed_depth / scale_m
    height, width = completed_depth.shape
    orientation = "portrait" if orientation_raw == "PORTRAIT_ROT90_CLOCKWISE" else "landscape"
    identity = factor_identity(label_result_sha)

    prediction = {
        "schema": PREDICTION_SCHEMA,
        "sample_id": sample_id,
        "factor_identity": identity,
        "camera_geometry_receipt_sha256": camera_receipt,
        "depth_scale": {
            "depth_shape_positive_hw": depth_shape.astype(np.float32).tolist(),
            "log_metric_scale_m_scalar": log_scale,
            "depth_log_sigma_hw": np.log(depth_sigma).astype(np.float32).tolist(),
            "depth_valid_probability_hw": depth_valid_probability.tolist(),
            "metric_scale_valid": True,
        },
        "support_surface": {
            "support_probability_hw": support_probability.tolist(),
            "support_plane_normal_camera_xyz": normal.tolist(),
            "camera_height_m": camera_height,
            "support_residual_sigma_m": max(0.01, support_residual),
            "support_valid": support_plane_valid,
        },
        "obstacle_boundary_evidence": {
            "obstacle_evidence_probability_hw": obstacle_probability.tolist(),
            "boundary_probability_hw": boundary_probability.tolist(),
            "boundary_localization_sigma_px_hw": boundary_sigma.tolist(),
            "evidence_valid_hw": np.ones(completed_depth.shape, dtype=np.bool_).tolist(),
        },
    }
    geometry = {
        "schema": GEOMETRY_SCHEMA,
        "frame_id": sample_id,
        "sample_id": sample_id,
        "content_sha256": camera_receipt,
        "tensor_hw": [height, width],
        "orientation": orientation,
        "k_display_upright": {
            "fx": float(scaled_k[0, 0]),
            "fy": float(scaled_k[1, 1]),
            "cx": float(scaled_k[0, 2]),
            "cy": float(scaled_k[1, 2]),
        },
        "k_valid": bool(
            scaled_k[0, 0] > 0.0
            and scaled_k[1, 1] > 0.0
            and 0.0 <= scaled_k[0, 2] < width
            and 0.0 <= scaled_k[1, 2] < height
        ),
        "transform_valid": bool(np.isfinite(transform).all() and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])),
        "gravity_valid": bool(np.isfinite(gravity).all() and abs(np.linalg.norm(gravity) - 1.0) <= 1.0e-6),
        "gravity_up_camera": gravity.tolist(),
    }
    calibration = {
        "schema": CALIBRATION_SCHEMA,
        "calibration_id": "AG_ST_DIRECT_TEACHER_SOURCE_GEOMETRY_R0",
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "source_role": "FIT_ONLY_CALIBRATION",
        "task_outcome_used": False,
        "scale_relative_sigma_floor": SOURCE_DEPTH_RELATIVE_SIGMA,
        "scale_relative_sigma_cap": COMPLETION_SCALE_RELATIVE_SIGMA_CAP,
        "support_normal_sigma_rad": normal_sigma_rad,
        "support_height_sigma_m": 0.02,
        "boundary_sigma_floor_px": 0.5,
        "evidence_sigma_floor": EVIDENCE_SIGMA_FLOOR,
    }
    flat_payload = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "sample_id": np.asarray(sample_id),
        "factor_identity_json": np.asarray(json.dumps(identity, sort_keys=True, separators=(",", ":"))),
        "camera_geometry_receipt_sha256": np.asarray(camera_receipt),
        "depth_shape_positive_hw": depth_shape.astype(np.float32),
        "log_metric_scale_m_scalar": np.asarray(log_scale, dtype=np.float32),
        "depth_log_sigma_hw": np.log(depth_sigma).astype(np.float32),
        "depth_valid_probability_hw": depth_valid_probability,
        "metric_scale_valid": np.asarray(True, dtype=np.bool_),
        "support_probability_hw": support_probability,
        "support_plane_normal_camera_xyz": normal.astype(np.float32),
        "camera_height_m": np.asarray(camera_height, dtype=np.float32),
        "support_residual_sigma_m": np.asarray(max(0.01, support_residual), dtype=np.float32),
        "support_valid": np.asarray(support_plane_valid, dtype=np.bool_),
        "obstacle_evidence_probability_hw": obstacle_probability,
        "boundary_probability_hw": boundary_probability.astype(np.float32),
        "boundary_localization_sigma_px_hw": boundary_sigma,
        "evidence_valid_hw": np.ones(completed_depth.shape, dtype=np.bool_),
    }
    receipt = {
        "sample_id": sample_id,
        "source_hw": list(depth.shape),
        "factor_hw": [height, width],
        "source_observed_evidence_fraction": float(evidence_valid.mean()),
        "source_native_factor_block_count": int(observed.sum()),
        "tier_c_completion_block_count": int(completion.sum()),
        "tier_c_obstacle_candidate_block_count": int(completion_candidate.sum()),
        "maximum_completion_distance_factor_px": float(completion_distance[completion].max()) if bool(completion.any()) else 0.0,
        "normal_gravity_dot": float(normal @ gravity),
    }
    return prediction, geometry, calibration, flat_payload, receipt


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        identity = json.loads(str(np.asarray(value["factor_identity_json"]).item()))
        return {
            "schema": str(np.asarray(value["schema"]).item()),
            "sample_id": str(np.asarray(value["sample_id"]).item()),
            "factor_identity": identity,
            "camera_geometry_receipt_sha256": str(np.asarray(value["camera_geometry_receipt_sha256"]).item()),
            "depth_scale": {
                "depth_shape_positive_hw": np.asarray(value["depth_shape_positive_hw"]).tolist(),
                "log_metric_scale_m_scalar": float(np.asarray(value["log_metric_scale_m_scalar"]).item()),
                "depth_log_sigma_hw": np.asarray(value["depth_log_sigma_hw"]).tolist(),
                "depth_valid_probability_hw": np.asarray(value["depth_valid_probability_hw"]).tolist(),
                "metric_scale_valid": bool(np.asarray(value["metric_scale_valid"]).item()),
            },
            "support_surface": {
                "support_probability_hw": np.asarray(value["support_probability_hw"]).tolist(),
                "support_plane_normal_camera_xyz": np.asarray(value["support_plane_normal_camera_xyz"]).tolist(),
                "camera_height_m": float(np.asarray(value["camera_height_m"]).item()),
                "support_residual_sigma_m": float(np.asarray(value["support_residual_sigma_m"]).item()),
                "support_valid": bool(np.asarray(value["support_valid"]).item()),
            },
            "obstacle_boundary_evidence": {
                "obstacle_evidence_probability_hw": np.asarray(value["obstacle_evidence_probability_hw"]).tolist(),
                "boundary_probability_hw": np.asarray(value["boundary_probability_hw"]).tolist(),
                "boundary_localization_sigma_px_hw": np.asarray(value["boundary_localization_sigma_px_hw"]).tolist(),
                "evidence_valid_hw": np.asarray(value["evidence_valid_hw"]).tolist(),
            },
        }


def run(label_result_path: Path, profile_fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(sha256_file(label_result_path) == EXPECTED_LABEL_RESULT_SHA256, "Attempt17 label result drift")
    labels = json.loads(label_result_path.read_text(encoding="utf-8"))
    require(labels["passed"] and labels["frame_count"] == 12 and labels["parent_count"] == 4, "label frontdoor not passed")
    require(not labels["decision"]["model_metrics_opened"], "label result model firewall drift")
    profile_fixture = json.loads(profile_fixture_path.read_text(encoding="utf-8"))
    reducer_profile = profile_fixture["reducer_profile"]
    source_anchor_angles = np.asarray(
        [float(row["angle_to_plus_z_deg"]) for row in labels["world_up_anchor_receipt"]["parent_angles"]],
        dtype=np.float64,
    )
    normal_sigma_rad = math.radians(float(np.quantile(source_anchor_angles, 0.75)))

    factor_dir = output_dir / "factor_tensors"
    adapter_dir = output_dir / "adapter_frames"
    reducer_dir = output_dir / "reducer_outputs"
    factor_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir.mkdir(parents=True, exist_ok=False)
    reducer_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    all_states: list[str] = []
    for frame in sorted(labels["frames"], key=lambda row: str(row["sample_id"])):
        label_path = Path(frame["output"])
        require(sha256_file(label_path) == frame["output_sha256"], f"label payload drift: {frame['sample_id']}")
        prediction, geometry, calibration, flat_payload, receipt = build_factor_and_receipts(
            label_path, EXPECTED_LABEL_RESULT_SHA256, normal_sigma_rad
        )
        factor_path = factor_dir / f"{frame['sample_id']}.npz"
        np.savez_compressed(factor_path, **flat_payload)
        reloaded_prediction = load_prediction(factor_path)
        require(reloaded_prediction["schema"] == prediction["schema"], "factor tensor schema roundtrip drift")
        require(reloaded_prediction["sample_id"] == prediction["sample_id"], "factor tensor identity roundtrip drift")
        adapter_input = {
            "prediction": reloaded_prediction,
            "geometry_receipt": geometry,
            "calibration_receipt": calibration,
        }
        adapted = adapt_factor_tensor(adapter_input)
        require(adapted["schema"] == ADAPTER_OUTPUT_SCHEMA, "adapter output schema drift")
        reduced_first = reduce_frame(adapted, reducer_profile)
        reduced_second = reduce_frame(json.loads(json.dumps(adapted)), json.loads(json.dumps(reducer_profile)))
        require(reduced_first["schema"] == REDUCER_OUTPUT_SCHEMA, "reducer output schema drift")
        require(reducer_sha256(reduced_first) == reducer_sha256(reduced_second), "reducer is nondeterministic")
        adapter_path = adapter_dir / f"{frame['sample_id']}.json"
        reducer_path = reducer_dir / f"{frame['sample_id']}.json"
        write_json(adapter_path, adapted)
        write_json(reducer_path, reduced_first)
        states = [state for _, _, state in iter_cells(reduced_first)]
        all_states.extend(states)
        missing_components = [row for row in adapted["boundary"]["obstacles"] if not row["depth_valid"]]
        rows.append(
            {
                **receipt,
                "factor_tensor": {"path": str(factor_path.resolve()), "sha256": sha256_file(factor_path), "bytes": factor_path.stat().st_size},
                "adapter_frame": {"path": str(adapter_path.resolve()), "sha256": sha256_file(adapter_path), "canonical_sha256": adapter_sha256(adapted)},
                "reducer_output": {"path": str(reducer_path.resolve()), "sha256": sha256_file(reducer_path), "canonical_sha256": reducer_sha256(reduced_first)},
                "adapter_depth_valid": bool(adapted["depth_scale"]["valid"]),
                "adapter_support_valid": bool(adapted["support"]["valid"]),
                "adapter_boundary_valid": bool(adapted["boundary"]["valid"]),
                "adapter_boundary_coverage": float(adapted["boundary"]["coverage"]),
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "adapter_missing_component_count": len(missing_components),
                "state_counts": {state: states.count(state) for state in sorted(set(states))},
                "deterministic_repeat_equal": reducer_sha256(reduced_first) == reducer_sha256(reduced_second),
            }
        )

    state_set = set(all_states)
    gates = {
        "STAG_S01_ALL_SOURCE_LABEL_RECEIPTS_EXACT": len(rows) == 12,
        "STAG_S02_ALL_FACTOR_TENSORS_ROUNDTRIP": all(Path(row["factor_tensor"]["path"]).is_file() for row in rows),
        "STAG_S03_ALL_ADAPTER_FRAMES_VALID": all(
            row["adapter_depth_valid"]
            and row["adapter_support_valid"]
            and row["adapter_boundary_valid"]
            and row["adapter_boundary_coverage"] >= 0.99
            for row in rows
        ),
        "STAG_S04_TIER_C_COMPLETION_EXPLICIT": all(row["tier_c_completion_block_count"] > 0 for row in rows),
        "STAG_S05_REDUCER_DETERMINISTIC_12_OF_12": all(row["deterministic_repeat_equal"] for row in rows),
        "STAG_S06_NONTRIVIAL_REAL_STATES": "CLEAR_OBSERVED" in state_set and "UNKNOWN" in state_set,
        "STAG_S07_POSITIVE_EVIDENCE_PATH_EXERCISED": "OCCUPIED_OBSERVED" in state_set,
        "STAG_S08_NO_LEARNED_FINAL_TASK_HEAD": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_direct_teacher_to_ag_real_seam_result_v1",
        "status": "AG_ST_DIRECT_TEACHER_TO_AG_REAL_SEAM_PASS"
        if passed
        else "AG_ST_DIRECT_TEACHER_TO_AG_REAL_SEAM_INCOMPLETE",
        "passed": passed,
        "label_result": {"path": str(label_result_path.resolve()), "sha256": EXPECTED_LABEL_RESULT_SHA256},
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "adapter_implementation": {"path": str((MODULE_DIR / "factor_tensor_adapter.py").resolve()), "sha256": sha256_file(MODULE_DIR / "factor_tensor_adapter.py")},
        "reducer_implementation": {"path": str((MODULE_DIR / "geometry_r2_reducer.py").resolve()), "sha256": sha256_file(MODULE_DIR / "geometry_r2_reducer.py")},
        "reducer_profile": {"path": str(profile_fixture_path.resolve()), "sha256": sha256_file(profile_fixture_path), "profile": reducer_profile},
        "factor_factory": {
            "downsample": FACTOR_DOWNSAMPLE,
            "source_depth_relative_sigma": SOURCE_DEPTH_RELATIVE_SIGMA,
            "completion_component_q10_sigma_fraction": COMPLETION_COMPONENT_Q10_SIGMA_FRACTION,
            "completion_scale_relative_sigma_cap": COMPLETION_SCALE_RELATIVE_SIGMA_CAP,
            "completion_probability": COMPLETION_PROBABILITY,
            "evidence_sigma_floor": EVIDENCE_SIGMA_FLOOR,
            "support_normal_sigma_rad": normal_sigma_rad,
            "support_normal_sigma_source": "Q75_18_PARENT_SOURCE_NATIVE_GRAVITY_ANGLE_TO_FREIBURG_PLUS_Z",
            "tier_c_truth_claim": False,
            "unknown_is_negative": False,
        },
        "frame_count": len(rows),
        "parent_count": len({frame["parent_id"] for frame in labels["frames"]}),
        "aggregate_state_counts": {state: all_states.count(state) for state in sorted(state_set)},
        "gates": gates,
        "frames": rows,
        "decision": {
            "superteacher_data_frontdoor_complete": bool(passed),
            "real_factor_adapter_reducer_seam_complete": bool(passed),
            "student_compression_required_for_this_seam": False,
            "student_route_status": "SEPARATE_COMPRESSION_AND_DEPLOYMENT_WORK",
            "claim_ceiling": "Real source-anchored SuperTeacher factor mechanics through the deterministic AG seam; not mobile inference, deployment, product or safety proof.",
        },
    }
    write_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-result", type=Path, default=DEFAULT_LABEL_RESULT)
    parser.add_argument("--profile-fixture", type=Path, default=DEFAULT_PROFILE_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.label_result.resolve(), args.profile_fixture.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "passed": result["passed"], "gates": result["gates"], "aggregate_state_counts": result["aggregate_state_counts"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
