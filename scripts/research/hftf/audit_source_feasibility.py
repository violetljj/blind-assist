#!/usr/bin/env python3
"""Audit whether a replay source can support an HFTF source canary.

This audit is intentionally narrower than a model experiment. It verifies the
source contract and reports whether a static metric-geometry projection canary
is supportable without silently authenticating pose/time, body calibration, or
human-event truth. Multi-height and future-teacher admission require separate,
source-specific verifiers and are never granted by this generic H0 audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


SCHEMA = "blindassist_hftf_source_feasibility_r0"
EXPECTED_REPLAY_SCHEMA = "blindassist_sanpo_synthetic_replay_v1"
POSE_BINDING_COLUMNS = {
    "frame",
    "frame_id",
    "frame_index",
    "image_id",
    "source_frame",
    "source_frame_index",
    "time_ns",
    "timestamp",
    "timestamp_ms",
    "timestamp_ns",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve_inside(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes replay root: {relative_path}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return None
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError):
        return None


def _depth_payload_contract(
    path: Path, width: int, height: int
) -> dict[str, Any]:
    payload = gzip.decompress(path.read_bytes())
    result: dict[str, Any] = {
        "shape_valid": False,
        "finite_positive_fraction": 0.0,
        "all_finite_positive": False,
        "ok": False,
    }
    expected_value_count = width * height + 2
    if len(payload) != expected_value_count * 2:
        return result
    values = np.frombuffer(payload, dtype="<f2")
    declared_height = float(values[0])
    declared_width = float(values[1])
    shape_valid = (
        math.isfinite(declared_height)
        and math.isfinite(declared_width)
        and declared_height.is_integer()
        and declared_width.is_integer()
        and int(declared_height) == height
        and int(declared_width) == width
        and values[2:].size == width * height
    )
    depth_values = values[2:].astype(np.float32, copy=False)
    finite_positive = np.isfinite(depth_values) & (depth_values > 0)
    fraction = float(finite_positive.mean()) if depth_values.size else 0.0
    result.update(
        {
            "shape_valid": shape_valid,
            "finite_positive_fraction": fraction,
            "all_finite_positive": fraction == 1.0,
            "ok": shape_valid and fraction == 1.0,
        }
    )
    return result


def _all_finite_positive(values: Iterable[Any]) -> bool:
    materialized = list(values)
    if not materialized or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in materialized
    ):
        return False
    try:
        numbers = [float(value) for value in materialized]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value > 0 for value in numbers)


def _is_exact_one(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number == 1.0


def _finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    )


def _normalized_quaternion(value: Any, tolerance: float = 1e-3) -> bool:
    if not _finite_vector(value, 4):
        return False
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    return norm > 0 and abs(norm - 1.0) <= tolerance


def _check_bound_files(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bindings = (
        ("image_path", "image_sha256", "png"),
        ("source_mask_path", "source_mask_sha256", "png"),
        ("source_depth_path", "source_depth_sha256", "depth"),
    )
    missing: list[str] = []
    hash_mismatches: list[str] = []
    png_shape_mismatches: list[str] = []
    depth_shape_mismatches: list[str] = []
    depth_nonfinite_or_nonpositive: list[str] = []
    depth_finite_positive_fractions: list[float] = []
    duplicate_canonical_paths: list[str] = []
    seen_canonical_paths: dict[str, set[str]] = {
        "png:image_path": set(),
        "png:source_mask_path": set(),
        "depth:source_depth_path": set(),
    }
    duplicate_observation_hash_triplets: list[str] = []
    seen_observation_hash_triplets: set[tuple[str, str, str]] = set()
    verified_files = 0

    for row in rows:
        row_id = str(row.get("id", "<missing-id>"))
        for path_key, hash_key, kind in bindings:
            relative_path = row.get(path_key)
            expected_hash = row.get(hash_key)
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                missing.append(f"{row_id}:{path_key}/{hash_key}")
                continue
            path = _resolve_inside(root, relative_path)
            if not path.is_file():
                missing.append(f"{row_id}:{relative_path}")
                continue
            binding_kind = f"{kind}:{path_key}"
            canonical_path = str(path)
            if canonical_path in seen_canonical_paths[binding_kind]:
                duplicate_canonical_paths.append(
                    f"{row_id}:{path_key}:{relative_path}"
                )
            else:
                seen_canonical_paths[binding_kind].add(canonical_path)
            actual_hash = _sha256(path)
            verified_files += 1
            if actual_hash != expected_hash:
                hash_mismatches.append(f"{row_id}:{relative_path}")
            width = row.get("width")
            height = row.get("height")
            dimensions_valid = (
                isinstance(width, int)
                and not isinstance(width, bool)
                and width > 0
                and isinstance(height, int)
                and not isinstance(height, bool)
                and height > 0
            )
            if not dimensions_valid:
                if kind == "depth":
                    depth_shape_mismatches.append(f"{row_id}:missing-dimensions")
                else:
                    png_shape_mismatches.append(f"{row_id}:missing-dimensions")
                continue
            if kind == "png":
                if _png_dimensions(path) != (width, height):
                    png_shape_mismatches.append(f"{row_id}:{relative_path}")
            else:
                depth_check = _depth_payload_contract(path, width, height)
                depth_finite_positive_fractions.append(
                    float(depth_check["finite_positive_fraction"])
                )
                if not depth_check["shape_valid"]:
                    depth_shape_mismatches.append(f"{row_id}:{relative_path}")
                if not depth_check["all_finite_positive"]:
                    depth_nonfinite_or_nonpositive.append(
                        f"{row_id}:{relative_path}"
                    )

        hash_triplet = tuple(
            row.get(hash_key)
            for hash_key in (
                "image_sha256",
                "source_mask_sha256",
                "source_depth_sha256",
            )
        )
        if all(isinstance(value, str) for value in hash_triplet):
            typed_hash_triplet = (
                str(hash_triplet[0]),
                str(hash_triplet[1]),
                str(hash_triplet[2]),
            )
            if typed_hash_triplet in seen_observation_hash_triplets:
                duplicate_observation_hash_triplets.append(row_id)
            else:
                seen_observation_hash_triplets.add(typed_hash_triplet)

    return {
        "verified_file_count": verified_files,
        "expected_file_count": len(rows) * len(bindings),
        "distinct_canonical_file_count": sum(
            len(paths) for paths in seen_canonical_paths.values()
        ),
        "missing": missing,
        "hash_mismatches": hash_mismatches,
        "duplicate_canonical_paths": duplicate_canonical_paths,
        "duplicate_observation_hash_triplets": (
            duplicate_observation_hash_triplets
        ),
        "png_shape_mismatches": png_shape_mismatches,
        "depth_shape_mismatches": depth_shape_mismatches,
        "depth_nonfinite_or_nonpositive": depth_nonfinite_or_nonpositive,
        "minimum_depth_finite_positive_fraction": (
            min(depth_finite_positive_fractions)
            if depth_finite_positive_fractions
            else 0.0
        ),
        "ok": not any(
            (
                missing,
                hash_mismatches,
                duplicate_canonical_paths,
                duplicate_observation_hash_triplets,
                png_shape_mismatches,
                depth_shape_mismatches,
                depth_nonfinite_or_nonpositive,
            )
        ),
    }


def _read_raw_pose_inventory(
    root: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    pose_paths: set[str] = set()
    declared_hashes: set[str] = set()
    declared_row_count = 0
    for row in rows:
        camera_poses = row.get("modalities", {}).get("camera_poses", {})
        path = camera_poses.get("path")
        digest = camera_poses.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            declared_row_count += 1
            pose_paths.add(path)
            declared_hashes.add(digest)

    result: dict[str, Any] = {
        "declared_row_count": declared_row_count,
        "all_rows_declared": declared_row_count == len(rows),
        "declared_path_count": len(pose_paths),
        "declared_hash_count": len(declared_hashes),
        "row_count": 0,
        "columns": [],
        "csv_has_explicit_frame_or_timestamp_column": False,
        "row_count_covers_requested_source_indices": False,
        "hash_bound": False,
        "admitted_for_hftf_future": False,
        "ok": False,
    }
    if (
        declared_row_count != len(rows)
        or len(pose_paths) != 1
        or len(declared_hashes) != 1
    ):
        return result

    relative_path = next(iter(pose_paths))
    pose_path = _resolve_inside(root, relative_path)
    if not pose_path.is_file():
        return result

    with pose_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [str(column).strip().lower() for column in (reader.fieldnames or [])]
        pose_rows = list(reader)

    source_indices = [
        row["source_frame_index"]
        for row in rows
        if isinstance(row.get("source_frame_index"), int)
    ]
    result.update(
        {
            "path": relative_path,
            "row_count": len(pose_rows),
            "columns": columns,
            "csv_has_explicit_frame_or_timestamp_column": bool(
                set(columns) & POSE_BINDING_COLUMNS
            ),
            "row_count_covers_requested_source_indices": bool(source_indices)
            and len(pose_rows) > max(source_indices),
            "hash_bound": _sha256(pose_path) == next(iter(declared_hashes)),
        }
    )
    result["ok"] = result["hash_bound"] and result["row_count"] > 0
    return result


def _validate_pose_binding(
    root: Path,
    spec: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = spec.get("hftf_pose_binding", {})
    if not isinstance(contract, dict):
        contract = {}
    result: dict[str, Any] = {
        "contract_present": bool(contract),
        "schema_valid": False,
        "metadata_valid": False,
        "path_hash_bound": False,
        "binding_row_count": 0,
        "one_to_one_manifest_binding": False,
        "raw_pose_rows_unique": False,
        "all_pose_values_valid": False,
        "trusted_source_mapping_admitted": False,
        "admission_blocker": (
            "source_specific_pose_time_mapping_verifier_not_implemented"
        ),
        "errors": [],
        "ok": False,
    }
    if not contract:
        result["errors"].append("hftf_pose_binding_contract_absent")
        return result

    admitted_states = contract.get("admitted_tracking_states")
    metadata_valid = (
        contract.get("schema") == "blindassist_hftf_pose_binding_v1"
        and contract.get("transform_direction") == "world_from_camera"
        and contract.get("position_unit") == "meter"
        and contract.get("time_unit") == "millisecond"
        and contract.get("quaternion_order") == "xyzw"
        and isinstance(contract.get("camera_frame"), str)
        and bool(contract["camera_frame"])
        and isinstance(contract.get("world_frame"), str)
        and bool(contract["world_frame"])
        and contract["camera_frame"] != contract["world_frame"]
        and isinstance(contract.get("source_authority"), str)
        and bool(contract["source_authority"])
        and isinstance(contract.get("binding_method"), str)
        and bool(contract["binding_method"])
        and isinstance(admitted_states, list)
        and bool(admitted_states)
        and all(isinstance(state, str) and state for state in admitted_states)
    )
    result["schema_valid"] = (
        contract.get("schema") == "blindassist_hftf_pose_binding_v1"
    )
    result["metadata_valid"] = metadata_valid

    relative_path = contract.get("path")
    expected_hash = contract.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        result["errors"].append("pose_binding_path_or_sha256_missing")
        return result
    binding_path = _resolve_inside(root, relative_path)
    if not binding_path.is_file() or _sha256(binding_path) != expected_hash:
        result["errors"].append("pose_binding_file_missing_or_hash_mismatch")
        return result
    result["path"] = relative_path
    result["path_hash_bound"] = True

    pose_paths = {
        row.get("modalities", {}).get("camera_poses", {}).get("path")
        for row in manifest_rows
    }
    pose_hashes = {
        row.get("modalities", {}).get("camera_poses", {}).get("sha256")
        for row in manifest_rows
    }
    if (
        len(pose_paths) != 1
        or len(pose_hashes) != 1
        or not isinstance(next(iter(pose_paths)), str)
        or not isinstance(next(iter(pose_hashes)), str)
    ):
        result["errors"].append("raw_pose_source_is_not_single_and_hash_bound")
        return result
    raw_pose_path = _resolve_inside(root, next(iter(pose_paths)))
    if (
        not raw_pose_path.is_file()
        or _sha256(raw_pose_path) != next(iter(pose_hashes))
    ):
        result["errors"].append("raw_pose_source_missing_or_hash_mismatch")
        return result
    with raw_pose_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_pose_rows = list(csv.DictReader(handle))

    binding_rows = _load_jsonl(binding_path)
    result["binding_row_count"] = len(binding_rows)
    manifest_by_id = {
        row.get("id"): row
        for row in manifest_rows
        if isinstance(row.get("id"), str) and row.get("id")
    }
    binding_ids = [
        row.get("manifest_id")
        for row in binding_rows
        if isinstance(row.get("manifest_id"), str) and row.get("manifest_id")
    ]
    one_to_one = (
        len(binding_ids) == len(binding_rows) == len(manifest_rows)
        and len(binding_ids) == len(set(binding_ids))
        and set(binding_ids) == set(manifest_by_id)
    )
    result["one_to_one_manifest_binding"] = one_to_one

    raw_indices: list[int] = []
    row_errors: list[str] = []
    for binding_row in binding_rows:
        manifest_id = binding_row.get("manifest_id")
        manifest_row = manifest_by_id.get(manifest_id)
        if manifest_row is None:
            row_errors.append(f"{manifest_id}:manifest_identity_mismatch")
            continue
        raw_index = binding_row.get("raw_pose_row_index")
        if (
            not isinstance(raw_index, int)
            or isinstance(raw_index, bool)
            or raw_index < 0
            or raw_index >= len(raw_pose_rows)
        ):
            row_errors.append(f"{manifest_id}:raw_pose_row_index_invalid")
            continue
        raw_indices.append(raw_index)
        binding_source_index = binding_row.get("source_frame_index")
        binding_timestamp = binding_row.get("source_timestamp_ms")
        if any(
            (
                not isinstance(binding_source_index, int),
                isinstance(binding_source_index, bool),
                not isinstance(binding_timestamp, int),
                isinstance(binding_timestamp, bool),
                binding_row.get("session_id") != manifest_row.get("session_id"),
                binding_row.get("sequence_id") != manifest_row.get("sequence_id"),
                binding_source_index != manifest_row.get("source_frame_index"),
                binding_timestamp != manifest_row.get("source_timestamp_ms"),
            )
        ):
            row_errors.append(f"{manifest_id}:manifest_pose_key_mismatch")
            continue

        tracking_state = binding_row.get("tracking_state")
        position = binding_row.get("position_m")
        quaternion = binding_row.get("quaternion_xyzw")
        if (
            tracking_state not in admitted_states
            or not _finite_vector(position, 3)
            or not _normalized_quaternion(quaternion)
        ):
            row_errors.append(f"{manifest_id}:pose_values_or_tracking_invalid")
            continue

        raw = raw_pose_rows[raw_index]
        try:
            raw_position = [float(raw[key]) for key in ("pos_x", "pos_y", "pos_z")]
            raw_quaternion = [
                float(raw[key]) for key in ("q_x", "q_y", "q_z", "q_w")
            ]
        except (KeyError, TypeError, ValueError):
            row_errors.append(f"{manifest_id}:raw_pose_values_invalid")
            continue
        if (
            raw.get("tracking_state") != tracking_state
            or not all(
                math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
                for left, right in zip(raw_position, position)
            )
            or not all(
                math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
                for left, right in zip(raw_quaternion, quaternion)
            )
        ):
            row_errors.append(f"{manifest_id}:binding_does_not_match_raw_pose")

    raw_rows_unique = (
        len(raw_indices) == len(binding_rows)
        and len(raw_indices) == len(set(raw_indices))
    )
    result["raw_pose_rows_unique"] = raw_rows_unique
    result["all_pose_values_valid"] = not row_errors
    result["errors"].extend(row_errors)
    result["ok"] = all(
        (
            metadata_valid,
            result["path_hash_bound"],
            one_to_one,
            raw_rows_unique,
            not row_errors,
        )
    )
    return result


def _validate_body_frame_contract(spec: dict[str, Any]) -> dict[str, Any]:
    contract = spec.get("hftf_body_frame_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    transform = contract.get("camera_to_body", {})
    ground = contract.get("ground_reference", {})
    if not isinstance(transform, dict):
        transform = {}
    if not isinstance(ground, dict):
        ground = {}
    schema_valid = (
        contract.get("schema") == "blindassist_hftf_body_frame_v1"
    )
    frame_semantics_valid = (
        isinstance(contract.get("camera_frame"), str)
        and bool(contract["camera_frame"])
        and isinstance(contract.get("body_frame"), str)
        and bool(contract["body_frame"])
        and contract["camera_frame"] != contract["body_frame"]
        and contract.get("transform_direction") == "body_from_camera"
        and contract.get("position_unit") == "meter"
        and contract.get("quaternion_order") == "xyzw"
        and contract.get("axis_convention") == "x_forward_y_left_z_up"
    )
    translation = transform.get("translation_m")
    camera_height = ground.get("camera_height_m")
    translation_valid = _finite_vector(translation, 3)
    camera_height_valid = (
        isinstance(camera_height, (int, float))
        and not isinstance(camera_height, bool)
        and math.isfinite(float(camera_height))
        and 0.5 <= float(camera_height) <= 2.2
    )
    geometry_internal_consistency_valid = (
        translation_valid
        and camera_height_valid
        and abs(float(translation[0])) <= 1.0
        and abs(float(translation[1])) <= 1.0
        and abs(float(translation[2]) - float(camera_height)) <= 0.05
    )
    transform_valid = (
        geometry_internal_consistency_valid
        and _normalized_quaternion(transform.get("quaternion_xyzw"))
    )
    ground_valid = (
        ground.get("kind") == "camera_height_along_body_z"
        and camera_height_valid
    )
    provenance_valid = (
        isinstance(contract.get("source_authority"), str)
        and bool(contract["source_authority"])
        and isinstance(contract.get("verification_method"), str)
        and bool(contract["verification_method"])
    )
    ok = all(
        (
            schema_valid,
            frame_semantics_valid,
            transform_valid,
            ground_valid,
            provenance_valid,
        )
    )
    return {
        "contract_present": bool(contract),
        "schema_valid": schema_valid,
        "frame_semantics_valid": frame_semantics_valid,
        "transform_valid": transform_valid,
        "ground_reference_valid": ground_valid,
        "geometry_internal_consistency_valid": (
            geometry_internal_consistency_valid
        ),
        "provenance_valid": provenance_valid,
        "trusted_calibration_admitted": False,
        "admission_blocker": "source_specific_calibration_receipt_not_implemented",
        "camera_frame": contract.get("camera_frame"),
        "ok": ok,
    }


def _check_metric_qa(
    root: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    row_count = len(rows)
    metric_candidates = sorted((root / "qa").glob("metric_replay_audit_*.json"))
    metric_path = metric_candidates[0] if len(metric_candidates) == 1 else None
    validation_path = root / "qa" / "replay_validation.json"
    result: dict[str, Any] = {
        "metric_report_candidate_count": len(metric_candidates),
        "metric_report_present": bool(metric_path and metric_path.is_file()),
        "replay_validation_present": validation_path.is_file(),
        "metric_schema_valid": False,
        "metric_ok_and_frame_count_valid": False,
        "depth_summary_paths_match_manifest": False,
        "metric_depth_source_integrity": False,
        "finite_positive_depth": False,
        "replay_validation_shape_valid": False,
        "replay_validation_ok_and_frame_count_valid": False,
        "rgb_mask_dimensions_match": False,
        "required_modalities_hash_bound": False,
        "validation_declares_official_train_split": False,
        "ok": False,
    }
    if (
        metric_path is None
        or not metric_path.is_file()
        or not validation_path.is_file()
    ):
        return result

    result["metric_report_path"] = str(metric_path.relative_to(root).as_posix())
    result["metric_report_sha256"] = _sha256(metric_path)
    result["replay_validation_sha256"] = _sha256(validation_path)
    metric = _load_json(metric_path)
    validation = _load_json(validation_path)
    summaries = metric.get("depth_summaries", [])
    expected_depth_names = sorted(
        Path(str(row.get("source_depth_path"))).name for row in rows
    )
    summary_depth_names = sorted(
        str(item.get("path"))
        for item in summaries
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ) if isinstance(summaries, list) else []
    metric_schema_valid = (
        metric.get("schema")
        == "blindassist_sanpo_synthetic_metric_replay_audit_v1"
    )
    metric_identity_valid = bool(
        metric.get("ok") is True
        and isinstance(metric.get("frame_count"), int)
        and not isinstance(metric.get("frame_count"), bool)
        and metric.get("frame_count") == row_count
    )
    validation_shape_valid = (
        isinstance(validation.get("ok"), bool)
        and isinstance(validation.get("frame_count"), int)
        and not isinstance(validation.get("frame_count"), bool)
        and isinstance(
            validation.get("all_rgb_mask_dimensions_match"), bool
        )
        and isinstance(
            validation.get("required_modalities_hash_bound"), bool
        )
        and isinstance(
            validation.get("all_frames_official_train_split"), bool
        )
        and isinstance(validation.get("imu_status"), str)
        and isinstance(validation.get("production_authorized"), bool)
    )
    paths_match = (
        len(summary_depth_names) == row_count
        and summary_depth_names == expected_depth_names
    )
    finite_positive = (
        isinstance(summaries, list)
        and len(summaries) == row_count
        and all(
            isinstance(item, dict)
            and _is_exact_one(item.get("finite_positive_fraction"))
            for item in summaries
        )
    )
    result.update(
        {
            "metric_depth_source_integrity": bool(
                metric_schema_valid
                and metric_identity_valid
                and paths_match
                and metric.get("metric_depth_source_integrity") is True
            ),
            "metric_schema_valid": metric_schema_valid,
            "metric_ok_and_frame_count_valid": metric_identity_valid,
            "depth_summary_paths_match_manifest": paths_match,
            "finite_positive_depth": finite_positive,
            "replay_validation_shape_valid": validation_shape_valid,
            "replay_validation_ok_and_frame_count_valid": bool(
                validation.get("ok") is True
                and validation.get("frame_count") == row_count
            ),
            "rgb_mask_dimensions_match": (
                validation.get("all_rgb_mask_dimensions_match") is True
            ),
            "required_modalities_hash_bound": (
                validation.get("required_modalities_hash_bound") is True
            ),
            "validation_declares_official_train_split": (
                validation.get("all_frames_official_train_split") is True
            ),
        }
    )
    result["ok"] = all(
        result[key]
        for key in (
            "metric_depth_source_integrity",
            "finite_positive_depth",
            "replay_validation_shape_valid",
            "replay_validation_ok_and_frame_count_valid",
            "rgb_mask_dimensions_match",
            "required_modalities_hash_bound",
            "validation_declares_official_train_split",
        )
    )
    return result


def audit_replay(replay_root: Path) -> dict[str, Any]:
    root = replay_root.resolve()
    spec = _load_json(root / "dataset_spec.json")
    rows = _load_jsonl(root / "manifest.replay.jsonl")
    if not rows:
        raise ValueError("Replay manifest is empty")

    raw_ids = [row.get("id") for row in rows]
    ids = [value for value in raw_ids if isinstance(value, str) and value]
    sessions = sorted(
        {
            str(row.get("session_id"))
            for row in rows
            if isinstance(row.get("session_id"), str) and row.get("session_id")
        }
    )
    sequences = sorted(
        {
            str(row.get("sequence_id"))
            for row in rows
            if isinstance(row.get("sequence_id"), str) and row.get("sequence_id")
        }
    )

    files = _check_bound_files(root, rows)
    raw_pose_inventory = _read_raw_pose_inventory(root, rows)
    pose_binding = _validate_pose_binding(root, spec, rows)
    metric_qa = _check_metric_qa(root, rows)
    body_binding = _validate_body_frame_contract(spec)

    camera = spec.get("camera", {})
    manifest_dimensions = {
        (row.get("width"), row.get("height"))
        for row in rows
        if isinstance(row.get("width"), int) and isinstance(row.get("height"), int)
    }
    camera_dimensions_match = (
        isinstance(camera, dict)
        and len(manifest_dimensions) == 1
        and (camera.get("image_width"), camera.get("image_height"))
        == next(iter(manifest_dimensions))
    )
    intrinsics_valid = isinstance(camera, dict) and _all_finite_positive(
        [
            camera.get("fx"),
            camera.get("fy"),
            camera.get("image_width"),
            camera.get("image_height"),
        ]
    )
    principal_point_valid = (
        isinstance(camera, dict)
        and all(
            isinstance(camera.get(key), (int, float))
            and not isinstance(camera.get(key), bool)
            and math.isfinite(float(camera[key]))
            for key in ("cx", "cy")
        )
        and 0 <= float(camera["cx"]) < float(camera["image_width"])
        and 0 <= float(camera["cy"]) < float(camera["image_height"])
    )
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        session_id = row.get("session_id")
        sequence_id = row.get("sequence_id")
        if (
            isinstance(session_id, str)
            and bool(session_id)
            and isinstance(sequence_id, str)
            and bool(sequence_id)
        ):
            grouped_rows.setdefault((session_id, sequence_id), []).append(row)
    all_rows_grouped = (
        bool(grouped_rows)
        and sum(len(group) for group in grouped_rows.values()) == len(rows)
    )
    group_frame_identity_valid = bool(grouped_rows) and all(
        all(
            isinstance(row.get("frame_index"), int)
            and not isinstance(row.get("frame_index"), bool)
            for row in group
        )
        and [row.get("frame_index") for row in group]
        == list(range(len(group)))
        for group in grouped_rows.values()
    )
    group_source_indices_valid = bool(grouped_rows) and all(
        all(
            isinstance(row.get("source_frame_index"), int)
            and not isinstance(row.get("source_frame_index"), bool)
            for row in group
        )
        and all(
            int(later["source_frame_index"])
            > int(earlier["source_frame_index"])
            for earlier, later in zip(group, group[1:])
        )
        for group in grouped_rows.values()
    )
    group_timestamps_valid = bool(grouped_rows) and all(
        len(group) > 0
        and all(
            isinstance(row.get("source_timestamp_ms"), int)
            and not isinstance(row.get("source_timestamp_ms"), bool)
            for row in group
        )
        and all(
            int(later["source_timestamp_ms"])
            > int(earlier["source_timestamp_ms"])
            for earlier, later in zip(group, group[1:])
        )
        for group in grouped_rows.values()
    )
    sequence_spans_ms = {
        f"{session_id}/{sequence_id}": (
            int(group[-1]["source_timestamp_ms"])
            - int(group[0]["source_timestamp_ms"])
            if group_timestamps_valid
            else 0
        )
        for (session_id, sequence_id), group in grouped_rows.items()
    }
    maximum_sequence_span_ms = (
        max(sequence_spans_ms.values()) if sequence_spans_ms else 0
    )
    manifest_identity_valid = (
        len(ids) == len(rows)
        and len(ids) == len(set(ids))
        and all_rows_grouped
        and group_frame_identity_valid
        and group_source_indices_valid
        and len(grouped_rows) > 0
    )
    declared_sanpo_synthetic_manifest_consistent = all(
        isinstance(row.get("source"), dict)
        and row.get("source_annotation_quality") == "SYNTHETIC"
        and row["source"].get("dataset") == "SANPO-Synthetic v0"
        and row["source"].get("official_split") == "train"
        and row["source"].get("session_id") == row.get("session_id")
        for row in rows
    )

    schema_valid = spec.get("schema") == EXPECTED_REPLAY_SCHEMA
    source_integrity = all(
        (
            schema_valid,
            manifest_identity_valid,
            files["ok"],
            metric_qa["ok"],
            intrinsics_valid,
            principal_point_valid,
            camera_dimensions_match,
            group_timestamps_valid,
            declared_sanpo_synthetic_manifest_consistent,
        )
    )
    static_projection_canary = source_integrity
    body_contract_structurally_valid = bool(
        static_projection_canary and body_binding["ok"]
    )
    pose_contract_spec = spec.get("hftf_pose_binding", {})
    if not isinstance(pose_contract_spec, dict):
        pose_contract_spec = {}
    camera_frame_consistent = (
        body_binding["ok"]
        and pose_binding["ok"]
        and body_binding.get("camera_frame")
        == pose_contract_spec.get("camera_frame")
    )
    future_mechanics_structure_ready = (
        body_contract_structurally_valid
        and pose_binding["ok"]
        and camera_frame_consistent
        and maximum_sequence_span_ms >= 1000
    )
    # A hash-bound sidecar can prove internal consistency, but it cannot
    # authenticate the source-specific frame/time mapping or physical body
    # calibration that produced it. H0 therefore never admits these stages.
    multi_height_teacher_canary = False
    future_teacher_canary = False

    event_truth_present = any(row.get("event_truth") is not None for row in rows)
    event_truth_authorized = all(
        bool(row.get("authorization", {}).get("human_event_truth")) for row in rows
    )
    independent_effect_evaluation = False

    if not source_integrity:
        terminal = "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
    else:
        terminal = "HFTF_H0_SOURCE_FEASIBILITY_PARTIAL"

    blockers: list[str] = []
    if not source_integrity:
        blockers.append("source_integrity_contract_failed")
    if static_projection_canary:
        if not body_binding["ok"]:
            blockers.append("body_frame_contract_structurally_invalid_or_absent")
        blockers.append("source_specific_body_calibration_verifier_required")
        if not pose_binding["ok"]:
            blockers.append("pose_binding_contract_structurally_invalid_or_absent")
        blockers.append("source_specific_pose_time_mapping_verifier_required")
        if (
            body_binding["ok"]
            and pose_binding["ok"]
            and not camera_frame_consistent
        ):
            blockers.append("pose_and_body_camera_frames_disagree")
        if (
            body_binding["ok"]
            and pose_binding["ok"]
            and camera_frame_consistent
            and maximum_sequence_span_ms < 1000
        ):
            blockers.append("no_single_sequence_spans_one_second")
    if len(sessions) < 2:
        blockers.append("single_parent_session_only")
    blockers.append("separate_hash_bound_parent_event_ledger_required_for_effect")

    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SOURCE_FEASIBILITY_ONLY",
        "mainline_changed": False,
        "default_app_changed": False,
        "source": {
            "replay_root": str(root),
            "dataset_schema": spec.get("schema"),
            "frame_count": len(rows),
            "parent_session_count": len(sessions),
            "parent_sessions": sessions,
            "sequence_count": len(sequences),
            "sequence_spans_ms": sequence_spans_ms,
            "maximum_single_sequence_span_ms": maximum_sequence_span_ms,
            "declared_source_identity": "SANPO-Synthetic v0",
            "source_identity_claim": (
                "DECLARED_SANPO_SYNTHETIC_REPLAY_NOT_CRYPTOGRAPHICALLY_"
                "AUTHENTICATED"
            ),
            "official_source_identity_authenticated": False,
        },
        "input_hashes": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "dataset_spec_sha256": _sha256(root / "dataset_spec.json"),
            "manifest_sha256": _sha256(root / "manifest.replay.jsonl"),
        },
        "checks": {
            "schema_valid": schema_valid,
            "manifest_identity_valid": manifest_identity_valid,
            "all_manifest_rows_have_nonempty_session_and_sequence": (
                all_rows_grouped
            ),
            "source_frame_indices_strictly_increase_within_sequence": (
                group_source_indices_valid
            ),
            "timestamps_strictly_monotonic_within_sequence": (
                group_timestamps_valid
            ),
            "declared_sanpo_synthetic_train_manifest_consistent": (
                declared_sanpo_synthetic_manifest_consistent
            ),
            "intrinsics_valid": intrinsics_valid and principal_point_valid,
            "camera_dimensions_match_manifest": camera_dimensions_match,
            "bound_files": files,
            "metric_qa": metric_qa,
            "raw_pose_inventory": raw_pose_inventory,
            "hftf_pose_binding_contract": pose_binding,
            "body_frame_contract": body_binding,
            "pose_body_camera_frame_consistent": camera_frame_consistent,
            "body_contract_structurally_valid_for_static_source": (
                body_contract_structurally_valid
            ),
            "future_mechanics_structure_ready_but_not_admitted": (
                future_mechanics_structure_ready
            ),
            "human_event_truth_present": event_truth_present,
            "human_event_truth_authorized": event_truth_authorized,
            "independent_parent_event_ledger_admitted": False,
        },
        "capability_decisions": {
            "static_metric_geometry_projection_canary": (
                "ELIGIBLE" if static_projection_canary else "NOT_EVALUABLE"
            ),
            "multi_height_human_envelope_teacher_canary": (
                "ELIGIBLE" if multi_height_teacher_canary else "NOT_EVALUABLE"
            ),
            "short_horizon_future_teacher_canary": (
                "ELIGIBLE" if future_teacher_canary else "NOT_EVALUABLE"
            ),
            "student_effect_evaluation": (
                "ELIGIBLE" if independent_effect_evaluation else "NOT_EVALUABLE"
            ),
        },
        "blockers": blockers,
        "allowed_next_step": (
            "repair_or_replace_source_contract"
            if not source_integrity
            else "static_metric_geometry_projection_canary_only"
        ),
        "prohibited_inferences": [
            "teacher_proxy_is_human_event_truth",
            "student_effectiveness",
            "research_mainline_promotion",
            "android_runtime_authorization",
            "production_or_safety_claim",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-root",
        type=Path,
        required=True,
        help="Root containing dataset_spec.json and manifest.replay.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON report path under artifacts.local/",
    )
    return parser.parse_args()


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    output = path.resolve()
    try:
        output.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {output}"
        ) from exc
    return output


def main() -> int:
    args = _parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output}")
        report = audit_replay(args.replay_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({"terminal": report["terminal"], "output": str(output)}))
        return 0
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
