from __future__ import annotations

"""Strict truth-blind CARLA RGB-D model-contract adapter.

This module deliberately accepts only a sanitized ``model`` root.  It has no
evaluator, raw-shard, sidecar, instance-segmentation, actor-state, or contact
loader.  Every JSON object is exact-key validated, every referenced file must
remain inside the supplied model root, and every reference is SHA-256 bound.

The image helpers are resolution independent.  Mask geometry is sampled on a
fixed 160 x 90 angular grid so increasing capture resolution does not silently
multiply the depth-support denominator.  Camera points use
``x=forward, y=left, z=up``; anchor points use
``x=forward, y=right, z=up``.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT_SCHEMA = "dtr-carla-rgbd-model-root-v1"
CALIBRATION_SCHEMA = "dtr-carla-rgbd-calibration-v1"
EPISODE_SCHEMA = "dtr-carla-rgbd-model-episode-v1"
OBSERVATION_SCHEMA = "dtr-carla-rgbd-model-observation-v1"
PLAN_SCHEMA = "dtr-c1-plan-receipt-v1"
DEPTH_ENCODING = "CARLA_BGR24_NORMALIZED"

C2_ROOT_SCHEMA_V1 = "dtr-c2-model-root-manifest-v1"
C2_ROOT_SCHEMA = "dtr-c2-model-root-manifest-v2"
C2_CALIBRATION_SCHEMA = "dtr-c2-model-camera-contract-v1"
C2_MODEL_CONTRACT_SCHEMA_V1 = "dtr-c2-model-contract-v1"
C2_MODEL_CONTRACT_SCHEMA = "dtr-c2-model-contract-v2"
C2_EPISODE_SCHEMA_V1 = "dtr-c2-model-episode-manifest-v1"
C2_EPISODE_SCHEMA = "dtr-c2-model-episode-manifest-v2"
C2_OBSERVATION_SCHEMA_V1 = "dtr-c2-model-observation-v1"
C2_OBSERVATION_SCHEMA = "dtr-c2-model-observation-v2"
C2_PLAN_WRAPPER_SCHEMA = "dtr-c2-model-plan-v1"
C2_PLAN_RECEIPT_SCHEMA = "dtr-c2-plan-receipt-v1"
C2_ALIGNMENT_RECEIPT_SCHEMA = (
    "dtr-c2-model-rgbd-deterministic-replay-alignment-receipt-v1"
)
C2_ALIGNMENT_AUTHORITY = "DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED"
C2_ALIGNMENT_WORLD_FRAME_RULE = (
    "world_frame equals wearable_rgb.source_world_frame; metric_depth is "
    "mapped into that namespace by the verified per-episode source offset"
)
C2_DEPTH_ENCODING = "CARLA_RGB24_NORMALIZED_DEPTH"
C2_DEPTH_FORMULA = "meters=1000*(R+256*G+65536*B)/(16777215)"

ANGULAR_GRID_WIDTH = 160
ANGULAR_GRID_HEIGHT = 90
MINIMUM_MASK_DEPTH_POINTS = 32
NEAR_DEPTH_QUANTILE = 0.15
MINIMUM_NEAR_SLAB_M = 0.60
MAXIMUM_NEAR_SLAB_M = 1.25
NEAR_SLAB_DEPTH_RATIO = 0.10
MINIMUM_DEPTH_M = 0.05
EPSILON = 1e-9

_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")

# These are exact path components, not substrings.  A file such as
# ``depth_raw_encoding.md`` is irrelevant because this adapter accepts only
# schema-declared JSON and image references, while a ``raw`` directory is a
# hard boundary violation.
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "evaluator",
        "evaluators",
        "instance",
        "oracle",
        "raw",
        "shard",
        "shards",
        "sidecar",
        "sidecars",
        "truth",
    }
)

FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "actors",
        "actor",
        "actor_ids",
        "actual_future_trajectory",
        "actor_id",
        "actor_state",
        "adherence_truth",
        "carla_actor_id",
        "collision_polygons_xy",
        "bbox",
        "bounding_box",
        "contact",
        "contact_label",
        "current_actors",
        "current_contact",
        "evaluator",
        "executed_route",
        "expected_contact",
        "expected_outcome",
        "first_contact_time_s",
        "future_contact",
        "future_contact_within_horizon",
        "future_pose",
        "gt_cv_route_risk",
        "gt_cv_ttc_seconds",
        "instance",
        "instance_path",
        "instance_visibility",
        "layout_id",
        "layout_role",
        "minimum_distance_m",
        "obb",
        "physical_loss",
        "occlusion",
        "occlusion_label",
        "realized_path",
        "realized_time_to_contact_seconds",
        "responsible_actor",
        "responsible_asset",
        "responsible_assets",
        "realized_future",
        "role",
        "route_truth",
        "scenario_role",
        "semantic_role",
        "target_obb_polygon_xy",
        "truth",
        "twin_role",
        "velocity",
        "witness",
        "witness_path",
    }
)

ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "truth_blind",
        "contains_evaluator_truth",
        "calibration",
        "episodes",
    }
)
FILE_REFERENCE_FIELDS = frozenset({"path", "sha256"})
IMAGE_REFERENCE_FIELDS = frozenset({"path", "sha256", "bytes", "width", "height"})
EPISODE_REFERENCE_FIELDS = frozenset({"episode_id", "path", "sha256"})
CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "resolution",
        "horizontal_fov_degrees",
        "depth_encoding",
        "depth_max_m",
    }
)
ANCHOR_FIELDS = frozenset({"center_xy_m", "z_origin_m", "forward_xy", "right_xy"})
EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "observation_count",
        "observations",
        "route_frame",
    }
)
OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "sample_index",
        "time_s",
        "world_frame",
        "navigation_session_id",
        "camera_transform",
        "wearable_rgb",
        "wearable_depth",
        "wearer",
        "issued_plan",
    }
)
TRANSFORM_FIELDS = frozenset({"pitch", "roll", "x", "y", "yaw", "z"})
XYZ_FIELDS = frozenset({"x", "y", "z"})
WEARER_FIELDS = frozenset(
    {"track_id", "transform", "command_velocity", "bounding_box_extent"}
)
ISSUED_PLAN_FIELDS = frozenset({"authority", "path", "receipt_sha256"})
PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "coordinate_frame",
        "plan_id",
        "session_id",
        "issued_at_s",
        "valid_from_s",
        "expires_at_s",
        "time_parameterized_waypoints",
        "receipt_sha256",
    }
)
WAYPOINT_FIELDS = frozenset({"time_s", "forward_m", "right_m"})

C2_ROOT_FIELDS_V1 = frozenset(
    {"schema_version", "experiment_id", "camera_calibration", "episodes"}
)
C2_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "camera_calibration",
        "model_contract",
        "rgbd_alignment_receipt",
        "episodes",
    }
)
C2_ROOT_EPISODE_REFERENCE_FIELDS = frozenset(
    {"episode_id", "manifest_path", "manifest_sha256"}
)
C2_MODEL_CONTRACT_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "current_actors_enabled",
        "dense_modalities",
        "evaluator_sibling_not_required",
        "record_top_level_allowlist",
    }
)
C2_MODEL_CONTRACT_FIELDS = frozenset(
    set(C2_MODEL_CONTRACT_FIELDS_V1) | {"rgbd_alignment"}
)
C2_OBSERVATION_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "episode_id",
        "sample_index",
        "world_frame",
        "time_s",
        "timestamp_s",
        "wearable_rgb",
        "metric_depth",
        "camera",
        "wearer_pose_current",
        "navigation",
    }
)
C2_OBSERVATION_FIELDS = frozenset(set(C2_OBSERVATION_FIELDS_V1) | {"frame_alignment"})
C2_CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "resolution",
        "fov_degrees",
        "K",
        "depth_codec",
        "wearable_rigid_extrinsic",
        "sensor_tick_seconds",
    }
)
C2_RESOLUTION_FIELDS = frozenset({"width", "height"})
C2_DEPTH_CODEC_FIELDS = frozenset({"name", "maximum_depth_m", "formula"})
C2_RIGID_EXTRINSIC_FIELDS = frozenset(
    {"x_m", "y_m", "z_m", "pitch_degrees", "yaw_degrees", "roll_degrees"}
)
C2_EPISODE_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "episode_id",
        "frames",
        "observations_sha256",
        "rgb_payloads",
        "depth_payloads",
        "navigation_session_id",
        "issued_plan",
    }
)
C2_EPISODE_FIELDS = frozenset(set(C2_EPISODE_FIELDS_V1) | {"rgbd_alignment"})
C2_EPISODE_PLAN_REFERENCE_FIELDS = frozenset(
    {"authority", "path", "receipt_sha256", "file_sha256"}
)
C2_CAMERA_FIELDS = frozenset(
    {"world_transform", "rigid_extrinsic", "width", "height", "fov_degrees", "K"}
)
C2_IMAGE_REFERENCE_FIELDS = frozenset(set(IMAGE_REFERENCE_FIELDS) | {"source_world_frame"})
C2_DEPTH_REFERENCE_FIELDS_V1 = frozenset(
    {"path", "sha256", "bytes", "width", "height", "codec"}
)
C2_DEPTH_REFERENCE_FIELDS = frozenset(
    set(C2_DEPTH_REFERENCE_FIELDS_V1) | {"source_world_frame"}
)
C2_NAVIGATION_FIELDS = frozenset({"navigation_session_id", "issued_plan"})
C2_OBSERVATION_PLAN_REFERENCE_FIELDS = frozenset(
    {"authority", "path", "receipt_sha256"}
)
C2_PLAN_WRAPPER_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "navigation_session_id",
        "layout_anchor",
        "issued_plan",
    }
)
C2_LAYOUT_ANCHOR_FIELDS = frozenset(
    {"world_center_xy_m", "world_forward_xy", "world_right_xy"}
)
C2_PLAN_WRAPPER_ISSUED_FIELDS = frozenset(
    {
        "authority",
        "receipt",
        "receipt_sha256",
        "world_coordinate_frame",
        "time_parameterized_waypoints_world",
    }
)
C2_PLAN_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "plan_id",
        "session_id",
        "issued_at_s",
        "expires_at_s",
        "coordinate_frame",
        "time_parameterized_waypoints",
        "receipt_sha256",
    }
)
C2_WORLD_WAYPOINT_FIELDS = frozenset({"time_s", "x_m", "y_m"})
C2_ROOT_ALIGNMENT_REFERENCE_FIELDS = frozenset(
    {"path", "receipt_sha256", "sha256"}
)
C2_MODEL_ALIGNMENT_FIELDS = frozenset(
    {
        "authority",
        "receipt_path",
        "receipt_sha256",
        "file_sha256",
        "world_frame_rule",
    }
)
C2_EPISODE_ALIGNMENT_FIELDS = frozenset(
    {
        "authority",
        "receipt_path",
        "receipt_sha256",
        "depth_minus_wearable_source_world_frame_offset",
    }
)
C2_FRAME_ALIGNMENT_FIELDS = frozenset(
    {
        "authority",
        "reference_modality",
        "receipt_path",
        "receipt_sha256",
        "depth_minus_wearable_source_world_frame_offset",
    }
)
C2_ALIGNMENT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "authority",
        "world_frame_rule",
        "matching_keys",
        "verified_equal_fields",
        "episodes",
        "receipt_sha256",
    }
)
C2_ALIGNMENT_EPISODE_FIELDS = frozenset(
    {
        "episode_id",
        "frames",
        "wearable_source_world_frame_first",
        "wearable_source_world_frame_last",
        "depth_source_world_frame_first",
        "depth_source_world_frame_last",
        "depth_minus_wearable_source_world_frame_offset",
        "alignment_projection_sha256",
    }
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(str(key) for key in value)
    require(actual == expected, f"{label}_fields:missing={sorted(expected - actual)}:extra={sorted(actual - expected)}")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label}_nonfinite")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    require(result > 0 and not isinstance(value, bool), f"{label}_not_positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = int(value)
    require(result >= 0 and not isinstance(value, bool), f"{label}_negative")
    return result


def _strict_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label}_not_integer")
    return int(value)


def _sha256(value: Any, label: str) -> str:
    result = str(value)
    require(bool(_SHA256.fullmatch(result)), f"{label}_invalid")
    return result.upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def assert_sanitized_model_value(value: Any, path: str = "root") -> None:
    """Reject privileged keys anywhere before schema-specific validation."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            require(
                normalized not in FORBIDDEN_MODEL_KEYS
                and not normalized.endswith("_actor_id"),
                f"rgbd_model_forbidden_key:{path}.{key}",
            )
            assert_sanitized_model_value(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_sanitized_model_value(child, f"{path}[{index}]")


def resolve_model_path(model_root: Path, raw_path: str | Path, label: str) -> Path:
    """Resolve a schema path without permitting absolute or symlink escape."""

    root = model_root.resolve(strict=True)
    candidate = Path(raw_path)
    require(not candidate.is_absolute(), f"{label}_absolute_path:{raw_path}")
    require(
        not ({part.lower() for part in candidate.parts} & FORBIDDEN_PATH_PARTS),
        f"{label}_privileged_path:{raw_path}",
    )
    resolved = (root / candidate).resolve(strict=True)
    require(resolved.is_relative_to(root), f"{label}_path_escape:{raw_path}")
    require(resolved.is_file(), f"{label}_not_file:{resolved}")
    return resolved


def validate_file_hash(path: Path, expected_sha256: str, label: str) -> None:
    expected = _sha256(expected_sha256, f"{label}_sha256")
    require(sha256_file(path) == expected, f"{label}_hash_mismatch:{path}")


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label}_not_object:{path}")
    assert_sanitized_model_value(value, label)
    return value


def _read_jsonl_file(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{label}_line_not_object:{line_number}")
            assert_sanitized_model_value(value, f"{label}[{line_number}]")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class FileReference:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ImageReference(FileReference):
    bytes: int
    width: int
    height: int
    source_world_frame: int | None = None


@dataclass(frozen=True)
class AnchorFrame:
    center_xy_m: tuple[float, float]
    z_origin_m: float
    forward_xy: tuple[float, float]
    right_xy: tuple[float, float]


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    horizontal_fov_degrees: float
    depth_max_m: float
    depth_encoding: str = DEPTH_ENCODING
    intrinsic_matrix: tuple[tuple[float, float, float], ...] | None = None
    wearable_rigid_extrinsic: Mapping[str, float] | None = None
    sensor_tick_seconds: float | None = None

    @property
    def intrinsic(self) -> np.ndarray:
        if self.intrinsic_matrix is not None:
            return np.asarray(self.intrinsic_matrix, dtype=np.float64)
        return camera_intrinsic(
            self.width,
            self.height,
            self.horizontal_fov_degrees,
        )


@dataclass(frozen=True)
class FrameObservation:
    episode_id: str
    sample_index: int
    time_s: float
    world_frame: int
    navigation_session_id: str
    camera_transform: Mapping[str, float]
    rgb: ImageReference
    depth: ImageReference
    wearer: Mapping[str, Any]
    issued_plan: Mapping[str, Any]
    frame_alignment: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    route_frame: AnchorFrame
    observations: tuple[FrameObservation, ...]
    rgbd_alignment: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SanitizedModelContract:
    model_root: Path
    manifest_path: Path
    manifest_sha256: str
    experiment_id: str
    calibration: CameraCalibration
    episodes: tuple[Episode, ...]
    source_schema_version: str = ROOT_SCHEMA
    rgbd_alignment: Mapping[str, Any] | None = None

    def episode(self, episode_id: str) -> Episode:
        matches = [value for value in self.episodes if value.episode_id == episode_id]
        require(len(matches) == 1, f"rgbd_model_unknown_episode:{episode_id}")
        return matches[0]


@dataclass(frozen=True)
class MaskDepthMeasurement:
    valid: bool
    grid_support: int
    foreground_support: int
    depth_q15_m: float | None
    slab_m: float | None
    position_camera_flu_m: tuple[float, float, float] | None
    position_world_m: tuple[float, float, float] | None
    position_anchor_fru_m: tuple[float, float, float] | None


def _parse_file_reference(
    model_root: Path,
    value: Any,
    label: str,
) -> FileReference:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, FILE_REFERENCE_FIELDS, label)
    path = resolve_model_path(model_root, str(value["path"]), label)
    expected = _sha256(value["sha256"], f"{label}_sha256")
    validate_file_hash(path, expected, label)
    return FileReference(path=path, sha256=expected)


def _parse_image_reference(
    model_root: Path,
    value: Any,
    calibration: CameraCalibration,
    label: str,
    *,
    validate_payload_hash: bool,
) -> ImageReference:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, IMAGE_REFERENCE_FIELDS, label)
    path = resolve_model_path(model_root, str(value["path"]), label)
    expected = _sha256(value["sha256"], f"{label}_sha256")
    expected_bytes = _positive_int(value["bytes"], f"{label}_bytes")
    width = _positive_int(value["width"], f"{label}_width")
    height = _positive_int(value["height"], f"{label}_height")
    require(
        (width, height) == (calibration.width, calibration.height),
        f"{label}_resolution:{width}x{height}",
    )
    require(path.stat().st_size == expected_bytes, f"{label}_size_mismatch:{path}")
    if validate_payload_hash:
        validate_file_hash(path, expected, label)
    return ImageReference(path, expected, expected_bytes, width, height)


def _unit_xy(value: Any, label: str) -> tuple[float, float]:
    require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{label}_not_pair")
    require(len(value) == 2, f"{label}_length")
    x, y = _finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]")
    norm = math.hypot(x, y)
    require(norm > EPSILON, f"{label}_zero")
    return x / norm, y / norm


def _parse_calibration(value: Mapping[str, Any]) -> CameraCalibration:
    _exact_fields(value, CALIBRATION_FIELDS, "rgbd_calibration")
    require(value["schema_version"] == CALIBRATION_SCHEMA, "rgbd_calibration_schema")
    resolution = value["resolution"]
    require(
        isinstance(resolution, Sequence)
        and not isinstance(resolution, (str, bytes))
        and len(resolution) == 2,
        "rgbd_calibration_resolution",
    )
    width = _positive_int(resolution[0], "rgbd_calibration_width")
    height = _positive_int(resolution[1], "rgbd_calibration_height")
    require(
        width >= ANGULAR_GRID_WIDTH and height >= ANGULAR_GRID_HEIGHT,
        "rgbd_calibration_below_angular_grid",
    )
    fov = _finite(value["horizontal_fov_degrees"], "rgbd_calibration_fov")
    require(0.0 < fov < 180.0, "rgbd_calibration_fov_range")
    require(value["depth_encoding"] == DEPTH_ENCODING, "rgbd_calibration_depth_encoding")
    depth_max_m = _finite(value["depth_max_m"], "rgbd_calibration_depth_max")
    require(depth_max_m > MINIMUM_DEPTH_M, "rgbd_calibration_depth_max_range")

    return CameraCalibration(width, height, fov, depth_max_m)


def _parse_anchor(value: Any, label: str) -> AnchorFrame:
    anchor_value = value
    require(isinstance(anchor_value, Mapping), "rgbd_calibration_anchor_not_object")
    _exact_fields(anchor_value, ANCHOR_FIELDS, label)
    center = anchor_value["center_xy_m"]
    require(
        isinstance(center, Sequence)
        and not isinstance(center, (str, bytes))
        and len(center) == 2,
        "rgbd_calibration_anchor_center",
    )
    center_xy = (
        _finite(center[0], "rgbd_calibration_anchor_center[0]"),
        _finite(center[1], "rgbd_calibration_anchor_center[1]"),
    )
    forward = _unit_xy(anchor_value["forward_xy"], "rgbd_calibration_anchor_forward")
    right = _unit_xy(anchor_value["right_xy"], "rgbd_calibration_anchor_right")
    require(abs(forward[0] * right[0] + forward[1] * right[1]) <= 1e-5, "rgbd_calibration_anchor_not_orthogonal")
    determinant = forward[0] * right[1] - forward[1] * right[0]
    require(determinant > 1.0 - 1e-5, "rgbd_calibration_anchor_not_forward_right_handed")
    anchor = AnchorFrame(
        center_xy_m=center_xy,
        z_origin_m=_finite(anchor_value["z_origin_m"], "rgbd_calibration_anchor_z"),
        forward_xy=forward,
        right_xy=right,
    )
    return anchor


def _parse_transform(value: Any, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, TRANSFORM_FIELDS, label)
    return {key: _finite(value[key], f"{label}.{key}") for key in sorted(TRANSFORM_FIELDS)}


def _parse_xyz(value: Any, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, XYZ_FIELDS, label)
    return {key: _finite(value[key], f"{label}.{key}") for key in sorted(XYZ_FIELDS)}


def _parse_wearer(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, WEARER_FIELDS, label)
    track_id = str(value["track_id"])
    require(bool(track_id), f"{label}_track_id")
    return {
        "track_id": track_id,
        "transform": _parse_transform(value["transform"], f"{label}.transform"),
        "command_velocity": _parse_xyz(value["command_velocity"], f"{label}.command_velocity"),
        "bounding_box_extent": _parse_xyz(value["bounding_box_extent"], f"{label}.bounding_box_extent"),
    }


def _validate_plan_receipt(value: Mapping[str, Any], expected_sha256: str, label: str) -> None:
    _exact_fields(value, PLAN_FIELDS, label)
    require(value["schema_version"] == PLAN_SCHEMA, f"{label}_schema")
    require(value["coordinate_frame"] == "ANCHOR_FORWARD_RIGHT", f"{label}_coordinate_frame")
    for key in ("plan_id", "session_id"):
        require(bool(str(value[key])), f"{label}_{key}")
    issued = _finite(value["issued_at_s"], f"{label}_issued")
    valid_from = _finite(value["valid_from_s"], f"{label}_valid_from")
    expires = _finite(value["expires_at_s"], f"{label}_expires")
    require(valid_from <= expires and issued <= expires, f"{label}_time_order")
    waypoints = value["time_parameterized_waypoints"]
    require(isinstance(waypoints, list) and len(waypoints) >= 2, f"{label}_waypoints")
    previous_time = -math.inf
    for index, waypoint in enumerate(waypoints):
        require(isinstance(waypoint, Mapping), f"{label}_waypoint_object:{index}")
        _exact_fields(waypoint, WAYPOINT_FIELDS, f"{label}.waypoints[{index}]")
        time_s = _finite(waypoint["time_s"], f"{label}_waypoint_time:{index}")
        _finite(waypoint["forward_m"], f"{label}_waypoint_forward:{index}")
        _finite(waypoint["right_m"], f"{label}_waypoint_right:{index}")
        require(time_s > previous_time, f"{label}_waypoint_order:{index}")
        previous_time = time_s
    supplied = _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
    require(supplied == expected_sha256, f"{label}_receipt_reference_mismatch")
    payload = {key: child for key, child in value.items() if key != "receipt_sha256"}
    computed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    require(computed == supplied, f"{label}_receipt_hash_mismatch")


def _parse_issued_plan(
    model_root: Path,
    value: Any,
    label: str,
    receipt_cache: dict[Path, str],
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, ISSUED_PLAN_FIELDS, label)
    authority = str(value["authority"])
    require(authority in {"VALID", "EXPIRED", "NO_PLAN"}, f"{label}_authority:{authority}")
    if authority == "NO_PLAN":
        require(value["path"] is None and value["receipt_sha256"] is None, f"{label}_no_plan_reference")
        return {"authority": authority, "path": None, "receipt_sha256": None}
    require(isinstance(value["path"], str) and value["path"], f"{label}_path")
    receipt_sha = _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
    path = resolve_model_path(model_root, value["path"], f"{label}_receipt")
    cached = receipt_cache.get(path)
    if cached is None:
        receipt = _read_json_file(path, f"{label}_receipt")
        _validate_plan_receipt(receipt, receipt_sha, f"{label}_receipt")
        receipt_cache[path] = receipt_sha
    else:
        require(cached == receipt_sha, f"{label}_receipt_cache_mismatch")
    return {"authority": authority, "path": str(path), "receipt_sha256": receipt_sha}


def _parse_observation(
    model_root: Path,
    value: Mapping[str, Any],
    calibration: CameraCalibration,
    episode_id: str,
    expected_index: int,
    receipt_cache: dict[Path, str],
    *,
    validate_payload_hashes: bool,
) -> FrameObservation:
    label = f"rgbd_observation:{episode_id}:{expected_index}"
    _exact_fields(value, OBSERVATION_FIELDS, label)
    require(value["schema_version"] == OBSERVATION_SCHEMA, f"{label}_schema")
    require(str(value["episode_id"]) == episode_id, f"{label}_episode")
    sample_index = _nonnegative_int(value["sample_index"], f"{label}_sample_index")
    require(sample_index == expected_index, f"{label}_sample_order:{sample_index}")
    time_s = _finite(value["time_s"], f"{label}_time")
    world_frame = _nonnegative_int(value["world_frame"], f"{label}_world_frame")
    navigation_session_id = str(value["navigation_session_id"])
    require(bool(navigation_session_id.strip()), f"{label}_navigation_session_id")
    camera_transform = _parse_transform(value["camera_transform"], f"{label}.camera_transform")
    rgb = _parse_image_reference(
        model_root,
        value["wearable_rgb"],
        calibration,
        f"{label}.wearable_rgb",
        validate_payload_hash=validate_payload_hashes,
    )
    depth = _parse_image_reference(
        model_root,
        value["wearable_depth"],
        calibration,
        f"{label}.wearable_depth",
        validate_payload_hash=validate_payload_hashes,
    )
    wearer = _parse_wearer(value["wearer"], f"{label}.wearer")
    issued_plan = _parse_issued_plan(
        model_root,
        value["issued_plan"],
        f"{label}.issued_plan",
        receipt_cache,
    )
    return FrameObservation(
        episode_id=episode_id,
        sample_index=sample_index,
        time_s=time_s,
        world_frame=world_frame,
        navigation_session_id=navigation_session_id,
        camera_transform=camera_transform,
        rgb=rgb,
        depth=depth,
        wearer=wearer,
        issued_plan=issued_plan,
    )


def _load_custom_model_contract(
    model_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    validate_payload_hashes: bool = True,
) -> SanitizedModelContract:
    """Load and fully validate a sanitized model root.

    Episode count, episode ids, frame count, and image resolution are taken
    from the sealed contract rather than hard-coded.  Observations must be
    dense: every row carries hashed RGB and depth references.
    """

    root = model_root.resolve(strict=True)
    require(root.is_dir(), f"rgbd_model_root_not_directory:{root}")
    require(
        not ({part.lower() for part in root.parts} & FORBIDDEN_PATH_PARTS),
        f"rgbd_model_root_privileged:{root}",
    )
    manifest_path = resolve_model_path(root, "manifest.json", "rgbd_model_manifest")
    manifest_hash = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        require(
            manifest_hash == _sha256(expected_manifest_sha256, "rgbd_model_manifest_expected"),
            "rgbd_model_manifest_hash_mismatch",
        )
    manifest = _read_json_file(manifest_path, "rgbd_model_manifest")
    _exact_fields(manifest, ROOT_FIELDS, "rgbd_model_manifest")
    require(manifest["schema_version"] == ROOT_SCHEMA, "rgbd_model_manifest_schema")
    require(manifest["truth_blind"] is True, "rgbd_model_manifest_not_truth_blind")
    require(manifest["contains_evaluator_truth"] is False, "rgbd_model_manifest_contains_truth")
    experiment_id = str(manifest["experiment_id"])
    require(bool(experiment_id), "rgbd_model_manifest_experiment")

    calibration_reference = _parse_file_reference(root, manifest["calibration"], "rgbd_calibration_reference")
    calibration_value = _read_json_file(calibration_reference.path, "rgbd_calibration")
    calibration = _parse_calibration(calibration_value)

    references = manifest["episodes"]
    require(isinstance(references, list) and references, "rgbd_model_manifest_episodes")
    seen: set[str] = set()
    episodes: list[Episode] = []
    receipt_cache: dict[Path, str] = {}
    for episode_number, reference in enumerate(references):
        label = f"rgbd_episode_reference:{episode_number}"
        require(isinstance(reference, Mapping), f"{label}_not_object")
        _exact_fields(reference, EPISODE_REFERENCE_FIELDS, label)
        episode_id = str(reference["episode_id"])
        require(bool(_EPISODE_ID.fullmatch(episode_id)), f"{label}_episode_id:{episode_id}")
        require(episode_id not in seen, f"{label}_duplicate:{episode_id}")
        seen.add(episode_id)
        episode_manifest_path = resolve_model_path(root, str(reference["path"]), label)
        validate_file_hash(episode_manifest_path, str(reference["sha256"]), label)
        episode_manifest = _read_json_file(episode_manifest_path, f"rgbd_episode_manifest:{episode_id}")
        _exact_fields(episode_manifest, EPISODE_FIELDS, f"rgbd_episode_manifest:{episode_id}")
        require(episode_manifest["schema_version"] == EPISODE_SCHEMA, f"rgbd_episode_manifest_schema:{episode_id}")
        require(str(episode_manifest["episode_id"]) == episode_id, f"rgbd_episode_manifest_identity:{episode_id}")
        observation_count = _positive_int(
            episode_manifest["observation_count"],
            f"rgbd_episode_observation_count:{episode_id}",
        )
        observation_reference = _parse_file_reference(
            root,
            episode_manifest["observations"],
            f"rgbd_observation_reference:{episode_id}",
        )
        observation_values = _read_jsonl_file(
            observation_reference.path,
            f"rgbd_observations:{episode_id}",
        )
        require(
            len(observation_values) == observation_count,
            f"rgbd_observation_count_mismatch:{episode_id}:{len(observation_values)}:{observation_count}",
        )
        observations = tuple(
            _parse_observation(
                root,
                value,
                calibration,
                episode_id,
                index,
                receipt_cache,
                validate_payload_hashes=validate_payload_hashes,
            )
            for index, value in enumerate(observation_values)
        )
        for previous, current in zip(observations, observations[1:]):
            require(current.time_s > previous.time_s, f"rgbd_observation_time_order:{episode_id}:{current.sample_index}")
            require(current.world_frame > previous.world_frame, f"rgbd_observation_world_frame_order:{episode_id}:{current.sample_index}")
        route_frame = _parse_anchor(
            episode_manifest["route_frame"],
            f"rgbd_episode_route_frame:{episode_id}",
        )
        episodes.append(Episode(episode_id, route_frame, observations))
    return SanitizedModelContract(
        model_root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        experiment_id=experiment_id,
        calibration=calibration,
        episodes=tuple(episodes),
        source_schema_version=ROOT_SCHEMA,
    )


def _rounded_alignment_value(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, Mapping):
        return {
            str(key): _rounded_alignment_value(child, digits)
            for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_rounded_alignment_value(child, digits) for child in value]
    return value


def _parse_c2_alignment_receipt(
    model_root: Path,
    reference: Any,
    experiment_id: str,
) -> dict[str, Any]:
    label = "rgbd_c2_alignment_receipt"
    require(isinstance(reference, Mapping), f"{label}_reference_not_object")
    _exact_fields(reference, C2_ROOT_ALIGNMENT_REFERENCE_FIELDS, f"{label}_reference")
    path = resolve_model_path(model_root, str(reference["path"]), f"{label}_reference")
    file_sha256 = _sha256(reference["sha256"], f"{label}_file_sha256")
    validate_file_hash(path, file_sha256, label)
    receipt_sha256 = _sha256(reference["receipt_sha256"], f"{label}_reference_receipt_sha256")
    value = _read_json_file(path, label)
    _exact_fields(value, C2_ALIGNMENT_RECEIPT_FIELDS, label)
    require(value["schema_version"] == C2_ALIGNMENT_RECEIPT_SCHEMA, f"{label}_schema")
    require(str(value["experiment_id"]) == experiment_id, f"{label}_experiment")
    require(value["authority"] == C2_ALIGNMENT_AUTHORITY, f"{label}_authority")
    require(value["world_frame_rule"] == C2_ALIGNMENT_WORLD_FRAME_RULE, f"{label}_world_frame_rule")
    require(
        value["matching_keys"] == ["episode_id", "sample_index", "time_s"],
        f"{label}_matching_keys",
    )
    require(
        value["verified_equal_fields"]
        == ["camera_world_transform", "wearer_pose_current"],
        f"{label}_verified_equal_fields",
    )
    supplied_receipt_sha256 = _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
    require(supplied_receipt_sha256 == receipt_sha256, f"{label}_reference_receipt_mismatch")
    payload = {key: child for key, child in value.items() if key != "receipt_sha256"}
    computed_receipt_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    require(computed_receipt_sha256 == receipt_sha256, f"{label}_canonical_hash_mismatch")
    episodes_value = value["episodes"]
    require(isinstance(episodes_value, list) and episodes_value, f"{label}_episodes")
    episode_map: dict[str, dict[str, Any]] = {}
    ordered_episode_ids: list[str] = []
    for index, episode_value in enumerate(episodes_value):
        episode_label = f"{label}.episodes[{index}]"
        require(isinstance(episode_value, Mapping), f"{episode_label}_not_object")
        _exact_fields(episode_value, C2_ALIGNMENT_EPISODE_FIELDS, episode_label)
        episode_id = str(episode_value["episode_id"])
        require(bool(_EPISODE_ID.fullmatch(episode_id)), f"{episode_label}_episode_id")
        require(episode_id not in episode_map, f"{episode_label}_duplicate")
        frames = _strict_int(episode_value["frames"], f"{episode_label}_frames")
        require(frames > 0, f"{episode_label}_frames_range")
        wearable_first = _strict_int(
            episode_value["wearable_source_world_frame_first"],
            f"{episode_label}_wearable_first",
        )
        wearable_last = _strict_int(
            episode_value["wearable_source_world_frame_last"],
            f"{episode_label}_wearable_last",
        )
        depth_first = _strict_int(
            episode_value["depth_source_world_frame_first"],
            f"{episode_label}_depth_first",
        )
        depth_last = _strict_int(
            episode_value["depth_source_world_frame_last"],
            f"{episode_label}_depth_last",
        )
        offset = _strict_int(
            episode_value["depth_minus_wearable_source_world_frame_offset"],
            f"{episode_label}_offset",
        )
        require(
            wearable_first >= 0
            and depth_first >= 0
            and wearable_last - wearable_first + 1 == frames
            and depth_last - depth_first + 1 == frames,
            f"{episode_label}_contiguous_range",
        )
        require(
            depth_first - wearable_first == offset
            and depth_last - wearable_last == offset,
            f"{episode_label}_offset_range_mismatch",
        )
        projection_sha256 = _sha256(
            episode_value["alignment_projection_sha256"],
            f"{episode_label}_projection_sha256",
        )
        episode_map[episode_id] = {
            "episode_id": episode_id,
            "frames": frames,
            "wearable_source_world_frame_first": wearable_first,
            "wearable_source_world_frame_last": wearable_last,
            "depth_source_world_frame_first": depth_first,
            "depth_source_world_frame_last": depth_last,
            "depth_minus_wearable_source_world_frame_offset": offset,
            "alignment_projection_sha256": projection_sha256,
        }
        ordered_episode_ids.append(episode_id)
    require(ordered_episode_ids == sorted(ordered_episode_ids), f"{label}_episode_order")
    return {
        "path": path,
        "relative_path": str(reference["path"]),
        "file_sha256": file_sha256,
        "receipt_sha256": receipt_sha256,
        "authority": C2_ALIGNMENT_AUTHORITY,
        "world_frame_rule": C2_ALIGNMENT_WORLD_FRAME_RULE,
        "episodes": episode_map,
        "receipt": dict(value),
    }


def _validate_c2_model_alignment(
    value: Any,
    alignment: Mapping[str, Any],
    label: str,
) -> None:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_MODEL_ALIGNMENT_FIELDS, label)
    require(value["authority"] == alignment["authority"], f"{label}_authority")
    require(value["receipt_path"] == alignment["relative_path"], f"{label}_receipt_path")
    require(
        _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
        == alignment["receipt_sha256"],
        f"{label}_receipt_sha256_mismatch",
    )
    require(
        _sha256(value["file_sha256"], f"{label}_file_sha256")
        == alignment["file_sha256"],
        f"{label}_file_sha256_mismatch",
    )
    require(value["world_frame_rule"] == alignment["world_frame_rule"], f"{label}_world_frame_rule")


def _validate_c2_episode_alignment(
    value: Any,
    alignment: Mapping[str, Any],
    episode_alignment: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_EPISODE_ALIGNMENT_FIELDS, label)
    require(value["authority"] == alignment["authority"], f"{label}_authority")
    require(value["receipt_path"] == alignment["relative_path"], f"{label}_receipt_path")
    require(
        _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
        == alignment["receipt_sha256"],
        f"{label}_receipt_sha256_mismatch",
    )
    offset = _strict_int(
        value["depth_minus_wearable_source_world_frame_offset"], f"{label}_offset"
    )
    require(
        offset == episode_alignment["depth_minus_wearable_source_world_frame_offset"],
        f"{label}_offset_mismatch",
    )
    return {
        "authority": alignment["authority"],
        "receipt_path": str(alignment["path"]),
        "receipt_sha256": alignment["receipt_sha256"],
        "file_sha256": alignment["file_sha256"],
        "frames": episode_alignment["frames"],
        "wearable_source_world_frame_first": episode_alignment[
            "wearable_source_world_frame_first"
        ],
        "wearable_source_world_frame_last": episode_alignment[
            "wearable_source_world_frame_last"
        ],
        "depth_source_world_frame_first": episode_alignment[
            "depth_source_world_frame_first"
        ],
        "depth_source_world_frame_last": episode_alignment[
            "depth_source_world_frame_last"
        ],
        "depth_minus_wearable_source_world_frame_offset": offset,
        "alignment_projection_sha256": episode_alignment["alignment_projection_sha256"],
    }


def _validate_c2_frame_alignment(
    value: Any,
    alignment: Mapping[str, Any],
    episode_alignment: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_FRAME_ALIGNMENT_FIELDS, label)
    require(value["authority"] == alignment["authority"], f"{label}_authority")
    require(value["reference_modality"] == "wearable_rgb", f"{label}_reference_modality")
    require(value["receipt_path"] == alignment["relative_path"], f"{label}_receipt_path")
    require(
        _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
        == alignment["receipt_sha256"],
        f"{label}_receipt_sha256_mismatch",
    )
    offset = _strict_int(
        value["depth_minus_wearable_source_world_frame_offset"], f"{label}_offset"
    )
    require(
        offset == episode_alignment["depth_minus_wearable_source_world_frame_offset"],
        f"{label}_offset_mismatch",
    )
    return {
        "authority": alignment["authority"],
        "reference_modality": "wearable_rgb",
        "receipt_path": str(alignment["path"]),
        "receipt_sha256": alignment["receipt_sha256"],
        "depth_minus_wearable_source_world_frame_offset": offset,
    }


def _c2_alignment_projection_sha256(
    observations: Sequence[FrameObservation],
) -> str:
    projection = [
        {
            "episode_id": observation.episode_id,
            "sample_index": observation.sample_index,
            "time_s": round(float(observation.time_s), 8),
            "world_frame": observation.world_frame,
            "camera_world_transform": _rounded_alignment_value(
                observation.camera_transform
            ),
            "wearer_pose_current": _rounded_alignment_value(
                observation.wearer["transform"]
            ),
        }
        for observation in observations
    ]
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest().upper()


def _parse_matrix3(value: Any, label: str) -> tuple[tuple[float, float, float], ...]:
    require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3,
        f"{label}_rows",
    )
    rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(value):
        require(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == 3,
            f"{label}_row:{row_index}",
        )
        rows.append(
            tuple(
                _finite(item, f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    require(abs(float(np.linalg.det(matrix))) > EPSILON, f"{label}_singular")
    return tuple(rows)


def _parse_c2_rigid_extrinsic(value: Any, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_RIGID_EXTRINSIC_FIELDS, label)
    return {
        key: _finite(value[key], f"{label}.{key}")
        for key in sorted(C2_RIGID_EXTRINSIC_FIELDS)
    }


def _parse_c2_depth_codec(value: Any, label: str) -> float:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_DEPTH_CODEC_FIELDS, label)
    require(value["name"] == C2_DEPTH_ENCODING, f"{label}_name")
    require(value["formula"] == C2_DEPTH_FORMULA, f"{label}_formula")
    maximum = _finite(value["maximum_depth_m"], f"{label}_maximum")
    require(abs(maximum - 1000.0) <= EPSILON, f"{label}_maximum_contract")
    return maximum


def _parse_c2_calibration(value: Mapping[str, Any]) -> CameraCalibration:
    label = "rgbd_c2_calibration"
    _exact_fields(value, C2_CALIBRATION_FIELDS, label)
    require(value["schema_version"] == C2_CALIBRATION_SCHEMA, f"{label}_schema")
    resolution = value["resolution"]
    require(isinstance(resolution, Mapping), f"{label}_resolution_not_object")
    _exact_fields(resolution, C2_RESOLUTION_FIELDS, f"{label}.resolution")
    width = _positive_int(resolution["width"], f"{label}_width")
    height = _positive_int(resolution["height"], f"{label}_height")
    require(
        width >= ANGULAR_GRID_WIDTH and height >= ANGULAR_GRID_HEIGHT,
        f"{label}_below_angular_grid",
    )
    fov = _finite(value["fov_degrees"], f"{label}_fov")
    require(0.0 < fov < 180.0, f"{label}_fov_range")
    intrinsic = _parse_matrix3(value["K"], f"{label}.K")
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    expected_intrinsic = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    require(
        np.allclose(np.asarray(intrinsic), expected_intrinsic, rtol=0.0, atol=1e-9),
        f"{label}_K_fov_resolution_mismatch",
    )
    depth_max_m = _parse_c2_depth_codec(value["depth_codec"], f"{label}.depth_codec")
    rigid_extrinsic = _parse_c2_rigid_extrinsic(
        value["wearable_rigid_extrinsic"], f"{label}.wearable_rigid_extrinsic"
    )
    sensor_tick = _finite(value["sensor_tick_seconds"], f"{label}_sensor_tick")
    require(sensor_tick > 0.0, f"{label}_sensor_tick_range")
    return CameraCalibration(
        width=width,
        height=height,
        horizontal_fov_degrees=fov,
        depth_max_m=depth_max_m,
        depth_encoding=C2_DEPTH_ENCODING,
        intrinsic_matrix=intrinsic,
        wearable_rigid_extrinsic=rigid_extrinsic,
        sensor_tick_seconds=sensor_tick,
    )


def _parse_c2_layout_anchor(value: Any, label: str) -> AnchorFrame:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_LAYOUT_ANCHOR_FIELDS, label)
    return _parse_anchor(
        {
            "center_xy_m": value["world_center_xy_m"],
            "z_origin_m": 0.0,
            "forward_xy": value["world_forward_xy"],
            "right_xy": value["world_right_xy"],
        },
        label,
    )


def _parse_c2_plan_receipt(
    value: Any,
    expected_sha256: str,
    navigation_session_id: str,
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(value, C2_PLAN_RECEIPT_FIELDS, label)
    require(value["schema_version"] == C2_PLAN_RECEIPT_SCHEMA, f"{label}_schema")
    require(value["coordinate_frame"] == "LAYOUT_FORWARD_RIGHT", f"{label}_coordinate_frame")
    require(bool(str(value["plan_id"]).strip()), f"{label}_plan_id")
    require(str(value["session_id"]) == navigation_session_id, f"{label}_session")
    issued_at = _finite(value["issued_at_s"], f"{label}_issued_at")
    expires_at = _finite(value["expires_at_s"], f"{label}_expires_at")
    require(issued_at <= expires_at, f"{label}_authority_order")
    waypoints = value["time_parameterized_waypoints"]
    require(isinstance(waypoints, list) and len(waypoints) >= 2, f"{label}_waypoints")
    previous_time = -math.inf
    for index, waypoint in enumerate(waypoints):
        require(isinstance(waypoint, Mapping), f"{label}_waypoint_not_object:{index}")
        _exact_fields(waypoint, WAYPOINT_FIELDS, f"{label}.waypoints[{index}]")
        time_s = _finite(waypoint["time_s"], f"{label}_waypoint_time:{index}")
        _finite(waypoint["forward_m"], f"{label}_waypoint_forward:{index}")
        _finite(waypoint["right_m"], f"{label}_waypoint_right:{index}")
        require(time_s > previous_time, f"{label}_waypoint_order:{index}")
        previous_time = time_s
    supplied = _sha256(value["receipt_sha256"], f"{label}_receipt_sha256")
    require(supplied == expected_sha256, f"{label}_receipt_reference_mismatch")
    payload = {key: child for key, child in value.items() if key != "receipt_sha256"}
    computed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    require(computed == supplied, f"{label}_receipt_hash_mismatch")
    return dict(value)


def _validate_c2_world_waypoints(
    value: Any,
    receipt: Mapping[str, Any],
    anchor: AnchorFrame,
    label: str,
) -> None:
    require(isinstance(value, list), f"{label}_not_list")
    local_waypoints = receipt["time_parameterized_waypoints"]
    require(len(value) == len(local_waypoints), f"{label}_count")
    for index, (world, local) in enumerate(zip(value, local_waypoints, strict=True)):
        require(isinstance(world, Mapping), f"{label}_not_object:{index}")
        _exact_fields(world, C2_WORLD_WAYPOINT_FIELDS, f"{label}[{index}]")
        local_forward = _finite(local["forward_m"], f"{label}_local_forward:{index}")
        local_right = _finite(local["right_m"], f"{label}_local_right:{index}")
        expected_x = (
            anchor.center_xy_m[0]
            + anchor.forward_xy[0] * local_forward
            + anchor.right_xy[0] * local_right
        )
        expected_y = (
            anchor.center_xy_m[1]
            + anchor.forward_xy[1] * local_forward
            + anchor.right_xy[1] * local_right
        )
        require(
            abs(_finite(world["time_s"], f"{label}_time:{index}") - float(local["time_s"])) <= 1e-8,
            f"{label}_time_mismatch:{index}",
        )
        require(
            abs(_finite(world["x_m"], f"{label}_x:{index}") - expected_x) <= 1e-4
            and abs(_finite(world["y_m"], f"{label}_y:{index}") - expected_y) <= 1e-4,
            f"{label}_position_mismatch:{index}",
        )


def _parse_c2_plan_wrapper(
    model_root: Path,
    plan_path: Path,
    plan_file_sha256: str,
    expected_episode_id: str,
    navigation_session_id: str,
    manifest_plan_reference: Mapping[str, Any],
) -> tuple[AnchorFrame, dict[str, Any]]:
    label = f"rgbd_c2_plan:{expected_episode_id}"
    value = _read_json_file(plan_path, label)
    _exact_fields(value, C2_PLAN_WRAPPER_FIELDS, label)
    require(value["schema_version"] == C2_PLAN_WRAPPER_SCHEMA, f"{label}_schema")
    require(str(value["episode_id"]) == expected_episode_id, f"{label}_episode")
    require(str(value["navigation_session_id"]) == navigation_session_id, f"{label}_session")
    anchor = _parse_c2_layout_anchor(value["layout_anchor"], f"{label}.layout_anchor")
    issued = value["issued_plan"]
    require(isinstance(issued, Mapping), f"{label}_issued_plan_not_object")
    _exact_fields(issued, C2_PLAN_WRAPPER_ISSUED_FIELDS, f"{label}.issued_plan")
    authority = str(issued["authority"])
    require(authority in {"VALID", "NO_PLAN"}, f"{label}_authority:{authority}")
    require(authority == str(manifest_plan_reference["authority"]), f"{label}_manifest_authority")
    require(issued["world_coordinate_frame"] == "CARLA_WORLD_XY", f"{label}_world_frame")
    receipt: dict[str, Any] | None
    receipt_sha256: str | None
    if authority == "NO_PLAN":
        require(
            issued["receipt"] is None
            and issued["receipt_sha256"] is None
            and manifest_plan_reference["receipt_sha256"] is None,
            f"{label}_no_plan_receipt",
        )
        require(issued["time_parameterized_waypoints_world"] == [], f"{label}_no_plan_waypoints")
        receipt = None
        receipt_sha256 = None
    else:
        receipt_sha256 = _sha256(issued["receipt_sha256"], f"{label}_receipt_sha256")
        require(
            receipt_sha256
            == _sha256(manifest_plan_reference["receipt_sha256"], f"{label}_manifest_receipt_sha256"),
            f"{label}_manifest_receipt_mismatch",
        )
        receipt = _parse_c2_plan_receipt(
            issued["receipt"], receipt_sha256, navigation_session_id, f"{label}.receipt"
        )
        _validate_c2_world_waypoints(
            issued["time_parameterized_waypoints_world"],
            receipt,
            anchor,
            f"{label}.world_waypoints",
        )
    return anchor, {
        "authority": authority,
        "path": str(plan_path) if receipt is not None else None,
        "receipt_sha256": receipt_sha256,
        "receipt": receipt,
        "source_plan_path": str(plan_path),
        "source_plan_sha256": plan_file_sha256,
    }


def _parse_c2_rgb_reference(
    model_root: Path,
    value: Any,
    calibration: CameraCalibration,
    label: str,
    *,
    validate_payload_hash: bool,
    sync_explicit: bool,
) -> ImageReference:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(
        value,
        C2_IMAGE_REFERENCE_FIELDS if sync_explicit else IMAGE_REFERENCE_FIELDS,
        label,
    )
    base = _parse_image_reference(
        model_root,
        {key: value[key] for key in IMAGE_REFERENCE_FIELDS},
        calibration,
        label,
        validate_payload_hash=validate_payload_hash,
    )
    source_world_frame = (
        _strict_int(value["source_world_frame"], f"{label}_source_world_frame")
        if sync_explicit
        else None
    )
    if source_world_frame is not None:
        require(source_world_frame >= 0, f"{label}_source_world_frame_range")
    return ImageReference(
        base.path,
        base.sha256,
        base.bytes,
        base.width,
        base.height,
        source_world_frame,
    )


def _parse_c2_depth_reference(
    model_root: Path,
    value: Any,
    calibration: CameraCalibration,
    label: str,
    *,
    validate_payload_hash: bool,
    sync_explicit: bool,
) -> ImageReference:
    require(isinstance(value, Mapping), f"{label}_not_object")
    _exact_fields(
        value,
        C2_DEPTH_REFERENCE_FIELDS if sync_explicit else C2_DEPTH_REFERENCE_FIELDS_V1,
        label,
    )
    _parse_c2_depth_codec(value["codec"], f"{label}.codec")
    base = _parse_image_reference(
        model_root,
        {key: value[key] for key in IMAGE_REFERENCE_FIELDS},
        calibration,
        label,
        validate_payload_hash=validate_payload_hash,
    )
    source_world_frame = (
        _strict_int(value["source_world_frame"], f"{label}_source_world_frame")
        if sync_explicit
        else None
    )
    if source_world_frame is not None:
        require(source_world_frame >= 0, f"{label}_source_world_frame_range")
    return ImageReference(
        base.path,
        base.sha256,
        base.bytes,
        base.width,
        base.height,
        source_world_frame,
    )


def _mapping_floats_equal(
    first: Mapping[str, float] | None,
    second: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    if first is None or set(first) != set(second):
        return False
    return all(abs(float(first[key]) - float(second[key])) <= tolerance for key in first)


def _parse_c2_observation(
    model_root: Path,
    value: Mapping[str, Any],
    calibration: CameraCalibration,
    episode_id: str,
    expected_index: int,
    navigation_session_id: str,
    manifest_plan_reference: Mapping[str, Any],
    normalized_issued_plan: Mapping[str, Any],
    previous_pose: Mapping[str, float] | None,
    previous_time_s: float | None,
    alignment: Mapping[str, Any] | None,
    episode_alignment: Mapping[str, Any] | None,
    *,
    validate_payload_hashes: bool,
    sync_explicit: bool,
) -> tuple[FrameObservation, dict[str, float]]:
    label = f"rgbd_c2_observation:{episode_id}:{expected_index}"
    _exact_fields(
        value,
        C2_OBSERVATION_FIELDS if sync_explicit else C2_OBSERVATION_FIELDS_V1,
        label,
    )
    require(
        value["schema_version"]
        == (C2_OBSERVATION_SCHEMA if sync_explicit else C2_OBSERVATION_SCHEMA_V1),
        f"{label}_schema",
    )
    require(str(value["episode_id"]) == episode_id, f"{label}_episode")
    sample_index = (
        _strict_int(value["sample_index"], f"{label}_sample_index")
        if sync_explicit
        else _nonnegative_int(value["sample_index"], f"{label}_sample_index")
    )
    require(sample_index >= 0, f"{label}_sample_index_range")
    require(sample_index == expected_index, f"{label}_sample_order:{sample_index}")
    time_s = _finite(value["time_s"], f"{label}_time")
    timestamp_s = _finite(value["timestamp_s"], f"{label}_timestamp")
    require(abs(time_s - timestamp_s) <= 1e-8, f"{label}_timestamp_mismatch")
    world_frame = (
        _strict_int(value["world_frame"], f"{label}_world_frame")
        if sync_explicit
        else _nonnegative_int(value["world_frame"], f"{label}_world_frame")
    )
    require(world_frame >= 0, f"{label}_world_frame_range")
    rgb = _parse_c2_rgb_reference(
        model_root,
        value["wearable_rgb"],
        calibration,
        f"{label}.wearable_rgb",
        validate_payload_hash=validate_payload_hashes,
        sync_explicit=sync_explicit,
    )
    depth = _parse_c2_depth_reference(
        model_root,
        value["metric_depth"],
        calibration,
        f"{label}.metric_depth",
        validate_payload_hash=validate_payload_hashes,
        sync_explicit=sync_explicit,
    )
    camera = value["camera"]
    require(isinstance(camera, Mapping), f"{label}_camera_not_object")
    _exact_fields(camera, C2_CAMERA_FIELDS, f"{label}.camera")
    camera_transform = _parse_transform(camera["world_transform"], f"{label}.camera.world_transform")
    camera_extrinsic = _parse_c2_rigid_extrinsic(
        camera["rigid_extrinsic"], f"{label}.camera.rigid_extrinsic"
    )
    require(
        _mapping_floats_equal(calibration.wearable_rigid_extrinsic, camera_extrinsic),
        f"{label}_camera_extrinsic_mismatch",
    )
    require(
        _positive_int(camera["width"], f"{label}_camera_width") == calibration.width
        and _positive_int(camera["height"], f"{label}_camera_height") == calibration.height,
        f"{label}_camera_resolution_mismatch",
    )
    require(
        abs(_finite(camera["fov_degrees"], f"{label}_camera_fov") - calibration.horizontal_fov_degrees)
        <= 1e-9,
        f"{label}_camera_fov_mismatch",
    )
    camera_intrinsic_value = _parse_matrix3(camera["K"], f"{label}.camera.K")
    require(
        np.allclose(
            np.asarray(camera_intrinsic_value), calibration.intrinsic, rtol=0.0, atol=1e-9
        ),
        f"{label}_camera_K_mismatch",
    )
    pose = _parse_transform(value["wearer_pose_current"], f"{label}.wearer_pose_current")
    if previous_pose is None:
        require(previous_time_s is None, f"{label}_previous_pose_time_pair")
        delta_s: float | None = None
        velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        velocity_source = "CAUSAL_INITIAL_ZERO_NO_PAST_POSE"
        velocity_valid = False
    else:
        require(previous_time_s is not None, f"{label}_previous_time_missing")
        delta_s = time_s - previous_time_s
        require(delta_s > 0.0, f"{label}_causal_velocity_delta")
        velocity = {
            axis: (pose[axis] - float(previous_pose[axis])) / delta_s
            for axis in ("x", "y", "z")
        }
        velocity_source = "CAUSAL_BACKWARD_DIFFERENCE"
        velocity_valid = True
    navigation = value["navigation"]
    require(isinstance(navigation, Mapping), f"{label}_navigation_not_object")
    _exact_fields(navigation, C2_NAVIGATION_FIELDS, f"{label}.navigation")
    require(
        str(navigation["navigation_session_id"]) == navigation_session_id,
        f"{label}_navigation_session",
    )
    observation_plan = navigation["issued_plan"]
    require(isinstance(observation_plan, Mapping), f"{label}_issued_plan_not_object")
    _exact_fields(
        observation_plan,
        C2_OBSERVATION_PLAN_REFERENCE_FIELDS,
        f"{label}.navigation.issued_plan",
    )
    require(
        dict(observation_plan)
        == {
            "authority": manifest_plan_reference["authority"],
            "path": manifest_plan_reference["path"],
            "receipt_sha256": manifest_plan_reference["receipt_sha256"],
        },
        f"{label}_issued_plan_manifest_mismatch",
    )
    frame_alignment: dict[str, Any] | None = None
    if sync_explicit:
        require(
            alignment is not None and episode_alignment is not None,
            f"{label}_alignment_context_missing",
        )
        frame_alignment = _validate_c2_frame_alignment(
            value["frame_alignment"],
            alignment,
            episode_alignment,
            f"{label}.frame_alignment",
        )
        require(
            rgb.source_world_frame is not None
            and depth.source_world_frame is not None,
            f"{label}_source_world_frames_missing",
        )
        require(
            world_frame == rgb.source_world_frame,
            f"{label}_world_frame_not_wearable_source",
        )
        require(
            depth.source_world_frame - rgb.source_world_frame
            == episode_alignment["depth_minus_wearable_source_world_frame_offset"],
            f"{label}_depth_wearable_offset",
        )
    else:
        require(
            alignment is None and episode_alignment is None,
            f"{label}_v1_alignment_context",
        )
    wearer = {
        "track_id": "wearer",
        "transform": pose,
        "command_velocity": velocity,
        "velocity_source": velocity_source,
        "velocity_valid": velocity_valid,
        "velocity_delta_s": delta_s,
    }
    return (
        FrameObservation(
            episode_id=episode_id,
            sample_index=sample_index,
            time_s=time_s,
            world_frame=world_frame,
            navigation_session_id=navigation_session_id,
            camera_transform=camera_transform,
            rgb=rgb,
            depth=depth,
            wearer=wearer,
            issued_plan=dict(normalized_issued_plan),
            frame_alignment=frame_alignment,
        ),
        pose,
    )


def _validate_c2_model_contract(
    model_root: Path,
    *,
    reference: Any | None,
    alignment: Mapping[str, Any] | None,
    sync_explicit: bool,
) -> None:
    if sync_explicit:
        model_contract_reference = _parse_file_reference(
            model_root, reference, "rgbd_c2_model_contract_reference"
        )
        path = model_contract_reference.path
    else:
        require(reference is None, "rgbd_c2_v1_model_contract_reference")
        path = resolve_model_path(model_root, "model_contract.json", "rgbd_c2_model_contract")
    value = _read_json_file(path, "rgbd_c2_model_contract")
    _exact_fields(
        value,
        C2_MODEL_CONTRACT_FIELDS if sync_explicit else C2_MODEL_CONTRACT_FIELDS_V1,
        "rgbd_c2_model_contract",
    )
    require(
        value["schema_version"]
        == (C2_MODEL_CONTRACT_SCHEMA if sync_explicit else C2_MODEL_CONTRACT_SCHEMA_V1),
        "rgbd_c2_model_contract_schema",
    )
    require(value["current_actors_enabled"] is False, "rgbd_c2_current_actors_enabled")
    require(value["evaluator_sibling_not_required"] is True, "rgbd_c2_evaluator_dependency")
    require(
        value["dense_modalities"] == ["wearable_rgb", "metric_depth"],
        "rgbd_c2_dense_modalities",
    )
    require(
        value["record_top_level_allowlist"]
        == sorted(C2_OBSERVATION_FIELDS if sync_explicit else C2_OBSERVATION_FIELDS_V1),
        "rgbd_c2_record_allowlist",
    )
    if sync_explicit:
        require(alignment is not None, "rgbd_c2_model_contract_alignment_missing")
        _validate_c2_model_alignment(
            value["rgbd_alignment"], alignment, "rgbd_c2_model_contract.rgbd_alignment"
        )
    else:
        require(alignment is None, "rgbd_c2_v1_alignment_present")


def _load_c2_model_contract(
    model_root: Path,
    *,
    expected_manifest_sha256: str | None,
    validate_payload_hashes: bool,
) -> SanitizedModelContract:
    root = model_root.resolve(strict=True)
    require(root.is_dir(), f"rgbd_model_root_not_directory:{root}")
    require(
        not ({part.lower() for part in root.parts} & FORBIDDEN_PATH_PARTS),
        f"rgbd_model_root_privileged:{root}",
    )
    manifest_path = resolve_model_path(root, "manifest.json", "rgbd_c2_model_manifest")
    manifest_hash = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        require(
            manifest_hash == _sha256(expected_manifest_sha256, "rgbd_c2_model_manifest_expected"),
            "rgbd_c2_model_manifest_hash_mismatch",
        )
    manifest = _read_json_file(manifest_path, "rgbd_c2_model_manifest")
    source_schema = manifest.get("schema_version")
    require(
        isinstance(source_schema, str)
        and source_schema in {C2_ROOT_SCHEMA_V1, C2_ROOT_SCHEMA},
        "rgbd_c2_model_manifest_schema",
    )
    sync_explicit = source_schema == C2_ROOT_SCHEMA
    _exact_fields(
        manifest,
        C2_ROOT_FIELDS if sync_explicit else C2_ROOT_FIELDS_V1,
        "rgbd_c2_model_manifest",
    )
    experiment_id = str(manifest["experiment_id"])
    require(bool(experiment_id.strip()), "rgbd_c2_model_manifest_experiment")
    alignment = (
        _parse_c2_alignment_receipt(
            root, manifest["rgbd_alignment_receipt"], experiment_id
        )
        if sync_explicit
        else None
    )
    _validate_c2_model_contract(
        root,
        reference=manifest["model_contract"] if sync_explicit else None,
        alignment=alignment,
        sync_explicit=sync_explicit,
    )
    calibration_reference = _parse_file_reference(
        root, manifest["camera_calibration"], "rgbd_c2_calibration_reference"
    )
    calibration_value = _read_json_file(calibration_reference.path, "rgbd_c2_calibration")
    calibration = _parse_c2_calibration(calibration_value)

    references = manifest["episodes"]
    require(isinstance(references, list) and references, "rgbd_c2_model_manifest_episodes")
    seen: set[str] = set()
    episodes: list[Episode] = []
    for episode_number, reference in enumerate(references):
        label = f"rgbd_c2_episode_reference:{episode_number}"
        require(isinstance(reference, Mapping), f"{label}_not_object")
        _exact_fields(reference, C2_ROOT_EPISODE_REFERENCE_FIELDS, label)
        episode_id = str(reference["episode_id"])
        require(bool(_EPISODE_ID.fullmatch(episode_id)), f"{label}_episode_id:{episode_id}")
        require(episode_id not in seen, f"{label}_duplicate:{episode_id}")
        seen.add(episode_id)
        episode_manifest_path = resolve_model_path(
            root, str(reference["manifest_path"]), label
        )
        validate_file_hash(
            episode_manifest_path, str(reference["manifest_sha256"]), label
        )
        episode_manifest = _read_json_file(
            episode_manifest_path, f"rgbd_c2_episode_manifest:{episode_id}"
        )
        _exact_fields(
            episode_manifest,
            C2_EPISODE_FIELDS if sync_explicit else C2_EPISODE_FIELDS_V1,
            f"rgbd_c2_episode_manifest:{episode_id}",
        )
        require(
            episode_manifest["schema_version"]
            == (C2_EPISODE_SCHEMA if sync_explicit else C2_EPISODE_SCHEMA_V1),
            f"rgbd_c2_episode_manifest_schema:{episode_id}",
        )
        require(
            str(episode_manifest["episode_id"]) == episode_id,
            f"rgbd_c2_episode_manifest_identity:{episode_id}",
        )
        frame_count = _positive_int(
            episode_manifest["frames"], f"rgbd_c2_episode_frames:{episode_id}"
        )
        require(
            _positive_int(
                episode_manifest["rgb_payloads"], f"rgbd_c2_episode_rgb_payloads:{episode_id}"
            )
            == frame_count
            and _positive_int(
                episode_manifest["depth_payloads"], f"rgbd_c2_episode_depth_payloads:{episode_id}"
            )
            == frame_count,
            f"rgbd_c2_episode_dense_payload_count:{episode_id}",
        )
        navigation_session_id = str(episode_manifest["navigation_session_id"])
        require(bool(navigation_session_id.strip()), f"rgbd_c2_episode_session:{episode_id}")
        alignment_episode: Mapping[str, Any] | None = None
        normalized_episode_alignment: Mapping[str, Any] | None = None
        if sync_explicit:
            require(alignment is not None, f"rgbd_c2_episode_alignment_root:{episode_id}")
            alignment_episode = alignment["episodes"].get(episode_id)
            require(
                alignment_episode is not None,
                f"rgbd_c2_episode_alignment_receipt_missing:{episode_id}",
            )
            require(
                alignment_episode["frames"] == frame_count,
                f"rgbd_c2_episode_alignment_frame_count:{episode_id}",
            )
            normalized_episode_alignment = _validate_c2_episode_alignment(
                episode_manifest["rgbd_alignment"],
                alignment,
                alignment_episode,
                f"rgbd_c2_episode_alignment:{episode_id}",
            )
        manifest_plan_reference = episode_manifest["issued_plan"]
        require(
            isinstance(manifest_plan_reference, Mapping),
            f"rgbd_c2_episode_plan_reference_not_object:{episode_id}",
        )
        _exact_fields(
            manifest_plan_reference,
            C2_EPISODE_PLAN_REFERENCE_FIELDS,
            f"rgbd_c2_episode_plan_reference:{episode_id}",
        )
        plan_authority = str(manifest_plan_reference["authority"])
        require(
            plan_authority in {"VALID", "NO_PLAN"},
            f"rgbd_c2_episode_plan_authority:{episode_id}:{plan_authority}",
        )
        require(
            isinstance(manifest_plan_reference["path"], str)
            and bool(str(manifest_plan_reference["path"]).strip()),
            f"rgbd_c2_episode_plan_path:{episode_id}",
        )
        if plan_authority == "NO_PLAN":
            require(
                manifest_plan_reference["receipt_sha256"] is None,
                f"rgbd_c2_episode_no_plan_receipt:{episode_id}",
            )
        else:
            _sha256(
                manifest_plan_reference["receipt_sha256"],
                f"rgbd_c2_episode_plan_receipt:{episode_id}",
            )
        plan_path = resolve_model_path(
            root,
            str(manifest_plan_reference["path"]),
            f"rgbd_c2_episode_plan:{episode_id}",
        )
        plan_file_sha256 = _sha256(
            manifest_plan_reference["file_sha256"],
            f"rgbd_c2_episode_plan_file:{episode_id}",
        )
        validate_file_hash(plan_path, plan_file_sha256, f"rgbd_c2_episode_plan:{episode_id}")
        route_frame, normalized_issued_plan = _parse_c2_plan_wrapper(
            root,
            plan_path,
            plan_file_sha256,
            episode_id,
            navigation_session_id,
            manifest_plan_reference,
        )

        observations_relative = (
            episode_manifest_path.parent.relative_to(root) / "observations.jsonl"
        ).as_posix()
        observations_path = resolve_model_path(
            root, observations_relative, f"rgbd_c2_observations:{episode_id}"
        )
        validate_file_hash(
            observations_path,
            str(episode_manifest["observations_sha256"]),
            f"rgbd_c2_observations:{episode_id}",
        )
        observation_values = _read_jsonl_file(
            observations_path, f"rgbd_c2_observations:{episode_id}"
        )
        require(
            len(observation_values) == frame_count,
            f"rgbd_c2_observation_count_mismatch:{episode_id}:{len(observation_values)}:{frame_count}",
        )
        parsed_observations: list[FrameObservation] = []
        previous_pose: Mapping[str, float] | None = None
        previous_time_s: float | None = None
        for expected_index, observation_value in enumerate(observation_values):
            observation, pose = _parse_c2_observation(
                root,
                observation_value,
                calibration,
                episode_id,
                expected_index,
                navigation_session_id,
                manifest_plan_reference,
                normalized_issued_plan,
                previous_pose,
                previous_time_s,
                alignment,
                alignment_episode,
                validate_payload_hashes=validate_payload_hashes,
                sync_explicit=sync_explicit,
            )
            if parsed_observations:
                previous = parsed_observations[-1]
                require(
                    observation.time_s > previous.time_s,
                    f"rgbd_c2_observation_time_order:{episode_id}:{observation.sample_index}",
                )
                require(
                    (
                        observation.world_frame == previous.world_frame + 1
                        if sync_explicit
                        else observation.world_frame > previous.world_frame
                    ),
                    f"rgbd_c2_observation_world_frame_order:{episode_id}:{observation.sample_index}",
                )
            parsed_observations.append(observation)
            previous_pose = pose
            previous_time_s = observation.time_s
        if sync_explicit:
            require(
                alignment_episode is not None and normalized_episode_alignment is not None,
                f"rgbd_c2_episode_alignment_context:{episode_id}",
            )
            wearable_source_frames = [
                observation.rgb.source_world_frame for observation in parsed_observations
            ]
            depth_source_frames = [
                observation.depth.source_world_frame for observation in parsed_observations
            ]
            require(
                all(value is not None for value in wearable_source_frames)
                and all(value is not None for value in depth_source_frames),
                f"rgbd_c2_episode_source_frames:{episode_id}",
            )
            require(
                wearable_source_frames[0]
                == alignment_episode["wearable_source_world_frame_first"]
                and wearable_source_frames[-1]
                == alignment_episode["wearable_source_world_frame_last"]
                and depth_source_frames[0]
                == alignment_episode["depth_source_world_frame_first"]
                and depth_source_frames[-1]
                == alignment_episode["depth_source_world_frame_last"],
                f"rgbd_c2_episode_source_frame_range:{episode_id}",
            )
            require(
                _c2_alignment_projection_sha256(parsed_observations)
                == alignment_episode["alignment_projection_sha256"],
                f"rgbd_c2_episode_alignment_projection_hash:{episode_id}",
            )
        episodes.append(
            Episode(
                episode_id,
                route_frame,
                tuple(parsed_observations),
                normalized_episode_alignment,
            )
        )
    if sync_explicit:
        require(alignment is not None, "rgbd_c2_alignment_missing")
        require(
            seen == set(alignment["episodes"]),
            "rgbd_c2_alignment_episode_set_mismatch",
        )
    return SanitizedModelContract(
        model_root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        experiment_id=experiment_id,
        calibration=calibration,
        episodes=tuple(episodes),
        source_schema_version=str(source_schema),
        rgbd_alignment=(dict(alignment) if alignment is not None else None),
    )


def load_model_contract(
    model_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    validate_payload_hashes: bool = True,
) -> SanitizedModelContract:
    """Load either the custom v1 contract or the native C2 model root.

    Dispatch depends only on the sanitized root manifest schema.  Native C2 is
    normalized into the same episode/frame objects as the custom contract.
    """

    root = model_root.resolve(strict=True)
    require(root.is_dir(), f"rgbd_model_root_not_directory:{root}")
    require(
        not ({part.lower() for part in root.parts} & FORBIDDEN_PATH_PARTS),
        f"rgbd_model_root_privileged:{root}",
    )
    manifest_path = resolve_model_path(root, "manifest.json", "rgbd_model_manifest_dispatch")
    manifest = _read_json_file(manifest_path, "rgbd_model_manifest_dispatch")
    schema = manifest.get("schema_version")
    if schema == ROOT_SCHEMA:
        return _load_custom_model_contract(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            validate_payload_hashes=validate_payload_hashes,
        )
    if isinstance(schema, str) and schema in {C2_ROOT_SCHEMA_V1, C2_ROOT_SCHEMA}:
        return _load_c2_model_contract(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            validate_payload_hashes=validate_payload_hashes,
        )
    raise RuntimeError(f"rgbd_model_manifest_schema_unsupported:{schema}")


def normalized_plan_receipt(observation: FrameObservation) -> dict[str, Any] | None:
    """Return an anchor-frame plan receipt for custom or native C2 input.

    Native C2 receipts are hash-validated in their source schema first, then
    translated in memory to the custom anchor-frame receipt shape.  The source
    receipt and source hashes remain in ``observation.issued_plan``.
    """

    issued_plan = observation.issued_plan
    if issued_plan["authority"] == "NO_PLAN":
        return None
    inline = issued_plan.get("receipt")
    if inline is None:
        path = Path(str(issued_plan["path"])).resolve(strict=True)
        inline = _read_json_file(path, f"rgbd_plan_receipt:{observation.episode_id}")
    require(isinstance(inline, Mapping), f"rgbd_plan_receipt_not_object:{observation.episode_id}")
    if inline.get("schema_version") == PLAN_SCHEMA:
        expected = _sha256(
            issued_plan["receipt_sha256"], f"rgbd_plan_receipt_expected:{observation.episode_id}"
        )
        _validate_plan_receipt(inline, expected, f"rgbd_plan_receipt:{observation.episode_id}")
        return dict(inline)
    require(
        inline.get("schema_version") == C2_PLAN_RECEIPT_SCHEMA,
        f"rgbd_plan_receipt_schema:{observation.episode_id}",
    )
    payload = {
        "schema_version": PLAN_SCHEMA,
        "coordinate_frame": "ANCHOR_FORWARD_RIGHT",
        "plan_id": str(inline["plan_id"]),
        "session_id": str(inline["session_id"]),
        "issued_at_s": float(inline["issued_at_s"]),
        "valid_from_s": float(inline["issued_at_s"]),
        "expires_at_s": float(inline["expires_at_s"]),
        "time_parameterized_waypoints": [dict(value) for value in inline["time_parameterized_waypoints"]],
    }
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    normalized = {**payload, "receipt_sha256": receipt_sha256}
    _validate_plan_receipt(normalized, receipt_sha256, f"rgbd_normalized_plan:{observation.episode_id}")
    return normalized


def camera_intrinsic(width: int, height: int, horizontal_fov_degrees: float) -> np.ndarray:
    width = _positive_int(width, "rgbd_intrinsic_width")
    height = _positive_int(height, "rgbd_intrinsic_height")
    fov = _finite(horizontal_fov_degrees, "rgbd_intrinsic_fov")
    require(0.0 < fov < 180.0, "rgbd_intrinsic_fov_range")
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.asarray(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_rgb(observation: FrameObservation) -> np.ndarray:
    from PIL import Image

    validate_file_hash(observation.rgb.path, observation.rgb.sha256, "rgbd_rgb")
    with Image.open(observation.rgb.path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(
        rgb.shape == (observation.rgb.height, observation.rgb.width, 3),
        f"rgbd_rgb_shape:{rgb.shape}:{observation.rgb.path}",
    )
    return rgb


def decode_carla_bgr_depth(
    encoded_bgr: np.ndarray,
    *,
    depth_max_m: float = 1000.0,
) -> np.ndarray:
    """Decode CARLA's 24-bit depth from a BGR or BGRA uint8 array."""

    encoded = np.asarray(encoded_bgr)
    require(encoded.ndim == 3 and encoded.shape[2] in {3, 4}, f"rgbd_depth_bgr_shape:{encoded.shape}")
    require(encoded.dtype == np.uint8, f"rgbd_depth_bgr_dtype:{encoded.dtype}")
    maximum = _finite(depth_max_m, "rgbd_depth_max")
    require(maximum > MINIMUM_DEPTH_M, "rgbd_depth_max_range")
    values = encoded[:, :, :3].astype(np.uint32)
    blue = values[:, :, 0]
    green = values[:, :, 1]
    red = values[:, :, 2]
    normalized = (
        red.astype(np.float64)
        + green.astype(np.float64) * 256.0
        + blue.astype(np.float64) * 65536.0
    ) / 16777215.0
    return (normalized * maximum).astype(np.float32)


def load_depth_m(
    observation: FrameObservation,
    calibration: CameraCalibration,
) -> np.ndarray:
    from PIL import Image

    require(
        calibration.depth_encoding in {DEPTH_ENCODING, C2_DEPTH_ENCODING},
        f"rgbd_depth_encoding_unsupported:{calibration.depth_encoding}",
    )
    validate_file_hash(observation.depth.path, observation.depth.sha256, "rgbd_depth")
    with Image.open(observation.depth.path) as image:
        encoded_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(
        encoded_rgb.shape == (observation.depth.height, observation.depth.width, 3),
        f"rgbd_depth_shape:{encoded_rgb.shape}:{observation.depth.path}",
    )
    # PIL exposes RGB; reverse once so the public decoder has one unambiguous
    # BGR contract matching CARLA's raw byte ordering.
    return decode_carla_bgr_depth(
        encoded_rgb[:, :, ::-1],
        depth_max_m=calibration.depth_max_m,
    )


def angular_grid_pixels(
    width: int,
    height: int,
    *,
    grid_width: int = ANGULAR_GRID_WIDTH,
    grid_height: int = ANGULAR_GRID_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one center pixel for each fixed normalized angular cell."""

    width = _positive_int(width, "rgbd_grid_width")
    height = _positive_int(height, "rgbd_grid_height")
    grid_width = _positive_int(grid_width, "rgbd_grid_angular_width")
    grid_height = _positive_int(grid_height, "rgbd_grid_angular_height")
    require(width >= grid_width and height >= grid_height, "rgbd_grid_resolution_too_small")
    columns = np.floor((np.arange(grid_width, dtype=np.float64) + 0.5) * width / grid_width).astype(np.int32)
    rows = np.floor((np.arange(grid_height, dtype=np.float64) + 0.5) * height / grid_height).astype(np.int32)
    columns = np.clip(columns, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)
    vv, uu = np.meshgrid(rows, columns, indexing="ij")
    return uu, vv


def unproject_pixels_camera_flu(
    pixels_uv: np.ndarray,
    forward_depth_m: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Unproject pixels into camera forward-left-up coordinates."""

    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(forward_depth_m, dtype=np.float64).reshape(-1)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    require(len(pixels) == len(depth), "rgbd_unproject_count")
    require(matrix.shape == (3, 3) and np.all(np.isfinite(matrix)), "rgbd_unproject_intrinsic")
    require(np.all(np.isfinite(pixels)) and np.all(np.isfinite(depth)), "rgbd_unproject_nonfinite")
    require(np.all(depth > 0.0), "rgbd_unproject_nonpositive_depth")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    require(fx > 0.0 and fy > 0.0, "rgbd_unproject_focal")
    forward = depth
    left = (cx - pixels[:, 0]) * forward / fx
    up = (cy - pixels[:, 1]) * forward / fy
    return np.column_stack((forward, left, up)).astype(np.float64)


def _rotation_ue(transform: Mapping[str, Any]) -> np.ndarray:
    pitch = math.radians(_finite(transform["pitch"], "rgbd_pose_pitch"))
    yaw = math.radians(_finite(transform["yaw"], "rgbd_pose_yaw"))
    roll = math.radians(_finite(transform["roll"], "rgbd_pose_roll"))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def camera_flu_to_world(
    points_camera_flu: np.ndarray,
    camera_transform: Mapping[str, Any],
) -> np.ndarray:
    """Transform camera FLU points into CARLA world coordinates."""

    _exact_fields(camera_transform, TRANSFORM_FIELDS, "rgbd_camera_transform")
    points = np.asarray(points_camera_flu, dtype=np.float64).reshape(-1, 3)
    require(np.all(np.isfinite(points)), "rgbd_camera_points_nonfinite")
    rotation = _rotation_ue(camera_transform) @ np.diag([1.0, -1.0, 1.0])
    translation = np.asarray(
        [
            _finite(camera_transform["x"], "rgbd_camera_x"),
            _finite(camera_transform["y"], "rgbd_camera_y"),
            _finite(camera_transform["z"], "rgbd_camera_z"),
        ],
        dtype=np.float64,
    )
    return points @ rotation.T + translation[None, :]


def world_to_anchor_fru(
    points_world: np.ndarray,
    anchor: AnchorFrame,
) -> np.ndarray:
    """Transform CARLA world points into anchor forward-right-up coordinates."""

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    require(np.all(np.isfinite(points)), "rgbd_world_points_nonfinite")
    delta_xy = points[:, :2] - np.asarray(anchor.center_xy_m, dtype=np.float64)[None, :]
    forward = delta_xy @ np.asarray(anchor.forward_xy, dtype=np.float64)
    right = delta_xy @ np.asarray(anchor.right_xy, dtype=np.float64)
    up = points[:, 2] - float(anchor.z_origin_m)
    return np.column_stack((forward, right, up))


def mask_near_depth_measurement(
    mask: np.ndarray,
    depth_m: np.ndarray,
    calibration: CameraCalibration,
    camera_transform: Mapping[str, Any],
    route_frame: AnchorFrame,
    *,
    minimum_points: int = MINIMUM_MASK_DEPTH_POINTS,
) -> MaskDepthMeasurement:
    """Measure a mask with a resolution-invariant near-depth slab."""

    mask_value = np.asarray(mask)
    depth_value = np.asarray(depth_m, dtype=np.float64)
    expected_shape = (calibration.height, calibration.width)
    require(mask_value.shape == expected_shape, f"rgbd_mask_shape:{mask_value.shape}:{expected_shape}")
    require(depth_value.shape == expected_shape, f"rgbd_depth_shape:{depth_value.shape}:{expected_shape}")
    minimum = _positive_int(minimum_points, "rgbd_mask_minimum_points")
    uu, vv = angular_grid_pixels(calibration.width, calibration.height)
    selected_mask = mask_value.astype(bool)[vv, uu]
    selected_depth = depth_value[vv, uu]
    valid = (
        selected_mask
        & np.isfinite(selected_depth)
        & (selected_depth > MINIMUM_DEPTH_M)
        & (selected_depth < calibration.depth_max_m)
    )
    support = int(np.count_nonzero(valid))
    if support < minimum:
        return MaskDepthMeasurement(False, support, 0, None, None, None, None, None)

    candidate_depth = selected_depth[valid]
    depth_q15 = float(np.quantile(candidate_depth, NEAR_DEPTH_QUANTILE))
    slab_m = max(
        MINIMUM_NEAR_SLAB_M,
        min(MAXIMUM_NEAR_SLAB_M, NEAR_SLAB_DEPTH_RATIO * depth_q15),
    )
    foreground_grid = valid & (selected_depth <= depth_q15 + slab_m)
    foreground_support = int(np.count_nonzero(foreground_grid))
    if foreground_support < minimum:
        return MaskDepthMeasurement(
            False,
            support,
            foreground_support,
            depth_q15,
            slab_m,
            None,
            None,
            None,
        )

    pixels = np.column_stack((uu[foreground_grid], vv[foreground_grid])).astype(np.float64)
    forward_depth = selected_depth[foreground_grid]
    camera_points = unproject_pixels_camera_flu(
        pixels,
        forward_depth,
        calibration.intrinsic,
    )
    position_camera = np.median(camera_points, axis=0)
    position_world = camera_flu_to_world(position_camera.reshape(1, 3), camera_transform)[0]
    position_anchor = world_to_anchor_fru(position_world.reshape(1, 3), route_frame)[0]
    return MaskDepthMeasurement(
        True,
        support,
        foreground_support,
        depth_q15,
        slab_m,
        tuple(float(value) for value in position_camera),
        tuple(float(value) for value in position_world),
        tuple(float(value) for value in position_anchor),
    )


__all__ = [
    "ANGULAR_GRID_HEIGHT",
    "ANGULAR_GRID_WIDTH",
    "CALIBRATION_SCHEMA",
    "C2_ALIGNMENT_AUTHORITY",
    "C2_ALIGNMENT_RECEIPT_SCHEMA",
    "C2_CALIBRATION_SCHEMA",
    "C2_DEPTH_ENCODING",
    "C2_EPISODE_SCHEMA",
    "C2_OBSERVATION_SCHEMA",
    "C2_PLAN_RECEIPT_SCHEMA",
    "C2_PLAN_WRAPPER_SCHEMA",
    "C2_ROOT_SCHEMA",
    "C2_ROOT_SCHEMA_V1",
    "DEPTH_ENCODING",
    "EPISODE_SCHEMA",
    "FORBIDDEN_MODEL_KEYS",
    "FORBIDDEN_PATH_PARTS",
    "MINIMUM_MASK_DEPTH_POINTS",
    "OBSERVATION_SCHEMA",
    "ROOT_SCHEMA",
    "AnchorFrame",
    "CameraCalibration",
    "Episode",
    "FileReference",
    "FrameObservation",
    "ImageReference",
    "MaskDepthMeasurement",
    "SanitizedModelContract",
    "angular_grid_pixels",
    "assert_sanitized_model_value",
    "camera_flu_to_world",
    "camera_intrinsic",
    "decode_carla_bgr_depth",
    "load_depth_m",
    "load_model_contract",
    "load_rgb",
    "mask_near_depth_measurement",
    "normalized_plan_receipt",
    "resolve_model_path",
    "sha256_file",
    "unproject_pixels_camera_flu",
    "validate_file_hash",
    "world_to_anchor_fru",
]
