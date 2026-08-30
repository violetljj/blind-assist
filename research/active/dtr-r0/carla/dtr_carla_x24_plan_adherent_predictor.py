"""Freeze and predict CARLA X23/X24 from sanitized dense RGB-D.

The predictor is deliberately evaluator-blind.  It consumes only the strict
model contract from :mod:`dtr_carla_rgbd_model_adapter` and normalized YOLO
candidate artifacts.  Both arms share every visual measurement and metric
track; their sole difference is the wearer route hypothesis:

* X23 extrapolates the currently observed wearer velocity.
* X24 admits an issued time-parameterized plan only while authority and
  current execution agree, otherwise it causally falls back to X23.

Scoring lives behind a separate post-prediction command that is added only
after a concrete evaluator contract has been sealed.  This file therefore
cannot open evaluator material by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_rgbd_model_adapter as adapter  # noqa: E402
import dtr_carla_x24_plan_route_core as route  # noqa: E402
import dtr_carla_yolo_metric_candidates as detector  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X24_PLAN_ADHERENT_METRIC_TRACK"
INDEX_SCHEMA = "blindassist-dtr-carla-x24-rgb-index-v1"
FREEZE_SCHEMA = "blindassist-dtr-carla-x24-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
ARM_X23 = "X23_OBSERVED_CV_ROUTE"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"

VELOCITY_WINDOW_S = 0.50
TRACK_HISTORY_S = 1.00
HOLD_WINDOW_S = 0.60
MINIMUM_FIT_SAMPLES = 4
MINIMUM_FIT_SPAN_S = 0.15
MINIMUM_SLOPE_SPAN_S = 0.10
ASSOCIATION_DISTANCE_M = 1.50
CONFIRMATION_SECONDS = 0.10
MAXIMUM_CONSECUTIVE_GAP_FACTOR = 1.50
EPSILON = 1e-9

ALGORITHM_FILES = {
    "adapter": Path(adapter.__file__).resolve(),
    "route_core": Path(route.__file__).resolve(),
    "candidate_materializer": Path(detector.__file__).resolve(),
    "predictor": Path(__file__).resolve(),
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label}_not_object:{path}")
    adapter.assert_sanitized_model_value(value, label)
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def contained_file(root: Path, raw_path: str | Path, label: str) -> Path:
    base = root.resolve(strict=True)
    candidate = Path(raw_path)
    require(not candidate.is_absolute(), f"{label}_absolute:{raw_path}")
    resolved = (base / candidate).resolve(strict=True)
    require(resolved.is_relative_to(base) and resolved.is_file(), f"{label}_escape:{raw_path}")
    return resolved


def run_paths(run_root: Path) -> dict[str, Path]:
    return {
        "index": run_root / "x24-rgb-index.jsonl",
        "index_receipt": run_root / "x24-rgb-index-receipt.json",
        "candidates": run_root / "candidates",
        "freeze": run_root / "freeze-x24.json",
        "predictions": run_root / "predictions-x24.json",
    }


def flatten_observations(
    contract: adapter.SanitizedModelContract,
) -> list[adapter.FrameObservation]:
    return [observation for episode in contract.episodes for observation in episode.observations]


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    require(run_root.is_dir(), f"x24_run_root_missing:{run_root}")
    paths = run_paths(run_root)
    contract = adapter.load_model_contract(
        args.model_root,
        expected_manifest_sha256=args.model_manifest_sha256,
        validate_payload_hashes=True,
    )
    observations = flatten_observations(contract)
    rows = [
        {
            "image_path": str(observation.rgb.path),
            "image_sha256": observation.rgb.sha256,
            "frame_id": f"{observation.episode_id}/{observation.sample_index:06d}",
            "episode_id": observation.episode_id,
            "sample_index": observation.sample_index,
            "time_s": observation.time_s,
            "world_frame": observation.world_frame,
        }
        for observation in observations
    ]
    write_jsonl_exclusive(paths["index"], rows)
    receipt = {
        "schema": INDEX_SCHEMA,
        "status": "SEALED_SANITIZED_RGB_INDEX",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "model_root": str(contract.model_root),
        "model_manifest": {
            "path": str(contract.manifest_path),
            "sha256": contract.manifest_sha256,
        },
        "episodes": len(contract.episodes),
        "frames": len(observations),
        "resolution": [contract.calibration.width, contract.calibration.height],
        "index": {"path": str(paths["index"]), "sha256": sha256_file(paths["index"])},
    }
    write_json_exclusive(paths["index_receipt"], receipt)
    return receipt


def candidate_set(
    candidate_root: Path,
    observations: Sequence[adapter.FrameObservation],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = candidate_root.resolve(strict=True)
    require(root.is_dir(), f"x24_candidate_root_not_directory:{root}")
    manifest_path = contained_file(root, "manifest.json", "x24_candidate_manifest")
    manifest = read_json(manifest_path, "x24_candidate_manifest")
    require(manifest.get("schema") == detector.MANIFEST_SCHEMA, "x24_candidate_manifest_schema")
    require(manifest.get("status") == "COMPLETE", "x24_candidate_manifest_status")
    require(manifest.get("run_kind") == "CANDIDATE_MATERIALIZATION", "x24_candidate_run_kind")
    require(int(manifest.get("frame_count", -1)) == len(observations), "x24_candidate_frame_count")
    references = manifest.get("frames")
    require(isinstance(references, list) and len(references) == len(observations), "x24_candidate_references")
    values: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for ordinal, (observation, reference) in enumerate(zip(observations, references)):
        require(isinstance(reference, Mapping), f"x24_candidate_reference:{ordinal}")
        require(int(reference.get("ordinal", -1)) == ordinal, f"x24_candidate_ordinal:{ordinal}")
        expected_frame_id = f"{observation.episode_id}/{observation.sample_index:06d}"
        require(reference.get("frame_id") == expected_frame_id, f"x24_candidate_frame_id:{ordinal}")
        require(
            int(reference.get("world_frame", -1)) == observation.world_frame,
            f"x24_candidate_world_frame:{ordinal}",
        )
        path = contained_file(root, str(reference.get("path", "")), f"x24_candidate_file:{ordinal}")
        digest = sha256_file(path)
        require(digest == str(reference.get("sha256", "")).upper(), f"x24_candidate_hash:{ordinal}")
        value = read_json(path, f"x24_candidate:{ordinal}")
        require(value.get("schema") == detector.SCHEMA, f"x24_candidate_schema:{ordinal}")
        require(value.get("run_kind") == "CANDIDATE_MATERIALIZATION", f"x24_candidate_kind:{ordinal}")
        source = value.get("source")
        require(isinstance(source, Mapping), f"x24_candidate_source:{ordinal}")
        require(
            source.get("frame_id") == expected_frame_id
            and source.get("episode_id") == observation.episode_id
            and int(source.get("sample_index", -1)) == observation.sample_index
            and abs(float(source.get("time_s", math.inf)) - observation.time_s) <= EPSILON,
            f"x24_candidate_identity:{ordinal}",
        )
        require(
            int(source.get("world_frame", -1)) == observation.world_frame,
            f"x24_candidate_source_world_frame:{ordinal}",
        )
        require(str(source.get("image_sha256", "")).upper() == observation.rgb.sha256, f"x24_candidate_image:{ordinal}")
        require(
            (int(source.get("image_width", -1)), int(source.get("image_height", -1)))
            == (observation.rgb.width, observation.rgb.height),
            f"x24_candidate_resolution:{ordinal}",
        )
        candidates = value.get("candidates")
        require(isinstance(candidates, list), f"x24_candidate_list:{ordinal}")
        for candidate_number, candidate in enumerate(candidates):
            validate_candidate(candidate, ordinal, candidate_number)
        aggregate.update(f"{path.name}:{digest}\n".encode("utf-8"))
        values.append(value)
    require(bool(values), "x24_candidate_values_empty")
    manifest_model_sha256 = str(manifest["model"]["sha256"]).upper()
    require(
        all(str(value["model"]["sha256"]).upper() == manifest_model_sha256 for value in values),
        "x24_candidate_model_identity",
    )
    return manifest, values, aggregate.hexdigest().upper()


def validate_candidate(candidate: Any, ordinal: int, candidate_number: int) -> None:
    label = f"x24_candidate_geometry:{ordinal}:{candidate_number}"
    require(isinstance(candidate, Mapping), f"{label}:not_object")
    require(str(candidate.get("class_name")) in detector.HAZARD_CLASS_NAMES, f"{label}:class")
    confidence = float(candidate.get("confidence", math.nan))
    require(math.isfinite(confidence) and 0.0 <= confidence <= 1.0, f"{label}:confidence")
    bbox = np.asarray(candidate.get("bbox_xyxy_normalized"), dtype=np.float64)
    polygon = np.asarray(candidate.get("polygon_xy_normalized"), dtype=np.float64)
    require(bbox.shape == (4,) and np.isfinite(bbox).all(), f"{label}:bbox")
    require(polygon.ndim == 2 and polygon.shape[1:] == (2,) and len(polygon) >= 3, f"{label}:polygon")
    require(np.isfinite(polygon).all(), f"{label}:polygon_finite")
    require(np.all((bbox >= 0.0) & (bbox <= 1.0)), f"{label}:bbox_range")
    require(np.all((polygon >= 0.0) & (polygon <= 1.0)), f"{label}:polygon_range")
    require(bbox[2] > bbox[0] and bbox[3] > bbox[1], f"{label}:bbox_order")


def fixed_constants() -> dict[str, Any]:
    return {
        "angular_depth_grid": [adapter.ANGULAR_GRID_WIDTH, adapter.ANGULAR_GRID_HEIGHT],
        "minimum_mask_depth_points": adapter.MINIMUM_MASK_DEPTH_POINTS,
        "velocity_window_seconds": VELOCITY_WINDOW_S,
        "hold_window_seconds": HOLD_WINDOW_S,
        "minimum_fit_samples": MINIMUM_FIT_SAMPLES,
        "minimum_fit_span_seconds": MINIMUM_FIT_SPAN_S,
        "association_distance_m": ASSOCIATION_DISTANCE_M,
        "confirmation_seconds": CONFIRMATION_SECONDS,
        "route_horizon_seconds": route.DEFAULT_ROUTE_HORIZON_S,
        "tube_radius_m": route.DEFAULT_TUBE_RADIUS_M,
        "minimum_closing_speed_mps": route.DEFAULT_MIN_CLOSING_SPEED_MPS,
        "maximum_plan_position_residual_m": route.DEFAULT_PLAN_POSITION_RESIDUAL_M,
        "maximum_plan_velocity_direction_error_degrees": route.DEFAULT_PLAN_VELOCITY_DIRECTION_ERROR_DEG,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    paths = run_paths(run_root)
    require(not paths["freeze"].exists(), f"x24_freeze_exists:{paths['freeze']}")
    contract = adapter.load_model_contract(
        args.model_root,
        expected_manifest_sha256=args.model_manifest_sha256,
        validate_payload_hashes=True,
    )
    observations = flatten_observations(contract)
    index_receipt = read_json(paths["index_receipt"].resolve(strict=True), "x24_index_receipt")
    require(index_receipt.get("schema") == INDEX_SCHEMA, "x24_index_receipt_schema")
    require(index_receipt["model_manifest"]["sha256"] == contract.manifest_sha256, "x24_index_model_drift")
    require(index_receipt["index"]["sha256"] == sha256_file(paths["index"]), "x24_index_hash_drift")
    candidate_manifest, _candidate_values, aggregate = candidate_set(paths["candidates"], observations)
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "model_root": str(contract.model_root),
        "model_manifest": {
            "path": str(contract.manifest_path),
            "sha256": contract.manifest_sha256,
        },
        "rgb_index": {
            "path": str(paths["index"]),
            "sha256": sha256_file(paths["index"]),
        },
        "candidates": {
            "path": str(paths["candidates"]),
            "manifest_sha256": sha256_file(paths["candidates"] / "manifest.json"),
            "aggregate_sha256": aggregate,
            "detector_model": candidate_manifest["model"],
        },
        "algorithm_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in ALGORITHM_FILES.items()
        },
        "episodes": len(contract.episodes),
        "frames": len(observations),
        "resolution": [contract.calibration.width, contract.calibration.height],
        "arms": [ARM_X23, ARM_X24],
        "fixed_constants": fixed_constants(),
        "forbidden_online_keys": sorted(adapter.FORBIDDEN_MODEL_KEYS),
    }
    write_json_exclusive(paths["freeze"], value)
    return {**value, "freeze_sha256": sha256_file(paths["freeze"])}


def require_freeze(run_root: Path) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    paths = run_paths(run_root)
    frozen = read_json(paths["freeze"].resolve(strict=True), "x24_freeze")
    require(
        frozen.get("schema") == FREEZE_SCHEMA
        and frozen.get("status") == "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "x24_freeze_schema",
    )
    require(frozen.get("fixed_constants") == fixed_constants(), "x24_frozen_constants_drift")
    for name, reference in frozen["algorithm_files"].items():
        require(name in ALGORITHM_FILES, f"x24_frozen_algorithm_unknown:{name}")
        path = Path(reference["path"]).resolve(strict=True)
        require(path == ALGORITHM_FILES[name] and sha256_file(path) == reference["sha256"], f"x24_algorithm_drift:{name}")
    contract = adapter.load_model_contract(
        Path(frozen["model_root"]),
        expected_manifest_sha256=frozen["model_manifest"]["sha256"],
        validate_payload_hashes=True,
    )
    observations = flatten_observations(contract)
    candidate_manifest, candidates, aggregate = candidate_set(Path(frozen["candidates"]["path"]), observations)
    require(sha256_file(Path(frozen["candidates"]["path"]) / "manifest.json") == frozen["candidates"]["manifest_sha256"], "x24_candidate_manifest_drift")
    require(aggregate == frozen["candidates"]["aggregate_sha256"], "x24_candidate_aggregate_drift")
    require(candidate_manifest["model"] == frozen["candidates"]["detector_model"], "x24_detector_model_drift")
    require(len(observations) == int(frozen["frames"]), "x24_frozen_frame_count")
    return frozen, contract, candidates


def normalized_polygon_mask(candidate: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    points = [
        (
            max(0, min(width - 1, int(round(float(value[0]) * (width - 1))))),
            max(0, min(height - 1, int(round(float(value[1]) * (height - 1))))),
        )
        for value in candidate["polygon_xy_normalized"]
    ]
    if len(points) >= 3:
        ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class Measurement:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    position_xy: np.ndarray
    depth_support: int


@dataclass
class Track:
    track_id: str
    class_id: int
    class_name: str
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    last_seen_s: float = -math.inf
    last_position_xy: np.ndarray | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    position_xy: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    state_time_s: float = -math.inf
    depth_support: int | None = None


def robust_motion(history: Sequence[tuple[float, np.ndarray]], now_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    window = [row for row in history if now_s - row[0] <= VELOCITY_WINDOW_S + EPSILON]
    if len(window) < MINIMUM_FIT_SAMPLES or window[-1][0] - window[0][0] < MINIMUM_FIT_SPAN_S - EPSILON:
        return None
    times = np.asarray([row[0] for row in window], dtype=np.float64)
    positions = np.stack([row[1] for row in window]).astype(np.float64)
    slopes: list[np.ndarray] = []
    for left in range(len(window)):
        for right_index in range(left + 1, len(window)):
            delta_s = times[right_index] - times[left]
            if delta_s >= MINIMUM_SLOPE_SPAN_S - EPSILON:
                slopes.append((positions[right_index] - positions[left]) / delta_s)
    if not slopes:
        return None
    velocity = np.median(np.stack(slopes), axis=0)
    position = np.median(positions - (times - now_s)[:, None] * velocity[None, :], axis=0)
    return position, velocity


class MetricTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, Track] = {}
        self.next_id = 1

    @staticmethod
    def predicted_position(track: Track, now_s: float) -> np.ndarray | None:
        if track.position_xy is not None and track.velocity_xy is not None:
            return track.position_xy + track.velocity_xy * max(0.0, now_s - track.state_time_s)
        return None if track.last_position_xy is None else track.last_position_xy

    def update(self, measurements: Sequence[Measurement], now_s: float) -> set[str]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= HOLD_WINDOW_S + EPSILON
        }
        costs: list[tuple[float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            predicted = self.predicted_position(track, now_s)
            if predicted is None:
                continue
            for index, measurement in enumerate(measurements):
                if measurement.class_id != track.class_id:
                    continue
                distance = float(np.linalg.norm(predicted - measurement.position_xy))
                if distance <= ASSOCIATION_DISTANCE_M + EPSILON:
                    overlap = 0.0 if track.last_bbox is None else bbox_iou(track.last_bbox, measurement.bbox)
                    costs.append((distance, -overlap, track_id, index))
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _distance, _overlap, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: set[str] = set()
        for index, measurement in enumerate(measurements):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"metric-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = Track(track_id, measurement.class_id, measurement.class_name)
            track = self.tracks[track_id]
            track.history.append((now_s, measurement.position_xy.copy()))
            track.history = [row for row in track.history if now_s - row[0] <= TRACK_HISTORY_S + EPSILON]
            track.last_seen_s = now_s
            track.last_position_xy = measurement.position_xy.copy()
            track.last_bbox = measurement.bbox
            track.depth_support = measurement.depth_support
            motion = robust_motion(track.history, now_s)
            if motion is not None:
                track.position_xy, track.velocity_xy = motion
                track.state_time_s = now_s
            measured_ids.add(track_id)
        return measured_ids

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            if track.position_xy is None or track.velocity_xy is None:
                continue
            age_s = now_s - track.last_seen_s
            if age_s > HOLD_WINDOW_S + EPSILON:
                continue
            position = track.position_xy + track.velocity_xy * max(0.0, now_s - track.state_time_s)
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "position_forward_m": float(position[0]),
                    "position_right_m": float(position[1]),
                    "velocity_forward_mps": float(track.velocity_xy[0]),
                    "velocity_right_mps": float(track.velocity_xy[1]),
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                }
            )
        return output


def candidate_measurements(
    observation: adapter.FrameObservation,
    candidate_value: Mapping[str, Any],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> list[Measurement]:
    if not candidate_value["candidates"]:
        return []
    depth_m = adapter.load_depth_m(observation, calibration)
    output: list[Measurement] = []
    for candidate in candidate_value["candidates"]:
        mask = normalized_polygon_mask(candidate, calibration.width, calibration.height)
        measured = adapter.mask_near_depth_measurement(
            mask,
            depth_m,
            calibration,
            observation.camera_transform,
            route_frame,
        )
        if not measured.valid or measured.position_anchor_fru_m is None:
            continue
        output.append(
            Measurement(
                class_id=int(candidate["class_id"]),
                class_name=str(candidate["class_name"]),
                confidence=float(candidate["confidence"]),
                bbox=tuple(float(value) for value in candidate["bbox_xyxy_normalized"]),
                position_xy=np.asarray(measured.position_anchor_fru_m[:2], dtype=np.float64),
                depth_support=measured.foreground_support,
            )
        )
    return output


def wearer_anchor_state(
    observation: adapter.FrameObservation,
    route_frame: adapter.AnchorFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    transform = observation.wearer["transform"]
    world = np.asarray([[transform["x"], transform["y"], transform["z"]]], dtype=np.float64)
    anchor_position = adapter.world_to_anchor_fru(world, route_frame)[0]
    velocity = observation.wearer["command_velocity"]
    forward = route_frame.forward_xy
    right = route_frame.right_xy
    anchor_velocity = (
        float(velocity["x"] * forward[0] + velocity["y"] * forward[1]),
        float(velocity["x"] * right[0] + velocity["y"] * right[1]),
    )
    return (float(anchor_position[0]), float(anchor_position[1])), anchor_velocity


def load_receipt(observation: adapter.FrameObservation, cache: dict[Path, dict[str, Any]]) -> dict[str, Any] | None:
    raw_path = observation.issued_plan["path"]
    if raw_path is None:
        return None
    path = Path(raw_path).resolve(strict=True)
    value = cache.get(path)
    if value is None:
        value = read_json(path, f"x24_plan_receipt:{observation.episode_id}")
        if value.get("schema_version") == "dtr-c2-model-plan-v1":
            require(
                set(value)
                == {
                    "schema_version",
                    "episode_id",
                    "navigation_session_id",
                    "layout_anchor",
                    "issued_plan",
                },
                "x24_c2_plan_wrapper_fields",
            )
            require(value["episode_id"] == observation.episode_id, "x24_c2_plan_episode")
            require(
                value["navigation_session_id"] == observation.navigation_session_id,
                "x24_c2_plan_session",
            )
            issued = value["issued_plan"]
            require(
                isinstance(issued, Mapping)
                and set(issued)
                == {
                    "authority",
                    "receipt",
                    "receipt_sha256",
                    "time_parameterized_waypoints_world",
                    "world_coordinate_frame",
                },
                "x24_c2_issued_plan_fields",
            )
            require(issued["authority"] == observation.issued_plan["authority"], "x24_c2_plan_authority")
            value = issued["receipt"]
            require(isinstance(value, Mapping), "x24_c2_plan_receipt_missing")
            value = dict(value)
            require(issued["receipt_sha256"] == value.get("receipt_sha256"), "x24_c2_wrapper_receipt_identity")
        route.validate_plan_receipt(value)
        require(value["receipt_sha256"] == observation.issued_plan["receipt_sha256"], "x24_plan_receipt_identity")
        cache[path] = value
    return value


def observed_selection(previous_mode: str | None) -> route.RouteSelection:
    return route.RouteSelection(
        mode=route.ROUTE_MODE_OBSERVED_CV,
        authority="PLAN_NOT_CONSULTED",
        receipt_valid=False,
        receipt_sha256=None,
        plan_position_residual_m=None,
        plan_velocity_direction_error_deg=None,
        fallback_reason="X23_BASELINE",
        mode_changed=route.route_mode_changed(previous_mode, route.ROUTE_MODE_OBSERVED_CV),
    )


@dataclass
class RiskConfirmation:
    since_s: dict[str, float] = field(default_factory=dict)
    last_s: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.since_s.clear()
        self.last_s.clear()

    def update(
        self,
        entries: Mapping[str, float | None],
        *,
        now_s: float,
        sample_period_s: float,
    ) -> set[str]:
        confirmed: set[str] = set()
        live = set(entries)
        for track_id in set(self.since_s) - live:
            self.since_s.pop(track_id, None)
            self.last_s.pop(track_id, None)
        for track_id, entry_s in entries.items():
            if entry_s is None:
                self.since_s.pop(track_id, None)
                self.last_s.pop(track_id, None)
                continue
            previous = self.last_s.get(track_id)
            if previous is None or now_s - previous > sample_period_s * MAXIMUM_CONSECUTIVE_GAP_FACTOR + EPSILON:
                self.since_s[track_id] = now_s
            self.last_s[track_id] = now_s
            if now_s - self.since_s[track_id] >= CONFIRMATION_SECONDS - EPSILON:
                confirmed.add(track_id)
        return confirmed


def arm_frame(
    selection: route.RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    observation: adapter.FrameObservation,
    wearer_position: tuple[float, float],
    wearer_velocity: tuple[float, float],
    tracks: Sequence[Mapping[str, Any]],
    confirmation: RiskConfirmation,
    sample_period_s: float,
) -> dict[str, Any]:
    if selection.mode_changed:
        confirmation.reset()
    entries: dict[str, float | None] = {}
    for track in tracks:
        entries[str(track["track_id"])] = route.first_selected_route_entry_s(
            selection,
            receipt=receipt,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            target_position_xy=(track["position_forward_m"], track["position_right_m"]),
            target_velocity_xy=(track["velocity_forward_mps"], track["velocity_right_mps"]),
        )
    confirmed = confirmation.update(
        entries,
        now_s=observation.time_s,
        sample_period_s=sample_period_s,
    )
    confirmed_entries = [float(entries[track_id]) for track_id in confirmed if entries[track_id] is not None]
    return {
        "route_mode": selection.mode,
        "authority": selection.authority,
        "plan_receipt_sha256": selection.receipt_sha256,
        "plan_position_residual_m": selection.plan_position_residual_m,
        "plan_velocity_direction_error_degrees": selection.plan_velocity_direction_error_deg,
        "fallback_reason": selection.fallback_reason,
        "route_mode_changed": selection.mode_changed,
        "route_risk": bool(confirmed_entries),
        "minimum_entry_s": min(confirmed_entries) if confirmed_entries else None,
        "candidate_risk_track_ids": sorted(track_id for track_id, value in entries.items() if value is not None),
        "confirmed_risk_track_ids": sorted(confirmed),
    }


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    require(len(candidate_values) == len(episode.observations), f"x24_episode_candidate_count:{episode.episode_id}")
    tracker = MetricTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmations = {ARM_X23: RiskConfirmation(), ARM_X24: RiskConfirmation()}
    previous_modes: dict[str, str | None] = {ARM_X23: None, ARM_X24: None}
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    raw_candidate_frames = metric_measurement_frames = emitted_track_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(zip(episode.observations, candidate_values)):
        previous_time = episode.observations[ordinal - 1].time_s if ordinal else None
        next_time = episode.observations[ordinal + 1].time_s if ordinal + 1 < len(episode.observations) else None
        if previous_time is not None:
            sample_period_s = observation.time_s - previous_time
        elif next_time is not None:
            sample_period_s = next_time - observation.time_s
        else:
            raise RuntimeError(f"x24_single_frame_episode:{episode.episode_id}")
        require(sample_period_s > 0.0, f"x24_sample_period:{episode.episode_id}:{ordinal}")
        measurements = candidate_measurements(
            observation,
            candidate_value,
            calibration,
            episode.route_frame,
        )
        raw_candidate_frames += int(bool(candidate_value["candidates"]))
        metric_measurement_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        emitted_track_frames += int(bool(tracks))
        hold_frames += int(any(track["disposition"] == "HOLD" for track in tracks))
        wearer_position, wearer_velocity = wearer_anchor_state(
            observation,
            episode.route_frame,
        )
        receipt = load_receipt(observation, receipt_cache)

        x23_selection = observed_selection(previous_modes[ARM_X23])
        x24_selection = route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_modes[ARM_X24],
        )
        if observation.issued_plan["authority"] == "NO_PLAN":
            require(x24_selection.authority == route.AUTHORITY_NO_PLAN, f"x24_authority_no_plan:{episode.episode_id}:{ordinal}")
        elif observation.issued_plan["authority"] == "EXPIRED":
            require(x24_selection.authority == route.AUTHORITY_EXPIRED, f"x24_authority_expired:{episode.episode_id}:{ordinal}")
        elif observation.issued_plan["authority"] == "VALID":
            require(x24_selection.authority == route.AUTHORITY_VALID, f"x24_authority_valid:{episode.episode_id}:{ordinal}")
        previous_modes[ARM_X23] = x23_selection.mode
        previous_modes[ARM_X24] = x24_selection.mode
        route_mode_counts[x24_selection.mode] += 1
        arms = {
            ARM_X23: arm_frame(
                x23_selection,
                receipt=None,
                observation=observation,
                wearer_position=wearer_position,
                wearer_velocity=wearer_velocity,
                tracks=tracks,
                confirmation=confirmations[ARM_X23],
                sample_period_s=sample_period_s,
            ),
            ARM_X24: arm_frame(
                x24_selection,
                receipt=receipt,
                observation=observation,
                wearer_position=wearer_position,
                wearer_velocity=wearer_velocity,
                tracks=tracks,
                confirmation=confirmations[ARM_X24],
                sample_period_s=sample_period_s,
            ),
        }
        frames.append(
            {
                "sample_index": observation.sample_index,
                "time_s": observation.time_s,
                "world_frame": observation.world_frame,
                "raw_candidates": len(candidate_value["candidates"]),
                "metric_measurements": len(measurements),
                "tracks": tracks,
                "arms": arms,
            }
        )
    arm_summaries = {
        arm: {
            "route_risk_frames": sum(bool(frame["arms"][arm]["route_risk"]) for frame in frames),
            "first_route_risk_time_s": next(
                (frame["time_s"] for frame in frames if frame["arms"][arm]["route_risk"]),
                None,
            ),
        }
        for arm in (ARM_X23, ARM_X24)
    }
    return {
        "episode_id": episode.episode_id,
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "raw_candidate_frames": raw_candidate_frames,
            "metric_measurement_frames": metric_measurement_frames,
            "emitted_track_frames": emitted_track_frames,
            "track_coverage": emitted_track_frames / max(1, len(frames)),
            "hold_frames": hold_frames,
            "x24_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": arm_summaries,
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    paths = run_paths(run_root)
    if not args.dry_run:
        require(not paths["predictions"].exists(), f"x24_predictions_exist:{paths['predictions']}")
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        limit = len(episode.observations) if not args.dry_run else min(args.limit_per_episode, len(episode.observations))
        require(limit >= MINIMUM_FIT_SAMPLES, f"x24_predict_limit:{limit}")
        selected_episode = adapter.Episode(
            episode.episode_id,
            episode.route_frame,
            episode.observations[:limit],
        )
        selected_candidates = candidate_values[cursor : cursor + limit]
        episodes[episode.episode_id] = predict_episode(selected_episode, selected_candidates, contract.calibration)
        cursor += len(episode.observations)
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "DRY_CHECK_COMPLETE_NO_WRITE" if args.dry_run else "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X23, ARM_X24],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {
            "freeze_sha256": sha256_file(paths["freeze"]),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen["candidates"]["aggregate_sha256"],
        },
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened": False,
            "current_actor_oracle_used": False,
            "old_320x180_c1_not_scored": True,
        },
    }
    if not args.dry_run:
        write_json_exclusive(paths["predictions"], value)
        value = {**value, "predictions_sha256": sha256_file(paths["predictions"])}
    return value


def status(args: argparse.Namespace) -> dict[str, Any]:
    paths = run_paths(args.run_root.resolve())
    existing = {
        name: None if not path.is_file() else {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "candidates"
    }
    if existing["predictions"] is not None:
        state = "SEALED_PENDING_SCORE"
    elif existing["freeze"] is not None:
        state = "FROZEN_PENDING_PREDICT"
    elif existing["index"] is not None:
        state = "INDEXED_PENDING_CANDIDATES_AND_FREEZE"
    else:
        state = "NOT_INDEXED"
    return {"state": state, "artifacts": existing}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--run-root", type=Path, required=True)
    index_parser.add_argument("--model-root", type=Path, required=True)
    index_parser.add_argument("--model-manifest-sha256")
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-root", type=Path, required=True)
    freeze_parser.add_argument("--model-root", type=Path, required=True)
    freeze_parser.add_argument("--model-manifest-sha256")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--run-root", type=Path, required=True)
    predict_parser.add_argument("--dry-run", action="store_true")
    predict_parser.add_argument("--limit-per-episode", type=int, default=12)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "index":
        value = build_index(args)
    elif args.command == "freeze":
        value = freeze(args)
    elif args.command == "predict":
        value = predict(args)
    else:
        value = status(args)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
