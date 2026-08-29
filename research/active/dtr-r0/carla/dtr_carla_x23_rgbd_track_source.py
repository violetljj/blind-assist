from __future__ import annotations

"""Freeze, predict, one-score, and report the CARLA V18 X23 RGBD source.

Prediction opens only the frozen V18 causal input index and its NPZ payloads,
the frozen YOLO candidate files, and the sealed X22 prediction artifact used
for the same-source X21 baseline.  Evaluator rows are opened only by ``score``
after an exclusive one-attempt receipt has been written.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
DTR_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import carla_raw_source as carla_source  # noqa: E402
import dtr_carla_x22_flow_veto as x22  # noqa: E402


EXPERIMENT_ID = "CARLA_DTR_V18_X23_RGBD_TRACK_SOURCE_ONE_SCORE"
FREEZE_SCHEMA = "blindassist-dtr-carla-v18-x23-rgbd-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-v18-x21-x23-predictions-v1"
RESULT_SCHEMA = "blindassist-dtr-carla-v18-x23-rgbd-one-score-result-v1"
SCORE_ATTEMPT_SCHEMA = "blindassist-dtr-carla-v18-x23-rgbd-score-attempt-v1"
ARM_X21 = x22.ARM_X21
ARM_X23 = "X23_CURRENT_MASK_RGBD_CAUSAL_TRACK"

SCENARIOS = (
    "static_clear_mirror",
    "moving_parallel_clear_mirror",
    "crossing_clear_mirror",
    "crossing_physical_occlusion_mirror",
)
FRAMES_PER_SCENARIO = 201
EXPECTED_FRAMES = len(SCENARIOS) * FRAMES_PER_SCENARIO

MIN_FOREGROUND_DEPTH_POINTS = 32
CAMERA_YAW_IN_CARLA_DEG = 10.0
CAMERA_FORWARD_OFFSET_M = 0.20
VELOCITY_WINDOW_S = 0.50
HOLD_WINDOW_S = 0.60
RISK_CONFIRM_FRAMES = 2
MIN_FIT_SAMPLES = 4
MIN_FIT_SPAN_S = 0.15
EPSILON = 1e-9

ALLOWED_FRAME_ARRAYS = frozenset(
    {
        "points_lidar",
        "intensity",
        "image_rgb",
        "flow_xy_px",
        "marker_world",
        "projection_world",
        "camera_intrinsic",
        "K",
        "pose_valid",
        "image_valid",
        "flow_valid",
        "source_sample_index",
    }
)
FORBIDDEN_PATH_PARTS = frozenset({"evaluator", "instance", "actor", "contact", "route-truth"})
DEFAULT_PROTOCOL = HERE / "dtr_carla_x23_rgbd_track_protocol.json"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"x23_json_object:{path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except BaseException:
        raise


def assert_truth_blind(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            require(str(key).lower() not in x22.ONLINE_FORBIDDEN_KEYS, f"x23_forbidden_key:{path}.{key}")
            assert_truth_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_truth_blind(child, f"{path}[{index}]")


def require_model_path(path: Path, label: str) -> Path:
    value = path.resolve(strict=True)
    require(not (FORBIDDEN_PATH_PARTS & {part.lower() for part in value.parts}), f"x23_privileged_path:{label}:{value}")
    return value


def _paths(run_root: Path) -> dict[str, Path]:
    return {
        "freeze": run_root / "freeze-v18-x23.json",
        "predictions": run_root / "predictions-v18-x23.json",
        "score_attempt": run_root / "score-attempt-v18-x23.json",
        "result": run_root / "result-v18-x23.json",
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve(strict=True))
    require(protocol.get("schema") == "blindassist-dtr-carla-v18-x23-protocol-v1", "x23_protocol_schema")
    require(protocol.get("experiment_id") == EXPERIMENT_ID, "x23_protocol_experiment")
    require(tuple(protocol.get("scenarios", [])) == SCENARIOS, "x23_protocol_scenarios")
    require(int(protocol.get("expected_frames", -1)) == EXPECTED_FRAMES, "x23_protocol_frames")
    constants = protocol["x23_contract"]["fixed_constants"]
    expected_constants = {
        "minimum_foreground_depth_points": MIN_FOREGROUND_DEPTH_POINTS,
        "camera_yaw_in_carla_degrees": CAMERA_YAW_IN_CARLA_DEG,
        "camera_forward_offset_m": CAMERA_FORWARD_OFFSET_M,
        "velocity_window_seconds": VELOCITY_WINDOW_S,
        "hold_window_seconds": HOLD_WINDOW_S,
        "risk_confirmation_consecutive_frames": RISK_CONFIRM_FRAMES,
        "route_half_width_m": x22.ROUTE_HALF_WIDTH_M,
        "route_horizon_seconds": x22.ROUTE_HORIZON_S,
        "minimum_closing_speed_mps": x22.MIN_CLOSING_SPEED_MPS,
        "track_radius_m": 0.0,
    }
    require(constants == expected_constants, "x23_protocol_constants")
    gates = protocol["x23_contract"]["one_score_gates"]
    require(
        gates
        == {
            "expected_physical_gap_frames": 11,
            "minimum_occluded_gap_frames": 9,
            "minimum_x23_minus_x21_gap_coverage": 0.50,
            "minimum_occluded_first_alert_lead_seconds": 2.5,
            "minimum_clear_first_alert_lead_seconds": 2.5,
            "maximum_new_static_route_risk_frames": 0,
            "maximum_new_parallel_route_risk_frames": 0,
            "maximum_new_prethreat_route_risk_frames": 0,
        },
        "x23_protocol_gates",
    )
    expected_files = {
        "project_adapter": Path(carla_source.__file__).resolve(),
        "project_x22_evaluator": Path(x22.__file__).resolve(),
        "project_x23": Path(__file__).resolve(),
    }
    frozen_files = protocol["frozen_files"]
    require(set(frozen_files) == set(expected_files), "x23_protocol_frozen_file_set")
    for name, expected_path in expected_files.items():
        reference = frozen_files[name]
        file_path = Path(reference["path"]).resolve(strict=True)
        require(file_path == expected_path and sha256_file(file_path) == str(reference["sha256"]).upper(), f"x23_protocol_file:{name}")
    return protocol


def flatten_index(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in index["sequences"]:
        rows.extend(dict(row) for row in sequence["frames"])
    return rows


def load_input_index(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = require_model_path(path, "input_index")
    index = read_json(path)
    require(index.get("schema") == carla_source.INPUT_SCHEMA and index.get("truth_blind") is True, "x23_input_schema")
    carla_source.assert_model_clean(index)
    require(index.get("online_modalities") == list(carla_source.MODEL_TOPICS), "x23_input_modalities")
    require(index.get("aggregate") == {"sequences": 4, "frames": EXPECTED_FRAMES, "expected_frames": EXPECTED_FRAMES}, "x23_input_aggregate")
    rows = flatten_index(index)
    require(len(rows) == EXPECTED_FRAMES, "x23_input_count")
    cursor = 0
    for scenario in SCENARIOS:
        for sample_index in range(FRAMES_PER_SCENARIO):
            row = rows[cursor]
            cursor += 1
            require(row.get("sequence") == scenario and int(row.get("sample_index", -1)) == sample_index, f"x23_input_order:{scenario}:{sample_index}")
            require(abs(float(row["sequence_time_s"]) - sample_index * 0.05) <= EPSILON, f"x23_input_clock:{scenario}:{sample_index}")
            require_model_path(path.parent / "frames" / f"{int(row['frame']):08d}.npz", "frame")
    return index, rows


def candidate_set_fingerprint(root: Path, rows: Sequence[Mapping[str, Any]], model_sha256: str) -> dict[str, Any]:
    root = require_model_path(root, "candidate_root")
    digest = hashlib.sha256()
    for row in rows:
        frame = int(row["frame"])
        path = require_model_path(root / f"{frame:08d}.json", "candidate")
        value = read_json(path)
        require(value.get("schema") == "blindassist-dtr-carla-v18-yolo-candidates-v1", f"x23_candidate_schema:{frame}")
        require(value.get("truth_blind") is True, f"x23_candidate_boundary:{frame}")
        require(
            int(value.get("frame", -1)) == frame
            and int(value.get("sample_index", -1)) == int(row["sample_index"])
            and value.get("sequence") == row["sequence"],
            f"x23_candidate_identity:{frame}",
        )
        require(str(value.get("frame_file_sha256", "")).upper() == str(row["frame_file_sha256"]).upper(), f"x23_candidate_frame_hash:{frame}")
        require(str(value.get("model_sha256", "")).upper() == model_sha256.upper(), f"x23_candidate_model:{frame}")
        assert_truth_blind(value)
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode("utf-8"))
    return {"path": str(root), "files": len(rows), "aggregate_sha256": digest.hexdigest().upper()}


def validate_baseline_predictions(path: Path) -> dict[str, Any]:
    value = read_json(require_model_path(path, "baseline_predictions"))
    require(value.get("schema") == x22.PREDICTION_SCHEMA and value.get("status") == "SEALED", "x23_baseline_schema")
    require(value.get("truth_blind") is True and value.get("experiment_id") == x22.EXPERIMENT_ID, "x23_baseline_boundary")
    require(ARM_X21 in value.get("arms", {}), "x23_baseline_arm")
    for scenario in SCENARIOS:
        arm = value["arms"][ARM_X21]["scenarios"][scenario]
        require(len(arm["frames"]) == FRAMES_PER_SCENARIO, f"x23_baseline_frames:{scenario}")
    assert_truth_blind({key: child for key, child in value.items() if key != "forbidden_fields"})
    return value


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    require(not paths["freeze"].exists(), f"x23_freeze_exists:{paths['freeze']}")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = validate_protocol(protocol_path)
    source_root = args.source_root.resolve(strict=True)
    input_index_path = require_model_path(args.input_index, "input_index")
    input_index, rows = load_input_index(input_index_path)
    require(Path(str(input_index["source_root"])).resolve() == source_root, "x23_source_root_drift")

    model_path = require_model_path(args.model, "model")
    model_sha256 = sha256_file(model_path)
    candidates = candidate_set_fingerprint(args.candidate_root, rows, model_sha256)
    model_manifest_path, _model_manifest = carla_source._load_model_manifest(source_root)
    require(str(input_index["model_manifest"]["sha256"]).upper() == sha256_file(model_manifest_path), "x23_model_manifest_hash")
    # Do not open, stat, or hash evaluator material before the one-score
    # receipt.  Freeze records only the protocol-known expected identity.
    evaluator_manifest_path = source_root / carla_source.EVALUATOR_MANIFEST_RELATIVE
    baseline_path = require_model_path(args.baseline_predictions, "baseline_predictions")
    validate_baseline_predictions(baseline_path)

    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_NO_EVALUATOR_ROWS_PARSED",
        "truth_blind_prediction": True,
        "experiment_id": EXPERIMENT_ID,
        "source_root": str(source_root),
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "input_index": {"path": str(input_index_path), "sha256": sha256_file(input_index_path), "frames": len(rows)},
        "source_model_manifest": {"path": str(model_manifest_path), "sha256": sha256_file(model_manifest_path)},
        "detector_model": {"path": str(model_path), "sha256": model_sha256},
        "detector_candidates": candidates,
        "evaluator_manifest_expected": {
            "path": str(evaluator_manifest_path),
            "sha256": str(protocol["evaluator_manifest_expected_sha256"]).upper(),
        },
        "x22_baseline_predictions": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
        "algorithm_files": {
            name: {"path": str(Path(reference["path"]).resolve()), "sha256": str(reference["sha256"]).upper()}
            for name, reference in protocol["frozen_files"].items()
        },
        "arms": [ARM_X21, ARM_X23],
        "fixed_constants": protocol["x23_contract"]["fixed_constants"],
        "one_score_gates": protocol["x23_contract"]["one_score_gates"],
        "forbidden": protocol["x23_contract"]["forbidden"],
    }
    atomic_json(paths["freeze"], value)
    return {**value, "freeze_sha256": sha256_file(paths["freeze"])}


def require_freeze(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = _paths(run_root)
    frozen = read_json(paths["freeze"].resolve(strict=True))
    require(frozen.get("schema") == FREEZE_SCHEMA and frozen.get("status") == "FROZEN_TRUTH_BLIND_NO_EVALUATOR_ROWS_PARSED", "x23_freeze_schema")
    protocol_path = Path(frozen["protocol"]["path"]).resolve(strict=True)
    require(sha256_file(protocol_path) == frozen["protocol"]["sha256"], "x23_protocol_drift")
    protocol = validate_protocol(protocol_path)
    for reference in frozen["algorithm_files"].values():
        path = Path(reference["path"]).resolve(strict=True)
        require(sha256_file(path) == reference["sha256"], f"x23_algorithm_drift:{path}")
    for key in ("input_index", "source_model_manifest", "detector_model", "x22_baseline_predictions"):
        reference = frozen[key]
        path = Path(reference["path"]).resolve(strict=True)
        require(sha256_file(path) == reference["sha256"], f"x23_frozen_input_drift:{key}")
    expected_evaluator_path = Path(frozen["evaluator_manifest_expected"]["path"])
    require(
        expected_evaluator_path == Path(frozen["source_root"]) / carla_source.EVALUATOR_MANIFEST_RELATIVE
        and frozen["evaluator_manifest_expected"]["sha256"] == str(protocol["evaluator_manifest_expected_sha256"]).upper(),
        "x23_evaluator_expected_identity_drift",
    )
    input_path = Path(frozen["input_index"]["path"]).resolve(strict=True)
    _index, rows = load_input_index(input_path)
    candidates = candidate_set_fingerprint(Path(frozen["detector_candidates"]["path"]), rows, frozen["detector_model"]["sha256"])
    require(candidates == frozen["detector_candidates"], "x23_candidate_set_drift")
    validate_baseline_predictions(Path(frozen["x22_baseline_predictions"]["path"]))
    return frozen, protocol, rows


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, self.y2)


@dataclass
class TrackerBox:
    bbox: BBox
    last_time_s: float


class CausalPersonTracker:
    """The V18 materializer's causal greedy bbox/footpoint association."""

    def __init__(self) -> None:
        self.tracks: dict[str, TrackerBox] = {}
        self.next_id = 1

    @staticmethod
    def iou(left: BBox, right: BBox) -> float:
        width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
        height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = width * height
        union = (left.x2 - left.x1) * (left.y2 - left.y1) + (right.x2 - right.x1) * (right.y2 - right.y1) - intersection
        return intersection / union if union > 0.0 else 0.0

    def update(self, boxes: Sequence[BBox], time_s: float) -> list[str]:
        self.tracks = {key: state for key, state in self.tracks.items() if time_s - state.last_time_s <= 0.75 + EPSILON}
        candidates: list[tuple[float, float, str, int]] = []
        for key, state in self.tracks.items():
            previous_x, previous_y = state.bbox.bottom_center
            for index, box in enumerate(boxes):
                current_x, current_y = box.bottom_center
                overlap = self.iou(state.bbox, box)
                distance = math.hypot(previous_x - current_x, previous_y - current_y)
                if overlap >= 0.10 or distance <= 80.0:
                    candidates.append((-overlap, distance, key, index))
        candidates.sort()
        assigned: dict[int, str] = {}
        used: set[str] = set()
        for _overlap, _distance, key, index in candidates:
            if key not in used and index not in assigned:
                assigned[index] = key
                used.add(key)
        output: list[str] = []
        for index, box in enumerate(boxes):
            key = assigned.get(index)
            if key is None:
                key = f"person-{self.next_id:06d}"
                self.next_id += 1
            self.tracks[key] = TrackerBox(box, time_s)
            output.append(key)
        return output


@dataclass
class TrackState:
    measurements: list[tuple[float, np.ndarray]] = field(default_factory=list)
    last_seen_s: float = -math.inf
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None


def polygon_mask(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    points = [
        (max(0, min(width - 1, int(round(float(x))))), max(0, min(height - 1, int(round(float(y))))))
        for x, y in polygon
    ]
    if len(points) >= 3:
        ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def project_points(points: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (x > 0.20) & (x < 60.0)
    pixels = np.full((len(points), 2), -1, dtype=np.int32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    pixels[valid, 0] = np.rint(cx - fx * y[valid] / x[valid]).astype(np.int32)
    pixels[valid, 1] = np.rint(cy - fy * z[valid] / x[valid]).astype(np.int32)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, valid


def robust_mask_position(points: np.ndarray, pixels: np.ndarray, projected_valid: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    indices = np.flatnonzero(projected_valid)
    selected = points[indices[mask[pixels[indices, 1], pixels[indices, 0]]]].astype(np.float64)
    if len(selected) < MIN_FOREGROUND_DEPTH_POINTS:
        return None, int(len(selected))
    depth_q15 = float(np.quantile(selected[:, 0], 0.15))
    slab_m = max(0.60, min(1.25, 0.10 * depth_q15))
    foreground = selected[selected[:, 0] <= depth_q15 + slab_m]
    if len(foreground) < MIN_FOREGROUND_DEPTH_POINTS:
        return None, int(len(foreground))
    return np.median(foreground, axis=0), int(len(foreground))


def camera_to_route(position: np.ndarray) -> np.ndarray:
    angle = math.radians(-CAMERA_YAW_IN_CARLA_DEG)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = [float(value) for value in position]
    return np.asarray([CAMERA_FORWARD_OFFSET_M + cosine * x - sine * y, sine * x + cosine * y, z], dtype=np.float64)


def robust_motion(history: Sequence[tuple[float, np.ndarray]], now_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    window = [row for row in history if now_s - row[0] <= VELOCITY_WINDOW_S + EPSILON]
    if len(window) < MIN_FIT_SAMPLES or window[-1][0] - window[0][0] < MIN_FIT_SPAN_S - EPSILON:
        return None
    times = np.asarray([row[0] for row in window], dtype=np.float64)
    positions = np.stack([row[1] for row in window]).astype(np.float64)
    slopes: list[np.ndarray] = []
    for left in range(len(window)):
        for right in range(left + 1, len(window)):
            delta_s = times[right] - times[left]
            if delta_s >= 0.10 - EPSILON:
                slopes.append((positions[right] - positions[left]) / delta_s)
    if not slopes:
        return None
    velocity = np.median(np.stack(slopes), axis=0)
    position = np.median(positions - (times - now_s)[:, None] * velocity[None, :], axis=0)
    return position, velocity


def first_tube_entry_s(position: np.ndarray, velocity: np.ndarray) -> float | None:
    x, y = float(position[0]), float(position[1])
    vx, vy = float(velocity[0]), float(velocity[1])
    radius = x22.ROUTE_HALF_WIDTH_M
    distance = math.hypot(x, y)
    speed_squared = vx * vx + vy * vy
    if distance <= radius + EPSILON:
        closing = math.hypot(vx, vy) if distance <= EPSILON else -(x * vx + y * vy) / distance
        return 0.0 if closing >= x22.MIN_CLOSING_SPEED_MPS else None
    if speed_squared <= EPSILON:
        return None
    b = 2.0 * (x * vx + y * vy)
    c = x * x + y * y - radius * radius
    discriminant = b * b - 4.0 * speed_squared * c
    if discriminant < 0.0:
        return None
    root = (-b - math.sqrt(max(0.0, discriminant))) / (2.0 * speed_squared)
    if root < -EPSILON or root > x22.ROUTE_HORIZON_S + EPSILON:
        return None
    entry_s = max(0.0, root)
    entry_x, entry_y = x + vx * entry_s, y + vy * entry_s
    inward = -(entry_x * vx + entry_y * vy) / max(EPSILON, math.hypot(entry_x, entry_y))
    return entry_s if inward + EPSILON >= x22.MIN_CLOSING_SPEED_MPS else None


def load_frame(input_index_path: Path, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = require_model_path(input_index_path.parent / "frames" / f"{int(row['frame']):08d}.npz", "frame")
    require(sha256_file(path) == str(row["frame_file_sha256"]).upper(), f"x23_frame_hash:{row['frame']}")
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == ALLOWED_FRAME_ARRAYS, f"x23_frame_arrays:{row['frame']}")
        require(bool(values["pose_valid"][0]) and bool(values["image_valid"][0]), f"x23_frame_validity:{row['frame']}")
        return {name: values[name].copy() for name in values.files}


def predict_scenario(
    scenario: str,
    rows: Sequence[Mapping[str, Any]],
    input_index_path: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    tracker = CausalPersonTracker()
    states: dict[str, TrackState] = {}
    risk_streaks: dict[str, int] = {}
    frames: list[dict[str, Any]] = []
    raw_detection_frames = metric_detection_frames = track_state_frames = hold_frames = 0
    for row in rows:
        sample_index = int(row["sample_index"])
        time_s = float(row["sequence_time_s"])
        frame = int(row["frame"])
        values = load_frame(input_index_path, row)
        points = values["points_lidar"].astype(np.float64)
        intrinsic = values["K"].astype(np.float64)
        height, width = values["image_rgb"].shape[:2]
        candidates = read_json(require_model_path(candidate_root / f"{frame:08d}.json", "candidate"))["candidates"]
        people = [candidate for candidate in candidates if int(candidate["class_id"]) == 0]
        raw_detection_frames += int(bool(people))
        boxes = [BBox(*[float(value) for value in candidate["bbox_xyxy"]]) for candidate in people]
        track_ids = tracker.update(boxes, time_s)
        pixels, projected_valid = project_points(points, intrinsic, width, height)
        measured_ids: set[str] = set()
        support_by_id: dict[str, int] = {}
        for track_id, candidate in zip(track_ids, people):
            mask = polygon_mask(candidate["polygon_xy"], width, height)
            position, support = robust_mask_position(points, pixels, projected_valid, mask)
            support_by_id[track_id] = support
            if position is None:
                continue
            state = states.setdefault(track_id, TrackState())
            state.measurements.append((time_s, camera_to_route(position)))
            state.measurements = [item for item in state.measurements if time_s - item[0] <= 1.0 + EPSILON]
            state.last_seen_s = time_s
            motion = robust_motion(state.measurements, time_s)
            if motion is not None:
                state.position, state.velocity = motion
            measured_ids.add(track_id)
        metric_detection_frames += int(bool(measured_ids))

        emitted: list[dict[str, Any]] = []
        emitted_ids: set[str] = set()
        for track_id, state in sorted(states.items()):
            if state.position is None or state.velocity is None:
                continue
            age_s = time_s - state.last_seen_s
            if age_s > HOLD_WINDOW_S + EPSILON:
                continue
            disposition = "MEASURED" if track_id in measured_ids else "HOLD"
            position = state.position if disposition == "MEASURED" else state.position + state.velocity * max(0.0, age_s)
            entry_s = first_tube_entry_s(position, state.velocity)
            risk_streaks[track_id] = risk_streaks.get(track_id, 0) + 1 if entry_s is not None else 0
            confirmed = entry_s is not None and risk_streaks[track_id] >= RISK_CONFIRM_FRAMES
            emitted_ids.add(track_id)
            emitted.append(
                {
                    "track_id": track_id,
                    "disposition": disposition,
                    "evidence_age_s": max(0.0, age_s),
                    "forward_m": float(position[0]),
                    "left_m": float(position[1]),
                    "velocity_forward_mps": float(state.velocity[0]),
                    "velocity_left_mps": float(state.velocity[1]),
                    "foreground_depth_points": support_by_id.get(track_id),
                    "tube_entry_s": entry_s,
                    "tube_risk_confirmed": confirmed,
                }
            )
        for track_id in set(risk_streaks) - emitted_ids:
            risk_streaks[track_id] = 0
        confirmed_entries = [float(value["tube_entry_s"]) for value in emitted if value["tube_risk_confirmed"]]
        route_risk = bool(confirmed_entries)
        track_state_frames += int(bool(emitted))
        hold_frames += int(any(value["disposition"] == "HOLD" for value in emitted))
        frames.append(
            {
                "sample_index": sample_index,
                "time_s": time_s,
                "route_risk": route_risk,
                "minimum_entry_s": min(confirmed_entries) if confirmed_entries else None,
                "emitted_states": len(emitted),
                "measured_states": sum(value["disposition"] == "MEASURED" for value in emitted),
                "held_states": sum(value["disposition"] == "HOLD" for value in emitted),
                "states": emitted,
            }
        )
    risk_indices = [int(row["sample_index"]) for row in frames if row["route_risk"]]
    return {
        "frames": frames,
        "route_risk_sample_indices": risk_indices,
        "route_risk_frames": len(risk_indices),
        "raw_person_detection_frames": raw_detection_frames,
        "metric_detection_frames": metric_detection_frames,
        "track_state_frames": track_state_frames,
        "track_coverage": track_state_frames / max(1, len(rows)),
        "hold_frames": hold_frames,
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    if not args.dry_run:
        require(not paths["predictions"].exists(), f"x23_predictions_exist:{paths['predictions']}")
    frozen, protocol, rows = require_freeze(run_root)
    input_index_path = Path(frozen["input_index"]["path"]).resolve(strict=True)
    candidate_root = Path(frozen["detector_candidates"]["path"]).resolve(strict=True)
    baseline = validate_baseline_predictions(Path(frozen["x22_baseline_predictions"]["path"]))
    limit = args.limit_per_scenario if args.dry_run else FRAMES_PER_SCENARIO
    require(1 <= limit <= FRAMES_PER_SCENARIO, "x23_predict_limit")
    x23_scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["sequence"] == scenario][:limit]
        x23_scenarios[scenario] = predict_scenario(scenario, scenario_rows, input_index_path, candidate_root)
    if args.dry_run:
        return {
            "schema": "blindassist-dtr-carla-v18-x23-predict-dry-check-v1",
            "status": "DRY_CHECK_COMPLETE_NO_PREDICTIONS_WRITTEN",
            "frames_per_scenario": limit,
            "scenarios": {
                name: {
                    key: value
                    for key, value in scenario.items()
                    if key != "frames" and key != "route_risk_sample_indices"
                }
                for name, scenario in x23_scenarios.items()
            },
        }
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED",
        "truth_blind": True,
        "experiment_id": EXPERIMENT_ID,
        "arms": {
            ARM_X21: {"scenarios": baseline["arms"][ARM_X21]["scenarios"]},
            ARM_X23: {"scenarios": x23_scenarios},
        },
        "fixed_constants": protocol["x23_contract"]["fixed_constants"],
        "diagnostics": {"frames": EXPECTED_FRAMES},
        "source": {
            "freeze_sha256": sha256_file(paths["freeze"]),
            "input_index_sha256": frozen["input_index"]["sha256"],
            "candidate_set_sha256": frozen["detector_candidates"]["aggregate_sha256"],
            "x22_baseline_predictions_sha256": frozen["x22_baseline_predictions"]["sha256"],
        },
        "forbidden_fields": sorted(x22.ONLINE_FORBIDDEN_KEYS),
    }
    assert_truth_blind({key: child for key, child in value.items() if key != "forbidden_fields"})
    atomic_json(paths["predictions"], value)
    return {**value, "predictions_sha256": sha256_file(paths["predictions"])}


def arm_scenario(predictions: Mapping[str, Any], arm: str, scenario: str) -> dict[str, Any]:
    value = predictions["arms"][arm]["scenarios"][scenario]
    require(len(value["frames"]) == FRAMES_PER_SCENARIO, f"x23_prediction_frames:{arm}:{scenario}")
    return value


def first_risk_time(value: Mapping[str, Any]) -> float | None:
    return next((float(row["time_s"]) for row in value["frames"] if row["route_risk"]), None)


def first_contact_time(rows: Sequence[Mapping[str, Any]], label: str) -> float:
    values = [float(row["elapsed_seconds"]) for row in rows if bool(row["truth"]["current_contact"])]
    require(bool(values), f"x23_contact_missing:{label}")
    return min(values)


def score(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    paths = _paths(run_root)
    require(not paths["score_attempt"].exists(), f"x23_score_attempt_consumed:{paths['score_attempt']}")
    require(not paths["result"].exists(), f"x23_result_exists:{paths['result']}")
    frozen, protocol, _rows = require_freeze(run_root)
    predictions_path = paths["predictions"].resolve(strict=True)
    predictions = read_json(predictions_path)
    require(predictions.get("schema") == PREDICTION_SCHEMA and predictions.get("status") == "SEALED", "x23_prediction_schema")
    require(predictions.get("truth_blind") is True, "x23_prediction_boundary")
    require(predictions["source"]["freeze_sha256"] == sha256_file(paths["freeze"]), "x23_prediction_freeze")
    source_root = Path(frozen["source_root"]).resolve(strict=True)
    evaluator_manifest_path = Path(frozen["evaluator_manifest_expected"]["path"])

    exclusive_json(
        paths["score_attempt"],
        {
            "schema": SCORE_ATTEMPT_SCHEMA,
            "attempt": 1,
            "status": "CONSUMED_BEFORE_EVALUATOR_ROWS_OPEN",
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "freeze_sha256": sha256_file(paths["freeze"]),
            "evaluator_manifest_expected": {
                "path": str(evaluator_manifest_path),
                "sha256": frozen["evaluator_manifest_expected"]["sha256"],
            },
        },
    )

    evaluator_manifest_path = evaluator_manifest_path.resolve(strict=True)
    require(
        sha256_file(evaluator_manifest_path) == frozen["evaluator_manifest_expected"]["sha256"],
        "x23_evaluator_manifest_hash",
    )
    evaluator = carla_source.load_evaluator_truth(source_root)
    require(set(evaluator) == set(SCENARIOS), "x23_evaluator_scenarios")
    clear_name = protocol["source_admission"]["clear_crossing_scenario"]
    occluded_name = protocol["source_admission"]["occluded_crossing_scenario"]
    static_name = "static_clear_mirror"
    parallel_name = "moving_parallel_clear_mirror"
    occluded_rows = evaluator[occluded_name]
    physical_gap = {
        int(row["sample_index"])
        for row in occluded_rows
        if bool(row["evaluator"]["physical_loss"])
    }
    gates = protocol["x23_contract"]["one_score_gates"]
    require(len(physical_gap) == int(gates["expected_physical_gap_frames"]), f"x23_physical_gap_count:{len(physical_gap)}")
    require(all(bool(occluded_rows[index]["truth"]["future_contact_within_horizon"]) for index in physical_gap), "x23_physical_gap_positive")

    x21_occluded = arm_scenario(predictions, ARM_X21, occluded_name)
    x23_occluded = arm_scenario(predictions, ARM_X23, occluded_name)
    x21_gap_hits = physical_gap & set(x21_occluded["route_risk_sample_indices"])
    x23_gap_hits = physical_gap & set(x23_occluded["route_risk_sample_indices"])
    denominator = len(physical_gap)
    x21_coverage = len(x21_gap_hits) / denominator
    x23_coverage = len(x23_gap_hits) / denominator
    improvement = x23_coverage - x21_coverage

    x21_clear = arm_scenario(predictions, ARM_X21, clear_name)
    x23_clear = arm_scenario(predictions, ARM_X23, clear_name)
    first_contact_occluded = first_contact_time(occluded_rows, occluded_name)
    first_contact_clear = first_contact_time(evaluator[clear_name], clear_name)
    x23_first_occluded = first_risk_time(x23_occluded)
    x23_first_clear = first_risk_time(x23_clear)
    occluded_lead_s = None if x23_first_occluded is None else first_contact_occluded - x23_first_occluded
    clear_lead_s = None if x23_first_clear is None else first_contact_clear - x23_first_clear

    new_static = sorted(set(arm_scenario(predictions, ARM_X23, static_name)["route_risk_sample_indices"]) - set(arm_scenario(predictions, ARM_X21, static_name)["route_risk_sample_indices"]))
    new_parallel = sorted(set(arm_scenario(predictions, ARM_X23, parallel_name)["route_risk_sample_indices"]) - set(arm_scenario(predictions, ARM_X21, parallel_name)["route_risk_sample_indices"]))
    new_prethreat: dict[str, list[int]] = {}
    for scenario in SCENARIOS:
        x21_risk = set(arm_scenario(predictions, ARM_X21, scenario)["route_risk_sample_indices"])
        x23_risk = set(arm_scenario(predictions, ARM_X23, scenario)["route_risk_sample_indices"])
        negative = {
            int(row["sample_index"])
            for row in evaluator[scenario]
            if not bool(row["truth"]["future_contact_within_horizon"])
        }
        added = sorted((x23_risk - x21_risk) & negative)
        if added:
            new_prethreat[scenario] = added

    checks = {
        "physical_gap_at_least_9_of_11": len(x23_gap_hits) >= int(gates["minimum_occluded_gap_frames"]),
        "x23_minus_x21_gap_coverage_at_least_50pp": improvement + EPSILON >= float(gates["minimum_x23_minus_x21_gap_coverage"]),
        "occluded_first_alert_lead_at_least_2_5s": occluded_lead_s is not None and occluded_lead_s + EPSILON >= float(gates["minimum_occluded_first_alert_lead_seconds"]),
        "clear_first_alert_lead_at_least_2_5s": clear_lead_s is not None and clear_lead_s + EPSILON >= float(gates["minimum_clear_first_alert_lead_seconds"]),
        "static_zero_new_risk": len(new_static) <= int(gates["maximum_new_static_route_risk_frames"]),
        "parallel_zero_new_risk": len(new_parallel) <= int(gates["maximum_new_parallel_route_risk_frames"]),
        "prethreat_zero_new_risk": sum(map(len, new_prethreat.values())) <= int(gates["maximum_new_prethreat_route_risk_frames"]),
        "prediction_truth_boundary": predictions.get("forbidden_fields") == sorted(x22.ONLINE_FORBIDDEN_KEYS),
        "one_score_attempt_consumed": paths["score_attempt"].is_file(),
    }
    passed = all(checks.values())
    result = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "CARLA_DTR_X23_RGBD_TRACK_SYNTHETIC_DEVELOPMENT_GATE_MET" if passed else "CARLA_DTR_X23_RGBD_TRACK_SYNTHETIC_DEVELOPMENT_GATE_NOT_MET",
        "gate": {"passed": passed, "checks": checks},
        "metrics": {
            "physical_gap_sample_indices": sorted(physical_gap),
            "physical_gap_frames": denominator,
            ARM_X21: {"gap_hits": sorted(x21_gap_hits), "gap_frames": len(x21_gap_hits), "gap_coverage": x21_coverage},
            ARM_X23: {"gap_hits": sorted(x23_gap_hits), "gap_frames": len(x23_gap_hits), "gap_coverage": x23_coverage},
            "x23_minus_x21_gap_coverage": improvement,
            "x23_minus_x21_gap_coverage_percentage_points": improvement * 100.0,
            "occluded_first_contact_time_s": first_contact_occluded,
            "x23_first_occluded_risk_time_s": x23_first_occluded,
            "x23_occluded_first_alert_lead_s": occluded_lead_s,
            "clear_first_contact_time_s": first_contact_clear,
            "x23_first_clear_risk_time_s": x23_first_clear,
            "x23_clear_first_alert_lead_s": clear_lead_s,
            "new_static_risk_sample_indices": new_static,
            "new_parallel_risk_sample_indices": new_parallel,
            "new_prethreat_risk_sample_indices": new_prethreat,
        },
        "sources": {
            "freeze": {"path": str(paths["freeze"]), "sha256": sha256_file(paths["freeze"])},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "score_attempt": {"path": str(paths["score_attempt"]), "sha256": sha256_file(paths["score_attempt"])},
            "evaluator_manifest": {"path": str(evaluator_manifest_path), "sha256": sha256_file(evaluator_manifest_path)},
        },
        "decision": {"next": "PROMOTE_X23_TO_NEW_SOURCE_DEVELOPMENT" if passed else "CLOSE_FROZEN_X23_ARM_WITHOUT_SWEEP"},
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(paths["result"], result)
    return {**result, "result_sha256": sha256_file(paths["result"])}


def status(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.run_root.resolve())
    artifacts = {
        name: None if not path.is_file() else {"path": str(path), "sha256": sha256_file(path), "value": read_json(path)}
        for name, path in paths.items()
    }
    result = artifacts["result"]
    if result is not None:
        gate = result["value"]["gate"]
        state = "GATE_MET" if gate["passed"] else "GATE_NOT_MET"
    elif artifacts["score_attempt"] is not None:
        gate = None
        state = "SCORE_ATTEMPT_CONSUMED_RESULT_MISSING"
    elif artifacts["predictions"] is not None:
        gate = None
        state = "SEALED_PENDING_ONE_SCORE"
    elif artifacts["freeze"] is not None:
        gate = None
        state = "FROZEN_PENDING_PREDICT"
    else:
        gate = None
        state = "NOT_FROZEN"
    return {"state": state, "gate": gate, "artifacts": artifacts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-root", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--input-index", type=Path, required=True)
    freeze_parser.add_argument("--candidate-root", type=Path, required=True)
    freeze_parser.add_argument("--model", type=Path, required=True)
    freeze_parser.add_argument("--baseline-predictions", type=Path, required=True)
    freeze_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--run-root", type=Path, required=True)
    predict_parser.add_argument("--dry-run", action="store_true")
    predict_parser.add_argument("--limit-per-scenario", type=int, default=3)
    for command in ("score", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        value = freeze(args)
    elif args.command == "predict":
        value = predict(args)
    elif args.command == "score":
        value = score(args)
    else:
        value = status(args)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "score":
        return 0 if value["gate"]["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
