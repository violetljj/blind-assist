"""Depth-teacher dense field generation for the USTRF U0 upper-bound arm.

The module deliberately has no event-label input.  It converts a licensed,
hash-bound relative-depth model output and an external causal route into dense
auxiliary fields plus route-intrusion evidence.  It does not create lifecycle,
event IDs, feedback decisions, metric depth, or production authority.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


FIELD_SCHEMA = "blindassist_ustrf_sc_u0_dense_teacher_field_bundle_v2"
FIELD_CONTRACT_ID = "ustrf_sc_u0_dense_teacher_field_v2"
ARTIFACT_SCHEMA = "blindassist_ustrf_sc_u0_dense_teacher_loso_artifact_v2"
MODEL_NAME = "Depth Anything V2 Small"
MODEL_VERSION = "depth_anything_v2_vits_252_nhwc_onnx"
MODEL_LICENSE = "Apache-2.0"
MODEL_INPUT_CONTRACT = "rgb_imagenet_normalized_nhwc_252_v1"
MODEL_OUTPUT_CONTRACT = "relative_inverse_depth_nhwc_252_v1"
DECODE_POLICY = "opencv_video_capture_requested_pts_v1"
QUANTIZATION = "uint32_le_round_clip_0_1000000_v1"
QUANTIZATION_SCALE = 1_000_000
RISK_SOURCES = ["depth-anything-v2-small-relative-depth", "relative-depth-boundary"]
SOURCE_FIELD_NAMES = (
    "boundary_field",
    "local_obstacle_field",
    "unknown_field",
    "walkability_field",
)
ROUTE_FIELD_NAMES = ("route_relative_risk_field", "route_weight_field")
ALL_FIELD_NAMES = SOURCE_FIELD_NAMES + ROUTE_FIELD_NAMES


class DenseTeacherError(ValueError):
    pass


@dataclass(frozen=True)
class DenseTeacherConfig:
    grid_width: int = 63
    grid_height: int = 63
    corridor_half_width_ratio: float = 0.08
    maximum_route_age_ms: int = 1_000
    minimum_route_confidence: float = 0.5
    field_ttl_ms: int = 500
    depth_weight: float = 0.60
    boundary_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise DenseTeacherError("dense grid dimensions must be positive")
        if not 0 < self.corridor_half_width_ratio <= 0.5:
            raise DenseTeacherError("corridor half width ratio must be in (0, .5]")
        if self.maximum_route_age_ms < 0 or self.field_ttl_ms <= 0:
            raise DenseTeacherError("route age and field TTL are invalid")
        if not 0 <= self.minimum_route_confidence <= 1:
            raise DenseTeacherError("minimum route confidence must be in [0, 1]")
        if not 0 <= self.depth_weight <= 1 or not 0 <= self.boundary_weight <= 1:
            raise DenseTeacherError("dense field weights must be in [0, 1]")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(text.encode("utf-8"))


def require_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DenseTeacherError(f"{where} must be lowercase SHA256")
    return value


def validate_loso_artifact(
    artifact: Mapping[str, Any],
    *,
    held_out_session_id: str,
    training_manifest_sha256: str,
    model_sha256: str,
) -> None:
    expected = {
        "schema": ARTIFACT_SCHEMA,
        "fit_policy": "leave_one_session_out_fit_v1",
        "held_out_session_id": held_out_session_id,
        "training_input_manifest_sha256": training_manifest_sha256,
        "teacher_model_sha256": model_sha256,
        "teacher_model_name": MODEL_NAME,
        "teacher_model_version": MODEL_VERSION,
        "teacher_model_license": MODEL_LICENSE,
        "teacher_model_input_contract": MODEL_INPUT_CONTRACT,
        "teacher_model_output_contract": MODEL_OUTPUT_CONTRACT,
        "teacher_decode_policy": DECODE_POLICY,
        "teacher_inference_runtime": "onnxruntime_cpu_v1",
        "calibration_input_schema": "blindassist_ustrf_sc_u0_dense_teacher_calibration_inputs_v1",
        "blind_accessed": False,
        "future_inputs_used": False,
        "human_event_truth_used": False,
        "production_authorized": False,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise DenseTeacherError(f"LOSO artifact {key} mismatch")
    training_sessions = artifact.get("training_session_ids")
    if (
        not isinstance(training_sessions, list)
        or not training_sessions
        or training_sessions != sorted(set(training_sessions))
        or held_out_session_id in training_sessions
    ):
        raise DenseTeacherError("LOSO artifact training session inventory is invalid")
    calibration = artifact.get("calibration")
    if not isinstance(calibration, dict):
        raise DenseTeacherError("LOSO artifact calibration is required")
    low = calibration.get("raw_depth_lower_quantile")
    high = calibration.get("raw_depth_upper_quantile")
    gradient = calibration.get("gradient_upper_quantile")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (low, high, gradient)):
        raise DenseTeacherError("LOSO artifact calibration values must be finite")
    if not low < high or gradient <= 0:
        raise DenseTeacherError("LOSO artifact calibration ranges are invalid")
    for key in (
        "fit_implementation_file_sha256",
        "fit_implementation_sha256",
        "training_sample_inventory_sha256",
    ):
        require_sha(artifact.get(key), where=f"LOSO artifact {key}")
    training_samples = artifact.get("training_samples")
    if (
        not isinstance(training_samples, list)
        or not training_samples
        or calibration.get("training_frame_count") != len(training_samples)
    ):
        raise DenseTeacherError("LOSO artifact training sample provenance is invalid")
    identities: set[tuple[Any, Any, Any, Any]] = set()
    for index, sample in enumerate(training_samples):
        if not isinstance(sample, Mapping):
            raise DenseTeacherError("LOSO artifact training sample must be an object")
        identity = (
            sample.get("session_id"), sample.get("episode_id"),
            sample.get("frame_id"), sample.get("video_pts_ms"),
        )
        if (
            any(not isinstance(value, str) or not value for value in identity[:3])
            or not isinstance(identity[3], int) or isinstance(identity[3], bool) or identity[3] < 0
            or identity in identities
        ):
            raise DenseTeacherError("LOSO artifact training sample identity is invalid")
        identities.add(identity)
        for key in ("teacher_decoded_rgb_sha256", "raw_depth_sha256"):
            require_sha(sample.get(key), where=f"LOSO artifact training_samples[{index}].{key}")


def select_route_sample(
    route_episode: Mapping[str, Any],
    frame_timestamp_ms: int,
    config: DenseTeacherConfig,
) -> tuple[int, Mapping[str, Any]]:
    provider = route_episode.get("provider")
    coordinate = route_episode.get("coordinate_contract")
    if not isinstance(provider, dict) or provider.get("inferred_by_risk_model") is not False:
        raise DenseTeacherError("route provider is missing or inferred by the risk model")
    if not isinstance(coordinate, dict) or coordinate.get("space") != "normalized_current_camera_frame_xy":
        raise DenseTeacherError("dense teacher requires normalized current-camera route coordinates")
    samples = route_episode.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DenseTeacherError("route sample inventory is empty")
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise DenseTeacherError("route sample must be an object")
        timestamp = sample.get("timestamp_ms")
        if isinstance(timestamp, int) and not isinstance(timestamp, bool) and timestamp <= frame_timestamp_ms:
            candidates.append((index, sample))
    if not candidates:
        raise DenseTeacherError("no causal route sample")
    index, selected = max(candidates, key=lambda value: (value[1]["timestamp_ms"], value[0]))
    timestamp = selected["timestamp_ms"]
    valid_until = selected.get("valid_until_timestamp_ms")
    confidence = selected.get("confidence")
    if selected.get("route_valid") is not True:
        raise DenseTeacherError("selected route sample is invalid")
    if not isinstance(valid_until, int) or isinstance(valid_until, bool) or valid_until < frame_timestamp_ms:
        raise DenseTeacherError("selected route sample is stale")
    if frame_timestamp_ms - timestamp > config.maximum_route_age_ms:
        raise DenseTeacherError("selected route sample is too old")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or confidence < config.minimum_route_confidence:
        raise DenseTeacherError("selected route confidence is too low")
    waypoints = selected.get("horizon_waypoints")
    if not isinstance(waypoints, list) or [row.get("horizon_ms") for row in waypoints if isinstance(row, dict)] != [1_000, 2_000, 3_000]:
        raise DenseTeacherError("selected route waypoint contract is invalid")
    for row in waypoints:
        xy = row.get("xy_norm")
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1 for v in xy)
        ):
            raise DenseTeacherError("route waypoint coordinate is invalid")
    return index, selected


def _lazy_numeric() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise DenseTeacherError("numpy and opencv-python-headless are required for dense teacher inference") from error
    return np, cv2


def route_weight_field(sample: Mapping[str, Any], config: DenseTeacherConfig) -> Any:
    np, _ = _lazy_numeric()
    waypoints = [(0.5, 1.0)] + [tuple(map(float, row["xy_norm"])) for row in sample["horizon_waypoints"]]
    xs = (np.arange(config.grid_width, dtype=np.float32) + 0.5) / config.grid_width
    ys = (np.arange(config.grid_height, dtype=np.float32) + 0.5) / config.grid_height
    xx, yy = np.meshgrid(xs, ys)
    minimum = np.full((config.grid_height, config.grid_width), np.inf, dtype=np.float32)
    for start, end in zip(waypoints, waypoints[1:]):
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            distance = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2)
        else:
            t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / length_squared, 0.0, 1.0)
            distance = np.sqrt((xx - (ax + t * dx)) ** 2 + (yy - (ay + t * dy)) ** 2)
        minimum = np.minimum(minimum, distance)
    return np.clip(1.0 - minimum / config.corridor_half_width_ratio, 0.0, 1.0).astype(np.float32)


class DepthAnythingOnnxTeacher:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as error:
            raise DenseTeacherError("onnxruntime is required for dense teacher inference") from error
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise DenseTeacherError(f"teacher model is missing: {self.model_path}")
        self.model_sha256 = sha256_file(self.model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 252, 252, 3] or inputs[0].type != "tensor(float)":
            raise DenseTeacherError("teacher ONNX input contract mismatch")
        if len(outputs) != 1 or outputs[0].shape != [1, 252, 252, 1] or outputs[0].type != "tensor(float)":
            raise DenseTeacherError("teacher ONNX output contract mismatch")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def infer_rgb(self, rgb: Any) -> tuple[Any, float]:
        np, cv2 = _lazy_numeric()
        resized = cv2.resize(rgb, (252, 252), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((resized - mean) / std)[None, ...].astype(np.float32)
        started = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        duration_ms = (time.perf_counter() - started) * 1_000.0
        depth = np.asarray(output[0, :, :, 0], dtype=np.float32)
        if depth.shape != (252, 252) or not np.isfinite(depth).all():
            raise DenseTeacherError("teacher produced invalid relative-depth output")
        return depth, duration_ms


def decode_video_frame_rgb(video_path: Path, requested_pts_ms: int) -> tuple[Any, float]:
    np, cv2 = _lazy_numeric()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise DenseTeacherError(f"cannot open input video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(requested_pts_ms))
        started = time.perf_counter()
        ok, bgr = capture.read()
        duration_ms = (time.perf_counter() - started) * 1_000.0
        if not ok or bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise DenseTeacherError(f"cannot decode teacher frame at {requested_pts_ms}ms")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), duration_ms
    finally:
        capture.release()


def _quantize(field: Any) -> Any:
    np, _ = _lazy_numeric()
    return np.rint(np.clip(field, 0.0, 1.0) * QUANTIZATION_SCALE).astype("<u4")


def _field_payload_sha(width: int, height: int, fields: Mapping[str, bytes], *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{FIELD_CONTRACT_ID}|{domain}|{width}|{height}|{QUANTIZATION}\n".encode("ascii"))
    for name in sorted(fields):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(fields[name])
    return digest.hexdigest()


def _decode_serialized_fields(payload: Mapping[str, Any]) -> tuple[int, int, dict[str, Any], dict[str, bytes]]:
    np, _ = _lazy_numeric()
    width = payload.get("grid_width")
    height = payload.get("grid_height")
    if (
        not isinstance(width, int) or isinstance(width, bool) or width <= 0
        or not isinstance(height, int) or isinstance(height, bool) or height <= 0
        or payload.get("quantization") != QUANTIZATION
    ):
        raise DenseTeacherError("serialized dense field dimensions/quantization are invalid")
    encoded = payload.get("fields_base64")
    if not isinstance(encoded, Mapping) or set(encoded) != set(ALL_FIELD_NAMES):
        raise DenseTeacherError("serialized dense field inventory is invalid")
    arrays: dict[str, Any] = {}
    binary: dict[str, bytes] = {}
    expected_bytes = width * height * 4
    for name in ALL_FIELD_NAMES:
        value = encoded.get(name)
        if not isinstance(value, str):
            raise DenseTeacherError(f"serialized dense field {name} is not base64 text")
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise DenseTeacherError(f"serialized dense field {name} is invalid base64") from error
        if len(raw) != expected_bytes:
            raise DenseTeacherError(f"serialized dense field {name} byte length mismatch")
        array = np.frombuffer(raw, dtype="<u4").reshape((height, width))
        if bool((array > QUANTIZATION_SCALE).any()):
            raise DenseTeacherError(f"serialized dense field {name} exceeds fixed-point range")
        arrays[name] = array
        binary[name] = raw
    return width, height, arrays, binary


def summarize_serialized_field(payload: Mapping[str, Any]) -> dict[str, Any]:
    np, _ = _lazy_numeric()
    width, height, fields, binary = _decode_serialized_fields(payload)
    route_weight = fields["route_weight_field"]
    active = route_weight > 0
    if not bool(active.any()):
        raise DenseTeacherError("serialized route field has no contributing cells")
    total_weight = int(route_weight[active].astype(np.uint64).sum())
    route_relative = fields["route_relative_risk_field"]
    local = fields["local_obstacle_field"]
    boundary = fields["boundary_field"]
    unknown = fields["unknown_field"]
    source_binary = {name: binary[name] for name in SOURCE_FIELD_NAMES}
    route_binary = {name: binary[name] for name in ROUTE_FIELD_NAMES}
    return {
        "source_field_sha256": _field_payload_sha(width, height, source_binary, domain="source"),
        "route_interaction_field_sha256": _field_payload_sha(width, height, route_binary, domain="route-interaction"),
        "field_cell_count": width * height,
        "risk_evidence_count": int(active.sum()),
        "route_intrusion_score": float(route_relative[active].astype(np.uint64).sum() / total_weight),
        "maximum_route_cell_risk": float(np.maximum(local, boundary)[active].max() / QUANTIZATION_SCALE),
        "route_unknown_fraction": float(
            (unknown[active].astype(np.uint64) * route_weight[active].astype(np.uint64)).sum()
            / (QUANTIZATION_SCALE * total_weight)
        ),
    }


def validate_serialized_field_summary(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    expected = summarize_serialized_field(payload)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool) or abs(float(actual) - value) > 1e-12:
                raise DenseTeacherError(f"serialized dense field summary mismatch: {key}")
        elif actual != value:
            raise DenseTeacherError(f"serialized dense field summary mismatch: {key}")


def build_frame_field(
    raw_depth: Any,
    *,
    artifact: Mapping[str, Any],
    route_sample: Mapping[str, Any],
    config: DenseTeacherConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np, cv2 = _lazy_numeric()
    calibration = artifact["calibration"]
    low = float(calibration["raw_depth_lower_quantile"])
    high = float(calibration["raw_depth_upper_quantile"])
    gradient_upper = float(calibration["gradient_upper_quantile"])
    resized = cv2.resize(raw_depth, (config.grid_width, config.grid_height), interpolation=cv2.INTER_AREA)
    closeness = np.clip((resized - low) / (high - low), 0.0, 1.0).astype(np.float32)
    gx = cv2.Sobel(closeness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(closeness, cv2.CV_32F, 0, 1, ksize=3)
    boundary = np.clip(np.sqrt(gx * gx + gy * gy) / gradient_upper, 0.0, 1.0).astype(np.float32)
    local_obstacle = np.maximum(closeness * config.depth_weight, boundary * config.boundary_weight)
    walkability = 1.0 - local_obstacle
    range_margin = (high - low) * 0.25
    unknown = ((resized < low - range_margin) | (resized > high + range_margin) | ~np.isfinite(resized)).astype(np.float32)
    route_weight = route_weight_field(route_sample, config)
    route_relative = np.maximum(local_obstacle, boundary) * route_weight
    quantized_fields = {
        "local_obstacle_field": _quantize(local_obstacle),
        "walkability_field": _quantize(walkability),
        "boundary_field": _quantize(boundary),
        "unknown_field": _quantize(unknown),
        "route_weight_field": _quantize(route_weight),
        "route_relative_risk_field": _quantize(route_relative),
    }
    binary_fields = {name: value.tobytes(order="C") for name, value in quantized_fields.items()}
    payload = {
        "grid_width": config.grid_width,
        "grid_height": config.grid_height,
        "quantization": QUANTIZATION,
        "fields_base64": {name: base64.b64encode(value).decode("ascii") for name, value in binary_fields.items()},
    }
    summary = {
        **summarize_serialized_field(payload),
        "risk_sources": list(RISK_SOURCES),
    }
    validate_serialized_field_summary(payload, summary)
    return payload, summary


def generate_field_bundle(
    *,
    request: Mapping[str, Any],
    video_path: Path,
    route_episode: Mapping[str, Any],
    artifact: Mapping[str, Any],
    teacher: DepthAnythingOnnxTeacher,
    generator_implementation_sha256: str,
    config: DenseTeacherConfig,
) -> dict[str, Any]:
    if route_episode.get("episode_id") != request.get("episode_id"):
        raise DenseTeacherError("route episode identity mismatch")
    validate_loso_artifact(
        artifact,
        held_out_session_id=str(request.get("fold_held_out_session_id")),
        training_manifest_sha256=str(request.get("fold_training_input_manifest_sha256")),
        model_sha256=teacher.model_sha256,
    )
    frames = request.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DenseTeacherError("request frame inventory is empty")
    output_frames: list[dict[str, Any]] = []
    provider = route_episode["provider"]
    route_intent_id = f"{route_episode['episode_id']}:{provider['provider_id']}"
    for request_frame in frames:
        if not isinstance(request_frame, dict):
            raise DenseTeacherError("request frame must be an object")
        frame_ms = request_frame.get("video_pts_ms")
        if not isinstance(frame_ms, int) or isinstance(frame_ms, bool) or frame_ms < 0:
            raise DenseTeacherError("request video_pts_ms is invalid")
        sample_index, sample = select_route_sample(route_episode, frame_ms, config)
        rgb, _decode_ms = decode_video_frame_rgb(video_path, frame_ms)
        rgb_sha = sha256_bytes(rgb.tobytes(order="C"))
        depth, _inference_ms = teacher.infer_rgb(rgb)
        field_payload, summary = build_frame_field(
            depth, artifact=artifact, route_sample=sample, config=config
        )
        valid_until = min(int(sample["valid_until_timestamp_ms"]), frame_ms + config.field_ttl_ms)
        output_frames.append({
            "frame_id": request_frame["frame_id"],
            "frame_timestamp_ms": frame_ms,
            "observed_at_ms": frame_ms,
            "valid_until_ms": valid_until,
            "source_frame_payload_sha256": request_frame["frame_payload_sha256"],
            "teacher_decoded_rgb_sha256": rgb_sha,
            "evidence_status": "AVAILABLE",
            "selected_route_sample_index": sample_index,
            "selected_route_sample_timestamp_ms": sample["timestamp_ms"],
            "selected_route_valid_until_ms": sample["valid_until_timestamp_ms"],
            "selected_route_confidence": sample["confidence"],
            "selected_route_waypoints": sample["horizon_waypoints"],
            "route_intent_id": route_intent_id,
            "event_key": f"{request['episode_id']}:{route_intent_id}",
            **summary,
            "field_payload": field_payload,
        })
    return {
        "schema": FIELD_SCHEMA,
        "dense_field_contract_id": FIELD_CONTRACT_ID,
        "episode_id": request["episode_id"],
        "input_video_sha256": request["input_video_sha256"],
        "source_capture_frame_ledger_sha256": request["source_capture_frame_ledger_sha256"],
        "route_input_sha256": request["adapter_route_input_sha256"],
        "fold_artifact_sha256": request["fold_artifact_sha256"],
        "teacher_model_name": MODEL_NAME,
        "teacher_model_version": MODEL_VERSION,
        "teacher_model_license": MODEL_LICENSE,
        "teacher_model_sha256": teacher.model_sha256,
        "teacher_model_input_contract": MODEL_INPUT_CONTRACT,
        "teacher_model_output_contract": MODEL_OUTPUT_CONTRACT,
        "teacher_decode_policy": DECODE_POLICY,
        "teacher_inference_runtime": "onnxruntime_cpu_v1",
        "generator_implementation_sha256": generator_implementation_sha256,
        "teacher_output_role": "auxiliary_only_not_human_truth",
        "blind_accessed": False,
        "future_inputs_used": False,
        "human_event_truth_used": False,
        "production_authorized": False,
        "frame_count": len(output_frames),
        "frames": output_frames,
    }
