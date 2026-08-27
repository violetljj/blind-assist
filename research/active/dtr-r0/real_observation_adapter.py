"""Truth-blind offline RGB observation materializer for DTR-R0.

The CLI consumes an explicitly addressed, frozen 24-event manifest and emits
one ledger row per episode plus an input coverage/track-continuity report.  It
does not read event truth, run a DTR arm, compute outcome metrics, or decide a
scientific advancement gate.

Optional ``cv2`` and ``ultralytics`` imports are deliberately local to the
real-video execution path.  Importing this module and running its unit tests
requires only the Python standard library.

Manifest shape (``dtr-r0-real-observation-manifest-v1``)::

    {
      "schema_version": "dtr-r0-real-observation-manifest-v1",
      "frozen": true,
      "episodes": [{
        "episode_id": "crossing_enters_route-00",
        "scene_type": "crossing_enters_route",
        "video_path": "inputs/crossing-00.mp4",
        "pose_jsonl_path": "inputs/crossing-00.pose.jsonl",
        "time_offset_s": 0.0,
        "camera": {
          "image_width_px": 1920, "image_height_px": 1080,
          "fx_px": 1200.0, "fy_px": 1200.0,
          "cx_px": 960.0, "cy_px": 540.0,
          "camera_height_m": 1.55, "pitch_down_rad": 0.10,
          "person_radius_m": 0.30
        }
      }]
    }

Relative input paths resolve only against the manifest directory.  Output is
accepted only outside the repository or under its ignored ``artifacts.local``
surface, and the caller must provide it with ``--output-dir``.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


MANIFEST_SCHEMA = "dtr-r0-real-observation-manifest-v1"
LEDGER_SCHEMA = "dtr-r0-real-observation-ledger-v1"
REPORT_SCHEMA = "dtr-r0-real-observation-input-report-v1"
REPORT_ROLE = "INPUT_MATERIALIZATION_COVERAGE_CONTINUITY_ONLY"
EXPECTED_SCENE_COUNTS = {
    "crossing_enters_route": 4,
    "oncoming": 4,
    "parallel_outside_route": 4,
    "static_roadside": 4,
    "ego_turn_pseudo_motion": 4,
    "enter_then_exit": 4,
}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "app" / "src" / "main" / "assets" / "yolo11n_fp16_320.tflite"
)
_FORBIDDEN_MANIFEST_KEYS = {
    "truth",
    "event_truth",
    "ground_truth",
    "annotation",
    "annotations",
    "annotation_path",
    "labels",
    "label_path",
    "outcome",
    "scientific_gate",
}
_EPSILON = 1e-9


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("bbox coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")


@dataclass(frozen=True)
class TrackedDetection:
    track_id: str
    detection: Detection
    is_new_track: bool


@dataclass(frozen=True)
class CameraCalibration:
    image_width_px: int
    image_height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    camera_height_m: float
    pitch_down_rad: float
    person_radius_m: float = 0.30

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("camera image dimensions must be positive")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.camera_height_m <= 0.0 or self.person_radius_m < 0.0:
            raise ValueError("camera height must be positive and radius non-negative")
        if not -math.pi / 2.0 < self.pitch_down_rad < math.pi / 2.0:
            raise ValueError("camera pitch must be inside (-pi/2, pi/2)")
        if not all(
            math.isfinite(value)
            for value in (
                self.fx_px,
                self.fy_px,
                self.cx_px,
                self.cy_px,
                self.camera_height_m,
                self.pitch_down_rad,
                self.person_radius_m,
            )
        ):
            raise ValueError("camera calibration values must be finite")


@dataclass(frozen=True)
class GroundProjection:
    forward_m: float
    left_m: float
    radius_m: float


def project_bbox_bottom_center(
    bbox: BBox, calibration: CameraCalibration
) -> Optional[GroundProjection]:
    """Project the bbox footpoint through a flat-ground pinhole model.

    Camera ray axes are right/down/forward. ``pitch_down_rad`` is positive when
    the optical axis points toward the ground.  DTR local coordinates are
    forward/left, hence the sign inversion from camera-right to local-left.
    A ray parallel to or above the ground returns ``None`` rather than a metric
    observation.
    """

    pixel_x, pixel_y = bbox.bottom_center
    ray_right = (pixel_x - calibration.cx_px) / calibration.fx_px
    ray_down = (pixel_y - calibration.cy_px) / calibration.fy_px
    pitch_cosine = math.cos(calibration.pitch_down_rad)
    pitch_sine = math.sin(calibration.pitch_down_rad)
    ground_down = pitch_cosine * ray_down + pitch_sine
    ground_forward = -pitch_sine * ray_down + pitch_cosine
    if ground_down <= _EPSILON or ground_forward <= _EPSILON:
        return None
    scale = calibration.camera_height_m / ground_down
    return GroundProjection(
        forward_m=scale * ground_forward,
        left_m=-scale * ray_right,
        radius_m=calibration.person_radius_m,
    )


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.width * (left.y2 - left.y1) + right.width * (right.y2 - right.y1) - intersection
    return intersection / union if union > 0.0 else 0.0


def footpoint_distance(left: BBox, right: BBox) -> float:
    left_x, left_y = left.bottom_center
    right_x, right_y = right.bottom_center
    return math.hypot(left_x - right_x, left_y - right_y)


@dataclass
class _TrackState:
    track_id: str
    bbox: BBox
    last_time_s: float


class CausalPersonTracker:
    """Greedy one-to-one tracker using only prior bbox/footpoint state."""

    def __init__(
        self,
        *,
        minimum_iou: float = 0.10,
        maximum_footpoint_distance_px: float = 80.0,
        maximum_track_age_s: float = 0.75,
    ) -> None:
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be in [0, 1]")
        self.minimum_iou = minimum_iou
        self.maximum_footpoint_distance_px = _positive_number(
            maximum_footpoint_distance_px, "maximum_footpoint_distance_px"
        )
        self.maximum_track_age_s = _positive_number(
            maximum_track_age_s, "maximum_track_age_s"
        )
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1
        self._last_time_s: Optional[float] = None

    def update(
        self, detections: Sequence[Detection], *, time_s: float
    ) -> list[TrackedDetection]:
        time_s = _finite_number(time_s, "tracker time_s")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("tracker timestamps must be strictly increasing")
        self._last_time_s = time_s
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if time_s - state.last_time_s <= self.maximum_track_age_s + _EPSILON
        }

        candidates: list[tuple[float, float, str, int]] = []
        for track_id, state in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = bbox_iou(state.bbox, detection.bbox)
                distance = footpoint_distance(state.bbox, detection.bbox)
                if overlap >= self.minimum_iou or distance <= self.maximum_footpoint_distance_px:
                    candidates.append((-overlap, distance, track_id, detection_index))
        candidates.sort()

        detection_to_track: dict[int, str] = {}
        used_tracks: set[str] = set()
        for _negative_iou, _distance, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in detection_to_track:
                continue
            used_tracks.add(track_id)
            detection_to_track[detection_index] = track_id

        tracked: list[TrackedDetection] = []
        for detection_index, detection in enumerate(detections):
            track_id = detection_to_track.get(detection_index)
            is_new = track_id is None
            if track_id is None:
                track_id = f"person-{self._next_track_number:06d}"
                self._next_track_number += 1
            self._tracks[track_id] = _TrackState(track_id, detection.bbox, time_s)
            tracked.append(TrackedDetection(track_id, detection, is_new))
        return tracked


@dataclass(frozen=True)
class PoseSample:
    time_s: float
    tracking_state: str
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    body_yaw_rad: Optional[float] = None
    sensor_yaw_rad: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("pose time_s must be finite")
        if not self.tracking_state:
            raise ValueError("pose tracking_state must be non-empty")
        if self.tracking_state == "TRACKING":
            values = (self.x_m, self.y_m, self.body_yaw_rad, self.sensor_yaw_rad)
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(
                    "TRACKING pose requires separate finite position/body/sensor yaw"
                )

    def to_ledger_dict(self, frame_time_s: float) -> dict[str, float | str]:
        if self.tracking_state != "TRACKING":
            raise ValueError("only TRACKING poses can enter the ledger")
        assert self.x_m is not None and self.y_m is not None
        assert self.body_yaw_rad is not None and self.sensor_yaw_rad is not None
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "body_yaw_rad": self.body_yaw_rad,
            "sensor_yaw_rad": self.sensor_yaw_rad,
            "pose_time_s": self.time_s,
            "pose_age_s": frame_time_s - self.time_s,
            "tracking_state": "TRACKING",
        }


@dataclass(frozen=True)
class PoseResolution:
    sample: Optional[PoseSample]
    input_health: str
    age_s: Optional[float]


class CausalPoseLookup:
    """Return the latest sample at/before the frame only when it is TRACKING.

    A newer LIMITED/LOST sample invalidates older tracking. Falling back across
    it would manufacture pose availability that the source explicitly revoked.
    """

    def __init__(self, samples: Sequence[PoseSample], *, maximum_age_s: float) -> None:
        self.maximum_age_s = _positive_number(maximum_age_s, "maximum_age_s")
        previous_time: Optional[float] = None
        for sample in samples:
            if previous_time is not None and sample.time_s <= previous_time:
                raise ValueError("pose samples must be strictly time-ordered")
            previous_time = sample.time_s
        self._samples = list(samples)
        self._times = [sample.time_s for sample in self._samples]

    def resolve(self, frame_time_s: float) -> PoseResolution:
        frame_time_s = _finite_number(frame_time_s, "frame_time_s")
        index = bisect_right(self._times, frame_time_s) - 1
        if index < 0:
            return PoseResolution(None, "NO_CAUSAL_POSE", None)
        sample = self._samples[index]
        age_s = frame_time_s - sample.time_s
        if sample.tracking_state != "TRACKING":
            return PoseResolution(
                None, f"LATEST_POSE_{sample.tracking_state}", age_s
            )
        if age_s > self.maximum_age_s + _EPSILON:
            return PoseResolution(None, "STALE_TRACKING_POSE", age_s)
        return PoseResolution(sample, "TRACKING", age_s)


def load_pose_jsonl(path: Path) -> list[PoseSample]:
    samples: list[PoseSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"pose line {line_number} must be an object")
            state = str(value.get("tracking_state", "")).strip().upper()
            samples.append(
                PoseSample(
                    time_s=_finite_number(value.get("time_s"), f"pose line {line_number} time_s"),
                    tracking_state=state,
                    x_m=(
                        _finite_number(value.get("x_m"), f"pose line {line_number} x_m")
                        if state == "TRACKING"
                        else None
                    ),
                    y_m=(
                        _finite_number(value.get("y_m"), f"pose line {line_number} y_m")
                        if state == "TRACKING"
                        else None
                    ),
                    body_yaw_rad=(
                        _finite_number(
                            value.get("body_yaw_rad"),
                            f"pose line {line_number} body_yaw_rad",
                        )
                        if state == "TRACKING"
                        else None
                    ),
                    sensor_yaw_rad=(
                        _finite_number(
                            value.get("sensor_yaw_rad"),
                            f"pose line {line_number} sensor_yaw_rad",
                        )
                        if state == "TRACKING"
                        else None
                    ),
                )
            )
    return samples


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    scene_type: str
    video_path: Path
    pose_jsonl_path: Path
    time_offset_s: float
    camera: CameraCalibration


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    episodes: tuple[EpisodeSpec, ...]


def _reject_truth_fields(value: Any, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in _FORBIDDEN_MANIFEST_KEYS
                or "truth" in normalized
                or "annotation" in normalized
            ):
                raise ValueError(f"truth/outcome field is forbidden at {location}.{key}")
            _reject_truth_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_truth_fields(child, f"{location}[{index}]")


def _resolve_manifest_input(
    raw_path: Any,
    *,
    manifest_directory: Path,
    label: str,
    require_files: bool,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} must be an explicit non-empty path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = manifest_directory / candidate
    try:
        return candidate.resolve(strict=require_files)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {candidate}") from error


def _camera_from_manifest(value: Any, label: str) -> CameraCalibration:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return CameraCalibration(
        image_width_px=int(value["image_width_px"]),
        image_height_px=int(value["image_height_px"]),
        fx_px=_positive_number(value["fx_px"], f"{label}.fx_px"),
        fy_px=_positive_number(value["fy_px"], f"{label}.fy_px"),
        cx_px=_finite_number(value["cx_px"], f"{label}.cx_px"),
        cy_px=_finite_number(value["cy_px"], f"{label}.cy_px"),
        camera_height_m=_positive_number(
            value["camera_height_m"], f"{label}.camera_height_m"
        ),
        pitch_down_rad=_finite_number(
            value["pitch_down_rad"], f"{label}.pitch_down_rad"
        ),
        person_radius_m=_finite_number(
            value.get("person_radius_m", 0.30), f"{label}.person_radius_m"
        ),
    )


def validate_frozen_manifest(
    value: Mapping[str, Any],
    *,
    manifest_path: Path,
    require_files: bool = True,
) -> FrozenManifest:
    _reject_truth_fields(value)
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    if value.get("frozen") is not True:
        raise ValueError("manifest must declare frozen=true")
    episode_values = value.get("episodes")
    if not isinstance(episode_values, list) or len(episode_values) != 24:
        raise ValueError("frozen manifest must contain exactly 24 episodes")

    resolved_manifest_path = manifest_path.resolve(strict=require_files)
    manifest_directory = resolved_manifest_path.parent
    episode_ids: set[str] = set()
    scene_counts: Counter[str] = Counter()
    episodes: list[EpisodeSpec] = []
    for index, episode_value in enumerate(episode_values):
        label = f"episodes[{index}]"
        if not isinstance(episode_value, Mapping):
            raise ValueError(f"{label} must be an object")
        episode_id = str(episode_value.get("episode_id", "")).strip()
        if not episode_id or episode_id in episode_ids:
            raise ValueError("episode ids must be non-empty and unique")
        episode_ids.add(episode_id)
        scene_type = str(episode_value.get("scene_type", ""))
        if scene_type not in EXPECTED_SCENE_COUNTS:
            raise ValueError(f"unsupported scene class: {scene_type}")
        scene_counts[scene_type] += 1
        episodes.append(
            EpisodeSpec(
                episode_id=episode_id,
                scene_type=scene_type,
                video_path=_resolve_manifest_input(
                    episode_value.get("video_path"),
                    manifest_directory=manifest_directory,
                    label=f"{label}.video_path",
                    require_files=require_files,
                ),
                pose_jsonl_path=_resolve_manifest_input(
                    episode_value.get("pose_jsonl_path"),
                    manifest_directory=manifest_directory,
                    label=f"{label}.pose_jsonl_path",
                    require_files=require_files,
                ),
                time_offset_s=_finite_number(
                    episode_value.get("time_offset_s", 0.0),
                    f"{label}.time_offset_s",
                ),
                camera=_camera_from_manifest(episode_value.get("camera"), f"{label}.camera"),
            )
        )
    if dict(scene_counts) != EXPECTED_SCENE_COUNTS:
        raise ValueError(
            f"manifest must contain exactly four episodes in each scene: {dict(scene_counts)}"
        )
    return FrozenManifest(resolved_manifest_path, tuple(episodes))


def load_frozen_manifest(path: Path) -> FrozenManifest:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("manifest root must be an object")
    return validate_frozen_manifest(value, manifest_path=resolved, require_files=True)


@dataclass(frozen=True)
class SampledVideoFrame:
    frame_index: int
    video_time_s: float
    image: Any
    source_fps: float


def iter_sampled_video_frames(
    video_path: Path, *, sample_hz: float
) -> Iterable[SampledVideoFrame]:
    """Decode sequentially and select the first frame at/after each sample tick."""

    sample_hz = _positive_number(sample_hz, "sample_hz")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("real video materialization requires optional package cv2") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS: {video_path}")
    sample_period_s = 1.0 / sample_hz
    next_sample_s = 0.0
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            video_time_s = frame_index / source_fps
            if video_time_s + _EPSILON >= next_sample_s:
                yield SampledVideoFrame(frame_index, video_time_s, image, source_fps)
                while next_sample_s <= video_time_s + _EPSILON:
                    next_sample_s += sample_period_s
            frame_index += 1
    finally:
        capture.release()


class UltralyticsPersonDetector:
    """Local Ultralytics model wrapper; the package is imported on first use."""

    def __init__(self, model_path: Path, *, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path.resolve(strict=True)
        self.confidence_threshold = _finite_number(
            confidence_threshold, "confidence_threshold"
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "real RGB materialization requires optional package ultralytics"
            ) from error
        self._model = YOLO(str(self.model_path), task="detect")

    def detect(self, image: Any) -> list[Detection]:
        self._load()
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            classes=[0],
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            names = getattr(result, "names", {})
            for coordinates_value, confidence, class_id in zip(
                coordinates, confidences, classes
            ):
                class_number = int(class_id)
                class_name = names.get(class_number, "") if isinstance(names, Mapping) else ""
                if class_number != 0 and class_name != "person":
                    continue
                detections.append(
                    Detection(
                        BBox(*(float(item) for item in coordinates_value[:4])),
                        float(confidence),
                    )
                )
        detections.sort(
            key=lambda item: (
                item.bbox.x1,
                item.bbox.y1,
                item.bbox.x2,
                item.bbox.y2,
                -item.confidence,
            )
        )
        return detections


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def materialize_episode(
    spec: EpisodeSpec,
    *,
    detector: UltralyticsPersonDetector,
    model_path: Path,
    model_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    sample_hz: float,
    maximum_pose_age_s: float,
    tracker_minimum_iou: float,
    tracker_footpoint_distance_px: float,
    tracker_maximum_age_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pose_lookup = CausalPoseLookup(
        load_pose_jsonl(spec.pose_jsonl_path), maximum_age_s=maximum_pose_age_s
    )
    tracker = CausalPersonTracker(
        minimum_iou=tracker_minimum_iou,
        maximum_footpoint_distance_px=tracker_footpoint_distance_px,
        maximum_track_age_s=tracker_maximum_age_s,
    )
    frames: list[dict[str, Any]] = []
    person_detection_count = 0
    projected_observation_count = 0
    frames_with_pose = 0
    frames_with_metric_observation = 0
    new_track_count = 0
    continued_track_assignment_count = 0
    calibration_mismatch_frames = 0

    for sampled in iter_sampled_video_frames(spec.video_path, sample_hz=sample_hz):
        frame_time_s = sampled.video_time_s + spec.time_offset_s
        detections = detector.detect(sampled.image)
        tracked = tracker.update(detections, time_s=frame_time_s)
        pose_resolution = pose_lookup.resolve(frame_time_s)
        if pose_resolution.sample is not None:
            frames_with_pose += 1
            ego_pose: Optional[dict[str, float | str]] = (
                pose_resolution.sample.to_ledger_dict(frame_time_s)
            )
        else:
            ego_pose = None

        image_height_px = int(sampled.image.shape[0])
        image_width_px = int(sampled.image.shape[1])
        calibration_matches = (
            image_width_px == spec.camera.image_width_px
            and image_height_px == spec.camera.image_height_px
        )
        if not calibration_matches:
            calibration_mismatch_frames += 1

        observations: list[dict[str, Any]] = []
        projection_rejected_count = 0
        for tracked_detection in tracked:
            if tracked_detection.is_new_track:
                new_track_count += 1
            else:
                continued_track_assignment_count += 1
            projection = (
                project_bbox_bottom_center(tracked_detection.detection.bbox, spec.camera)
                if calibration_matches
                else None
            )
            if projection is None:
                projection_rejected_count += 1
                continue
            projected_observation_count += 1
            observations.append(
                {
                    "track_id": tracked_detection.track_id,
                    "forward_m": projection.forward_m,
                    "left_m": projection.left_m,
                    "radius_m": projection.radius_m,
                    "source_bbox_xyxy": tracked_detection.detection.bbox.to_list(),
                    "person_confidence": tracked_detection.detection.confidence,
                }
            )
        if observations:
            frames_with_metric_observation += 1
        person_detection_count += len(detections)
        frames.append(
            {
                "frame_index": sampled.frame_index,
                "video_time_s": sampled.video_time_s,
                "time_s": frame_time_s,
                "person_detection_count": len(detections),
                "ego_pose": ego_pose,
                "observations": observations,
                "input_health": {
                    "video_decode": "OK",
                    "person_detector": "OK",
                    "pose": pose_resolution.input_health,
                    "camera_calibration": (
                        "OK" if calibration_matches else "IMAGE_SIZE_MISMATCH"
                    ),
                    "projection_rejected_count": projection_rejected_count,
                },
            }
        )

    sampled_frame_count = len(frames)
    if sampled_frame_count == 0:
        raise RuntimeError(f"video produced no sampled frames: {spec.video_path}")
    tracked_detection_count = new_track_count + continued_track_assignment_count
    summary = {
        "episode_id": spec.episode_id,
        "scene_type": spec.scene_type,
        "sampled_frame_count": sampled_frame_count,
        "person_detection_count": person_detection_count,
        "projected_observation_count": projected_observation_count,
        "projection_success_fraction": _fraction(
            projected_observation_count, person_detection_count
        ),
        "frames_with_tracking_pose": frames_with_pose,
        "tracking_pose_frame_coverage": _fraction(frames_with_pose, sampled_frame_count),
        "frames_with_metric_observation": frames_with_metric_observation,
        "metric_observation_frame_coverage": _fraction(
            frames_with_metric_observation, sampled_frame_count
        ),
        "tracked_detection_count": tracked_detection_count,
        "new_track_count": new_track_count,
        "continued_track_assignment_count": continued_track_assignment_count,
        "continued_track_assignment_fraction": _fraction(
            continued_track_assignment_count, tracked_detection_count
        ),
        "calibration_mismatch_frames": calibration_mismatch_frames,
    }
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "episode_id": spec.episode_id,
        "scene_type": spec.scene_type,
        "sample_hz": sample_hz,
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "video_path": str(spec.video_path),
            "video_sha256": sha256_file(spec.video_path),
            "pose_jsonl_path": str(spec.pose_jsonl_path),
            "pose_jsonl_sha256": sha256_file(spec.pose_jsonl_path),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
        },
        "frames": frames,
        "input_health": summary,
        "authority": REPORT_ROLE,
    }
    return ledger, summary


def _ensure_untracked_output_directory(output_directory: Path) -> Path:
    resolved = output_directory.resolve(strict=False)
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0].lower() != "artifacts.local":
        raise ValueError(
            "output directory inside the repository must be under artifacts.local"
        )
    return resolved


def _atomic_write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def build_input_report(
    manifest: FrozenManifest,
    *,
    manifest_sha256: str,
    model_path: Path,
    model_sha256: str,
    sample_hz: float,
    episode_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sampled_frames = sum(int(item["sampled_frame_count"]) for item in episode_summaries)
    pose_frames = sum(int(item["frames_with_tracking_pose"]) for item in episode_summaries)
    metric_frames = sum(
        int(item["frames_with_metric_observation"]) for item in episode_summaries
    )
    tracked_detections = sum(
        int(item["tracked_detection_count"]) for item in episode_summaries
    )
    continued = sum(
        int(item["continued_track_assignment_count"]) for item in episode_summaries
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "report_role": REPORT_ROLE,
        "result_status": "NO_SCIENTIFIC_RESULT",
        "scientific_gate_evaluated": False,
        "truth_accessed": False,
        "forbidden_uses": ["scientific_result", "advancement_decision"],
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest_sha256,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "sample_hz": sample_hz,
        "episode_count": len(episode_summaries),
        "scene_counts": dict(EXPECTED_SCENE_COUNTS),
        "coverage": {
            "sampled_frame_count": sampled_frames,
            "frames_with_tracking_pose": pose_frames,
            "tracking_pose_frame_coverage": _fraction(pose_frames, sampled_frames),
            "frames_with_metric_observation": metric_frames,
            "metric_observation_frame_coverage": _fraction(metric_frames, sampled_frames),
        },
        "continuity": {
            "tracked_detection_count": tracked_detections,
            "new_track_count": sum(
                int(item["new_track_count"]) for item in episode_summaries
            ),
            "continued_track_assignment_count": continued,
            "continued_track_assignment_fraction": _fraction(
                continued, tracked_detections
            ),
        },
        "episodes": list(episode_summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-hz", type=float, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--maximum-pose-age-s", type=float, default=0.25)
    parser.add_argument("--tracker-minimum-iou", type=float, default=0.10)
    parser.add_argument("--tracker-footpoint-distance-px", type=float, default=80.0)
    parser.add_argument("--tracker-maximum-age-s", type=float, default=0.75)
    args = parser.parse_args()

    manifest = load_frozen_manifest(args.manifest)
    model_path = args.model.resolve(strict=True)
    output_directory = _ensure_untracked_output_directory(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    ledger_path = output_directory / "real-observation-ledger.jsonl"
    report_path = output_directory / "real-observation-input-report.json"
    if ledger_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing materialization output")

    sample_hz = _positive_number(args.sample_hz, "sample_hz")
    manifest_sha256 = sha256_file(manifest.path)
    model_sha256 = sha256_file(model_path)
    detector = UltralyticsPersonDetector(
        model_path, confidence_threshold=args.confidence_threshold
    )
    ledgers: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in manifest.episodes:
        ledger, summary = materialize_episode(
            spec,
            detector=detector,
            model_path=model_path,
            model_sha256=model_sha256,
            manifest_path=manifest.path,
            manifest_sha256=manifest_sha256,
            sample_hz=sample_hz,
            maximum_pose_age_s=args.maximum_pose_age_s,
            tracker_minimum_iou=args.tracker_minimum_iou,
            tracker_footpoint_distance_px=args.tracker_footpoint_distance_px,
            tracker_maximum_age_s=args.tracker_maximum_age_s,
        )
        ledgers.append(ledger)
        summaries.append(summary)

    report = build_input_report(
        manifest,
        manifest_sha256=manifest_sha256,
        model_path=model_path,
        model_sha256=model_sha256,
        sample_hz=sample_hz,
        episode_summaries=summaries,
    )
    _atomic_write_jsonl(ledger_path, ledgers)
    _atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "ledger_path": str(ledger_path),
                "report_path": str(report_path),
                "episode_count": len(ledgers),
                "report_role": REPORT_ROLE,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
