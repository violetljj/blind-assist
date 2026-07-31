from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


REFERENCE_BRANCH = "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY"
DETECTION_DUMP_SCHEMA = "blindassist.dual_loop_all_detection_dump.v1"
Q0_R01_SCHEMA = "blindassist.dual_loop_semantic_refresh_q0_r0_1_result.v1"
PROPAGATION_MODE = "ZERO_ORDER_HOLD_SEMANTIC_PROPAGATION_R0"
NANOS_PER_MILLISECOND = 1_000_000
NANOS_PER_SECOND = 1_000_000_000
EPISODE_FEEDBACK_MATCH_TOLERANCE_MS = 100.0
EPISODE_FEEDBACK_MATCH_TOLERANCE_NS = int(
    EPISODE_FEEDBACK_MATCH_TOLERANCE_MS * NANOS_PER_MILLISECOND
)
EPISODE_ADMISSION_CONSTRAINTS = {
    "reference_event_missed_count_max": 0,
    "reference_episode_match_recall_min": 0.95,
    "episode_temporal_iou_mean_min": 0.80,
    "episode_abs_onset_delay_p95_ms_max": 150.0,
    "risk_signature_match_rate_min": 0.95,
}


class Q0InputError(ValueError):
    """Raised when a Q0 input violates the causal or identity contract."""


class MissingFastFeatureStream(Q0InputError):
    """Raised when a feature-dependent arm has no independent fast-loop trace."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Q0InputError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Q0InputError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_text_exclusive(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class TruthItem:
    item_id: str
    item_kind: str
    session_id: str
    start_rel_ns: int
    end_rel_ns: int
    scoring_status: str


@dataclass(frozen=True)
class Frame:
    session_id: str
    frame_id: str
    timestamp_ns: int
    detector_output_sha256: str
    detections: tuple[dict[str, Any], ...]
    stable_risk: dict[str, Any]
    reference_feedback_triggered: bool
    reference_risk_event: dict[str, Any] | None

    @property
    def key(self) -> tuple[str, str]:
        return self.session_id, self.frame_id


@dataclass
class FastFeatureRow:
    session_id: str
    frame_id: str
    timestamp_ns: int
    features: dict[str, float | bool | None]

    @property
    def key(self) -> tuple[str, str]:
        return self.session_id, self.frame_id


@dataclass
class ArmState:
    session_id: str | None = None
    last_refresh_timestamp_ns: int | None = None
    last_refresh_frame_id: str | None = None
    cached_risk: dict[str, Any] | None = None
    refresh_count: int = 0
    active_event_id: str | None = None
    next_event_sequence_number: int = 1

    def reset_for_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.last_refresh_timestamp_ns = None
        self.last_refresh_frame_id = None
        self.cached_risk = None
        self.refresh_count = 0
        self.active_event_id = None
        self.next_event_sequence_number = 1


class RefreshPolicy(Protocol):
    name: str
    requires_fast_features: bool

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class FullRatePolicy:
    name: str = "FULL_RATE_REFERENCE"
    requires_fast_features: bool = False

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]:
        return True, "FULL_RATE"


@dataclass(frozen=True)
class FixedIntervalPolicy:
    interval_ns: int
    name: str
    requires_fast_features: bool = False

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]:
        if state.last_refresh_timestamp_ns is None:
            return True, "SESSION_START"
        elapsed = frame.timestamp_ns - state.last_refresh_timestamp_ns
        if elapsed >= self.interval_ns:
            return True, "MAX_INTERVAL_ELAPSED"
        return False, "CACHE_WITHIN_INTERVAL"


@dataclass(frozen=True)
class MaxAgePolicy:
    max_age_ns: int
    name: str
    requires_fast_features: bool = True

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]:
        if self.requires_fast_features and features is None:
            raise MissingFastFeatureStream(
                f"{self.name}: no independent fast-feature row for {frame.key}"
            )
        if state.last_refresh_timestamp_ns is None:
            return True, "SESSION_START"
        if frame.timestamp_ns - state.last_refresh_timestamp_ns >= self.max_age_ns:
            return True, "MAX_CACHE_AGE"
        if features is not None and bool(features.features.get("track_failure", False)):
            return True, "TRACK_FAILURE"
        return False, "CACHE_WITHIN_MAX_AGE"


@dataclass(frozen=True)
class FeatureRulePolicy:
    max_age_ns: int
    motion_threshold: float = math.inf
    blur_threshold: float = math.inf
    name: str = "FEATURE_RULE_COMBINATION"
    requires_fast_features: bool = True

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]:
        if features is None:
            raise MissingFastFeatureStream(
                f"{self.name}: no independent fast-feature row for {frame.key}"
            )
        if state.last_refresh_timestamp_ns is None:
            return True, "SESSION_START"
        if frame.timestamp_ns - state.last_refresh_timestamp_ns >= self.max_age_ns:
            return True, "MAX_CACHE_AGE"
        values = features.features
        if bool(values.get("track_failure", False)):
            return True, "TRACK_FAILURE"
        if bool(values.get("association_ambiguous", False)):
            return True, "ASSOCIATION_AMBIGUOUS"
        motion = values.get("image_motion")
        if motion is not None and float(motion) >= self.motion_threshold:
            return True, "IMAGE_MOTION"
        blur = values.get("blur")
        if blur is not None and float(blur) >= self.blur_threshold:
            return True, "BLUR"
        return False, "FEATURES_WITHIN_POLICY"


@dataclass(frozen=True)
class LogisticPolicy:
    intercept: float
    weights: dict[str, float]
    threshold: float
    max_age_ns: int
    name: str = "LOGISTIC_REFRESH_POLICY"
    requires_fast_features: bool = True

    def should_refresh(
        self,
        frame: Frame,
        state: ArmState,
        features: FastFeatureRow | None,
    ) -> tuple[bool, str]:
        if features is None:
            raise MissingFastFeatureStream(
                f"{self.name}: no independent fast-feature row for {frame.key}"
            )
        if state.last_refresh_timestamp_ns is None:
            return True, "SESSION_START"
        if frame.timestamp_ns - state.last_refresh_timestamp_ns >= self.max_age_ns:
            return True, "MAX_CACHE_AGE"
        linear = self.intercept
        for name, weight in self.weights.items():
            value = features.features.get(name)
            if value is not None:
                linear += weight * float(value)
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, linear))))
        if probability >= self.threshold:
            return True, "LOGISTIC_REFRESH_PROBABILITY"
        return False, "LOGISTIC_BELOW_THRESHOLD"


def fixed_time_policies(intervals_ms: tuple[int, ...]) -> list[RefreshPolicy]:
    return [
        FixedIntervalPolicy(
            interval_ns=interval_ms * NANOS_PER_MILLISECOND,
            name=f"FIXED_TIME_{interval_ms}MS",
        )
        for interval_ms in intervals_ms
    ]


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Q0InputError(f"{label}: expected {expected!r}, got {actual!r}")


def load_frames(
    dump_path: Path,
    baseline_path: Path,
    expected_dump_receipt_path: Path | None = None,
    expected_baseline_sha256: str | None = None,
) -> list[Frame]:
    dump_rows = read_jsonl(dump_path)
    baseline_rows = [
        row
        for row in read_jsonl(baseline_path)
        if row.get("branch_id") == REFERENCE_BRANCH
    ]
    if expected_dump_receipt_path is not None:
        receipt = read_json(expected_dump_receipt_path)
        _require_equal(receipt.get("status"), "COMPLETE", "detection dump receipt status")
        _require_equal(receipt.get("truth_read"), False, "detection dump truth access")
        _require_equal(
            receipt.get("trace_sha256"),
            sha256_file(dump_path),
            "detection dump receipt trace hash",
        )
        _require_equal(receipt.get("frame_count"), len(dump_rows), "detection dump frame count")
    if expected_baseline_sha256 is not None:
        _require_equal(
            sha256_file(baseline_path), expected_baseline_sha256, "baseline trace hash"
        )
    if len(dump_rows) == 0 or len(dump_rows) != len(baseline_rows):
        raise Q0InputError("detection dump and reference trace frame counts differ")

    dump_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    baseline_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in dump_rows:
        if row.get("schema_version") != DETECTION_DUMP_SCHEMA:
            raise Q0InputError("detection dump schema mismatch")
        key = (str(row["session_id"]), str(row["frame_id"]))
        if key in dump_by_key:
            raise Q0InputError(f"duplicate detection frame {key}")
        dump_by_key[key] = row
    for row in baseline_rows:
        key = (str(row["session_id"]), str(row["frame_id"]))
        if key in baseline_by_key:
            raise Q0InputError(f"duplicate reference frame {key}")
        baseline_by_key[key] = row
    if set(dump_by_key) != set(baseline_by_key):
        raise Q0InputError("detection/reference frame identity mismatch")

    frames: list[Frame] = []
    previous_by_session: dict[str, int] = {}
    for dump_row in dump_rows:
        key = (str(dump_row["session_id"]), str(dump_row["frame_id"]))
        reference = baseline_by_key[key]
        timestamp_ns = int(dump_row["source_capture_timestamp_ns"])
        previous = previous_by_session.get(key[0])
        if previous is not None and timestamp_ns <= previous:
            raise Q0InputError(f"non-increasing capture time at {key}")
        previous_by_session[key[0]] = timestamp_ns
        if dump_row.get("detector_output_sha256") != reference.get(
            "detector_output_sha256"
        ):
            raise Q0InputError(f"detector output hash mismatch at {key}")
        stable_risk = reference.get("stable_risk")
        if not isinstance(stable_risk, dict):
            raise Q0InputError(f"missing stable risk at {key}")
        detections = dump_row.get("detections")
        if not isinstance(detections, list):
            raise Q0InputError(f"missing detection list at {key}")
        frames.append(
            Frame(
                session_id=key[0],
                frame_id=key[1],
                timestamp_ns=timestamp_ns,
                detector_output_sha256=str(dump_row["detector_output_sha256"]),
                detections=tuple(copy.deepcopy(detections)),
                stable_risk=copy.deepcopy(stable_risk),
                reference_feedback_triggered=bool(reference.get("feedback_triggered", False)),
                reference_risk_event=copy.deepcopy(reference.get("risk_event")),
            )
        )
    return frames


def load_truth_items(path: Path) -> list[TruthItem]:
    payload = read_json(path)
    table = payload.get("truth_item_table")
    if not isinstance(table, list) or not table:
        raise Q0InputError("baseline evaluation has no truth_item_table")
    items: list[TruthItem] = []
    for row in table:
        if row.get("scoring_status") != "SCORED":
            continue
        interval = row.get("interval_ns")
        if row.get("item_kind") == "positive_event":
            interval = row.get("valid_interval_ns")
        if not isinstance(interval, list) or len(interval) != 2:
            raise Q0InputError(f"invalid scored truth interval for {row.get('item_id')}")
        items.append(
            TruthItem(
                item_id=str(row["item_id"]),
                item_kind=str(row["item_kind"]),
                session_id=str(row["session_id"]),
                start_rel_ns=int(interval[0]),
                end_rel_ns=int(interval[1]),
                scoring_status=str(row["scoring_status"]),
            )
        )
    if not items:
        raise Q0InputError("no scored truth items")
    return items


def load_fast_features(path: Path, frames: list[Frame]) -> dict[tuple[str, str], FastFeatureRow]:
    rows = read_jsonl(path)
    feature_schema = "blindassist.dual_loop_semantic_refresh_fast_features.v1"
    features: dict[tuple[str, str], FastFeatureRow] = {}
    for row in rows:
        if row.get("schema_version") != feature_schema:
            raise Q0InputError(f"fast-feature schema mismatch at {row.get('frame_id')}")
        if row.get("causal_source") != "CURRENT_FRAME_FAST_LOOP_ONLY":
            raise Q0InputError("fast features are not declared current-frame-only")
        if row.get("uses_full_detector_output") is not False:
            raise Q0InputError("fast features must explicitly reject full detector output")
        key = (str(row["session_id"]), str(row["frame_id"]))
        if key in features:
            raise Q0InputError(f"duplicate fast-feature row {key}")
        values = row.get("features")
        if not isinstance(values, dict):
            raise Q0InputError(f"missing fast-feature map at {key}")
        features[key] = FastFeatureRow(
            session_id=key[0],
            frame_id=key[1],
            timestamp_ns=int(row["source_capture_timestamp_ns"]),
            features={
                str(name): value
                for name, value in values.items()
                if isinstance(value, (bool, int, float)) or value is None
            },
        )
    frame_keys = {frame.key for frame in frames}
    if set(features) != frame_keys:
        raise Q0InputError("fast-feature frame identity does not exactly match input frames")
    for frame in frames:
        feature = features[frame.key]
        if feature.timestamp_ns != frame.timestamp_ns:
            raise Q0InputError(f"fast-feature timestamp mismatch at {frame.key}")
    return features


def _detection(risk: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(risk, Mapping):
        return None
    value = risk.get("source_detection")
    return dict(value) if isinstance(value, Mapping) else None


def _identity(detection: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    if detection is None:
        return None
    return tuple(detection.get(field) for field in ("class_id", "label", "source"))


def _box_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(first["right"]), float(second["right"]))
        - max(float(first["left"]), float(second["left"])),
    ) * max(
        0.0,
        min(float(first["bottom"]), float(second["bottom"]))
        - max(float(first["top"]), float(second["top"])),
    )
    first_area = max(0.0, float(first["right"]) - float(first["left"])) * max(
        0.0, float(first["bottom"]) - float(first["top"])
    )
    second_area = max(0.0, float(second["right"]) - float(second["left"])) * max(
        0.0, float(second["bottom"]) - float(second["top"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _same_target(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if _identity(first) != _identity(second):
        return False
    if _box_iou(first, second) >= 0.25:
        return True
    first_center = (
        (float(first["left"]) + float(first["right"]))
        / 2.0
        / max(1.0, float(first["frame_width"])),
        (float(first["top"]) + float(first["bottom"]))
        / 2.0
        / max(1.0, float(first["frame_height"])),
    )
    second_center = (
        (float(second["left"]) + float(second["right"]))
        / 2.0
        / max(1.0, float(second["frame_width"])),
        (float(second["top"]) + float(second["bottom"]))
        / 2.0
        / max(1.0, float(second["frame_height"])),
    )
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]) <= 0.12


def _risk_signature(risk: Mapping[str, Any] | None) -> tuple[Any, ...]:
    if risk is None:
        return ("NONE", "NONE", "FAR", "NO_SUPPORTED_TARGET_EVIDENCE")
    return tuple(
        risk.get(field)
        for field in ("level", "direction", "proximity", "evidence_state")
    )


def divergence_levels(
    cached_risk: Mapping[str, Any] | None,
    reference_risk: Mapping[str, Any] | None,
    candidate_feedback: bool,
    reference_feedback: bool,
) -> dict[str, Any]:
    cached_detection = _detection(cached_risk)
    reference_detection = _detection(reference_risk)
    level1 = {
        "new_target": cached_detection is None and reference_detection is not None,
        "target_disappeared": cached_detection is not None and reference_detection is None,
        "target_identity_changed": (
            cached_detection is not None
            and reference_detection is not None
            and _identity(cached_detection) != _identity(reference_detection)
        ),
        "detection_changed": (
            cached_detection is not None
            and reference_detection is not None
            and _identity(cached_detection) == _identity(reference_detection)
            and not _same_target(cached_detection, reference_detection)
        ),
    }
    level2 = {
        "stable_risk_changed": _risk_signature(cached_risk) != _risk_signature(reference_risk),
    }
    level3 = {
        "feedback_decision_changed": candidate_feedback != reference_feedback,
        "event_active_changed": (
            (cached_risk is not None and cached_risk.get("level") != "NONE")
            != (reference_risk is not None and reference_risk.get("level") != "NONE")
        ),
    }
    return {
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "level1_any": any(level1.values()),
        "level2_any": any(level2.values()),
        "level3_any": any(level3.values()),
    }


@dataclass
class FeedbackState:
    session_id: str | None = None
    origin_timestamp_ns: int = 0
    last_alert_ms: dict[tuple[Any, Any], int] = field(default_factory=dict)
    fatigue_last_ms: int = 0
    fatigue_count: int = 0
    previous_low_confidence: dict[str, Any] | None = None

    def reset_for_session(self, session_id: str, timestamp_ns: int) -> None:
        self.session_id = session_id
        self.origin_timestamp_ns = timestamp_ns
        self.last_alert_ms.clear()
        self.fatigue_last_ms = 0
        self.fatigue_count = 0
        self.previous_low_confidence = None

    def update(self, risk: Mapping[str, Any] | None, timestamp_ns: int) -> bool:
        if self.session_id is None:
            raise Q0InputError("feedback state was not initialized")
        now_ms = (timestamp_ns - self.origin_timestamp_ns) // NANOS_PER_MILLISECOND
        if risk is None:
            self.previous_low_confidence = None
            return False
        detection = _detection(risk)
        requires_confirmation = (
            detection is not None
            and detection.get("source") == "OBJECT_DETECTOR"
            and detection.get("label") == "person"
            and float(detection.get("confidence", 1.0)) < 0.5
            and risk.get("direction") != "CENTER"
        )
        confirmed = True
        if requires_confirmation:
            confirmed = (
                self.previous_low_confidence is not None
                and self.previous_low_confidence.get("label") == detection.get("label")
                and self.previous_low_confidence.get("source") == detection.get("source")
                and _same_target(self.previous_low_confidence, detection)
            )
            self.previous_low_confidence = detection
        else:
            self.previous_low_confidence = None

        base_cooldown: int | None = None
        if risk.get("proximity") == "CRITICAL" and risk.get("level") == "HIGH":
            base_cooldown = 850
        elif risk.get("proximity") == "NEAR" and risk.get("level") in {"HIGH", "MEDIUM"}:
            base_cooldown = 1500
        if not confirmed or base_cooldown is None:
            return False

        alert_key = (risk.get("direction"), risk.get("proximity"))
        critical = base_cooldown == 850
        fatigue_level = (
            0
            if critical
            else self.fatigue_count if now_ms - self.fatigue_last_ms <= 12_000 else 0
        )
        multiplier = 2.0 if fatigue_level >= 4 else 1.5 if fatigue_level >= 2 else 1.0
        cooldown = base_cooldown if critical else int(base_cooldown * multiplier)
        previous_alert = self.last_alert_ms.get(alert_key)
        if previous_alert is not None and now_ms - previous_alert < cooldown:
            return False
        self.last_alert_ms[alert_key] = now_ms
        if critical:
            self.fatigue_count = 0
            self.fatigue_last_ms = 0
        else:
            self.fatigue_count = (
                self.fatigue_count + 1
                if now_ms - self.fatigue_last_ms <= 12_000
                else 1
            )
            self.fatigue_last_ms = now_ms
        return True


def _event_snapshot(
    state: ArmState,
    event_active: bool,
    feedback_triggered: bool,
) -> dict[str, Any]:
    if event_active and state.active_event_id is None:
        state.active_event_id = (
            f"Q0-{state.session_id}-event-{state.next_event_sequence_number:04d}"
        )
        state.next_event_sequence_number += 1
    elif not event_active:
        state.active_event_id = None
    return {
        "event_id": state.active_event_id,
        "active_event_id": state.active_event_id,
        "next_event_sequence_number": state.next_event_sequence_number,
        "state": "ACTIVE" if event_active else "CLEAR",
        "active": event_active,
        "feedback_count": 1 if feedback_triggered else 0,
    }


@dataclass
class ArmResult:
    name: str
    status: str
    propagation_mode: str
    outputs: list[dict[str, Any]]
    metrics: dict[str, Any]
    reason: str | None = None


def simulate_arm(
    frames: list[Frame],
    policy: RefreshPolicy,
    fast_features: dict[tuple[str, str], FastFeatureRow] | None = None,
) -> ArmResult:
    state = ArmState()
    feedback_state = FeedbackState()
    outputs: list[dict[str, Any]] = []
    for frame in frames:
        if state.session_id != frame.session_id:
            state.reset_for_session(frame.session_id)
            feedback_state.reset_for_session(frame.session_id, frame.timestamp_ns)
        feature = fast_features.get(frame.key) if fast_features is not None else None
        refresh, refresh_reason = policy.should_refresh(frame, state, feature)
        if refresh:
            state.cached_risk = copy.deepcopy(frame.stable_risk)
            state.last_refresh_timestamp_ns = frame.timestamp_ns
            state.last_refresh_frame_id = frame.frame_id
            state.refresh_count += 1
        candidate_risk = copy.deepcopy(state.cached_risk)
        cache_age_ns = (
            None
            if state.last_refresh_timestamp_ns is None
            else frame.timestamp_ns - state.last_refresh_timestamp_ns
        )
        candidate_feedback = feedback_state.update(candidate_risk, frame.timestamp_ns)
        event_active = candidate_risk is not None and candidate_risk.get("level") != "NONE"
        divergence = divergence_levels(
            candidate_risk,
            frame.stable_risk,
            candidate_feedback,
            frame.reference_feedback_triggered,
        )
        outputs.append(
            {
                "session_id": frame.session_id,
                "frame_id": frame.frame_id,
                "source_capture_timestamp_ns": frame.timestamp_ns,
                "refresh_requested": refresh,
                "refresh_reason": refresh_reason,
                "detector_ran": refresh,
                "cache_age_ns": cache_age_ns,
                "cache_source_frame_id": state.last_refresh_frame_id,
                "candidate_risk": candidate_risk,
                "candidate_feedback_triggered": candidate_feedback,
                "candidate_event": _event_snapshot(
                    state, event_active, candidate_feedback
                ),
                "reference_feedback_triggered": frame.reference_feedback_triggered,
                "reference_event": frame.reference_risk_event,
                "divergence": divergence,
            }
        )
    return ArmResult(
        name=policy.name,
        status="VALID",
        propagation_mode=PROPAGATION_MODE,
        outputs=outputs,
        metrics={},
    )


def _session_origins(frames: list[Frame]) -> dict[str, int]:
    origins: dict[str, int] = {}
    for frame in frames:
        origins.setdefault(frame.session_id, frame.timestamp_ns)
    return origins


def _window_outputs(
    outputs: list[dict[str, Any]], item: TruthItem, origins: Mapping[str, int]
) -> list[dict[str, Any]]:
    origin = origins[item.session_id]
    return [
        row
        for row in outputs
        if row["session_id"] == item.session_id
        and item.start_rel_ns
        <= int(row["source_capture_timestamp_ns"]) - origin
        <= item.end_rel_ns
        and row["candidate_feedback_triggered"]
    ]


@dataclass(frozen=True)
class RiskEpisode:
    episode_id: str
    session_id: str
    start_timestamp_ns: int
    end_timestamp_ns_exclusive: int
    start_frame_id: str
    end_frame_id: str
    risk_signature: tuple[Any, Any, Any]
    feedback_timestamps_ns: tuple[int, ...]
    frame_count: int
    max_stale_duration_ms: float

    @property
    def duration_ms(self) -> float:
        return (
            self.end_timestamp_ns_exclusive - self.start_timestamp_ns
        ) / NANOS_PER_MILLISECOND


def _active_risk_signature(
    risk: Mapping[str, Any] | None,
) -> tuple[Any, Any, Any] | None:
    if not isinstance(risk, Mapping) or risk.get("level") == "NONE":
        return None
    return tuple(risk.get(field) for field in ("level", "direction", "proximity"))


def _session_step_ns(rows: list[Mapping[str, Any]]) -> int:
    timestamps = [int(row["source_capture_timestamp_ns"]) for row in rows]
    differences = [
        right - left for left, right in zip(timestamps, timestamps[1:]) if right > left
    ]
    if not differences:
        return NANOS_PER_MILLISECOND
    ordered = sorted(differences)
    return ordered[(len(ordered) - 1) // 2]


def segment_risk_episodes(
    rows: list[Mapping[str, Any]],
    source_label: str,
) -> list[RiskEpisode]:
    """Segment active risk-signature runs without smoothing or future reads.

    A new episode starts at an active transition or when level/direction/proximity
    changes. This is an evaluation unit, not a claim that the runtime's event
    lifecycle has been independently validated.
    """

    episodes: list[RiskEpisode] = []
    session_ids = sorted({str(row["session_id"]) for row in rows})
    for session_id in session_ids:
        session_rows = [
            row for row in rows if str(row["session_id"]) == session_id
        ]
        step_ns = _session_step_ns(session_rows)
        current: dict[str, Any] | None = None

        def finish(end_timestamp_ns: int) -> None:
            nonlocal current
            if current is None:
                return
            sequence = sum(
                1
                for episode in episodes
                if episode.session_id == session_id
            ) + 1
            episodes.append(
                RiskEpisode(
                    episode_id=(
                        f"Q0-R0.1-{source_label}-{session_id}-episode-"
                        f"{sequence:04d}"
                    ),
                    session_id=session_id,
                    start_timestamp_ns=int(current["start_timestamp_ns"]),
                    end_timestamp_ns_exclusive=int(end_timestamp_ns),
                    start_frame_id=str(current["start_frame_id"]),
                    end_frame_id=str(current["end_frame_id"]),
                    risk_signature=tuple(current["risk_signature"]),
                    feedback_timestamps_ns=tuple(current["feedback_timestamps_ns"]),
                    frame_count=int(current["frame_count"]),
                    max_stale_duration_ms=max(
                        [float(value) for value in current["stale_duration_ms"]],
                        default=0.0,
                    ),
                )
            )
            current = None

        for index, row in enumerate(session_rows):
            timestamp_ns = int(row["source_capture_timestamp_ns"])
            signature = _active_risk_signature(row.get("risk"))
            if signature is None:
                finish(timestamp_ns)
                continue
            if current is None or tuple(current["risk_signature"]) != signature:
                finish(timestamp_ns)
                current = {
                    "start_timestamp_ns": timestamp_ns,
                    "start_frame_id": str(row["frame_id"]),
                    "end_frame_id": str(row["frame_id"]),
                    "risk_signature": signature,
                    "feedback_timestamps_ns": [],
                    "stale_duration_ms": [],
                    "frame_count": 0,
                }
            if index + 1 < len(session_rows):
                end_timestamp_ns = int(
                    session_rows[index + 1]["source_capture_timestamp_ns"]
                )
            else:
                end_timestamp_ns = timestamp_ns + step_ns
            current["end_frame_id"] = str(row["frame_id"])
            current["frame_count"] += 1
            if bool(row.get("feedback_triggered", False)):
                current["feedback_timestamps_ns"].append(timestamp_ns)
            cache_age_ns = row.get("cache_age_ns")
            if cache_age_ns is not None:
                current["stale_duration_ms"].append(
                    max(0.0, float(cache_age_ns) / NANOS_PER_MILLISECOND)
                )
            current["end_timestamp_ns_exclusive"] = end_timestamp_ns
        if current is not None:
            finish(int(current["end_timestamp_ns_exclusive"]))
    return episodes


def _episode_record(episode: RiskEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "session_id": episode.session_id,
        "start_timestamp_ns": episode.start_timestamp_ns,
        "end_timestamp_ns_exclusive": episode.end_timestamp_ns_exclusive,
        "start_frame_id": episode.start_frame_id,
        "end_frame_id": episode.end_frame_id,
        "duration_ms": episode.duration_ms,
        "risk_level": episode.risk_signature[0],
        "direction": episode.risk_signature[1],
        "proximity": episode.risk_signature[2],
        "feedback_count": len(episode.feedback_timestamps_ns),
        "frame_count": episode.frame_count,
        "max_stale_duration_ms": episode.max_stale_duration_ms,
    }


def _interval_overlap_ns(first: RiskEpisode, second: RiskEpisode) -> int:
    return max(
        0,
        min(first.end_timestamp_ns_exclusive, second.end_timestamp_ns_exclusive)
        - max(first.start_timestamp_ns, second.start_timestamp_ns),
    )


def _interval_iou(first: RiskEpisode, second: RiskEpisode) -> float:
    overlap = _interval_overlap_ns(first, second)
    union = max(
        first.end_timestamp_ns_exclusive,
        second.end_timestamp_ns_exclusive,
    ) - min(first.start_timestamp_ns, second.start_timestamp_ns)
    return overlap / union if union > 0 else 0.0


def _match_episode_indices(
    candidate: list[RiskEpisode], reference: list[RiskEpisode]
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, float, int, int]] = []
    for candidate_index, candidate_episode in enumerate(candidate):
        for reference_index, reference_episode in enumerate(reference):
            if candidate_episode.session_id != reference_episode.session_id:
                continue
            overlap = _interval_overlap_ns(candidate_episode, reference_episode)
            if overlap > 0:
                candidates.append(
                    (
                        -overlap,
                        -_interval_iou(candidate_episode, reference_episode),
                        candidate_index,
                        reference_index,
                    )
                )
    matched_candidate: set[int] = set()
    matched_reference: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, candidate_index, reference_index in sorted(candidates):
        if candidate_index in matched_candidate or reference_index in matched_reference:
            continue
        matched_candidate.add(candidate_index)
        matched_reference.add(reference_index)
        matches.append((candidate_index, reference_index))
    return sorted(matches)


def _match_feedback_timestamps(
    candidate: tuple[int, ...],
    reference: tuple[int, ...],
    tolerance_ns: int = EPISODE_FEEDBACK_MATCH_TOLERANCE_NS,
) -> tuple[int, int, int]:
    possible: list[tuple[int, int, int]] = []
    for candidate_index, candidate_timestamp_ns in enumerate(candidate):
        for reference_index, reference_timestamp_ns in enumerate(reference):
            difference = abs(candidate_timestamp_ns - reference_timestamp_ns)
            if difference <= tolerance_ns:
                possible.append((difference, candidate_index, reference_index))
    used_candidate: set[int] = set()
    used_reference: set[int] = set()
    matched = 0
    for _, candidate_index, reference_index in sorted(possible):
        if candidate_index in used_candidate or reference_index in used_reference:
            continue
        used_candidate.add(candidate_index)
        used_reference.add(reference_index)
        matched += 1
    return matched, len(candidate) - matched, len(reference) - matched


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
    }


def evaluate_episode_alignment(
    candidate_rows: list[Mapping[str, Any]],
    reference_rows: list[Mapping[str, Any]],
    feedback_tolerance_ms: float = EPISODE_FEEDBACK_MATCH_TOLERANCE_MS,
) -> dict[str, Any]:
    tolerance_ns = int(feedback_tolerance_ms * NANOS_PER_MILLISECOND)
    candidate_episodes = segment_risk_episodes(candidate_rows, "candidate")
    reference_episodes = segment_risk_episodes(reference_rows, "reference")
    matches = _match_episode_indices(candidate_episodes, reference_episodes)
    matched_candidate = {candidate_index for candidate_index, _ in matches}
    matched_reference = {reference_index for _, reference_index in matches}

    matched_records: list[dict[str, Any]] = []
    onset_delays: list[float] = []
    absolute_onset_delays: list[float] = []
    offset_delays: list[float] = []
    temporal_ious: list[float] = []
    risk_signature_matches: list[bool] = []
    feedback_matches = 0
    candidate_unmatched_feedback = 0
    reference_unmatched_feedback = 0
    for match_index, (candidate_index, reference_index) in enumerate(matches, start=1):
        candidate_episode = candidate_episodes[candidate_index]
        reference_episode = reference_episodes[reference_index]
        onset_delay_ms = (
            candidate_episode.start_timestamp_ns
            - reference_episode.start_timestamp_ns
        ) / NANOS_PER_MILLISECOND
        offset_delay_ms = (
            candidate_episode.end_timestamp_ns_exclusive
            - reference_episode.end_timestamp_ns_exclusive
        ) / NANOS_PER_MILLISECOND
        temporal_iou = _interval_iou(candidate_episode, reference_episode)
        signature_match = (
            candidate_episode.risk_signature == reference_episode.risk_signature
        )
        matched, candidate_unmatched, reference_unmatched = _match_feedback_timestamps(
            candidate_episode.feedback_timestamps_ns,
            reference_episode.feedback_timestamps_ns,
            tolerance_ns,
        )
        feedback_matches += matched
        candidate_unmatched_feedback += candidate_unmatched
        reference_unmatched_feedback += reference_unmatched
        onset_delays.append(onset_delay_ms)
        absolute_onset_delays.append(abs(onset_delay_ms))
        offset_delays.append(offset_delay_ms)
        temporal_ious.append(temporal_iou)
        risk_signature_matches.append(signature_match)
        matched_records.append(
            {
                "match_id": f"Q0-R0.1-episode-match-{match_index:04d}",
                "candidate": _episode_record(candidate_episode),
                "reference": _episode_record(reference_episode),
                "onset_delay_ms": onset_delay_ms,
                "offset_delay_ms": offset_delay_ms,
                "temporal_iou": temporal_iou,
                "risk_level_match": candidate_episode.risk_signature[0]
                == reference_episode.risk_signature[0],
                "direction_match": candidate_episode.risk_signature[1]
                == reference_episode.risk_signature[1],
                "proximity_match": candidate_episode.risk_signature[2]
                == reference_episode.risk_signature[2],
                "risk_signature_match": signature_match,
                "feedback_match_tolerance_ms": feedback_tolerance_ms,
                "feedback_matched_count": matched,
                "candidate_unmatched_feedback_count": candidate_unmatched,
                "reference_unmatched_feedback_count": reference_unmatched,
                "feedback_count_delta": len(
                    candidate_episode.feedback_timestamps_ns
                ) - len(reference_episode.feedback_timestamps_ns),
            }
        )

    reference_count = len(reference_episodes)
    candidate_count = len(candidate_episodes)
    return {
        "status": "VALID",
        "segmentation_unit": "contiguous active risk-signature run",
        "active_definition": "risk.level != NONE",
        "matching_rule": "one-to-one same-session positive-overlap greedy by overlap then IoU",
        "feedback_match_tolerance_ms": feedback_tolerance_ms,
        "candidate_episode_count": candidate_count,
        "reference_episode_count": reference_count,
        "matched_episode_count": len(matches),
        "reference_episode_unmatched_count": reference_count - len(matches),
        "candidate_episode_unmatched_count": candidate_count - len(matches),
        "reference_episode_match_recall": (
            len(matches) / reference_count if reference_count else 1.0
        ),
        "candidate_episode_match_precision": (
            len(matches) / candidate_count if candidate_count else 1.0
        ),
        "onset_delay_ms": _summary_stats(onset_delays),
        "absolute_onset_delay_ms": _summary_stats(absolute_onset_delays),
        "offset_delay_ms": _summary_stats(offset_delays),
        "temporal_iou": _summary_stats(temporal_ious),
        "risk_level_match_rate": (
            sum(
                record["risk_level_match"] for record in matched_records
            )
            / len(matched_records)
            if matched_records
            else 1.0
        ),
        "direction_match_rate": (
            sum(record["direction_match"] for record in matched_records)
            / len(matched_records)
            if matched_records
            else 1.0
        ),
        "proximity_match_rate": (
            sum(record["proximity_match"] for record in matched_records)
            / len(matched_records)
            if matched_records
            else 1.0
        ),
        "risk_signature_match_rate": (
            sum(risk_signature_matches) / len(risk_signature_matches)
            if risk_signature_matches
            else 1.0
        ),
        "feedback": {
            "matched_count": feedback_matches,
            "candidate_unmatched_count": candidate_unmatched_feedback,
            "reference_unmatched_count": reference_unmatched_feedback,
            "count_delta": sum(
                len(episode.feedback_timestamps_ns)
                for episode in candidate_episodes
            )
            - sum(
                len(episode.feedback_timestamps_ns)
                for episode in reference_episodes
            ),
        },
        "candidate_longest_stale_duration_ms": max(
            [episode.max_stale_duration_ms for episode in candidate_episodes],
            default=0.0,
        ),
        "reference_longest_stale_duration_ms": max(
            [episode.max_stale_duration_ms for episode in reference_episodes],
            default=0.0,
        ),
        "matched_episodes": matched_records,
        "unmatched_candidate_episodes": [
            _episode_record(episode)
            for index, episode in enumerate(candidate_episodes)
            if index not in matched_candidate
        ],
        "unmatched_reference_episodes": [
            _episode_record(episode)
            for index, episode in enumerate(reference_episodes)
            if index not in matched_reference
        ],
    }


def evaluate_arm(
    arm: ArmResult,
    frames: list[Frame],
    truth_items: list[TruthItem],
) -> dict[str, Any]:
    outputs = arm.outputs
    origins = _session_origins(frames)
    reference_by_key = {frame.key: frame.reference_feedback_triggered for frame in frames}
    candidate_by_key = {
        (row["session_id"], row["frame_id"]): bool(row["candidate_feedback_triggered"])
        for row in outputs
    }
    candidate_episode_rows = [
        {
            "session_id": row["session_id"],
            "frame_id": row["frame_id"],
            "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
            "risk": row["candidate_risk"],
            "feedback_triggered": row["candidate_feedback_triggered"],
            "cache_age_ns": row["cache_age_ns"],
        }
        for row in outputs
    ]
    reference_episode_rows = [
        {
            "session_id": frame.session_id,
            "frame_id": frame.frame_id,
            "source_capture_timestamp_ns": frame.timestamp_ns,
            "risk": frame.stable_risk,
            "feedback_triggered": frame.reference_feedback_triggered,
            "cache_age_ns": 0,
        }
        for frame in frames
    ]
    episode_alignment = evaluate_episode_alignment(
        candidate_episode_rows,
        reference_episode_rows,
    )
    positive_items = [item for item in truth_items if item.item_kind == "positive_event"]
    negative_items = [item for item in truth_items if item.item_kind == "negative_window"]
    item_results: list[dict[str, Any]] = []
    for item in truth_items:
        origin = origins[item.session_id]
        reference_rows = [
            frame
            for frame in frames
            if frame.session_id == item.session_id
            and item.start_rel_ns
            <= frame.timestamp_ns - origin
            <= item.end_rel_ns
            and frame.reference_feedback_triggered
        ]
        candidate_rows = _window_outputs(outputs, item, origins)
        reference_first = reference_rows[0].timestamp_ns if reference_rows else None
        candidate_first = (
            int(candidate_rows[0]["source_capture_timestamp_ns"])
            if candidate_rows
            else None
        )
        item_results.append(
            {
                "item_id": item.item_id,
                "item_kind": item.item_kind,
                "session_id": item.session_id,
                "reference_trigger_count": len(reference_rows),
                "candidate_trigger_count": len(candidate_rows),
                "reference_hit": bool(reference_rows),
                "candidate_hit": bool(candidate_rows),
                "reference_event_missed": bool(reference_rows) and not candidate_rows,
                "candidate_added_reference_event": bool(candidate_rows) and not reference_rows,
                "feedback_count_delta": len(candidate_rows) - len(reference_rows),
                "first_feedback_delay_ms": (
                    None
                    if item.item_kind != "positive_event"
                    or reference_first is None
                    or candidate_first is None
                    else (candidate_first - reference_first) / NANOS_PER_MILLISECOND
                ),
            }
        )

    def count(rows: list[dict[str, Any]], key: str) -> int:
        return sum(bool(row[key]) for row in rows)

    level1_count = sum(bool(row["divergence"]["level1_any"]) for row in outputs)
    level2_count = sum(bool(row["divergence"]["level2_any"]) for row in outputs)
    level3_count = sum(bool(row["divergence"]["level3_any"]) for row in outputs)
    level3_feedback_count = sum(
        bool(row["divergence"]["level3"]["feedback_decision_changed"])
        for row in outputs
    )
    level3_event_active_count = sum(
        bool(row["divergence"]["level3"]["event_active_changed"])
        for row in outputs
    )
    positive_results = [
        row for row in item_results if row["item_kind"] == "positive_event"
    ]
    negative_results = [
        row for row in item_results if row["item_kind"] == "negative_window"
    ]
    delays = [
        row["first_feedback_delay_ms"]
        for row in positive_results
        if row["first_feedback_delay_ms"] is not None
    ]
    delay_tolerance_ms = EPISODE_FEEDBACK_MATCH_TOLERANCE_MS
    delay_stats = _summary_stats(delays)
    absolute_delay_stats = _summary_stats([abs(value) for value in delays])
    first_feedback_delay = {
        "tolerance_ms": delay_tolerance_ms,
        "eligible_positive_event_count": len(positive_results),
        "matched_positive_event_count": len(delays),
        "missing_candidate_feedback_count": len(positive_results) - len(delays),
        "early_count": sum(value < -delay_tolerance_ms for value in delays),
        "within_tolerance_count": sum(
            abs(value) <= delay_tolerance_ms for value in delays
        ),
        "late_count": sum(value > delay_tolerance_ms for value in delays),
        "signed_ms": delay_stats,
        "absolute_ms": absolute_delay_stats,
    }
    session_metrics: dict[str, Any] = {}
    for session_id in sorted({frame.session_id for frame in frames}):
        rows = [row for row in item_results if row["session_id"] == session_id]
        session_delays = [
            float(row["first_feedback_delay_ms"])
            for row in rows
            if row["first_feedback_delay_ms"] is not None
        ]
        session_metrics[session_id] = {
            "truth_item_count": len(rows),
            "positive_count": sum(row["item_kind"] == "positive_event" for row in rows),
            "negative_count": sum(row["item_kind"] == "negative_window" for row in rows),
            "candidate_positive_hits": count(
                [row for row in rows if row["item_kind"] == "positive_event"],
                "candidate_hit",
            ),
            "candidate_negative_hits": count(
                [row for row in rows if row["item_kind"] == "negative_window"],
                "candidate_hit",
            ),
            "feedback_count_delta": sum(
                int(row["feedback_count_delta"]) for row in rows
            ),
            "first_feedback_delay_ms": _summary_stats(session_delays),
            "first_feedback_delay_absolute_ms": _summary_stats(
                [abs(value) for value in session_delays]
            ),
        }
    return {
        "status": arm.status,
        "policy": arm.name,
        "propagation_mode": arm.propagation_mode,
        "frame_count": len(outputs),
        "detector_call_count": sum(bool(row["detector_ran"]) for row in outputs),
        "detector_call_rate": sum(bool(row["detector_ran"]) for row in outputs) / len(outputs),
        "cache_age_max_ms": max(
            [
                row["cache_age_ns"] / NANOS_PER_MILLISECOND
                for row in outputs
                if row["cache_age_ns"] is not None
            ],
            default=None,
        ),
        "divergence": {
            "level1_detection_frame_count": level1_count,
            "level2_risk_decision_frame_count": level2_count,
            "level3_event_or_feedback_frame_count": level3_count,
            "level3_feedback_decision_frame_count": level3_feedback_count,
            "level3_event_active_frame_count": level3_event_active_count,
        },
        "truth_item_metrics": {
            "item_count": len(item_results),
            "positive_count": len(positive_items),
            "negative_count": len(negative_items),
            "reference_positive_hits": count(positive_results, "reference_hit"),
            "candidate_positive_hits": count(positive_results, "candidate_hit"),
            "reference_negative_hits": count(negative_results, "reference_hit"),
            "candidate_negative_hits": count(negative_results, "candidate_hit"),
            "reference_event_missed_count": count(
                positive_results, "reference_event_missed"
            ),
            "candidate_added_reference_event_count": count(
                item_results, "candidate_added_reference_event"
            ),
            "feedback_count_delta": sum(
                int(row["feedback_count_delta"]) for row in item_results
            ),
            "feedback_count_absolute_delta": sum(
                abs(int(row["feedback_count_delta"])) for row in item_results
            ),
            "first_feedback_delay": first_feedback_delay,
            "first_feedback_delay_max_ms": delay_stats["max"],
            "first_feedback_delay_min_ms": delay_stats["min"],
            "first_feedback_delay_p50_ms": delay_stats["p50"],
            "first_feedback_delay_p90_ms": delay_stats["p90"],
            "first_feedback_delay_p95_ms": delay_stats["p95"],
            "first_feedback_delay_values_ms": delays,
            "items": item_results,
        },
        "episode_alignment": episode_alignment,
        "session_metrics": session_metrics,
        "reference_frame_count": sum(reference_by_key.values()),
        "candidate_frame_count": sum(candidate_by_key.values()),
    }


def _nested_metric_value(
    metrics: Mapping[str, Any], section: str, key: str
) -> float:
    section_value = metrics.get(section)
    if not isinstance(section_value, Mapping):
        return math.inf
    value = section_value.get(key)
    if value is None:
        return math.inf
    return float(value)


def _raw_pareto_objectives(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["detector_call_count"]),
        float(metrics["divergence"]["level3_event_or_feedback_frame_count"]),
        float(metrics["truth_item_metrics"]["reference_event_missed_count"]),
        float(metrics["episode_alignment"]["reference_episode_unmatched_count"]),
        _nested_metric_value(
            metrics["episode_alignment"], "absolute_onset_delay_ms", "p95"
        ),
        1.0
        - _nested_metric_value(metrics["episode_alignment"], "temporal_iou", "mean"),
    )


def raw_nondominated_set(arm_metrics: list[dict[str, Any]]) -> list[str]:
    usable = [metrics for metrics in arm_metrics if metrics.get("status") == "VALID"]
    objectives = {
        metrics["policy"]: _raw_pareto_objectives(metrics) for metrics in usable
    }
    front: list[str] = []
    for name, values in objectives.items():
        dominated = False
        for other_name, other in objectives.items():
            if other_name == name:
                continue
            no_worse = all(left <= right for left, right in zip(other, values))
            strictly_better = any(left < right for left, right in zip(other, values))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(name)
    return sorted(front)


def _admission_failures(metrics: Mapping[str, Any]) -> list[str]:
    if metrics.get("status") != "VALID":
        return ["STATUS_NOT_VALID"]
    truth = metrics["truth_item_metrics"]
    episodes = metrics["episode_alignment"]
    failures: list[str] = []
    if (
        int(truth["reference_event_missed_count"])
        > EPISODE_ADMISSION_CONSTRAINTS["reference_event_missed_count_max"]
    ):
        failures.append("REFERENCE_EVENT_MISS")
    if (
        float(episodes["reference_episode_match_recall"])
        < EPISODE_ADMISSION_CONSTRAINTS["reference_episode_match_recall_min"]
    ):
        failures.append("REFERENCE_EPISODE_COVERAGE")
    if (
        _nested_metric_value(episodes, "temporal_iou", "mean")
        < EPISODE_ADMISSION_CONSTRAINTS["episode_temporal_iou_mean_min"]
    ):
        failures.append("EPISODE_TEMPORAL_IOU")
    if (
        _nested_metric_value(episodes, "absolute_onset_delay_ms", "p95")
        > EPISODE_ADMISSION_CONSTRAINTS["episode_abs_onset_delay_p95_ms_max"]
    ):
        failures.append("EPISODE_ONSET_DELAY")
    if (
        float(episodes["risk_signature_match_rate"])
        < EPISODE_ADMISSION_CONSTRAINTS["risk_signature_match_rate_min"]
    ):
        failures.append("RISK_SIGNATURE_MISMATCH")
    return failures


def constrained_operating_point(
    arm_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = {
        metrics["policy"]: _admission_failures(metrics) for metrics in arm_metrics
    }
    admissible = sorted(
        name for name, reasons in failures.items() if not reasons
    )
    valid_admissible = [
        metrics for metrics in arm_metrics if metrics["policy"] in admissible
    ]
    valid_admissible.sort(
        key=lambda metrics: (
            int(metrics["detector_call_count"]),
            _nested_metric_value(
                metrics["episode_alignment"], "absolute_onset_delay_ms", "p95"
            ),
            int(metrics["divergence"]["level3_event_or_feedback_frame_count"]),
            metrics["policy"],
        )
    )
    best = valid_admissible[0]["policy"] if valid_admissible else None
    return {
        "constraints": EPISODE_ADMISSION_CONSTRAINTS,
        "admissible_set": admissible,
        "best_operating_point": best,
        "decision_rule": (
            "Among admissible VALID policies, minimize detector_call_count; "
            "break ties by absolute episode onset-delay p95, then Level-3 divergence."
        ),
        "rejection_reasons": failures,
    }


def run_q0(
    dump_path: Path,
    baseline_path: Path,
    baseline_evaluation_path: Path,
    dump_receipt_path: Path | None,
    fast_features_path: Path | None,
    intervals_ms: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    frames = load_frames(
        dump_path,
        baseline_path,
        expected_dump_receipt_path=dump_receipt_path,
        expected_baseline_sha256=read_json(baseline_evaluation_path).get("trace_sha256"),
    )
    truth_items = load_truth_items(baseline_evaluation_path)
    fast_features = (
        load_fast_features(fast_features_path, frames)
        if fast_features_path is not None
        else None
    )
    policies: list[RefreshPolicy] = [FullRatePolicy()]
    policies.extend(fixed_time_policies(intervals_ms))
    policies.append(
        MaxAgePolicy(
            max_age_ns=100 * NANOS_PER_MILLISECOND,
            name="AGE_100MS_TRACK_FAILURE",
        )
    )
    policies.append(
        FeatureRulePolicy(
            max_age_ns=100 * NANOS_PER_MILLISECOND,
            motion_threshold=0.15,
            blur_threshold=1.0,
        )
    )
    policies.append(
        LogisticPolicy(
            intercept=0.0,
            weights={},
            threshold=0.5,
            max_age_ns=100 * NANOS_PER_MILLISECOND,
        )
    )

    arm_metrics: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    not_evaluable: dict[str, str] = {}
    for policy in policies:
        try:
            arm = simulate_arm(frames, policy, fast_features)
            metrics = evaluate_arm(arm, frames, truth_items)
            if policy.name == "FULL_RATE_REFERENCE":
                mismatches = [
                    row
                    for row in arm.outputs
                    if row["candidate_feedback_triggered"]
                    != row["reference_feedback_triggered"]
                ]
                if mismatches:
                    raise Q0InputError(
                        "full-rate reference reproduction mismatch at "
                        f"{mismatches[0]['session_id']}/{mismatches[0]['frame_id']}"
                    )
                metrics["reference_reproduction"] = {
                    "status": "VALID",
                    "frame_mismatch_count": 0,
                }
            arm_metrics.append(metrics)
            traces[policy.name] = arm.outputs
        except MissingFastFeatureStream as error:
            not_evaluable[policy.name] = str(error)
            arm_metrics.append(
                {
                    "status": "NOT_EVALUABLE",
                    "policy": policy.name,
                    "reason": str(error),
                    "requires_fast_features": True,
                }
            )

    valid_metrics = [metrics for metrics in arm_metrics if metrics["status"] == "VALID"]
    raw_nondominated = raw_nondominated_set(arm_metrics)
    constrained = constrained_operating_point(arm_metrics)
    result = {
        "schema_version": Q0_R01_SCHEMA,
        "protocol_id": "DUAL_LOOP_SEMANTIC_REFRESH_Q0_R0_1",
        "status": "COMPLETE" if valid_metrics else "NOT_EVALUABLE",
        "stage": "DEVELOPMENT_OFFLINE_EVALUATION_REPAIR",
        "authority": "DEVELOPMENT_ONLY_BURNED_RGB",
        "claim_ceiling": (
            "FIXED_MODEL_FULL_RATE_REFERENCE_PRESERVATION_WITH_RISK_EPISODE_ALIGNMENT; "
            "NO_REAL_WORLD_SEMANTIC_OR_SAFETY_TRUTH"
        ),
        "propagation_mode": PROPAGATION_MODE,
        "analysis_design": {
            "frame_unit": "one source capture frame for causal state transitions",
            "primary_unit": (
                "source-session risk episode alignment plus scored truth-item guardrails"
            ),
            "source_count": len({frame.session_id for frame in frames}),
            "frame_count": len(frames),
            "uncertainty": "DESCRIPTIVE_ONLY_TWO_SESSION_CLUSTER_COUNT; NO_P_VALUES",
            "feature_leakage_guard": "fast features are optional and must be independently declared current-frame-only",
            "episode_segmentation": "contiguous active risk-signature run; risk.level != NONE",
            "episode_matching": "one-to-one same-session positive-overlap greedy by overlap then IoU",
            "feedback_matching_tolerance_ms": EPISODE_FEEDBACK_MATCH_TOLERANCE_MS,
            "delay_summary": "signed and absolute P50/P90/P95; missing positive events remain separate failures",
        },
        "inputs": {
            "detection_dump": {
                "path": str(dump_path),
                "sha256": sha256_file(dump_path),
            },
            "reference_trace": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "baseline_evaluation": {
                "path": str(baseline_evaluation_path),
                "sha256": sha256_file(baseline_evaluation_path),
            },
            "fast_features": (
                {
                    "path": str(fast_features_path),
                    "sha256": sha256_file(fast_features_path),
                    "status": "LOADED",
                }
                if fast_features_path is not None
                else {"status": "NOT_PROVIDED"}
            ),
        },
        "policies": arm_metrics,
        "not_evaluable_policies": not_evaluable,
        "raw_nondominated_set": raw_nondominated,
        "constrained_operating_point": constrained,
        "stop_gate": {
            "status": "NOT_APPLIED_TO_LEARNED_POLICY",
            "reason": (
                "R0.1 only repairs offline episode-level evaluation; no learned policy, "
                "fast-feature stream, Android, energy or safety claim is authorized"
            ),
        },
    }
    return result, traces
