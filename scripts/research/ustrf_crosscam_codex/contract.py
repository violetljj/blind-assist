"""Contracts shared by the isolated cross-camera Codex proxy benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_SCHEMA = "blindassist_ustrf_crosscam_source_receipt_v1"
BUNDLE_SCHEMA = "blindassist_ustrf_crosscam_codex_review_bundle_v1"
REVIEW_SCHEMA = "blindassist_ustrf_crosscam_codex_teacher_review_v1"
CONSENSUS_SCHEMA = "blindassist_ustrf_crosscam_codex_teacher_consensus_v1"
CANDIDATE_SCHEMA = "blindassist_ustrf_crosscam_candidate_trace_v1"
REPORT_SCHEMA = "blindassist_ustrf_crosscam_proxy_benchmark_report_v1"
CONTRACT_ID = "ustrf_crosscam_codex_proxy_r0_v1"

RISK_LEVELS = ("none", "caution", "critical", "unknown")
ROUTE_VALIDITY = ("yes", "no", "uncertain")
ROUTE_RELATIONS = ("inside", "entering", "adjacent", "outside", "uncertain")
DISTANCE_BANDS = ("0-2m", "2-5m", "over-5m", "unknown")
TTC_BANDS = ("0-1.5s", "1.5-3s", "over-3s", "unknown")
ACTIONS = ("none", "slow", "detour", "stop", "step_over", "uncertain")
CATEGORIES = (
    "static_obstacle", "dropoff", "overhead", "vehicle", "person", "animal", "surface", "other"
)
ABSTAIN_REASONS = (
    "blur", "occlusion", "route_invalid", "insufficient_frames", "geometry_ambiguous", "other"
)
ROLES = ("full_context_teacher", "causal_codex_baseline")


class ContractError(ValueError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def require_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def require_enum(value: Any, allowed: Sequence[str], where: str) -> str:
    if value not in allowed:
        raise ContractError(f"{where} must be one of {', '.join(allowed)}")
    return str(value)


def require_false_flags(value: Mapping[str, Any], where: str) -> None:
    for key in (
        "human_event_truth_present", "metric_geometry_present", "training_authorized",
        "u0_authority_granted", "android_runtime_change_authorized",
        "production_model_replacement_authorized",
    ):
        if value.get(key) is not False:
            raise ContractError(f"{where}.{key} must be false")


def validate_source_receipt(value: Any, *, video_path: Path | None = None) -> Mapping[str, Any]:
    row = require_object(value, "source receipt")
    if row.get("schema") != SOURCE_SCHEMA or row.get("contract_id") != CONTRACT_ID:
        raise ContractError("source receipt schema/contract mismatch")
    for key in (
        "source_id", "dataset_name", "dataset_page", "citation",
        "camera_domain",
    ):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ContractError(f"source receipt lacks {key}")
    if row.get("public_data") is not True and row.get("ordinary_public_download") is not True:
        raise ContractError("source must be downloadable through an ordinary public channel")
    for optional_text in ("license_name", "license_url", "privacy_review_status"):
        value = row.get(optional_text)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ContractError(f"source receipt.{optional_text} must be non-empty when present")
    require_false_flags(row, "source receipt")
    expected = require_sha256(row.get("video_sha256"), "source receipt.video_sha256")
    if video_path is not None:
        if not video_path.is_file():
            raise ContractError(f"video is missing: {video_path}")
        if sha256_file(video_path) != expected:
            raise ContractError("video SHA differs from source receipt")
    return row


def validate_geometry(value: Any) -> Mapping[str, Any]:
    row = require_object(value, "assumed geometry")
    expected = {
        "authority": "assumed_geometry_v1",
        "pseudo_metric": True,
        "camera_height_m": (0.5, 2.5),
        "horizontal_fov_deg": (30.0, 150.0),
        "vertical_fov_deg": (20.0, 140.0),
        "walking_speed_mps": (0.1, 3.0),
        "route_width_m": (0.3, 2.0),
        "risk_horizon_s": (0.5, 10.0),
    }
    if row.get("authority") != expected["authority"] or row.get("pseudo_metric") is not True:
        raise ContractError("geometry must be explicit assumed_geometry_v1 pseudo metric")
    for key, bounds in expected.items():
        if key in ("authority", "pseudo_metric"):
            continue
        number = row.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(float(number)) or not bounds[0] <= float(number) <= bounds[1]:
            raise ContractError(f"assumed geometry {key} is outside bounded research range")
    polygon = row.get("route_polygon_xy_norm")
    if not isinstance(polygon, list) or len(polygon) < 4:
        raise ContractError("assumed geometry needs a route polygon")
    for index, point in enumerate(polygon):
        if not isinstance(point, list) or len(point) != 2 or any(
            not isinstance(number, (int, float)) or not 0.0 <= float(number) <= 1.0 for number in point
        ):
            raise ContractError(f"route polygon point {index} is invalid")
    return row


def validate_event(value: Any, *, frame_ids: set[str], where: str) -> Mapping[str, Any]:
    row = require_object(value, where)
    require_enum(row.get("category"), CATEGORIES, f"{where}.category")
    require_enum(row.get("route_relation"), ROUTE_RELATIONS, f"{where}.route_relation")
    require_enum(row.get("distance_band"), DISTANCE_BANDS, f"{where}.distance_band")
    require_enum(row.get("ttc_band"), TTC_BANDS, f"{where}.ttc_band")
    require_enum(row.get("required_action"), ACTIONS, f"{where}.required_action")
    if row.get("confidence") not in ("low", "medium", "high"):
        raise ContractError(f"{where}.confidence is invalid")
    for key in ("start_frame", "end_frame", "peak_frame"):
        if row.get(key) not in frame_ids:
            raise ContractError(f"{where}.{key} is not an admitted input frame")
    evidence = row.get("evidence_frames")
    if not isinstance(evidence, list) or not evidence or any(frame not in frame_ids for frame in evidence):
        raise ContractError(f"{where}.evidence_frames are invalid")
    return row


def validate_review(value: Any, *, bundle: Mapping[str, Any], bundle_sha256: str) -> Mapping[str, Any]:
    row = require_object(value, "teacher review")
    if row.get("schema") != REVIEW_SCHEMA or row.get("contract_id") != CONTRACT_ID:
        raise ContractError("teacher review schema/contract mismatch")
    role = require_enum(row.get("role"), ROLES, "teacher review.role")
    if row.get("teacher_id") != "codex_visual_teacher_provisional_v1":
        raise ContractError("teacher identity mismatch")
    if row.get("reviewer_surface") != "codex_interactive_visual_model":
        raise ContractError("reviewer surface mismatch")
    if row.get("reproducibility_class") != "surface_snapshot_not_weight_pinned":
        raise ContractError("teacher reproducibility limitation must be explicit")
    if row.get("bundle_manifest_sha256") != bundle_sha256:
        raise ContractError("teacher review is bound to another bundle")
    artifact = bundle["review_artifacts"][role]
    if row.get("input_inventory_sha256") != artifact["input_inventory_sha256"]:
        raise ContractError("teacher review input inventory mismatch")
    if row.get("prompt_sha256") != artifact["prompt_sha256"] or row.get("output_schema_sha256") != artifact["output_schema_sha256"]:
        raise ContractError("teacher review prompt/schema binding mismatch")
    if not isinstance(row.get("round"), int) or not 1 <= row["round"] <= 3:
        raise ContractError("teacher review round must be 1..3")
    require_enum(row.get("route_valid"), ROUTE_VALIDITY, "teacher review.route_valid")
    require_enum(row.get("overall_risk"), RISK_LEVELS, "teacher review.overall_risk")
    reasons = row.get("abstain_reasons")
    if not isinstance(reasons, list) or any(reason not in ABSTAIN_REASONS for reason in reasons):
        raise ContractError("teacher review abstain reasons are invalid")
    frame_ids = {frame["frame_id"] for frame in artifact["frames"]}
    events = row.get("events")
    if not isinstance(events, list):
        raise ContractError("teacher review events must be an array")
    for index, event in enumerate(events):
        validate_event(event, frame_ids=frame_ids, where=f"teacher review event {index}")
    if row["overall_risk"] == "unknown" and not reasons:
        raise ContractError("unknown review must disclose an abstain reason")
    if row["overall_risk"] == "none" and events:
        raise ContractError("none review cannot contain risk events")
    require_false_flags(row, "teacher review")
    return row


def majority(values: Iterable[str]) -> tuple[str | None, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None, 0
    selected, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return (selected if count >= 2 else None), count
