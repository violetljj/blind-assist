#!/usr/bin/env python3
"""Materialize model-blind, role-isolated inputs for a D7 review batch.

This command is deliberately an input-only step.  It never writes a review
decision, event bucket, phase interval, or admission result.  RGB reviewers
receive only contact sheets and a generic taxonomy; the geometry reviewer
receives only source-native geometry; the counterexample reviewer receives a
separate RGB copy.  A fresh batch directory is required so an existing review
bundle cannot be silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


RGB_ROLES = ("RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C")
GEOMETRY_ROLE = "GEOMETRY_EVIDENCE_REVIEWER"
COUNTEREXAMPLE_ROLE = "COUNTEREXAMPLE_REVIEWER"
REVIEW_ROLES = RGB_ROLES + (GEOMETRY_ROLE, COUNTEREXAMPLE_ROLE)
EGOWALK_POSE_RATE_HZ = 5.0

# These fields can encode a discovery selector, a previous label, or a
# promotion decision.  They are intentionally not copied into review input.
MODEL_OR_LABEL_KEYS = {
    "model_hint",
    "candidate_selection",
    "model_output_visible_to_selector",
    "selection_role",
    "truth_status",
    "event_bucket",
    "native_geometry_used_for_selection",
    "parent_event_id",
    "parent_independence_status",
    "required_confirmation_selection",
    "review_state",
    "admission_status",
    "not_evaluable_reason",
}

ALLOWED_EVENT_BUCKETS = [
    "BLOCKING_BODY_POSITIVE",
    "BOUNDARY_LEVEL_CHANGE_POSITIVE",
    "HEAD_HAZARD_POSITIVE",
    "DYNAMIC_INTRUSION_POSITIVE",
    "PARALLEL_STRUCTURE_NEGATIVE",
    "SIDE_OBJECT_NONBLOCKING_NEGATIVE",
    "NORMAL_WALKABLE_NEGATIVE",
    "EGOMOTION_VISUAL_HARD_NEGATIVE",
    "HEAD_NONACTIONABLE_NEGATIVE",
    "NOT_EVALUABLE",
]

GENERIC_REVIEW_INSTRUCTIONS = (
    "Review only the supplied visual/geometry evidence. Do not infer a label "
    "from the sampling method, candidate identifier, source name, or any "
    "model output. Select exactly one allowed event bucket or NOT_EVALUABLE. "
    "Use decision SUPPORT when the supplied evidence supports the selected "
    "bucket, including a negative bucket; use REJECT when the candidate does "
    "not support the proposed bucket and should not be promoted. "
    "Use NOT_EVALUABLE when the view is occluded, the event is ambiguous, the "
    "required phase cannot be established, or required evidence is absent. "
    "A positive requires observable pre, alertable, and passed-clearance phases "
    "and a SUPPORT decision must include all three intervals; a supported "
    "negative requires a continuous negative interval and a SUPPORT decision "
    "must include that interval. Missing optional geometry is not negative "
    "evidence."
)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"required JSONL missing: {path}")
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"JSONL row is not an object: {path}:{line_number}")
            yield row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def _assert_model_blind(value: Any, *, path: str = "$", allow_false_visibility: bool = True) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in MODEL_OR_LABEL_KEYS:
                raise ContractError(f"model/label field leaked into review input: {path}.{key}")
            if key == "model_output_visible" and allow_false_visibility and child is False:
                continue
            _assert_model_blind(child, path=f"{path}.{key}", allow_false_visibility=allow_false_visibility)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_model_blind(child, path=f"{path}[{index}]", allow_false_visibility=allow_false_visibility)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_roles(value: str) -> list[str]:
    roles = [item.strip() for item in value.split(",") if item.strip()]
    if not roles:
        raise ContractError("--roles must contain at least one review role")
    unknown = sorted(set(roles) - set(REVIEW_ROLES))
    if unknown:
        raise ContractError(f"unknown review roles: {unknown}")
    if len(set(roles)) != len(roles):
        raise ContractError("duplicate review role in --roles")
    return roles


def _select_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_id = {str(row.get("candidate_id", "")): row for row in rows}
    if len(by_id) != len(rows) or "" in by_id:
        raise ContractError("candidate index has missing or duplicate candidate_id")
    requested_ids = [item for value in args.candidate_id for item in value.split(",") if item]
    if requested_ids:
        if len(set(requested_ids)) != len(requested_ids):
            raise ContractError("duplicate --candidate-id")
        missing = sorted(set(requested_ids) - set(by_id))
        if missing:
            raise ContractError(f"requested candidate_id not found: {missing[:5]}")
        selected = [by_id[item] for item in requested_ids]
    else:
        selected = rows
        if args.dataset_id:
            selected = [row for row in selected if str(row.get("dataset_id")) == args.dataset_id]
        selected = selected[args.offset :]
        if args.count is not None:
            selected = selected[: args.count]
    if not selected:
        raise ContractError("candidate selection is empty")
    if args.count is not None and args.count <= 0:
        raise ContractError("--count must be positive")
    if args.offset < 0:
        raise ContractError("--offset must be non-negative")
    return selected


def _safe_rgb_path(candidate: dict[str, Any], *, output_root: Path) -> Path:
    candidate_path = str(candidate.get("rgb_local_path") or "").strip()
    if candidate_path:
        path = Path(candidate_path).resolve()
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ContractError(f"candidate RGB path is outside D7 output root: {path}") from exc
        if path.is_file():
            return path
    if str(candidate.get("dataset_id")) == "EgoWalk":
        source_id = str(candidate.get("source_id") or "")
        if not source_id or Path(source_id).name != source_id:
            raise ContractError(f"EgoWalk candidate has unsafe source_id: {source_id!r}")
        path = output_root / "raw" / "egowalk-rgb" / f"{source_id}__rgb.mp4"
        if path.is_file():
            return path.resolve()
    raise ContractError(f"local RGB media is not available for candidate {candidate.get('candidate_id')}")


def _probe_video(path: Path, ffprobe_path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ContractError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ContractError(f"ffprobe found no video stream: {path}")
    return streams[0]


def _rate_hz(value: object) -> float | None:
    """Parse an ffprobe rational frame rate without accepting bad metadata."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            parsed = float(numerator) / denominator_value
        else:
            parsed = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _source_start_times(rows: list[dict[str, Any]]) -> dict[str, int]:
    starts: dict[str, int] = {}
    for row in rows:
        session_id = str(row.get("source_session_id") or "")
        timestamp = row.get("start_timestamp_ns")
        if not session_id or timestamp is None:
            continue
        try:
            value = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid candidate start timestamp: {row.get('candidate_id')}") from exc
        starts[session_id] = min(value, starts.get(session_id, value))
    return starts


def _sample_times(candidate: dict[str, Any], source_start_ns: int, duration_s: float | None) -> list[float]:
    try:
        start = (int(candidate["start_timestamp_ns"]) - source_start_ns) / 1_000_000_000.0
        end = (int(candidate["end_timestamp_ns"]) - source_start_ns) / 1_000_000_000.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"candidate has invalid timestamp range: {candidate.get('candidate_id')}") from exc
    if start < -0.001 or end < start:
        raise ContractError(f"candidate timestamp range is not monotone: {candidate.get('candidate_id')}")
    start = max(0.0, start)
    end = max(start, end)
    values = [max(0.0, start - 0.75), start, start + (end - start) / 2.0, end, end + 0.75]
    if duration_s is not None and duration_s > 0:
        values = [min(max(0.0, duration_s - 0.02), value) for value in values]
    result: list[float] = []
    for value in values:
        if not result or abs(value - result[-1]) > 0.001:
            result.append(value)
    return result


def _sample_egowalk_times(
    candidate: dict[str, Any],
    *,
    video_rate_hz: float,
    video_frame_count: int | None,
    pose_rate_hz: float,
) -> tuple[list[float], list[int]]:
    """Map EgoWalk pose-row ordinals to the extracted video timeline.

    EgoWalk's extracted videos contain one video frame per trajectory row but
    advertise a container rate of 100 Hz while the pose timeline is 5 Hz.
    Timestamp-based seeking therefore collapses later windows onto the end of
    a short video.  D7 review inputs must use the source row/frame binding and
    only then convert the ordinal to the container's playback time.
    """

    try:
        start = int(candidate["start_frame_index"])
        end = int(candidate["end_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"EgoWalk candidate has invalid frame range: {candidate.get('candidate_id')}") from exc
    if start < 0 or end < start:
        raise ContractError(f"EgoWalk candidate frame range is not monotone: {candidate.get('candidate_id')}")
    if not math.isfinite(pose_rate_hz) or pose_rate_hz <= 0:
        raise ContractError("EgoWalk pose rate must be positive")
    if not math.isfinite(video_rate_hz) or video_rate_hz <= 0:
        raise ContractError("EgoWalk video rate must be positive")
    padding = max(1, int(round(0.75 * pose_rate_hz)))
    midpoint = int(round((start + end) / 2.0))
    frame_indices = [start - padding, start, midpoint, end, end + padding]
    if video_frame_count is not None and video_frame_count > 0:
        frame_indices = [min(max(0, value), video_frame_count - 1) for value in frame_indices]
    else:
        frame_indices = [max(0, value) for value in frame_indices]
    deduped: list[int] = []
    for value in frame_indices:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return [value / video_rate_hz for value in deduped], deduped


def _extract_contact_sheet(
    *,
    ffmpeg_path: Path,
    video_path: Path,
    sample_times: list[float],
    staging_dir: Path,
) -> tuple[Path, list[Path]]:
    staging_dir.mkdir(parents=True, exist_ok=False)
    frame_paths: list[Path] = []
    for index, seconds in enumerate(sample_times):
        frame_path = staging_dir / f"frame_{index:02d}.jpg"
        command = [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seconds:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "2",
            str(frame_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not frame_path.is_file():
            detail = result.stderr.strip() or "no frame was written"
            raise ContractError(f"ffmpeg frame extraction failed for {video_path} at {seconds:.3f}s: {detail}")
        frame_paths.append(frame_path)

    sheet_path = staging_dir / "contact_sheet.jpg"
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "1",
        "-i",
        str(staging_dir / "frame_%02d.jpg"),
        "-vf",
        "tile=3x2:padding=8:margin=8",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(sheet_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not sheet_path.is_file():
        detail = result.stderr.strip() or "no contact sheet was written"
        raise ContractError(f"ffmpeg contact-sheet assembly failed for {video_path}: {detail}")
    return sheet_path, frame_paths


def _load_selected_geometry(
    *,
    output_root: Path,
    selected: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected_ids = {
        str(frame_id)
        for candidate in selected
        for frame_id in (candidate.get("frame_ids") if isinstance(candidate.get("frame_ids"), list) else [])
    }
    selected_sessions = {str(candidate.get("source_session_id") or "") for candidate in selected}
    frame_path = output_root / "canonical" / "egowalk_frame_registry.jsonl"
    by_id: dict[str, dict[str, Any]] = {}
    if frame_path.is_file() and selected_ids:
        for row in _iter_jsonl(frame_path):
            frame_id = str(row.get("frame_id") or "")
            if frame_id in selected_ids:
                by_id[frame_id] = row
    missing = sorted(selected_ids - set(by_id))
    if missing:
        raise ContractError(f"selected candidate frames missing from native geometry registry: {missing[:5]}")
    result: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        candidate_id = str(candidate.get("candidate_id"))
        frame_ids = candidate.get("frame_ids") if isinstance(candidate.get("frame_ids"), list) else []
        pose_rows = []
        for frame_id in frame_ids:
            source = by_id.get(str(frame_id))
            if source is None:
                continue
            pose = source.get("pose_optional")
            if not isinstance(pose, dict):
                continue
            pose_rows.append({
                "frame_index": source.get("frame_index"),
                "timestamp_ns": source.get("timestamp_ns"),
                "pose": {
                    key: pose[key]
                    for key in ("cart_x", "cart_y", "cart_z", "quat_x", "quat_y", "quat_z", "quat_w")
                    if key in pose
                },
            })
        result[candidate_id] = {
            "source_native_fields": ["pose"] if pose_rows else [],
            "missing_source_native_fields": ["depth", "segmentation", "tracks", "obstacle_geometry"],
            "pose_rows": pose_rows,
        }
    # A non-EgoWalk pilot may not have a canonical pose file.  That is a
    # legitimate missing-evidence terminal, not a reason to invent geometry.
    for candidate in selected:
        candidate_id = str(candidate.get("candidate_id"))
        result.setdefault(candidate_id, {
            "source_native_fields": [],
            "missing_source_native_fields": ["pose", "depth", "segmentation", "tracks", "obstacle_geometry"],
            "pose_rows": [],
        })
    return result


def _review_common(candidate: dict[str, Any], *, batch_id: str, role: str, index: int, source_start_ns: int) -> dict[str, Any]:
    source_session_id = str(candidate.get("source_session_id") or "")
    source_session_token = _sha256_text(source_session_id)
    return {
        "schema": "hftf_d7_public_real_review_input_v1",
        "record_kind": "REVIEW_INPUT",
        "batch_id": batch_id,
        "review_role": role,
        "review_index": index,
        "review_input_id": stable_id("d7review-input", batch_id, role, str(candidate.get("candidate_id"))),
        "candidate_id": str(candidate.get("candidate_id")),
        "dataset_id": str(candidate.get("dataset_id") or "UNKNOWN"),
        "source_session_token": source_session_token,
        "source_start_timestamp_ns": source_start_ns,
        "window_start_frame_index": candidate.get("start_frame_index"),
        "window_end_frame_index": candidate.get("end_frame_index"),
        "window_start_timestamp_ns": candidate.get("start_timestamp_ns"),
        "window_end_timestamp_ns": candidate.get("end_timestamp_ns"),
        "allowed_event_buckets": ALLOWED_EVENT_BUCKETS,
        "instructions": GENERIC_REVIEW_INSTRUCTIONS,
        "model_output_visible": False,
    }


def _build_rgb_input(
    candidate: dict[str, Any],
    *,
    batch_id: str,
    role: str,
    index: int,
    source_start_ns: int,
    sheet_path: Path,
    sample_times: list[float],
    sample_frame_indices: list[int] | None,
    role_root: Path,
) -> dict[str, Any]:
    destination = role_root / "contact_sheets" / f"{candidate['candidate_id']}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sheet_path, destination)
    row = _review_common(candidate, batch_id=batch_id, role=role, index=index, source_start_ns=source_start_ns)
    row.update({
        "input_scope": "RGB_ONLY",
        "contact_sheet_path": str(destination.resolve()),
        "contact_sheet_sha256": sha256_file(destination),
        "sample_times_seconds_from_source_start": [round(value, 6) for value in sample_times],
        "sample_frame_indices": sample_frame_indices,
        "native_geometry_included": False,
        "counterexample_search_required": role == COUNTEREXAMPLE_ROLE,
    })
    return row


def _build_geometry_input(
    candidate: dict[str, Any],
    *,
    batch_id: str,
    index: int,
    source_start_ns: int,
    geometry: dict[str, Any],
    role_root: Path,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    geometry_path = role_root / "native_geometry" / f"{candidate_id}.json"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "hftf_d7_public_real_geometry_review_input_v1",
        "record_kind": "REVIEW_INPUT",
        "batch_id": batch_id,
        "review_role": GEOMETRY_ROLE,
        "candidate_id": candidate_id,
        "dataset_id": str(candidate.get("dataset_id") or "UNKNOWN"),
        "source_session_token": _sha256_text(str(candidate.get("source_session_id") or "")),
        "source_start_timestamp_ns": source_start_ns,
        "window_start_frame_index": candidate.get("start_frame_index"),
        "window_end_frame_index": candidate.get("end_frame_index"),
        "window_start_timestamp_ns": candidate.get("start_timestamp_ns"),
        "window_end_timestamp_ns": candidate.get("end_timestamp_ns"),
        "source_native_fields": geometry["source_native_fields"],
        "missing_source_native_fields": geometry["missing_source_native_fields"],
        "pose_rows": geometry["pose_rows"],
        "model_output_visible": False,
        "instructions": (
            "Use only the source-native geometry shown here. Do not use RGB, "
            "detector outputs, trigger names, or another reviewer's judgment. "
            "Missing depth, tracks, segmentation, or obstacle geometry is "
            "NOT_EVALUABLE, never a negative."
        ),
    }
    _assert_model_blind(payload)
    write_json(geometry_path, payload)
    return {
        "schema": "hftf_d7_public_real_review_input_v1",
        "record_kind": "REVIEW_INPUT",
        "batch_id": batch_id,
        "review_role": GEOMETRY_ROLE,
        "review_index": index,
        "review_input_id": stable_id("d7review-input", batch_id, GEOMETRY_ROLE, candidate_id),
        "candidate_id": candidate_id,
        "dataset_id": str(candidate.get("dataset_id") or "UNKNOWN"),
        "source_session_token": _sha256_text(str(candidate.get("source_session_id") or "")),
        "native_geometry_path": str(geometry_path.resolve()),
        "native_geometry_sha256": sha256_file(geometry_path),
        "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY",
        "rgb_included": False,
        "allowed_event_buckets": ALLOWED_EVENT_BUCKETS,
        "model_output_visible": False,
        "instructions": payload["instructions"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    batch_root = output_root / "reviews" / "input_bundles" / args.batch_id
    if batch_root.exists():
        raise ContractError(f"review batch already exists; refusing overwrite: {batch_root}")
    if args.count is not None and args.count <= 0:
        raise ContractError("--count must be positive")
    if args.offset < 0:
        raise ContractError("--offset must be non-negative")
    roles = _parse_roles(args.roles)
    default_candidate_path = (output_root / "candidates" / "candidate_index.jsonl").resolve()
    candidate_path = Path(args.candidate_artifact).resolve() if args.candidate_artifact else default_candidate_path
    if not candidate_path.is_file():
        raise ContractError(f"candidate artifact is missing: {candidate_path}")
    candidate_rows = _jsonl_rows(candidate_path)
    selected = _select_candidates(candidate_rows, args)
    if any(str(row.get("dataset_id")) != "EgoWalk" for row in selected):
        raise ContractError("the current review-bundle extractor is limited to EgoWalk RGB/pose pilot inputs")
    # A selection artifact may contain one window per session rather than the
    # full candidate index.  Keep source-time binding anchored to the full
    # canonical index when available, while selecting only from the supplied
    # review artifact.
    source_rows = candidate_rows
    if candidate_path != default_candidate_path:
        source_rows = _jsonl_rows(default_candidate_path)
    source_starts = _source_start_times(source_rows)
    geometry_by_candidate = _load_selected_geometry(output_root=output_root, selected=selected)
    ffmpeg_path = Path(args.ffmpeg_path).resolve()
    ffprobe_path = Path(args.ffprobe_path).resolve()
    if not ffmpeg_path.is_file() or not ffprobe_path.is_file():
        raise ContractError(f"ffmpeg/ffprobe not found: {ffmpeg_path}, {ffprobe_path}")

    batch_root.mkdir(parents=True, exist_ok=False)
    staging_root = batch_root / "staging"
    manifests_root = batch_root / "manifests"
    for role in roles:
        (batch_root / role).mkdir(parents=True, exist_ok=False)
    staging_root.mkdir(parents=True, exist_ok=False)
    manifest_paths: dict[str, Path] = {}
    manifest_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_video_receipts: dict[str, dict[str, Any]] = {}
    try:
        for index, candidate in enumerate(selected):
            candidate_id = str(candidate["candidate_id"])
            source_session_id = str(candidate.get("source_session_id") or "")
            source_start_ns = source_starts.get(source_session_id)
            if source_start_ns is None:
                raise ContractError(f"source session start timestamp missing: {source_session_id}")
            rgb_roles = [role for role in roles if role in RGB_ROLES or role == COUNTEREXAMPLE_ROLE]
            sheet_path: Path | None = None
            sample_times: list[float] = []
            sample_frame_indices: list[int] | None = None
            if rgb_roles:
                video_path = _safe_rgb_path(candidate, output_root=output_root)
                video_key = str(video_path)
                if video_key not in source_video_receipts:
                    probe = _probe_video(video_path, ffprobe_path)
                    duration_s = _float_or_none(probe.get("duration"))
                    frame_count: int | None = None
                    try:
                        parsed_frame_count = int(str(probe.get("nb_frames")))
                    except (TypeError, ValueError):
                        parsed_frame_count = 0
                    if parsed_frame_count > 0:
                        frame_count = parsed_frame_count
                    source_video_receipts[video_key] = {
                        "path": video_key,
                        "sha256": sha256_file(video_path),
                        "bytes": video_path.stat().st_size,
                        "probe": probe,
                        "duration_seconds": duration_s,
                        "frame_count": frame_count,
                        "video_rate_hz": _rate_hz(probe.get("r_frame_rate") or probe.get("avg_frame_rate")),
                    }
                duration_s = source_video_receipts[video_key]["duration_seconds"]
                if str(candidate.get("dataset_id")) == "EgoWalk":
                    video_rate_hz = source_video_receipts[video_key].get("video_rate_hz")
                    if not isinstance(video_rate_hz, (int, float)) or video_rate_hz <= 0:
                        raise ContractError(f"EgoWalk video rate is unavailable: {video_path}")
                    sample_times, sample_frame_indices = _sample_egowalk_times(
                        candidate,
                        video_rate_hz=float(video_rate_hz),
                        video_frame_count=source_video_receipts[video_key].get("frame_count"),
                        pose_rate_hz=EGOWALK_POSE_RATE_HZ,
                    )
                else:
                    sample_times = _sample_times(candidate, source_start_ns, duration_s)
                staging_dir = staging_root / candidate_id
                sheet_path, _ = _extract_contact_sheet(
                    ffmpeg_path=ffmpeg_path,
                    video_path=video_path,
                    sample_times=sample_times,
                    staging_dir=staging_dir,
                )
            for role in roles:
                if role in RGB_ROLES or role == COUNTEREXAMPLE_ROLE:
                    assert sheet_path is not None
                    row = _build_rgb_input(
                        candidate,
                        batch_id=args.batch_id,
                        role=role,
                        index=index,
                        source_start_ns=source_start_ns,
                        sheet_path=sheet_path,
                        sample_times=sample_times,
                        sample_frame_indices=sample_frame_indices,
                        role_root=batch_root / role,
                    )
                else:
                    row = _build_geometry_input(
                        candidate,
                        batch_id=args.batch_id,
                        index=index,
                        source_start_ns=source_start_ns,
                        geometry=geometry_by_candidate[candidate_id],
                        role_root=batch_root / role,
                    )
                _assert_model_blind(row)
                manifest_rows[role].append(row)
        manifests_root.mkdir(parents=True, exist_ok=False)
        for role in roles:
            manifest_path = manifests_root / f"{role}.jsonl"
            _write_jsonl(manifest_path, manifest_rows[role])
            manifest_paths[role] = manifest_path
        bundle_manifest = {
            "schema": "hftf_d7_public_real_review_bundle_v1",
            "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
            "batch_id": args.batch_id,
            "generated_at_utc": utc_now(),
            "status": "READY_FOR_ISOLATED_REVIEW",
            "candidate_count": len(selected),
            "candidate_ids": [str(row["candidate_id"]) for row in selected],
            "candidate_index_sha256": sha256_file(candidate_path),
            "candidate_artifact": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "roles": {
                role: {
                    "manifest_path": str(manifest_paths[role].resolve()),
                    "manifest_sha256": sha256_file(manifest_paths[role]),
                    "row_count": len(manifest_rows[role]),
                    "input_scope": "RGB_ONLY" if role in RGB_ROLES or role == COUNTEREXAMPLE_ROLE else "SOURCE_NATIVE_GEOMETRY_ONLY",
                }
                for role in roles
            },
            "source_video_receipts": list(source_video_receipts.values()),
            "model_output_visible_in_any_input": False,
            "review_assignments_are_not_labels": True,
            "final_adjudication_written": False,
            "notes": [
                "Each role has a separate bundle directory and manifest.",
                "RGB contact sheets are generated from the public extracted EgoWalk RGB receipt only.",
                "EgoWalk RGB uses the frozen pose-row-to-video-ordinal binding; container timestamps are not used as the physical pose timeline.",
                "The staging directory is not a review decision and contains no model output.",
                "This bundle does not authorize training, confirmation, production, or safety claims.",
            ],
        }
        _assert_model_blind(bundle_manifest)
        write_json(batch_root / "bundle_manifest.json", bundle_manifest)
        receipt = {
            "schema": "hftf_d7_public_real_review_bundle_receipt_v1",
            "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
            "run_id": args.run_id,
            "batch_id": args.batch_id,
            "generated_at_utc": utc_now(),
            "status": "READY_FOR_ISOLATED_REVIEW",
            "output_root": str(output_root),
            "batch_root": str(batch_root),
            "candidate_count": len(selected),
            "review_roles": roles,
            "bundle_manifest": {
                "path": str((batch_root / "bundle_manifest.json").resolve()),
                "sha256": sha256_file(batch_root / "bundle_manifest.json"),
            },
            "manifest_files": {
                role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for role, path in manifest_paths.items()
            },
            "source_video_count": len(source_video_receipts),
            "candidate_artifact": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "source_video_sha256": sorted(item["sha256"] for item in source_video_receipts.values()),
            "model_output_visible_in_any_input": False,
            "review_assignments_are_not_labels": True,
            "training_authorized": False,
            "confirmation_authorized": False,
            "production_authorized": False,
        }
        receipt_path = output_root / "receipts" / f"review_bundle_receipt_{args.batch_id}.json"
        if receipt_path.exists():
            raise ContractError(f"review bundle receipt already exists; refusing overwrite: {receipt_path}")
        write_json(receipt_path, receipt)
        return receipt
    except Exception:
        # A partial batch is unsafe to reuse as if it were complete.  Keep the
        # evidence for diagnosis, but the next run must use a new batch ID.
        raise


def parse_args() -> argparse.Namespace:
    default_ffmpeg = r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffmpeg.exe"
    default_ffprobe = r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffprobe.exe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--dataset-id", default="EgoWalk")
    parser.add_argument(
        "--candidate-artifact",
        help="optional model-blind candidate JSONL; selection defaults to the canonical candidate index",
    )
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--roles", default=",".join(REVIEW_ROLES))
    parser.add_argument("--ffmpeg-path", default=default_ffmpeg)
    parser.add_argument("--ffprobe-path", default=default_ffprobe)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
