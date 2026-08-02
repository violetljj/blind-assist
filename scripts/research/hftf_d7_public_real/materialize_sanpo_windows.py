#!/usr/bin/env python3
"""Materialize bounded SANPO-Real candidate/evidence rows.

This is an intake-only materializer. It reads an existing
"sanpo-gcs-inventory.json" plus an explicit JSON/JSONL list of
session/camera/view selections. The selection may optionally contain
"frames", "start_frame"/"end_frame" (inclusive), or
"start_frame"/"frame_count". When no frame bound is present, all frame
indices present in the selected view of the inventory are used.

The output is append-only JSONL. Every selected frame must have RGB, depth,
segmentation, intrinsics, and pose evidence. Missing or ambiguous evidence,
duplicate candidates, an absent FPS, and malformed inputs fail before the
output is changed. "--fps" has no default: the caller must pass the
official source rate explicitly (for example "--fps 15").

Rows intentionally contain only source identity/timing/evidence fields:
"dataset", "session", "camera", "view", "ancestry", "frame",
"timestamp", "timestamp_ns", "nominal_time_ns", "time_semantics",
"capture_timestamp_authoritative", "pose_row_binding", "rgb_depth_mask_binding",
"rgb", "intrinsics", "pose", "depth", "segmentation",
"source", "license", "revision", "hash", and a stable "candidate_id".
Review labels, event buckets, admission decisions, and model output are
neither read into rows nor generated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from pipeline import ContractError, canonical_json, canonical_sha256
except ImportError:  # pragma: no cover - package import fallback
    from .pipeline import ContractError, canonical_json, canonical_sha256


DATASET = "SANPO-Real"
_FRAME_KINDS = frozenset({"rgb", "depth", "segmentation"})
_EVIDENCE_KINDS = ("rgb", "intrinsics", "pose", "depth", "segmentation")
_OUTPUT_KEYS = frozenset(
    {
        "candidate_id",
        "dataset",
        "session",
        "camera",
        "view",
        "ancestry",
        "frame",
        "timestamp",
        "timestamp_ns",
        "nominal_time_ns",
        "time_semantics",
        "capture_timestamp_authoritative",
        "pose_row_binding",
        "rgb_depth_mask_binding",
        "rgb",
        "intrinsics",
        "pose",
        "depth",
        "segmentation",
        "source",
        "license",
        "revision",
        "hash",
    }
)
_FORBIDDEN_KEY_PARTS = (
    "review",
    "label",
    "event_bucket",
    "admission",
    "model_output",
)
_CAMERA_RE = re.compile(r"^camera_[a-z0-9][a-z0-9_-]*$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FRAME_PATH_RE = re.compile(
    r"(?:^|[\\/])(?:video_frames|depth_maps|segmentation_masks|rgb|depth|mask)"
    r"[\\/]([0-9]+)(?:\.[^\\/]+)+$",
    re.IGNORECASE,
)

_KIND_ALIASES = {
    "rgb": "rgb",
    "image": "rgb",
    "video": "rgb",
    "video_frame": "rgb",
    "video_frames": "rgb",
    "color": "rgb",
    "depth": "depth",
    "depth_map": "depth",
    "depth_maps": "depth",
    "segmentation": "segmentation",
    "segmentation_mask": "segmentation",
    "segmentation_masks": "segmentation",
    "mask": "segmentation",
    "masks": "segmentation",
    "intrinsic": "intrinsics",
    "intrinsics": "intrinsics",
    "camera_intrinsics": "intrinsics",
    "camera_params": "intrinsics",
    "calibration": "intrinsics",
    "description": "intrinsics",
    "pose": "pose",
    "camera_pose": "pose",
    "camera_poses": "pose",
    "fixed_camera_poses": "pose",
}


def _kind_alias(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _KIND_ALIASES.get(value.strip().lower().replace("-", "_"))


def _forbidden_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _FORBIDDEN_KEY_PARTS)


def _json_value(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-safe copy while rejecting evidence-role contamination."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if _forbidden_key(key):
                raise ContractError(f"forbidden output field at {path}: {key}")
            result[str(key)] = _json_value(child, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child, path=f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"non-finite value at {path}")
        return value
    raise ContractError(f"non-JSON evidence value at {path}: {type(value).__name__}")


def _text(value: object, field_name: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ContractError(f"{field_name} is required")
    result = str(value).strip()
    if not result:
        raise ContractError(f"{field_name} is required")
    return result


def _optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    result = str(value).strip()
    return result or None


def _normalise_camera(value: object, *, required: bool = True) -> str | None:
    if value is None or value == "":
        if required:
            raise ContractError("camera is required in every explicit selection")
        return None
    result = _text(value, "camera").lower()
    if result in {"chest", "head"}:
        result = f"camera_{result}"
    if not _CAMERA_RE.fullmatch(result):
        raise ContractError(f"unsafe camera value: {value!r}")
    return result


def _normalise_view(value: object, *, required: bool = True) -> str | None:
    if value is None or value == "":
        if required:
            raise ContractError("view is required in every explicit selection")
        return None
    result = _text(value, "view").lower()
    if not _SAFE_COMPONENT_RE.fullmatch(result) or "/" in result or "\\" in result:
        raise ContractError(f"unsafe view value: {value!r}")
    return result


def _frame_index(value: object, field_name: str = "frame") -> int:
    if isinstance(value, bool) or value is None:
        raise ContractError(f"{field_name} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        result = int(value.strip())
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raise ContractError(f"{field_name} must be a non-negative integer")
    if result < 0:
        raise ContractError(f"{field_name} must be non-negative")
    return result


def _frame_from_mapping(value: Mapping[str, Any]) -> int | None:
    for key in ("frame_index", "source_frame_index", "frame", "index"):
        if key not in value:
            continue
        candidate = value[key]
        if isinstance(candidate, Mapping) and "index" in candidate:
            candidate = candidate["index"]
        if candidate in (None, ""):
            continue
        return _frame_index(candidate, key)
    return None


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field_name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ContractError(f"{field_name} must be a finite positive number")
    return result


def _nominal_time_seconds(frame_index: int, fps: float | None) -> float:
    """Calculate non-authoritative relative seconds from frame index and FPS."""

    if fps is None:
        raise ContractError("--fps is required; no FPS default is allowed")
    rate = _number(fps, "--fps")
    index = _frame_index(frame_index)
    return round(index / rate, 12)


def _nominal_time_ns(frame_index: int, fps: float | None) -> int:
    rate = _number(fps, "--fps")
    index = _frame_index(frame_index)
    return round(index * 1_000_000_000 / rate)


def _load_payload(path: Path, description: str) -> Any:
    if not path.is_file():
        raise ContractError(f"{description} does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read {description}: {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"invalid {description} JSON/JSONL at line {line_number}: {exc}"
                ) from exc
        if not rows:
            raise ContractError(f"{description} is empty or invalid JSON")
        return rows


def _rows_from_payload(payload: Any, *, keys: tuple[str, ...], description: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None and any(
            key in payload for key in ("session", "session_id", "source_session_id")
        ):
            rows = [payload]
        if rows is None:
            raise ContractError(
                f"{description} must be a list or contain one of {', '.join(keys)}"
            )
    else:
        raise ContractError(f"{description} must contain JSON objects")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractError(f"{description} row {index} is not an object")
        result.append(dict(row))
    return result


def _object_name(value: Mapping[str, Any]) -> str | None:
    for key in ("name", "object_name", "remote_name", "path", "url", "object"):
        raw_candidate = value.get(key)
        if key == "object" and isinstance(raw_candidate, Mapping):
            continue
        candidate = _optional_text(raw_candidate)
        if candidate:
            return candidate
    return None


def _path_context(name: str) -> tuple[str | None, str | None, int | None]:
    parts = [part for part in re.split(r"[\\/]", name) if part]
    camera: str | None = None
    view: str | None = None
    for part in parts:
        lowered = part.lower()
        if _CAMERA_RE.fullmatch(lowered):
            camera = _normalise_camera(lowered, required=False)
        elif lowered in {"chest", "head"} and camera is None:
            camera = _normalise_camera(lowered, required=False)
        elif lowered in {"left", "right"}:
            view = lowered
    match = _FRAME_PATH_RE.search(name)
    frame = int(match.group(1)) if match else None
    return camera, view, frame


def _looks_like_object(value: object) -> bool:
    return isinstance(value, Mapping) and _object_name(value) is not None


def _explicit_kind(value: Mapping[str, Any]) -> str | None:
    for key in ("kind", "type", "asset_type", "evidence_kind", "role"):
        kind = _kind_alias(value.get(key))
        if kind:
            return kind
    return None


def _kind_for_object(value: Mapping[str, Any], key_hint: str | None) -> str | None:
    explicit = _explicit_kind(value)
    if explicit:
        return explicit
    hinted = _kind_alias(key_hint)
    if hinted:
        return hinted
    name = _object_name(value)
    if not name:
        return None
    lowered = name.lower()
    if "fixed_camera_poses" in lowered or "camera_poses" in lowered:
        return "pose"
    if "segmentation_masks" in lowered or "/mask/" in lowered:
        return "segmentation"
    if "depth_maps" in lowered or "/depth/" in lowered:
        return "depth"
    if "video_frames" in lowered or "/rgb/" in lowered:
        return "rgb"
    if "camera_poses" in lowered or "fixed_camera_poses" in lowered or "pose" in lowered:
        return "pose"
    if (
        "intrinsic" in lowered
        or "camera_params" in lowered
        or lowered == "description.json"
        or lowered.endswith("/description.json")
    ):
        return "intrinsics"
    return None


def _hash_hint(value: Mapping[str, Any]) -> str | None:
    for key in (
        "sha256",
        "sha256_hash",
        "source_sha256",
        "local_sha256",
        "hash",
        "md5Hash",
        "md5",
        "provider_md5_base64",
        "crc32c",
    ):
        if key not in value or value[key] in (None, ""):
            continue
        candidate = value[key]
        if isinstance(candidate, Mapping):
            algorithm = _optional_text(candidate.get("algorithm")) or "hash"
            digest = _optional_text(candidate.get("value"))
            if digest:
                return f"{algorithm}:{digest}"
        else:
            if key in {"sha256", "sha256_hash", "source_sha256", "local_sha256"}:
                return f"sha256:{_text(candidate, key)}"
            if key == "provider_md5_base64":
                return f"md5-base64:{_text(candidate, key)}"
            return f"{key}:{_text(candidate, key)}"
    return None


def _revision_hint(value: Mapping[str, Any], fallback: str | None) -> str | None:
    for key in ("revision", "generation", "metageneration"):
        candidate = _optional_text(value.get(key))
        if candidate:
            return candidate
    return fallback


@dataclass(frozen=True)
class _Evidence:
    kind: str
    camera: str | None
    view: str | None
    frame: int | None
    raw: Any
    object_like: bool
    revision: str | None
    hash_value: str | None
    name: str | None


@dataclass
class _Catalog:
    session: str
    ancestry: str
    license: str
    revision: str | None
    source: dict[str, Any]
    evidence: list[_Evidence] = field(default_factory=list)
    _identities: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    def add(self, evidence: _Evidence) -> None:
        identity_payload = evidence.raw if not evidence.object_like else {
            "name": evidence.name,
            "revision": evidence.revision,
            "hash": evidence.hash_value,
        }
        identity = (
            evidence.kind,
            evidence.camera,
            evidence.view,
            evidence.frame,
            canonical_sha256(_json_value(identity_payload, path="inventory")),
        )
        if identity in self._identities:
            return
        for existing in self.evidence:
            if (
                existing.kind == evidence.kind
                and existing.camera == evidence.camera
                and existing.view == evidence.view
                and existing.frame == evidence.frame
            ):
                raise ContractError(
                    "duplicate inventory evidence slot: "
                    f"{self.session}/{evidence.camera}/{evidence.view}/"
                    f"{evidence.kind}/{evidence.frame}"
                )
        self._identities.add(identity)
        self.evidence.append(evidence)


def _walk_inventory(
    catalog: _Catalog,
    node: Any,
    *,
    key_hint: str | None,
    frame_hint: int | None,
    camera_hint: str | None,
    view_hint: str | None,
    fallback_revision: str | None,
    prefer_fixed_pose: bool = False,
) -> None:
    if isinstance(node, Mapping):
        local_frame = _frame_from_mapping(node)
        frame = local_frame if local_frame is not None else frame_hint
        local_camera = _normalise_camera(
            node.get("camera") or node.get("camera_id") or node.get("camera_name"),
            required=False,
        )
        camera = local_camera or camera_hint
        local_view = _normalise_view(
            node.get("view") or node.get("lens") or node.get("camera_view"),
            required=False,
        )
        view = local_view or view_hint

        if _looks_like_object(node):
            kind = _kind_for_object(node, key_hint)
            if kind:
                name = _object_name(node)
                lowered_name = (name or "").lower()
                # The receipt carries both CSV variants.  Use the corrected
                # fixed-camera pose as the evidence role and retain the raw
                # CSV as a non-selected alternate, instead of treating two
                # pose files as one silently interchangeable slot.
                if prefer_fixed_pose and kind == "pose" and "fixed_camera_poses" not in lowered_name and "camera_poses" in lowered_name:
                    kind = "pose_raw"
                path_camera, path_view, path_frame = _path_context(name or "")
                catalog.add(
                    _Evidence(
                        kind=kind,
                        camera=path_camera or camera,
                        view=path_view or view,
                        frame=path_frame if path_frame is not None else frame,
                        raw=dict(node),
                        object_like=True,
                        revision=_revision_hint(node, fallback_revision),
                        hash_value=_hash_hint(node),
                        name=name,
                    )
                )

        for key, child in node.items():
            child_kind = _kind_alias(key)
            if child_kind in {"intrinsics", "pose"} and not _looks_like_object(child):
                direct_frame = frame
                catalog.add(
                    _Evidence(
                        kind=child_kind,
                        camera=camera,
                        view=view,
                        frame=direct_frame,
                        raw=_json_value(child, path=f"inventory.{key}"),
                        object_like=False,
                        revision=_revision_hint(node, fallback_revision),
                        hash_value=None,
                        name=None,
                    )
                )
            _walk_inventory(
                catalog,
                child,
                key_hint=child_kind,
                frame_hint=frame,
                camera_hint=camera,
                view_hint=view,
                fallback_revision=fallback_revision,
                prefer_fixed_pose=prefer_fixed_pose,
            )
    elif isinstance(node, list):
        for child in node:
            _walk_inventory(
                catalog,
                child,
                key_hint=key_hint,
                frame_hint=frame_hint,
                camera_hint=camera_hint,
                view_hint=view_hint,
                fallback_revision=fallback_revision,
                prefer_fixed_pose=prefer_fixed_pose,
            )


def _inventory_records(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise ContractError("sanpo-gcs-inventory.json must be a JSON object")
    dataset = payload.get("dataset") or payload.get("dataset_id") or DATASET
    if str(dataset) != DATASET:
        raise ContractError(f"inventory dataset must be {DATASET}, got {dataset!r}")
    records_value = payload.get("records")
    if records_value is None:
        records_value = payload.get("sessions")
    if records_value is None and isinstance(payload.get("objects"), list) and payload.get("source_session_id"):
        # A materialized SANPO media receipt is also a valid one-session
        # evidence inventory.  It remains source-intake only; receipt objects
        # do not supply authoritative capture timestamps or pose-row binding.
        records_value = [payload]
    if not isinstance(records_value, list) or not records_value:
        raise ContractError("inventory records must be a non-empty list")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(records_value):
        if not isinstance(value, Mapping):
            raise ContractError(f"inventory record {index} is not an object")
        records.append(dict(value))
    return dict(payload), records


def _build_catalogs(payload: Any) -> tuple[dict[str, Any], dict[str, _Catalog]]:
    inventory, records = _inventory_records(payload)
    root_license = _optional_text(inventory.get("license") or inventory.get("source_license"))
    root_revision = _optional_text(
        inventory.get("revision") or inventory.get("generation") or inventory.get("run_id")
    )
    source: dict[str, Any] = {"provider": "SANPO-GCS"}
    for key in ("schema", "official_url", "gcs_api", "gcs_prefix"):
        value = inventory.get(key)
        if value not in (None, ""):
            source[key] = _json_value(value, path=f"inventory.{key}")

    catalogs: dict[str, _Catalog] = {}
    for index, record in enumerate(records):
        session_value = (
            record.get("session")
            or record.get("session_id")
            or record.get("source_session_id")
        )
        session = _text(session_value, f"inventory record {index} session")
        if session in catalogs:
            raise ContractError(f"duplicate inventory session: {session}")
        license_value = _optional_text(
            record.get("license")
            or record.get("source_license")
            or root_license
        )
        if not license_value:
            raise ContractError(f"license missing for inventory session {session}")
        ancestry = _optional_text(
            record.get("ancestry")
            or record.get("ancestry_group")
            or record.get("source_ancestry_id")
        ) or session
        record_revision = _optional_text(
            record.get("revision")
            or record.get("generation")
            or root_revision
        )
        catalog = _Catalog(
            session=session,
            ancestry=ancestry,
            license=license_value,
            revision=record_revision,
            source=dict(source),
        )
        record_camera = _normalise_camera(
            record.get("camera") or record.get("camera_id") or record.get("camera_name"),
            required=False,
        )
        record_view = _normalise_view(
            record.get("view") or record.get("lens") or record.get("camera_view"),
            required=False,
        )
        prefer_fixed_pose = "fixed_camera_poses" in json.dumps(record, ensure_ascii=False).lower()
        _walk_inventory(
            catalog,
            record,
            key_hint=None,
            frame_hint=None,
            camera_hint=record_camera,
            view_hint=record_view,
            fallback_revision=record_revision,
            prefer_fixed_pose=prefer_fixed_pose,
        )
        catalogs[session] = catalog
    return inventory, catalogs


@dataclass(frozen=True)
class _Plan:
    session: str
    camera: str
    view: str
    ancestry: str | None
    frames: tuple[int, ...] | None


def _selection_frames(row: Mapping[str, Any], index: int) -> tuple[int, ...] | None:
    window = row.get("window")
    merged: dict[str, Any] = dict(row)
    if isinstance(window, Mapping):
        for key, value in window.items():
            merged.setdefault(str(key), value)

    for key in ("frame_indices", "frames"):
        if key in merged:
            value = merged[key]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                value = [value]
            values = tuple(_frame_index(item, f"selection row {index} {key}") for item in value)
            if not values:
                raise ContractError(f"selection row {index} has an empty frame list")
            if len(set(values)) != len(values):
                raise ContractError(f"duplicate frame in selection row {index}")
            return tuple(sorted(values))

    if "frame" in merged and not isinstance(merged["frame"], Mapping):
        return (_frame_index(merged["frame"], f"selection row {index} frame"),)

    frame_range = merged.get("frame_range")
    if isinstance(frame_range, Sequence) and not isinstance(frame_range, (str, bytes)):
        if len(frame_range) != 2:
            raise ContractError(f"selection row {index} frame_range must have two values")
        start = _frame_index(frame_range[0], f"selection row {index} frame_range start")
        end = _frame_index(frame_range[1], f"selection row {index} frame_range end")
        if end < start:
            raise ContractError(f"selection row {index} frame_range is descending")
        return tuple(range(start, end + 1))

    start_value = merged.get("start_frame", merged.get("window_start_frame"))
    end_value = merged.get("end_frame", merged.get("window_end_frame"))
    count_value = merged.get("frame_count")
    if start_value is None and (end_value is not None or count_value is not None):
        raise ContractError(f"selection row {index} needs start_frame with its bound")
    if start_value is not None:
        start = _frame_index(start_value, f"selection row {index} start_frame")
        if count_value is not None:
            if isinstance(count_value, bool):
                raise ContractError(f"selection row {index} frame_count is invalid")
            count = _frame_index(count_value, f"selection row {index} frame_count")
            if count <= 0:
                raise ContractError(f"selection row {index} frame_count must be positive")
            return tuple(range(start, start + count))
        if end_value is not None:
            end = _frame_index(end_value, f"selection row {index} end_frame")
            if end < start:
                raise ContractError(f"selection row {index} frame range is descending")
            return tuple(range(start, end + 1))
        return (start,)
    return None


def _build_plans(payload: Any) -> list[_Plan]:
    rows = _rows_from_payload(
        payload,
        keys=("selections", "session_camera_views", "windows", "records"),
        description="session/camera/view list",
    )
    plans: list[_Plan] = []
    for index, row in enumerate(rows):
        session = _text(
            row.get("session") or row.get("session_id") or row.get("source_session_id"),
            f"selection row {index} session",
        )
        camera = _normalise_camera(
            row.get("camera") or row.get("camera_id") or row.get("camera_name")
        )
        view = _normalise_view(
            row.get("view") or row.get("lens") or row.get("camera_view")
        )
        ancestry = _optional_text(
            row.get("ancestry")
            or row.get("ancestry_group")
            or row.get("source_ancestry_id")
        )
        plans.append(
            _Plan(
                session=session,
                camera=camera or "",
                view=view or "",
                ancestry=ancestry,
                frames=_selection_frames(row, index),
            )
        )
    if not plans:
        raise ContractError("session/camera/view list is empty")
    return plans


def _available_frames(catalog: _Catalog, camera: str, view: str) -> list[int]:
    values: set[int] = set()
    for evidence in catalog.evidence:
        if evidence.kind not in _FRAME_KINDS or evidence.frame is None:
            continue
        if evidence.camera not in (None, camera) or evidence.view not in (None, view):
            continue
        values.add(evidence.frame)
    return sorted(values)


def _matching_evidence(
    catalog: _Catalog,
    *,
    kind: str,
    camera: str,
    view: str,
    frame: int,
) -> _Evidence:
    candidates: list[tuple[int, _Evidence]] = []
    for evidence in catalog.evidence:
        if evidence.kind != kind:
            continue
        if evidence.camera not in (None, camera):
            continue
        if evidence.view not in (None, view):
            continue
        if kind in _FRAME_KINDS:
            if evidence.frame != frame:
                continue
        elif evidence.frame not in (None, frame):
            continue
        score = (
            int(evidence.camera == camera)
            + int(evidence.view == view)
            + int(evidence.frame == frame)
        )
        candidates.append((score, evidence))
    if not candidates:
        raise ContractError(
            f"missing {kind} object for {catalog.session}/{camera}/{view}/frame {frame}"
        )
    best_score = max(score for score, _ in candidates)
    best = [evidence for score, evidence in candidates if score == best_score]
    if len(best) != 1:
        raise ContractError(
            f"ambiguous {kind} evidence for {catalog.session}/{camera}/{view}/frame {frame}"
        )
    return best[0]


def _evidence_ref(evidence: _Evidence, fallback_revision: str | None) -> dict[str, Any]:
    revision = evidence.revision or fallback_revision
    if not revision:
        raise ContractError(f"{evidence.kind} evidence has no revision")
    if evidence.object_like:
        if not evidence.name:
            raise ContractError(f"{evidence.kind} evidence has no object name")
        if not evidence.hash_value:
            raise ContractError(f"{evidence.kind} object has no source hash: {evidence.name}")
        return {
            "object": evidence.name,
            "revision": revision,
            "hash": evidence.hash_value,
        }
    value = _json_value(evidence.raw, path=f"inventory.{evidence.kind}")
    return {
        "value": value,
        "revision": revision,
        "hash": canonical_sha256(value),
    }


def _source_payload(catalog: _Catalog, fps: float) -> dict[str, Any]:
    source = dict(catalog.source)
    source.update(
        {
            "selection": "explicit_session_camera_view",
            "timestamp_unit": "nominal_seconds_from_source_frame_zero",
            "fps": fps,
            "time_semantics": "DERIVED_RELATIVE_NOMINAL",
            "capture_timestamp_authoritative": False,
            "pose_row_binding": "NOT_EVALUABLE",
            "rgb_depth_mask_binding": "INDEX_KEYED",
        }
    )
    return source


def _make_row(
    catalog: _Catalog,
    plan: _Plan,
    frame: int,
    fps: float,
) -> dict[str, Any]:
    refs = {
        kind: _evidence_ref(
            _matching_evidence(
                catalog,
                kind=kind,
                camera=plan.camera,
                view=plan.view,
                frame=frame,
            ),
            catalog.revision,
        )
        for kind in _EVIDENCE_KINDS
    }
    base: dict[str, Any] = {
        "dataset": DATASET,
        "session": plan.session,
        "camera": plan.camera,
        "view": plan.view,
        "ancestry": plan.ancestry or catalog.ancestry,
        "frame": frame,
        "timestamp": {
            "value": _nominal_time_seconds(frame, fps),
            "unit": "seconds_from_source_frame_zero",
            "semantics": "DERIVED_RELATIVE_NOMINAL",
            "authoritative": False,
        },
        "timestamp_ns": None,
        "nominal_time_ns": _nominal_time_ns(frame, fps),
        "time_semantics": "DERIVED_RELATIVE_NOMINAL",
        "capture_timestamp_authoritative": False,
        "pose_row_binding": "NOT_EVALUABLE",
        "rgb_depth_mask_binding": "INDEX_KEYED",
        "rgb": refs["rgb"],
        "intrinsics": refs["intrinsics"],
        "pose": refs["pose"],
        "depth": refs["depth"],
        "segmentation": refs["segmentation"],
        "source": _source_payload(catalog, fps),
        "license": catalog.license,
        "revision": {
            **({"inventory": catalog.revision} if catalog.revision else {}),
            **{kind: refs[kind]["revision"] for kind in _EVIDENCE_KINDS},
        },
    }
    digest = canonical_sha256(base)
    row = {
        **base,
        "candidate_id": f"sanpo-real-{digest[:24]}",
        "hash": {
            "algorithm": "sha256",
            "candidate": digest,
            "evidence": {kind: refs[kind]["hash"] for kind in _EVIDENCE_KINDS},
        },
    }
    _validate_output_row(row)
    return row


def _validate_output_row(row: Mapping[str, Any]) -> None:
    missing = sorted(_OUTPUT_KEYS.difference(row.keys()))
    if missing:
        raise ContractError(f"candidate row missing fields: {', '.join(missing)}")
    extra = sorted(set(row.keys()).difference(_OUTPUT_KEYS))
    if extra:
        raise ContractError(f"candidate row has unsupported fields: {', '.join(extra)}")
    for key in row:
        if _forbidden_key(key):
            raise ContractError(f"candidate row has forbidden field: {key}")
    _json_value(dict(row), path="candidate")
    if row["dataset"] != DATASET:
        raise ContractError("candidate row has the wrong dataset")
    if not isinstance(row["frame"], int) or isinstance(row["frame"], bool) or row["frame"] < 0:
        raise ContractError("candidate frame must be a non-negative integer")
    timestamp = row["timestamp"]
    if not isinstance(timestamp, Mapping):
        raise ContractError("candidate timestamp must carry explicit semantics")
    if timestamp.get("semantics") != "DERIVED_RELATIVE_NOMINAL" or timestamp.get("authoritative") is not False:
        raise ContractError("candidate timestamp must be non-authoritative nominal time")
    if not isinstance(timestamp.get("value"), (int, float)) or isinstance(timestamp.get("value"), bool):
        raise ContractError("candidate timestamp value must be numeric seconds")
    if not math.isfinite(float(timestamp["value"])):
        raise ContractError("candidate timestamp value must be finite")
    if row["timestamp_ns"] is not None:
        raise ContractError("SANPO source timestamp_ns must remain NOT_EVALUABLE")
    if row["time_semantics"] != "DERIVED_RELATIVE_NOMINAL":
        raise ContractError("SANPO time semantics drift")
    if row["capture_timestamp_authoritative"] is not False:
        raise ContractError("SANPO capture timestamp authority drift")
    if row["pose_row_binding"] != "NOT_EVALUABLE":
        raise ContractError("SANPO pose row binding must remain NOT_EVALUABLE")


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    _validate_output_row(row)
    return (
        str(row["dataset"]),
        str(row["session"]),
        str(row["camera"]),
        str(row["view"]),
        int(row["frame"]),
    )


def _read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file():
        raise ContractError(f"output path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read existing output {path}: {exc}") from exc
    if text.lstrip().startswith("["):
        raise ContractError("incremental output must be JSONL, not a JSON array")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid existing output JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ContractError(f"existing output line {line_number} is not an object")
        row = dict(value)
        key = _candidate_key(row)
        if key in seen:
            raise ContractError(f"duplicate existing candidate: {key}")
        seen.add(key)
        rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_separator = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_separator:
            handle.write("\n")
        for row in rows:
            handle.write(canonical_json(dict(row)).decode("utf-8"))


def materialize_windows(
    inventory_path: Path | str,
    selection_path: Path | str,
    output_path: Path | str,
    *,
    fps: float | None,
) -> dict[str, Any]:
    """Validate and append one explicit batch of SANPO-Real evidence rows."""

    if fps is None:
        raise ContractError("--fps is required; no FPS default is allowed")
    rate = _number(fps, "--fps")
    inventory_file = Path(inventory_path)
    selection_file = Path(selection_path)
    output_file = Path(output_path)
    inventory_payload = _load_payload(inventory_file, "sanpo-gcs-inventory.json")
    selection_payload = _load_payload(selection_file, "session/camera/view list")
    _, catalogs = _build_catalogs(inventory_payload)
    plans = _build_plans(selection_payload)
    existing = _read_existing_jsonl(output_file)
    existing_keys = {_candidate_key(row) for row in existing}
    batch_keys: set[tuple[str, str, str, str, int]] = set()
    rows: list[dict[str, Any]] = []

    for plan in plans:
        catalog = catalogs.get(plan.session)
        if catalog is None:
            raise ContractError(f"session is absent from inventory: {plan.session}")
        frames = list(plan.frames) if plan.frames is not None else _available_frames(
            catalog, plan.camera, plan.view
        )
        if not frames:
            raise ContractError(
                f"no inventory frame objects for {plan.session}/{plan.camera}/{plan.view}"
            )
        for frame in frames:
            key = (DATASET, plan.session, plan.camera, plan.view, frame)
            if key in existing_keys or key in batch_keys:
                raise ContractError(f"duplicate candidate rejected: {key}")
            row = _make_row(catalog, plan, frame, rate)
            batch_keys.add(key)
            rows.append(row)

    if not rows:
        raise ContractError("selection produced no candidate rows")
    _append_jsonl(output_file, rows)
    return {
        "status": "APPENDED",
        "fps": rate,
        "rows_appended": len(rows),
        "rows_total": len(existing) + len(rows),
        "output": str(output_file.resolve()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument(
        "--selection",
        "--session-camera-view-list",
        dest="selection",
        required=True,
        type=Path,
        help="explicit JSON/JSONL session/camera/view selection list",
    )
    parser.add_argument("--output", required=True, type=Path, help="append-only JSONL evidence manifest")
    parser.add_argument(
        "--fps",
        required=True,
        type=float,
        help="explicit source FPS; pass the official SANPO rate at the call site",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    return materialize_windows(
        args.inventory,
        args.selection,
        args.output,
        fps=args.fps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
    except ContractError as exc:
        print(json.dumps({"status": "REJECTED", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
