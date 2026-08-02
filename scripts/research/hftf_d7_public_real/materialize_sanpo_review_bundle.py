#!/usr/bin/env python3
"""Materialize a model-blind SANPO-Real review batch.

The batch is an input-only artifact.  It downloads the exact provider objects
for selected contiguous RGB+depth windows, verifies provider MD5 values, and
creates separate RGB, geometry, and counterexample role manifests.  It never
writes a review decision, event bucket, phase interval, or admission result.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


MEDIA = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o"
RGB_ROLES = ("RGB_REVIEWER_A", "RGB_REVIEWER_B", "RGB_REVIEWER_C")
GEOMETRY_ROLE = "GEOMETRY_EVIDENCE_REVIEWER"
COUNTEREXAMPLE_ROLE = "COUNTEREXAMPLE_REVIEWER"
ROLES = (*RGB_ROLES, GEOMETRY_ROLE, COUNTEREXAMPLE_ROLE)
ALLOWED_EVENT_BUCKETS = (
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
)
INSTRUCTIONS = (
    "Review only the evidence in this role's bundle. Do not use detector, "
    "HFTF, model, trigger, or another reviewer's output. Select one fixed "
    "event bucket or NOT_EVALUABLE. Use NOT_EVALUABLE when the view, timing, "
    "source binding, or event phase is insufficient."
)


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _download_object(item: dict[str, Any], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item.get("size", -1))
    expected_md5 = str(item.get("md5Hash") or "")
    if destination.exists():
        if destination.stat().st_size != expected_size or _md5_base64(destination) != expected_md5:
            raise ContractError(f"existing SANPO review media does not match provider metadata: {destination}")
    else:
        encoded = urllib.parse.quote(str(item["name"]), safe="")
        request = urllib.request.Request(
            f"{MEDIA}/{encoded}?alt=media",
            headers={"User-Agent": "blindassist-hftf-d7/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        except OSError as exc:
            raise ContractError(f"SANPO review media download failed: {item['name']}: {exc}") from exc
    actual_md5 = _md5_base64(destination)
    if actual_md5 != expected_md5:
        raise ContractError(f"SANPO review media MD5 mismatch: {destination}")
    return {
        "remote_name": item["name"],
        "generation": item.get("generation"),
        "provider_md5_base64": expected_md5,
        "size": expected_size,
        "local_path": str(destination.resolve()),
        "local_sha256": sha256_file(destination),
        "md5_verified": True,
    }


def _object_maps(record: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    media = record.get("media") if isinstance(record.get("media"), dict) else {}
    raw = media.get("objects_by_kind") if isinstance(media.get("objects_by_kind"), dict) else {}
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for kind in ("rgb", "depth", "mask"):
        values = raw.get(kind) if isinstance(raw.get(kind), list) else []
        mapping: dict[int, dict[str, Any]] = {}
        for item in values:
            name = str(item.get("name") or "")
            filename = Path(name).name
            index_text = filename.split(".", 1)[0]
            if not index_text.isdigit():
                raise ContractError(f"invalid SANPO provider frame name: {name}")
            mapping[int(index_text)] = item
        result[kind] = mapping
    return result


def _select_candidates(
    rows: list[dict[str, Any]],
    *,
    count: int,
    session_count: int,
    session_offset: int = 0,
    allowed_sessions: set[str] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ContractError("--count must be positive")
    if session_offset < 0:
        raise ContractError("--session-offset must be non-negative")
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_id") != "SANPO-Real":
            continue
        raw_session = str(row.get("source_metadata", {}).get("raw_source_session_id") or row.get("source_id") or "")
        if raw_session and (allowed_sessions is None or raw_session in allowed_sessions):
            by_session[raw_session].append(row)
    ordered_sessions = sorted(by_session)[session_offset:]
    if session_count > 0:
        ordered_sessions = ordered_sessions[:session_count]
    selected: list[dict[str, Any]] = []
    for raw_session in ordered_sessions:
        for row in sorted(by_session[raw_session], key=lambda item: (int(item.get("start_frame_index", -1)), str(item.get("candidate_id")))):
            selected.append(row)
            if len(selected) >= count:
                return selected
    if len(selected) < count:
        raise ContractError(f"requested {count} SANPO candidates but only {len(selected)} fit the session selection")
    return selected


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_candidate_media_root(batch_root: Path, raw_session: str, candidate_id: str) -> Path:
    if not raw_session or not candidate_id:
        raise ContractError("missing SANPO review media identity")
    return batch_root / "source_media" / raw_session / candidate_id


def _provider_kind(name: str) -> str:
    if "/video_frames/" in name:
        return "rgb"
    if "/depth_maps/" in name:
        return "depth"
    if "/segmentation_masks/" in name:
        return "mask"
    if name.endswith(".csv"):
        return "pose"
    return "aux"


def _provider_destination(batch_root: Path, item: dict[str, Any]) -> Path:
    name = str(item["name"])
    filename = Path(name).name
    return batch_root / "provider_cache" / _provider_kind(name) / f"{_sha256_text(name)[:24]}_{filename}"


POSE_NUMERIC_FIELDS = (
    "pos_x",
    "pos_y",
    "pos_z",
    "q_x",
    "q_y",
    "q_z",
    "q_w",
)


def _read_pose_csv(path: Path, *, expected_rows: int) -> list[dict[str, str]] | None:
    """Read a SANPO pose CSV only when its row cardinality is source-closed."""

    if not path.is_file() or expected_rows <= 0:
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not set(POSE_NUMERIC_FIELDS).issubset(fieldnames):
                return None
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return None
    if len(rows) != expected_rows:
        return None
    return rows


def _pose_value(row: dict[str, str]) -> dict[str, Any] | None:
    values: dict[str, float] = {}
    for field in POSE_NUMERIC_FIELDS:
        value = row.get(field)
        try:
            parsed = float(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            return None
        if parsed is None or not math.isfinite(parsed):
            return None
        values[field] = parsed
    return {
        "tracking_state": row.get("tracking_state"),
        "position": {field: values[field] for field in ("pos_x", "pos_y", "pos_z")},
        "orientation": {field: values[field] for field in ("q_x", "q_y", "q_z", "q_w")},
    }


def _bind_pose_rows(
    *,
    pose_path: Path,
    fixed_pose_path: Path | None,
    expected_rows: int,
    frame_indices: list[int],
) -> tuple[str, dict[int, dict[str, Any]], str]:
    """Bind source pose rows to source frame indices, or fail closed.

    SANPO publishes frame-named media and a pose CSV with one row per source
    frame.  The binding is admitted only when the complete media frame count
    equals the CSV data-row count and every requested row has finite pose
    values.  No capture timestamp is inferred by this helper.
    """

    pose_rows = _read_pose_csv(pose_path, expected_rows=expected_rows)
    fixed_rows = _read_pose_csv(fixed_pose_path, expected_rows=expected_rows) if fixed_pose_path else None
    if pose_rows is None and fixed_rows is None:
        return "NOT_EVALUABLE", {}, "pose_csv_row_count_or_schema_mismatch"
    bound: dict[int, dict[str, Any]] = {}
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= expected_rows:
            return "NOT_EVALUABLE", {}, "requested_frame_outside_pose_row_range"
        row: dict[str, Any] = {"frame_index": frame_index}
        if pose_rows is not None:
            value = _pose_value(pose_rows[frame_index])
            if value is None:
                return "NOT_EVALUABLE", {}, "camera_pose_row_has_nonfinite_value"
            row["camera_pose"] = value
        if fixed_rows is not None:
            value = _pose_value(fixed_rows[frame_index])
            if value is None:
                return "NOT_EVALUABLE", {}, "fixed_camera_pose_row_has_nonfinite_value"
            row["fixed_camera_pose"] = value
        bound[frame_index] = row
    return "FRAME_INDEX_ROW_KEYED", bound, "complete_frame_count_equals_pose_data_rows"


def _download_required_items(
    *,
    batch_root: Path,
    items: list[dict[str, Any]],
    workers: int,
) -> dict[str, dict[str, Any]]:
    if workers <= 0:
        raise ContractError("--workers must be positive")
    ordered = sorted(items, key=lambda value: str(value["name"]))
    downloaded: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sanpo-review") as pool:
        futures = {
            pool.submit(_download_object, item, _provider_destination(batch_root, item)): str(item["name"])
            for item in ordered
        }
        try:
            for future in as_completed(futures):
                name = futures[future]
                downloaded[name] = future.result()
        except Exception:
            for pending in futures:
                pending.cancel()
            raise
    return {name: downloaded[name] for name in sorted(downloaded)}


def _contact_sheet(ffmpeg: Path, frame_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "15",
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-vf",
        "scale=320:-2,tile=6x10:padding=6:margin=6",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file():
        raise ContractError(f"SANPO contact-sheet creation failed: {result.stderr.strip()}")


def _review_input(
    candidate: dict[str, Any],
    *,
    batch_id: str,
    role: str,
    index: int,
    contact_sheet: Path | None,
    temporal_manifest: Path,
    geometry_manifest: Path | None,
) -> dict[str, Any]:
    raw_session = str(candidate.get("source_metadata", {}).get("raw_source_session_id") or candidate.get("source_id") or "")
    row = {
        "schema": "hftf_d7_public_real_review_input_v1",
        "record_kind": "REVIEW_INPUT",
        "batch_id": batch_id,
        "review_role": role,
        "review_index": index,
        "review_input_id": stable_id("d7review-input", batch_id, role, str(candidate["candidate_id"])),
        "candidate_id": candidate["candidate_id"],
        "dataset_id": "SANPO-Real",
        "source_session_token": _sha256_text(raw_session),
        "window_start_frame_index": candidate.get("start_frame_index"),
        "window_end_frame_index": candidate.get("end_frame_index"),
        "window_start_timestamp_ns": None,
        "window_end_timestamp_ns": None,
        "nominal_start_time_ns": candidate.get("nominal_start_time_ns"),
        "nominal_end_time_ns": candidate.get("nominal_end_time_ns"),
        "timestamp_semantics": "DERIVED_RELATIVE_NOMINAL",
        "allowed_event_buckets": list(ALLOWED_EVENT_BUCKETS),
        "instructions": INSTRUCTIONS,
        "model_output_visible": False,
        "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY" if role == GEOMETRY_ROLE else "RGB_ONLY",
        "rgb_included": role != GEOMETRY_ROLE,
        "native_geometry_included": role == GEOMETRY_ROLE,
        "contact_sheet_path": str(contact_sheet.resolve()) if contact_sheet is not None else None,
        "contact_sheet_sha256": sha256_file(contact_sheet) if contact_sheet is not None else None,
        "temporal_manifest_path": str(temporal_manifest.resolve()),
        "temporal_manifest_sha256": sha256_file(temporal_manifest),
        "native_geometry_path": str(geometry_manifest.resolve()) if geometry_manifest is not None else None,
        "native_geometry_sha256": sha256_file(geometry_manifest) if geometry_manifest is not None else None,
    }
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    candidate_path = Path(args.candidate_artifact).resolve()
    inventory_path = Path(args.inventory).resolve()
    ffmpeg = Path(args.ffmpeg_path).resolve()
    if not candidate_path.is_file() or not inventory_path.is_file():
        raise ContractError("SANPO candidate artifact and span inventory are required")
    if not ffmpeg.is_file():
        raise ContractError(f"ffmpeg not found: {ffmpeg}")
    if args.count <= 0 or args.session_count <= 0:
        raise ContractError("--count and --session-count must be positive")
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    if batch_root.exists():
        raise ContractError(f"review batch already exists; refusing overwrite: {batch_root}")
    candidates = load_jsonl(candidate_path)
    inventory = load_json(inventory_path)
    inventory_records = inventory.get("records") if isinstance(inventory, dict) else None
    if not isinstance(inventory_records, list):
        raise ContractError("SANPO span inventory has no records")
    by_raw_session = {str(record.get("source_session_id")): record for record in inventory_records}
    complete_sessions = {
        str(record.get("source_session_id"))
        for record in inventory_records
        if int(
            (record.get("media") if isinstance(record.get("media"), dict) else {}).get(
                "complete_frame_count", 0
            )
            or 0
        ) > 0
    }
    selected = _select_candidates(
        candidates,
        count=args.count,
        session_count=args.session_count,
        session_offset=args.session_offset,
        allowed_sessions=complete_sessions if args.require_complete_media else None,
    )
    batch_root.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        (batch_root / role).mkdir(parents=True, exist_ok=False)
    (batch_root / "manifests").mkdir(parents=True, exist_ok=False)
    (batch_root / "staging").mkdir(parents=True, exist_ok=False)

    required_provider_items: dict[str, dict[str, Any]] = {}
    candidate_context: list[dict[str, Any]] = []
    for candidate in selected:
        raw_session = str(candidate.get("source_metadata", {}).get("raw_source_session_id") or candidate.get("source_id") or "")
        record = by_raw_session.get(raw_session)
        if record is None:
            raise ContractError(f"SANPO span record missing for candidate: {candidate['candidate_id']}")
        maps = _object_maps(record)
        start = int(candidate.get("start_frame_index", -1))
        end = int(candidate.get("end_frame_index", -1))
        if start < 0 or end < start or end - start + 1 != int(candidate.get("frame_count", 0)):
            raise ContractError(f"invalid SANPO candidate frame interval: {candidate['candidate_id']}")
        frames = list(range(start, end + 1))
        for index in frames:
            for kind in ("rgb", "depth"):
                item = maps[kind].get(index)
                if item is None:
                    raise ContractError(f"SANPO candidate lacks {kind} provider frame: {candidate['candidate_id']}:{index}")
                required_provider_items[str(item["name"])] = item
            mask = maps["mask"].get(index)
            if mask is not None:
                required_provider_items[str(mask["name"])] = mask
        auxiliary = record.get("auxiliary") if isinstance(record.get("auxiliary"), dict) else {}
        for kind in ("intrinsics", "pose"):
            values = auxiliary.get(kind) if isinstance(auxiliary.get(kind), list) else []
            for item in values:
                required_provider_items[str(item["name"])] = item
        candidate_context.append({"candidate": candidate, "record": record, "maps": maps, "frames": frames})
    estimated_bytes = sum(int(item.get("size", 0) or 0) for item in required_provider_items.values())
    if estimated_bytes > args.max_bytes:
        raise ContractError(f"bounded SANPO review batch exceeds max-bytes: {estimated_bytes} > {args.max_bytes}")

    # Keep provider downloads bounded and deterministic.  Each worker still
    # performs the full size and provider-MD5 verification in _download_object;
    # concurrency only removes avoidable network/IO serialization.
    downloaded = _download_required_items(
        batch_root=batch_root,
        items=list(required_provider_items.values()),
        workers=args.workers,
    )

    manifest_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_receipts: dict[str, dict[str, Any]] = {}
    pose_binding_cache: dict[str, tuple[str, dict[int, dict[str, Any]], str]] = {}
    for index, context in enumerate(candidate_context):
        candidate = context["candidate"]
        record = context["record"]
        maps = context["maps"]
        frames = context["frames"]
        raw_session = str(record["source_session_id"])
        candidate_id = str(candidate["candidate_id"])
        if raw_session not in pose_binding_cache:
            auxiliary = record.get("auxiliary") if isinstance(record.get("auxiliary"), dict) else {}
            pose_items = auxiliary.get("pose") if isinstance(auxiliary.get("pose"), list) else []
            camera_pose_item = next(
                (item for item in pose_items if "fixed_camera_poses.csv" not in str(item.get("name"))),
                None,
            )
            fixed_pose_item = next(
                (item for item in pose_items if "fixed_camera_poses.csv" in str(item.get("name"))),
                None,
            )
            expected_rows = int(
                (record.get("media") if isinstance(record.get("media"), dict) else {}).get(
                    "complete_frame_count", 0
                )
                or 0
            )
            pose_binding_cache[raw_session] = _bind_pose_rows(
                pose_path=Path(downloaded[str(camera_pose_item["name"])] ["local_path"])
                if camera_pose_item is not None and str(camera_pose_item.get("name")) in downloaded
                else Path(),
                fixed_pose_path=Path(downloaded[str(fixed_pose_item["name"])] ["local_path"])
                if fixed_pose_item is not None and str(fixed_pose_item.get("name")) in downloaded
                else None,
                expected_rows=expected_rows,
                frame_indices=list(range(expected_rows)),
            )
        pose_row_binding, pose_by_frame, pose_binding_reason = pose_binding_cache[raw_session]
        media_root = _safe_candidate_media_root(batch_root, raw_session, candidate_id)
        rgb_dir = media_root / "rgb"
        depth_dir = media_root / "depth"
        mask_dir = media_root / "mask"
        rgb_dir.mkdir(parents=True, exist_ok=False)
        depth_dir.mkdir(parents=True, exist_ok=False)
        mask_dir.mkdir(parents=True, exist_ok=False)
        temporal_rows: list[dict[str, Any]] = []
        geometry_rows: list[dict[str, Any]] = []
        for local_index, frame_index in enumerate(frames):
            rgb_item = maps["rgb"][frame_index]
            depth_item = maps["depth"][frame_index]
            mask_item = maps["mask"].get(frame_index)
            rgb_local = media_root / "rgb" / f"frame_{local_index:06d}.png"
            depth_local = media_root / "depth" / f"frame_{local_index:06d}.float16.gz"
            shutil.copy2(downloaded[str(rgb_item["name"])] ["local_path"], rgb_local)
            shutil.copy2(downloaded[str(depth_item["name"])] ["local_path"], depth_local)
            mask_local: Path | None = None
            if mask_item is not None:
                mask_local = media_root / "mask" / f"frame_{local_index:06d}.png"
                shutil.copy2(downloaded[str(mask_item["name"])] ["local_path"], mask_local)
            temporal_rows.append({
                "frame_index": frame_index,
                "nominal_time_ns": round(frame_index * 1_000_000_000 / args.fps),
                "rgb_path": str(rgb_local.resolve()),
                "rgb_sha256": sha256_file(rgb_local),
                "depth_path": str(depth_local.resolve()),
                "depth_sha256": sha256_file(depth_local),
                "mask_path": str(mask_local.resolve()) if mask_local is not None else None,
                "mask_sha256": sha256_file(mask_local) if mask_local is not None else None,
                "capture_timestamp_authoritative": False,
                "pose_row_binding": pose_row_binding,
                "rgb_depth_mask_binding": "INDEX_KEYED",
            })
            geometry_row: dict[str, Any] = {
                "frame_index": frame_index,
                "nominal_time_ns": round(frame_index * 1_000_000_000 / args.fps),
                "depth_path": str(depth_local.resolve()),
                "depth_sha256": sha256_file(depth_local),
                "mask_path": str(mask_local.resolve()) if mask_local is not None else None,
                "mask_sha256": sha256_file(mask_local) if mask_local is not None else None,
                "pose_row_binding": pose_row_binding,
            }
            if pose_row_binding == "FRAME_INDEX_ROW_KEYED":
                geometry_row["pose"] = pose_by_frame[frame_index]
            geometry_rows.append(geometry_row)
        contact_sheet = batch_root / "staging" / candidate_id / "contact_sheet.jpg"
        _contact_sheet(ffmpeg, rgb_dir, contact_sheet)
        temporal_path = batch_root / "staging" / candidate_id / "temporal_manifest.jsonl"
        write_jsonl(temporal_path, temporal_rows)
        auxiliary = record.get("auxiliary") if isinstance(record.get("auxiliary"), dict) else {}
        source_native_fields = ["depth", "intrinsics"]
        if pose_row_binding == "FRAME_INDEX_ROW_KEYED":
            source_native_fields.append("pose")
        if any(row["mask_path"] for row in geometry_rows):
            source_native_fields.append("segmentation")
        missing_source_native_fields = ["capture_timestamp"]
        if pose_row_binding != "FRAME_INDEX_ROW_KEYED":
            missing_source_native_fields.append("pose_row_binding")
        geometry_path = batch_root / "staging" / candidate_id / "native_geometry.json"
        write_json(geometry_path, {
            "schema": "hftf_d7_public_real_sanpo_geometry_review_input_v1",
            "record_kind": "REVIEW_INPUT",
            "dataset_id": "SANPO-Real",
            "candidate_id": candidate_id,
            "source_session_token": _sha256_text(raw_session),
            "camera": record.get("camera"),
            "view": record.get("view"),
            "source_native_fields": source_native_fields,
            "missing_source_native_fields": missing_source_native_fields,
            "capture_timestamp_authoritative": False,
            "pose_row_binding": pose_row_binding,
            "pose_binding_rule": pose_binding_reason,
            "intrinsics_objects": [downloaded[str(item["name"])] for item in auxiliary.get("intrinsics", []) if str(item.get("name")) in downloaded],
            "pose_objects": [downloaded[str(item["name"])] for item in auxiliary.get("pose", []) if str(item.get("name")) in downloaded],
            "frames": geometry_rows,
            "model_output_visible": False,
            "instructions": "Use only source-native depth, segmentation when present, intrinsics, and pose metadata. Frame-index pose binding is source-native row pairing only; capture timestamp remains non-authoritative. Missing binding is NOT_EVALUABLE, never a negative.",
        })
        source_receipts[raw_session] = {
            "source_session_id": raw_session,
            "candidate_id": candidate_id,
            "provider_object_count": sum(1 for name in required_provider_items if name.startswith(str(record.get("session_prefix")))) if record.get("session_prefix") else 0,
            "candidate_media_bytes": sum(int(item.get("size", 0) or 0) for item in required_provider_items.values()),
            "media_hashes_verified": True,
        }
        role_contact_paths: dict[str, Path] = {}
        for role in (*RGB_ROLES, COUNTEREXAMPLE_ROLE):
            destination = batch_root / role / "contact_sheets" / f"{candidate_id}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(contact_sheet, destination)
            role_contact_paths[role] = destination
        role_temporal: dict[str, Path] = {}
        for role in (*RGB_ROLES, COUNTEREXAMPLE_ROLE):
            destination = batch_root / role / "temporal_manifests" / f"{candidate_id}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporal_path, destination)
            role_temporal[role] = destination
        geometry_manifest = batch_root / GEOMETRY_ROLE / "native_geometry" / f"{candidate_id}.json"
        geometry_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(geometry_path, geometry_manifest)
        geometry_temporal = batch_root / GEOMETRY_ROLE / "temporal_manifests" / f"{candidate_id}.jsonl"
        geometry_temporal.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporal_path, geometry_temporal)
        for role in (*RGB_ROLES, COUNTEREXAMPLE_ROLE):
            manifest_rows[role].append(_review_input(
                candidate,
                batch_id=args.batch_id,
                role=role,
                index=index,
                contact_sheet=role_contact_paths[role],
                temporal_manifest=role_temporal[role],
                geometry_manifest=None,
            ))
        manifest_rows[GEOMETRY_ROLE].append(_review_input(
            candidate,
            batch_id=args.batch_id,
            role=GEOMETRY_ROLE,
            index=index,
            contact_sheet=None,
            temporal_manifest=geometry_temporal,
            geometry_manifest=geometry_manifest,
        ))

    manifest_paths: dict[str, Path] = {}
    for role in ROLES:
        path = batch_root / "manifests" / f"{role}.jsonl"
        write_jsonl(path, manifest_rows[role])
        manifest_paths[role] = path
    bundle_manifest = {
        "schema": "hftf_d7_public_real_review_bundle_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_ISOLATED_REVIEW",
        "dataset_id": "SANPO-Real",
        "candidate_ids": [str(row["candidate_id"]) for row in selected],
        "candidate_count": len(selected),
        "review_roles": list(ROLES),
        "roles": {
            role: {
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "row_count": len(manifest_rows[role]),
                "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY" if role == GEOMETRY_ROLE else "RGB_ONLY",
            }
            for role, path in manifest_paths.items()
        },
        "candidate_artifact": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "inventory_artifact": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "source_media_receipts": sorted(source_receipts.values(), key=lambda row: str(row["candidate_id"])),
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "final_adjudication_written": False,
        "notes": [
            "RGB roles receive only RGB contact sheets and temporal frame manifests.",
            "Geometry role receives only source-native depth/mask/calibration/pose evidence and no RGB.",
            "Counterexample role receives RGB evidence and a dedicated counterexample instruction.",
            "Relative nominal frame times are not capture-authoritative.",
        ],
    }
    bundle_path = batch_root / "bundle_manifest.json"
    write_json(bundle_path, bundle_manifest)
    receipt_path = root / "receipts" / f"review_bundle_receipt_{args.batch_id}.json"
    if receipt_path.exists():
        raise ContractError(f"review bundle receipt already exists: {receipt_path}")
    receipt = {
        "schema": "hftf_d7_public_real_review_bundle_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": bundle_manifest["generated_at_utc"],
        "status": "READY_FOR_ISOLATED_REVIEW",
        "output_root": str(root),
        "batch_root": str(batch_root),
        "dataset_id": "SANPO-Real",
        "candidate_count": len(selected),
        "review_roles": list(ROLES),
        "estimated_provider_bytes": estimated_bytes,
        "provider_object_count": len(required_provider_items),
        "bundle_manifest": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        "manifest_files": {role: {"path": str(path), "sha256": sha256_file(path)} for role, path in manifest_paths.items()},
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--session-count", type=int, default=20)
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument(
        "--require-complete-media",
        action="store_true",
        help="select only sessions with a nonzero contiguous RGB/depth/mask intersection in the span inventory",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg-path", default=r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffmpeg.exe")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
