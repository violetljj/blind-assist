#!/usr/bin/env python3
"""Materialize model-blind SANPO-Real source-coverage candidates.

This command converts a verified SANPO frame-span inventory into a separate
D7 candidate/frame intake package.  It requires contiguous RGB+depth frames
for each window, treats segmentation as optional evidence, and retains the
published frame-rate conversion as relative nominal time only.  It does not
download media, read model output, assign an event bucket other than
``NOT_EVALUABLE``, or merge rows into the top-level D7 package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pipeline import ContractError, canonical_json, load_json, sha256_file, stable_id, utc_now, write_json, write_jsonl


DATASET = "SANPO-Real"
FPS_DEFAULT = 15.0


def _metadata_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _window_starts(run: dict[str, int], *, window_frames: int, stride_frames: int) -> list[int]:
    if window_frames <= 0 or stride_frames <= 0:
        raise ContractError("window_frames and stride_frames must be positive")
    start = int(run["start"])
    end = int(run["end"])
    if int(run.get("count", end - start + 1)) < window_frames:
        return []
    return list(range(start, end - window_frames + 2, stride_frames))


def _required_runs(frame_indices_by_kind: dict[str, Any], *, required: tuple[str, ...]) -> list[dict[str, int]]:
    sets: list[set[int]] = []
    for kind in required:
        values = frame_indices_by_kind.get(kind)
        if not isinstance(values, list):
            return []
        sets.append({int(value) for value in values})
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        return []
    runs: list[dict[str, int]] = []
    start = previous = common[0]
    for value in common[1:]:
        if value != previous + 1:
            runs.append({"start": start, "end": previous, "count": previous - start + 1})
            start = value
        previous = value
    runs.append({"start": start, "end": previous, "count": previous - start + 1})
    return runs


def _object_map(record: dict[str, Any], kind: str) -> dict[int, dict[str, Any]]:
    media = record.get("media")
    if not isinstance(media, dict):
        raise ContractError("SANPO span record has no media object")
    objects_by_kind = media.get("objects_by_kind")
    if not isinstance(objects_by_kind, dict):
        raise ContractError("SANPO span record has no objects_by_kind")
    objects = objects_by_kind.get(kind)
    if not isinstance(objects, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for item in objects:
        if not isinstance(item, dict):
            raise ContractError("SANPO object metadata must be an object")
        name = str(item.get("name") or "")
        marker = f"/{kind}/"
        if kind == "rgb":
            marker = "/video_frames/"
        elif kind == "depth":
            marker = "/depth_maps/"
        elif kind == "mask":
            marker = "/segmentation_masks/"
        if marker not in name:
            raise ContractError(f"SANPO {kind} object has unexpected path: {name}")
        filename = Path(name).name
        index_text = filename.split(".", 1)[0]
        if not index_text.isdigit():
            raise ContractError(f"SANPO {kind} object has invalid frame filename: {name}")
        index = int(index_text)
        if index in result:
            raise ContractError(f"duplicate SANPO {kind} object index: {index}")
        result[index] = item
    return result


def _gs_uri(name: str) -> str:
    if not name:
        raise ContractError("empty SANPO provider object name")
    return f"gs://gresearch/{name}"


def _object_hash(item: dict[str, Any]) -> str:
    md5 = str(item.get("md5Hash") or "")
    if md5:
        return f"md5-base64:{md5}"
    return f"metadata-sha256:{_metadata_sha256(item)}"


def _frame_id(session_id: str, camera: str, view: str, index: int) -> str:
    return stable_id("d7frm", DATASET, session_id, camera, view, index)


def _session_id(raw_session_id: str) -> str:
    return stable_id("d7sess", DATASET, raw_session_id)


def _ancestry_id(raw_session_id: str) -> str:
    return stable_id("d7anc", DATASET, raw_session_id)


def _make_frame_row(
    record: dict[str, Any],
    *,
    raw_session_id: str,
    session_id: str,
    ancestry_group: str,
    index: int,
    rgb: dict[str, Any],
    depth: dict[str, Any],
    mask: dict[str, Any] | None,
    intrinsics: list[dict[str, Any]],
    pose: list[dict[str, Any]],
) -> dict[str, Any]:
    camera = str(record.get("camera"))
    view = str(record.get("view"))
    return {
        "schema": "hftf_d7_public_real_frame_v1",
        "dataset_id": DATASET,
        "source_session_id": session_id,
        "ancestry_group": ancestry_group,
        "frame_id": _frame_id(raw_session_id, camera, view, index),
        "frame_index": index,
        "timestamp_ns": None,
        "rgb_path": _gs_uri(str(rgb["name"])),
        "intrinsics_optional": [_gs_uri(str(item["name"])) for item in intrinsics],
        "pose_optional": None,
        "depth_optional": _gs_uri(str(depth["name"])),
        "segmentation_optional": _gs_uri(str(mask["name"])) if mask is not None else None,
        "tracks_optional": None,
        "provider_revision": "gcs-generation-metadata",
        "source_hash": _object_hash(rgb),
        "source_license": "CC-BY-4.0",
        "source_metadata": {
            "raw_source_session_id": raw_session_id,
            "camera": camera,
            "view": view,
            "depth_source_hash": _object_hash(depth),
            "segmentation_source_hash": _object_hash(mask) if mask is not None else None,
            "pose_sources": [_gs_uri(str(item["name"])) for item in pose],
            "capture_timestamp_authoritative": False,
            "pose_row_binding": "NOT_EVALUABLE",
            "rgb_depth_mask_binding": "INDEX_KEYED",
        },
    }


def _make_candidate(
    record: dict[str, Any],
    *,
    raw_session_id: str,
    session_id: str,
    ancestry_group: str,
    start: int,
    window_frames: int,
    fps: float,
    rgb_map: dict[int, dict[str, Any]],
    depth_map: dict[int, dict[str, Any]],
    mask_map: dict[int, dict[str, Any]],
    intrinsics: list[dict[str, Any]],
    pose: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    camera = str(record.get("camera"))
    view = str(record.get("view"))
    indices = list(range(start, start + window_frames))
    frame_rows: list[dict[str, Any]] = []
    frame_ids: list[str] = []
    for index in indices:
        rgb = rgb_map.get(index)
        depth = depth_map.get(index)
        if rgb is None or depth is None:
            raise ContractError(f"window is not RGB+depth complete: {raw_session_id}:{start}:{index}")
        frame = _make_frame_row(
            record,
            raw_session_id=raw_session_id,
            session_id=session_id,
            ancestry_group=ancestry_group,
            index=index,
            rgb=rgb,
            depth=depth,
            mask=mask_map.get(index),
            intrinsics=intrinsics,
            pose=pose,
        )
        frame_rows.append(frame)
        frame_ids.append(frame["frame_id"])
    first_rgb = rgb_map[start]
    last_rgb = rgb_map[start + window_frames - 1]
    metadata = {
        "raw_source_session_id": raw_session_id,
        "camera": camera,
        "view": view,
        "window_selection": "MODEL_BLIND_UNIFORM_CONTIGUOUS_RGB_DEPTH",
        "window_start_frame": start,
        "window_end_frame": start + window_frames - 1,
        "fps": fps,
        "timestamp_semantics": "DERIVED_RELATIVE_NOMINAL",
        "capture_timestamp_authoritative": False,
        "pose_row_binding": "NOT_EVALUABLE",
        "rgb_depth_mask_binding": "INDEX_KEYED",
        "segmentation_complete": all(index in mask_map for index in indices),
        "segmentation_frame_count": sum(1 for index in indices if index in mask_map),
        "provider_metadata_sha256": str(record.get("media", {}).get("object_metadata_sha256") or ""),
        "source_object_generations": {
            "rgb_start": first_rgb.get("generation"),
            "rgb_end": last_rgb.get("generation"),
            "depth_start": depth_map[start].get("generation"),
            "depth_end": depth_map[start + window_frames - 1].get("generation"),
        },
    }
    source_hash = _metadata_sha256({
        "session": raw_session_id,
        "camera": camera,
        "view": view,
        "start": start,
        "end": start + window_frames - 1,
        "frame_hashes": [frame["source_hash"] for frame in frame_rows],
        "depth_hashes": [frame["source_metadata"]["depth_source_hash"] for frame in frame_rows],
        "span_inventory": metadata["provider_metadata_sha256"],
    })
    candidate_id = stable_id("d7cand", DATASET, raw_session_id, camera, view, start, window_frames, metadata["provider_metadata_sha256"])
    candidate = {
        "schema": "hftf_d7_public_real_candidate_v1",
        "candidate_id": candidate_id,
        "dataset_id": DATASET,
        "source_id": raw_session_id,
        "source_session_id": session_id,
        "ancestry_group": ancestry_group,
        "parent_event_id": stable_id("d7parent", candidate_id),
        "parent_independence_status": "UNVERIFIED",
        "segment_index": 0,
        "start_frame_index": start,
        "end_frame_index": start + window_frames - 1,
        "frame_count": window_frames,
        "frame_ids": frame_ids,
        "start_timestamp_ns": None,
        "end_timestamp_ns": None,
        "nominal_start_time_ns": round(start * 1_000_000_000 / fps),
        "nominal_end_time_ns": round((start + window_frames - 1) * 1_000_000_000 / fps),
        "timestamp_semantics": "DERIVED_RELATIVE_NOMINAL",
        "rgb_uri": _gs_uri(str(first_rgb["name"])).rsplit("/", 1)[0] + "/",
        "geometry_uri": record.get("media", {}).get("object_metadata_sha256"),
        "provider_revision": "gcs-generation-metadata",
        "source_license": "CC-BY-4.0",
        "source_hash": source_hash,
        "candidate_selection": "MODEL_BLIND_UNIFORM_CONTIGUOUS_RGB_DEPTH",
        "required_confirmation_selection": "MODEL_BLIND",
        "model_output_visible_to_selector": False,
        "native_geometry_available": bool(depth_map) and bool(intrinsics) and bool(pose),
        "native_geometry_used_for_selection": False,
        "event_bucket": "NOT_EVALUABLE",
        "truth_status": "NOT_EVALUABLE",
        "source_metadata": metadata,
    }
    return candidate, frame_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    inventory_path = Path(args.inventory).resolve() if args.inventory else root / "raw" / "sanpo-frame-span-inventory-d7-r1-sanpo-frame-spans-chest-left-20260802.json"
    if not inventory_path.is_file():
        raise ContractError(f"SANPO frame-span inventory is required: {inventory_path}")
    if args.fps <= 0 or not math.isfinite(args.fps):
        raise ContractError("--fps must be finite and positive")
    if args.window_frames <= 0 or args.stride_frames <= 0:
        raise ContractError("window and stride frames must be positive")
    inventory = load_json(inventory_path)
    records = inventory.get("records") if isinstance(inventory, dict) else None
    if not isinstance(records, list) or not records:
        raise ContractError("SANPO frame-span inventory has no records")
    requested = {str(value) for value in args.session_id}
    selected_records = [record for record in records if not requested or str(record.get("source_session_id")) in requested]
    if args.max_sessions:
        selected_records = selected_records[: args.max_sessions]
    if not selected_records:
        raise ContractError("SANPO candidate selection is empty")
    candidates: list[dict[str, Any]] = []
    frame_by_id: dict[str, dict[str, Any]] = {}
    sessions: dict[str, dict[str, Any]] = {}
    candidate_by_session: dict[str, int] = {}
    mask_complete_count = 0
    for record in sorted(selected_records, key=lambda item: str(item.get("source_session_id"))):
        raw_session_id = str(record.get("source_session_id") or "")
        session_id = _session_id(raw_session_id)
        ancestry_group = _ancestry_id(raw_session_id)
        camera = str(record.get("camera"))
        view = str(record.get("view"))
        media = record.get("media") if isinstance(record.get("media"), dict) else {}
        frame_indices_by_kind = media.get("frame_indices_by_kind") if isinstance(media.get("frame_indices_by_kind"), dict) else {}
        runs = _required_runs(frame_indices_by_kind, required=("rgb", "depth"))
        rgb_map = _object_map(record, "rgb")
        depth_map = _object_map(record, "depth")
        mask_map = _object_map(record, "mask")
        auxiliary = record.get("auxiliary") if isinstance(record.get("auxiliary"), dict) else {}
        intrinsics = auxiliary.get("intrinsics") if isinstance(auxiliary.get("intrinsics"), list) else []
        pose = auxiliary.get("pose") if isinstance(auxiliary.get("pose"), list) else []
        sessions[session_id] = {
            "schema": "hftf_d7_public_real_session_v1",
            "dataset_id": DATASET,
            "source_session_id": session_id,
            "ancestry_group": ancestry_group,
            "session_root": str(record.get("session_prefix")),
            "data_role": "DEVELOPMENT_CANDIDATE_DISCOVERY",
            "source_license_status": "CC-BY-4.0",
            "source_hashes": [str(media.get("object_metadata_sha256") or "")],
            "history_roles": ["public_gcs_frame_span_inventory"],
            "camera": camera,
            "view": view,
            "raw_source_session_id": raw_session_id,
        }
        session_count = 0
        for run in runs:
            for start in _window_starts(run, window_frames=args.window_frames, stride_frames=args.stride_frames):
                candidate, frames = _make_candidate(
                    record,
                    raw_session_id=raw_session_id,
                    session_id=session_id,
                    ancestry_group=ancestry_group,
                    start=start,
                    window_frames=args.window_frames,
                    fps=args.fps,
                    rgb_map=rgb_map,
                    depth_map=depth_map,
                    mask_map=mask_map,
                    intrinsics=intrinsics,
                    pose=pose,
                )
                candidates.append(candidate)
                session_count += 1
                if bool(candidate["source_metadata"]["segmentation_complete"]):
                    mask_complete_count += 1
                for frame in frames:
                    frame_by_id.setdefault(str(frame["frame_id"]), frame)
                if args.max_candidates and len(candidates) >= args.max_candidates:
                    break
            if args.max_candidates and len(candidates) >= args.max_candidates:
                break
        candidate_by_session[session_id] = session_count
        if args.max_candidates and len(candidates) >= args.max_candidates:
            break
    if not candidates:
        raise ContractError("no contiguous SANPO RGB+depth windows found")
    candidates.sort(key=lambda row: str(row["candidate_id"]))
    frame_rows = sorted(frame_by_id.values(), key=lambda row: (str(row["source_session_id"]), int(row["frame_index"])))
    tag = args.run_id
    candidate_path = root / "candidates" / f"sanpo_candidate_index_{tag}.jsonl"
    frame_path = root / "canonical" / f"sanpo_frame_registry_{tag}.jsonl"
    session_path = root / "manifests" / f"sanpo_session_registry_{tag}.jsonl"
    manifest_path = root / "manifests" / f"sanpo_candidate_manifest_{tag}.json"
    for path in (candidate_path, frame_path, session_path, manifest_path):
        if path.exists():
            raise ContractError(f"SANPO candidate artifact already exists; refusing overwrite: {path}")
    write_jsonl(candidate_path, candidates)
    write_jsonl(frame_path, frame_rows)
    write_jsonl(session_path, sorted(sessions.values(), key=lambda row: str(row["source_session_id"])))
    manifest = {
        "schema": "hftf_d7_public_real_sanpo_candidate_manifest_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "dataset_id": DATASET,
        "camera": args.camera,
        "view": args.view,
        "fps": args.fps,
        "window_frames": args.window_frames,
        "stride_frames": args.stride_frames,
        "selection_authority": "MODEL_BLIND_UNIFORM_CONTIGUOUS_RGB_DEPTH",
        "event_truth_authority": False,
        "candidate_count": len(candidates),
        "frame_count": len(frame_rows),
        "session_count": len(sessions),
        "segmentation_complete_candidate_count": mask_complete_count,
        "candidate_count_by_session": candidate_by_session,
        "candidate_index_path": str(candidate_path),
        "frame_registry_path": str(frame_path),
        "session_registry_path": str(session_path),
        "candidate_index_sha256": sha256_file(candidate_path),
        "frame_registry_sha256": sha256_file(frame_path),
        "session_registry_sha256": sha256_file(session_path),
        "merge_status": "NOT_MERGED_TOP_LEVEL",
        "notes": [
            "This package is source-intake only and is not an event-truth package.",
            "Segmentation is optional; absent segmentation is not negative evidence.",
            "SANPO capture timestamp and pose-row binding remain non-authoritative/NOT_EVALUABLE.",
        ],
    }
    write_json(manifest_path, manifest)
    receipt_path = root / "receipts" / f"sanpo_candidate_receipt_{tag}.json"
    if receipt_path.exists():
        raise ContractError(f"receipt already exists; refusing overwrite: {receipt_path}")
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_candidate_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": manifest["generated_at_utc"],
        "dataset_id": DATASET,
        "camera": args.camera,
        "view": args.view,
        "status": "PUBLIC_SOURCE_COVERAGE_CANDIDATES_MATERIALIZED_NOT_MERGED",
        "candidate_count": len(candidates),
        "frame_count": len(frame_rows),
        "session_count": len(sessions),
        "segmentation_complete_candidate_count": mask_complete_count,
        "selection_authority": manifest["selection_authority"],
        "event_truth_authority": False,
        "candidate_index": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "frame_registry": {"path": str(frame_path), "sha256": sha256_file(frame_path)},
        "session_registry": {"path": str(session_path), "sha256": sha256_file(session_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--inventory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--camera", default="chest")
    parser.add_argument("--view", default="left")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--stride-frames", type=int, default=30)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--session-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
