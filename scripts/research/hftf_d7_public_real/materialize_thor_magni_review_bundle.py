#!/usr/bin/env python3
"""Materialize a model-blind THOR-MAGNI review input bundle.

RGB roles receive only contact sheets and source-time manifests.  The geometry
role receives only source-native QTM time, pose, tracks, and calibration
metadata; it receives no RGB path or image.  This command never writes a
decision, event bucket, phase interval, or admission result.
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
from typing import Any

from materialize_review_bundle import _probe_video
from pipeline import ContractError, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


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
    "Review only this role's evidence. Do not use detector, HFTF, model, "
    "trigger, or another reviewer's output. Use NOT_EVALUABLE when the view, "
    "source binding, or required phase is insufficient."
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select(rows: list[dict[str, Any]], *, count: int, session_count: int) -> list[dict[str, Any]]:
    if count <= 0 or session_count <= 0:
        raise ContractError("--count and --session-count must be positive")
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_id") != "THOR-MAGNI":
            continue
        session = str(row.get("source_session_id") or "")
        if session:
            by_session[session].append(row)
    sessions = sorted(by_session)[:session_count]
    ordered = {
        session: sorted(by_session[session], key=lambda row: (int(row.get("start_timestamp_ns", 0)), str(row.get("candidate_id"))))
        for session in sessions
    }
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < count:
        advanced = False
        for session in sessions:
            if index < len(ordered[session]):
                selected.append(ordered[session][index])
                advanced = True
                if len(selected) >= count:
                    return selected
        index += 1
        if not advanced:
            break
    raise ContractError(f"requested {count} THOR-MAGNI candidates but only {len(selected)} fit selection")


def _safe_rgb_path(root: Path, candidate: dict[str, Any]) -> Path:
    uri = str(candidate.get("rgb_uri") or "")
    if not uri or uri.startswith(("http://", "https://", "gs://")):
        raise ContractError(f"THOR-MAGNI candidate lacks a local RGB path: {candidate.get('candidate_id')}")
    path = (root / uri).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"THOR-MAGNI RGB path escapes output root: {path}") from exc
    if not path.is_file():
        raise ContractError(f"THOR-MAGNI RGB media missing: {path}")
    return path


def _dense_sample_times(candidate: dict[str, Any], source_start_ns: int, duration_s: float | None, sample_count: int) -> list[float]:
    if sample_count < 2:
        raise ContractError("--temporal-samples must be at least 2")
    try:
        start = (int(candidate["start_timestamp_ns"]) - source_start_ns) / 1_000_000_000.0
        end = (int(candidate["end_timestamp_ns"]) - source_start_ns) / 1_000_000_000.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid THOR-MAGNI candidate time range: {candidate.get('candidate_id')}") from exc
    if start < -0.001 or end < start:
        raise ContractError(f"non-monotone THOR-MAGNI candidate time range: {candidate.get('candidate_id')}")
    start = max(0.0, start - 0.25)
    end = max(start, end + 0.25)
    if duration_s is not None and duration_s > 0:
        end = min(end, max(0.0, duration_s - 0.02))
        start = min(start, end)
    if end <= start:
        return [round(start, 6)] * sample_count
    step = (end - start) / (sample_count - 1)
    return [round(start + step * index, 6) for index in range(sample_count)]


def _extract_dense_contact_sheet(*, ffmpeg_path: Path, video_path: Path, sample_times: list[float], staging_dir: Path) -> tuple[Path, list[Path]]:
    staging_dir.mkdir(parents=True, exist_ok=False)
    frame_paths: list[Path] = []
    for index, seconds in enumerate(sample_times):
        frame_path = staging_dir / f"frame_{index:03d}.jpg"
        command = [
            str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.6f}",
            "-i", str(video_path), "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "3", str(frame_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not frame_path.is_file():
            raise ContractError(f"THOR-MAGNI frame extraction failed at {seconds:.3f}s: {result.stderr.strip()}")
        frame_paths.append(frame_path)
    columns = 6
    rows = max(1, math.ceil(len(frame_paths) / columns))
    sheet_path = staging_dir / "contact_sheet.jpg"
    command = [
        str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-framerate", "1",
        "-i", str(staging_dir / "frame_%03d.jpg"), "-vf", f"tile={columns}x{rows}:padding=5:margin=5",
        "-frames:v", "1", "-q:v", "3", str(sheet_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not sheet_path.is_file():
        raise ContractError(f"THOR-MAGNI contact-sheet assembly failed: {result.stderr.strip()}")
    return sheet_path, frame_paths


def _load_geometry(frame_path: Path, selected: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ids = {str(frame_id) for candidate in selected for frame_id in candidate.get("frame_ids", [])}
    # The canonical THOR frame registry is large (currently 1.46M rows).  A
    # review bundle needs only the frame IDs referenced by its selected
    # candidates; stream the JSONL and retain those rows instead of building a
    # second in-memory copy of the complete registry for every batch.
    frames: dict[str, dict[str, Any]] = {}
    with frame_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_id = str(row.get("frame_id") or "")
            if frame_id in ids:
                frames[frame_id] = row
    missing = sorted(ids - set(frames))
    if missing:
        raise ContractError(f"THOR-MAGNI frame registry missing selected frame IDs: {missing[:5]}")
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate in selected:
        rows: list[dict[str, Any]] = []
        for frame_id in candidate.get("frame_ids", []):
            source = frames[str(frame_id)]
            rows.append({
                "frame_id": source.get("frame_id"),
                "frame_index": source.get("frame_index"),
                "timestamp_ns": source.get("timestamp_ns"),
                "pose_optional": source.get("pose_optional"),
                "tracks_optional": source.get("tracks_optional"),
                "intrinsics_optional": source.get("intrinsics_optional"),
                "source_metadata": source.get("source_metadata"),
            })
        result[str(candidate["candidate_id"])] = rows
    return result


def _common(candidate: dict[str, Any], *, batch_id: str, role: str, index: int, source_start_ns: int) -> dict[str, Any]:
    return {
        "schema": "hftf_d7_public_real_review_input_v1",
        "record_kind": "REVIEW_INPUT",
        "batch_id": batch_id,
        "review_role": role,
        "review_index": index,
        "review_input_id": stable_id("d7review-input", batch_id, role, str(candidate["candidate_id"])),
        "candidate_id": candidate["candidate_id"],
        "dataset_id": "THOR-MAGNI",
        "source_session_token": _sha256_text(str(candidate["source_session_id"])),
        "source_start_timestamp_ns": source_start_ns,
        "window_start_frame_index": candidate.get("start_frame_index"),
        "window_end_frame_index": candidate.get("end_frame_index"),
        "window_start_timestamp_ns": candidate.get("start_timestamp_ns"),
        "window_end_timestamp_ns": candidate.get("end_timestamp_ns"),
        "timestamp_semantics": candidate.get("timestamp_semantics"),
        "allowed_event_buckets": list(ALLOWED_EVENT_BUCKETS),
        "instructions": INSTRUCTIONS,
        "model_output_visible": False,
    }


def _geometry_row(candidate: dict[str, Any], *, batch_id: str, index: int, source_start_ns: int, geometry_path: Path) -> dict[str, Any]:
    row = _common(candidate, batch_id=batch_id, role=GEOMETRY_ROLE, index=index, source_start_ns=source_start_ns)
    row.update({
        "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY",
        "rgb_included": False,
        "native_geometry_included": True,
        "route_evidence_included": bool(candidate.get("source_native_route_evidence")),
        "native_geometry_path": str(geometry_path.resolve()),
        "native_geometry_sha256": sha256_file(geometry_path),
    })
    return row


def _rgb_row(candidate: dict[str, Any], *, batch_id: str, role: str, index: int, source_start_ns: int, sheet_path: Path, temporal_path: Path) -> dict[str, Any]:
    row = _common(candidate, batch_id=batch_id, role=role, index=index, source_start_ns=source_start_ns)
    row.update({
        "input_scope": "RGB_ONLY",
        "rgb_included": True,
        "native_geometry_included": False,
        "contact_sheet_path": str(sheet_path.resolve()),
        "contact_sheet_sha256": sha256_file(sheet_path),
        "temporal_manifest_path": str(temporal_path.resolve()),
        "temporal_manifest_sha256": sha256_file(temporal_path),
        "counterexample_search_required": role == COUNTEREXAMPLE_ROLE,
    })
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    candidate_path = Path(args.candidate_artifact).resolve()
    frame_path = Path(args.frame_artifact).resolve()
    ffmpeg = Path(args.ffmpeg_path).resolve()
    ffprobe = Path(args.ffprobe_path).resolve()
    if not candidate_path.is_file() or not frame_path.is_file():
        raise ContractError("THOR-MAGNI candidate and frame artifacts are required")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise ContractError(f"ffmpeg/ffprobe not found: {ffmpeg}, {ffprobe}")
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    if batch_root.exists():
        raise ContractError(f"review batch already exists; refusing overwrite: {batch_root}")
    candidates = load_jsonl(candidate_path)
    selected = _select(candidates, count=args.count, session_count=args.session_count)
    geometry_by_candidate = _load_geometry(frame_path, selected)
    source_starts: dict[str, int] = {}
    for row in candidates:
        if row.get("dataset_id") != "THOR-MAGNI":
            continue
        session = str(row.get("source_session_id") or "")
        try:
            start_timestamp_ns = int(row["start_timestamp_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"THOR-MAGNI candidate lacks source start timestamp: {row.get('candidate_id')}") from exc
        source_starts[session] = min(start_timestamp_ns, source_starts.get(session, start_timestamp_ns))
    batch_root.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        (batch_root / role).mkdir(parents=True, exist_ok=False)
    staging_root = batch_root / "staging"
    staging_root.mkdir(parents=True, exist_ok=False)
    manifests_root = batch_root / "manifests"
    manifest_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_receipts: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(selected):
        candidate_id = str(candidate["candidate_id"])
        source_start = source_starts[str(candidate["source_session_id"])]
        video = _safe_rgb_path(root, candidate)
        key = str(video)
        if key not in video_receipts:
            probe = _probe_video(video, ffprobe)
            duration = float(probe.get("duration")) if probe.get("duration") is not None else None
            video_receipts[key] = {"path": key, "sha256": sha256_file(video), "bytes": video.stat().st_size, "probe": probe, "duration_seconds": duration}
        sample_times = _dense_sample_times(candidate, source_start, video_receipts[key]["duration_seconds"], args.temporal_samples)
        staging_dir = staging_root / candidate_id
        sheet, frame_paths = _extract_dense_contact_sheet(ffmpeg_path=ffmpeg, video_path=video, sample_times=sample_times, staging_dir=staging_dir)
        temporal_path = staging_dir / "temporal_manifest.jsonl"
        write_jsonl(temporal_path, [
            {
                "sample_index": sample_index,
                "relative_time_seconds": round(seconds, 6),
                "source_timestamp_ns": round(source_start + seconds * 1_000_000_000),
                "frame_path": str(frame_path.resolve()),
            }
            for sample_index, (seconds, frame_path) in enumerate(zip(sample_times, frame_paths))
        ])
        for role in (*RGB_ROLES, COUNTEREXAMPLE_ROLE):
            role_sheet = batch_root / role / "contact_sheets" / f"{candidate_id}.jpg"
            role_sheet.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sheet, role_sheet)
            role_temporal = batch_root / role / "temporal_manifests" / f"{candidate_id}.jsonl"
            role_temporal.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporal_path, role_temporal)
            manifest_rows[role].append(_rgb_row(candidate, batch_id=args.batch_id, role=role, index=index, source_start_ns=source_start, sheet_path=role_sheet, temporal_path=role_temporal))
        geometry_path = batch_root / GEOMETRY_ROLE / "native_geometry" / f"{candidate_id}.json"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_payload = {
            "schema": "hftf_d7_public_real_geometry_review_input_v1",
            "record_kind": "REVIEW_INPUT",
            "batch_id": args.batch_id,
            "review_role": GEOMETRY_ROLE,
            "candidate_id": candidate_id,
            "dataset_id": "THOR-MAGNI",
            "source_session_token": _sha256_text(str(candidate["source_session_id"])),
            "window_start_timestamp_ns": candidate.get("start_timestamp_ns"),
            "window_end_timestamp_ns": candidate.get("end_timestamp_ns"),
            "timestamp_semantics": candidate.get("timestamp_semantics"),
            "source_native_fields": ["qtm_time", "pose", "tracks", "intrinsics"],
            "missing_source_native_fields": ["depth", "segmentation"],
            "frame_rows": geometry_by_candidate[candidate_id],
            "model_output_visible": False,
            "instructions": "Use only QTM time, source-native pose/tracks/intrinsics, and any explicitly marked local-route measurements. Do not open RGB or other-role files. Local-route measurements are Development candidate evidence, not event truth; withheld proxy booleans must not be reconstructed as labels. Missing depth/segmentation is NOT_EVALUABLE, never negative.",
        }
        route_evidence = candidate.get("source_native_route_evidence")
        if route_evidence:
            if not isinstance(route_evidence, list):
                raise ContractError(f"invalid route evidence payload: {candidate_id}")
            geometry_payload["source_native_fields"].append("local_route_supervision_measurements")
            geometry_payload["local_route_supervision_measurements"] = route_evidence
            geometry_payload["local_route_supervision_contract"] = candidate.get("route_evidence_contract", {
                "source_native_geometry_only": True,
                "human_event_truth": False,
                "promotion": False,
            })
        write_json(geometry_path, geometry_payload)
        manifest_rows[GEOMETRY_ROLE].append(_geometry_row(candidate, batch_id=args.batch_id, index=index, source_start_ns=source_start, geometry_path=geometry_path))
    manifests_root.mkdir(parents=True, exist_ok=False)
    manifest_paths: dict[str, Path] = {}
    for role in ROLES:
        path = manifests_root / f"{role}.jsonl"
        write_jsonl(path, manifest_rows[role])
        manifest_paths[role] = path
    bundle_manifest = {
        "schema": "hftf_d7_public_real_review_bundle_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_ISOLATED_REVIEW",
        "dataset_id": "THOR-MAGNI",
        "candidate_count": len(selected),
        "candidate_ids": [str(row["candidate_id"]) for row in selected],
        "roles": {
            role: {"manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path), "row_count": len(manifest_rows[role]), "input_scope": "SOURCE_NATIVE_GEOMETRY_ONLY" if role == GEOMETRY_ROLE else "RGB_ONLY"}
            for role, path in manifest_paths.items()
        },
        "candidate_artifact_sha256": sha256_file(candidate_path),
        "frame_artifact_sha256": sha256_file(frame_path),
        "source_video_count": len(video_receipts),
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "final_adjudication_written": False,
        "notes": [
            "RGB roles receive only RGB contact sheets and source-time temporal manifests.",
            "Geometry receives only QTM time, pose, tracks, intrinsics, and copied native geometry JSON; route measurements are source-native Development evidence only; no RGB.",
            "Member-specific license terms remain review-gated; no production or Confirmation authority is granted.",
        ],
    }
    write_json(batch_root / "bundle_manifest.json", bundle_manifest)
    receipt = {
        "schema": "hftf_d7_public_real_review_bundle_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": bundle_manifest["generated_at_utc"],
        "status": "READY_FOR_ISOLATED_REVIEW",
        "output_root": str(root),
        "batch_root": str(batch_root),
        "dataset_id": "THOR-MAGNI",
        "candidate_count": len(selected),
        "review_roles": list(ROLES),
        "bundle_manifest": {"path": str((batch_root / "bundle_manifest.json").resolve()), "sha256": sha256_file(batch_root / "bundle_manifest.json")},
        "manifest_files": {role: {"path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in manifest_paths.items()},
        "source_video_count": len(video_receipts),
        "model_output_visible_in_any_input": False,
        "review_assignments_are_not_labels": True,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(root / "receipts" / f"review_bundle_receipt_{args.batch_id}.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    default_ffmpeg = r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffmpeg.exe"
    default_ffprobe = r"E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffprobe.exe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--candidate-artifact", default=r"F:\ba-data\hftf-d7-public-real\candidates\candidate_index.jsonl")
    parser.add_argument("--frame-artifact", default=r"F:\ba-data\hftf-d7-public-real\canonical\frame_registry.jsonl")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--session-count", type=int, default=5)
    parser.add_argument("--temporal-samples", type=int, default=30)
    parser.add_argument("--ffmpeg-path", default=default_ffmpeg)
    parser.add_argument("--ffprobe-path", default=default_ffprobe)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
