#!/usr/bin/env python3
"""Augment an untouched EgoWalk D7 review bundle with source-native depth.

The EgoWalk extracted RGB/pose review bundle deliberately has no depth
because the D7 metadata intake never opens depth.  This command is a separate,
receipt-bound Development-only step for an explicitly supplied local media
root.  It reads only the source depth stream, creates descriptive previews and
statistics, and rewrites the geometry-role input before any review output
exists.  It never infers an event, a phase, a threshold, or an admission.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from materialize_review_bundle import _assert_model_blind
from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


DEFAULT_DISPLAY_MIN_M = 0.5
DEFAULT_DISPLAY_MAX_M = 40.0
EGOWALK_POSE_PADDING_FRAMES = 4
GEOMETRY_ROLE = "GEOMETRY_EVIDENCE_REVIEWER"


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime-specific
        raise ContractError("EgoWalk depth evidence requires numpy") from exc
    return np


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _selected_depth_bindings(media_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = media_root / "acquisition_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ContractError(f"EgoWalk acquisition manifest is not an object: {manifest_path}")
    if manifest.get("dataset_repo") != "EgoWalk/trajectories":
        raise ContractError("depth media root is not the frozen EgoWalk trajectories source")
    bindings: dict[str, dict[str, Any]] = {}
    for item in manifest.get("downloaded_files", []):
        if not isinstance(item, dict) or item.get("kind") != "depth":
            continue
        trajectory = str(item.get("trajectory") or "")
        if not trajectory or trajectory in bindings:
            raise ContractError(f"duplicate or missing EgoWalk depth trajectory binding: {trajectory}")
        path = media_root / "video" / "depth" / f"{trajectory}__depth.mkv"
        if not path.is_file():
            raise ContractError(f"bound EgoWalk depth media missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(item.get("size_bytes") or -1) or actual_hash != str(item.get("sha256") or ""):
            raise ContractError(f"EgoWalk depth media hash/size mismatch: {path}")
        bindings[trajectory] = {
            "trajectory": trajectory,
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "size_bytes": actual_size,
            "source_role": "DEVELOPMENT_ONLY_CONSUMED_STAGE_C_MEDIA"
            if manifest.get("selected_sources_burned") is True
            else "D7_DEVELOPMENT_SOURCE",
        }
    if not bindings:
        raise ContractError("EgoWalk acquisition manifest has no bound depth media")
    return bindings


def _sample_indices(candidate: dict[str, Any], frame_count: int) -> list[int]:
    try:
        start = int(candidate["start_frame_index"])
        end = int(candidate["end_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"candidate has invalid EgoWalk frame range: {candidate.get('candidate_id')}") from exc
    if start < 0 or end < start or frame_count <= 0:
        raise ContractError(f"candidate frame range is invalid: {candidate.get('candidate_id')}")
    midpoint = int(round((start + end) / 2.0))
    values = [start - EGOWALK_POSE_PADDING_FRAMES, start, midpoint, end, end + EGOWALK_POSE_PADDING_FRAMES]
    result: list[int] = []
    for value in values:
        value = min(max(0, value), frame_count - 1)
        if not result or value != result[-1]:
            result.append(value)
    return result


def _decode_depth(path: Path, indices: list[int], np: Any) -> tuple[dict[str, Any], dict[int, Any]]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - runtime-specific
        raise ContractError("EgoWalk depth evidence requires PyAV") from exc
    samples: dict[int, Any] = {}
    pts: list[int] = []
    selected_indices = set(indices)
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ContractError(f"expected one depth video stream: {path}")
        stream = streams[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ContractError(f"depth frame has no PTS: {path}:{index}")
            pts.append(int(frame.pts))
            if index in selected_indices:
                raw = frame.to_ndarray(format="gray16le")
                depth = raw.astype(np.float32) / 1000.0
                depth[raw == 0] = np.nan
                samples[index] = depth
        stream_rate = stream.average_rate or stream.base_rate
        rate_hz = float(stream_rate) if stream_rate is not None else None
        report = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "width": int(stream.width),
            "height": int(stream.height),
            "source_pixel_format": stream.format.name if stream.format is not None else None,
            "rate_hz": rate_hz,
            "decoded_frame_count": len(pts),
            "first_pts": pts[0] if pts else None,
            "last_pts": pts[-1] if pts else None,
            "pts_strictly_increasing": all(b > a for a, b in zip(pts, pts[1:])),
            "pts_constant_step": len(pts) > 1 and len({b - a for a, b in zip(pts, pts[1:])}) == 1,
            "source_pixel_unit": "millimetres_divided_by_1000_to_metres",
        }
    # The requested upper bound includes a conservative post-window padding
    # frame.  The actual decoded count is authoritative, and the caller
    # clamps review samples after this report is available.
    return report, samples


def _stats(array: Any, np: Any) -> dict[str, Any]:
    valid = np.isfinite(array) & (array > 0)
    count = int(valid.sum())
    values = array[valid]
    return {
        "pixel_count": int(array.size),
        "valid_positive_finite_pixel_count": count,
        "valid_positive_finite_fraction": count / int(array.size) if array.size else 0.0,
        "min_m": float(np.min(values)) if count else None,
        "p50_m": float(np.percentile(values, 50)) if count else None,
        "p95_m": float(np.percentile(values, 95)) if count else None,
        "max_m": float(np.max(values)) if count else None,
    }


def _write_depth_pgm(path: Path, array: Any, np: Any, display_min_m: float, display_max_m: float) -> None:
    if not 0 < display_min_m < display_max_m:
        raise ContractError("invalid depth display range")
    height, width = array.shape
    step_y = max(1, math.ceil(height / 360))
    step_x = max(1, math.ceil(width / 640))
    sampled = array[::step_y, ::step_x].astype("<f4", copy=False)
    valid = np.isfinite(sampled) & (sampled > 0)
    clipped = np.clip(sampled, display_min_m, display_max_m)
    pixels = np.zeros(sampled.shape, dtype=np.uint8)
    pixels[valid] = np.asarray(
        (display_max_m - clipped[valid]) / (display_max_m - display_min_m) * 255.0,
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii"))
        handle.write(pixels.tobytes())


def _contact_sheet(ffmpeg: Path, preview_dir: Path, output: Path, count: int) -> None:
    if count <= 0:
        raise ContractError("no depth previews were written")
    columns = min(3, count)
    rows = math.ceil(count / columns)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-framerate", "5",
        "-i", str(preview_dir / "frame_%06d.pgm"),
        "-vf", f"scale=480:-2,tile={columns}x{rows}:padding=6:margin=6",
        "-frames:v", "1", "-q:v", "2", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file():
        raise ContractError(f"depth contact sheet failed: {result.stderr.strip()}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    bundle_path = batch_root / "bundle_manifest.json"
    if not bundle_path.is_file():
        raise ContractError(f"review bundle missing: {bundle_path}")
    if (batch_root / GEOMETRY_ROLE / "completed_review.jsonl").exists():
        raise ContractError("refusing depth augmentation after geometry review output exists")
    for receipt_name in (
        f"review_ingest_receipt_{args.batch_id}.json",
        f"adjudication_bundle_receipt_{args.batch_id}.json",
        f"adjudication_ingest_receipt_{args.batch_id}.json",
    ):
        if (root / "receipts" / receipt_name).exists():
            raise ContractError("refusing depth augmentation after review/adjudication state exists")
    bundle = load_json(bundle_path)
    if not isinstance(bundle, dict) or bundle.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError("review bundle is not ready for isolated review")
    if bundle.get("model_output_visible_in_any_input") is not False:
        raise ContractError("review bundle model-output firewall is not closed")
    geometry_manifest_path = batch_root / "manifests" / f"{GEOMETRY_ROLE}.jsonl"
    geometry_rows = _iter_jsonl(geometry_manifest_path)
    media_root = Path(args.media_root).resolve()
    bindings = _selected_depth_bindings(media_root)
    np = _numpy()
    ffmpeg = Path(args.ffmpeg_path).resolve()
    if not ffmpeg.is_file():
        raise ContractError(f"ffmpeg not found: {ffmpeg}")
    evidence_root = batch_root / GEOMETRY_ROLE / "depth_evidence"
    if evidence_root.exists():
        raise ContractError(f"depth evidence output already exists: {evidence_root}")
    evidence_root.mkdir(parents=True, exist_ok=False)
    candidate_rows = {str(row.get("candidate_id")): row for row in load_jsonl(root / "candidates" / "candidate_index.jsonl")}
    selected_candidate_ids = {str(item.get("candidate_id")) for item in geometry_rows}
    updated_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    decoded_by_source: dict[str, tuple[dict[str, Any], dict[int, Any]]] = {}
    for row in geometry_rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidate_rows.get(candidate_id)
        if candidate is None:
            raise ContractError(f"candidate missing from registry: {candidate_id}")
        source_id = str(candidate.get("source_id") or "")
        binding = bindings.get(source_id)
        if binding is None:
            raise ContractError(f"no bound depth media for selected source session: {source_id}")
        geometry_path = Path(str(row.get("native_geometry_path") or "")).resolve()
        geometry = load_json(geometry_path)
        if not isinstance(geometry, dict):
            raise ContractError(f"geometry input is not an object: {geometry_path}")
        if source_id not in decoded_by_source:
            # Decode once per source after calculating a conservative upper
            # bound from all selected candidates in that source.
            source_candidates = [
                item for item in candidate_rows.values()
                if str(item.get("candidate_id")) in selected_candidate_ids
                and str(item.get("source_id")) == source_id
            ]
            max_frame = max(
                int(item.get("end_frame_index") or 0) + EGOWALK_POSE_PADDING_FRAMES
                for item in source_candidates
            )
            report, samples = _decode_depth(Path(binding["path"]), list(range(max_frame + 1)), np)
            decoded_by_source[source_id] = (report, samples)
        report, decoded = decoded_by_source[source_id]
        frame_count = int(report["decoded_frame_count"])
        indices = _sample_indices(candidate, frame_count)
        preview_root = evidence_root / candidate_id / "depth_preview_frames"
        preview_root.mkdir(parents=True, exist_ok=False)
        frame_stats: list[dict[str, Any]] = []
        for preview_index, frame_index in enumerate(indices):
            array = decoded[frame_index]
            preview_path = preview_root / f"frame_{preview_index:06d}.pgm"
            _write_depth_pgm(preview_path, array, np, args.display_min_m, args.display_max_m)
            frame_stats.append({
                "frame_index": frame_index,
                "preview_path": str(preview_path.resolve()),
                "preview_sha256": sha256_file(preview_path),
                "source_native_depth_stats": _stats(array, np),
            })
        contact_sheet = evidence_root / candidate_id / "depth_contact_sheet.jpg"
        _contact_sheet(args.ffmpeg_path, preview_root, contact_sheet, len(indices))
        enhanced = dict(geometry)
        enhanced["source_native_fields"] = sorted(set(list(enhanced.get("source_native_fields") or []) + ["depth"]))
        enhanced["missing_source_native_fields"] = [
            item for item in (enhanced.get("missing_source_native_fields") or []) if item != "depth"
        ]
        enhanced["depth_evidence_source_role"] = binding["source_role"]
        enhanced["depth_video_path"] = binding["path"]
        enhanced["depth_video_sha256"] = binding["sha256"]
        enhanced["depth_video_size_bytes"] = binding["size_bytes"]
        enhanced["depth_encoding"] = "GRAY16LE_MILLIMETRES_DIVIDED_BY_1000_TO_METRES_ZERO_INVALID"
        enhanced["depth_frame_indices"] = indices
        enhanced["depth_frame_stats"] = frame_stats
        enhanced["depth_contact_sheet_path"] = str(contact_sheet.resolve())
        enhanced["depth_contact_sheet_sha256"] = sha256_file(contact_sheet)
        enhanced["depth_display_mapping"] = {
            "display_min_m": args.display_min_m,
            "display_max_m": args.display_max_m,
            "closer_depth_is_brighter": True,
            "invalid_depth_is_black": True,
            "mapping_is_for_visualization_only": True,
        }
        enhanced["source_native_numeric_summary"] = True
        enhanced["event_truth_inferred"] = False
        enhanced["event_threshold_applied"] = False
        enhanced["model_output_visible"] = False
        enhanced["rgb_included"] = False
        enhanced["instructions"] = (
            str(enhanced.get("instructions") or "")
            + " Source-native depth previews and statistics are descriptive aids only; "
            "they do not establish event truth, a hazard threshold, or a clearance label."
        ).strip()
        _assert_model_blind(enhanced)
        write_json(geometry_path, enhanced)
        updated = dict(row)
        updated["native_geometry_sha256"] = sha256_file(geometry_path)
        updated["depth_contact_sheet_path"] = str(contact_sheet.resolve())
        updated["depth_contact_sheet_sha256"] = sha256_file(contact_sheet)
        updated_rows.append(updated)
        summaries.append({
            "candidate_id": candidate_id,
            "source_id": source_id,
            "depth_video_sha256": binding["sha256"],
            "depth_decoded_frame_count": frame_count,
            "depth_frame_indices": indices,
            "depth_contact_sheet_path": str(contact_sheet.resolve()),
            "depth_contact_sheet_sha256": sha256_file(contact_sheet),
        })
    source_roles: set[str] = set()
    for item in summaries:
        source_roles.add(str(bindings[str(item["source_id"])]["source_role"]))
    if len(source_roles) != 1:
        raise ContractError(f"selected depth sources have mixed role authority: {sorted(source_roles)}")
    source_role = next(iter(source_roles))
    write_jsonl(geometry_manifest_path, updated_rows)
    summary_path = batch_root / GEOMETRY_ROLE / "depth_evidence_manifest.jsonl"
    write_jsonl(summary_path, summaries)
    bundle["geometry_evidence_augmented"] = True
    bundle["geometry_evidence_run_id"] = args.run_id
    bundle["geometry_evidence_generated_at_utc"] = utc_now()
    bundle["geometry_evidence_source_native_only"] = True
    bundle["geometry_evidence_model_output_visible"] = False
    bundle["geometry_evidence_event_truth_inferred"] = False
    bundle["geometry_evidence_summary_count"] = len(summaries)
    bundle["geometry_evidence_source_role"] = source_role
    bundle["geometry_evidence_summary_artifact"] = {
        "path": str(summary_path.resolve()),
        "sha256": sha256_file(summary_path),
    }
    bundle["roles"][GEOMETRY_ROLE]["manifest_sha256"] = sha256_file(geometry_manifest_path)
    bundle["notes"] = list(bundle.get("notes") or []) + [
        "Depth is source-native and model-blind, but this selected media root is consumed/burned Stage C evidence and is Development-only; it cannot receive fresh Confirmation credit.",
        "Depth previews are descriptive aids and do not create obstacle segmentation, event truth, or an admission.",
    ]
    _assert_model_blind(bundle)
    write_json(bundle_path, bundle)
    receipt = {
        "schema": "hftf_d7_public_real_egowalk_depth_evidence_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "batch_id": args.batch_id,
        "generated_at_utc": utc_now(),
        "status": "GEOMETRY_EVIDENCE_AUGMENTED_DEVELOPMENT_ONLY",
        "candidate_count": len(updated_rows),
        "source_count": len({item["source_id"] for item in summaries}),
        "source_role": source_role,
        "media_root": str(media_root),
        "summary_manifest": str(summary_path.resolve()),
        "summary_manifest_sha256": sha256_file(summary_path),
        "geometry_manifest": str(geometry_manifest_path.resolve()),
        "geometry_manifest_sha256": sha256_file(geometry_manifest_path),
        "bundle_manifest": str(bundle_path.resolve()),
        "bundle_manifest_sha256": sha256_file(bundle_path),
        "model_output_visible": False,
        "event_truth_inferred": False,
        "confirmation_authorized": False,
        "training_authorized": False,
    }
    write_json(root / "receipts" / f"egowalk_depth_evidence_receipt_{args.batch_id}.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--ffmpeg-path", required=True)
    parser.add_argument("--display-min-m", type=float, default=DEFAULT_DISPLAY_MIN_M)
    parser.add_argument("--display-max-m", type=float, default=DEFAULT_DISPLAY_MAX_M)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
