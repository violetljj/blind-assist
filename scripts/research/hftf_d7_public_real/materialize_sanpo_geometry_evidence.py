#!/usr/bin/env python3
"""Add source-native SANPO depth evidence to an unreviewed D7 batch.

The SANPO review materializer already binds RGB, depth, masks, intrinsics, and
pose by provider frame index.  This preparation step derives only descriptive
depth statistics and depth/mask contact sheets from those immutable source
objects.  It does not infer an event, a label, a threshold, or an admission.

The batch must still be untouched by review or adjudication.  The geometry
manifest is rewritten before isolated agents receive it, and the bundle
manifest records the augmentation and all generated-artifact hashes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


DEFAULT_DISPLAY_MIN_M = 0.5
DEFAULT_DISPLAY_MAX_M = 40.0
PREVIEW_MAX_HEIGHT = 360
PREVIEW_MAX_WIDTH = 640
MAX_CONTACT_SHEET_FRAMES = 6


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in the CLI environment
        raise ContractError(
            "SANPO geometry evidence requires numpy; run with uv --with numpy"
        ) from exc
    return np


def _read_depth(path: Path, np: Any) -> Any:
    """Read SANPO's float16-gzip depth encoding with its H/W header."""

    if not path.is_file():
        raise ContractError(f"SANPO depth object missing: {path}")
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ContractError(f"SANPO depth object cannot be decompressed: {path}: {exc}") from exc
    values = np.frombuffer(raw, dtype="<f2")
    if values.size < 2:
        raise ContractError(f"SANPO depth object has no H/W header: {path}")
    height_value = float(values[0])
    width_value = float(values[1])
    height = int(height_value)
    width = int(width_value)
    if (
        not math.isfinite(height_value)
        or not math.isfinite(width_value)
        or height_value != height
        or width_value != width
        or height <= 0
        or width <= 0
    ):
        raise ContractError(f"SANPO depth object has invalid H/W header: {path}")
    pixels = values[2:]
    if pixels.size != height * width:
        raise ContractError(
            f"SANPO depth object pixel count mismatch: {path}: "
            f"header={height}x{width}, pixels={pixels.size}"
        )
    return pixels.reshape((height, width))


def _sample_indices(frame_count: int, *, limit: int = MAX_CONTACT_SHEET_FRAMES) -> list[int]:
    if frame_count <= 0:
        return []
    limit = max(1, min(int(limit), frame_count))
    if limit == 1:
        return [0]
    values = [round(index * (frame_count - 1) / (limit - 1)) for index in range(limit)]
    return sorted(set(int(value) for value in values))


def _finite_stats(array: Any, np: Any) -> dict[str, Any]:
    valid = np.isfinite(array) & (array > 0)
    count = int(array.size)
    valid_count = int(valid.sum())
    result: dict[str, Any] = {
        "pixel_count": count,
        "valid_positive_finite_pixel_count": valid_count,
        "valid_positive_finite_fraction": valid_count / count if count else 0.0,
        "min_m": None,
        "max_m": None,
        "quantiles_m": {},
        "tile_medians_m": [],
    }
    if valid_count == 0:
        return result
    values = array[valid].astype("<f4", copy=False)
    result["min_m"] = float(np.min(values))
    result["max_m"] = float(np.max(values))
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    result["quantiles_m"] = {
        key: float(value)
        for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), quantiles)
    }
    height, width = array.shape
    tiles: list[list[float | None]] = []
    for row_index in range(3):
        row_values: list[float | None] = []
        y0 = row_index * height // 3
        y1 = (row_index + 1) * height // 3
        for column_index in range(3):
            x0 = column_index * width // 3
            x1 = (column_index + 1) * width // 3
            tile = array[y0:y1, x0:x1]
            tile_valid = np.isfinite(tile) & (tile > 0)
            row_values.append(float(np.median(tile[tile_valid])) if tile_valid.any() else None)
        tiles.append(row_values)
    result["tile_medians_m"] = tiles
    return result


def _downsample(array: Any, np: Any) -> Any:
    height, width = array.shape
    step_y = max(1, math.ceil(height / PREVIEW_MAX_HEIGHT))
    step_x = max(1, math.ceil(width / PREVIEW_MAX_WIDTH))
    return array[::step_y, ::step_x]


def _write_depth_pgm(
    path: Path,
    array: Any,
    np: Any,
    *,
    display_min_m: float,
    display_max_m: float,
) -> None:
    if not display_min_m < display_max_m:
        raise ContractError("depth display range must be increasing")
    sampled = _downsample(array, np).astype("<f4", copy=False)
    valid = np.isfinite(sampled) & (sampled > 0)
    pixels = np.zeros(sampled.shape, dtype=np.uint8)
    clipped = np.clip(sampled, display_min_m, display_max_m)
    scaled = (display_max_m - clipped) / (display_max_m - display_min_m) * 255.0
    pixels[valid] = np.asarray(scaled[valid], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii"))
        handle.write(pixels.tobytes())


def _run_contact_sheet(
    ffmpeg: Path,
    pattern: Path,
    output: Path,
    *,
    frame_count: int,
    extension: str,
) -> None:
    if frame_count <= 0:
        return
    columns = min(3, frame_count)
    rows = math.ceil(frame_count / columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "5",
        "-i",
        str(pattern),
        "-vf",
        f"scale=480:-2,tile={columns}x{rows}:padding=6:margin=6",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file():
        raise ContractError(
            f"SANPO geometry {extension} contact-sheet creation failed: "
            f"{result.stderr.strip()}"
        )


def _source_refs(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in geometry.get("intrinsics_objects", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("local_path") or ""))
        if path.is_file():
            refs.append({
                "kind": "intrinsics_description",
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "remote_name": item.get("remote_name"),
            })
    return refs


def _summarize_candidate(
    geometry: dict[str, Any],
    *,
    output_root: Path,
    ffmpeg: Path,
    np: Any,
    display_min_m: float,
    display_max_m: float,
    allow_relative_nominal_phase: bool,
) -> tuple[dict[str, Any], Path]:
    candidate_id = str(geometry.get("candidate_id") or "")
    frames = geometry.get("frames")
    if not candidate_id or not isinstance(frames, list) or not frames:
        raise ContractError("SANPO native geometry manifest has no candidate or frames")
    sample_indices = _sample_indices(len(frames))
    candidate_root = output_root / candidate_id
    depth_dir = candidate_root / "depth_preview_frames"
    mask_dir = candidate_root / "mask_preview_frames"
    if candidate_root.exists():
        raise ContractError(f"geometry evidence output already exists: {candidate_root}")
    candidate_root.mkdir(parents=True, exist_ok=False)
    previous_small: Any = None
    frame_summaries: list[dict[str, Any]] = []
    for local_position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ContractError(f"SANPO geometry frame is not an object: {candidate_id}:{local_position}")
        depth_path = Path(str(frame.get("depth_path") or ""))
        array = _read_depth(depth_path, np)
        stats = _finite_stats(array, np)
        small = _downsample(array, np).astype("<f4", copy=False)
        delta: float | None = None
        if previous_small is not None and previous_small.shape == small.shape:
            both_valid = (
                np.isfinite(previous_small)
                & (previous_small > 0)
                & np.isfinite(small)
                & (small > 0)
            )
            if both_valid.any():
                delta = float(np.median(np.abs(small[both_valid] - previous_small[both_valid])))
        previous_small = small
        summary_row = {
            "frame_index": frame.get("frame_index"),
            "nominal_time_ns": frame.get("nominal_time_ns"),
            "depth_path": str(depth_path.resolve()),
            "depth_sha256": sha256_file(depth_path),
            "depth_encoding": "SANPO_FLOAT16_GZIP_HEADER_H_W_THEN_H_W_PIXELS",
            "depth_shape_hw": [int(array.shape[0]), int(array.shape[1])],
            "source_native_depth_stats": stats,
            "median_abs_depth_delta_m_from_previous_source_frame": delta,
        }
        if local_position in sample_indices:
            preview_index = sample_indices.index(local_position)
            depth_preview = depth_dir / f"frame_{preview_index:06d}.pgm"
            _write_depth_pgm(
                depth_preview,
                array,
                np,
                display_min_m=display_min_m,
                display_max_m=display_max_m,
            )
            summary_row["depth_preview_path"] = str(depth_preview.resolve())
            mask_path = Path(str(frame.get("mask_path") or ""))
            if mask_path.is_file():
                mask_preview = mask_dir / f"frame_{preview_index:06d}.png"
                mask_preview.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, mask_preview)
                summary_row["mask_preview_path"] = str(mask_preview.resolve())
            else:
                summary_row["mask_preview_path"] = None
        else:
            summary_row["depth_preview_path"] = None
            summary_row["mask_preview_path"] = None
        frame_summaries.append(summary_row)

    depth_contact_sheet = candidate_root / "depth_contact_sheet.jpg"
    _run_contact_sheet(
        ffmpeg,
        depth_dir / "frame_%06d.pgm",
        depth_contact_sheet,
        frame_count=len(sample_indices),
        extension="depth",
    )
    mask_count = len(list(mask_dir.glob("frame_*.png")))
    mask_contact_sheet: Path | None = None
    if mask_count == len(sample_indices) and mask_count > 0:
        mask_contact_sheet = candidate_root / "mask_contact_sheet.jpg"
        _run_contact_sheet(
            ffmpeg,
            mask_dir / "frame_%06d.png",
            mask_contact_sheet,
            frame_count=mask_count,
            extension="mask",
        )

    summary = {
        "schema": "hftf_d7_public_real_sanpo_geometry_evidence_summary_v1",
        "record_kind": "SOURCE_NATIVE_GEOMETRY_SUMMARY",
        "dataset_id": "SANPO-Real",
        "candidate_id": candidate_id,
        "source_session_token": geometry.get("source_session_token"),
        "source_native_only": True,
        "rgb_included": False,
        "model_output_visible": False,
        "event_truth_inferred": False,
        "event_threshold_applied": False,
        "summary_semantics": "descriptive source-native depth measurements and display previews; not event truth",
        "source_intrinsics_refs": _source_refs(geometry),
        "pose_row_binding": geometry.get("pose_row_binding"),
        "relative_nominal_phase_contract": bool(
            allow_relative_nominal_phase
            and geometry.get("pose_row_binding") == "FRAME_INDEX_ROW_KEYED"
        ),
        "relative_nominal_phase_rule": (
            "frame_index_at_declared_fps_with_complete_pose_row_binding"
            if allow_relative_nominal_phase
            else "not_enabled"
        ),
        "frame_count": len(frames),
        "preview_frame_positions": sample_indices,
        "depth_display_mapping": {
            "display_min_m": display_min_m,
            "display_max_m": display_max_m,
            "closer_depth_is_brighter": True,
            "invalid_depth_is_black": True,
            "mapping_is_for_visualization_only": True,
        },
        "depth_contact_sheet_path": str(depth_contact_sheet.resolve()),
        "depth_contact_sheet_sha256": sha256_file(depth_contact_sheet),
        "mask_contact_sheet_path": str(mask_contact_sheet.resolve()) if mask_contact_sheet else None,
        "mask_contact_sheet_sha256": sha256_file(mask_contact_sheet) if mask_contact_sheet else None,
        "frames": frame_summaries,
    }
    summary_path = candidate_root / "geometry_evidence_summary.json"
    write_json(summary_path, summary)
    return summary, summary_path


def _copy_role_assets(row: dict[str, Any], *, output_batch: Path, role: str) -> dict[str, Any]:
    updated = dict(row)
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        raise ContractError(f"review manifest row has no candidate_id: {role}")
    if row.get("contact_sheet_path"):
        source = Path(str(row["contact_sheet_path"]))
        destination = output_batch / role / "contact_sheets" / f"{candidate_id}.jpg"
        if not source.is_file():
            raise ContractError(f"SANPO review contact sheet missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        updated["contact_sheet_path"] = str(destination.resolve())
        updated["contact_sheet_sha256"] = sha256_file(destination)
    temporal = Path(str(row.get("temporal_manifest_path") or ""))
    if not temporal.is_file():
        raise ContractError(f"SANPO review temporal manifest missing: {temporal}")
    temporal_destination = output_batch / role / "temporal_manifests" / f"{candidate_id}.jsonl"
    temporal_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temporal, temporal_destination)
    updated["temporal_manifest_path"] = str(temporal_destination.resolve())
    updated["temporal_manifest_sha256"] = sha256_file(temporal_destination)
    return updated


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    batch_root = root / "reviews" / "input_bundles" / args.batch_id
    bundle_path = batch_root / "bundle_manifest.json"
    if not bundle_path.is_file():
        raise ContractError(f"SANPO review bundle missing: {bundle_path}")
    if (batch_root / "GEOMETRY_EVIDENCE_REVIEWER" / "completed_review.jsonl").exists():
        raise ContractError("refusing to augment SANPO geometry after review output exists")
    receipt_root = root / "receipts"
    post_review_receipts = (
        receipt_root / f"review_ingest_receipt_{args.batch_id}.json",
        receipt_root / f"adjudication_bundle_receipt_{args.batch_id}.json",
        receipt_root / f"adjudication_ingest_receipt_{args.batch_id}.json",
    )
    if any(path.exists() for path in post_review_receipts):
        raise ContractError("refusing to augment SANPO geometry after review/adjudication exists")
    ffmpeg = Path(args.ffmpeg_path).resolve()
    if not ffmpeg.is_file():
        raise ContractError(f"ffmpeg not found: {ffmpeg}")
    if args.display_min_m <= 0 or args.display_max_m <= args.display_min_m:
        raise ContractError("invalid depth display range")
    np = _numpy()
    bundle = load_json(bundle_path)
    if not isinstance(bundle, dict) or bundle.get("status") != "READY_FOR_ISOLATED_REVIEW":
        raise ContractError("SANPO review bundle is not ready for isolated review")
    if bundle.get("model_output_visible_in_any_input") is not False:
        raise ContractError("SANPO review bundle model-output firewall is not closed")
    geometry_manifest_path = batch_root / "manifests" / "GEOMETRY_EVIDENCE_REVIEWER.jsonl"
    geometry_rows = load_jsonl(geometry_manifest_path)
    evidence_root = batch_root / "GEOMETRY_EVIDENCE_REVIEWER" / "geometry_evidence"
    summaries: list[dict[str, Any]] = []
    updated_rows: list[dict[str, Any]] = []
    for row in geometry_rows:
        geometry_path = Path(str(row.get("native_geometry_path") or ""))
        geometry = load_json(geometry_path)
        if not isinstance(geometry, dict):
            raise ContractError(f"SANPO native geometry is not an object: {geometry_path}")
        summary, summary_path = _summarize_candidate(
            geometry,
            output_root=evidence_root,
            ffmpeg=ffmpeg,
            np=np,
            display_min_m=float(args.display_min_m),
            display_max_m=float(args.display_max_m),
            allow_relative_nominal_phase=bool(args.relative_nominal_phase_contract),
        )
        enhanced = copy.deepcopy(geometry)
        enhanced["geometry_evidence_summary_path"] = str(summary_path.resolve())
        enhanced["geometry_evidence_summary_sha256"] = sha256_file(summary_path)
        enhanced["depth_contact_sheet_path"] = summary["depth_contact_sheet_path"]
        enhanced["depth_contact_sheet_sha256"] = summary["depth_contact_sheet_sha256"]
        enhanced["mask_contact_sheet_path"] = summary["mask_contact_sheet_path"]
        enhanced["mask_contact_sheet_sha256"] = summary["mask_contact_sheet_sha256"]
        enhanced["preview_frame_positions"] = summary["preview_frame_positions"]
        enhanced["source_native_numeric_summary"] = True
        enhanced["event_truth_inferred"] = False
        enhanced["event_threshold_applied"] = False
        enhanced["model_output_visible"] = False
        enhanced["rgb_included"] = False
        relative_phase_allowed = bool(
            args.relative_nominal_phase_contract
            and geometry.get("pose_row_binding") == "FRAME_INDEX_ROW_KEYED"
        )
        enhanced["relative_nominal_phase_contract"] = relative_phase_allowed
        enhanced["phase_timing_contract"] = {
            "nominal_time_semantics": "DERIVED_RELATIVE_FRAME_INDEX_AT_DECLARED_FPS",
            "capture_timestamp_required": False,
            "pose_row_binding_required": True,
            "relative_phase_allowed": relative_phase_allowed,
            "phase_contract_is_not_event_truth": True,
        }
        base_instruction = str(geometry.get("instructions") or "")
        if relative_phase_allowed:
            phase_instruction = (
                " Relative nominal frame time may be used for pre/alertable/passed-clearance "
                "phase intervals because the complete pose CSV is row-keyed to every source frame; "
                "missing capture_timestamp alone is not NOT_EVALUABLE."
            )
        else:
            phase_instruction = (
                " Relative phase timing is NOT_EVALUABLE unless complete frame-index pose binding "
                "is present; missing capture_timestamp remains a recorded limitation."
            )
        enhanced["instructions"] = (
            base_instruction
            + phase_instruction
            + " Descriptive depth statistics and contact sheets are source-native aids only; "
            "they do not establish event truth or an alert threshold."
        ).strip()
        write_json(geometry_path, enhanced)
        updated = dict(row)
        updated["native_geometry_sha256"] = sha256_file(geometry_path)
        updated_rows.append(updated)
        summaries.append({
            "candidate_id": summary["candidate_id"],
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "depth_contact_sheet_path": summary["depth_contact_sheet_path"],
            "depth_contact_sheet_sha256": summary["depth_contact_sheet_sha256"],
            "mask_contact_sheet_path": summary["mask_contact_sheet_path"],
            "mask_contact_sheet_sha256": summary["mask_contact_sheet_sha256"],
            "frame_count": summary["frame_count"],
            "preview_frame_positions": summary["preview_frame_positions"],
        })
    write_jsonl(geometry_manifest_path, updated_rows)
    bundle["geometry_evidence_augmented"] = True
    bundle["geometry_evidence_run_id"] = args.run_id
    bundle["geometry_evidence_generated_at_utc"] = utc_now()
    bundle["geometry_evidence_source_native_only"] = True
    bundle["geometry_evidence_model_output_visible"] = False
    bundle["geometry_evidence_event_truth_inferred"] = False
    bundle["geometry_evidence_summary_count"] = len(summaries)
    bundle["geometry_evidence_summary_artifact"] = {
        "path": str((batch_root / "GEOMETRY_EVIDENCE_REVIEWER" / "geometry_evidence_manifest.jsonl").resolve()),
    }
    bundle["roles"]["GEOMETRY_EVIDENCE_REVIEWER"]["manifest_sha256"] = sha256_file(geometry_manifest_path)
    summary_manifest_path = batch_root / "GEOMETRY_EVIDENCE_REVIEWER" / "geometry_evidence_manifest.jsonl"
    write_jsonl(summary_manifest_path, summaries)
    bundle["geometry_evidence_summary_artifact"]["sha256"] = sha256_file(summary_manifest_path)
    write_json(bundle_path, bundle)
    return {
        "status": "GEOMETRY_EVIDENCE_AUGMENTED",
        "batch_id": args.batch_id,
        "candidate_count": len(updated_rows),
        "summary_manifest": str(summary_manifest_path.resolve()),
        "bundle_manifest": str(bundle_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ffmpeg-path", required=True)
    parser.add_argument("--display-min-m", type=float, default=DEFAULT_DISPLAY_MIN_M)
    parser.add_argument("--display-max-m", type=float, default=DEFAULT_DISPLAY_MAX_M)
    parser.add_argument(
        "--relative-nominal-phase-contract",
        action="store_true",
        help="permit relative nominal phase intervals only with complete frame-index pose binding",
    )
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
