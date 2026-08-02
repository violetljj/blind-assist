#!/usr/bin/env python3
"""Materialize model-blind EgoWalk trajectory windows for D7 intake.

This adapter deliberately consumes only the public extracted trajectory
parquets downloaded by ``download_egowalk_metadata.py``.  It does not open raw
recordings, run a detector, infer an event class, or promote a trajectory
window to an HFTF parent event.  Every emitted candidate therefore remains
``NOT_EVALUABLE`` until RGB/geometry review has actually been completed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pipeline import (
    ContractError,
    canonical_sha256,
    load_json,
    read_int,
    sha256_file,
    stable_id,
    utc_now,
    write_json,
)


REQUIRED_COLUMNS = {
    "timestamp",
    "trajectory",
    "frame",
    "cart_x",
    "cart_y",
    "cart_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
}


def _load_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ContractError(
            "EgoWalk parquet intake requires the bundled polars runtime"
        ) from exc
    return pl


def _is_finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_row(row: dict[str, Any]) -> bool:
    if read_int(row.get("timestamp"), -1) < 0 or read_int(row.get("frame"), -1) < 0:
        return False
    return all(_is_finite_number(row.get(name)) for name in (
        "cart_x", "cart_y", "cart_z", "quat_x", "quat_y", "quat_z", "quat_w"
    ))


def _segments(
    rows: list[dict[str, Any]],
    *,
    max_gap_ms: int,
    max_position_jump_m: float,
) -> list[list[dict[str, Any]]]:
    """Return monotone, pose-valid source segments without repairing gaps."""

    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in sorted(rows, key=lambda item: (read_int(item.get("frame")), read_int(item.get("timestamp")))):
        valid = _valid_row(row)
        split = not valid
        if previous is not None and valid and _valid_row(previous):
            timestamp_delta = read_int(row.get("timestamp")) - read_int(previous.get("timestamp"))
            position_delta = math.sqrt(sum(
                (float(row.get(axis)) - float(previous.get(axis))) ** 2
                for axis in ("cart_x", "cart_y", "cart_z")
            ))
            split = (
                timestamp_delta <= 0
                or timestamp_delta > max_gap_ms
                or read_int(row.get("frame")) != read_int(previous.get("frame")) + 1
                or position_delta > max_position_jump_m
            )
        if split:
            if current:
                result.append(current)
            current = []
        if valid:
            current.append(row)
        previous = row
    if current:
        result.append(current)
    return result


def _source_hash(receipt: dict[str, Any], relative_path: str, fallback: Path) -> str:
    for item in receipt.get("files", []):
        if isinstance(item, dict) and item.get("path") == relative_path:
            value = str(item.get("sha256") or item.get("provider_lfs_oid") or "")
            if value:
                return value
    return sha256_file(fallback)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _trajectory_name(path: Path) -> str:
    return path.stem


def run(args: argparse.Namespace) -> dict[str, Any]:
    pl = _load_polars()
    metadata_root = Path(args.metadata_root).resolve()
    output_root = Path(args.output_root).resolve()
    data_root = metadata_root / "data"
    receipt_path = metadata_root.parent.parent / "receipts" / "egowalk_metadata_receipt.json"
    if not data_root.is_dir():
        raise ContractError(f"EgoWalk data directory not found: {data_root}")
    parquet_paths = sorted(data_root.glob("*.parquet"))
    if not parquet_paths:
        raise ContractError(f"no EgoWalk parquet files found under {data_root}")
    if not receipt_path.is_file():
        raise ContractError(f"metadata receipt not found: {receipt_path}")
    metadata_receipt = load_json(receipt_path)
    if not isinstance(metadata_receipt, dict):
        raise ContractError("EgoWalk metadata receipt must be an object")
    if args.window_seconds <= 0 or args.fps <= 0:
        raise ContractError("window_seconds and fps must be positive")
    rows_per_window = max(2, int(round(args.window_seconds * args.fps)))
    frame_step_ms = max(1, int(round(1000 / args.fps)))

    frame_path = output_root / "canonical" / "egowalk_frame_registry.jsonl"
    candidate_path = output_root / "candidates" / "egowalk_candidate_index.jsonl"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    frame_handle = frame_path.open("w", encoding="utf-8", newline="\n")
    candidate_handle = candidate_path.open("w", encoding="utf-8", newline="\n")
    frame_count = 0
    candidate_count = 0
    segment_count = 0
    invalid_row_count = 0
    gap_cut_count = 0
    trajectory_records: list[dict[str, Any]] = []

    try:
        for parquet_path in parquet_paths:
            df = pl.read_parquet(parquet_path)
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ContractError(f"{parquet_path.name} missing columns: {sorted(missing)}")
            rows = df.to_dicts()
            trajectory = _trajectory_name(parquet_path)
            source_session_id = stable_id("d7sess", "EgoWalk", trajectory.lower())
            ancestry_group = stable_id("d7anc", "EgoWalk", trajectory.lower())
            parquet_rel = f"data/{parquet_path.name}"
            parquet_hash = _source_hash(metadata_receipt, parquet_rel, parquet_path)
            segments = _segments(
                rows,
                max_gap_ms=args.max_gap_ms,
                max_position_jump_m=args.max_position_jump_m,
            )
            segment_count += len(segments)
            valid_rows = sum(len(segment) for segment in segments)
            invalid_row_count += len(rows) - valid_rows
            if len(segments) > 1:
                gap_cut_count += len(segments) - 1
            local_candidate_count = 0
            for segment_index, segment in enumerate(segments):
                if len(segment) < rows_per_window:
                    continue
                for offset in range(0, len(segment) - rows_per_window + 1, rows_per_window):
                    window = segment[offset : offset + rows_per_window]
                    if len(window) != rows_per_window:
                        continue
                    first = window[0]
                    last = window[-1]
                    first_frame = read_int(first.get("frame"))
                    last_frame = read_int(last.get("frame"))
                    first_timestamp_ms = read_int(first.get("timestamp"))
                    last_timestamp_ms = read_int(last.get("timestamp"))
                    frame_ids: list[str] = []
                    for item in window:
                        frame_index = read_int(item.get("frame"))
                        frame_id = stable_id("d7frm", source_session_id, frame_index, parquet_hash)
                        frame_ids.append(frame_id)
                        frame_handle.write(json.dumps({
                            "schema": "hftf_d7_public_real_frame_v1",
                            "dataset_id": "EgoWalk",
                            "source_session_id": source_session_id,
                            "ancestry_group": ancestry_group,
                            "frame_id": frame_id,
                            "frame_index": frame_index,
                            "timestamp_ns": read_int(item.get("timestamp")) * 1_000_000,
                            "rgb_path": f"raw/egowalk-rgb/{trajectory}__rgb.mp4",
                            "intrinsics_optional": "raw/egowalk-trajectories/meta/camera_rgb.json",
                            "pose_optional": {
                                "cart_x": float(item["cart_x"]),
                                "cart_y": float(item["cart_y"]),
                                "cart_z": float(item["cart_z"]),
                                "quat_x": float(item["quat_x"]),
                                "quat_y": float(item["quat_y"]),
                                "quat_z": float(item["quat_z"]),
                                "quat_w": float(item["quat_w"]),
                            },
                            "depth_optional": None,
                            "segmentation_optional": None,
                            "tracks_optional": None,
                            "source_metadata": {
                                "trajectory": trajectory,
                                "source_parquet": parquet_rel,
                                "source_row_frame": frame_index,
                                "source_row_timestamp_ms": read_int(item.get("timestamp")),
                                "source_truth_status": "METADATA_GEOMETRY_ONLY",
                                "raw_recording_access": "ACCESS_BLOCKED",
                                "rgb_video_downloaded": False,
                            },
                            "source_license": "TERMS_NOT_RESOLVED",
                            "provider_revision": "main",
                            "source_hash": parquet_hash,
                        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                        frame_count += 1
                    candidate_id = stable_id(
                        "d7cand", "EgoWalk", trajectory, segment_index, first_frame, last_frame, parquet_hash
                    )
                    candidate_handle.write(json.dumps({
                        "schema": "hftf_d7_public_real_candidate_v1",
                        "candidate_id": candidate_id,
                        "parent_event_id": stable_id("d7parent", candidate_id),
                        "dataset_id": "EgoWalk",
                        "source_session_id": source_session_id,
                        "ancestry_group": ancestry_group,
                        "source_id": trajectory,
                        "segment_index": segment_index,
                        "start_frame_index": first_frame,
                        "end_frame_index": last_frame,
                        "start_timestamp_ns": first_timestamp_ms * 1_000_000,
                        "end_timestamp_ns": last_timestamp_ms * 1_000_000,
                        "frame_ids": frame_ids,
                        "frame_count": len(frame_ids),
                        "candidate_selection": "MODEL_BLIND_UNIFORM_SOURCE_TRAJECTORY_COVERAGE",
                        "model_output_visible_to_selector": False,
                        "native_geometry_used_for_selection": False,
                        "event_bucket": "NOT_EVALUABLE",
                        "truth_status": "NOT_EVALUABLE",
                        "parent_independence_status": "UNVERIFIED",
                        "required_confirmation_selection": "MODEL_BLIND",
                        "source_license": "TERMS_NOT_RESOLVED",
                        "source_hash": parquet_hash,
                        "provider_revision": "main",
                        "rgb_uri": f"https://huggingface.co/datasets/EgoWalk/trajectories/resolve/main/video/rgb/{trajectory}__rgb.mp4",
                        "geometry_uri": f"raw/egowalk-trajectories/{parquet_rel}",
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    candidate_count += 1
                    local_candidate_count += 1
            trajectory_records.append({
                "trajectory": trajectory,
                "source_session_id": source_session_id,
                "ancestry_group": ancestry_group,
                "source_parquet": parquet_rel,
                "source_hash": parquet_hash,
                "row_count": len(rows),
                "valid_pose_row_count": valid_rows,
                "segment_count": len(segments),
                "candidate_count": local_candidate_count,
                "rgb_uri": f"https://huggingface.co/datasets/EgoWalk/trajectories/resolve/main/video/rgb/{trajectory}__rgb.mp4",
                "access_status": "PUBLIC_EXTRACTED_VIDEO_NOT_DOWNLOADED",
                "truth_status": "NOT_EVALUABLE",
            })
    finally:
        frame_handle.close()
        candidate_handle.close()

    receipt = {
        "schema": "hftf_d7_public_real_egowalk_metadata_ingest_receipt_v1",
        "run_id": args.run_id,
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "input": {
            "metadata_root": str(metadata_root),
            "metadata_receipt": str(receipt_path),
            "metadata_receipt_sha256": sha256_file(receipt_path),
            "trajectory_count": len(parquet_paths),
            "window_seconds": args.window_seconds,
            "fps": args.fps,
            "rows_per_window": rows_per_window,
            "frame_step_ms": frame_step_ms,
            "max_gap_ms": args.max_gap_ms,
            "max_position_jump_m": args.max_position_jump_m,
        },
        "counts": {
            "trajectory_files": len(parquet_paths),
            "source_segments": segment_count,
            "invalid_or_split_rows": invalid_row_count,
            "gap_or_reinitialization_cuts": gap_cut_count,
            "frame_rows": frame_count,
            "candidate_windows": candidate_count,
            "parent_events_admitted": 0,
            "not_evaluable_candidates": candidate_count,
        },
        "authority": {
            "candidate_selection": "MODEL_BLIND_UNIFORM_COVERAGE",
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
        },
        "license_status": "NOT_RESOLVED_BY_METADATA_ONLY_INTAKE",
        "raw_recording_status": "ACCESS_BLOCKED",
        "files": {
            "frame_registry": str(frame_path),
            "candidate_index": str(candidate_path),
            "trajectory_manifest": str(output_root / "manifests" / "egowalk_trajectory_manifest.json"),
        },
        "trajectory_manifest_sha256": None,
        "notes": [
            "Extracted trajectory/pose rows are geometry candidates, not HFTF event truth.",
            "No YOLO, HFTF, segmentation, or other model output was read by this adapter.",
            "Missing raw RGB/depth access, tracks, ancestry, and event labels remain UNKNOWN/NOT_EVALUABLE.",
        ],
    }
    manifest_path = output_root / "manifests" / "egowalk_trajectory_manifest.json"
    write_json(manifest_path, {
        "schema": "hftf_d7_public_real_egowalk_trajectory_manifest_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "source": "EgoWalk/trajectories",
        "records": trajectory_records,
        "count": len(trajectory_records),
        "authority": "DEVELOPMENT_CANDIDATE_DISCOVERY_ONLY",
    })
    receipt["trajectory_manifest_sha256"] = sha256_file(manifest_path)
    receipt_path_out = output_root / "receipts" / f"egowalk_metadata_ingest_receipt_{args.run_id}.json"
    write_json(receipt_path_out, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", default=r"F:\ba-data\hftf-d7-public-real\raw\egowalk-trajectories")
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-gap-ms", type=int, default=1000)
    parser.add_argument("--max-position-jump-m", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
