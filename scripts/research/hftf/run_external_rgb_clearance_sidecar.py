#!/usr/bin/env python3
"""Run candidate-only RGB clearance with causal sparse metric-scale anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_metric3d_clearance_field_a0 import HORIZONS_M, clearance_field
from external_camera_calibration import (
    FrameRectifier,
    finite_ratio,
    load_calibration,
    pinhole_calibration,
)
from metric_scale_anchor import MetricScaleTracker
from metric_traversability_field import (
    AlertMapper,
    build_metric_traversability_field,
)
from multizone_tof_anchor import (
    TofAnchorPolicy,
    TofFrameStream,
    estimate_tof_scale_anchor,
    load_registration,
    load_tof_frames,
)
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource
from sparse_scale_anchor_io import ScaleAnchorStream, load_scale_anchors

SCHEMA = "hftf_external_rgb_sparse_scale_clearance_sidecar_r0"


def load_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "sequence_id",
                "frame_path",
                "timestamp_ns",
                "intrinsics_fx_fy_cx_cy",
            } - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            frame = Path(str(row["frame_path"]))
            if not frame.is_absolute():
                frame = (path.parent / frame).resolve()
            row["frame_path"] = str(frame)
            unique.setdefault(str(frame), row)
    rows = sorted(
        unique.values(),
        key=lambda row: (str(row["sequence_id"]), int(row["timestamp_ns"])),
    )
    previous: dict[str, int] = {}
    for row in rows:
        sequence = str(row["sequence_id"])
        timestamp = int(row["timestamp_ns"])
        if previous.get(sequence, -1) >= timestamp:
            raise ValueError(f"timestamps must increase within {sequence}")
        previous[sequence] = timestamp
    if not rows:
        raise ValueError("manifests contain no frames")
    return rows


def manifest_frames(paths: list[Path]) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    for index, row in enumerate(load_manifest_rows(paths)):
        bgr = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {row['frame_path']}")
        yield bgr, {
            "frame_index": int(row.get("frame_index", index)),
            "sequence_id": str(row["sequence_id"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "intrinsics_fx_fy_cx_cy": list(row["intrinsics_fx_fy_cx_cy"]),
            "source": "manifest_replay",
            "frame_path": str(row["frame_path"]),
        }


def capture_frames(
    camera_index: int | None,
    video: Path | None,
    sequence_id: str,
    rectifier: FrameRectifier,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video) if video is not None else int(camera_index))
    if video is None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, rectifier.calibration.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.calibration.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError("cannot open RGB source")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if video is not None else None
    if video is not None and (not math.isfinite(float(fps)) or float(fps) <= 0):
        capture.release()
        raise RuntimeError("video FPS is invalid")
    previous_timestamp = -1
    frame_index = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                if video is not None:
                    break
                raise RuntimeError("camera read failed")
            timestamp = (
                round(float(capture.get(cv2.CAP_PROP_POS_MSEC)) * 1e6)
                if video is not None
                else time.monotonic_ns()
            )
            if video is not None and timestamp <= previous_timestamp:
                timestamp = round(frame_index / float(fps) * 1e9)
            if timestamp <= previous_timestamp:
                raise RuntimeError("capture timestamps are not strictly increasing")
            previous_timestamp = timestamp
            bgr, valid_mask = rectifier.rectify(raw)
            yield bgr, {
                "frame_index": frame_index,
                "sequence_id": sequence_id,
                "timestamp_ns": timestamp,
                "intrinsics_fx_fy_cx_cy": rectifier.calibration.intrinsics,
                "rectification_valid_mask": valid_mask,
                "source": f"video:{video.resolve()}" if video is not None else f"camera:{camera_index}",
                "capture_fps": fps,
            }
            frame_index += 1
    finally:
        capture.release()


def calibrated_field(
    raw_field: dict[str, Any],
    tracker: MetricScaleTracker,
    timestamp_ns: int,
) -> dict[str, Any]:
    if raw_field.get("status") != "VALID":
        return {"status": "UNKNOWN_RAW_CLEARANCE"}
    clearances = {
        band: raw_field["bands"][band].get("clearance_m")
        for band in ("left", "center", "right")
    }
    scaled = tracker.apply(timestamp_ns, clearances)
    if scaled["status"] != "VALID":
        return scaled
    bands = {}
    for band, value in scaled["bands_m"].items():
        bands[band] = {
            "clearance_m": value,
            "occupied_by_horizon": {
                str(horizon): value is not None and value <= horizon
                for horizon in HORIZONS_M
            },
        }
    return {**scaled, "bands": bands}


def _safe_asset_stem(sequence_id: str, frame_index: int) -> str:
    safe_sequence = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in sequence_id
    )
    return f"{safe_sequence}_{frame_index:06d}"


def write_visualization_assets(
    bgr: np.ndarray,
    calibrated_depth_m: np.ndarray | None,
    output_dir: Path,
    sequence_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """Write optional display assets; these files carry no evidence authority."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_asset_stem(sequence_id, frame_index)
    rgb_path = (output_dir / f"{stem}_rgb.jpg").resolve()
    depth_path = (output_dir / f"{stem}_metric_depth.png").resolve()
    if not cv2.imwrite(str(rgb_path), bgr):
        raise RuntimeError(f"failed to write {rgb_path}")
    if calibrated_depth_m is None:
        preview = np.full(bgr.shape[:2], 96, dtype=np.uint8)
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        cv2.putText(
            preview,
            "UNKNOWN METRIC DEPTH",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        depth_status = "UNKNOWN"
    else:
        depth = np.asarray(calibrated_depth_m, dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            preview = np.zeros(depth.shape, dtype=np.uint8)
        else:
            low, high = np.quantile(depth[valid], [0.05, 0.95])
            span = max(float(high - low), 1e-6)
            normalized = np.clip((depth - low) / span, 0.0, 1.0)
            preview = np.where(valid, np.round(255.0 * (1.0 - normalized)), 0).astype(
                np.uint8
            )
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        preview[~valid] = (80, 80, 80)
        depth_status = "VALID_DEVELOPMENT_DISPLAY"
    if not cv2.imwrite(str(depth_path), preview):
        raise RuntimeError(f"failed to write {depth_path}")
    return {
        "rgb_path": str(rgb_path),
        "metric_depth_heatmap_path": str(depth_path),
        "metric_depth_heatmap_status": depth_status,
        "authority": "DISPLAY_ONLY_NO_EVIDENCE_AUTHORITY",
    }


def assess_image_quality(bgr: np.ndarray) -> dict[str, Any]:
    """Frozen display-independent blur/exposure diagnostics for fail-closed output."""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    preview = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
    laplacian_variance = float(cv2.Laplacian(preview, cv2.CV_64F).var())
    underexposed_fraction = float(np.mean(preview <= 10))
    overexposed_fraction = float(np.mean(preview >= 245))
    sharpness_pass = laplacian_variance >= 20.0
    exposure_pass = underexposed_fraction <= 0.80 and overexposed_fraction <= 0.80
    return {
        "status": "PASS" if sharpness_pass and exposure_pass else "FAIL",
        "pass": sharpness_pass and exposure_pass,
        "laplacian_variance_320x240": laplacian_variance,
        "minimum_laplacian_variance": 20.0,
        "underexposed_fraction": underexposed_fraction,
        "overexposed_fraction": overexposed_fraction,
        "maximum_extreme_exposure_fraction": 0.80,
        "claim_ceiling": "frozen software quality gate; not calibrated perception confidence",
    }


def write_research_depth_artifact(
    raw_depth: np.ndarray,
    calibrated_depth_m: np.ndarray | None,
    output_dir: Path,
    sequence_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """Retain replayable depth under ignored artifacts.local, never in Git."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / f"{_safe_asset_stem(sequence_id, frame_index)}_depth.npz").resolve()
    np.savez_compressed(
        path,
        raw_depth=np.asarray(raw_depth, dtype=np.float32),
        calibrated_depth_m=(
            np.asarray(calibrated_depth_m, dtype=np.float32)
            if calibrated_depth_m is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        calibrated_available=np.asarray([calibrated_depth_m is not None], dtype=np.bool_),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "raw_depth_retained": True,
        "calibrated_depth_retained": calibrated_depth_m is not None,
        "authority": "RESEARCH_REPLAY_ARTIFACT_NO_EVIDENCE_PROMOTION",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, nargs="+")
    source.add_argument("--camera-index", type=int)
    source.add_argument("--video", type=Path)
    parser.add_argument("--sequence-id")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--calibration-size", type=int, nargs=2)
    parser.add_argument("--calibration-json", type=Path)
    anchor_source = parser.add_mutually_exclusive_group(required=True)
    anchor_source.add_argument("--scale-anchor-jsonl", type=Path)
    anchor_source.add_argument("--tof-jsonl", type=Path)
    parser.add_argument("--tof-registration-json", type=Path)
    parser.add_argument("--rgb-calibration-id")
    parser.add_argument("--rgb-clock-domain")
    parser.add_argument("--max-tof-rgb-skew-ms", type=float)
    parser.add_argument("--max-tof-sigma-m", type=float)
    parser.add_argument("--min-tof-zones", type=int)
    parser.add_argument("--min-tof-bands", type=int)
    parser.add_argument("--max-tof-scale-mad", type=float)
    parser.add_argument("--tof-depth-patch-radius-px", type=int, default=2)
    parser.add_argument("--max-anchor-age-ms", type=float, required=True)
    parser.add_argument("--depth-anything-repo", type=Path, required=True)
    parser.add_argument("--depth-anything-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-anything-input-size", type=int, default=392)
    parser.add_argument("--depth-anything-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--research-depth-dir", type=Path)
    parser.add_argument("--demo-alert-horizon-m", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_mode = args.camera_index is not None or args.video is not None
    if not math.isfinite(args.max_anchor_age_ms) or args.max_anchor_age_ms <= 0:
        raise ValueError("--max-anchor-age-ms must be finite and positive")
    if capture_mode and not args.sequence_id:
        raise ValueError("camera/video mode requires --sequence-id for anchor binding")
    if not capture_mode and (
        args.sequence_id
        or args.intrinsics
        or args.calibration_size
        or args.calibration_json
    ):
        raise ValueError("manifest mode uses per-row sequence and calibration")
    if capture_mode and args.calibration_json is None and (
        args.intrinsics is None or args.calibration_size is None
    ):
        raise ValueError("camera/video mode requires a complete calibration")
    if args.calibration_json is not None and (
        args.intrinsics is not None or args.calibration_size is not None
    ):
        raise ValueError("choose JSON calibration or CLI calibration, not both")

    tof_mode = args.tof_jsonl is not None
    tof_required = (
        "tof_registration_json",
        "rgb_clock_domain",
        "max_tof_rgb_skew_ms",
        "max_tof_sigma_m",
        "min_tof_zones",
        "min_tof_bands",
        "max_tof_scale_mad",
    )
    if tof_mode:
        missing = [name for name in tof_required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"ToF mode requires {missing}")
    elif any(getattr(args, name) is not None for name in tof_required) or args.rgb_calibration_id:
        raise ValueError("ToF registration, clock, calibration, and policy require --tof-jsonl")

    anchor_stream = (
        ScaleAnchorStream(load_scale_anchors(args.scale_anchor_jsonl))
        if args.scale_anchor_jsonl is not None
        else None
    )
    tof_stream = TofFrameStream(load_tof_frames(args.tof_jsonl)) if tof_mode else None
    registration = load_registration(args.tof_registration_json) if tof_mode else None
    tof_policy = (
        TofAnchorPolicy(
            max_rgb_tof_skew_ns=round(args.max_tof_rgb_skew_ms * 1e6),
            max_sigma_m=args.max_tof_sigma_m,
            minimum_zones=args.min_tof_zones,
            minimum_bands=args.min_tof_bands,
            maximum_scale_mad=args.max_tof_scale_mad,
            depth_patch_radius_px=args.tof_depth_patch_radius_px,
        )
        if tof_mode
        else None
    )
    if tof_policy is not None:
        tof_policy.validate()
    trackers: dict[str, MetricScaleTracker] = {}
    previous_fields: dict[str, tuple[int, dict[str, Any]]] = {}
    alert_mapper = AlertMapper(args.demo_alert_horizon_m)
    max_age_ns = round(args.max_anchor_age_ms * 1e6)
    source = DepthAnythingV2MetricSource(
        args.depth_anything_repo,
        args.depth_anything_checkpoint,
        args.device,
        args.depth_anything_input_size,
        args.depth_anything_precision,
    )
    if args.manifest is not None:
        if tof_mode and not args.rgb_calibration_id:
            raise ValueError("manifest ToF mode requires --rgb-calibration-id")
        active_rgb_calibration_id = args.rgb_calibration_id
        frames = manifest_frames(args.manifest)
    else:
        calibration = (
            load_calibration(args.calibration_json)
            if args.calibration_json is not None
            else pinhole_calibration(list(args.intrinsics), list(args.calibration_size))
        )
        frames = capture_frames(
            args.camera_index,
            args.video,
            args.sequence_id,
            FrameRectifier(calibration),
        )
        active_rgb_calibration_id = calibration.source_id

    if tof_mode and active_rgb_calibration_id != registration.rgb_calibration_id:
        raise ValueError("active RGB calibration does not match ToF registration")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for bgr, packet in frames:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            sequence = str(packet["sequence_id"])
            tracker = trackers.setdefault(sequence, MetricScaleTracker(max_age_ns))
            anchor_updates = []
            if anchor_stream is not None:
                for anchor in anchor_stream.take_available(
                    sequence, int(packet["timestamp_ns"])
                ):
                    tracker.update(anchor)
                    anchor_updates.append(
                        {
                            "status": "VALID_PRECOMPUTED_SCALE_ANCHOR",
                            "timestamp_ns": anchor.timestamp_ns,
                        }
                    )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_quality = assess_image_quality(bgr)
            depth_started = time.perf_counter()
            depth, metadata = source.infer(
                rgb,
                {"intrinsics_fx_fy_cx_cy": packet["intrinsics_fx_fy_cx_cy"]},
            )
            valid_mask = packet.get("rectification_valid_mask")
            if valid_mask is not None:
                depth = np.asarray(depth).copy()
                depth[~valid_mask] = np.nan
            depth_ms = (time.perf_counter() - depth_started) * 1000.0
            if tof_stream is not None:
                available_tof = tof_stream.take_available(
                    sequence, int(packet["timestamp_ns"])
                )
                for tof_frame in available_tof:
                    anchor, diagnostic = estimate_tof_scale_anchor(
                        np.asarray(depth),
                        list(packet["intrinsics_fx_fy_cx_cy"]),
                        int(packet["timestamp_ns"]),
                        args.rgb_clock_domain,
                        tof_frame,
                        registration,
                        active_rgb_calibration_id,
                        tof_policy,
                    )
                    anchor_updates.append(diagnostic)
                    if anchor is not None:
                        tracker.update(anchor)
            geometry_started = time.perf_counter()
            raw_field = clearance_field(
                depth,
                np.asarray(
                    [
                        [packet["intrinsics_fx_fy_cx_cy"][0], 0, packet["intrinsics_fx_fy_cx_cy"][2]],
                        [0, packet["intrinsics_fx_fy_cx_cy"][1], packet["intrinsics_fx_fy_cx_cy"][3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
            )
            scaled_field = calibrated_field(raw_field, tracker, int(packet["timestamp_ns"]))
            scale_receipt = tracker.resolve(int(packet["timestamp_ns"]))
            metric_depth = (
                np.asarray(depth, dtype=np.float64) * float(scale_receipt["scale"])
                if scale_receipt.get("status") == "VALID"
                else None
            )
            previous_timestamp, previous_field = previous_fields.get(
                sequence, (None, None)
            )
            traversability_field = build_metric_traversability_field(
                metric_depth if metric_depth is not None else np.asarray(depth),
                np.asarray(
                    [
                        [packet["intrinsics_fx_fy_cx_cy"][0], 0, packet["intrinsics_fx_fy_cx_cy"][2]],
                        [0, packet["intrinsics_fx_fy_cx_cy"][1], packet["intrinsics_fx_fy_cx_cy"][3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
                metric_scale=scale_receipt,
                source_model=source.model_id,
                timestamp_ns=int(packet["timestamp_ns"]),
                previous_field=previous_field,
                previous_timestamp_ns=previous_timestamp,
                image_quality=image_quality,
            )
            previous_fields[sequence] = (
                int(packet["timestamp_ns"]),
                traversability_field,
            )
            alert_projection = alert_mapper.map(traversability_field)
            geometry_ms = (time.perf_counter() - geometry_started) * 1000.0
            visualization_assets = (
                write_visualization_assets(
                    bgr,
                    metric_depth,
                    args.visualization_dir,
                    sequence,
                    int(packet["frame_index"]),
                )
                if args.visualization_dir is not None
                else None
            )
            research_depth_artifact = (
                write_research_depth_artifact(
                    depth,
                    metric_depth,
                    args.research_depth_dir,
                    sequence,
                    int(packet["frame_index"]),
                )
                if args.research_depth_dir is not None
                else None
            )
            record = {
                "schema": SCHEMA,
                "sequence_id": sequence,
                "frame_index": int(packet["frame_index"]),
                "timestamp_ns": int(packet["timestamp_ns"]),
                "source": packet["source"],
                "intrinsics_fx_fy_cx_cy": packet["intrinsics_fx_fy_cx_cy"],
                "model_id": source.model_id,
                "model_metadata": metadata,
                "depth_latency_ms": depth_ms,
                "geometry_and_scale_latency_ms": geometry_ms,
                "rectification_valid_fraction": finite_ratio(valid_mask) if valid_mask is not None else 1.0,
                "image_quality": image_quality,
                "metric_scale_receipt": scale_receipt,
                "raw_clearance": raw_field,
                "scaled_clearance": scaled_field,
                "metric_traversability_field": traversability_field,
                "shadow_demo_alert_projection": alert_projection,
                "visualization_assets": visualization_assets,
                "research_depth_artifact": research_depth_artifact,
                "metric_anchor_updates": anchor_updates,
                "claim_ceiling": (
                    "candidate-only clearance sidecar; alert projection is non-actuating "
                    "shadow/demo only; no safety, navigation, or production decision"
                ),
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            processed += 1
            print(
                f"sequence={sequence} frame={packet['frame_index']} status={scaled_field['status']} depth_ms={depth_ms:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
