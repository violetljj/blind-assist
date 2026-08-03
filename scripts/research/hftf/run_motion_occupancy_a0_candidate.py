#!/usr/bin/env python3
"""Run the frozen A0.1 current-occupancy candidate on an RGB manifest."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_metric3d_clearance_field_a0 import BANDS, HORIZONS_M, clearance_field
from evaluate_motion_conditioned_occupancy_a0 import (
    EXPECTED_RAFT_SHA256,
    FEATURE_NAMES,
    extract_motion,
)
from produce_external_rgb_metric_depth_observations import (
    UniDepthSource,
    intrinsics_matrix,
)


SCHEMA = "blindassist_hftf_motion_occupancy_a0_candidate_output_r0"


def load_manifest(path: Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_frames is not None:
        rows = rows[:max_frames]
    if len(rows) < 2:
        raise ValueError("manifest needs at least two frames")
    for row in rows:
        if len(row.get("intrinsics_fx_fy_cx_cy", [])) != 4:
            raise ValueError("every row needs four intrinsics values")
        row["frame_path"] = str(Path(row["frame_path"]).resolve())
        if "timestamp" not in row:
            if "timestamp_ns" not in row:
                raise ValueError("every row needs timestamp or timestamp_ns")
            row["timestamp"] = float(row["timestamp_ns"]) / 1_000_000_000.0
    by_sequence: dict[str, list[float]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["sequence_id"]), []).append(float(row["timestamp"]))
    for sequence, timestamps in by_sequence.items():
        timestamps.sort()
        deltas = np.diff(timestamps)
        if len(deltas) and not 0.075 <= float(np.median(deltas)) <= 0.125:
            raise ValueError(f"{sequence}: frozen model expects approximately 10 FPS")
    return rows


def predict_field(
    field: dict[str, Any], motion: np.ndarray, model: dict[str, Any]
) -> dict[str, Any]:
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("frozen feature order mismatch")
    if model.get("raft_sha256") != EXPECTED_RAFT_SHA256:
        raise ValueError("frozen RAFT identity mismatch")
    if field["status"] != "VALID":
        return {"status": field["status"]}
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=np.float64)
    output = {}
    for band in BANDS:
        value = field["bands"][band]
        clearance = value["clearance_m"]
        confidence = value.get("clearance_log1p_confidence")
        if clearance is None or confidence is None:
            output[band] = {"status": "UNKNOWN_CLEARANCE_OR_CONFIDENCE"}
            continue
        probabilities = {}
        for horizon in HORIZONS_M:
            static = np.asarray(
                [
                    float(clearance) - horizon,
                    float(clearance),
                    horizon,
                    float(confidence),
                    float(field["ground_plane_median_residual_m"]),
                    math.log1p(int(value["obstacle_points"])),
                    float(band == "left"),
                    float(band == "center"),
                ]
            )
            features = np.concatenate((static, motion))
            design = np.concatenate(([1.0], (features - mean) / scale))
            probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40)))
            probabilities[str(horizon)] = float(probability)
        output[band] = {
            "status": "VALID",
            "clearance_m": float(clearance),
            "occupancy_probability_by_horizon_m": probabilities,
        }
    return {"status": "VALID", "bands": output}


def render_video(rows: list[dict[str, Any]], output: Path, fps: float = 10.0) -> None:
    first = cv2.imread(rows[0]["frame_path"], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(rows[0]["frame_path"])
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"cannot create video: {output}")
    colors = {"low": (70, 150, 70), "medium": (40, 150, 210), "high": (55, 55, 195)}
    for row in rows:
        frame = cv2.imread(row["frame_path"], cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(row["frame_path"])
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 34), (25, 25, 25), -1)
        cv2.putText(
            overlay,
            "CURRENT occupancy only | 1.5 m metric bands | NOT future prediction",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        strip_top = height - 54
        for index, band in enumerate(BANDS):
            left = round(index * width / 3)
            right = round((index + 1) * width / 3)
            band_result = row["candidate"].get("bands", {}).get(band, {})
            probability = band_result.get("occupancy_probability_by_horizon_m", {}).get("1.5")
            if probability is None:
                color, label = (90, 90, 90), f"{band}: UNKNOWN"
            else:
                level = "high" if probability >= 0.50 else "medium" if probability >= 0.20 else "low"
                color, label = colors[level], f"{band}: P={probability:.2f}"
            cv2.rectangle(overlay, (left, strip_top), (right, height), color, -1)
            cv2.putText(
                overlay,
                label,
                (left + 8, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0))
    writer.release()


def run(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    raft_weights: Path,
    source: UniDepthSource,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    motion_input = [
        {
            "sequence_id": row["sequence_id"],
            "timestamp": row["timestamp"],
            "frame_path": row["frame_path"],
        }
        for row in rows
    ]
    started = time.perf_counter()
    motion = extract_motion(motion_input, raft_weights)
    raft_total_ms = (time.perf_counter() - started) * 1000.0
    outputs = []
    inference_ms = []
    geometry_ms = []
    for row in rows:
        bgr = cv2.imread(row["frame_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["frame_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        depth, metadata = source.infer(rgb, row)
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        field = clearance_field(
            depth,
            intrinsics_matrix(row),
            confidence_map=metadata.get("confidence_map"),
        )
        geometry_ms.append((time.perf_counter() - started) * 1000.0)
        outputs.append(
            {
                "schema": SCHEMA,
                "sequence_id": row["sequence_id"],
                "timestamp": float(row["timestamp"]),
                "frame_path": row["frame_path"],
                "candidate": predict_field(field, motion[row["frame_path"]], model),
            }
        )
    valid = sum(value["candidate"]["status"] == "VALID" for value in outputs)
    summary = {
        "schema": SCHEMA + "_summary",
        "frames": len(outputs),
        "candidate_valid_frames": valid,
        "candidate_valid_fraction": valid / len(outputs),
        "unidepth_mean_ms": statistics.fmean(inference_ms),
        "geometry_mean_ms": statistics.fmean(geometry_ms),
        "raft_cold_total_ms": raft_total_ms,
        "scope": "candidate-only Development output; no truth, alert, or safety claim",
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model-name", default="lpiccinelli/unidepth-v2-vits14")
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--output-video", type=Path)
    args = parser.parse_args()
    rows = load_manifest(args.manifest, args.max_frames)
    source = UniDepthSource(
        args.unidepth_repo,
        args.unidepth_model_name,
        args.unidepth_resolution_level,
        args.device,
    )
    outputs, summary = run(
        rows,
        json.loads(args.model.read_text(encoding="utf-8")),
        args.raft_weights,
        source,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_video is not None:
        render_video(outputs, args.output_video)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
