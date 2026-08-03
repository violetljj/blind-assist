"""Run the frozen known-height operator on consumed TartanGround Development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import core as scale_core


REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import clearance_field  # noqa: E402
from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)
from run_stage_c_d5_tartanground_development_pilot import (  # noqa: E402
    decode_depth,
    load_metadata,
    pose_matrix,
)


BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
INTRINSICS = np.asarray(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
WORLD_UP_NED = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def up_optical_from_pose(pose: np.ndarray) -> np.ndarray:
    _, rotation_local_ned_to_world = pose_matrix(pose)
    up_local_ned = rotation_local_ned_to_world.T @ WORLD_UP_NED
    # local NED [forward, right, down] -> optical [right, down, forward]
    up_optical = np.asarray(
        [up_local_ned[1], up_local_ned[2], up_local_ned[0]], dtype=np.float64
    )
    up_optical /= np.linalg.norm(up_optical)
    return up_optical


def strict_band_values(field: dict[str, Any]) -> dict[str, float] | None:
    if field.get("status") != "VALID":
        return None
    values = {}
    for band in BANDS:
        value = field.get("bands", {}).get(band, {}).get("clearance_m")
        if value is None or not np.isfinite(value):
            return None
        values[band] = float(value)
    return values


def aligned_scale_diagnostic(sensor_depth: np.ndarray, da_depth: np.ndarray) -> float | None:
    sensor = np.asarray(sensor_depth, dtype=np.float64)[::4, ::4]
    candidate = np.asarray(da_depth, dtype=np.float64)[::4, ::4]
    valid = (
        np.isfinite(sensor)
        & np.isfinite(candidate)
        & (sensor >= 0.25)
        & (sensor <= 6.0)
        & (candidate > 0.0)
    )
    if int(np.sum(valid)) < 500:
        return None
    ratios = sensor[valid] / candidate[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    return float(np.median(ratios)) if len(ratios) >= 500 else None


def summarize_arm(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    parent_rows = []
    for parent_id in sorted({row["parent_id"] for row in records}):
        rows = sorted(
            (row for row in records if row["parent_id"] == parent_id),
            key=lambda row: row["anchor_frame_id"],
        )
        truth_exposures = 0
        known_exposures = 0
        errors = []
        agreements = []
        false_clears = []
        temporal_errors = []
        previous = None
        for row in rows:
            truth = row["truth"]
            values = row[arm]
            if truth is None:
                previous = None
                continue
            for band in BANDS:
                truth_exposures += 1
                if values is None:
                    continue
                known_exposures += 1
                errors.append(abs(values[band] - truth[band]))
                for horizon in HORIZONS:
                    truth_occupied = truth[band] <= horizon
                    predicted_occupied = values[band] <= horizon
                    agreements.append(truth_occupied == predicted_occupied)
                    false_clears.append(truth_occupied and not predicted_occupied)
            if previous is not None and values is not None and previous[arm] is not None:
                for band in BANDS:
                    truth_delta = truth[band] - previous["truth"][band]
                    arm_delta = values[band] - previous[arm][band]
                    temporal_errors.append(abs(arm_delta - truth_delta))
            previous = row if values is not None else None
        parent_rows.append(
            {
                "parent_id": parent_id,
                "truth_band_exposures": truth_exposures,
                "known_band_exposures": known_exposures,
                "known_coverage": known_exposures / truth_exposures if truth_exposures else 0.0,
                "clearance_mae_m": float(np.mean(errors)) if errors else None,
                "envelope_agreement": float(np.mean(agreements)) if agreements else None,
                "false_clear_rate": float(np.mean(false_clears)) if false_clears else None,
                "temporal_delta_mae_m": float(np.mean(temporal_errors)) if temporal_errors else None,
            }
        )
    metric_names = (
        "known_coverage",
        "clearance_mae_m",
        "envelope_agreement",
        "false_clear_rate",
        "temporal_delta_mae_m",
    )
    parent_macro = {}
    for name in metric_names:
        values = [row[name] for row in parent_rows if row[name] is not None]
        parent_macro[name] = float(np.mean(values)) if values else None
    return {"parent_macro": parent_macro, "parents": parent_rows}


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_root.exists():
        raise FileExistsError(arguments.output_root)
    amendment = json.loads(arguments.amendment.read_text(encoding="utf-8"))
    if amendment.get("status") != "USER_AUTHORIZED_CONSUMED_DATA_FROZEN_BEFORE_DA_AND_EFFECT_EXECUTION":
        raise ValueError("consumed Development amendment is not frozen")
    eligible = {
        row["parent_id"]: float(row["robot_height_m"])
        for row in amendment["height_eligible_parents"]
    }
    manifest = json.loads((arguments.corpus_root / "manifest.json").read_text(encoding="utf-8"))
    source_by_parent = {row["parent_id"]: row for row in manifest["sources"]}
    if set(eligible) - set(source_by_parent):
        raise ValueError("amendment parents missing from corpus manifest")
    if sha256(arguments.dav2_checkpoint) != "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545":
        raise ValueError("unexpected DA V2 checkpoint")

    selected_samples = [
        row
        for row in load_jsonl(arguments.corpus_root / "samples.jsonl")
        if row["parent_id"] in eligible
    ]
    if len(selected_samples) != len(eligible) * 33:
        raise ValueError("expected 33 frozen anchors per eligible parent")

    arguments.output_root.mkdir(parents=True)
    prediction_root = arguments.output_root / "predictions"
    source = DepthAnythingV2MetricSource(
        arguments.dav2_repo,
        arguments.dav2_checkpoint,
        arguments.device,
        input_size=518,
        precision="fp16" if arguments.device.startswith("cuda") else "fp32",
    )
    metadata_cache = {
        parent_id: load_metadata(arguments.metadata_root, parent_id)
        for parent_id in eligible
    }
    records = []
    for index, sample in enumerate(selected_samples, 1):
        parent_id = sample["parent_id"]
        anchor = int(sample["anchor_frame_id"])
        current = sample["history_rgb"][-1]
        if int(current["frame_id"]) != anchor or float(current["relative_time_s"]) != 0.0:
            raise ValueError("sample current-frame binding mismatch")
        rgb_path = Path(current["image_path"])
        if sha256(rgb_path).lower() != str(current["image_sha256"]).lower():
            raise ValueError("RGB hash mismatch")
        depth_path = rgb_path.parent.parent / "depth" / f"{anchor:06d}.png"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (640, 640):
            raise ValueError("unexpected RGB")
        sensor_depth = decode_depth(depth_path.read_bytes())
        metadata, poses = metadata_cache[parent_id]
        height_m = float(metadata["robot_height"])
        if height_m != eligible[parent_id] or anchor >= len(poses):
            raise ValueError("height or pose binding mismatch")
        up_optical = up_optical_from_pose(poses[anchor])
        truth_field = clearance_field(
            sensor_depth,
            INTRINSICS,
            plane_override=(up_optical, height_m, 0.0),
        )
        truth = strict_band_values(truth_field)

        started = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        da_depth, _ = source.infer(rgb, {})
        latency_ms = (time.perf_counter() - started) * 1000.0
        if da_depth.shape != sensor_depth.shape:
            raise ValueError("DA/sensor depth shape mismatch")
        prediction_path = prediction_root / parent_id.replace("/", "__") / f"{anchor:06d}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.exists():
            raise FileExistsError(prediction_path)
        np.savez_compressed(prediction_path, da_depth=da_depth.astype(np.float32))

        raw = strict_band_values(clearance_field(da_depth, INTRINSICS))
        receipt = scale_core.CameraHeightReceipt(parent_id, parent_id, height_m, 0.0)
        recovery = scale_core.recover_metric_scale(
            da_depth, INTRINSICS, receipt, parent_id, parent_id
        )
        candidate = None
        scale = None
        scale_reason = None
        normalized_plane_residual = None
        if recovery["status"] == "VALID":
            plane = recovery["ground"]
            scale = float(recovery["scale"])
            normalized_plane_residual = float(plane.normalized_median_residual)
            candidate_field = clearance_field(
                recovery["metric_depth"],
                INTRINSICS,
                plane_override=(
                    plane.normal,
                    height_m,
                    plane.normalized_median_residual * height_m,
                ),
            )
            candidate = strict_band_values(candidate_field)
            if candidate is None:
                scale_reason = "STRICT_CLEARANCE_BAND_UNKNOWN"
        else:
            scale_reason = str(recovery["reason"])
        oracle_scale = aligned_scale_diagnostic(sensor_depth, da_depth)
        records.append(
            {
                "parent_id": parent_id,
                "environment": sample["environment"],
                "anchor_frame_id": anchor,
                "rgb_sha256": current["image_sha256"],
                "depth_sha256": sha256(depth_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256(prediction_path),
                "latency_ms": latency_ms,
                "height_m": height_m,
                "truth": truth,
                "raw": raw,
                "candidate": candidate,
                "candidate_unknown_reason": scale_reason,
                "candidate_scale": scale,
                "normalized_plane_residual": normalized_plane_residual,
                "aligned_scale_diagnostic": oracle_scale,
                "candidate_scale_relative_error_diagnostic": abs(scale / oracle_scale - 1.0)
                if scale is not None and oracle_scale is not None
                else None,
            }
        )
        if index % 20 == 0:
            print(json.dumps({"processed": index, "total": len(selected_samples)}), flush=True)

    raw_summary = summarize_arm(records, "raw")
    candidate_summary = summarize_arm(records, "candidate")
    raw_by_parent = {row["parent_id"]: row for row in raw_summary["parents"]}
    jointly_better = []
    for row in candidate_summary["parents"]:
        raw_row = raw_by_parent[row["parent_id"]]
        better = (
            row["clearance_mae_m"] is not None
            and raw_row["clearance_mae_m"] is not None
            and row["clearance_mae_m"] < raw_row["clearance_mae_m"]
            and row["false_clear_rate"] is not None
            and raw_row["false_clear_rate"] is not None
            and row["false_clear_rate"] <= raw_row["false_clear_rate"]
        )
        jointly_better.append({"parent_id": row["parent_id"], "jointly_better": better})
    scale_errors = [
        row["candidate_scale_relative_error_diagnostic"]
        for row in records
        if row["candidate_scale_relative_error_diagnostic"] is not None
    ]
    macro = candidate_summary["parent_macro"]
    gates = {
        "known_coverage": macro["known_coverage"] is not None and macro["known_coverage"] >= 0.60,
        "clearance_mae": macro["clearance_mae_m"] is not None and macro["clearance_mae_m"] <= 0.25,
        "envelope_agreement": macro["envelope_agreement"] is not None and macro["envelope_agreement"] >= 0.90,
        "false_clear": macro["false_clear_rate"] is not None and macro["false_clear_rate"] <= 0.05,
        "temporal_delta_mae": macro["temporal_delta_mae_m"] is not None and macro["temporal_delta_mae_m"] <= 0.15,
        "jointly_better_parents": sum(row["jointly_better"] for row in jointly_better) >= 3,
    }
    unknown_reasons: dict[str, int] = {}
    for row in records:
        if row["candidate"] is None:
            reason = row["candidate_unknown_reason"] or "UNKNOWN"
            unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
    result = {
        "schema": "blindassist_known_camera_height_ground_scale_consumed_tartanground_development_result",
        "data_role": "CONSUMED_DEVELOPMENT",
        "claim_ceiling": amendment["claim_ceiling"],
        "amendment_sha256": sha256(arguments.amendment),
        "corpus_manifest_sha256": sha256(arguments.corpus_root / "manifest.json"),
        "corpus_samples_sha256": sha256(arguments.corpus_root / "samples.jsonl"),
        "dav2_checkpoint_sha256": sha256(arguments.dav2_checkpoint),
        "record_count": len(records),
        "parent_count": len(eligible),
        "records": records,
        "raw_da": raw_summary,
        "known_height_candidate": candidate_summary,
        "jointly_better_parents": jointly_better,
        "candidate_unknown_reason_counts": unknown_reasons,
        "scale_diagnostic": {
            "count": len(scale_errors),
            "median_absolute_relative_error": float(np.median(scale_errors)) if scale_errors else None,
            "p90_absolute_relative_error": float(np.quantile(scale_errors, 0.90)) if scale_errors else None,
            "authority": "POSTHOC_PIXEL_MEDIAN_ALIGNMENT_DIAGNOSTIC_NOT_INDEPENDENT_TRUTH",
        },
        "latency_ms": {
            "median": float(np.median([row["latency_ms"] for row in records])),
            "p95": float(np.quantile([row["latency_ms"] for row in records], 0.95)),
        },
        "gates": gates,
        "terminal": "CONSUMED_DEVELOPMENT_KNOWN_HEIGHT_SIGNAL_SUPPORTED"
        if all(gates.values())
        else "CONSUMED_DEVELOPMENT_KNOWN_HEIGHT_NOT_SUPPORTED_DIAGNOSE",
    }
    write_json_new(arguments.output_root / "result.json", result)
    print(
        json.dumps(
            {
                "record_count": len(records),
                "raw_da": raw_summary["parent_macro"],
                "known_height_candidate": candidate_summary["parent_macro"],
                "jointly_better_parents": jointly_better,
                "candidate_unknown_reason_counts": unknown_reasons,
                "scale_diagnostic": result["scale_diagnostic"],
                "latency_ms": result["latency_ms"],
                "gates": gates,
                "terminal": result["terminal"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
