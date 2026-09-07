"""Diagnose saved depth/reference disagreement without model execution or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HEIGHTS = ("low", "body", "head")
EDGES = np.asarray((.065, .65, 1.4, 1.85), dtype=np.float32)
BUCKETS = (
    "prediction_nonfinite", "prediction_depth_outside_0p08_12",
    "prediction_forward_nonpositive", "prediction_forward_over_3m",
    "prediction_height_below_0p065m", "prediction_height_at_or_above_1p85m",
    "retained_changed_height_band", "retained_same_height_band",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite diagnostic values require an explicit denominator")
    return {"count": int(values.size), "mean": float(values.mean()),
            **{key: float(np.quantile(values, quantile))
               for key, quantile in (("p05", .05), ("p50", .50), ("p95", .95))}}


def counted(labels: np.ndarray, mask: np.ndarray) -> dict:
    selected = labels[mask]
    if np.any(selected < 0):
        raise AssertionError("Some native eligible pixels were not classified")
    counts = {name: int((selected == index).sum()) for index, name in enumerate(BUCKETS)}
    if sum(counts.values()) != int(mask.sum()):
        raise AssertionError("Disjoint buckets must partition the native eligible denominator")
    return {"native_eligible_pixels": int(mask.sum()), "buckets": counts}


def add_counts(target: dict, source: dict):
    target["native_eligible_pixels"] += source["native_eligible_pixels"]
    for name in BUCKETS:
        target["buckets"][name] += source["buckets"][name]


def empty_counts() -> dict:
    return {"native_eligible_pixels": 0, "buckets": {name: 0 for name in BUCKETS}}


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Sensor path escapes the recorded model directory")
    return path


def analyze(run: Path, output: Path) -> dict:
    run, output = run.resolve(strict=True), output.resolve()
    if output.exists() or not output.is_relative_to(run):
        raise ValueError("Output must be new and remain inside the original run directory")
    receipt_path = run / "result.json"
    before = {receipt_path: sha(receipt_path)}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("status") != "COMPLETED" or receipt.get("attention_forward_m") != 3.0:
        raise ValueError("Require the completed fixed 3 m representation probe")
    replay = receipt["replay"]
    model_dir = Path(replay["source"]["model_dir"]).resolve(strict=True)
    manifest_path = model_dir / "manifest.json"
    manifest_hash = sha(manifest_path)
    if manifest_hash != replay["source"]["manifest_sha256"] or manifest_hash != receipt["source_manifest_sha256"]:
        raise ValueError("Source manifest no longer matches the run receipt")
    before[manifest_path] = manifest_hash
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "ue-sample-segment-v1" or len(manifest["episodes"]) != 1:
        raise ValueError("Unexpected sensor source schema")
    frames = manifest["episodes"][0]["frames"]
    recorded = replay["rows"]
    if len(frames) != 11 or len(recorded) != 11:
        raise ValueError("Diagnose all eleven fixed frames exactly once")
    calibration = manifest["calibration"]
    h, w = int(calibration["height"]), int(calibration["width"])
    v, u = np.mgrid[:h, :w].astype(np.float32)
    fx = w / (2 * math.tan(math.radians(float(calibration["horizontal_fov_degrees"]) / 2)))
    ray_y = (u - (w - 1) / 2) / fx
    ray_up = -(v - (h - 1) / 2) / fx
    verified = []
    # Validate every receipt hash before inspecting arrays or deriving outcomes.
    for index, (frame, record) in enumerate(zip(frames, recorded)):
        if frame["sample_index"] != index or record["sample_index"] != index:
            raise ValueError("Recorded and source frame identities disagree")
        native_path, rgb_path = (contained(model_dir, frame[key]) for key in ("depth_path", "rgb_path"))
        predicted_path = Path(record["predicted_path"]).resolve(strict=True)
        if not predicted_path.is_relative_to(run):
            raise ValueError("Saved prediction must belong to this run")
        for path, expected in ((native_path, record["native_depth_sha256"]),
                               (rgb_path, record["rgb_sha256"]),
                               (predicted_path, record["predicted_sha256"])):
            actual = sha(path)
            if actual != expected:
                raise ValueError("Input hash mismatch: " + str(path))
            before[path] = actual
        verified.append((frame, record, native_path, predicted_path))
    total, native_alert_subset = empty_counts(), empty_counts()
    by_band = {height: empty_counts() for height in HEIGHTS}
    rows, samples = [], {key: [] for key in (
        "native_depth_m", "predicted_depth_m", "predicted_minus_native_depth_m",
        "predicted_minus_native_forward_m", "predicted_div_native_depth",
        "native_height_m", "predicted_height_m", "predicted_minus_native_height_m")}
    cross_total = {x: {z: 0 for z in ("below", "within", "above")} for x in ("nonpositive", "near", "far")}
    native_alert_cells = 0
    for frame, record, native_path, predicted_path in verified:
        native = np.load(native_path, allow_pickle=False)
        predicted = np.load(predicted_path, allow_pickle=False)
        if native.dtype != np.float32 or predicted.dtype != np.float32 or native.shape != (h, w) or predicted.shape != (h, w):
            raise ValueError("Saved depth raster contract changed")
        native.flags.writeable = predicted.flags.writeable = False
        cam, wearer = frame["camera_transform"], frame["wearer_transform"]
        camera_height = float(cam["z"] - wearer["z"])
        if cam["yaw"] != wearer["yaw"] or cam["roll"] != 0 or wearer["roll"] != 0:
            raise ValueError("Unsupported yaw/roll in the recorded calibration")
        if not np.isclose(camera_height, record["camera"]["camera_height_m"]) or cam["pitch"] != record["camera"]["pitch_deg"]:
            raise ValueError("Camera source and run receipt disagree")
        pitch = math.radians(cam["pitch"])
        ray_x = math.cos(pitch) - math.sin(pitch) * ray_up
        ray_z = math.sin(pitch) + math.cos(pitch) * ray_up
        native_x, predicted_x = native * ray_x, predicted * ray_x
        native_height, predicted_height = camera_height + native * ray_z, camera_height + predicted * ray_z
        native_band = np.searchsorted(EDGES, native_height, side="right") - 1
        predicted_band = np.searchsorted(EDGES, predicted_height, side="right") - 1
        eligible = (np.isfinite(native) & (native > .08) & (native < 12) & (ray_x > 0)
                    & (native_x <= 3) & (native_band >= 0) & (native_band < 3))
        valid_pred = np.isfinite(predicted) & (predicted > .08) & (predicted < 12)
        conditions = (~np.isfinite(predicted), ~valid_pred,
                      predicted_x <= 0, predicted_x > 3,
                      predicted_height < EDGES[0], predicted_height >= EDGES[-1],
                      predicted_band != native_band, np.ones((h, w), dtype=bool))
        labels = np.full((h, w), -1, dtype=np.int8)
        for index, condition in enumerate(conditions):
            labels[eligible & (labels < 0) & condition] = index
        partition = counted(labels, eligible)
        add_counts(total, partition)
        band_rows = {}
        for index, height in enumerate(HEIGHTS):
            band_rows[height] = counted(labels, eligible & (native_band == index))
            add_counts(by_band[height], band_rows[height])
        direction = np.searchsorted(np.radians([-10., 10.]), np.arctan2(ray_y, ray_x), side="right")
        alert_cells = np.asarray(record["native"]["near_field"]["alerts"], dtype=bool)
        native_alert_cells += int(alert_cells.sum())
        alert_pixels = eligible & alert_cells[direction, native_band.clip(0, 2)]
        alert_partition = counted(labels, alert_pixels)
        add_counts(native_alert_subset, alert_partition)
        cross = {x: {z: 0 for z in ("below", "within", "above")} for x in cross_total}
        valid_subset = eligible & valid_pred
        x_masks = {"nonpositive": predicted_x <= 0, "near": (predicted_x > 0) & (predicted_x <= 3), "far": predicted_x > 3}
        z_masks = {"below": predicted_height < EDGES[0],
                   "within": (predicted_height >= EDGES[0]) & (predicted_height < EDGES[-1]),
                   "above": predicted_height >= EDGES[-1]}
        for x, x_mask in x_masks.items():
            for z, z_mask in z_masks.items():
                cross[x][z] = int((valid_subset & x_mask & z_mask).sum())
                cross_total[x][z] += cross[x][z]
        if sum(sum(columns.values()) for columns in cross.values()) != int(valid_subset.sum()):
            raise AssertionError("Distance/height cross-table must partition valid predictions")
        finite_subset = eligible & np.isfinite(predicted)
        values = {"native_depth_m": native[finite_subset], "predicted_depth_m": predicted[finite_subset],
                  "predicted_minus_native_depth_m": predicted[finite_subset] - native[finite_subset],
                  "predicted_minus_native_forward_m": predicted_x[finite_subset] - native_x[finite_subset],
                  "predicted_div_native_depth": predicted[finite_subset] / native[finite_subset],
                  "native_height_m": native_height[finite_subset], "predicted_height_m": predicted_height[finite_subset],
                  "predicted_minus_native_height_m": predicted_height[finite_subset] - native_height[finite_subset]}
        for key, value in values.items():
            samples[key].append(value)
        rows.append({"sample_index": frame["sample_index"], "partition": partition,
                     "by_native_height_band": band_rows, "native_alert_cell_pixels": alert_partition,
                     "distance_by_height_for_valid_predictions": cross,
                     "finite_prediction_reference_diagnostics": {key: distribution(value) for key, value in values.items()}})
    for path, original in before.items():
        if sha(path) != original:
            raise AssertionError("An input changed during the read-only diagnosis: " + str(path))
    result = {"schema": "nearfield-saved-depth-loss-diagnosis-v1", "frames": 11,
              "backend": {"device": "cpu", "workload": "IO_ETL_AND_SCALAR_DIAGNOSTICS", "reason": "TASK_NOT_GPU_SUITABLE"},
              "code_sha256": sha(Path(__file__)), "run_receipt_sha256": before[receipt_path],
              "source_manifest_sha256": manifest_hash,
              "validated_input_files": {str(path): digest for path, digest in before.items()},
              "verified_receipt_bound_sensor_files": len(before) - 1,
              "inputs_unchanged_after_analysis": True, "all_buckets_partition_native_pixels": True,
              "denominator": "Native valid forward depth 0.08<d<12; gravity-aligned x in (0,3]; world height [0.065,1.85).",
              "disjoint_bucket_precedence": list(BUCKETS), "total": total, "by_native_height_band": by_band,
              "native_alert_cells": native_alert_cells, "native_alert_cell_pixels": native_alert_subset,
              "distance_by_height_for_valid_predictions": cross_total,
              "finite_prediction_reference_diagnostics": {key: distribution(np.concatenate(value)) for key, value in samples.items()},
              "rows": rows,
              "limits": ["Privileged synthetic native reference and recorded camera calibration; not independent obstacle ground truth.",
                         "Pixels are same-ray correspondences; no model rerun, scale fitting, parameter changes or new source.",
                         "Depth changes affect forward distance and reconstructed height jointly; cross-table avoids hiding overlapping effects.",
                         "This partition precedes spatial support and representation reduction, so it cannot attribute retained losses to support without further evidence.",
                         "No statements about unseen pixels, natural categories, phone performance or real-world safety."]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": sha(args.output),
                      "total": result["total"], "cross_table": result["distance_by_height_for_valid_predictions"],
                      "native_alert_cells": result["native_alert_cells"], "verified_files": result["verified_receipt_bound_sensor_files"]}))


if __name__ == "__main__":
    main()
