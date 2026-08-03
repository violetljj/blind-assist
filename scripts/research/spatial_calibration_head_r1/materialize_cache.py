#!/usr/bin/env python3
"""Materialize the development-only ARKitScenes feature/truth cache for R1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import regional_feature_inputs
from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, sha256, validate

HFTF_DIR = REPO_ROOT / "scripts/research/hftf"
SUCCESSOR_DIR = REPO_ROOT / "scripts/research/metric_depth_successors_r0"
DEPENDENCY_DIR = REPO_ROOT / "artifacts.local/vendor/python-packages-hftf-metric-depth-r0"
for path in (DEPENDENCY_DIR, HFTF_DIR, SUCCESSOR_DIR):
    sys.path.insert(0, str(path))

from common import fit_dense_affine
from evaluate_metric3d_clearance_field_a0 import clearance_field
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource, intrinsics_matrix


def timestamp_from_stem(stem: str) -> float:
    try:
        return float(stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"cannot parse timestamp from {stem}") from error


def matched_frame_stems(video_root: Path) -> list[str]:
    folders = [video_root / name for name in ("lowres_wide", "lowres_depth", "confidence")]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    stem_sets = [{path.stem for path in folder.glob("*.png")} for folder in folders]
    common = set.intersection(*stem_sets)
    return sorted(common, key=lambda stem: (timestamp_from_stem(stem), stem))


def sample_150(stems: list[str]) -> list[str]:
    if len(stems) < 150:
        raise ValueError("fewer than 150 matched frame triples")
    indices = np.round(np.linspace(0, len(stems) - 1, 150)).astype(int)
    if len(set(indices.tolist())) != 150:
        raise ValueError("sampling produced duplicate indices")
    return [stems[index] for index in indices]


def intrinsics_files(video_root: Path) -> list[tuple[float, Path]]:
    folder = video_root / "lowres_wide_intrinsics"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted((timestamp_from_stem(path.stem), path) for path in folder.glob("*.pincam"))


def nearest_intrinsics(timestamp: float, candidates: list[tuple[float, Path]]) -> Path:
    differences = [(abs(value - timestamp), path) for value, path in candidates]
    if not differences:
        raise ValueError("no intrinsics files")
    difference, path = min(differences, key=lambda row: (row[0], str(row[1])))
    if difference > 0.0015:
        raise ValueError(f"no intrinsics within 1.5 ms of {timestamp}")
    if sum(abs(value - timestamp) == difference for value, _path in candidates) != 1:
        raise ValueError("ambiguous nearest intrinsics")
    return path


def read_intrinsics(path: Path, expected_width: int, expected_height: int) -> list[float]:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    if len(values) != 6:
        raise ValueError(f"invalid pincam: {path}")
    width, height, fx, fy, cx, cy = values
    if int(width) != expected_width or int(height) != expected_height:
        raise ValueError("intrinsics/image size mismatch")
    return [fx, fy, cx, cy]


def dav2_depth_tokens_cls(source: DepthAnythingV2MetricSource, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = source.model
    image, (height, width) = model.image2tensor(bgr, source.input_size)
    patch_height, patch_width = image.shape[-2] // 14, image.shape[-1] // 14
    with source.torch.inference_mode(), source.torch.autocast(
        device_type=source.device.type,
        dtype=source.torch.float16,
        enabled=source.precision == "fp16",
    ):
        features = model.pretrained.get_intermediate_layers(
            image, model.intermediate_layer_idx[model.encoder], return_class_token=True
        )
        depth = model.depth_head(features, patch_height, patch_width) * model.max_depth
        depth = source.torch.nn.functional.interpolate(
            depth, (height, width), mode="bilinear", align_corners=True
        )[0, 0]
        patch = features[-1][0][0].reshape(patch_height, patch_width, 384)
        cls = features[-1][1][0]
    return (
        depth.float().cpu().numpy().astype(np.float32),
        patch.float().cpu().numpy().astype(np.float32),
        cls.float().cpu().numpy().astype(np.float32),
    )


def band_clearances(field: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(3, np.nan, dtype=np.float32)
    valid = np.zeros(3, dtype=bool)
    if field.get("status") != "VALID":
        return values, valid
    for index, name in enumerate(("left", "center", "right")):
        value = field.get("bands", {}).get(name, {}).get("clearance_m")
        if value is not None and np.isfinite(value):
            values[index] = float(value)
            valid[index] = True
    return values, valid


def load_roster(lock_path: Path, protocol_sha256: str) -> list[dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "METADATA_ROSTER_24_LOCKED_MEDIA_UNOPENED_LICENSE_REVIEW_REQUIRED":
        raise ValueError("unexpected roster lock state")
    if lock.get("protocol_sha256") != protocol_sha256:
        raise ValueError("roster lock protocol mismatch")
    rows = []
    for role in ("train", "validation"):
        for row in lock["roles"][role]:
            rows.append({**row, "role": role, "official_fold": "Training"})
    if len(rows) != 20 or len({row["visit_id"] for row in rows}) != 20:
        raise ValueError("development roster must contain 20 unique parents")
    return rows


def write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    if partial.exists():
        raise FileExistsError(partial)
    np.savez_compressed(partial, **arrays)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--roster-lock", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing raw/Training/<video_id>")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    receipt = json.loads(args.license_receipt.read_text(encoding="utf-8"))
    if receipt.get("media_download_authorized") is not True or receipt.get("license_sha256") != protocol["data"]["license_sha256"]:
        raise ValueError("explicit bound license receipt required")
    roster = load_roster(args.roster_lock, sha256(args.protocol))
    args.output_root.mkdir(parents=True)

    source = DepthAnythingV2MetricSource(
        REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main",
        REPO_ROOT / protocol["dav2"]["checkpoint_path"],
        args.device,
        input_size=int(protocol["dav2"]["input_size"]),
        precision=protocol["dav2"]["precision"],
    )
    arrays: dict[str, list[Any]] = {name: [] for name in ("region_inputs", "raw_clearance", "truth_clearance", "truth_valid", "cls_features", "affine_targets", "affine_valid")}
    records = []
    latencies = []
    for roster_row in roster:
        video_id = str(roster_row["video_id"])
        video_root = args.dataset_root / "raw" / roster_row["official_fold"] / video_id
        trajectory = video_root / "lowres_wide.traj"
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        candidates = intrinsics_files(video_root)
        for stem in sample_150(matched_frame_stems(video_root)):
            rgb_path = video_root / "lowres_wide" / f"{stem}.png"
            depth_path = video_root / "lowres_depth" / f"{stem}.png"
            confidence_path = video_root / "confidence" / f"{stem}.png"
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth_raw is None or confidence is None:
                raise OSError(f"decode failed: {stem}")
            if bgr.shape[:2] != depth_raw.shape or depth_raw.shape != confidence.shape:
                raise ValueError("RGB/depth/confidence shape mismatch")
            height, width = depth_raw.shape
            timestamp = timestamp_from_stem(stem)
            intrinsics_path = nearest_intrinsics(timestamp, candidates)
            intrinsics = read_intrinsics(intrinsics_path, width, height)
            truth_depth = depth_raw.astype(np.float32) / 1000.0
            truth_mask = (confidence == 2) & np.isfinite(truth_depth) & (truth_depth >= 0.25) & (truth_depth <= 6.0)
            truth_depth[~truth_mask] = np.nan
            truth_fraction = float(np.mean(truth_mask))
            started = time.perf_counter()
            da_depth, patch, cls = dav2_depth_tokens_cls(source, bgr)
            latencies.append((time.perf_counter() - started) * 1000.0)
            matrix = intrinsics_matrix({"intrinsics_fx_fy_cx_cy": intrinsics})
            raw_clearance, _raw_valid = band_clearances(clearance_field(da_depth, matrix))
            truth_clearance, truth_valid = band_clearances(clearance_field(truth_depth, matrix))
            if truth_fraction < float(protocol["data"]["truth"]["minimum_truth_valid_fraction_per_frame"]):
                truth_valid[:] = False
            fit = fit_dense_affine(da_depth, truth_depth, protocol["arms"]["global_affine_label_fit"])
            affine_valid = fit.get("status") == "VALID"
            affine_target = [fit.get("slope", np.nan), fit.get("intercept_m", np.nan)]
            arrays["region_inputs"].append(regional_feature_inputs(patch, da_depth, intrinsics))
            arrays["raw_clearance"].append(raw_clearance)
            arrays["truth_clearance"].append(truth_clearance)
            arrays["truth_valid"].append(truth_valid)
            arrays["cls_features"].append(cls)
            arrays["affine_targets"].append(affine_target)
            arrays["affine_valid"].append(affine_valid)
            records.append({
                "parent_id": str(roster_row["visit_id"]), "video_id": video_id,
                "timestamp": timestamp, "frame_stem": stem, "role": roster_row["role"],
                "cv_fold": roster_row.get("cv_fold"), "truth_confidence2_fraction": truth_fraction,
                "intrinsics_path": str(intrinsics_path.resolve()), "affine_status": fit["status"],
            })
    output_arrays = {name: np.asarray(values) for name, values in arrays.items()}
    array_path = args.output_root / "arrays.npz"
    write_npz_new(array_path, output_arrays)
    manifest = {
        "schema": "blindassist_spatial_calibration_head_r1_cache",
        "protocol_sha256": sha256(args.protocol),
        "roster_lock_sha256": sha256(args.roster_lock),
        "license_receipt_sha256": sha256(args.license_receipt),
        "records": records,
        "arrays": {"path": str(array_path.resolve()), "sha256": sha256(array_path)},
        "sealed_truth_included": False,
        "sealed_media_opened": False,
        "frame_count": len(records),
        "dav2_latency_ms": {"mean": float(np.mean(latencies)), "p95": float(np.quantile(latencies, 0.95))},
        "terminal": "SPATIAL_CALIBRATION_HEAD_R1_DEVELOPMENT_CACHE_MATERIALIZED",
    }
    write_json_new(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
