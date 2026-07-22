from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contract import parse_rows, sha256, write_json
from lilocbench_calibration import (
    compose_transform_chain,
    parse_depth_to_color_yaml,
    parse_intrinsics_yaml,
    parse_transformations_yaml,
    register_depth_to_color,
    transform_matrix,
    validate_front_color_optical,
)


RECEIPT_SCHEMA = "blindassist_ustrf_lilocbench_rgbd_preparation_v1"


def _image_rows(path: Path) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    for image in path.glob("*.png"):
        try:
            stamp = float(image.stem)
        except ValueError as error:
            raise ValueError(f"non-timestamp LILocBench image: {image.name}") from error
        if not math.isfinite(stamp):
            raise ValueError(f"non-finite LILocBench image timestamp: {image.name}")
        rows.append((stamp, image))
    rows.sort(key=lambda row: row[0])
    if not rows or any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"empty or non-monotonic LILocBench image directory: {path}")
    return rows


def _associate_sorted(
    color: list[tuple[float, Path]],
    depth: list[tuple[float, Path]],
    maximum_delta_s: float,
) -> list[tuple[float, Path, float, Path]]:
    """Return monotonic one-to-one nearest pairs within the frozen delta."""
    if not math.isfinite(maximum_delta_s) or maximum_delta_s <= 0:
        raise ValueError("invalid LILocBench RGB-depth association delta")
    result: list[tuple[float, Path, float, Path]] = []
    depth_index = 0
    for color_stamp, color_path in color:
        while depth_index + 1 < len(depth) and abs(depth[depth_index + 1][0] - color_stamp) < abs(depth[depth_index][0] - color_stamp):
            depth_index += 1
        depth_stamp, depth_path = depth[depth_index]
        if abs(depth_stamp - color_stamp) <= maximum_delta_s:
            result.append((color_stamp, color_path, depth_stamp, depth_path))
            depth_index += 1
            if depth_index >= len(depth):
                break
    if not result or any(a[0] >= b[0] or a[2] >= b[2] for a, b in zip(result, result[1:])):
        raise ValueError("empty or non-monotonic LILocBench RGB-depth association")
    return result


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _hash_chain_update(digest: Any, relative: str, file_hash: str) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_hash.encode("ascii"))
    digest.update(b"\n")


def sanitize_raw_depth(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Map the archive's uint16 zero/saturation sentinels to unknown depth."""
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("invalid LILocBench uint16 depth raster")
    saturated = raw == np.iinfo(np.uint16).max
    count = int(np.count_nonzero(saturated))
    if count == 0:
        return raw, 0
    sanitized = raw.copy()
    sanitized[saturated] = 0
    return sanitized, count


def prepare(
    source_root: Path,
    ground_truth: Path,
    manifest_path: Path,
    output: Path,
    maximum_delta_ms: float,
    depth_scale_units_per_meter: float,
    limit: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    ground_truth = ground_truth.resolve()
    output = output.resolve()
    work = output.with_name(output.name + ".incomplete")
    if output.exists() or work.exists():
        raise ValueError(f"refusing to overwrite prepared LILocBench package: {output}")
    if not ground_truth.is_file():
        raise ValueError(f"missing LILocBench ground truth: {ground_truth}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest["sequence"]["individual_files_rgbd_archive"]
    camera_plan = archive["camera_optical_plan"]

    paths = {
        "transformations": source_root / "transformations.yaml",
        "color_intrinsics": source_root / "camera_front/color/intrinsics.yaml",
        "depth_intrinsics": source_root / "camera_front/depth/intrinsics.yaml",
        "depth_to_color": source_root / "camera_front/extrinsics_depth_to_color.yaml",
    }
    expected_hashes = {
        "transformations": camera_plan["expected_transformations_yaml_sha256"],
        "color_intrinsics": camera_plan["expected_color_intrinsics_yaml_sha256"],
        "depth_intrinsics": camera_plan["expected_depth_intrinsics_yaml_sha256"],
        "depth_to_color": camera_plan["expected_depth_to_color_yaml_sha256"],
    }
    actual_hashes = {name: sha256(path) for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise ValueError("LILocBench calibration member hash mismatch")
    if sha256(ground_truth) != manifest["sequence"]["ground_truth"]["sha256"]:
        raise ValueError("LILocBench ground-truth hash mismatch")

    color_calibration = parse_intrinsics_yaml(paths["color_intrinsics"])
    depth_calibration = parse_intrinsics_yaml(paths["depth_intrinsics"])
    color_from_depth = transform_matrix(parse_depth_to_color_yaml(paths["depth_to_color"]))
    transforms = parse_transformations_yaml(paths["transformations"])
    transform_chain = camera_plan["transform_chain"]
    base_from_color = compose_transform_chain(transforms, transform_chain)
    forward_axis = validate_front_color_optical(base_from_color)

    color = _image_rows(source_root / "camera_front/color/images")
    depth = _image_rows(source_root / "camera_front/depth/images")
    associated = _associate_sorted(color, depth, maximum_delta_ms / 1000.0)
    if limit > 0:
        associated = associated[:limit]
    minimum_fraction = 0.95
    associated_fraction = len(associated) / max(len(color), len(depth))
    if limit <= 0 and associated_fraction < minimum_fraction:
        raise ValueError(f"LILocBench RGB-depth association fraction {associated_fraction:.6f} < {minimum_fraction}")

    (work / "color/images").mkdir(parents=True)
    (work / "aligned_depth/images").mkdir(parents=True)
    raw_chain = hashlib.sha256()
    aligned_chain = hashlib.sha256()
    color_rows: list[str] = []
    aligned_rows: list[str] = []
    deltas_ms: list[float] = []
    valid_fractions: list[float] = []
    link_modes: set[str] = set()
    saturated_depth_pixel_count = 0
    for index, (color_stamp, color_path, depth_stamp, depth_path) in enumerate(associated):
        color_relative = f"color/images/{color_path.name}"
        color_target = work / color_relative
        link_modes.add(_link_or_copy(color_path, color_target))
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(f"invalid LILocBench uint16 depth PNG: {depth_path}")
        raw, saturated_count = sanitize_raw_depth(raw)
        saturated_depth_pixel_count += saturated_count
        aligned_m = register_depth_to_color(
            raw,
            depth_scale_units_per_meter,
            depth_calibration,
            color_calibration,
            color_from_depth,
        )
        aligned_units = np.rint(aligned_m * depth_scale_units_per_meter)
        if not np.all(np.isfinite(aligned_units)) or np.any(aligned_units < 0) or np.any(aligned_units > np.iinfo(np.uint16).max):
            raise ValueError(f"out-of-range LILocBench aligned depth: {depth_path}")
        aligned_relative = f"aligned_depth/images/{depth_path.name}"
        aligned_target = work / aligned_relative
        if not cv2.imwrite(str(aligned_target), aligned_units.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"failed to write LILocBench aligned depth: {aligned_target}")
        _hash_chain_update(raw_chain, depth_path.name, sha256(depth_path))
        _hash_chain_update(aligned_chain, aligned_relative, sha256(aligned_target))
        color_rows.append(f"{color_stamp:.9f} {color_relative}\n")
        aligned_rows.append(f"{depth_stamp:.9f} {aligned_relative}\n")
        deltas_ms.append(abs(depth_stamp - color_stamp) * 1000.0)
        valid_fractions.append(float(np.count_nonzero(aligned_units)) / float(aligned_units.size))
        if (index + 1) % 100 == 0:
            print(json.dumps({"prepared_frames": index + 1, "total_frames": len(associated)}), flush=True)

    (work / "color.txt").write_text("".join(color_rows), encoding="utf-8")
    (work / "aligned_depth.txt").write_text("".join(aligned_rows), encoding="utf-8")
    shutil.copy2(ground_truth, work / "groundtruth.txt")
    calibration_targets = {
        "transformations": Path("transformations.yaml"),
        "color_intrinsics": Path("calibration/color_intrinsics.yaml"),
        "depth_intrinsics": Path("calibration/depth_intrinsics.yaml"),
        "depth_to_color": Path("calibration/extrinsics_depth_to_color.yaml"),
    }
    for name, path in paths.items():
        target = work / calibration_targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "sequence_id": manifest["sequence"]["sequence_id"],
        "selected_camera": "camera_front",
        "source_root": str(source_root),
        "frame_count": len(associated),
        "original_color_count": len(color),
        "original_depth_count": len(depth),
        "associated_fraction": associated_fraction,
        "maximum_rgb_depth_delta_ms": maximum_delta_ms,
        "rgb_depth_delta_ms_p95": float(np.quantile(deltas_ms, 0.95)),
        "rgb_depth_delta_ms_max": max(deltas_ms),
        "minimum_aligned_valid_depth_fraction": min(valid_fractions),
        "median_aligned_valid_depth_fraction": float(np.median(valid_fractions)),
        "depth_encoding": "uint16_png_z_meters",
        "depth_scale_units_per_meter": depth_scale_units_per_meter,
        "depth_scale_basis": "RealSense ROS 16UC1 millimeter convention plus uint16 archive inspection",
        "raw_depth_invalid_values": [0, 65535],
        "raw_depth_saturated_pixel_count": saturated_depth_pixel_count,
        "depth_registered_to_color": True,
        "registration_hole_fill": False,
        "registration_collision_policy": "nearest_z",
        "transform_convention": "parent_T_child",
        "transform_chain": transform_chain,
        "base_link_optical_forward_axis": forward_axis.tolist(),
        "calibration_member_sha256": actual_hashes,
        "ground_truth_sha256": sha256(ground_truth),
        "raw_depth_hash_chain_sha256": raw_chain.hexdigest(),
        "aligned_depth_hash_chain_sha256": aligned_chain.hexdigest(),
        "color_materialization": sorted(link_modes),
        "candidate_alerts_used": False,
        "source_count_credit": 0,
        "evaluator_ran": False,
        "production_authority": False,
    }
    write_json(work / "preparation_receipt.json", receipt)
    work.rename(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-delta-ms", type=float, default=20.0)
    parser.add_argument("--depth-scale-units-per-meter", type=float, default=1000.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        receipt = prepare(
            args.source_root,
            args.ground_truth,
            args.manifest.resolve(),
            args.output,
            args.maximum_delta_ms,
            args.depth_scale_units_per_meter,
            args.limit,
        )
        print(json.dumps(receipt))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
