#!/usr/bin/env python3
"""Recover and validate the TUM accelerometer-to-RGB optical axis mapping."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ag_st_tum_rgbd import (
    _read_member,
    _tar_member_map,
    interpolate_camera_to_world,
    load_tum_cohort,
    parse_tum_index,
    parse_tum_poses,
)
from download_b0_arkitscenes_assets import require
from plan_ag_st_tum_third_teacher_cohort import DEFAULT_OUTPUT as DEFAULT_COHORT


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-gravity-diagnostic-r0/result.json"
WINDOW_SECONDS = 0.03


def parse_accelerometer(text: str) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        require(len(values) == 4 and all(math.isfinite(value) for value in values), "TUM accelerometer row invalid")
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    require(result.ndim == 2 and result.shape[1] == 4 and len(result) > 0, "TUM accelerometer empty")
    require(np.all(np.diff(result[:, 0]) >= 0), "TUM accelerometer timestamps nonmonotonic")
    return result


def proper_signed_permutations() -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                output.append(matrix)
    require(len(output) == 24, "signed permutation enumeration drift")
    return output


def _axis_description(matrix: np.ndarray) -> list[str]:
    axes = ("x", "y", "z")
    output: list[str] = []
    for row in range(3):
        column = int(np.argmax(np.abs(matrix[row])))
        sign = "+" if matrix[row, column] > 0 else "-"
        output.append(f"camera_{axes[row]}={sign}imu_{axes[column]}")
    return output


def _angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(vectors, axis=0)
    mean /= np.linalg.norm(mean)
    cosine = np.clip(vectors @ mean, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)), mean


def _selected_measurements(cohort: Path) -> tuple[list[dict[str, Any]], list[str]]:
    measurements: list[dict[str, Any]] = []
    missing_parents: list[str] = []
    for role in ("fit", "evaluation"):
        _, parents = load_tum_cohort(cohort, role)
        for row in parents:
            parent = str(row["parent_id"])
            archive_path = (REPO_ROOT / str(row["source_path"])).resolve()
            with tarfile.open(archive_path, "r:gz") as archive:
                members = _tar_member_map(archive, parent)
                require("accelerometer.txt" in members, f"TUM accelerometer missing: {parent}")
                rgb_rows = parse_tum_index(_read_member(archive, members["rgb.txt"]).decode("utf-8"))
                poses = parse_tum_poses(_read_member(archive, members["groundtruth.txt"]).decode("utf-8"))
                accelerometer_text = _read_member(
                    archive, members["accelerometer.txt"]
                ).decode("utf-8")
                if not any(
                    line.strip() and not line.lstrip().startswith("#")
                    for line in accelerometer_text.splitlines()
                ):
                    missing_parents.append(parent)
                    continue
                accelerometer = parse_accelerometer(accelerometer_text)
            rgb_by_index = {value.row_index: value for value in rgb_rows}
            for rgb_index in row["rgb_row_indices_zero_based"]:
                rgb = rgb_by_index[int(rgb_index)]
                camera_to_world, _ = interpolate_camera_to_world(poses, rgb.timestamp_seconds)
                delta = np.abs(accelerometer[:, 0] - rgb.timestamp_seconds)
                local = accelerometer[delta <= WINDOW_SECONDS, 1:]
                require(len(local) >= 3, f"TUM accelerometer window sparse: {parent}/{rgb_index}")
                vector = np.median(local, axis=0)
                norm = float(np.linalg.norm(vector))
                require(norm > 1e-6, "TUM accelerometer zero norm")
                measurements.append(
                    {
                        "role": role,
                        "parent_id": parent,
                        "rgb_row_index": int(rgb_index),
                        "accelerometer_mps2": vector,
                        "accelerometer_norm_mps2": norm,
                        "camera_to_world_rotation": camera_to_world[:3, :3],
                        "window_sample_count": len(local),
                    }
                )
    require(len(measurements) == 15, "TUM gravity frame count drift")
    require(len(set(missing_parents)) == 2, "TUM no-accelerometer parent count drift")
    return measurements, sorted(set(missing_parents))


def diagnose(cohort: Path) -> dict[str, Any]:
    measurements, missing_parents = _selected_measurements(cohort)
    candidates: list[dict[str, Any]] = []
    for matrix in proper_signed_permutations():
        vectors = np.stack(
            [
                row["camera_to_world_rotation"]
                @ matrix
                @ (row["accelerometer_mps2"] / row["accelerometer_norm_mps2"])
                for row in measurements
            ]
        )
        angles, mean = _angles(vectors)
        parent_p95: dict[str, float] = {}
        for parent in sorted({row["parent_id"] for row in measurements}):
            mask = np.asarray([row["parent_id"] == parent for row in measurements])
            parent_p95[parent] = float(np.quantile(angles[mask], 0.95))
        candidates.append(
            {
                "imu_to_rgb_optical": matrix.tolist(),
                "axis_description": _axis_description(matrix),
                "world_specific_force_unit": mean.tolist(),
                "median_angle_deg": float(np.median(angles)),
                "p95_angle_deg": float(np.quantile(angles, 0.95)),
                "maximum_angle_deg": float(np.max(angles)),
                "worst_parent_p95_angle_deg": max(parent_p95.values()),
                "parent_p95_angle_deg": parent_p95,
            }
        )
    candidates.sort(key=lambda row: (row["p95_angle_deg"], row["median_angle_deg"]))
    best = candidates[0]
    second = candidates[1]
    norms = np.asarray([row["accelerometer_norm_mps2"] for row in measurements])
    passed = bool(
        8.5 <= float(np.median(norms)) <= 10.8
        and best["p95_angle_deg"] <= 10.0
        and best["worst_parent_p95_angle_deg"] <= 15.0
        and second["p95_angle_deg"] - best["p95_angle_deg"] >= 2.0
    )
    return {
        "schema": "blindassist_ag_st_tum_gravity_diagnostic_v1",
        "status": "TUM_GRAVITY_BASIS_VALIDATED" if passed else "TUM_GRAVITY_BASIS_NOT_VALIDATED",
        "cohort": str(cohort.resolve()),
        "frame_count": len(measurements),
        "parent_count": len({row["parent_id"] for row in measurements}),
        "parents_without_accelerometer": missing_parents,
        "accelerometer_window_seconds": WINDOW_SECONDS,
        "accelerometer_norm_mps2": {
            "median": float(np.median(norms)),
            "p05": float(np.quantile(norms, 0.05)),
            "p95": float(np.quantile(norms, 0.95)),
        },
        "best": best,
        "runner_up": second,
        "runner_up_p95_margin_deg": second["p95_angle_deg"] - best["p95_angle_deg"],
        "all_candidates": candidates,
        "decision": {
            "gravity_basis_validated": passed,
            "support_boundary_materialization_authorized": passed,
            "world_specific_force_is_up_not_down": True,
        },
        "claim_boundary": "Accelerometer-to-RGB optical axis and gravity consistency on the frozen 21 TUM frames only; not support correctness, task utility, product, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    require(not args.output.exists(), f"gravity result already exists: {args.output}")
    result = diagnose(args.cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "best", "runner_up_p95_margin_deg")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
