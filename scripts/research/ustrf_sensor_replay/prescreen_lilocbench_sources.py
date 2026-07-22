"""Run the fail-closed LILocBench GT-only route prescreen for R3.

The official ground truth is sampled at 20 Hz while RGB-D is 15 Hz.  This
tool preserves the frozen 24/12 *RGB frame* semantics by constructing a
nominal 15 Hz timeline and associating the nearest base_link pose using the
unchanged R3 pose delta.  A pass has rejection-only authority: it cannot grant
source credit or substitute for camera-frame adaptation and full-sequence
isolated review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from contract import read_json, sha256, write_json


SCHEMA = "blindassist_ustrf_sensor_replay_r3_lilocbench_gt_prescreen_report_v1"


def load_ground_truth(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] != 8:
        raise ValueError(f"invalid LILocBench ground-truth table: {path}")
    if not np.all(np.isfinite(rows)):
        raise ValueError("non-finite LILocBench ground truth")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("non-monotonic LILocBench ground-truth time")
    quaternion_norms = np.linalg.norm(rows[:, 4:8], axis=1)
    if np.any(np.abs(quaternion_norms - 1.0) > 1e-3):
        raise ValueError("invalid LILocBench ground-truth quaternion")
    return rows


def quaternion_rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def nominal_rgb_pose_timeline(
    rows: np.ndarray,
    rgb_rate_hz: float,
    maximum_pose_delta_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rgb_rate_hz <= 0 or maximum_pose_delta_s < 0:
        raise ValueError("invalid RGB timeline contract")
    start = float(rows[0, 0])
    duration = float(rows[-1, 0] - start)
    count = int(math.floor(duration * rgb_rate_hz)) + 1
    timestamps = start + np.arange(count, dtype=np.float64) / rgb_rate_hz
    gt_timestamps = rows[:, 0]
    right = np.searchsorted(gt_timestamps, timestamps, side="left")
    right = np.clip(right, 0, len(rows) - 1)
    left = np.clip(right - 1, 0, len(rows) - 1)
    choose_left = np.abs(gt_timestamps[left] - timestamps) <= np.abs(gt_timestamps[right] - timestamps)
    indices = np.where(choose_left, left, right)
    deltas = np.abs(gt_timestamps[indices] - timestamps)
    aligned = deltas <= maximum_pose_delta_s
    return timestamps, indices, aligned, deltas


def _classify_delta(current: np.ndarray, other: np.ndarray, minimum_displacement_m: float) -> tuple[str, float]:
    delta_world = other[1:4] - current[1:4]
    displacement = float(np.linalg.norm(delta_world))
    if displacement < minimum_displacement_m:
        return "stationary", displacement
    delta_base = quaternion_rotation_xyzw(current[4:8]).T @ delta_world
    if float(delta_base[0]) > abs(float(delta_base[1])):
        return "forward_dominant", displacement
    if float(delta_base[0]) < -abs(float(delta_base[1])):
        return "reverse_dominant", displacement
    return "lateral_dominant", displacement


def route_proxy_stats(
    rows: np.ndarray,
    pose_indices: np.ndarray,
    aligned: np.ndarray,
    offset_frames: int,
    minimum_displacement_m: float,
    mode: str,
) -> dict[str, Any]:
    if offset_frames <= 0 or mode not in {"truth_future", "causal_history"}:
        raise ValueError("invalid route proxy window")
    total = len(pose_indices)
    counts = {
        "forward_dominant": 0,
        "reverse_dominant": 0,
        "lateral_dominant": 0,
        "stationary": 0,
        "unaligned_or_edge": 0,
    }
    displacements: list[float] = []
    for index in range(total):
        other_index = index + offset_frames if mode == "truth_future" else index - offset_frames
        if other_index < 0 or other_index >= total or not aligned[index] or not aligned[other_index]:
            counts["unaligned_or_edge"] += 1
            continue
        if mode == "truth_future":
            current = rows[pose_indices[index]]
            other = rows[pose_indices[other_index]]
        else:
            current = rows[pose_indices[index]]
            past = rows[pose_indices[other_index]]
            other = current.copy()
            other[1:4] = current[1:4] + (current[1:4] - past[1:4])
        category, displacement = _classify_delta(current, other, minimum_displacement_m)
        counts[category] += 1
        displacements.append(displacement)
    known = counts["forward_dominant"]
    unknown_rate = 1.0 - known / max(1, total)
    return {
        "mode": mode,
        "offset_frames": offset_frames,
        "total_nominal_rgb_frames": total,
        "route_proxy_known_frames": known,
        "route_proxy_unknown_rate": unknown_rate,
        "classification_counts": counts,
        "median_window_displacement_m": float(np.median(displacements)) if displacements else None,
        "p05_window_displacement_m": float(np.quantile(displacements, 0.05)) if displacements else None,
        "authority": "reject_only_base_link_forward_proxy",
    }


def build_report(repo: Path, config_path: Path, ground_truth_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    prereg_receipt = config["frozen_r3_prereg"]
    prereg_path = (repo / prereg_receipt["path"]).resolve()
    prereg_sha256 = sha256(prereg_path)
    if prereg_sha256 != prereg_receipt["sha256"]:
        raise ValueError("frozen R3 prereg hash mismatch; source replacement must not tune the round")
    prereg = read_json(prereg_path)
    route = prereg["route"]
    review = config["complete_sequence_two_model_review"]
    if int(review["anchor_tolerance_frames"]) != 15:
        raise ValueError("R3 complete-sequence review tolerance changed")
    if int(review["minimum_admitted_trajectories_before_source_count_credit"]) != 3:
        raise ValueError("R3 minimum admitted source count changed")

    expected_gt = config["sequence"]["ground_truth"]
    actual_gt_size = ground_truth_path.stat().st_size
    actual_gt_sha256 = sha256(ground_truth_path)
    if actual_gt_size != int(expected_gt["size_bytes"]):
        raise ValueError("LILocBench ground-truth size mismatch")
    if actual_gt_sha256 != expected_gt["sha256"]:
        raise ValueError("LILocBench ground-truth hash mismatch")
    rows = load_ground_truth(ground_truth_path)

    precheck = config["gt_only_route_prescreen"]
    rgb_rate_hz = float(precheck["rgb_frame_rate_hz"])
    maximum_pose_delta_s = float(prereg["synchronization"]["maximum_rgb_pose_delta_ms"]) / 1000.0
    timestamps, pose_indices, aligned, alignment_deltas = nominal_rgb_pose_timeline(rows, rgb_rate_hz, maximum_pose_delta_s)
    aligned_fraction = float(np.mean(aligned))
    truth = route_proxy_stats(
        rows,
        pose_indices,
        aligned,
        int(route["truth_horizon_frames"]),
        float(route["minimum_forward_displacement_m"]),
        "truth_future",
    )
    causal = route_proxy_stats(
        rows,
        pose_indices,
        aligned,
        int(route["causal_history_frames"]),
        float(route["minimum_forward_displacement_m"]),
        "causal_history",
    )
    maximum_unknown_rate = float(route["maximum_unknown_rate"])
    gt_steps = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1)
    gt_deltas = np.diff(rows[:, 0])
    gates = {
        "ground_truth_hash_and_size_match": True,
        "frozen_r3_prereg_hash_match": True,
        "nominal_rgb_pose_alignment_passed": aligned_fraction >= float(prereg["synchronization"]["minimum_source_aligned_fraction"]),
        "truth_route_proxy_unknown_rate_passed": truth["route_proxy_unknown_rate"] <= maximum_unknown_rate,
        "causal_route_proxy_unknown_rate_passed": causal["route_proxy_unknown_rate"] <= maximum_unknown_rate,
    }
    gt_route_prescreen_passed = all(gates.values())
    rights = config["dataset"]["data_rights"]
    rights_gate_passed = rights["source_admission_rights_gate_passed"] is True
    ordinary_public_download = rights.get("official_direct_download") is True
    use_policy_gate_passed = (
        ordinary_public_download
        or rights.get("source_admission_use_policy_gate_passed", rights_gate_passed) is True
    )
    full_rgbd_download_authorized = ordinary_public_download or (
        use_policy_gate_passed and rights["full_rgbd_download_authorized"] is True
    )
    archive = config["sequence"]["individual_files_rgbd_archive"]
    return {
        "schema": SCHEMA,
        "authority": "discovery_candidate_only_reject_only",
        "config_sha256": sha256(config_path),
        "frozen_r3_prereg_sha256": prereg_sha256,
        "sequence_id": config["sequence"]["sequence_id"],
        "ground_truth": {
            "path": str(ground_truth_path),
            "size_bytes": actual_gt_size,
            "sha256": actual_gt_sha256,
            "sample_count": int(rows.shape[0]),
            "duration_s": float(rows[-1, 0] - rows[0, 0]),
            "median_rate_hz": float(1.0 / np.median(gt_deltas)),
            "maximum_gap_s": float(np.max(gt_deltas)),
            "path_length_m": float(gt_steps.sum()),
            "frame": config["dataset"]["ground_truth_frame"],
        },
        "source_identity": {
            "official_page_url": config["dataset"]["official_page_url"],
            "artifact_url": expected_gt["url"],
            "http_etag": expected_gt["http_etag"],
            "http_last_modified": expected_gt["http_last_modified"],
        },
        "time_semantics": {
            "ground_truth_median_rate_hz": float(1.0 / np.median(gt_deltas)),
            "nominal_rgb_rate_hz": rgb_rate_hz,
            "nominal_rgb_frame_count": int(len(timestamps)),
            "truth_horizon_frames": int(route["truth_horizon_frames"]),
            "truth_horizon_seconds": float(route["truth_horizon_frames"] / rgb_rate_hz),
            "causal_history_frames": int(route["causal_history_frames"]),
            "causal_history_seconds": float(route["causal_history_frames"] / rgb_rate_hz),
            "maximum_rgb_pose_delta_ms": float(prereg["synchronization"]["maximum_rgb_pose_delta_ms"]),
            "nominal_rgb_pose_aligned_fraction": aligned_fraction,
            "nominal_rgb_pose_delta_p95_ms": float(np.quantile(alignment_deltas, 0.95) * 1000.0),
            "nominal_rgb_pose_delta_max_ms": float(np.max(alignment_deltas) * 1000.0),
            "actual_rgb_timestamps_available": False,
        },
        "frozen_route_thresholds": {
            "minimum_forward_displacement_m": float(route["minimum_forward_displacement_m"]),
            "maximum_unknown_rate": maximum_unknown_rate,
        },
        "truth_route_proxy": truth,
        "causal_route_proxy": causal,
        "gates": gates,
        "gt_route_prescreen_passed": gt_route_prescreen_passed,
        "data_rights_status": rights["status"],
        "ordinary_public_download": ordinary_public_download,
        "source_admission_rights_gate_passed": rights_gate_passed,
        "source_admission_use_policy_gate_passed": use_policy_gate_passed,
        "full_rgbd_download_authorized": full_rgbd_download_authorized,
        "full_rgbd_archive": {
            "downloaded": archive.get("downloaded") is True,
            "path": archive.get("downloaded_path"),
            "size_bytes": archive.get("http_content_length_bytes"),
            "sha256": archive.get("sha256"),
            "zip_crc_verified": archive.get("zip_crc_verified") is True,
        },
        "camera_optical_geometry_evaluable": False,
        "visible_obstacle_lifecycle_evaluable": False,
        "complete_sequence_two_model_admitted": False,
        "source_count_credit": 0,
        "admitted_trajectory_count": 0,
        "three_source_count_credit": False,
        "evaluator_ran": False,
        "next_gate": (
            "prepare registered camera_front RGB-D and run the frozen candidate plus two isolated complete-sequence reviews"
            if archive.get("downloaded") is True
            else (
                "download and hash full RGB-D, then verify camera calibration and synchronization"
                if full_rgbd_download_authorized
                else "locate an ordinary public download or obtain access without bypassing authentication, payment, or access controls"
            )
        ),
        "production_authority": False,
        "hardware_selection_authorized": False,
        "u0_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError("refusing to overwrite LILocBench prescreen output")
        report = build_report(args.repo.resolve(), args.config.resolve(), args.ground_truth.resolve())
        write_json(output, report)
        print(json.dumps({
            "sequence_id": report["sequence_id"],
            "gt_route_prescreen_passed": report["gt_route_prescreen_passed"],
            "full_rgbd_download_authorized": report["full_rgbd_download_authorized"],
            "source_count_credit": report["source_count_credit"],
            "evaluator_ran": report["evaluator_ran"],
        }))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
