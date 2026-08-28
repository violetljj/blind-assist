"""Validate R7 occupancy flow with ego-rigid RGB point-track residuals.

This consumed Development canary changes the motion information source rather
than the route decision.  Each R7 component proposes a previous world position
from its measured velocity.  Sparse RGB tracks are compared with the image
trajectory that position would follow if it were static under measured ego
motion.  Forward/backward track consistency and agreement with the proposed
motion become one component confidence before the unchanged R7 route test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_r7_occupancy_flow_canary as r7
from dtr_r5_dropout_canary import cases_from_tracks
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
    FIRST_FRAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LAST_FRAME,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    load_calibration,
    load_truth_and_associate,
    project_logical_to_stitched,
    read_jsonl,
    write_json,
)
from tools.research_backend import (
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
)


SCHEMA = "blindassist-dtr-c22-ego-rigid-visual-motion-v1"
LEDGER_SCHEMA = "blindassist-dtr-c22-visual-motion-confidence-ledger-v1"
STATUS_MET = "DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_NOT_MET"
CLAIM_CEILING = "CONSUMED_CURATED_PUBLIC_REAL_RGB_LIDAR_DEVELOPMENT_CANARY_ONLY"

# Fixed representation constants, not validation-tuned thresholds.
HEIGHT_ANCHORS_M = (-0.36, 0.12, 0.60, 1.08, 1.56)
FB_SIGMA_PX = 1.5
AGREEMENT_SIGMA_PX = 1.5
MOTION_SIGMA_PX = 0.5
DECISION_CONFIDENCE = 0.5
COMPONENT_QUANTILE = 0.75
LK_WINDOW = (21, 21)
LK_LEVELS = 3
TARGET_FALSE_SEGMENTS = 14  # remove at least six of R7's eight added segments


@dataclass(frozen=True)
class VisualLedger:
    base: r7.FlowLedger
    confidence: np.ndarray
    decision_confidence: float
    manifest: dict[str, Any]

    def frame_cells(self, frame: int) -> tuple[Any, Any, Any, Any, Any]:
        index = int(np.searchsorted(self.base.frames, frame))
        require(
            index < len(self.base.frames) and int(self.base.frames[index]) == frame,
            f"visual_frame_missing:{frame}",
        )
        start = int(self.base.offsets[index])
        stop = int(self.base.offsets[index + 1])
        keep = self.confidence[start:stop] >= self.decision_confidence
        forward, left, vf, vl, component = self.base.frame_cells(frame)
        return forward[keep], left[keep], vf[keep], vl[keep], component[keep]


def _pose_xy(pose: dict[str, Any]) -> tuple[float, float, float]:
    return float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])


def _ego_to_world_xy(local: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    x_m, y_m, yaw = _pose_xy(pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.column_stack(
        (
            x_m + cosine * local[:, 0] - sine * local[:, 1],
            y_m + sine * local[:, 0] + cosine * local[:, 1],
        )
    )


def _world_to_ego_xy(world: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    x_m, y_m, yaw = _pose_xy(pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    delta = world - np.asarray([x_m, y_m], dtype=np.float64)
    return np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        )
    )


def _ego_velocity_to_world(values: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    _x_m, _y_m, yaw = _pose_xy(pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.column_stack(
        (
            cosine * values[:, 0] - sine * values[:, 1],
            sine * values[:, 0] + cosine * values[:, 1],
        )
    )


def _project(world: np.ndarray, pose: dict[str, Any], calibration: dict[str, Any]) -> np.ndarray:
    local = _world_to_ego_xy(world[:, :2], pose)
    logical = np.column_stack(
        (
            local[:, 0] - BASE_LINK_FROM_LOGICAL_RGB360_X_M,
            local[:, 1] - BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
            world[:, 2],
        )
    )
    return project_logical_to_stitched(logical, calibration)


def _unwrap(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return values + np.round((reference - values) / IMAGE_WIDTH) * IMAGE_WIDTH


def _lk_tracks(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_tiled = np.concatenate((previous_gray, previous_gray, previous_gray), axis=1)
    current_tiled = np.concatenate((current_gray, current_gray, current_gray), axis=1)
    source = points.astype(np.float32).reshape(-1, 1, 2).copy()
    source[:, 0, 0] += IMAGE_WIDTH
    tracked, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_tiled,
        current_tiled,
        source,
        None,
        winSize=LK_WINDOW,
        maxLevel=LK_LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if tracked is None:
        count = len(points)
        return (
            np.zeros((count, 2), dtype=np.float64),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, dtype=np.float64),
            np.full(count, np.inf, dtype=np.float64),
        )
    returned, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        current_tiled,
        previous_tiled,
        tracked,
        None,
        winSize=LK_WINDOW,
        maxLevel=LK_LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if returned is None:
        returned = np.full_like(source, np.nan)
        backward_status = np.zeros_like(forward_status)
    actual = tracked[:, 0, :].astype(np.float64)
    actual[:, 0] -= IMAGE_WIDTH
    valid = (forward_status[:, 0] > 0) & (backward_status[:, 0] > 0)
    fb_error = np.linalg.norm(returned[:, 0, :] - source[:, 0, :], axis=1)
    error = np.nan_to_num(
        forward_error[:, 0], nan=np.inf, posinf=np.inf, neginf=np.inf
    ).astype(np.float64)
    return actual, valid, fb_error, error


def _select_lk_backend(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    receipt: Path,
) -> dict[str, Any]:
    probe = points[: min(512, len(points))]
    require(bool(len(probe)), "c22_lk_probe_missing")
    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "opencv-cpu-sparse-pyrlk",
            "cpu",
            lambda: _lk_tracks(previous_gray, current_gray, probe),
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"opencv-{cv2.__version__}"
            ),
        ),
        gpu=None,
        cpu_reason="GPU_BACKEND_UNAVAILABLE",
        record_path=receipt,
    )


def _frame_confidence(
    *,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_pose: dict[str, Any],
    current_pose: dict[str, Any],
    dt_s: float,
    cells: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: dict[str, Any],
    receipt: Path,
    select_receipt: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    forward, left, vf, vl, component = cells
    if not len(forward):
        return np.empty(0, dtype=np.float32), {"valid_tracks": 0, "tracks": 0}
    local = np.column_stack((forward, left)).astype(np.float64)
    current_world_xy = _ego_to_world_xy(local, current_pose)
    velocity_world = _ego_velocity_to_world(
        np.column_stack((vf, vl)).astype(np.float64), current_pose
    )
    previous_world_xy = current_world_xy - velocity_world * dt_s
    count = len(forward)
    anchors = np.asarray(HEIGHT_ANCHORS_M, dtype=np.float64)
    previous_world = np.column_stack(
        (
            np.repeat(previous_world_xy[:, 0], len(anchors)),
            np.repeat(previous_world_xy[:, 1], len(anchors)),
            np.tile(anchors, count),
        )
    )
    current_world = np.column_stack(
        (
            np.repeat(current_world_xy[:, 0], len(anchors)),
            np.repeat(current_world_xy[:, 1], len(anchors)),
            np.tile(anchors, count),
        )
    )
    source = _project(previous_world, previous_pose, calibration)
    rigid = _project(previous_world, current_pose, calibration)
    moving = _project(current_world, current_pose, calibration)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(rigid), axis=1)
    finite &= np.all(np.isfinite(moving), axis=1)
    finite &= (source[:, 1] >= 2.0) & (source[:, 1] < IMAGE_HEIGHT - 2.0)
    indices = np.nonzero(finite)[0]
    confidence = np.zeros(len(source), dtype=np.float64)
    if len(indices):
        points = source[indices]
        if select_receipt:
            _select_lk_backend(previous_gray, current_gray, points, receipt)
        actual, valid, fb_error, _lk_error = _lk_tracks(
            previous_gray, current_gray, points
        )
        rigid_u = _unwrap(rigid[indices, 0], points[:, 0])
        moving_u = _unwrap(moving[indices, 0], points[:, 0])
        actual_u = _unwrap(actual[:, 0], points[:, 0])
        actual_residual = np.column_stack(
            (actual_u - rigid_u, actual[:, 1] - rigid[indices, 1])
        )
        proposed_residual = np.column_stack(
            (moving_u - rigid_u, moving[indices, 1] - rigid[indices, 1])
        )
        agreement_error = np.linalg.norm(actual_residual - proposed_residual, axis=1)
        proposed_motion = np.linalg.norm(proposed_residual, axis=1)
        q = np.exp(-0.5 * (fb_error / FB_SIGMA_PX) ** 2)
        q *= np.exp(-0.5 * (agreement_error / AGREEMENT_SIGMA_PX) ** 2)
        q *= 1.0 - np.exp(-0.5 * (proposed_motion / MOTION_SIGMA_PX) ** 2)
        q[~valid] = 0.0
        confidence[indices] = q

    sample_component = np.repeat(component.astype(np.int64), len(anchors))
    output = np.zeros(count, dtype=np.float32)
    supported = 0
    for component_id in np.unique(component):
        cell_mask = component == component_id
        sample_mask = sample_component == int(component_id)
        values = confidence[sample_mask]
        valid_count = int(np.count_nonzero(values > 0.0))
        coverage = valid_count / max(1, len(values))
        value = (
            float(np.quantile(values[values > 0.0], COMPONENT_QUANTILE)) * coverage
            if valid_count
            else 0.0
        )
        output[cell_mask] = value
        supported += int(value >= DECISION_CONFIDENCE)
    return output, {
        "tracks": int(len(indices)),
        "valid_tracks": int(np.count_nonzero(confidence > 0.0)),
        "components": int(len(np.unique(component))),
        "supported_components": supported,
    }


def confidence_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".visual-motion.npz"),
        output.with_name(output.stem + ".visual-motion.json"),
    )


def materialize_confidence(
    *,
    base: r7.FlowLedger,
    images_dir: Path,
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
    output: Path,
    manifest_path: Path,
    receipt: Path,
) -> dict[str, Any]:
    values = np.zeros(len(base.forward_m), dtype=np.float32)
    diagnostics: dict[str, Any] = {}
    previous_gray = None
    selected = receipt.exists()
    for frame_index, frame in enumerate(range(FIRST_FRAME, LAST_FRAME + 1)):
        image = cv2.imread(str(images_dir / f"{frame:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        require(image is not None, f"c22_image_missing:{frame}")
        require(image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH), f"c22_image_shape:{frame}")
        current_pose = interpolate_pose(poses, round(timestamps[frame] * 1e9))
        if previous_gray is not None:
            previous_frame = frame - 1
            previous_pose = interpolate_pose(
                poses, round(timestamps[previous_frame] * 1e9)
            )
            dt_s = timestamps[frame] - timestamps[previous_frame]
            start = int(base.offsets[frame_index])
            stop = int(base.offsets[frame_index + 1])
            frame_values, frame_diag = _frame_confidence(
                previous_gray=previous_gray,
                current_gray=image,
                previous_pose=previous_pose,
                current_pose=current_pose,
                dt_s=dt_s,
                cells=base.frame_cells(frame),
                calibration=calibration,
                receipt=receipt,
                select_receipt=not selected,
            )
            if len(frame_values):
                selected = True
            values[start:stop] = frame_values
            diagnostics[f"{frame:06d}"] = frame_diag
        previous_gray = image
        if (frame_index + 1) % 20 == 0 or frame == LAST_FRAME:
            print(json.dumps({"c22_frames": frame_index + 1, "total": LAST_FRAME - FIRST_FRAME + 1}), flush=True)
    require(selected and receipt.exists(), "c22_backend_receipt_missing")
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, confidence=values)
    os.replace(temporary, output)
    manifest = {
        "schema_version": LEDGER_SCHEMA,
        "truth_blind": True,
        "base_flow_ledger_sha256": base.manifest["ledger_sha256"],
        "frames": {"first": FIRST_FRAME, "last": LAST_FRAME},
        "representation": {
            "height_anchors_m": list(HEIGHT_ANCHORS_M),
            "forward_backward_sigma_px": FB_SIGMA_PX,
            "motion_agreement_sigma_px": AGREEMENT_SIGMA_PX,
            "motion_strength_sigma_px": MOTION_SIGMA_PX,
            "component_quantile": COMPONENT_QUANTILE,
            "decision_confidence": DECISION_CONFIDENCE,
        },
        "diagnostics": {
            "cells": int(len(values)),
            "supported_cells": int(np.count_nonzero(values >= DECISION_CONFIDENCE)),
            "confidence_mean": float(values.mean()) if len(values) else None,
            "confidence_maximum": float(values.max()) if len(values) else None,
            "by_frame": diagnostics,
        },
        "ledger": str(output.resolve()),
        "ledger_sha256": sha256_file(output),
        "backend_receipt": str(receipt.resolve()),
        "backend_receipt_sha256": sha256_file(receipt),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_visual(base: r7.FlowLedger, path: Path, manifest_path: Path) -> VisualLedger:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == LEDGER_SCHEMA, "c22_manifest_schema")
    require(manifest.get("truth_blind") is True, "c22_manifest_not_truth_blind")
    require(manifest["base_flow_ledger_sha256"] == base.manifest["ledger_sha256"], "c22_base_hash_drift")
    require(sha256_file(path) == manifest["ledger_sha256"], "c22_ledger_hash_drift")
    confidence = np.load(path, allow_pickle=False)["confidence"]
    require(len(confidence) == len(base.forward_m), "c22_confidence_shape")
    return VisualLedger(base, confidence, DECISION_CONFIDENCE, manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    r7_result_path = args.r7_result.resolve(strict=True)
    r7_result = json.loads(r7_result_path.read_text(encoding="utf-8"))
    flow_path, flow_manifest_path = r7.ledger_paths(r7_result_path)
    base = r7.load_flow_ledger(flow_path, flow_manifest_path)
    timestamps_path = Path(r7_result["source"]["timestamps"]).resolve(strict=True)
    bag_path = Path(r7_result["source"]["bag"]).resolve(strict=True)
    images_dir = args.images_dir.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    calibration = load_calibration(calibration_dir)
    confidence_path, confidence_manifest = confidence_paths(args.output.resolve())
    if not (args.reuse_confidence and confidence_path.exists() and confidence_manifest.exists()):
        materialize_confidence(
            base=base,
            images_dir=images_dir,
            timestamps=timestamps,
            poses=poses,
            calibration=calibration,
            output=confidence_path,
            manifest_path=confidence_manifest,
            receipt=args.backend_receipt.resolve(),
        )
    ledger = load_visual(base, confidence_path, confidence_manifest)

    known_tracks_path = Path(r7_result["source"]["known_height_tracks"]).resolve(strict=True)
    labels_path = Path(r7_result["source"]["labels"]).resolve(strict=True)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, read_jsonl(known_tracks_path), context
    )
    cases = cases_from_tracks(tracks)
    original = r7.evaluate_original(cases, ledger)
    stress = r7.evaluate_stress(cases, ledger)
    nuisance = r7.global_nuisance(cases, ledger)
    recovered = sum(
        row["occupancy_flow"]["recovered_track_only_window_misses"]
        for row in stress.values()
    )
    baseline = r7_result["original_cohort"]["r7_p_occupancy_flow"]
    checks = {
        "preserves_all_nine_dropout_recoveries": recovered == 9,
        "critical_event_recall_not_lower": original["critical_event_recall"] >= baseline["critical_event_recall"],
        "false_segments_reduced_by_at_least_six": original["false_alert_segments"] <= TARGET_FALSE_SEGMENTS,
        "event_f1_higher": original["event_detection_f1"] > baseline["event_detection_f1"],
    }
    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "claim_ceiling": CLAIM_CEILING,
        "question": "Can ego-rigid visual track residual validate R7 movers while preserving 9/9 dropout recovery and removing at least six of eight added false segments?",
        "source": {
            "r7_result": str(r7_result_path),
            "r7_result_sha256": sha256_file(r7_result_path),
            "images_dir": str(images_dir),
            "bag": str(bag_path),
            "bag_authority": bag_authority,
            "calibration": calibration,
        },
        "visual_motion_ledger": ledger.manifest,
        "original_cohort": {"r7": baseline, "c22": original},
        "stress_by_duration_s": stress,
        "global_nuisance": nuisance,
        "gate": {
            "passed": passed,
            "checks": checks,
            "recovered_window_misses": recovered,
            "target_false_segments": TARGET_FALSE_SEGMENTS,
        },
        "evaluator_firewall": {
            "visual_ledger": "sealed from current/past RGB, ego pose, and truth-blind R7 flow before labels",
            "labels": "opened only for target attribution and scoring after visual ledger seal",
            "geometry_quality": geometry_quality,
        },
        "limitations": [
            "One already consumed 143-frame Development window with three events and nine repeated induced-dropout trials.",
            "Sparse cylindrical RGB tracking validates tangential image motion; it is not full 3-D scene flow.",
            "No fresh-sequence, product, user-benefit, reliability, or safety claim is authorized.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    evidence = repo / "artifacts.local" / "evidence"
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-result", type=Path, default=evidence / "dtr-r7" / "occupancy-flow-canary" / "result.json")
    parser.add_argument("--images-dir", type=Path, default=dataset / "images" / "packard-poster-session-2019-03-20_1")
    parser.add_argument("--calibration-dir", type=Path, default=repo / "artifacts.local" / "datasets" / "ustrf-canonical-observation-source-authority-data-pack-r0" / "jrdb_toolkit" / "calibration")
    root = evidence / "dtr-c22" / "ego-rigid-visual-motion"
    parser.add_argument("--output", type=Path, default=root / "result.json")
    parser.add_argument("--backend-receipt", type=Path, default=root / "backend-optical-flow.json")
    parser.add_argument("--reuse-confidence", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "gate": result["gate"], "c22": result["original_cohort"]["c22"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
