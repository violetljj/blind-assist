"""C23 rigid-temporal object flow over the truth-blind R7 cell ledger.

R7 assigns one translation to a frame-local voxel component and immediately
lets every cell contribute route risk.  C23 changes that representation:

1. associate whole components across consecutive frames;
2. fit one rigid SE(2) transform by ICP before deriving point velocities;
3. measure attribution margin, rigidity, support, and R7/rigid agreement;
4. accumulate up to five causal velocity observations and score consistency;
5. expose the resulting object-motion confidence to the unchanged R7 route.

This is a consumed Development canary.  It does not alter route geometry,
probability thresholds, lifecycle, target truth, or fresh cohorts.
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

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_c22_ego_rigid_visual_motion as c22
import dtr_r7_occupancy_flow_canary as r7
from dtr_r5_dropout_canary import cases_from_tracks
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    LAST_FRAME,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    load_truth_and_associate,
    read_jsonl,
    write_json,
)
from tools.research_backend import (
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


SCHEMA = "blindassist-dtr-c23-rigid-temporal-object-flow-v1"
LEDGER_SCHEMA = "blindassist-dtr-c23-rigid-temporal-confidence-ledger-v1"
STATUS_MET = "DTR_C23_RIGID_TEMPORAL_OBJECT_FLOW_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C23_RIGID_TEMPORAL_OBJECT_FLOW_DEVELOPMENT_GATE_NOT_MET"
ICP_ITERATIONS = 5
TEMPORAL_OBSERVATIONS = 5
RIGIDITY_SIGMA_M = 2.0 * r7.FROZEN_FLOW_CONFIG.voxel_size_m
VELOCITY_SIGMA_MPS = 0.55
DECISION_CONFIDENCE = 0.5
TARGET_FALSE_SEGMENTS = 14

_ICP_BACKEND: str | None = None
_ICP_SELECTION: dict[str, Any] | None = None


@dataclass(frozen=True)
class ComponentObservation:
    frame: int
    component_id: int
    points_world: np.ndarray
    centroid_world: np.ndarray
    r7_velocity_world: np.ndarray


@dataclass(frozen=True)
class RigidFit:
    rotation_rad: float
    centroid_velocity_world: np.ndarray
    rmse_m: float
    inlier_fraction: float


@dataclass(frozen=True)
class Association:
    previous: ComponentObservation
    fit: RigidFit
    rigidity: float
    size_similarity: float
    base_score: float


def _numpy_icp(
    previous: np.ndarray,
    current: np.ndarray,
    span_s: float,
) -> RigidFit:
    source = np.asarray(previous, dtype=np.float64)
    target = np.asarray(current, dtype=np.float64)
    rotation = np.eye(2, dtype=np.float64)
    translation = target.mean(axis=0) - source.mean(axis=0)
    for _ in range(ICP_ITERATIONS):
        transformed = source @ rotation.T + translation
        distance2 = np.sum((transformed[:, None, :] - target[None, :, :]) ** 2, axis=2)
        paired = target[np.argmin(distance2, axis=1)]
        source_mean = source.mean(axis=0)
        paired_mean = paired.mean(axis=0)
        left = source - source_mean
        right = paired - paired_mean
        u, _singular, vt = np.linalg.svd(left.T @ right)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        translation = paired_mean - source_mean @ rotation.T
    transformed = source @ rotation.T + translation
    left_distance2 = np.min(
        np.sum((transformed[:, None, :] - target[None, :, :]) ** 2, axis=2), axis=1
    )
    right_distance2 = np.min(
        np.sum((target[:, None, :] - transformed[None, :, :]) ** 2, axis=2), axis=1
    )
    residual2 = np.concatenate((left_distance2, right_distance2))
    inlier_radius = math.sqrt(2.0) * r7.FROZEN_FLOW_CONFIG.voxel_size_m
    centroid_velocity = (target.mean(axis=0) - source.mean(axis=0)) / span_s
    return RigidFit(
        rotation_rad=float(math.atan2(rotation[1, 0], rotation[0, 0])),
        centroid_velocity_world=centroid_velocity,
        rmse_m=float(math.sqrt(float(residual2.mean()))),
        inlier_fraction=float(np.mean(residual2 <= inlier_radius**2)),
    )


def _torch_icp(
    previous: np.ndarray,
    current: np.ndarray,
    span_s: float,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    source = torch.as_tensor(previous, dtype=torch.float64, device="cuda")
    target = torch.as_tensor(current, dtype=torch.float64, device="cuda")
    rotation = torch.eye(2, dtype=torch.float64, device="cuda")
    translation = target.mean(dim=0) - source.mean(dim=0)
    for _ in range(ICP_ITERATIONS):
        transformed = source @ rotation.T + translation
        paired = target[torch.argmin(torch.cdist(transformed, target), dim=1)]
        source_mean = source.mean(dim=0)
        paired_mean = paired.mean(dim=0)
        u, _singular, vh = torch.linalg.svd((source - source_mean).T @ (paired - paired_mean))
        rotation = vh.T @ u.T
        if float(torch.linalg.det(rotation).item()) < 0.0:
            vh = vh.clone()
            vh[-1, :] *= -1.0
            rotation = vh.T @ u.T
        translation = paired_mean - source_mean @ rotation.T
    transformed = source @ rotation.T + translation
    distance = torch.cdist(transformed, target)
    residual2 = torch.cat((torch.min(distance, dim=1).values**2, torch.min(distance, dim=0).values**2))
    inlier_radius = math.sqrt(2.0) * r7.FROZEN_FLOW_CONFIG.voxel_size_m
    output = {
        "rotation": torch.atan2(rotation[1, 0], rotation[0, 0]),
        "velocity": (target.mean(dim=0) - source.mean(dim=0)) / span_s,
        "rmse": torch.sqrt(torch.mean(residual2)),
        "inlier": torch.mean((residual2 <= inlier_radius**2).to(torch.float64)),
    }
    if not numpy_output:
        return output
    return RigidFit(
        rotation_rad=float(output["rotation"].cpu().item()),
        centroid_velocity_world=output["velocity"].cpu().numpy(),
        rmse_m=float(output["rmse"].cpu().item()),
        inlier_fraction=float(output["inlier"].cpu().item()),
    )


def _select_icp_backend(
    previous: np.ndarray,
    current: np.ndarray,
    span_s: float,
    receipt: Path,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_icp(previous, current, span_s)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_icp(previous, current, span_s, numpy_output=False)
        return cache["gpu"]

    def observe(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        gpu = _torch_icp(previous, current, span_s, numpy_output=True)
        cpu = cache["cpu"]
        require(abs(cpu.rmse_m - gpu.rmse_m) <= 1e-8, "c23_icp_rmse_mismatch")
        require(abs(cpu.inlier_fraction - gpu.inlier_fraction) <= 1e-8, "c23_icp_inlier_mismatch")
        require(
            np.allclose(cpu.centroid_velocity_world, gpu.centroid_velocity_world, atol=1e-10, rtol=1e-10),
            "c23_icp_velocity_mismatch",
        )
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "numpy-rigid-component-icp",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-rigid-component-icp",
            "cuda",
            gpu_probe,
            observe,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _icp(
    previous: np.ndarray,
    current: np.ndarray,
    span_s: float,
    receipt: Path,
) -> RigidFit:
    global _ICP_BACKEND, _ICP_SELECTION
    if _ICP_BACKEND is None:
        _ICP_SELECTION = _select_icp_backend(previous, current, span_s, receipt)
        _ICP_BACKEND = str(_ICP_SELECTION["selected_backend"])
    if _ICP_BACKEND == "torch-cuda-rigid-component-icp":
        return _torch_icp(previous, current, span_s, numpy_output=True)
    return _numpy_icp(previous, current, span_s)


def _world_velocity(local: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1]],
        dtype=np.float64,
    )


def _components(
    base: r7.FlowLedger,
    frame: int,
    pose: dict[str, Any],
) -> tuple[list[ComponentObservation], np.ndarray]:
    forward, left, vf, vl, component = base.frame_cells(frame)
    observations = []
    for component_id in np.unique(component):
        mask = component == component_id
        local = np.column_stack((forward[mask], left[mask], np.zeros(int(mask.sum()))))
        world = r7._ego_to_world(local, pose)[:, :2]
        velocity_local = np.asarray([np.median(vf[mask]), np.median(vl[mask])])
        observations.append(
            ComponentObservation(
                frame=frame,
                component_id=int(component_id),
                points_world=world,
                centroid_world=world.mean(axis=0),
                r7_velocity_world=_world_velocity(velocity_local, float(pose["yaw_rad"])),
            )
        )
    return observations, component


def _associations(
    current: ComponentObservation,
    previous: Sequence[ComponentObservation],
    span_s: float,
    receipt: Path,
) -> list[Association]:
    output = []
    for candidate in previous:
        ratio = len(current.points_world) / len(candidate.points_world)
        if not r7.FROZEN_FLOW_CONFIG.minimum_size_ratio <= ratio <= r7.FROZEN_FLOW_CONFIG.maximum_size_ratio:
            continue
        centroid_speed = float(
            np.linalg.norm(current.centroid_world - candidate.centroid_world) / span_s
        )
        if centroid_speed > r7.FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps:
            continue
        fit = _icp(candidate.points_world, current.points_world, span_s, receipt)
        rigidity = math.exp(-0.5 * (fit.rmse_m / RIGIDITY_SIGMA_M) ** 2)
        size_similarity = min(ratio, 1.0 / ratio)
        base_score = rigidity * fit.inlier_fraction * size_similarity
        output.append(Association(candidate, fit, rigidity, size_similarity, base_score))
    return sorted(output, key=lambda item: item.base_score, reverse=True)


def _temporal_confidence(history: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.asarray(history[-TEMPORAL_OBSERVATIONS:], dtype=np.float64)
    if not len(values):
        return 0.0, math.inf
    center = np.median(values, axis=0)
    variance = float(np.mean(np.sum((values - center) ** 2, axis=1)))
    consistency = math.exp(-0.5 * variance / VELOCITY_SIGMA_MPS**2)
    coverage = min(1.0, len(values) / TEMPORAL_OBSERVATIONS)
    return consistency * math.sqrt(coverage), variance


def _confidence(
    current: ComponentObservation,
    associations: Sequence[Association],
    previous_history: dict[tuple[int, int], tuple[np.ndarray, ...]],
) -> tuple[float, tuple[np.ndarray, ...], dict[str, Any]]:
    if not associations:
        return 0.0, (), {"reason": "no_component_association"}
    best = associations[0]
    second = associations[1].base_score if len(associations) > 1 else 0.0
    attribution = max(0.0, 1.0 - second / max(best.base_score, np.finfo(float).tiny))
    agreement_error = float(
        np.linalg.norm(best.fit.centroid_velocity_world - current.r7_velocity_world)
    )
    agreement = math.exp(-0.5 * (agreement_error / VELOCITY_SIGMA_MPS) ** 2)
    speed = float(np.linalg.norm(best.fit.centroid_velocity_world))
    dynamic = 1.0 - math.exp(
        -0.5 * (speed / r7.FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps) ** 2
    )
    prior = previous_history.get(
        (best.previous.frame, best.previous.component_id), ()
    )
    history = (*prior[-(TEMPORAL_OBSERVATIONS - 1) :], best.fit.centroid_velocity_world)
    temporal, variance = _temporal_confidence(history)
    factors = np.asarray(
        [
            best.rigidity,
            best.fit.inlier_fraction,
            best.size_similarity,
            attribution,
            agreement,
            temporal,
            dynamic,
        ],
        dtype=np.float64,
    )
    value = float(np.prod(np.clip(factors, 0.0, 1.0)) ** (1.0 / len(factors)))
    return value, tuple(history), {
        "previous_component": best.previous.component_id,
        "rigid_velocity_world_mps": best.fit.centroid_velocity_world.tolist(),
        "r7_velocity_world_mps": current.r7_velocity_world.tolist(),
        "rigidity_rmse_m": best.fit.rmse_m,
        "rigidity": best.rigidity,
        "inlier_fraction": best.fit.inlier_fraction,
        "size_similarity": best.size_similarity,
        "attribution_margin": attribution,
        "velocity_agreement_error_mps": agreement_error,
        "temporal_observations": len(history),
        "temporal_variance_mps2": variance,
        "dynamic_speed_mps": speed,
        "confidence": value,
    }


def confidence_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".rigid-temporal.npz"),
        output.with_name(output.stem + ".rigid-temporal.json"),
    )


def materialize(
    *,
    base: r7.FlowLedger,
    timestamps: dict[int, float],
    poses: Sequence[dict[str, Any]],
    output: Path,
    manifest_path: Path,
    backend_receipt: Path,
) -> dict[str, Any]:
    confidence = np.zeros(len(base.forward_m), dtype=np.float32)
    history: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
    previous_components: list[ComponentObservation] = []
    previous_time: float | None = None
    diagnostics: dict[str, Any] = {}
    for frame_index, frame in enumerate(range(FIRST_FRAME, LAST_FRAME + 1)):
        time_s = timestamps[frame]
        pose = r7._causal_pose(poses, round(time_s * 1e9))
        current_components, component_ids = _components(base, frame, pose)
        start = int(base.offsets[frame_index])
        stop = int(base.offsets[frame_index + 1])
        frame_confidence = np.zeros(stop - start, dtype=np.float32)
        frame_rows = []
        if previous_time is not None:
            span_s = time_s - previous_time
            require(span_s > 0.0, "c23_frame_time_not_increasing")
            for current in current_components:
                candidates = _associations(
                    current, previous_components, span_s, backend_receipt
                )
                value, current_history, row = _confidence(current, candidates, history)
                history[(frame, current.component_id)] = current_history
                frame_confidence[component_ids == current.component_id] = value
                frame_rows.append({"component_id": current.component_id, **row})
        confidence[start:stop] = frame_confidence
        diagnostics[f"{frame:06d}"] = {
            "components": len(current_components),
            "supported_components": int(
                len(np.unique(component_ids[frame_confidence >= DECISION_CONFIDENCE]))
            ),
            "rows": frame_rows,
        }
        previous_components = current_components
        previous_time = time_s
    require(backend_receipt.exists(), "c23_backend_receipt_missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, confidence=confidence)
    os.replace(temporary, output)
    manifest = {
        "schema_version": LEDGER_SCHEMA,
        "truth_blind": True,
        "base_flow_ledger_sha256": base.manifest["ledger_sha256"],
        "representation": {
            "icp_iterations": ICP_ITERATIONS,
            "temporal_observations": TEMPORAL_OBSERVATIONS,
            "rigidity_sigma_m": RIGIDITY_SIGMA_M,
            "velocity_sigma_mps": VELOCITY_SIGMA_MPS,
            "decision_confidence": DECISION_CONFIDENCE,
            "features": [
                "rigidity_rmse",
                "inlier_fraction",
                "size_similarity",
                "attribution_margin",
                "r7_rigid_velocity_agreement",
                "five_observation_velocity_consistency",
                "nonzero_motion_evidence",
            ],
        },
        "diagnostics": {
            "cells": int(len(confidence)),
            "supported_cells": int(np.count_nonzero(confidence >= DECISION_CONFIDENCE)),
            "confidence_mean": float(confidence.mean()),
            "confidence_maximum": float(confidence.max()),
            "by_frame": diagnostics,
        },
        "backend_receipt": str(backend_receipt.resolve()),
        "backend_receipt_sha256": sha256_file(backend_receipt),
        "ledger": str(output.resolve()),
        "ledger_sha256": sha256_file(output),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_ledger(
    base: r7.FlowLedger,
    path: Path,
    manifest_path: Path,
) -> c22.VisualLedger:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == LEDGER_SCHEMA, "c23_manifest_schema")
    require(manifest.get("truth_blind") is True, "c23_manifest_truth")
    require(
        manifest["base_flow_ledger_sha256"] == base.manifest["ledger_sha256"],
        "c23_base_flow_hash",
    )
    require(sha256_file(path) == manifest["ledger_sha256"], "c23_ledger_hash")
    values = np.load(path, allow_pickle=False)["confidence"]
    require(len(values) == len(base.forward_m), "c23_confidence_shape")
    return c22.VisualLedger(base, values, DECISION_CONFIDENCE, manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    r7_result_path = args.r7_result.resolve(strict=True)
    r7_result = json.loads(r7_result_path.read_text(encoding="utf-8"))
    flow_path, flow_manifest = r7.ledger_paths(r7_result_path)
    base = r7.load_flow_ledger(flow_path, flow_manifest)
    timestamps_path = Path(r7_result["source"]["timestamps"]).resolve(strict=True)
    bag_path = Path(r7_result["source"]["bag"]).resolve(strict=True)
    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    ledger_path, ledger_manifest = confidence_paths(args.output.resolve())
    if not (args.reuse_confidence and ledger_path.exists() and ledger_manifest.exists()):
        materialize(
            base=base,
            timestamps=timestamps,
            poses=poses,
            output=ledger_path,
            manifest_path=ledger_manifest,
            backend_receipt=args.backend_receipt.resolve(),
        )
    ledger = load_ledger(base, ledger_path, ledger_manifest)

    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    known_tracks = Path(r7_result["source"]["known_height_tracks"]).resolve(strict=True)
    labels = Path(r7_result["source"]["labels"]).resolve(strict=True)
    tracks, geometry_quality = load_truth_and_associate(
        labels, read_jsonl(known_tracks), context
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
        "question": "Can object-rigid attribution plus five-observation velocity consistency preserve R7 9/9 recovery while removing at least six of eight added false segments?",
        "source": {
            "r7_result": str(r7_result_path),
            "r7_result_sha256": sha256_file(r7_result_path),
            "bag": str(bag_path),
            "bag_authority": bag_authority,
        },
        "rigid_temporal_ledger": ledger.manifest,
        "original_cohort": {"r7": baseline, "c23": original},
        "stress_by_duration_s": stress,
        "global_nuisance": nuisance,
        "gate": {"passed": passed, "checks": checks, "recovered_window_misses": recovered},
        "evaluator_firewall": {
            "ledger": "sealed from the truth-blind R7 raw-LiDAR voxel ledger before labels",
            "labels": "opened only for target attribution and scoring after confidence seal",
            "geometry_quality": geometry_quality,
        },
        "limitations": [
            "One consumed 143-frame Development canary with three events and nine repeated induced-dropout trials.",
            "R7 voxel components are a coarse object proxy, not instance segmentation.",
            "No fresh, product, user-benefit, reliability, or safety claim is authorized.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def main() -> int:
    evidence = REPO / "artifacts.local" / "evidence"
    root = evidence / "dtr-c23" / "rigid-temporal-object-flow"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-result", type=Path, default=evidence / "dtr-r7" / "occupancy-flow-canary" / "result.json")
    parser.add_argument("--output", type=Path, default=root / "result.json")
    parser.add_argument("--backend-receipt", type=Path, default=root / "backend-rigid-icp.json")
    parser.add_argument("--reuse-confidence", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "gate": result["gate"], "c23": result["original_cohort"]["c23"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
