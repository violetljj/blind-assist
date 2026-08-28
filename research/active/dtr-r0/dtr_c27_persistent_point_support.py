"""Evaluate a PDC-seeded, identity-free persistent point-support memory.

This consumed C25 Development canary replays sealed M1-PD/M1-PDC ledgers.  A
PDC cell may originate a lineage; M1-PD may only refresh an existing lineage by
reciprocal world-space point association.  Confidence blends the observed and
advected state but never deletes a live lineage.  Missing cells are propagated
for the already exercised 0.8 second maximum dropout duration, then the arm
falls back to the frozen PDC contract.  The frozen PDC lifecycle is mechanically
unioned with the extension lifecycle so an UNKNOWN extension cannot erase an
already-active baseline frame.  Evaluator identity and component IDs are never
used for association, and no component velocity is broadcast.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research_backend import (  # noqa: E402
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)
from dtr_c1_global_obb_cohort_admission import (  # noqa: E402
    ROSTER_SCHEMA,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import (  # noqa: E402
    _tracks,
    aggregate_scores,
    dropout_stress,
    score_sequence,
)
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_m1_confident_direct_velocity import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    POSITION_SIGMA_M,
    SEARCH_RADIUS_M,
    VELOCITY_SIGMA_MPS,
    _local_to_world,
    _velocity_to_world,
    load_ledger as load_confident_ledger,
)
from dtr_m1_raw_point_direct_velocity import load_ledger as load_point_ledger  # noqa: E402
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS, DROPOUT_DURATIONS_S, cases_from_tracks  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
    FlowLedger,
    _entry_s,
    _rotate_world_velocity_to_ego,
    _world_to_ego_xy,
    load_flow_ledger,
)

SCHEMA = "blindassist-dtr-c27-persistent-point-support-development-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c27-sealed-persistent-point-support-v1"
C25_PREDICTION_SCHEMA = "blindassist-dtr-c25-sealed-point-flow-predictions-v1"
C25_RESULT_SCHEMA = "blindassist-dtr-c25-fresh-point-flow-confirmation-v1"
STATUS_MET = "DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_NOT_MET"
ARM = "M1_PDM_GLOBAL"
REFERENCE_ARMS = ("R7_P_GLOBAL", "M1_PD_GLOBAL", "M1_PDC_GLOBAL")
MAX_MEMORY_AGE_S = max(DROPOUT_DURATIONS_S)
HALF_LIFE_S = MAX_MEMORY_AGE_S
EPSILON = 1e-9


@dataclass
class Particle:
    lineage_id: int
    position_world: np.ndarray
    velocity_world: np.ndarray
    state_time_s: float
    last_seed_time_s: float
    seed_confidence: float
    status: str


def _load_arrays(path: Path, manifest_path: Path, required: set[str]) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(sha256_file(path) == manifest["ledger_sha256"], f"ledger_hash:{path}")
    with np.load(path, allow_pickle=False) as values:
        require(required <= set(values.files), f"ledger_arrays:{path}")
        return {name: np.asarray(values[name]) for name in values.files}


def _cost_matrix(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_position = source_position[:, None, :] - target_position[None, :, :]
    distance = np.linalg.norm(delta_position, axis=2)
    delta_velocity = source_velocity[:, None, :] - target_velocity[None, :, :]
    velocity_error = np.linalg.norm(delta_velocity, axis=2)
    cost = (distance / POSITION_SIGMA_M) ** 2 + (velocity_error / VELOCITY_SIGMA_MPS) ** 2
    cost[distance > SEARCH_RADIUS_M + EPSILON] = np.inf
    return cost, distance


def _mutual_from_cost(cost: np.ndarray) -> np.ndarray:
    output = np.full(cost.shape[0], -1, dtype=np.int64)
    if not cost.size or cost.shape[1] == 0:
        return output
    source_best = np.argmin(cost, axis=1)
    source_value = cost[np.arange(len(source_best)), source_best]
    target_best = np.argmin(cost, axis=0)
    for source_index, target_index in enumerate(source_best):
        if math.isfinite(float(source_value[source_index])) and int(target_best[target_index]) == source_index:
            output[source_index] = int(target_index)
    return output


def _match_cpu(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> np.ndarray:
    output = np.full(len(source_position), -1, dtype=np.int64)
    if not len(source_position) or not len(target_position):
        return output
    target_tree = cKDTree(target_position)
    source_tree = cKDTree(source_position)
    source_best: dict[int, tuple[float, int]] = {}
    for source_index, candidates in enumerate(
        target_tree.query_ball_point(source_position, SEARCH_RADIUS_M + EPSILON)
    ):
        if not candidates:
            continue
        candidate = np.asarray(candidates, dtype=np.int64)
        dp = np.linalg.norm(target_position[candidate] - source_position[source_index], axis=1)
        dv = np.linalg.norm(target_velocity[candidate] - source_velocity[source_index], axis=1)
        costs = (dp / POSITION_SIGMA_M) ** 2 + (dv / VELOCITY_SIGMA_MPS) ** 2
        local = int(np.argmin(costs))
        source_best[source_index] = (float(costs[local]), int(candidate[local]))
    target_best: dict[int, int] = {}
    for target_index, candidates in enumerate(
        source_tree.query_ball_point(target_position, SEARCH_RADIUS_M + EPSILON)
    ):
        if not candidates:
            continue
        candidate = np.asarray(candidates, dtype=np.int64)
        dp = np.linalg.norm(source_position[candidate] - target_position[target_index], axis=1)
        dv = np.linalg.norm(source_velocity[candidate] - target_velocity[target_index], axis=1)
        costs = (dp / POSITION_SIGMA_M) ** 2 + (dv / VELOCITY_SIGMA_MPS) ** 2
        target_best[target_index] = int(candidate[int(np.argmin(costs))])
    for source_index, (_cost, target_index) in source_best.items():
        if target_best.get(target_index) == source_index:
            output[source_index] = target_index
    return output


def _match_gpu_tensor(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> Any:
    import torch

    if not len(source_position) or not len(target_position):
        return torch.full((len(source_position),), -1, dtype=torch.int64, device="cuda")
    source_p = torch.as_tensor(source_position, dtype=torch.float64, device="cuda")
    source_v = torch.as_tensor(source_velocity, dtype=torch.float64, device="cuda")
    target_p = torch.as_tensor(target_position, dtype=torch.float64, device="cuda")
    target_v = torch.as_tensor(target_velocity, dtype=torch.float64, device="cuda")
    distance = torch.cdist(source_p, target_p)
    velocity_error = torch.cdist(source_v, target_v)
    cost = (distance / POSITION_SIGMA_M) ** 2 + (velocity_error / VELOCITY_SIGMA_MPS) ** 2
    cost = torch.where(distance <= SEARCH_RADIUS_M + EPSILON, cost, torch.inf)
    source_value, source_best = torch.min(cost, dim=1)
    _target_value, target_best = torch.min(cost, dim=0)
    source_index = torch.arange(len(source_position), device="cuda")
    mutual = torch.isfinite(source_value) & (target_best[source_best] == source_index)
    return torch.where(mutual, source_best, torch.full_like(source_best, -1))


def _select_backend(
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    receipt_path: Path,
) -> dict[str, Any]:
    cpu_cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cpu_cache["value"] = _match_cpu(
            source_position, source_velocity, target_position, target_velocity
        )
        return cpu_cache["value"]

    def gpu_probe() -> Any:
        return _match_gpu_tensor(
            source_position, source_velocity, target_position, target_velocity
        )

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "value" in cpu_cache and not np.array_equal(
            cpu_cache["value"], output.detach().cpu().numpy()
        ):
            raise BackendSelectionError("C27_RECIPROCAL_MATCH_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "scipy-cKDTree-reciprocal-point-lineage",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"scipy-cKDTree/numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-reciprocal-point-lineage",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def _match(
    backend: str,
    source_position: np.ndarray,
    source_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> np.ndarray:
    if backend == "scipy-cKDTree-reciprocal-point-lineage":
        return _match_cpu(source_position, source_velocity, target_position, target_velocity)
    if backend == "torch-cuda-reciprocal-point-lineage":
        return _match_gpu_tensor(
            source_position, source_velocity, target_position, target_velocity
        ).detach().cpu().numpy()
    raise ValueError(f"unknown_backend:{backend}")


def _pose(arrays: Mapping[str, np.ndarray], index: int) -> dict[str, float]:
    return {
        "x_m": float(arrays["frame_ego_x_m"][index]),
        "y_m": float(arrays["frame_ego_y_m"][index]),
        "yaw_rad": float(arrays["frame_ego_yaw_rad"][index]),
    }


def _frame_values(arrays: Mapping[str, np.ndarray], index: int) -> tuple[np.ndarray, ...]:
    start = int(arrays["offsets"][index])
    stop = int(arrays["offsets"][index + 1])

    def optional(name: str) -> np.ndarray:
        if name not in arrays:
            return np.ones(stop - start, dtype=np.float64)
        return np.asarray(arrays[name][start:stop], dtype=np.float64)

    return (
        np.column_stack([arrays["forward_m"][start:stop], arrays["left_m"][start:stop]]).astype(np.float64),
        np.column_stack(
            [arrays["velocity_forward_mps"][start:stop], arrays["velocity_left_mps"][start:stop]]
        ).astype(np.float64),
        optional("source_point_count"),
        optional("flow_support"),
        optional("confidence"),
    )


def _build_memory(
    *,
    sequence: str,
    pd: Mapping[str, np.ndarray],
    pdc: Mapping[str, np.ndarray],
    backend: str,
) -> tuple[FlowLedger, dict[int, dict[str, Any]], dict[str, Any]]:
    frames = np.asarray(pd["frames"], dtype=np.int32)
    times = np.asarray(pd["frame_time_s"], dtype=np.float64)
    require(np.array_equal(frames, pdc["frames"]), f"pdc_frame_drift:{sequence}")
    particles: list[Particle] = []
    next_lineage = 0
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    offsets = [0]
    sidecar: dict[int, dict[str, Any]] = {}
    global_counts: Counter[str] = Counter()

    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        now_s = float(times[index])
        pose = _pose(pd, index)
        pd_local, pd_velocity_local, source_count, flow_support, _pd_conf = _frame_values(pd, index)
        pdc_local, pdc_velocity_local, _a, _b, pdc_confidence = _frame_values(pdc, index)
        pd_world = _local_to_world(pd_local, pose["x_m"], pose["y_m"], pose["yaw_rad"])
        pd_velocity_world = _velocity_to_world(pd_velocity_local, pose["yaw_rad"])
        pdc_world = _local_to_world(pdc_local, pose["x_m"], pose["y_m"], pose["yaw_rad"])
        pdc_velocity_world = _velocity_to_world(pdc_velocity_local, pose["yaw_rad"])

        particles = [
            particle
            for particle in particles
            if now_s - particle.last_seed_time_s <= MAX_MEMORY_AGE_S + EPSILON
        ]
        predicted_position = np.asarray(
            [
                particle.position_world
                + particle.velocity_world * (now_s - particle.state_time_s)
                for particle in particles
            ],
            dtype=np.float64,
        ).reshape((-1, 2))
        predicted_velocity = np.asarray(
            [particle.velocity_world for particle in particles], dtype=np.float64
        ).reshape((-1, 2))
        particle_to_pd = _match(
            backend, predicted_position, predicted_velocity, pd_world, pd_velocity_world
        )
        pdc_to_pd = _match(
            backend, pdc_world, pdc_velocity_world, pd_world, pd_velocity_world
        )
        pd_to_pdc = {int(pd_index): pdc_index for pdc_index, pd_index in enumerate(pdc_to_pd) if pd_index >= 0}
        used_pdc: set[int] = set()
        frame_rows: list[dict[str, Any]] = []

        for particle_index, particle in enumerate(particles):
            pd_index = int(particle_to_pd[particle_index])
            age_s = now_s - particle.last_seed_time_s
            half_life = 2.0 ** (-age_s / HALF_LIFE_S)
            if pd_index >= 0 and pd_index in pd_to_pdc:
                pdc_index = int(pd_to_pdc[pd_index])
                used_pdc.add(pdc_index)
                particle.position_world = pdc_world[pdc_index].copy()
                particle.velocity_world = pdc_velocity_world[pdc_index].copy()
                particle.last_seed_time_s = now_s
                particle.seed_confidence = float(pdc_confidence[pdc_index])
                particle.status = "OBSERVED_PDC"
                half_life = 1.0
                q = 1.0
                weight = 1.0
                dp = 0.0
                dv = 0.0
            elif pd_index >= 0:
                dp = float(np.linalg.norm(pd_world[pd_index] - predicted_position[particle_index]))
                dv = float(np.linalg.norm(pd_velocity_world[pd_index] - predicted_velocity[particle_index]))
                q = min(
                    min(1.0, float(source_count[pd_index]) / 3.0),
                    float(flow_support[pd_index]),
                    math.exp(-0.5 * (dp / POSITION_SIGMA_M) ** 2),
                    math.exp(-0.5 * (dv / VELOCITY_SIGMA_MPS) ** 2),
                )
                weight = particle.seed_confidence * half_life * q
                particle.position_world = (
                    weight * pd_world[pd_index]
                    + (1.0 - weight) * predicted_position[particle_index]
                )
                particle.velocity_world = (
                    weight * pd_velocity_world[pd_index]
                    + (1.0 - weight) * predicted_velocity[particle_index]
                )
                particle.status = "OBSERVED_PD"
            else:
                particle.position_world = predicted_position[particle_index].copy()
                particle.status = "PREDICTED_UNKNOWN"
                q = 0.0
                weight = 0.0
                dp = None
                dv = None
            particle.state_time_s = now_s
            frame_rows.append(
                {
                    "lineage_id": particle.lineage_id,
                    "status": particle.status,
                    "age_s": now_s - particle.last_seed_time_s,
                    "seed_confidence": particle.seed_confidence,
                    "q": q,
                    "h": half_life,
                    "w": weight,
                    "dp_m": dp,
                    "dv_mps": dv,
                }
            )

        for pdc_index in range(len(pdc_world)):
            if pdc_index in used_pdc:
                continue
            particles.append(
                Particle(
                    lineage_id=next_lineage,
                    position_world=pdc_world[pdc_index].copy(),
                    velocity_world=pdc_velocity_world[pdc_index].copy(),
                    state_time_s=now_s,
                    last_seed_time_s=now_s,
                    seed_confidence=float(pdc_confidence[pdc_index]),
                    status="OBSERVED_PDC",
                )
            )
            frame_rows.append(
                {
                    "lineage_id": next_lineage,
                    "status": "OBSERVED_PDC",
                    "age_s": 0.0,
                    "seed_confidence": float(pdc_confidence[pdc_index]),
                    "q": 1.0,
                    "h": 1.0,
                    "w": 1.0,
                    "dp_m": 0.0,
                    "dv_mps": 0.0,
                }
            )
            next_lineage += 1

        if particles:
            world_position = np.asarray([particle.position_world for particle in particles])
            world_velocity = np.asarray([particle.velocity_world for particle in particles])
            local_position = _world_to_ego_xy(world_position, pose)
            local_velocity = _rotate_world_velocity_to_ego(world_velocity, pose)
            component = np.asarray([particle.lineage_id for particle in particles], dtype=np.int32)
        else:
            local_position = np.empty((0, 2), dtype=np.float64)
            local_velocity = np.empty((0, 2), dtype=np.float64)
            component = np.empty(0, dtype=np.int32)
        rows.append((local_position, local_velocity, component))
        offsets.append(offsets[-1] + len(local_position))
        statuses = Counter(row["status"] for row in frame_rows)
        global_counts.update(statuses)
        sidecar[frame] = {
            "lineages": len(particles),
            "statuses": dict(sorted(statuses.items())),
            "predicted_unknown": statuses["PREDICTED_UNKNOWN"],
            "rows": frame_rows,
        }

    arrays = {
        "frames": frames,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(np.float32),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(np.float32),
        "component_id": np.concatenate([row[2] for row in rows]).astype(np.int32),
    }
    manifest = {
        "schema": "blindassist-dtr-c27-persistent-point-support-ledger-v1",
        "sequence": sequence,
        "frames": len(frames),
        "identity": "internal causal lineage only; evaluator identity and source component_id unused",
        "max_memory_age_s": MAX_MEMORY_AGE_S,
        "status_counts": dict(sorted(global_counts.items())),
    }
    return FlowLedger(manifest=manifest, **arrays), sidecar, manifest


def _predict_memory(
    *,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    ledger: FlowLedger,
    sidecar: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard_s = HORIZON_S * FROZEN_R2_CONFIG.imminent_horizon_fraction
    origin_s = float(timestamps[int(frames[0])])
    raw_frames = []
    active_frames = []
    urgent_frames = []
    unknown_frames = []
    minimum_entries: dict[str, float] = {}
    risky_cells: dict[str, int] = {}
    for frame in frames:
        forward, left, vf, vl, _component = ledger.frame_cells(int(frame))
        entries = [
            entry
            for values in zip(forward, left, vf, vl)
            if (entry := _entry_s(*(float(value) for value in values))) is not None
        ]
        entry = min(entries) if entries else None
        if entry is not None:
            evidence: bool | None = True
        elif int(sidecar[int(frame)]["predicted_unknown"]) > 0:
            evidence = None
            unknown_frames.append(int(frame))
        else:
            evidence = False
        urgent = bool(entry is not None and entry <= guard_s + EPSILON)
        signal = lifecycle.update(
            float(timestamps[int(frame)]) - origin_s, evidence, urgent=urgent
        )
        if evidence is True:
            raw_frames.append(int(frame))
            minimum_entries[str(int(frame))] = float(entry)
            risky_cells[str(int(frame))] = len(entries)
        if urgent:
            urgent_frames.append(int(frame))
        if signal in ACTIVE_SIGNALS:
            active_frames.append(int(frame))
    return {
        "raw_alert_frames": raw_frames,
        "active_alert_frames": active_frames,
        "urgent_frames": urgent_frames,
        "unknown_frames": unknown_frames,
        "minimum_entry_s_by_frame": minimum_entries,
        "risky_cells_by_frame": risky_cells,
    }


def _aggregate_stress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    misses = sum(int(row["track_only_window_misses"]) for row in rows)
    recovered = {
        "r7": sum(int(row["r7_recovered_track_only_window_misses"]) for row in rows),
        "m1_pd": sum(int(row["m1_recovered_track_only_window_misses"]) for row in rows),
        "c27": sum(int(row["m1_ct_recovered_track_only_window_misses"]) for row in rows),
    }
    return {
        "trials": sum(int(row["trials"]) for row in rows),
        "track_only_window_misses": misses,
        **{f"{name}_recovered_track_only_window_misses": value for name, value in recovered.items()},
    }


def _representative(c25: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    for row in c25["sequences"]:
        sources = row["sources"]["ledgers"]
        pd = _load_arrays(
            Path(sources["M1_PD_GLOBAL"]["ledger"]),
            Path(sources["M1_PD_GLOBAL"]["manifest"]),
            {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
        )
        pdc = _load_arrays(
            Path(sources["M1_PDC_GLOBAL"]["ledger"]),
            Path(sources["M1_PDC_GLOBAL"]["manifest"]),
            {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
        )
        for index in range(len(pd["frames"])):
            pd_position, pd_velocity, *_ = _frame_values(pd, index)
            pdc_position, pdc_velocity, *_ = _frame_values(pdc, index)
            if len(pd_position) >= 32 and len(pdc_position) >= 16:
                return (
                    pdc_position[: min(512, len(pdc_position))],
                    pdc_velocity[: min(512, len(pdc_velocity))],
                    pd_position[: min(1024, len(pd_position))],
                    pd_velocity[: min(1024, len(pd_velocity))],
                )
    raise RuntimeError("c27_representative_point_lineage_missing")


def run(args: argparse.Namespace) -> dict[str, Any]:
    c25_predictions_path = args.c25_predictions.resolve(strict=True)
    c25 = json.loads(c25_predictions_path.read_text(encoding="utf-8"))
    require(c25.get("schema") == C25_PREDICTION_SCHEMA, "c25_prediction_schema")
    require(c25.get("truth_blind") is True, "c25_prediction_not_sealed")
    representative = _representative(c25)
    selection = _select_backend(*representative, args.backend_receipt.resolve())
    backend = str(selection["selected_backend"])

    candidates: dict[str, tuple[FlowLedger, dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
    sealed_rows = []
    for row in c25["sequences"]:
        sequence = str(row["sequence"])
        sources = row["sources"]["ledgers"]
        pd_source = sources["M1_PD_GLOBAL"]
        pdc_source = sources["M1_PDC_GLOBAL"]
        pd = _load_arrays(
            Path(pd_source["ledger"]), Path(pd_source["manifest"]),
            {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps", "source_point_count", "flow_support"},
        )
        pdc = _load_arrays(
            Path(pdc_source["ledger"]), Path(pdc_source["manifest"]),
            {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps", "confidence"},
        )
        ledger, sidecar, manifest = _build_memory(
            sequence=sequence, pd=pd, pdc=pdc, backend=backend
        )
        frames = [int(value) for value in pd["frames"]]
        timestamps = {frame: float(pd["frame_time_s"][index]) for index, frame in enumerate(frames)}
        prediction = _predict_memory(
            frames=frames, timestamps=timestamps, ledger=ledger, sidecar=sidecar
        )
        pdc_prediction = row["arms"]["M1_PDC_GLOBAL"]
        union_added: dict[str, int] = {}
        for field in ("raw_alert_frames", "active_alert_frames", "urgent_frames"):
            candidate_frames = {int(value) for value in prediction[field]}
            baseline_frames = {int(value) for value in pdc_prediction[field]}
            union_added[field] = len(baseline_frames - candidate_frames)
            prediction[field] = sorted(candidate_frames | baseline_frames)
        prediction["pdc_lifecycle_union_added_frames"] = union_added
        pdc_raw = set(int(value) for value in pdc_prediction["raw_alert_frames"])
        require(pdc_raw <= set(prediction["raw_alert_frames"]), f"pdc_raw_not_subset:{sequence}")
        candidates[sequence] = (ledger, sidecar, {**manifest, "pd_arrays": pd}, prediction)
        sealed_rows.append({
            "sequence": sequence,
            "arm": prediction,
            "source": {
                "m1_pd_ledger_sha256": sha256_file(Path(pd_source["ledger"])),
                "m1_pdc_ledger_sha256": sha256_file(Path(pdc_source["ledger"])),
            },
            "diagnostics": manifest,
        })
    sealed = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "prediction_boundary": "sealed C25 point ledgers and causal internal lineages only; no roster, labels, evaluator identity, or prior C27 score",
        "backend": selection,
        "sequences": sealed_rows,
    }
    write_json(args.predictions.resolve(), sealed)

    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    c25_result_path = args.c25_result.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    c25_result = json.loads(c25_result_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema")
    require(c25_result.get("schema") == C25_RESULT_SCHEMA, "c25_result_schema")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "timestamps_hash")
    c25_rows = {str(row["sequence"]): row for row in c25["sequences"]}
    roster_rows = {str(row["sequence"]): row for row in roster["selected_sequences"]}
    require(set(candidates) == set(roster_rows), "candidate_roster_coverage")
    per_arm = {arm: [] for arm in REFERENCE_ARMS}
    candidate_scores = []
    stress_rows = []
    per_sequence = []
    event_nonregression = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sorted(candidates):
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            row = c25_rows[sequence]
            references = {}
            for arm in REFERENCE_ARMS:
                score = score_sequence(
                    sequence=sequence, timeline=timeline,
                    prediction_frames=_prediction_frames(frames, row["arms"][arm]),
                )
                references[arm] = score
                per_arm[arm].append(score)
            ledger, sidecar, manifest, prediction = candidates[sequence]
            candidate = score_sequence(
                sequence=sequence, timeline=timeline,
                prediction_frames=_prediction_frames(frames, prediction),
            )
            candidate_scores.append(candidate)
            pdc_events = {event["event_id"]: event for event in references["M1_PDC_GLOBAL"]["event_rows"]}
            for event in candidate["event_rows"]:
                baseline = pdc_events[event["event_id"]]
                event_nonregression.append(
                    event["recalled"]
                    and baseline["recalled"]
                    and float(event["first_alert_lead_s"]) + EPSILON
                    >= float(baseline["first_alert_lead_s"])
                )
            pd_arrays = manifest["pd_arrays"]
            frame_poses = {frame: _pose(pd_arrays, index) for index, frame in enumerate(frames)}
            cases = cases_from_tracks(
                _tracks(boxes_by_frame=boxes, timestamps=timestamps, frame_poses=frame_poses)
            )
            sources = row["sources"]["ledgers"]
            stress = dropout_stress(
                roster_sequence=roster_rows[sequence],
                cases={(case.label_id, case.segment_index): case for case in cases},
                r7=load_flow_ledger(
                    Path(sources["R7_P_GLOBAL"]["ledger"]),
                    Path(sources["R7_P_GLOBAL"]["manifest"]),
                    expected_sequence=sequence, expected_frames=frames,
                ),
                m1=load_point_ledger(
                    Path(sources["M1_PD_GLOBAL"]["ledger"]),
                    Path(sources["M1_PD_GLOBAL"]["manifest"]),
                    expected_sequence=sequence, expected_frames=frames,
                ),
                m1_ct=ledger,
            )
            stress_rows.append(stress)
            per_sequence.append({
                "sequence": sequence, "candidate": candidate,
                "references": references, "dropout_stress": stress,
                "memory": {key: value for key, value in manifest.items() if key != "pd_arrays"},
            })
    aggregate = {arm: aggregate_scores(rows) for arm, rows in per_arm.items()}
    candidate = aggregate_scores(candidate_scores)
    stress = _aggregate_stress(stress_rows)
    for arm in REFERENCE_ARMS:
        for metric in (
            "bounded_contact_events_recalled", "false_alert_segments",
            "bounded_contact_event_f1", "median_first_alert_lead_s",
        ):
            require(abs(float(aggregate[arm][metric]) - float(c25_result["aggregate"][arm][metric])) <= 1e-9, f"c25_replay_drift:{arm}:{metric}")
    pdc = aggregate["M1_PDC_GLOBAL"]
    r7_dropout = int(c25_result["dropout_stress"]["r7_recovered_track_only_window_misses"])
    checks = {
        "contact_recall_retains_pdc_12_of_12": candidate["bounded_contact_events_recalled"] == pdc["bounded_contact_events_recalled"],
        "false_segments_not_higher_than_pdc": candidate["false_alert_segments"] <= pdc["false_alert_segments"],
        "dropout_recovery_not_lower_than_r7": stress["c27_recovered_track_only_window_misses"] >= r7_dropout,
        "median_first_lead_not_lower_than_pdc": candidate["median_first_alert_lead_s"] + EPSILON >= pdc["median_first_alert_lead_s"],
        "every_event_lead_not_later_than_pdc": all(event_nonregression),
    }
    passed = all(checks.values())
    result = {
        "schema": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "question": "Can a PDC-seeded, PD-refreshed identity-free point lineage retain low false alerts while restoring R7 detector-dropout recovery?",
        "candidate": candidate,
        "references": aggregate,
        "dropout_stress": stress,
        "gate": {"passed": passed, "checks": checks},
        "per_sequence": per_sequence,
        "source": {
            "sealed_predictions": str(args.predictions.resolve()),
            "sealed_predictions_sha256": sha256_file(args.predictions.resolve()),
            "c25_predictions_sha256": sha256_file(c25_predictions_path),
            "c25_result_sha256": sha256_file(c25_result_path),
            "roster_sha256": sha256_file(roster_path),
            "backend_receipt": str(args.backend_receipt.resolve()),
            "backend_receipt_sha256": sha256_file(args.backend_receipt.resolve()),
        },
        "fixed_algorithm": {
            "seed": "sealed M1-PDC cells only; confidence threshold is seed authority, not an alert veto",
            "refresh": "reciprocal one-to-one world point matching to sealed M1-PD; component_id ignored",
            "search_radius_m": SEARCH_RADIUS_M,
            "position_sigma_m": POSITION_SIGMA_M,
            "velocity_sigma_mps": VELOCITY_SIGMA_MPS,
            "seed_confidence_floor": CONFIDENCE_THRESHOLD,
            "maximum_age_s": MAX_MEMORY_AGE_S,
            "half_life_s": HALF_LIFE_S,
            "confidence_role": "continuous state fusion and diagnostics only; live lineage output is not thresholded",
            "unknown_role": "within the bounded memory window, unmatched positive support sends UNKNOWN to lifecycle when it produces no route entry; at expiry the arm reverts to the frozen PDC baseline contract",
        },
        "compute": selection,
        "claim_limits": [
            "This is a consumed five-sequence Development canary, not fresh confirmation.",
            "The existing detector-dropout stress is a continuity guardrail; it does not remove raw point cells and therefore is not a point-occlusion benchmark.",
            "Sealed dynamic ledgers lack full visibility and known-empty evidence, so this is bounded positive-support memory, not a complete UNKNOWN-to-CLEAR occupancy belief.",
            "No learned future occupancy, Android runtime, product, user-benefit, reliability, or safety claim follows.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    c25 = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-c27" / "persistent-point-support"
    parser = argparse.ArgumentParser()
    parser.add_argument("--c25-predictions", type=Path, default=c25 / "predictions.json")
    parser.add_argument("--c25-result", type=Path, default=c25 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c25_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_timestamps.zip")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.json")
    parser.add_argument("--backend-receipt", type=Path, default=root / "backend.json")
    parser.add_argument("--output", type=Path, default=root / "result.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"], "gate": result["gate"],
        "candidate": result["candidate"], "dropout": result["dropout_stress"],
        "backend": result["compute"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
