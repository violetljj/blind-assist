"""C28 visibility-conditioned point memory on the consumed C25 Development canary.

The arm keeps M1-PDC as its alert baseline and extends it only with causal
point lineages.  Unlike C27, an absent lineage is not one undifferentiated
positive: its remembered 3-D support is classified from the current, separate
upper/lower LiDAR rays as HIT, KNOWN_FREE, OCCLUDED, or UNSENSED.

This file deliberately reuses C27's sealed evaluator and lifecycle machinery.
Only the truth-blind candidate builder and its raw-ray information source are
replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_c27_persistent_point_support as c27
from dtr_c28_fast_ray_visibility import (
    CODE_STATE,
    CpuRayIndex,
    STATE_CODE,
    classify_gpu_codes,
)
from dtr_c28_raw_ray_frames import RawRayFrame, iter_raw_ray_frames
from dtr_c28_ray_visibility import VOXEL_SIZE_M, Visibility
from dtr_r7_occupancy_flow_canary import (
    FROZEN_FLOW_CONFIG,
    FlowLedger,
    _rotate_world_velocity_to_ego,
    _world_to_ego_xy,
)
from jrdb_rgb_bridge import require, sha256_file, write_json
from tools.research_backend import (
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


REPO = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist-dtr-c28-visibility-conditioned-point-memory-development-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c28-sealed-visibility-conditioned-point-memory-v1"
LEDGER_SCHEMA = "blindassist-dtr-c28-visibility-conditioned-point-memory-ledger-v1"
STATUS_MET = "DTR_C28_VISIBILITY_CONDITIONED_POINT_MEMORY_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C28_VISIBILITY_CONDITIONED_POINT_MEMORY_DEVELOPMENT_GATE_NOT_MET"
STATUS_NOT_EVALUABLE = "DTR_C28_VISIBILITY_CONDITIONED_POINT_MEMORY_NOT_EVALUABLE"
ARM = "M1_PDV_GLOBAL"
EPSILON = 1e-9


@dataclass
class VisibleParticle:
    lineage_id: int
    position_world: np.ndarray
    velocity_world: np.ndarray
    state_time_s: float
    last_seed_time_s: float
    seed_confidence: float
    height_voxel_keys: tuple[int, ...]
    status: str


_SOURCE_BY_SEQUENCE: dict[str, Mapping[str, Any]] = {}
_RAY_BACKEND = ""
_TRACE_BY_SEQUENCE: dict[str, dict[str, Any]] = {}


def _xy_codes(keys: np.ndarray) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64)
    return (values[:, 0] << np.int64(32)) ^ (values[:, 1] & np.int64(0xFFFFFFFF))


def _endpoint_heights(
    raw: RawRayFrame,
    wanted_xy: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Return exact raw endpoint z-voxels for requested BEV cells."""

    if not wanted_xy:
        return {}
    wanted = np.asarray(sorted(set(wanted_xy)), dtype=np.int64)
    wanted_codes = _xy_codes(wanted)
    output: dict[tuple[int, int], set[int]] = {tuple(row): set() for row in wanted.tolist()}
    z_min, z_max = FROZEN_FLOW_CONFIG.roi_height_m
    for name in ("upper", "lower"):
        sweep = raw.sensor(name)
        if sweep is None:
            continue
        endpoints = np.asarray(sweep.world_endpoints_m, dtype=np.float64)
        keep_height = (endpoints[:, 2] >= z_min) & (endpoints[:, 2] <= z_max)
        endpoint_keys = np.floor(endpoints[keep_height] / VOXEL_SIZE_M).astype(np.int64)
        if not len(endpoint_keys):
            continue
        keep = np.isin(_xy_codes(endpoint_keys[:, :2]), wanted_codes)
        for x_key, y_key, z_key in endpoint_keys[keep]:
            key = (int(x_key), int(y_key))
            if key in output:
                output[key].add(int(z_key))
    return {key: tuple(sorted(values)) for key, values in output.items() if values}


def _cpu_ray_codes(
    origin: Sequence[float], endpoints: Sequence[Sequence[float]], queries: Sequence[Sequence[int]]
) -> np.ndarray:
    return CpuRayIndex.build(origin, endpoints).classify_codes(queries)


def _gpu_ray_codes(
    origin: Sequence[float], endpoints: Sequence[Sequence[float]], queries: Sequence[Sequence[int]]
):
    return classify_gpu_codes(origin, endpoints, queries)


def _classify_sensor(
    backend: str,
    origin: Sequence[float],
    endpoints: Sequence[Sequence[float]],
    queries: Sequence[Sequence[int]],
) -> np.ndarray:
    if backend == "scipy-cKDTree-ray-box":
        return _cpu_ray_codes(origin, endpoints, queries)
    if backend == "torch-cuda-batched-ray-box":
        return _gpu_ray_codes(origin, endpoints, queries).detach().cpu().numpy()
    raise ValueError(f"unknown_ray_backend:{backend}")


def _particle_visibilities(
    raw: RawRayFrame,
    positions_world: Sequence[np.ndarray],
    height_voxel_keys: Sequence[tuple[int, ...]],
) -> list[Visibility]:
    """Classify all absent lineages with one upload/index per sensor sweep."""

    queries: list[tuple[int, int, int]] = []
    spans: list[tuple[int, int]] = []
    for position_world, heights in zip(positions_world, height_voxel_keys, strict=True):
        start = len(queries)
        xy = np.floor(np.asarray(position_world, dtype=np.float64) / VOXEL_SIZE_M).astype(np.int64)
        queries.extend((int(xy[0]), int(xy[1]), int(z_key)) for z_key in heights)
        spans.append((start, len(queries)))
    if not queries:
        return [Visibility.UNSENSED for _ in spans]
    combined = np.zeros(len(queries), dtype=np.int8)
    for name in ("upper", "lower"):
        sweep = raw.sensor(name)
        if sweep is None:
            continue
        combined = np.maximum(
            combined,
            _classify_sensor(
                _RAY_BACKEND,
                sweep.world_origin_m,
                sweep.world_endpoints_m,
                queries,
            ),
        )
    output: list[Visibility] = []
    for start, stop in spans:
        states = tuple(CODE_STATE[int(value)] for value in combined[start:stop])
        if any(state is Visibility.HIT for state in states):
            output.append(Visibility.HIT)
        elif states and all(state is Visibility.KNOWN_FREE for state in states):
            output.append(Visibility.KNOWN_FREE)
        elif any(state is Visibility.OCCLUDED for state in states):
            output.append(Visibility.OCCLUDED)
        else:
            output.append(Visibility.UNSENSED)
    return output


def _ray_frames(
    sequence: str,
    frames: np.ndarray,
    times: np.ndarray,
) -> dict[int, RawRayFrame]:
    source = _SOURCE_BY_SEQUENCE[sequence]
    return {
        row.frame: row
        for row in iter_raw_ray_frames(
            bag_path=Path(source["bag"]).resolve(strict=True),
            frames=[int(value) for value in frames],
            frame_time_s=[float(value) for value in times],
            calibration_dir=Path(source["calibration"]).resolve(strict=True),
        )
    }


def _geometry_fields(particle: VisibleParticle, pose: Mapping[str, float]) -> dict[str, float]:
    local_position = _world_to_ego_xy(
        np.asarray([particle.position_world], dtype=np.float64), pose
    )[0]
    local_velocity = _rotate_world_velocity_to_ego(
        np.asarray([particle.velocity_world], dtype=np.float64), pose
    )[0]
    return {
        "forward_m": float(local_position[0]),
        "left_m": float(local_position[1]),
        "velocity_forward_mps": float(local_velocity[0]),
        "velocity_left_mps": float(local_velocity[1]),
    }


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
    raw_by_frame = _ray_frames(sequence, frames, times)
    particles: list[VisibleParticle] = []
    next_lineage = 0
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    offsets = [0]
    sidecar: dict[int, dict[str, Any]] = {}
    global_counts: Counter[str] = Counter()
    trace_frames: list[dict[str, Any]] = []

    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        now_s = float(times[index])
        pose = c27._pose(pd, index)
        pd_local, pd_velocity_local, source_count, flow_support, _pd_conf = c27._frame_values(pd, index)
        pdc_local, pdc_velocity_local, _a, _b, pdc_confidence = c27._frame_values(pdc, index)
        pd_world = c27._local_to_world(pd_local, pose["x_m"], pose["y_m"], pose["yaw_rad"])
        pd_velocity_world = c27._velocity_to_world(pd_velocity_local, pose["yaw_rad"])
        pdc_world = c27._local_to_world(pdc_local, pose["x_m"], pose["y_m"], pose["yaw_rad"])
        pdc_velocity_world = c27._velocity_to_world(pdc_velocity_local, pose["yaw_rad"])

        particles = [
            particle
            for particle in particles
            if now_s - particle.last_seed_time_s <= c27.MAX_MEMORY_AGE_S + EPSILON
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
        particle_to_pd = c27._match(
            backend, predicted_position, predicted_velocity, pd_world, pd_velocity_world
        )
        pdc_to_pd = c27._match(
            backend, pdc_world, pdc_velocity_world, pd_world, pd_velocity_world
        )
        pd_to_pdc = {
            int(pd_index): pdc_index
            for pdc_index, pd_index in enumerate(pdc_to_pd)
            if pd_index >= 0
        }
        wanted_world = np.concatenate([pd_world, pdc_world], axis=0) if len(pdc_world) else pd_world
        wanted_xy = [
            tuple(np.floor(point / VOXEL_SIZE_M).astype(np.int64).tolist())
            for point in wanted_world
        ]
        height_map = _endpoint_heights(raw_by_frame[frame], wanted_xy)
        used_pdc: set[int] = set()
        retained: list[VisibleParticle] = []
        emitted: list[VisibleParticle] = []
        frame_rows: list[dict[str, Any]] = []
        absent_indices = [
            particle_index
            for particle_index in range(len(particles))
            if int(particle_to_pd[particle_index]) < 0
        ]
        absent_states = _particle_visibilities(
            raw_by_frame[frame],
            [predicted_position[index] for index in absent_indices],
            [particles[index].height_voxel_keys for index in absent_indices],
        )
        visibility_by_index = dict(zip(absent_indices, absent_states, strict=True))

        for particle_index, particle in enumerate(particles):
            pd_index = int(particle_to_pd[particle_index])
            age_s = now_s - particle.last_seed_time_s
            half_life = 2.0 ** (-age_s / c27.HALF_LIFE_S)
            q = 0.0
            weight = 0.0
            dp: float | None = None
            dv: float | None = None
            visibility: Visibility | None = None
            emit = True
            keep = True
            if pd_index >= 0 and pd_index in pd_to_pdc:
                pdc_index = int(pd_to_pdc[pd_index])
                used_pdc.add(pdc_index)
                particle.position_world = pdc_world[pdc_index].copy()
                particle.velocity_world = pdc_velocity_world[pdc_index].copy()
                particle.last_seed_time_s = now_s
                particle.seed_confidence = float(pdc_confidence[pdc_index])
                particle.status = "OBSERVED_PDC"
                half_life = q = weight = 1.0
                cell = tuple(np.floor(pdc_world[pdc_index] / VOXEL_SIZE_M).astype(np.int64).tolist())
                particle.height_voxel_keys = height_map.get(cell, particle.height_voxel_keys)
                dp = dv = 0.0
            elif pd_index >= 0:
                dp = float(np.linalg.norm(pd_world[pd_index] - predicted_position[particle_index]))
                dv = float(np.linalg.norm(pd_velocity_world[pd_index] - predicted_velocity[particle_index]))
                q = min(
                    min(1.0, float(source_count[pd_index]) / 3.0),
                    float(flow_support[pd_index]),
                    math.exp(-0.5 * (dp / c27.POSITION_SIGMA_M) ** 2),
                    math.exp(-0.5 * (dv / c27.VELOCITY_SIGMA_MPS) ** 2),
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
                cell = tuple(np.floor(pd_world[pd_index] / VOXEL_SIZE_M).astype(np.int64).tolist())
                current_heights = height_map.get(cell, ())
                if current_heights:
                    particle.height_voxel_keys = current_heights
                    particle.last_seed_time_s = now_s
                    particle.status = "OBSERVED_PD_HIT"
                else:
                    particle.status = "OBSERVED_PD_NO_3D_HIT"
            else:
                particle.position_world = predicted_position[particle_index].copy()
                visibility = visibility_by_index[particle_index]
                particle.status = f"VISIBILITY_{visibility.value}"
                if visibility is Visibility.HIT:
                    # Occupancy at the predicted cell does not establish that
                    # the same mover or its old velocity is still present.
                    # Keep the lineage available for a later PD association,
                    # but do not emit or refresh it.  The frozen C28 contract
                    # allows persistence only when the support is occluded.
                    emit = False
                elif visibility is Visibility.KNOWN_FREE:
                    keep = False
                    emit = False
                elif visibility is Visibility.UNSENSED:
                    emit = False
            particle.state_time_s = now_s
            if keep:
                retained.append(particle)
            if emit:
                emitted.append(particle)
            frame_rows.append(
                {
                    "lineage_id": particle.lineage_id,
                    "status": particle.status,
                    "visibility": visibility.value if visibility is not None else None,
                    "emitted": emit,
                    "age_s": now_s - particle.last_seed_time_s,
                    "height_voxels": len(particle.height_voxel_keys),
                    "seed_confidence": particle.seed_confidence,
                    "q": q,
                    "h": half_life,
                    "w": weight,
                    "dp_m": dp,
                    "dv_mps": dv,
                    **_geometry_fields(particle, pose),
                }
            )
        particles = retained

        for pdc_index in range(len(pdc_world)):
            if pdc_index in used_pdc:
                continue
            cell = tuple(np.floor(pdc_world[pdc_index] / VOXEL_SIZE_M).astype(np.int64).tolist())
            particle = VisibleParticle(
                lineage_id=next_lineage,
                position_world=pdc_world[pdc_index].copy(),
                velocity_world=pdc_velocity_world[pdc_index].copy(),
                state_time_s=now_s,
                last_seed_time_s=now_s,
                seed_confidence=float(pdc_confidence[pdc_index]),
                height_voxel_keys=height_map.get(cell, ()),
                status="OBSERVED_PDC",
            )
            particles.append(particle)
            emitted.append(particle)
            frame_rows.append(
                {
                    "lineage_id": next_lineage,
                    "status": "OBSERVED_PDC",
                    "visibility": None,
                    "emitted": True,
                    "age_s": 0.0,
                    "height_voxels": len(particle.height_voxel_keys),
                    "seed_confidence": particle.seed_confidence,
                    "q": 1.0,
                    "h": 1.0,
                    "w": 1.0,
                    "dp_m": 0.0,
                    "dv_mps": 0.0,
                    **_geometry_fields(particle, pose),
                }
            )
            next_lineage += 1

        if emitted:
            world_position = np.asarray([particle.position_world for particle in emitted])
            world_velocity = np.asarray([particle.velocity_world for particle in emitted])
            local_position = _world_to_ego_xy(world_position, pose)
            local_velocity = _rotate_world_velocity_to_ego(world_velocity, pose)
            component = np.asarray([particle.lineage_id for particle in emitted], dtype=np.int32)
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
            "emitted_lineages": len(emitted),
            "statuses": dict(sorted(statuses.items())),
            "predicted_unknown": statuses["VISIBILITY_UNSENSED"],
            "rows": frame_rows,
        }
        trace_frames.append(
            {
                "frame": frame,
                "frame_time_s": now_s,
                "rows": frame_rows,
            }
        )

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
        "schema": LEDGER_SCHEMA,
        "sequence": sequence,
        "frames": len(frames),
        "identity": "internal causal lineage with raw-ray height support; evaluator identity and source component_id unused",
        "max_memory_age_s": c27.MAX_MEMORY_AGE_S,
        "voxel_size_m": VOXEL_SIZE_M,
        "state_transition": {
            "HIT": "occupied now but no old-velocity emission or refresh without PD motion consistency",
            "KNOWN_FREE": "departed and clear",
            "OCCLUDED": "propagate with bounded age",
            "UNSENSED": "retain internally but emit UNKNOWN",
        },
        "status_counts": dict(sorted(global_counts.items())),
    }
    _TRACE_BY_SEQUENCE[sequence] = {
        "sequence": sequence,
        "frames": trace_frames,
        "status_counts": dict(sorted(global_counts.items())),
    }
    return FlowLedger(manifest=manifest, **arrays), sidecar, manifest


def _representative_ray_workload(c25_data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = c25_data["sequences"][0]
    source = row["sources"]
    pd_source = source["ledgers"]["M1_PD_GLOBAL"]
    pdc_source = source["ledgers"]["M1_PDC_GLOBAL"]
    pd = c27._load_arrays(
        Path(pd_source["ledger"]), Path(pd_source["manifest"]),
        {"frames", "frame_time_s", "frame_ego_x_m", "frame_ego_y_m", "frame_ego_yaw_rad", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
    )
    pdc = c27._load_arrays(
        Path(pdc_source["ledger"]), Path(pdc_source["manifest"]),
        {"frames", "offsets", "forward_m", "left_m", "velocity_forward_mps", "velocity_left_mps"},
    )
    chosen = next(
        index
        for index in range(len(pd["frames"]))
        if int(pdc["offsets"][index + 1] - pdc["offsets"][index]) >= 8
    )
    frame = int(pd["frames"][chosen])
    raw = next(
        iter_raw_ray_frames(
            bag_path=Path(source["bag"]).resolve(strict=True),
            frames=[frame],
            frame_time_s=[float(pd["frame_time_s"][chosen])],
            calibration_dir=Path(source["calibration"]).resolve(strict=True),
        )
    )
    sweep = raw.upper or raw.lower
    require(sweep is not None, "c28_representative_raw_sweep_missing")
    endpoints = np.asarray(sweep.world_endpoints_m, dtype=np.float64)
    endpoint_keys = np.floor(endpoints / VOXEL_SIZE_M).astype(np.int64)
    stride = max(1, len(endpoint_keys) // 32)
    hit_queries = endpoint_keys[::stride][:32]
    occlusion_queries = hit_queries.copy()
    origin = np.asarray(sweep.world_origin_m, dtype=np.float64)
    centers = (occlusion_queries.astype(np.float64) + 0.5) * VOXEL_SIZE_M
    direction = centers - origin
    shifted = centers + direction / np.maximum(np.linalg.norm(direction, axis=1)[:, None], EPSILON) * (2.0 * VOXEL_SIZE_M)
    occlusion_queries = np.floor(shifted / VOXEL_SIZE_M).astype(np.int64)
    queries = np.concatenate([hit_queries, occlusion_queries], axis=0)
    return origin, endpoints, queries


def _select_ray_backend(
    origin: np.ndarray,
    endpoints: np.ndarray,
    queries: np.ndarray,
    receipt_path: Path,
) -> dict[str, Any]:
    cpu_cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cpu_cache["value"] = _cpu_ray_codes(origin, endpoints, queries)
        return cpu_cache["value"]

    def gpu_probe():
        return _gpu_ray_codes(origin, endpoints, queries)

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "value" in cpu_cache and not np.array_equal(
            cpu_cache["value"], output.detach().cpu().numpy()
        ):
            raise BackendSelectionError("C28_RAY_VISIBILITY_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "scipy-cKDTree-ray-box",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"scipy-cKDTree/numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-batched-ray-box",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _SOURCE_BY_SEQUENCE, _RAY_BACKEND, _TRACE_BY_SEQUENCE
    c25_path = args.c25_predictions.resolve(strict=True)
    c25_data = json.loads(c25_path.read_text(encoding="utf-8"))
    require(c25_data.get("schema") == c27.C25_PREDICTION_SCHEMA, "c25_prediction_schema")
    _SOURCE_BY_SEQUENCE = {
        str(row["sequence"]): row["sources"] for row in c25_data["sequences"]
    }
    _TRACE_BY_SEQUENCE = {}
    ray_selection = _select_ray_backend(
        *_representative_ray_workload(c25_data), args.ray_backend_receipt.resolve()
    )
    _RAY_BACKEND = str(ray_selection["selected_backend"])

    original_builder = c27._build_memory
    original_constants = {
        name: getattr(c27, name)
        for name in ("SCHEMA", "PREDICTION_SCHEMA", "STATUS_MET", "STATUS_NOT_MET", "ARM")
    }
    try:
        c27._build_memory = _build_memory
        c27.SCHEMA = SCHEMA
        c27.PREDICTION_SCHEMA = PREDICTION_SCHEMA
        c27.STATUS_MET = STATUS_MET
        c27.STATUS_NOT_MET = STATUS_NOT_MET
        c27.ARM = ARM
        result = c27.run(args)
    finally:
        c27._build_memory = original_builder
        for name, value in original_constants.items():
            setattr(c27, name, value)

    candidate = result["candidate"]
    pdc = result["references"]["M1_PDC_GLOBAL"]
    recovered = int(result["dropout_stress"]["c27_recovered_track_only_window_misses"])
    result["dropout_stress"]["c28_recovered_track_only_window_misses"] = recovered
    visibility_counts = Counter()
    for row in result["per_sequence"]:
        visibility_counts.update(row["memory"]["status_counts"])
    coverage = sum(
        visibility_counts[f"VISIBILITY_{state.value}"]
        for state in Visibility
    )
    causal = (
        visibility_counts["VISIBILITY_HIT"]
        + visibility_counts["VISIBILITY_KNOWN_FREE"]
        + visibility_counts["VISIBILITY_OCCLUDED"]
    )
    evaluable = coverage > 0 and causal > 0
    checks = {
        "contact_recall_12_of_12": int(candidate["bounded_contact_events_recalled"]) == 12,
        "false_segments_at_most_21": int(candidate["false_alert_segments"]) <= 21,
        "dropout_recovery_at_least_30_of_36": recovered >= 30,
        "every_event_lead_not_later_than_pdc": bool(
            result["gate"]["checks"]["every_event_lead_not_later_than_pdc"]
        ),
    }
    passed = evaluable and all(checks.values())
    result["status"] = STATUS_MET if passed else (STATUS_NOT_MET if evaluable else STATUS_NOT_EVALUABLE)
    result["question"] = "Can explicit LiDAR visibility cause preserve PDC precision while restoring detector-independent point continuity?"
    result["gate"] = {
        "passed": passed,
        "evaluable": evaluable,
        "checks": checks,
        "frozen_reference": {
            "pdc_false_segments": pdc["false_alert_segments"],
            "pdc_median_first_alert_lead_s": pdc["median_first_alert_lead_s"],
        },
    }
    result["visibility"] = {
        "status_counts": dict(sorted(visibility_counts.items())),
        "classified_absences": coverage,
        "causally_observed_absences": causal,
    }
    result["fixed_algorithm"] = {
        "baseline": "sealed M1-PDC alert lifecycle is a mechanical subset",
        "height_support": "raw endpoint z-voxels in the same frozen 0.12 m BEV cell as each observed lineage",
        "state_precedence": "HIT > KNOWN_FREE > OCCLUDED > UNSENSED per remembered 3-D support",
        "column_clear": "clear only when every remembered height voxel is KNOWN_FREE",
        "transition": "HIT records present occupancy without old velocity; KNOWN_FREE clears; OCCLUDED alone emits bounded propagation; UNSENSED is internal UNKNOWN; OBSERVED_PD_HIT refreshes motion",
        "maximum_age_s": c27.MAX_MEMORY_AGE_S,
        "route_lifecycle_change": "none",
    }
    result["compute"] = {
        "point_lineage": result["compute"],
        "ray_visibility": ray_selection,
    }
    result["source"]["ray_backend_receipt"] = str(args.ray_backend_receipt.resolve())
    result["source"]["ray_backend_receipt_sha256"] = sha256_file(args.ray_backend_receipt.resolve())
    result["claim_limits"] = [
        "This is the consumed five-sequence C25 Development canary, not fresh confirmation.",
        "The dropout stress removes detector tracks, not raw LiDAR points; it is a continuity guardrail rather than a physical point-occlusion benchmark.",
        "Ray state is geometric evidence over remembered raw endpoint height voxels, not semantic identity or calibrated occupancy probability.",
        "No Android runtime, natural wearer behavior, product reliability, user benefit, or safety claim follows.",
    ]
    trace = {
        "schema": "blindassist-dtr-c29-truth-blind-authority-trace-v1",
        "truth_blind": True,
        "prediction_boundary": "C28 causal lineage/ray features sealed before labels; no sequence identity is passed to candidate policy",
        "sequences": [_TRACE_BY_SEQUENCE[key] for key in sorted(_TRACE_BY_SEQUENCE)],
        "source": {
            "c25_predictions_sha256": sha256_file(c25_path),
            "c28_algorithm_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.observation_cache.resolve(), trace)
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    c25 = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-c28" / "visibility-conditioned-point-memory"
    parser = argparse.ArgumentParser()
    parser.add_argument("--c25-predictions", type=Path, default=c25 / "predictions.json")
    parser.add_argument("--c25-result", type=Path, default=c25 / "result.json")
    parser.add_argument("--roster", type=Path, default=REPO / "research" / "active" / "dtr-r0" / "dtr_c25_fresh_confirmation_roster.json")
    parser.add_argument("--labels", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1" / "train_timestamps.zip")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.json")
    parser.add_argument("--backend-receipt", type=Path, default=root / "point-backend.json")
    parser.add_argument("--ray-backend-receipt", type=Path, default=root / "ray-backend.json")
    parser.add_argument("--observation-cache", type=Path, default=root / "authority-trace.json")
    parser.add_argument("--output", type=Path, default=root / "result.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "gate": result["gate"],
        "candidate": result["candidate"],
        "dropout": result["dropout_stress"],
        "visibility": result["visibility"],
        "compute": result["compute"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
