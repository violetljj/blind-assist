"""Predict global path conflict with wearer-relative world occupancy belief.

C6 fixes the remaining geometry mismatch in C4: scene-flow velocity is an
absolute world motion estimate, while collision risk depends on motion relative
to the wearer's short-term route.  The truth-blind phase estimates wearer
velocity from past ego poses, transforms high-confidence M1-CT cells into the
world frame, and evaluates relative continuous collision geometry.  A second
arm retains those cells for the already frozen 0.5 second lifecycle grace and
advects them through short observation gaps.

Predictions are hash sealed before the C1 roster or future native OBB labels are
opened.  No detector box, target identity, future pose, or tuned route threshold
enters prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import (  # noqa: E402
    _ledger_paths,
    _load_arm_ledger,
    _prediction_frames,
    _sequence_names,
)
from dtr_m1_confident_direct_velocity import (  # noqa: E402
    _local_to_world,
    _velocity_to_world,
)
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    MINIMUM_CLOSING_SPEED_MPS,
    ROUTE_HALF_WIDTH_M,
    _history_index,
    load_flow_ledger,
)


SCHEMA = "blindassist-dtr-c6-world-route-occupancy-belief-development-v1"
PREDICTION_SCHEMA = "blindassist-dtr-c6-sealed-world-route-occupancy-belief-v1"
STATUS = "DTR_C6_WORLD_ROUTE_OCCUPANCY_BELIEF_DEVELOPMENT_MEASURED"
ARMS = ("M1_WR_GLOBAL", "M1_WB_GLOBAL")
PROBE_CELLS = 2048
EPSILON = 1e-9


@dataclass(frozen=True)
class SequenceData:
    sequence: str
    frames: np.ndarray
    times_s: np.ndarray
    ego_x_m: np.ndarray
    ego_y_m: np.ndarray
    ego_yaw_rad: np.ndarray
    ct_offsets: np.ndarray
    ct_forward_m: np.ndarray
    ct_left_m: np.ndarray
    ct_velocity_forward_mps: np.ndarray
    ct_velocity_left_mps: np.ndarray
    ct_confidence: np.ndarray
    ct_path: Path
    r7_path: Path


def _load_sequence(
    *,
    sequence: str,
    timestamps: Mapping[int, float],
    c2_root: Path,
    c3_root: Path,
) -> SequenceData:
    frames = np.asarray(sorted(timestamps), dtype=np.int64)
    paths = _ledger_paths(c2_root, c3_root, sequence)
    ct_path = paths["M1_CT_GLOBAL"].resolve(strict=True)
    r7_path = paths["R7_P_GLOBAL"].resolve(strict=True)
    _load_arm_ledger(
        "M1_CT_GLOBAL",
        ct_path,
        sequence=sequence,
        frames=frames.tolist(),
    )
    load_flow_ledger(
        r7_path,
        r7_path.with_suffix(".json").resolve(strict=True),
        expected_sequence=sequence,
        expected_frames=frames.tolist(),
    )
    with np.load(ct_path, allow_pickle=False) as ct_values:
        ct = {name: np.asarray(ct_values[name]).copy() for name in ct_values.files}
    with np.load(r7_path, allow_pickle=False) as r7_values:
        pose = {
            name: np.asarray(r7_values[name]).copy()
            for name in (
                "frames",
                "frame_time_s",
                "frame_ego_x_m",
                "frame_ego_y_m",
                "frame_ego_yaw_rad",
            )
        }
    require(np.array_equal(ct["frames"], frames), f"ct_frame_drift:{sequence}")
    require(np.array_equal(pose["frames"], frames), f"pose_frame_drift:{sequence}")
    expected_times = np.asarray([timestamps[int(frame)] for frame in frames], dtype=np.float64)
    require(
        np.allclose(pose["frame_time_s"], expected_times, atol=1e-6, rtol=0.0),
        f"pose_timestamp_drift:{sequence}",
    )
    return SequenceData(
        sequence=sequence,
        frames=frames,
        times_s=pose["frame_time_s"].astype(np.float64),
        ego_x_m=pose["frame_ego_x_m"].astype(np.float64),
        ego_y_m=pose["frame_ego_y_m"].astype(np.float64),
        ego_yaw_rad=pose["frame_ego_yaw_rad"].astype(np.float64),
        ct_offsets=ct["offsets"].astype(np.int64),
        ct_forward_m=ct["forward_m"].astype(np.float64),
        ct_left_m=ct["left_m"].astype(np.float64),
        ct_velocity_forward_mps=ct["velocity_forward_mps"].astype(np.float64),
        ct_velocity_left_mps=ct["velocity_left_mps"].astype(np.float64),
        ct_confidence=ct["confidence"].astype(np.float64),
        ct_path=ct_path,
        r7_path=r7_path,
    )


def _numpy_entry(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    require(position.shape == velocity.shape and position.ndim == 2 and position.shape[1] == 2, "entry_shape")
    count = len(position)
    entries = np.full(count, np.nan, dtype=np.float64)
    if not count:
        return entries
    radius = ROUTE_HALF_WIDTH_M
    distance = np.linalg.norm(position, axis=1)
    speed_squared = np.sum(velocity * velocity, axis=1)
    dot = np.sum(position * velocity, axis=1)
    inside = distance <= radius + EPSILON
    closing = np.where(
        distance <= EPSILON,
        np.sqrt(speed_squared),
        -dot / np.maximum(distance, EPSILON),
    )
    valid_inside = inside & (closing + EPSILON >= MINIMUM_CLOSING_SPEED_MPS)
    entries[valid_inside] = 0.0

    outside = ~inside & (speed_squared > EPSILON)
    discriminant = 4.0 * dot * dot - 4.0 * speed_squared * (distance * distance - radius * radius)
    candidate = outside & (discriminant >= 0.0)
    safe_speed = np.maximum(speed_squared, EPSILON)
    root = (-2.0 * dot - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * safe_speed)
    candidate &= (root >= -EPSILON) & (root <= HORIZON_S + EPSILON)
    entry = np.maximum(root, 0.0)
    entry_position = position + velocity * entry[:, None]
    entry_distance = np.maximum(np.linalg.norm(entry_position, axis=1), EPSILON)
    inward = -np.sum(entry_position * velocity, axis=1) / entry_distance
    valid_outside = candidate & (inward + EPSILON >= MINIMUM_CLOSING_SPEED_MPS)
    entries[valid_outside] = entry[valid_outside]
    return entries


def _torch_entry(position: np.ndarray, velocity: np.ndarray, *, numpy_output: bool) -> Any:
    import torch

    p = torch.as_tensor(position, dtype=torch.float64, device="cuda")
    v = torch.as_tensor(velocity, dtype=torch.float64, device="cuda")
    count = p.shape[0]
    entries = torch.full((count,), float("nan"), dtype=torch.float64, device="cuda")
    if count:
        distance = torch.linalg.vector_norm(p, dim=1)
        speed_squared = torch.sum(v * v, dim=1)
        dot = torch.sum(p * v, dim=1)
        inside = distance <= ROUTE_HALF_WIDTH_M + EPSILON
        closing = torch.where(
            distance <= EPSILON,
            torch.sqrt(speed_squared),
            -dot / torch.clamp(distance, min=EPSILON),
        )
        valid_inside = inside & (closing + EPSILON >= MINIMUM_CLOSING_SPEED_MPS)
        entries[valid_inside] = 0.0
        outside = (~inside) & (speed_squared > EPSILON)
        discriminant = 4.0 * dot * dot - 4.0 * speed_squared * (
            distance * distance - ROUTE_HALF_WIDTH_M * ROUTE_HALF_WIDTH_M
        )
        candidate = outside & (discriminant >= 0.0)
        root = (-2.0 * dot - torch.sqrt(torch.clamp(discriminant, min=0.0))) / (
            2.0 * torch.clamp(speed_squared, min=EPSILON)
        )
        candidate = candidate & (root >= -EPSILON) & (root <= HORIZON_S + EPSILON)
        entry = torch.clamp(root, min=0.0)
        entry_position = p + v * entry[:, None]
        entry_distance = torch.clamp(torch.linalg.vector_norm(entry_position, dim=1), min=EPSILON)
        inward = -torch.sum(entry_position * v, dim=1) / entry_distance
        valid_outside = candidate & (inward + EPSILON >= MINIMUM_CLOSING_SPEED_MPS)
        entries[valid_outside] = entry[valid_outside]
    return entries.cpu().numpy() if numpy_output else entries


def _select_backend(
    position: np.ndarray,
    velocity: np.ndarray,
    receipt_path: Path,
) -> dict[str, Any]:
    require(bool(len(position)), "m1_world_route_probe_missing")
    position = np.asarray(position[:PROBE_CELLS], dtype=np.float64)
    velocity = np.asarray(velocity[:PROBE_CELLS], dtype=np.float64)
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_entry(position, velocity)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_entry(position, velocity, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_entry(position, velocity, numpy_output=True)
            cpu = cache["cpu"]
            if not np.array_equal(np.isfinite(cpu), np.isfinite(gpu)):
                raise BackendSelectionError("M1_WORLD_ROUTE_CPU_GPU_VALIDITY_MISMATCH")
            finite = np.isfinite(cpu)
            if not np.allclose(cpu[finite], gpu[finite], atol=1e-9, rtol=1e-9):
                raise BackendSelectionError("M1_WORLD_ROUTE_CPU_GPU_ENTRY_MISMATCH")
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-relative-collision",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation("cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"),
        ),
        gpu=BackendCandidate(
            "torch-cuda-relative-collision",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def _entries(position: np.ndarray, velocity: np.ndarray, backend: str) -> np.ndarray:
    if backend == "torch-cuda-relative-collision":
        return _torch_entry(position, velocity, numpy_output=True)
    return _numpy_entry(position, velocity)


def _rotate_world_to_local(values: np.ndarray, yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return np.asarray(values, dtype=np.float64) @ rotation.T


def _wearer_velocity(data: SequenceData, index: int) -> np.ndarray:
    history = _history_index(data.times_s.tolist(), index, FROZEN_FLOW_CONFIG)
    if history is None:
        return np.zeros(2, dtype=np.float64)
    delta_s = float(data.times_s[index] - data.times_s[history])
    require(delta_s > 0.0, f"wearer_velocity_delta:{data.sequence}:{index}")
    return np.asarray(
        [
            (data.ego_x_m[index] - data.ego_x_m[history]) / delta_s,
            (data.ego_y_m[index] - data.ego_y_m[history]) / delta_s,
        ],
        dtype=np.float64,
    )


def _current_world_cells(data: SequenceData, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(data.ct_offsets[index])
    stop = int(data.ct_offsets[index + 1])
    local_position = np.column_stack(
        [data.ct_forward_m[start:stop], data.ct_left_m[start:stop]]
    )
    local_velocity = np.column_stack(
        [
            data.ct_velocity_forward_mps[start:stop],
            data.ct_velocity_left_mps[start:stop],
        ]
    )
    world_position = _local_to_world(
        local_position,
        float(data.ego_x_m[index]),
        float(data.ego_y_m[index]),
        float(data.ego_yaw_rad[index]),
    )
    world_velocity = _velocity_to_world(local_velocity, float(data.ego_yaw_rad[index]))
    return world_position, world_velocity, data.ct_confidence[start:stop]


def _relative_state(
    *,
    seed_position: np.ndarray,
    seed_velocity: np.ndarray,
    seed_time_s: np.ndarray,
    data: SequenceData,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_time_s = float(data.times_s[index])
    age_s = current_time_s - seed_time_s
    world_position = seed_position + seed_velocity * age_s[:, None]
    relative_world_position = world_position - np.asarray(
        [data.ego_x_m[index], data.ego_y_m[index]], dtype=np.float64
    )
    relative_world_velocity = seed_velocity - _wearer_velocity(data, index)[None, :]
    return (
        _rotate_world_to_local(relative_world_position, float(data.ego_yaw_rad[index])),
        _rotate_world_to_local(relative_world_velocity, float(data.ego_yaw_rad[index])),
    )


def _prediction_template() -> dict[str, Any]:
    return {
        "raw_alert_frames": [],
        "active_alert_frames": [],
        "urgent_frames": [],
        "minimum_entry_s_by_frame": {},
        "risky_cells_by_frame": {},
    }


def _record(
    output: dict[str, Any],
    *,
    frame: int,
    entries: np.ndarray,
    lifecycle: RiskEventLifecycle,
    time_s: float,
    guard_boundary_s: float,
) -> None:
    finite = entries[np.isfinite(entries)]
    minimum = None if not len(finite) else float(finite.min())
    raw = minimum is not None
    urgent = bool(raw and minimum <= guard_boundary_s + EPSILON)
    signal = lifecycle.update(time_s, raw, urgent=urgent)
    if raw:
        output["raw_alert_frames"].append(frame)
        output["minimum_entry_s_by_frame"][str(frame)] = minimum
        output["risky_cells_by_frame"][str(frame)] = int(len(finite))
    if urgent:
        output["urgent_frames"].append(frame)
    if signal in ACTIVE_SIGNALS:
        output["active_alert_frames"].append(frame)


def _predict_sequence(data: SequenceData, backend: str) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    output = {arm: _prediction_template() for arm in ARMS}
    lifecycle = {arm: RiskEventLifecycle(config.clear_grace_s) for arm in ARMS}
    belief_position = np.empty((0, 2), dtype=np.float64)
    belief_velocity = np.empty((0, 2), dtype=np.float64)
    belief_time_s = np.empty(0, dtype=np.float64)
    belief_confidence = np.empty(0, dtype=np.float64)
    current_cells = 0
    retained_cell_evaluations = 0
    relative_route_speed = []

    for index, frame_value in enumerate(data.frames):
        frame = int(frame_value)
        now_s = float(data.times_s[index])
        world_position, world_velocity, confidence = _current_world_cells(data, index)
        current_cells += len(world_position)
        keep = now_s - belief_time_s <= config.clear_grace_s + EPSILON
        belief_position = belief_position[keep]
        belief_velocity = belief_velocity[keep]
        belief_time_s = belief_time_s[keep]
        belief_confidence = belief_confidence[keep]
        if len(world_position):
            belief_position = np.concatenate([belief_position, world_position], axis=0)
            belief_velocity = np.concatenate([belief_velocity, world_velocity], axis=0)
            belief_time_s = np.concatenate(
                [belief_time_s, np.full(len(world_position), now_s, dtype=np.float64)]
            )
            belief_confidence = np.concatenate([belief_confidence, confidence])
        require(
            np.all(belief_confidence >= 0.5 - EPSILON),
            f"belief_confidence_below_source_gate:{data.sequence}:{frame}",
        )
        relative_route_speed.append(float(np.linalg.norm(_wearer_velocity(data, index))))

        current_count = len(world_position)
        if current_count:
            current_position, current_velocity = _relative_state(
                seed_position=world_position,
                seed_velocity=world_velocity,
                seed_time_s=np.full(current_count, now_s, dtype=np.float64),
                data=data,
                index=index,
            )
            current_entries = _entries(current_position, current_velocity, backend)
        else:
            current_entries = np.empty(0, dtype=np.float64)
        _record(
            output["M1_WR_GLOBAL"],
            frame=frame,
            entries=current_entries,
            lifecycle=lifecycle["M1_WR_GLOBAL"],
            time_s=now_s - float(data.times_s[0]),
            guard_boundary_s=guard_boundary_s,
        )

        retained_cell_evaluations += len(belief_position)
        if len(belief_position):
            belief_local_position, belief_local_velocity = _relative_state(
                seed_position=belief_position,
                seed_velocity=belief_velocity,
                seed_time_s=belief_time_s,
                data=data,
                index=index,
            )
            belief_entries = _entries(belief_local_position, belief_local_velocity, backend)
        else:
            belief_entries = np.empty(0, dtype=np.float64)
        _record(
            output["M1_WB_GLOBAL"],
            frame=frame,
            entries=belief_entries,
            lifecycle=lifecycle["M1_WB_GLOBAL"],
            time_s=now_s - float(data.times_s[0]),
            guard_boundary_s=guard_boundary_s,
        )

    for arm in ARMS:
        output[arm]["diagnostics"] = {
            "frames": len(data.frames),
            "current_confident_cells": current_cells,
            "retained_cell_evaluations": (
                current_cells if arm == "M1_WR_GLOBAL" else retained_cell_evaluations
            ),
            "frames_with_route_entry": len(output[arm]["raw_alert_frames"]),
            "active_alert_frames": len(output[arm]["active_alert_frames"]),
            "wearer_speed_mps": {
                "median": float(np.median(relative_route_speed)),
                "p95": float(np.quantile(relative_route_speed, 0.95)),
                "maximum": float(np.max(relative_route_speed)),
            },
        }
    return output


def _probe(data_rows: Sequence[SequenceData]) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    velocities = []
    for data in data_rows:
        for index in range(len(data.frames)):
            world_position, world_velocity, _confidence = _current_world_cells(data, index)
            if not len(world_position):
                continue
            local_position, local_velocity = _relative_state(
                seed_position=world_position,
                seed_velocity=world_velocity,
                seed_time_s=np.full(len(world_position), float(data.times_s[index])),
                data=data,
                index=index,
            )
            positions.append(local_position)
            velocities.append(local_velocity)
            if sum(len(row) for row in positions) >= PROBE_CELLS:
                return np.concatenate(positions)[:PROBE_CELLS], np.concatenate(velocities)[:PROBE_CELLS]
    require(bool(positions), "world_route_probe_no_cells")
    return np.concatenate(positions), np.concatenate(velocities)


def seal_predictions(args: argparse.Namespace) -> dict[str, Any]:
    timestamps_path = args.timestamps.resolve(strict=True)
    c2_root = args.c2_ledger_root.resolve(strict=True)
    c3_root = args.c3_ledger_root.resolve(strict=True)
    sequence_names = _sequence_names(c2_root, c3_root)
    data_rows = []
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in sequence_names:
            data_rows.append(
                _load_sequence(
                    sequence=sequence,
                    timestamps=_load_timestamps(timestamps_zip, sequence),
                    c2_root=c2_root,
                    c3_root=c3_root,
                )
            )
    probe_position, probe_velocity = _probe(data_rows)
    backend_receipt = args.backend_receipt.resolve()
    selection = _select_backend(probe_position, probe_velocity, backend_receipt)
    selected_backend = str(selection["selected_backend"])
    rows = []
    for data in data_rows:
        arms = _predict_sequence(data, selected_backend)
        rows.append(
            {
                "sequence": data.sequence,
                "frames": len(data.frames),
                "arms": arms,
                "sources": {
                    "m1_ct": str(data.ct_path),
                    "m1_ct_sha256": sha256_file(data.ct_path),
                    "r7_pose_source": str(data.r7_path),
                    "r7_pose_source_sha256": sha256_file(data.r7_path),
                },
            }
        )
        print(
            json.dumps(
                {
                    "c6_truth_blind_sequence": data.sequence,
                    "raw_alert_frames": {
                        arm: len(arms[arm]["raw_alert_frames"]) for arm in ARMS
                    },
                }
            ),
            flush=True,
        )
    prediction = {
        "schema": PREDICTION_SCHEMA,
        "truth_blind": True,
        "development_after_c4": True,
        "prediction_boundary": (
            "past/current ego poses plus sealed M1-CT scene motion only; no roster, detector/native boxes, future pose, future labels, or prior scores"
        ),
        "mechanism": {
            "wearer_route": "causal constant velocity from the frozen 0.25-0.45 second pose history",
            "relative_collision": "scene world velocity minus wearer route velocity",
            "belief_frame": "world",
            "belief_retention_s": DTRConfig().clear_grace_s,
            "belief_motion": "constant-velocity advection",
            "source_confidence_gate": "frozen M1-CT >= 0.5",
            "route_thresholds_lifecycle_and_source_ledger": "UNCHANGED",
        },
        "backend": {
            "receipt": str(backend_receipt),
            "receipt_sha256": sha256_file(backend_receipt),
            "selected_backend": selected_backend,
            "selected_device_type": selection["selected_device_type"],
            "selected_device_name": selection["selected_device_name"],
            "selection_reason": selection["selection_reason"],
        },
        "source": {
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "sequences": rows,
    }
    write_json(args.predictions.resolve(), prediction)
    return prediction


def score_predictions(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    require(prediction.get("schema") == PREDICTION_SCHEMA, "prediction_schema_drift")
    require(prediction.get("truth_blind") is True, "prediction_not_truth_blind")
    roster_path = args.roster.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash_drift")
    require(
        roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path),
        "timestamps_hash_drift",
    )
    by_sequence = {str(row["sequence"]): row for row in prediction["sequences"]}
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(set(by_sequence) == set(expected), "prediction_roster_sequence_drift")
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    per_sequence = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence in expected:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(timestamps)
            boxes = _load_boxes(labels, sequence)
            timeline = global_truth_timeline(frames=frames, timestamps=timestamps, boxes_by_frame=boxes)
            row = by_sequence[sequence]
            scores = {}
            for arm in ARMS:
                score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, row["arms"][arm]),
                )
                scores[arm] = score
                per_arm[arm].append(score)
            per_sequence.append(
                {
                    "sequence": sequence,
                    "scores": scores,
                    "prediction_diagnostics": {
                        arm: row["arms"][arm]["diagnostics"] for arm in ARMS
                    },
                }
            )
    aggregate = {arm: aggregate_scores(per_arm[arm]) for arm in ARMS}
    c4_path = args.c4_result.resolve(strict=True)
    c4 = json.loads(c4_path.read_text(encoding="utf-8"))
    require(c4.get("schema") == "blindassist-dtr-c4-detector-independent-global-risk-v1", "c4_schema_drift")
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": (
            "Does wearer-relative world occupancy belief improve detector-independent future path-conflict alerts without route tuning?"
        ),
        "aggregate": aggregate,
        "comparators": {
            "M1_CT_GLOBAL_C4": c4["aggregate"]["M1_CT_GLOBAL"],
            "M1_PDC_GLOBAL_C4": c4["aggregate"]["M1_PDC_GLOBAL"],
        },
        "per_sequence": per_sequence,
        "source": {
            "sealed_predictions": str(prediction_path),
            "sealed_predictions_sha256": prediction_sha256,
            "backend_receipt": prediction["backend"]["receipt"],
            "backend_receipt_sha256": prediction["backend"]["receipt_sha256"],
            "c4_result": str(c4_path),
            "c4_result_sha256": sha256_file(c4_path),
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
        },
        "algorithm_increment": (
            "C6 evaluates high-confidence scene motion relative to a causal wearer route in world coordinates and retains advected occupancy through the unchanged lifecycle grace."
        ),
        "dropout_composition": (
            "The previously sealed M1-HYBRID raw-point/R7 bounded track-gap bridge remains unchanged, so its consumed 9/9 recovery is preserved by composition; C6 changes only detector-independent natural alert origination."
        ),
        "claim_limits": [
            "C6 was designed after C4 on the same seven sequences and is Development evidence, not fresh confirmation.",
            "The wearer route is a causal constant-velocity estimate from past poses, not a semantic navigation plan or turn forecast.",
            "The belief is deterministic constant-velocity occupancy retained for 0.5 seconds, not calibrated probabilistic or multimodal forecasting.",
            "This curated public replay is not product, user-benefit, or safety evidence.",
        ],
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c2 = repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay"
    c3 = repo / "artifacts.local" / "evidence" / "dtr-c3" / "raw-point-direct-velocity-canary"
    c4 = repo / "artifacts.local" / "evidence" / "dtr-c4" / "detector-independent-global-risk"
    c6 = repo / "artifacts.local" / "evidence" / "dtr-c6" / "world-route-occupancy-belief"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster", type=Path, default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json")
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--c2-ledger-root", type=Path, default=c2 / "ledgers")
    parser.add_argument("--c3-ledger-root", type=Path, default=c3 / "ledgers")
    parser.add_argument("--c4-result", type=Path, default=c4 / "result.json")
    parser.add_argument("--backend-receipt", type=Path, default=c6 / "backend.json")
    parser.add_argument("--predictions", type=Path, default=c6 / "predictions.json")
    parser.add_argument("--output", type=Path, default=c6 / "result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal_predictions(args)
    result = score_predictions(args)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
