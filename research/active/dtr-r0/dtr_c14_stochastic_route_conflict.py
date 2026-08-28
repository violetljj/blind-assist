"""Develop confidence-covariant stochastic route conflict over frozen C11.

M1-CT keeps only the final confidence, but its R7 source ledger still contains
the causal observations needed to reproduce the admitted match.  C14 recovers
the per-cell position and velocity disagreement, converts those measured
residuals into a 4-D state covariance, and propagates it through the unchanged
continuous route-collision geometry with a third-degree cubature rule.

This is a consumed-cohort development test.  It replaces probabilistic C11
ONSET (while retaining the imminent-geometry bypass and frozen maintenance),
uses the native CONTACT target, and keeps the decision probability at 0.5.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
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
    require,
    sha256_file,
    write_json,
)
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_c6_world_route_occupancy_belief import (  # noqa: E402
    EPSILON,
    _current_world_cells,
    _entries,
    _probe,
    _relative_state,
    _rotate_world_to_local,
    _select_backend,
)
from dtr_c11_fresh_confirmation import _select_probability_backend  # noqa: E402
from dtr_c11_route_region_probability import (  # noqa: E402
    PROBABILITY_THRESHOLD,
    SequenceEvidence,
    _probability,
    _select_fit_backend,
    extract_sequence,
    fit_platt,
)
from dtr_c12_c13_route_time_research import (  # noqa: E402
    MINIMUM_LEAD_GAIN_S,
    _future_first_passage_target,
    _load_group,
    _timelines,
)
from dtr_m1_confident_direct_velocity import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    POSITION_SIGMA_M,
    SEARCH_RADIUS_M,
    VELOCITY_SIGMA_MPS,
    _history_index,
    _local_to_world,
    _velocity_to_world,
)
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import FROZEN_FLOW_CONFIG, HORIZON_S  # noqa: E402


SCHEMA = "blindassist-dtr-c14-stochastic-route-conflict-v1"
ARM = "M1_SRC_GLOBAL"
C11_ARM = "M1_RROQ_GLOBAL"
STATE_DIMENSION = 4
CUBATURE_POINTS = 2 * STATE_DIMENSION
QUANTIZATION_M = FROZEN_FLOW_CONFIG.voxel_size_m
SEARCH_CELLS = math.ceil(SEARCH_RADIUS_M / FROZEN_FLOW_CONFIG.voxel_size_m)
PROBE_CELLS = 2048


@dataclass(frozen=True)
class C14Evidence:
    base: SequenceEvidence
    stochastic_score: np.ndarray
    stochastic_min_entry_s: np.ndarray
    admitted_cells: int


def _numpy_match(projected: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = np.asarray(projected, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    output_index = np.full(len(current), -1, dtype=np.int64)
    output_residual = np.full((len(current), 2), np.nan, dtype=np.float64)
    if not len(projected) or not len(current):
        return output_index, output_residual
    previous_cells = np.floor(projected / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    for start in range(0, len(current), 512):
        stop = min(start + 512, len(current))
        query = current[start:stop]
        query_cells = np.floor(query / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
        delta_cell = np.abs(query_cells[:, None, :] - previous_cells[None, :, :])
        valid = np.all(delta_cell <= SEARCH_CELLS, axis=2)
        delta = query[:, None, :] - projected[None, :, :]
        distance2 = np.sum(delta * delta, axis=2)
        distance2[~valid] = np.inf
        nearest = np.argmin(distance2, axis=1)
        has_match = np.isfinite(distance2[np.arange(len(query)), nearest])
        rows = np.nonzero(has_match)[0]
        output_index[start + rows] = nearest[rows]
        output_residual[start + rows] = delta[rows, nearest[rows]]
    return output_index, output_residual


def _torch_match(
    projected: np.ndarray, current: np.ndarray, *, numpy_output: bool
) -> Any:
    import torch

    previous = torch.as_tensor(projected, dtype=torch.float64, device="cuda")
    query = torch.as_tensor(current, dtype=torch.float64, device="cuda")
    indices = torch.full((len(current),), -1, dtype=torch.int64, device="cuda")
    residual = torch.full((len(current), 2), float("nan"), dtype=torch.float64, device="cuda")
    if len(previous) and len(query):
        previous_cells = torch.floor(previous / FROZEN_FLOW_CONFIG.voxel_size_m).to(torch.int64)
        for start in range(0, len(query), 512):
            stop = min(start + 512, len(query))
            block = query[start:stop]
            query_cells = torch.floor(block / FROZEN_FLOW_CONFIG.voxel_size_m).to(torch.int64)
            valid = torch.all(
                torch.abs(query_cells[:, None, :] - previous_cells[None, :, :]) <= SEARCH_CELLS,
                dim=2,
            )
            delta = block[:, None, :] - previous[None, :, :]
            distance2 = torch.sum(delta * delta, dim=2)
            distance2 = torch.where(valid, distance2, torch.full_like(distance2, float("inf")))
            minimum, nearest = torch.min(distance2, dim=1)
            has_match = torch.isfinite(minimum)
            rows = torch.nonzero(has_match, as_tuple=False).flatten()
            indices[start + rows] = nearest[rows]
            residual[start + rows] = delta[rows, nearest[rows]]
    if numpy_output:
        return indices.cpu().numpy(), residual.cpu().numpy()
    return indices, residual


def _select_matching_backend(
    projected: np.ndarray, current: np.ndarray, receipt: Path
) -> dict[str, Any]:
    projected = np.asarray(projected[:PROBE_CELLS], dtype=np.float64)
    current = np.asarray(current[:PROBE_CELLS], dtype=np.float64)
    require(bool(len(projected)) and bool(len(current)), "c14_matching_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_match(projected, current)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_match(projected, current, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_match(projected, current, numpy_output=True)
            require(np.array_equal(cache["cpu"][0], gpu[0]), "c14_cpu_gpu_match_index_mismatch")
            finite = np.isfinite(cache["cpu"][1])
            require(np.array_equal(finite, np.isfinite(gpu[1])), "c14_cpu_gpu_match_validity_mismatch")
            require(np.allclose(cache["cpu"][1][finite], gpu[1][finite], atol=1e-10, rtol=1e-10), "c14_cpu_gpu_match_residual_mismatch")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate(
            "numpy-causal-cell-match",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation("cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"),
        ),
        gpu=BackendCandidate(
            "torch-cuda-causal-cell-match",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _match(projected: np.ndarray, current: np.ndarray, backend: str) -> tuple[np.ndarray, np.ndarray]:
    if backend == "torch-cuda-causal-cell-match":
        return _torch_match(projected, current, numpy_output=True)
    return _numpy_match(projected, current)


def _sigma_numpy(
    position: np.ndarray,
    velocity: np.ndarray,
    position_residual: np.ndarray,
    velocity_delta: np.ndarray,
    support: np.ndarray,
    history_span_s: float,
) -> np.ndarray:
    count = len(position)
    if not count:
        return np.empty((0, CUBATURE_POINTS, STATE_DIMENSION), dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    support = np.maximum(np.asarray(support, dtype=np.float64), EPSILON)
    sigma_p = (
        0.5 * position_residual[:, :, None] * position_residual[:, None, :]
        + (QUANTIZATION_M**2 / 12.0) * eye[None, :, :]
    ) / support[:, None, None]
    sigma_v = (
        0.5 * velocity_delta[:, :, None] * velocity_delta[:, None, :]
        + (QUANTIZATION_M**2 / (6.0 * history_span_s**2)) * eye[None, :, :]
    ) / support[:, None, None]
    covariance = np.zeros((count, STATE_DIMENSION, STATE_DIMENSION), dtype=np.float64)
    covariance[:, :2, :2] = sigma_p
    covariance[:, 2:, 2:] = sigma_v
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[:, None, :]
    offsets = np.concatenate([root, -root], axis=2).transpose(0, 2, 1) * math.sqrt(STATE_DIMENSION)
    mean = np.concatenate([position, velocity], axis=1)
    return mean[:, None, :] + offsets


def _sigma_torch(
    position: np.ndarray,
    velocity: np.ndarray,
    position_residual: np.ndarray,
    velocity_delta: np.ndarray,
    support: np.ndarray,
    history_span_s: float,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    p = torch.as_tensor(position, dtype=torch.float64, device="cuda")
    v = torch.as_tensor(velocity, dtype=torch.float64, device="cuda")
    dp = torch.as_tensor(position_residual, dtype=torch.float64, device="cuda")
    dv = torch.as_tensor(velocity_delta, dtype=torch.float64, device="cuda")
    s = torch.clamp(torch.as_tensor(support, dtype=torch.float64, device="cuda"), min=EPSILON)
    eye = torch.eye(2, dtype=torch.float64, device="cuda")
    sigma_p = (0.5 * dp[:, :, None] * dp[:, None, :] + (QUANTIZATION_M**2 / 12.0) * eye) / s[:, None, None]
    sigma_v = (0.5 * dv[:, :, None] * dv[:, None, :] + (QUANTIZATION_M**2 / (6.0 * history_span_s**2)) * eye) / s[:, None, None]
    covariance = torch.zeros((len(p), STATE_DIMENSION, STATE_DIMENSION), dtype=torch.float64, device="cuda")
    covariance[:, :2, :2] = sigma_p
    covariance[:, 2:, 2:] = sigma_v
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = eigenvectors * torch.sqrt(torch.clamp(eigenvalues, min=0.0))[:, None, :]
    offsets = torch.cat([root, -root], dim=2).transpose(1, 2) * math.sqrt(STATE_DIMENSION)
    mean = torch.cat([p, v], dim=1)
    output = mean[:, None, :] + offsets
    return output.cpu().numpy() if numpy_output else output


def _select_sigma_backend(probe: tuple[np.ndarray, ...], receipt: Path) -> dict[str, Any]:
    position, velocity, dp, dv, support, span = probe
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _sigma_numpy(position, velocity, dp, dv, support, float(span[0]))
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _sigma_torch(position, velocity, dp, dv, support, float(span[0]), numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _sigma_torch(position, velocity, dp, dv, support, float(span[0]), numpy_output=True)
            cpu = cache["cpu"]
            cpu_mean = cpu.mean(axis=1)
            gpu_mean = gpu.mean(axis=1)
            cpu_centered = cpu - cpu_mean[:, None, :]
            gpu_centered = gpu - gpu_mean[:, None, :]
            cpu_covariance = np.einsum("nki,nkj->nij", cpu_centered, cpu_centered) / CUBATURE_POINTS
            gpu_covariance = np.einsum("nki,nkj->nij", gpu_centered, gpu_centered) / CUBATURE_POINTS
            require(
                np.allclose(cpu_mean, gpu_mean, atol=1e-9, rtol=1e-9)
                and np.allclose(cpu_covariance, gpu_covariance, atol=1e-8, rtol=1e-8),
                "c14_cpu_gpu_sigma_moment_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate("numpy-cubature", "cpu", cpu_probe, lambda _output: DeviceObservation("cpu", platform.processor() or "CPU", f"numpy-{np.__version__}")),
        gpu=BackendCandidate("torch-cuda-cubature", "cuda", gpu_probe, observe_gpu, synchronize=lambda: __import__("torch").cuda.synchronize()),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _sigma_points(probe: tuple[np.ndarray, ...], backend: str) -> np.ndarray:
    position, velocity, dp, dv, support, span = probe
    if backend == "torch-cuda-cubature":
        return _sigma_torch(position, velocity, dp, dv, support, float(span[0]), numpy_output=True)
    return _sigma_numpy(position, velocity, dp, dv, support, float(span[0]))


def _r7_values(data: Any) -> dict[str, np.ndarray]:
    manifest_path = data.r7_path.with_suffix(".json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m1_manifest_path = data.ct_path.with_suffix(".json").resolve(strict=True)
    m1_manifest = json.loads(m1_manifest_path.read_text(encoding="utf-8"))
    require(sha256_file(data.r7_path) == manifest["ledger_sha256"], "c14_r7_hash_drift")
    require(sha256_file(data.ct_path) == m1_manifest["ledger_sha256"], "c14_m1_hash_drift")
    require(m1_manifest["source"]["direct_velocity_ledger_sha256"] == manifest["ledger_sha256"], "c14_m1_r7_source_drift")
    with np.load(data.r7_path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]).copy() for name in values.files}


def _raw_pair(data: Any, values: Mapping[str, np.ndarray], index: int) -> tuple[Any, ...] | None:
    history = _history_index(data.times_s, index)
    if history is None:
        return None
    offsets = values["offsets"]
    start, stop = int(offsets[index]), int(offsets[index + 1])
    previous_start, previous_stop = int(offsets[history]), int(offsets[history + 1])
    if start == stop or previous_start == previous_stop:
        return None
    current_local = np.column_stack([values["forward_m"][start:stop], values["left_m"][start:stop]]).astype(np.float64)
    current_velocity_local = np.column_stack([values["velocity_forward_mps"][start:stop], values["velocity_left_mps"][start:stop]]).astype(np.float64)
    previous_local = np.column_stack([values["forward_m"][previous_start:previous_stop], values["left_m"][previous_start:previous_stop]]).astype(np.float64)
    previous_velocity_local = np.column_stack([values["velocity_forward_mps"][previous_start:previous_stop], values["velocity_left_mps"][previous_start:previous_stop]]).astype(np.float64)
    current_world = _local_to_world(current_local, float(data.ego_x_m[index]), float(data.ego_y_m[index]), float(data.ego_yaw_rad[index]))
    current_velocity_world = _velocity_to_world(current_velocity_local, float(data.ego_yaw_rad[index]))
    previous_world = _local_to_world(previous_local, float(data.ego_x_m[history]), float(data.ego_y_m[history]), float(data.ego_yaw_rad[history]))
    previous_velocity_world = _velocity_to_world(previous_velocity_local, float(data.ego_yaw_rad[history]))
    span = float(data.times_s[index] - data.times_s[history])
    projected = previous_world + previous_velocity_world * span
    return history, start, stop, previous_start, current_local, current_velocity_local, current_world, current_velocity_world, previous_velocity_world, projected, span


def _recover_frame(data: Any, values: Mapping[str, np.ndarray], index: int, matching_backend: str) -> tuple[np.ndarray, ...] | None:
    pair = _raw_pair(data, values, index)
    ct_start, ct_stop = int(data.ct_offsets[index]), int(data.ct_offsets[index + 1])
    if pair is None:
        require(ct_start == ct_stop, f"c14_unmatched_m1_frame:{data.sequence}:{int(data.frames[index])}")
        return None
    history, start, stop, previous_start, current_local, current_velocity_local, current_world, current_velocity_world, previous_velocity_world, projected, span = pair
    nearest, residual_world = _match(projected, current_world, matching_backend)
    valid = nearest >= 0
    confidence = np.zeros(stop - start, dtype=np.float64)
    support = np.zeros(stop - start, dtype=np.float64)
    velocity_delta_world = np.full((stop - start, 2), np.nan, dtype=np.float64)
    rows = np.nonzero(valid)[0]
    if len(rows):
        previous = nearest[rows]
        current_support = np.minimum(1.0, values["source_point_count"][start:stop][rows].astype(np.float64) / 3.0)
        previous_support = np.minimum(1.0, values["source_point_count"][previous_start + previous].astype(np.float64) / 3.0)
        support[rows] = np.minimum.reduce([current_support, previous_support, values["flow_support"][start:stop][rows].astype(np.float64), values["flow_support"][previous_start + previous].astype(np.float64)])
        velocity_delta_world[rows] = current_velocity_world[rows] - previous_velocity_world[previous]
        position_confidence = np.exp(-0.5 * (np.linalg.norm(residual_world[rows], axis=1) / POSITION_SIGMA_M) ** 2)
        velocity_confidence = np.exp(-0.5 * (np.linalg.norm(velocity_delta_world[rows], axis=1) / VELOCITY_SIGMA_MPS) ** 2)
        confidence[rows] = np.minimum.reduce([support[rows], position_confidence, velocity_confidence])
    admitted = confidence >= CONFIDENCE_THRESHOLD
    expected_position = np.column_stack([data.ct_forward_m[ct_start:ct_stop], data.ct_left_m[ct_start:ct_stop]])
    expected_velocity = np.column_stack([data.ct_velocity_forward_mps[ct_start:ct_stop], data.ct_velocity_left_mps[ct_start:ct_stop]])
    require(np.array_equal(current_local[admitted], expected_position), f"c14_m1_position_replay_mismatch:{data.sequence}:{int(data.frames[index])}")
    require(np.array_equal(current_velocity_local[admitted], expected_velocity), f"c14_m1_velocity_replay_mismatch:{data.sequence}:{int(data.frames[index])}")
    require(np.allclose(confidence[admitted], data.ct_confidence[ct_start:ct_stop], atol=1e-7, rtol=1e-7), f"c14_m1_confidence_replay_mismatch:{data.sequence}:{int(data.frames[index])}")
    if not np.any(admitted):
        return None
    world_position, world_velocity, _ = _current_world_cells(data, index)
    seed_time = np.full(len(world_position), float(data.times_s[index]), dtype=np.float64)
    relative_position, relative_velocity = _relative_state(
        seed_position=world_position,
        seed_velocity=world_velocity,
        seed_time_s=seed_time,
        data=data,
        index=index,
    )
    dp_local = _rotate_world_to_local(residual_world[admitted], float(data.ego_yaw_rad[index]))
    dv_local = _rotate_world_to_local(velocity_delta_world[admitted], float(data.ego_yaw_rad[index]))
    return relative_position, relative_velocity, dp_local, dv_local, support[admitted], np.full(len(relative_position), span, dtype=np.float64)


def _matching_probe(rows: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    for data in rows:
        values = _r7_values(data)
        for index in range(len(data.frames)):
            pair = _raw_pair(data, values, index)
            if pair is not None:
                return pair[9], pair[6]
    raise RuntimeError("c14_matching_probe_missing")


def _uncertainty_probe(rows: Sequence[Any], matching_backend: str) -> tuple[np.ndarray, ...]:
    for data in rows:
        values = _r7_values(data)
        for index in range(len(data.frames)):
            recovered = _recover_frame(data, values, index, matching_backend)
            if recovered is not None and len(recovered[0]):
                return tuple(value[:PROBE_CELLS] for value in recovered)
    raise RuntimeError("c14_uncertainty_probe_missing")


def _frame_score(recovered: tuple[np.ndarray, ...] | None, sigma_backend: str, route_backend: str, confidence: np.ndarray) -> tuple[float, float]:
    if recovered is None:
        return 0.0, float("nan")
    samples = _sigma_points(recovered, sigma_backend)
    entries = _entries(samples[:, :, :2].reshape(-1, 2), samples[:, :, 2:].reshape(-1, 2), route_backend).reshape(len(samples), CUBATURE_POINTS)
    mass = np.where(np.isfinite(entries), np.exp(-np.nan_to_num(entries, nan=0.0) / HORIZON_S), 0.0).mean(axis=1) * confidence
    cells = np.floor(recovered[0] / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    collapsed: dict[tuple[int, int], float] = {}
    for cell, value in zip(cells, mass):
        key = (int(cell[0]), int(cell[1]))
        collapsed[key] = max(collapsed.get(key, 0.0), float(value))
    finite = entries[np.isfinite(entries)]
    return math.log1p(FROZEN_FLOW_CONFIG.voxel_size_m**2 * sum(collapsed.values())), (float(finite.min()) if len(finite) else float("nan"))


def extract_c14(data: Any, matching_backend: str, sigma_backend: str, route_backend: str) -> C14Evidence:
    base = extract_sequence(data, route_backend)
    values = _r7_values(data)
    scores, minimum = [], []
    admitted = 0
    for index in range(len(data.frames)):
        recovered = _recover_frame(data, values, index, matching_backend)
        start, stop = int(data.ct_offsets[index]), int(data.ct_offsets[index + 1])
        score, entry = _frame_score(recovered, sigma_backend, route_backend, data.ct_confidence[start:stop])
        scores.append(score)
        minimum.append(entry)
        admitted += stop - start
    return C14Evidence(base, np.asarray(scores), np.asarray(minimum), admitted)


def predict_c14(evidence: C14Evidence, onset_model: Sequence[float], c11_maintenance: Sequence[float], probability_backend: str) -> dict[str, Any]:
    onset_probability = _probability(evidence.stochastic_score, onset_model, probability_backend)
    maintenance_probability = _probability(evidence.base.reachable_score, c11_maintenance, probability_backend)
    config = DTRConfig()
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard = HORIZON_S * FROZEN_R2_CONFIG.imminent_horizon_fraction
    output: dict[str, Any] = {"raw_alert_frames": [], "active_alert_frames": [], "urgent_frames": [], "minimum_entry_s_by_frame": {}, "risky_cells_by_frame": {}, "route_region_probability_by_frame": {}}
    stochastic_onset_frames = maintenance_only_frames = imminent_origin_frames = 0
    for index, frame_value in enumerate(evidence.base.frames):
        frame = int(frame_value)
        imminent = bool(np.isfinite(evidence.base.current_min_entry_s[index]) and evidence.base.current_min_entry_s[index] <= guard + EPSILON)
        probabilistic = bool(onset_probability[index] >= PROBABILITY_THRESHOLD)
        onset = probabilistic or imminent
        maintain = bool(lifecycle.active and maintenance_probability[index] >= PROBABILITY_THRESHOLD)
        raw = onset or maintain
        minimum = float(evidence.stochastic_min_entry_s[index]) if onset else float(evidence.base.reachable_min_entry_s[index]) if maintain else float("nan")
        urgent = bool(raw and np.isfinite(minimum) and minimum <= guard + EPSILON)
        signal = lifecycle.update(float(evidence.base.times_s[index] - evidence.base.times_s[0]), raw, urgent=urgent)
        output["route_region_probability_by_frame"][str(frame)] = float(max(onset_probability[index], maintenance_probability[index] if lifecycle.active else 0.0))
        stochastic_onset_frames += int(probabilistic)
        imminent_origin_frames += int(imminent and not probabilistic)
        maintenance_only_frames += int(maintain and not onset)
        if raw:
            output["raw_alert_frames"].append(frame)
            output["minimum_entry_s_by_frame"][str(frame)] = minimum
            output["risky_cells_by_frame"][str(frame)] = 1
        if urgent:
            output["urgent_frames"].append(frame)
        if signal in ACTIVE_SIGNALS:
            output["active_alert_frames"].append(frame)
    output["diagnostics"] = {"frames": len(evidence.base.frames), "admitted_cells": evidence.admitted_cells, "stochastic_evidence_frames": int(np.count_nonzero(evidence.stochastic_score)), "stochastic_onset_frames": stochastic_onset_frames, "imminent_geometry_origin_frames": imminent_origin_frames, "maintenance_only_frames": maintenance_only_frames, "active_alert_frames": len(output["active_alert_frames"])}
    return output


def _score_group(rows: Sequence[Any], evidence: Mapping[str, C14Evidence], timelines: Mapping[str, Sequence[Mapping[str, Any]]], *, c11_onset: Sequence[float], c11_maintenance: Sequence[float], c14_onset: Sequence[float], probability_backend: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from dtr_c11_route_region_probability import predict as predict_c11

    baseline_rows, candidate_rows, details = [], [], []
    for data in rows:
        item = evidence[data.sequence]
        baseline = predict_c11(item.base, c11_onset, c11_maintenance, probability_backend=probability_backend)
        candidate = predict_c14(item, c14_onset, c11_maintenance, probability_backend)
        timeline = timelines[data.sequence]
        baseline_score = score_sequence(sequence=data.sequence, timeline=timeline, prediction_frames=_prediction_frames(item.base.frames.tolist(), baseline))
        candidate_score = score_sequence(sequence=data.sequence, timeline=timeline, prediction_frames=_prediction_frames(item.base.frames.tolist(), candidate))
        baseline_rows.append(baseline_score)
        candidate_rows.append(candidate_score)
        details.append({"sequence": data.sequence, "scores": {C11_ARM: baseline_score, ARM: candidate_score}, "diagnostics": candidate["diagnostics"]})
    return aggregate_scores(baseline_rows), aggregate_scores(candidate_rows), details


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamps = args.timestamps.resolve(strict=True)
    labels = args.labels.resolve(strict=True)
    calibrator_path = args.c11_calibrator.resolve(strict=True)
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    c11_onset = [float(calibrator["model"]["onset_platt_slope"]), float(calibrator["model"]["onset_platt_intercept"])]
    c11_maintenance = [float(calibrator["model"]["maintenance_platt_slope"]), float(calibrator["model"]["maintenance_platt_intercept"])]
    train_rows, train_timestamps = _load_group(args.training_ledger_roots, timestamps)
    validation_rows, validation_timestamps = _load_group(args.validation_ledger_roots, timestamps)
    all_rows = [*train_rows, *validation_rows]
    projected, current = _matching_probe(all_rows)
    matching_selection = _select_matching_backend(projected, current, args.matching_backend_receipt.resolve())
    matching_backend = str(matching_selection["selected_backend"])
    uncertainty_probe = _uncertainty_probe(all_rows, matching_backend)
    sigma_selection = _select_sigma_backend(uncertainty_probe, args.sigma_backend_receipt.resolve())
    sigma_backend = str(sigma_selection["selected_backend"])
    sample_probe = _sigma_points(uncertainty_probe, sigma_backend)
    route_selection = _select_backend(sample_probe[:, :, :2].reshape(-1, 2), sample_probe[:, :, 2:].reshape(-1, 2), args.route_backend_receipt.resolve())
    route_backend = str(route_selection["selected_backend"])
    evidence = {data.sequence: extract_c14(data, matching_backend, sigma_backend, route_backend) for data in all_rows}
    train_timelines = _timelines(train_rows, train_timestamps, labels)
    validation_timelines = _timelines(validation_rows, validation_timestamps, labels)
    train_x, train_y, target_rows = [], [], {}
    for data in train_rows:
        eligible, target = _future_first_passage_target(train_timelines[data.sequence])
        train_x.append(evidence[data.sequence].stochastic_score[eligible])
        train_y.append(target[eligible])
        target_rows[data.sequence] = {"eligible_frames": int(np.count_nonzero(eligible)), "positive_frames": int(target[eligible].sum())}
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    require(bool(len(x)) and 0 < float(y.mean()) < 1, "c14_target_degenerate")
    fit_selection = _select_fit_backend(x, y, args.fit_backend_receipt.resolve())
    fit_backend = str(fit_selection["selected_backend"])
    development_model = fit_platt(x, y, fit_backend)
    probability_selection = _select_probability_backend(np.concatenate([item.base.current_score for item in evidence.values()]), np.concatenate([item.stochastic_score for item in evidence.values()]), c11_onset, development_model.tolist(), args.probability_backend_receipt.resolve())
    probability_backend = str(probability_selection["selected_backend"])
    baseline, candidate, details = _score_group(validation_rows, evidence, validation_timelines, c11_onset=c11_onset, c11_maintenance=c11_maintenance, c14_onset=development_model, probability_backend=probability_backend)
    lead_gain = float(candidate["median_first_alert_lead_s"]) - float(baseline["median_first_alert_lead_s"])
    gate = {"recall_not_lower": candidate["bounded_contact_events_recalled"] >= baseline["bounded_contact_events_recalled"], "false_segments_not_higher": candidate["false_alert_segments"] <= baseline["false_alert_segments"], "median_lead_gain_at_least_s": MINIMUM_LEAD_GAIN_S, "observed_median_lead_gain_s": lead_gain}
    passed = bool(gate["recall_not_lower"] and gate["false_segments_not_higher"] and lead_gain >= MINIMUM_LEAD_GAIN_S - EPSILON)
    result = {
        "schema": SCHEMA,
        "status": "DTR_C14_STOCHASTIC_ROUTE_CONFLICT_DEVELOPMENT_GATE_MET" if passed else "DTR_C14_STOCHASTIC_ROUTE_CONFLICT_DEVELOPMENT_GATE_NOT_MET",
        "question": "Can measured point-wise velocity uncertainty suppress pseudo-motion while recovering earlier true route conflicts?",
        "fixed_gate": gate,
        "development_model": {"slope": float(development_model[0]), "intercept": float(development_model[1]), "decision_probability": PROBABILITY_THRESHOLD},
        "development_validation": {C11_ARM: baseline, ARM: candidate, "per_sequence": details},
        "feature": {"name": "confidence-covariant stochastic route conflict", "state_dimension": STATE_DIMENSION, "cubature_points": CUBATURE_POINTS, "cubature_weight": 1.0 / CUBATURE_POINTS, "position_covariance": "(0.5*dp*dp^T + voxel_size^2/12*I)/support", "velocity_covariance": "(0.5*dv*dv^T + voxel_size^2/(6*history_span^2)*I)/support", "collision_mass": "mean_sigma[finite_entry * exp(-entry/3s)] * M1 confidence", "decision": "C14 probabilistic onset replaces C11 probabilistic onset; imminent geometry and C11 maintenance unchanged"},
        "backends": {"matching": matching_selection, "sigma": sigma_selection, "route": route_selection, "fit": fit_selection, "probability": probability_selection},
        "sources": {"c11_calibrator": str(calibrator_path), "c11_calibrator_sha256": sha256_file(calibrator_path), "timestamps": str(timestamps), "timestamps_sha256": sha256_file(timestamps), "labels": str(labels), "labels_sha256": sha256_file(labels), "training_ledger_roots": [str(path.resolve()) for path in args.training_ledger_roots], "validation_ledger_roots": [str(path.resolve()) for path in args.validation_ledger_roots]},
        "training_target": {"name": "native CONTACT: realized path intersection within frozen future horizon", "negative_censoring": "UNKNOWN excluded; CLEAR and PROXIMITY negatives", "by_sequence": target_rows},
        "external_basis": ["Vinod and Oishi: probabilistic occupancy from forward stochastic reachability", "Li et al. RigidFlow++: correspondence confidence from spatial and temporal consistency", "DifFlow3D: per-point scene-flow uncertainty as reliability"],
        "claim_limits": ["The four C11 confirmation sequences are consumed Development validation for C14.", "Future CONTACT truth enters fitting/scoring only, never inference.", "No remaining algorithm-fresh sequence is opened unless this fixed gate passes."],
    }
    write_json(args.output.resolve(), result)
    if passed:
        frozen = {"schema": SCHEMA, "status": "DTR_C14_STOCHASTIC_ROUTE_CONFLICT_MODEL_FROZEN", "model": result["development_model"], "c11_model": calibrator["model"], "feature": result["feature"], "development_validation": {C11_ARM: baseline, ARM: candidate, "fixed_gate": gate}, "sources": result["sources"], "claim_limits": result["claim_limits"]}
        write_json(args.frozen_model.resolve(), frozen)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = REPO / "artifacts.local" / "evidence"
    output_root = evidence / "dtr-c14" / "stochastic-route-conflict"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ledger-roots", type=Path, nargs="+", default=[evidence / "dtr-c2" / "fresh-global-obb-replay" / "ledgers", evidence / "dtr-c10" / "fresh-confirmation" / "ledgers"])
    parser.add_argument("--validation-ledger-roots", type=Path, nargs="+", default=[evidence / "dtr-c11" / "fresh-confirmation" / "ledgers"])
    parser.add_argument("--c11-calibrator", type=Path, default=Path(__file__).resolve().with_name("dtr_c11_route_region_calibrator.json"))
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--matching-backend-receipt", type=Path, default=output_root / "backend-matching.json")
    parser.add_argument("--sigma-backend-receipt", type=Path, default=output_root / "backend-sigma.json")
    parser.add_argument("--route-backend-receipt", type=Path, default=output_root / "backend-route.json")
    parser.add_argument("--fit-backend-receipt", type=Path, default=output_root / "backend-fit.json")
    parser.add_argument("--probability-backend-receipt", type=Path, default=output_root / "backend-probability.json")
    parser.add_argument("--output", type=Path, default=output_root / "result.json")
    parser.add_argument("--frozen-model", type=Path, default=Path(__file__).resolve().with_name("dtr_c14_stochastic_route_conflict_model.json"))
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "gate": result["fixed_gate"], "development": result["development_validation"]}))


if __name__ == "__main__":
    main()
