"""Calibrate C9 route-region occupancy probability on consumed sequences.

The representation is deliberately small.  Confidence-aware M1-CT cells that
can enter the route tube contribute a resolution-aware hazard intensity.  Flow-
advected belief is spatially collapsed after transport so repeated observations
cannot multiply risk merely because the grid or frame rate is denser.  Separate
Platt maps calibrate current-evidence ONSET and active-state maintenance.
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
    CLEAR,
    CONTACT,
    PROXIMITY,
    _load_boxes,
    _load_timestamps,
    global_truth_timeline,
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
    _load_sequence,
    _probe,
    _relative_state,
    _select_backend,
)
from dtr_c8_global_risk_belief_bridge import _current_absolute_entries  # noqa: E402
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
)


SCHEMA = "blindassist-dtr-c11-route-region-probability-calibrator-v1"
ARM = "M1_RROQ_GLOBAL"
PROBABILITY_THRESHOLD = 0.5
PLATT_L2 = 1e-3
PLATT_STEPS = 25
ECE_BINS = 10


@dataclass(frozen=True)
class SequenceEvidence:
    sequence: str
    frames: np.ndarray
    times_s: np.ndarray
    current_score: np.ndarray
    reachable_score: np.ndarray
    current_min_entry_s: np.ndarray
    reachable_min_entry_s: np.ndarray


def _hazard_score(
    position: np.ndarray,
    entries: np.ndarray,
    confidence: np.ndarray,
    age_s: np.ndarray,
) -> tuple[float, float]:
    finite = np.isfinite(entries)
    if not np.any(finite):
        return 0.0, float("nan")
    position = np.asarray(position[finite], dtype=np.float64)
    entries = np.asarray(entries[finite], dtype=np.float64)
    confidence = np.asarray(confidence[finite], dtype=np.float64)
    age_s = np.asarray(age_s[finite], dtype=np.float64)
    weights = (
        confidence
        * np.exp(-entries / HORIZON_S)
        * np.exp(-age_s / DTRConfig().clear_grace_s)
    )
    cells = np.floor(position / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    collapsed: dict[tuple[int, int], float] = {}
    for cell, weight in zip(cells, weights):
        key = (int(cell[0]), int(cell[1]))
        collapsed[key] = max(collapsed.get(key, 0.0), float(weight))
    cell_area_m2 = FROZEN_FLOW_CONFIG.voxel_size_m**2
    intensity = cell_area_m2 * sum(collapsed.values())
    return math.log1p(intensity), float(entries.min())


def extract_sequence(data: Any, backend: str) -> SequenceEvidence:
    current_scores = []
    reachable_scores = []
    current_minimum = []
    reachable_minimum = []
    belief_position = np.empty((0, 2), dtype=np.float64)
    belief_velocity = np.empty((0, 2), dtype=np.float64)
    belief_time_s = np.empty(0, dtype=np.float64)
    belief_confidence = np.empty(0, dtype=np.float64)
    grace_s = DTRConfig().clear_grace_s

    for index in range(len(data.frames)):
        now_s = float(data.times_s[index])
        start = int(data.ct_offsets[index])
        stop = int(data.ct_offsets[index + 1])
        local_position = np.column_stack(
            [data.ct_forward_m[start:stop], data.ct_left_m[start:stop]]
        )
        current_confidence = data.ct_confidence[start:stop]
        current_entries = _current_absolute_entries(data, index, backend)
        current_score, current_entry = _hazard_score(
            local_position,
            current_entries,
            current_confidence,
            np.zeros(len(local_position), dtype=np.float64),
        )

        world_position, world_velocity, confidence = _current_world_cells(data, index)
        keep = now_s - belief_time_s <= grace_s + EPSILON
        belief_position = belief_position[keep]
        belief_velocity = belief_velocity[keep]
        belief_time_s = belief_time_s[keep]
        belief_confidence = belief_confidence[keep]
        if len(world_position):
            belief_position = np.concatenate([belief_position, world_position])
            belief_velocity = np.concatenate([belief_velocity, world_velocity])
            belief_time_s = np.concatenate(
                [belief_time_s, np.full(len(world_position), now_s, dtype=np.float64)]
            )
            belief_confidence = np.concatenate([belief_confidence, confidence])
        require(
            np.all(belief_confidence >= 0.5 - EPSILON),
            f"c11_confidence_below_m1_gate:{data.sequence}:{int(data.frames[index])}",
        )
        if len(belief_position):
            relative_position, relative_velocity = _relative_state(
                seed_position=belief_position,
                seed_velocity=belief_velocity,
                seed_time_s=belief_time_s,
                data=data,
                index=index,
            )
            reachable_entries = _entries(relative_position, relative_velocity, backend)
            reachable_score, reachable_entry = _hazard_score(
                relative_position,
                reachable_entries,
                belief_confidence,
                now_s - belief_time_s,
            )
        else:
            reachable_score, reachable_entry = 0.0, float("nan")
        current_scores.append(current_score)
        reachable_scores.append(reachable_score)
        current_minimum.append(current_entry)
        reachable_minimum.append(reachable_entry)

    return SequenceEvidence(
        sequence=data.sequence,
        frames=np.asarray(data.frames, dtype=np.int64),
        times_s=np.asarray(data.times_s, dtype=np.float64),
        current_score=np.asarray(current_scores, dtype=np.float64),
        reachable_score=np.asarray(reachable_scores, dtype=np.float64),
        current_min_entry_s=np.asarray(current_minimum, dtype=np.float64),
        reachable_min_entry_s=np.asarray(reachable_minimum, dtype=np.float64),
    )


def _fit_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray([1.0, math.log((float(y.mean()) + 1e-6) / (1.0 - float(y.mean()) + 1e-6))])
    for _ in range(PLATT_STEPS):
        logits = np.clip(params[0] * x + params[1], -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = probability - y
        weight = probability * (1.0 - probability)
        gradient = np.asarray(
            [np.sum(residual * x) + PLATT_L2 * params[0], np.sum(residual)]
        )
        hessian = np.asarray(
            [
                [np.sum(weight * x * x) + PLATT_L2, np.sum(weight * x)],
                [np.sum(weight * x), np.sum(weight) + 1e-12],
            ]
        )
        params -= np.linalg.solve(hessian, gradient)
    return params


def _fit_torch(x: np.ndarray, y: np.ndarray, *, numpy_output: bool) -> Any:
    import torch

    tx = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    ty = torch.as_tensor(y, dtype=torch.float64, device="cuda")
    prevalence = torch.mean(ty)
    params = torch.stack(
        [
            torch.ones((), dtype=torch.float64, device="cuda"),
            torch.log((prevalence + 1e-6) / (1.0 - prevalence + 1e-6)),
        ]
    )
    for _ in range(PLATT_STEPS):
        logits = torch.clamp(params[0] * tx + params[1], -40.0, 40.0)
        probability = torch.sigmoid(logits)
        residual = probability - ty
        weight = probability * (1.0 - probability)
        gradient = torch.stack(
            [torch.sum(residual * tx) + PLATT_L2 * params[0], torch.sum(residual)]
        )
        hessian = torch.stack(
            [
                torch.stack([torch.sum(weight * tx * tx) + PLATT_L2, torch.sum(weight * tx)]),
                torch.stack([torch.sum(weight * tx), torch.sum(weight) + 1e-12]),
            ]
        )
        params = params - torch.linalg.solve(hessian, gradient)
    return params.detach().cpu().numpy() if numpy_output else params


def _select_fit_backend(x: np.ndarray, y: np.ndarray, receipt: Path) -> dict[str, Any]:
    cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cache["cpu"] = _fit_numpy(x, y)
        return cache["cpu"]

    def gpu_probe() -> Any:
        return _fit_torch(x, y, numpy_output=False)

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = output.detach().cpu().numpy()
            if not np.allclose(cache["cpu"], gpu, atol=1e-7, rtol=1e-7):
                raise BackendSelectionError("C11_PLATT_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-newton-platt",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-newton-platt",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def fit_platt(x: np.ndarray, y: np.ndarray, backend: str) -> np.ndarray:
    if backend == "torch-cuda-newton-platt":
        return _fit_torch(x, y, numpy_output=True)
    return _fit_numpy(x, y)


def _probability(
    score: np.ndarray,
    params: Sequence[float],
    backend: str = "numpy-platt-inference",
) -> np.ndarray:
    if backend == "torch-cuda-platt-inference":
        import torch

        values = torch.as_tensor(score, dtype=torch.float64, device="cuda")
        probability = torch.sigmoid(float(params[0]) * values + float(params[1]))
        return probability.cpu().numpy()
    logits = np.clip(float(params[0]) * score + float(params[1]), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _calibration_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    epsilon = 1e-12
    order = np.argsort(probability)
    bins = [values for values in np.array_split(order, min(ECE_BINS, len(order))) if len(values)]
    ece = sum(
        len(values) / len(order) * abs(float(probability[values].mean() - truth[values].mean()))
        for values in bins
    )
    return {
        "frames": int(len(truth)),
        "positive_fraction": float(truth.mean()),
        "brier": float(np.mean((probability - truth) ** 2)),
        "nll": float(
            -np.mean(
                truth * np.log(np.clip(probability, epsilon, 1.0))
                + (1.0 - truth) * np.log(np.clip(1.0 - probability, epsilon, 1.0))
            )
        ),
        "ece_equal_count_10": float(ece),
    }


def predict(
    evidence: SequenceEvidence,
    onset: Sequence[float],
    maintenance: Sequence[float],
    probability_backend: str = "numpy-platt-inference",
) -> dict[str, Any]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    onset_probability = _probability(evidence.current_score, onset, probability_backend)
    maintenance_probability = _probability(
        evidence.reachable_score, maintenance, probability_backend
    )
    output: dict[str, Any] = {
        "raw_alert_frames": [],
        "active_alert_frames": [],
        "urgent_frames": [],
        "minimum_entry_s_by_frame": {},
        "risky_cells_by_frame": {},
        "route_region_probability_by_frame": {},
    }
    onset_frames = 0
    maintenance_only_frames = 0
    blocked_maintenance_frames = 0
    imminent_geometry_origin_frames = 0
    for index, frame_value in enumerate(evidence.frames):
        frame = int(frame_value)
        current_evidence = bool(np.isfinite(evidence.current_min_entry_s[index]))
        imminent_geometry = bool(
            current_evidence
            and evidence.current_min_entry_s[index] <= guard + EPSILON
        )
        probabilistic_onset = bool(onset_probability[index] >= PROBABILITY_THRESHOLD)
        onset = bool(probabilistic_onset or imminent_geometry)
        maintain = bool(
            lifecycle.active and maintenance_probability[index] >= PROBABILITY_THRESHOLD
        )
        imminent_geometry_origin_frames += int(imminent_geometry and not probabilistic_onset)
        blocked_maintenance_frames += int(
            not lifecycle.active
            and maintenance_probability[index] >= PROBABILITY_THRESHOLD
            and not onset
        )
        raw = bool(onset or maintain)
        onset_frames += int(onset)
        maintenance_only_frames += int(maintain and not onset)
        probability = float(
            max(
                onset_probability[index],
                maintenance_probability[index] if lifecycle.active else 0.0,
            )
        )
        minimum = (
            float(evidence.current_min_entry_s[index])
            if onset
            else float(evidence.reachable_min_entry_s[index])
            if maintain
            else float("nan")
        )
        urgent = bool(raw and np.isfinite(minimum) and minimum <= guard + EPSILON)
        signal = lifecycle.update(
            float(evidence.times_s[index] - evidence.times_s[0]), raw, urgent=urgent
        )
        output["route_region_probability_by_frame"][str(frame)] = probability
        if raw:
            output["raw_alert_frames"].append(frame)
            output["minimum_entry_s_by_frame"][str(frame)] = minimum
            output["risky_cells_by_frame"][str(frame)] = 1
        if urgent:
            output["urgent_frames"].append(frame)
        if signal in ACTIVE_SIGNALS:
            output["active_alert_frames"].append(frame)
    output["diagnostics"] = {
        "frames": len(evidence.frames),
        "onset_probability_frames": onset_frames,
        "maintenance_only_probability_frames": maintenance_only_frames,
        "blocked_maintenance_probability_frames": blocked_maintenance_frames,
        "imminent_geometry_origin_frames": imminent_geometry_origin_frames,
        "active_alert_frames": len(output["active_alert_frames"]),
    }
    return output


def _sequence_names(roots: Sequence[Path]) -> list[tuple[str, Path]]:
    output: dict[str, Path] = {}
    for root in roots:
        for path in root.iterdir():
            if path.is_dir():
                require(path.name not in output, f"duplicate_calibration_sequence:{path.name}")
                output[path.name] = root
    require(bool(output), "no_calibration_sequences")
    return sorted(output.items())


def _truth(
    *, sequence: str, frames: Sequence[int], labels: zipfile.ZipFile, timestamps: Mapping[int, float]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    timeline = global_truth_timeline(
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=_load_boxes(labels, sequence),
    )
    known = np.asarray([row["label"] in {CONTACT, PROXIMITY, CLEAR} for row in timeline])
    truth = np.asarray([row["label"] == CONTACT for row in timeline], dtype=np.float64)
    return known, timeline


def fit(args: argparse.Namespace) -> dict[str, Any]:
    timestamps_path = args.timestamps.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    roots = [path.resolve(strict=True) for path in args.ledger_roots]
    sequence_roots = _sequence_names(roots)
    data_rows = []
    timestamps_by_sequence: dict[str, dict[int, float]] = {}
    with zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence, root in sequence_roots:
            timestamps = _load_timestamps(timestamps_zip, sequence)
            timestamps_by_sequence[sequence] = timestamps
            data_rows.append(
                _load_sequence(
                    sequence=sequence,
                    timestamps=timestamps,
                    c2_root=root,
                    c3_root=root,
                )
            )
    probe_position, probe_velocity = _probe(data_rows)
    route_receipt = args.route_backend_receipt.resolve()
    route_selection = _select_backend(probe_position, probe_velocity, route_receipt)
    route_backend = str(route_selection["selected_backend"])
    evidence = {
        data.sequence: extract_sequence(data, route_backend) for data in data_rows
    }

    known_by_sequence: dict[str, np.ndarray] = {}
    truth_by_sequence: dict[str, np.ndarray] = {}
    timeline_by_sequence: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(labels_path) as labels:
        for sequence, _root in sequence_roots:
            known, timeline = _truth(
                sequence=sequence,
                frames=evidence[sequence].frames.tolist(),
                labels=labels,
                timestamps=timestamps_by_sequence[sequence],
            )
            known_by_sequence[sequence] = known
            truth_by_sequence[sequence] = np.asarray(
                [row["label"] == CONTACT for row in timeline], dtype=np.float64
            )
            timeline_by_sequence[sequence] = timeline

    all_onset = np.concatenate(
        [evidence[name].current_score[known_by_sequence[name]] for name, _ in sequence_roots]
    )
    all_maintenance = np.concatenate(
        [evidence[name].reachable_score[known_by_sequence[name]] for name, _ in sequence_roots]
    )
    all_truth = np.concatenate(
        [truth_by_sequence[name][known_by_sequence[name]] for name, _ in sequence_roots]
    )
    fit_receipt = args.fit_backend_receipt.resolve()
    fit_selection = _select_fit_backend(all_onset, all_truth, fit_receipt)
    fit_backend = str(fit_selection["selected_backend"])

    oof_probability_onset = []
    oof_probability_maintenance = []
    oof_truth = []
    oof_scores = []
    per_sequence = []
    names = [name for name, _root in sequence_roots]
    for held_out in names:
        train_names = [name for name in names if name != held_out]
        train_truth = np.concatenate(
            [truth_by_sequence[name][known_by_sequence[name]] for name in train_names]
        )
        onset_params = fit_platt(
            np.concatenate(
                [evidence[name].current_score[known_by_sequence[name]] for name in train_names]
            ),
            train_truth,
            fit_backend,
        )
        maintenance_params = fit_platt(
            np.concatenate(
                [evidence[name].reachable_score[known_by_sequence[name]] for name in train_names]
            ),
            train_truth,
            fit_backend,
        )
        held = evidence[held_out]
        known = known_by_sequence[held_out]
        truth = truth_by_sequence[held_out][known]
        onset_probability = _probability(held.current_score[known], onset_params)
        maintenance_probability = _probability(held.reachable_score[known], maintenance_params)
        oof_probability_onset.append(onset_probability)
        oof_probability_maintenance.append(maintenance_probability)
        oof_truth.append(truth)
        arm = predict(held, onset_params, maintenance_params)
        sequence_score = score_sequence(
            sequence=held_out,
            timeline=timeline_by_sequence[held_out],
            prediction_frames=_prediction_frames(held.frames.tolist(), arm),
        )
        oof_scores.append(sequence_score)
        per_sequence.append(
            {
                "sequence": held_out,
                "onset": [float(value) for value in onset_params],
                "maintenance": [float(value) for value in maintenance_params],
                "score": sequence_score,
                "diagnostics": arm["diagnostics"],
            }
        )

    final_onset = fit_platt(all_onset, all_truth, fit_backend)
    final_maintenance = fit_platt(all_maintenance, all_truth, fit_backend)
    result = {
        "schema": SCHEMA,
        "status": "DTR_C11_ROUTE_REGION_PROBABILITY_CALIBRATOR_FROZEN",
        "model": {
            "onset_platt_slope": float(final_onset[0]),
            "onset_platt_intercept": float(final_onset[1]),
            "maintenance_platt_slope": float(final_maintenance[0]),
            "maintenance_platt_intercept": float(final_maintenance[1]),
            "decision_probability": PROBABILITY_THRESHOLD,
            "l2": PLATT_L2,
            "newton_steps": PLATT_STEPS,
        },
        "feature": {
            "name": "flow-reachable route-region hazard intensity",
            "formula": (
                "log1p(voxel_area * sum over unique advected route cells of "
                "max(confidence * exp(-entry/3s) * exp(-age/0.5s))))"
            ),
            "onset_source": "current M1-CT route-entry cells",
            "maintenance_source": "0.5s flow-advected M1-CT belief collapsed after transport",
            "belief_cannot_originate_from_clear": True,
            "onset_authority": (
                "calibrated current route-region probability >= 0.5 or unchanged "
                "imminent continuous-collision geometry"
            ),
            "grid_resolution_m": FROZEN_FLOW_CONFIG.voxel_size_m,
        },
        "leave_one_sequence_out": {
            "sequences": len(names),
            "onset_calibration": _calibration_metrics(
                np.concatenate(oof_probability_onset), np.concatenate(oof_truth)
            ),
            "maintenance_calibration": _calibration_metrics(
                np.concatenate(oof_probability_maintenance), np.concatenate(oof_truth)
            ),
            "event_score": aggregate_scores(oof_scores),
            "per_sequence": per_sequence,
        },
        "training": {
            "sequences": names,
            "known_frames": int(len(all_truth)),
            "contact_frames": int(all_truth.sum()),
            "truth_role": "consumed Development plus consumed fixed-C9 confirmation only",
        },
        "backends": {
            "route": {
                "receipt": str(route_receipt),
                "receipt_sha256": sha256_file(route_receipt),
                "selected_backend": route_backend,
                "selected_device_type": route_selection["selected_device_type"],
                "selection_reason": route_selection["selection_reason"],
            },
            "fit": {
                "receipt": str(fit_receipt),
                "receipt_sha256": sha256_file(fit_receipt),
                "selected_backend": fit_backend,
                "selected_device_type": fit_selection["selected_device_type"],
                "selection_reason": fit_selection["selection_reason"],
            },
        },
        "sources": {
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "ledger_roots": [str(path) for path in roots],
        },
        "claim_limits": [
            "The coefficients use only already consumed sequences; leave-one-sequence-out results are Development model-selection evidence.",
            "M1 confidence is calibrated into route-region CONTACT probability, not asserted to be cell occupancy probability.",
            "A separate algorithm-fresh cohort is required before C11 performance is claimed.",
        ],
    }
    write_json(args.output.resolve(), result)
    frozen = {
        "schema": SCHEMA,
        "status": result["status"],
        "model": result["model"],
        "feature": result["feature"],
        "development_leave_one_sequence_out": {
            key: value
            for key, value in result["leave_one_sequence_out"].items()
            if key != "per_sequence"
        },
        "training": result["training"],
        "backends": result["backends"],
        "sources": result["sources"],
        "claim_limits": result["claim_limits"],
    }
    write_json(args.frozen_model.resolve(), frozen)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    c11 = repo / "artifacts.local" / "evidence" / "dtr-c11" / "route-region-probability"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit",))
    parser.add_argument(
        "--ledger-roots",
        type=Path,
        nargs="+",
        default=[
            repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay" / "ledgers",
            repo / "artifacts.local" / "evidence" / "dtr-c10" / "fresh-confirmation" / "ledgers",
        ],
    )
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--route-backend-receipt", type=Path, default=c11 / "backend-route.json")
    parser.add_argument("--fit-backend-receipt", type=Path, default=c11 / "backend-fit.json")
    parser.add_argument("--output", type=Path, default=c11 / "calibrator.json")
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c11_route_region_calibrator.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "cross_validation": result["leave_one_sequence_out"]["event_score"],
            }
        )
    )


if __name__ == "__main__":
    main()
