"""Develop detector-independent static/mover velocity mixtures over C11.

C14 showed that symmetric point-wise covariance spreading recovers a contact
but converts uncertainty mass into false route conflict.  C15 instead uses the
same measured temporal residual as observation covariance, then asks a more
specific question inside each frame-local occupancy component: is its velocity
better explained by a static world hypothesis or one shared rigid translation?

The two hypotheses are compared with a fixed Gaussian BIC approximation and
their posterior weights are propagated through the unchanged continuous route
geometry.  There is no component-speed threshold, route-threshold change, or
trajectory model.
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
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)

from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence  # noqa: E402
from dtr_c4_detector_independent_global_risk import _prediction_frames  # noqa: E402
from dtr_c6_world_route_occupancy_belief import (  # noqa: E402
    EPSILON,
    _current_world_cells,
    _entries,
    _rotate_world_to_local,
    _select_backend,
    _wearer_velocity,
)
from dtr_c11_fresh_confirmation import _select_probability_backend  # noqa: E402
from dtr_c11_route_region_probability import (  # noqa: E402
    PROBABILITY_THRESHOLD,
    SequenceEvidence,
    _probability,
    _select_fit_backend,
    extract_sequence,
    fit_platt,
    predict as predict_c11,
)
from dtr_c12_c13_route_time_research import (  # noqa: E402
    MINIMUM_LEAD_GAIN_S,
    _future_first_passage_target,
    _load_group,
    _timelines,
)
from dtr_c14_stochastic_route_conflict import (  # noqa: E402
    _matching_probe,
    _r7_values,
    _recover_frame,
    _select_matching_backend,
)
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import FROZEN_FLOW_CONFIG, HORIZON_S  # noqa: E402


SCHEMA = "blindassist-dtr-c15-component-velocity-mixture-v1"
ARM = "M1_CVM_GLOBAL"
C11_ARM = "M1_RROQ_GLOBAL"
MODEL_DIMENSION = 2
PROBE_CELLS = 2048


@dataclass(frozen=True)
class C15Evidence:
    base: SequenceEvidence
    mixture_score: np.ndarray
    mixture_min_entry_s: np.ndarray
    admitted_cells: int
    components: int
    mean_mover_probability: float


def _velocity_covariance(
    velocity_delta: np.ndarray,
    support: np.ndarray,
    history_span_s: float,
) -> np.ndarray:
    eye = np.eye(2, dtype=np.float64)
    return (
        0.5 * velocity_delta[:, :, None] * velocity_delta[:, None, :]
        + (
            FROZEN_FLOW_CONFIG.voxel_size_m**2
            / (6.0 * history_span_s**2)
        )
        * eye[None, :, :]
    ) / np.maximum(support[:, None, None], EPSILON)


def _numpy_component_mixture(
    velocity: np.ndarray,
    covariance: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(velocity, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    component = np.asarray(component)
    output_velocity = np.zeros_like(velocity)
    mover_probability = np.zeros(len(velocity), dtype=np.float64)
    for value in np.unique(component):
        rows = np.nonzero(component == value)[0]
        precision = np.linalg.inv(covariance[rows])
        summed_precision = precision.sum(axis=0)
        weighted_velocity = np.einsum("nij,nj->ni", precision, velocity[rows]).sum(axis=0)
        mean = np.linalg.solve(summed_precision, weighted_velocity)
        static_chi2 = float(
            np.einsum("ni,nij,nj->", velocity[rows], precision, velocity[rows])
        )
        residual = velocity[rows] - mean
        mover_chi2 = float(np.einsum("ni,nij,nj->", residual, precision, residual))
        mover_bic = mover_chi2 + MODEL_DIMENSION * math.log(max(len(rows), 2))
        logit = float(np.clip(0.5 * (static_chi2 - mover_bic), -40.0, 40.0))
        probability = 1.0 / (1.0 + math.exp(-logit))
        output_velocity[rows] = mean
        mover_probability[rows] = probability
    return output_velocity, mover_probability


def _torch_component_mixture(
    velocity: np.ndarray,
    covariance: np.ndarray,
    component: np.ndarray,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    v = torch.as_tensor(velocity, dtype=torch.float64, device="cuda")
    cov = torch.as_tensor(covariance, dtype=torch.float64, device="cuda")
    comp = torch.as_tensor(component.astype(np.int64), dtype=torch.int64, device="cuda")
    output_velocity = torch.zeros_like(v)
    mover_probability = torch.zeros((len(v),), dtype=torch.float64, device="cuda")
    for value in torch.unique(comp):
        rows = torch.nonzero(comp == value, as_tuple=False).flatten()
        precision = torch.linalg.inv(cov[rows])
        summed_precision = precision.sum(dim=0)
        weighted_velocity = torch.einsum("nij,nj->ni", precision, v[rows]).sum(dim=0)
        mean = torch.linalg.solve(summed_precision, weighted_velocity)
        static_chi2 = torch.einsum("ni,nij,nj->", v[rows], precision, v[rows])
        residual = v[rows] - mean
        mover_chi2 = torch.einsum("ni,nij,nj->", residual, precision, residual)
        mover_bic = mover_chi2 + MODEL_DIMENSION * math.log(max(len(rows), 2))
        probability = torch.sigmoid(0.5 * (static_chi2 - mover_bic))
        output_velocity[rows] = mean
        mover_probability[rows] = probability
    output = (output_velocity, mover_probability)
    if numpy_output:
        return output_velocity.cpu().numpy(), mover_probability.cpu().numpy()
    return output


def _select_component_backend(
    velocity: np.ndarray,
    covariance: np.ndarray,
    component: np.ndarray,
    receipt: Path,
) -> dict[str, Any]:
    velocity = velocity[:PROBE_CELLS]
    covariance = covariance[:PROBE_CELLS]
    component = component[:PROBE_CELLS]
    require(bool(len(velocity)), "c15_component_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_component_mixture(velocity, covariance, component)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_component_mixture(
            velocity, covariance, component, numpy_output=False
        )
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_component_mixture(
                velocity, covariance, component, numpy_output=True
            )
            require(
                np.allclose(cache["cpu"][0], gpu[0], atol=1e-8, rtol=1e-8)
                and np.allclose(cache["cpu"][1], gpu[1], atol=1e-8, rtol=1e-8),
                "c15_cpu_gpu_component_mixture_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-component-bic-mixture",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-component-bic-mixture",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _component_mixture(
    velocity: np.ndarray,
    covariance: np.ndarray,
    component: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "torch-cuda-component-bic-mixture":
        return _torch_component_mixture(
            velocity, covariance, component, numpy_output=True
        )
    return _numpy_component_mixture(velocity, covariance, component)


def _component_inputs(
    data: Any,
    index: int,
    recovered: tuple[np.ndarray, ...],
    component_all: np.ndarray,
) -> tuple[np.ndarray, ...]:
    start, stop = int(data.ct_offsets[index]), int(data.ct_offsets[index + 1])
    world_position, world_velocity, confidence = _current_world_cells(data, index)
    absolute_velocity = _rotate_world_to_local(
        world_velocity, float(data.ego_yaw_rad[index])
    )
    wearer_velocity = _rotate_world_to_local(
        _wearer_velocity(data, index)[None, :], float(data.ego_yaw_rad[index])
    )[0]
    span = float(recovered[5][0])
    covariance = _velocity_covariance(recovered[3], recovered[4], span)
    component = np.asarray(component_all[start:stop])
    require(
        len(recovered[0])
        == len(absolute_velocity)
        == len(covariance)
        == len(component)
        == len(confidence),
        f"c15_component_cardinality:{data.sequence}:{int(data.frames[index])}",
    )
    return (
        recovered[0],
        absolute_velocity,
        covariance,
        component,
        confidence,
        wearer_velocity,
    )


def _component_probe(
    rows: Sequence[Any], matching_backend: str
) -> tuple[np.ndarray, ...]:
    for data in rows:
        values = _r7_values(data)
        with np.load(data.ct_path, allow_pickle=False) as ct:
            component = np.asarray(ct["component_id"]).copy()
        for index in range(len(data.frames)):
            recovered = _recover_frame(data, values, index, matching_backend)
            if recovered is not None:
                return _component_inputs(data, index, recovered, component)
    raise RuntimeError("c15_component_probe_missing")


def _mixture_frame(
    inputs: tuple[np.ndarray, ...] | None,
    component_backend: str,
    route_backend: str,
) -> tuple[float, float, int, float]:
    if inputs is None:
        return 0.0, float("nan"), 0, float("nan")
    position, velocity, covariance, component, confidence, wearer_velocity = inputs
    mover_velocity, mover_probability = _component_mixture(
        velocity, covariance, component, component_backend
    )
    static_relative = np.broadcast_to(-wearer_velocity, velocity.shape)
    mover_relative = mover_velocity - wearer_velocity
    hypotheses = np.stack([static_relative, mover_relative], axis=1)
    entries = _entries(
        np.repeat(position, 2, axis=0),
        hypotheses.reshape(-1, 2),
        route_backend,
    ).reshape(len(position), 2)
    weights = np.column_stack([1.0 - mover_probability, mover_probability])
    discounted = np.where(
        np.isfinite(entries),
        np.exp(-np.nan_to_num(entries, nan=0.0) / HORIZON_S),
        0.0,
    )
    mass = confidence * np.sum(weights * discounted, axis=1)
    cells = np.floor(position / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    collapsed: dict[tuple[int, int], float] = {}
    for cell, value in zip(cells, mass):
        key = (int(cell[0]), int(cell[1]))
        collapsed[key] = max(collapsed.get(key, 0.0), float(value))
    finite = entries[np.isfinite(entries)]
    score = math.log1p(
        FROZEN_FLOW_CONFIG.voxel_size_m**2 * sum(collapsed.values())
    )
    return (
        score,
        float(finite.min()) if len(finite) else float("nan"),
        len(np.unique(component)),
        float(mover_probability.mean()),
    )


def extract_c15(
    data: Any,
    matching_backend: str,
    component_backend: str,
    route_backend: str,
) -> C15Evidence:
    base = extract_sequence(data, route_backend)
    values = _r7_values(data)
    with np.load(data.ct_path, allow_pickle=False) as ct:
        component_all = np.asarray(ct["component_id"]).copy()
    scores: list[float] = []
    minimum: list[float] = []
    component_count = 0
    mover_probability: list[float] = []
    for index in range(len(data.frames)):
        recovered = _recover_frame(data, values, index, matching_backend)
        inputs = (
            None
            if recovered is None
            else _component_inputs(data, index, recovered, component_all)
        )
        score, entry, components, probability = _mixture_frame(
            inputs, component_backend, route_backend
        )
        scores.append(score)
        minimum.append(entry)
        component_count += components
        if np.isfinite(probability):
            mover_probability.append(probability)
    return C15Evidence(
        base=base,
        mixture_score=np.asarray(scores, dtype=np.float64),
        mixture_min_entry_s=np.asarray(minimum, dtype=np.float64),
        admitted_cells=int(len(component_all)),
        components=component_count,
        mean_mover_probability=(
            float(np.mean(mover_probability)) if mover_probability else float("nan")
        ),
    )


def predict_c15(
    evidence: C15Evidence,
    onset_model: Sequence[float],
    c11_maintenance: Sequence[float],
    probability_backend: str,
) -> dict[str, Any]:
    onset_probability = _probability(
        evidence.mixture_score, onset_model, probability_backend
    )
    maintenance_probability = _probability(
        evidence.base.reachable_score, c11_maintenance, probability_backend
    )
    config = DTRConfig()
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard = HORIZON_S * FROZEN_R2_CONFIG.imminent_horizon_fraction
    output: dict[str, Any] = {
        "raw_alert_frames": [],
        "active_alert_frames": [],
        "urgent_frames": [],
        "minimum_entry_s_by_frame": {},
        "risky_cells_by_frame": {},
        "route_region_probability_by_frame": {},
    }
    onset_frames = maintenance_frames = imminent_origin_frames = 0
    for index, frame_value in enumerate(evidence.base.frames):
        frame = int(frame_value)
        imminent = bool(
            np.isfinite(evidence.base.current_min_entry_s[index])
            and evidence.base.current_min_entry_s[index] <= guard + EPSILON
        )
        probabilistic = bool(onset_probability[index] >= PROBABILITY_THRESHOLD)
        onset = probabilistic or imminent
        maintain = bool(
            lifecycle.active
            and maintenance_probability[index] >= PROBABILITY_THRESHOLD
        )
        raw = onset or maintain
        minimum = (
            float(evidence.mixture_min_entry_s[index])
            if onset
            else float(evidence.base.reachable_min_entry_s[index])
            if maintain
            else float("nan")
        )
        urgent = bool(raw and np.isfinite(minimum) and minimum <= guard + EPSILON)
        signal = lifecycle.update(
            float(evidence.base.times_s[index] - evidence.base.times_s[0]),
            raw,
            urgent=urgent,
        )
        output["route_region_probability_by_frame"][str(frame)] = float(
            max(
                onset_probability[index],
                maintenance_probability[index] if lifecycle.active else 0.0,
            )
        )
        onset_frames += int(probabilistic)
        imminent_origin_frames += int(imminent and not probabilistic)
        maintenance_frames += int(maintain and not onset)
        if raw:
            output["raw_alert_frames"].append(frame)
            output["minimum_entry_s_by_frame"][str(frame)] = minimum
            output["risky_cells_by_frame"][str(frame)] = 1
        if urgent:
            output["urgent_frames"].append(frame)
        if signal in ACTIVE_SIGNALS:
            output["active_alert_frames"].append(frame)
    output["diagnostics"] = {
        "frames": len(evidence.base.frames),
        "admitted_cells": evidence.admitted_cells,
        "component_observations": evidence.components,
        "mean_mover_probability": evidence.mean_mover_probability,
        "mixture_evidence_frames": int(np.count_nonzero(evidence.mixture_score)),
        "mixture_onset_frames": onset_frames,
        "imminent_geometry_origin_frames": imminent_origin_frames,
        "maintenance_only_frames": maintenance_frames,
        "active_alert_frames": len(output["active_alert_frames"]),
    }
    return output


def _score_group(
    rows: Sequence[Any],
    evidence: Mapping[str, C15Evidence],
    timelines: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    c11_onset: Sequence[float],
    c11_maintenance: Sequence[float],
    c15_onset: Sequence[float],
    probability_backend: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    baseline_rows, candidate_rows, details = [], [], []
    for data in rows:
        item = evidence[data.sequence]
        baseline = predict_c11(
            item.base,
            c11_onset,
            c11_maintenance,
            probability_backend=probability_backend,
        )
        candidate = predict_c15(
            item, c15_onset, c11_maintenance, probability_backend
        )
        timeline = timelines[data.sequence]
        baseline_score = score_sequence(
            sequence=data.sequence,
            timeline=timeline,
            prediction_frames=_prediction_frames(item.base.frames.tolist(), baseline),
        )
        candidate_score = score_sequence(
            sequence=data.sequence,
            timeline=timeline,
            prediction_frames=_prediction_frames(item.base.frames.tolist(), candidate),
        )
        baseline_rows.append(baseline_score)
        candidate_rows.append(candidate_score)
        details.append(
            {
                "sequence": data.sequence,
                "scores": {C11_ARM: baseline_score, ARM: candidate_score},
                "diagnostics": candidate["diagnostics"],
            }
        )
    return aggregate_scores(baseline_rows), aggregate_scores(candidate_rows), details


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamps = args.timestamps.resolve(strict=True)
    labels = args.labels.resolve(strict=True)
    calibrator_path = args.c11_calibrator.resolve(strict=True)
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    c11_onset = [
        float(calibrator["model"]["onset_platt_slope"]),
        float(calibrator["model"]["onset_platt_intercept"]),
    ]
    c11_maintenance = [
        float(calibrator["model"]["maintenance_platt_slope"]),
        float(calibrator["model"]["maintenance_platt_intercept"]),
    ]
    train_rows, train_timestamps = _load_group(
        args.training_ledger_roots, timestamps
    )
    validation_rows, validation_timestamps = _load_group(
        args.validation_ledger_roots, timestamps
    )
    all_rows = [*train_rows, *validation_rows]

    projected, current = _matching_probe(all_rows)
    matching_selection = _select_matching_backend(
        projected, current, args.matching_backend_receipt.resolve()
    )
    matching_backend = str(matching_selection["selected_backend"])
    probe = _component_probe(all_rows, matching_backend)
    component_selection = _select_component_backend(
        probe[1], probe[2], probe[3], args.component_backend_receipt.resolve()
    )
    component_backend = str(component_selection["selected_backend"])
    probe_velocity, probe_probability = _component_mixture(
        probe[1], probe[2], probe[3], component_backend
    )
    static_relative = np.broadcast_to(-probe[5], probe_velocity.shape)
    mover_relative = probe_velocity - probe[5]
    route_position = np.repeat(probe[0], 2, axis=0)
    route_velocity = np.stack([static_relative, mover_relative], axis=1).reshape(-1, 2)
    route_selection = _select_backend(
        route_position,
        route_velocity,
        args.route_backend_receipt.resolve(),
    )
    route_backend = str(route_selection["selected_backend"])
    evidence = {
        data.sequence: extract_c15(
            data, matching_backend, component_backend, route_backend
        )
        for data in all_rows
    }

    train_timelines = _timelines(train_rows, train_timestamps, labels)
    validation_timelines = _timelines(
        validation_rows, validation_timestamps, labels
    )
    train_x, train_y, target_rows = [], [], {}
    for data in train_rows:
        eligible, target = _future_first_passage_target(
            train_timelines[data.sequence]
        )
        train_x.append(evidence[data.sequence].mixture_score[eligible])
        train_y.append(target[eligible])
        target_rows[data.sequence] = {
            "eligible_frames": int(np.count_nonzero(eligible)),
            "positive_frames": int(target[eligible].sum()),
        }
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    require(bool(len(x)) and 0 < float(y.mean()) < 1, "c15_target_degenerate")
    fit_selection = _select_fit_backend(
        x, y, args.fit_backend_receipt.resolve()
    )
    fit_backend = str(fit_selection["selected_backend"])
    development_model = fit_platt(x, y, fit_backend)
    probability_selection = _select_probability_backend(
        np.concatenate([item.base.current_score for item in evidence.values()]),
        np.concatenate([item.mixture_score for item in evidence.values()]),
        c11_onset,
        development_model.tolist(),
        args.probability_backend_receipt.resolve(),
    )
    probability_backend = str(probability_selection["selected_backend"])
    baseline, candidate, details = _score_group(
        validation_rows,
        evidence,
        validation_timelines,
        c11_onset=c11_onset,
        c11_maintenance=c11_maintenance,
        c15_onset=development_model,
        probability_backend=probability_backend,
    )
    lead_gain = float(candidate["median_first_alert_lead_s"]) - float(
        baseline["median_first_alert_lead_s"]
    )
    gate = {
        "recall_not_lower": candidate["bounded_contact_events_recalled"]
        >= baseline["bounded_contact_events_recalled"],
        "false_segments_not_higher": candidate["false_alert_segments"]
        <= baseline["false_alert_segments"],
        "median_lead_gain_at_least_s": MINIMUM_LEAD_GAIN_S,
        "observed_median_lead_gain_s": lead_gain,
    }
    passed = bool(
        gate["recall_not_lower"]
        and gate["false_segments_not_higher"]
        and lead_gain >= MINIMUM_LEAD_GAIN_S - EPSILON
    )
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_C15_COMPONENT_VELOCITY_MIXTURE_DEVELOPMENT_GATE_MET"
            if passed
            else "DTR_C15_COMPONENT_VELOCITY_MIXTURE_DEVELOPMENT_GATE_NOT_MET"
        ),
        "question": (
            "Can a detector-independent static/shared-rigid velocity mixture "
            "separate pseudo-motion from real future route conflict?"
        ),
        "fixed_gate": gate,
        "development_model": {
            "slope": float(development_model[0]),
            "intercept": float(development_model[1]),
            "decision_probability": PROBABILITY_THRESHOLD,
        },
        "development_validation": {
            C11_ARM: baseline,
            ARM: candidate,
            "per_sequence": details,
        },
        "feature": {
            "name": "component static/shared-rigid velocity mixture",
            "component_source": "R7 frame-local connected occupancy component; no detector identity",
            "observation_covariance": "(0.5*dv*dv^T + voxel_size^2/(6*history_span^2)*I)/support",
            "hypotheses": ["absolute world velocity = 0", "one precision-weighted velocity per component"],
            "model_probability": "equal-prior Gaussian BIC posterior with k_static=0 and k_mover=2",
            "collision_mass": "confidence * sum_hypothesis probability * finite_entry * exp(-entry/3s)",
            "decision": "C15 onset replaces C11 probabilistic onset; imminent geometry and C11 maintenance unchanged",
        },
        "backends": {
            "matching": matching_selection,
            "component": component_selection,
            "route": route_selection,
            "fit": fit_selection,
            "probability": probability_selection,
        },
        "sources": {
            "c11_calibrator": str(calibrator_path),
            "c11_calibrator_sha256": sha256_file(calibrator_path),
            "timestamps": str(timestamps),
            "timestamps_sha256": sha256_file(timestamps),
            "labels": str(labels),
            "labels_sha256": sha256_file(labels),
            "training_ledger_roots": [
                str(path.resolve()) for path in args.training_ledger_roots
            ],
            "validation_ledger_roots": [
                str(path.resolve()) for path in args.validation_ledger_roots
            ],
        },
        "training_target": {
            "name": "native CONTACT: realized path intersection within frozen future horizon",
            "negative_censoring": "UNKNOWN excluded; CLEAR and PROXIMITY negatives",
            "by_sequence": target_rows,
        },
        "external_basis": [
            "Nuss et al.: dynamic occupancy retains occupancy/existence and velocity distributions",
            "Wang et al. WSAFlowNet: local shared aggregation implicitly enforces rigid scene flow",
            "Chi et al. PUA-MOS: uncertainty-weighted aggregation across same-motion points",
        ],
        "claim_limits": [
            "The four C11 confirmation sequences are consumed Development validation for C15.",
            "Component IDs are frame-local occupancy connectivity, not stable object identity.",
            "Future CONTACT truth enters fitting/scoring only and never inference.",
            "No algorithm-fresh sequence is opened unless this fixed gate passes.",
        ],
    }
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C15_COMPONENT_VELOCITY_MIXTURE_MODEL_FROZEN",
                "model": result["development_model"],
                "c11_model": calibrator["model"],
                "feature": result["feature"],
                "development_validation": {
                    C11_ARM: baseline,
                    ARM: candidate,
                    "fixed_gate": gate,
                },
                "sources": result["sources"],
                "claim_limits": result["claim_limits"],
            },
        )
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = REPO / "artifacts.local" / "evidence"
    output_root = evidence / "dtr-c15" / "component-velocity-mixture"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-ledger-roots",
        type=Path,
        nargs="+",
        default=[
            evidence / "dtr-c2" / "fresh-global-obb-replay" / "ledgers",
            evidence / "dtr-c10" / "fresh-confirmation" / "ledgers",
        ],
    )
    parser.add_argument(
        "--validation-ledger-roots",
        type=Path,
        nargs="+",
        default=[evidence / "dtr-c11" / "fresh-confirmation" / "ledgers"],
    )
    parser.add_argument(
        "--c11-calibrator",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "dtr_c11_route_region_calibrator.json"
        ),
    )
    parser.add_argument(
        "--timestamps", type=Path, default=dataset / "train_timestamps.zip"
    )
    parser.add_argument(
        "--labels", type=Path, default=dataset / "train_labels.zip"
    )
    parser.add_argument(
        "--matching-backend-receipt",
        type=Path,
        default=output_root / "backend-matching.json",
    )
    parser.add_argument(
        "--component-backend-receipt",
        type=Path,
        default=output_root / "backend-component.json",
    )
    parser.add_argument(
        "--route-backend-receipt",
        type=Path,
        default=output_root / "backend-route.json",
    )
    parser.add_argument(
        "--fit-backend-receipt",
        type=Path,
        default=output_root / "backend-fit.json",
    )
    parser.add_argument(
        "--probability-backend-receipt",
        type=Path,
        default=output_root / "backend-probability.json",
    )
    parser.add_argument("--output", type=Path, default=output_root / "result.json")
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "dtr_c15_component_velocity_mixture_model.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["fixed_gate"],
                "development": result["development_validation"],
            }
        )
    )


if __name__ == "__main__":
    main()
