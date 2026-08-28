"""Develop joint early-motion and temporal-confidence route risk over C11.

C16's signed current/history modes improved lead, while C18's three-frame
confidence reduced false alerts.  C19 preserves those as two distinct causal
features and learns one training-only logistic calibration.  It does not alter
the route geometry, probability threshold, maintenance model, or lifecycle.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_c11_route_region_probability as c11
import dtr_c14_stochastic_route_conflict as c14
import dtr_c16_empirical_velocity_modes as c16
import dtr_c18_three_frame_motion_confidence as c18
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json
from dtr_c2_fresh_global_obb_replay import aggregate_scores, score_sequence
from dtr_c4_detector_independent_global_risk import _prediction_frames
from dtr_c6_world_route_occupancy_belief import EPSILON, _select_backend
from dtr_c12_c13_route_time_research import (
    MINIMUM_LEAD_GAIN_S,
    _future_first_passage_target,
    _load_group,
    _timelines,
)
from dtr_r0 import DTRConfig
from dtr_r1 import RiskEventLifecycle
from dtr_r2 import FROZEN_R2_CONFIG
from dtr_r5_dropout_canary import ACTIVE_SIGNALS
from tools.research_backend import (
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


SCHEMA = "blindassist-dtr-c19-joint-motion-confidence-v1"
ARM = "M1_JMC_GLOBAL"
C11_ARM = "M1_RROQ_GLOBAL"
STATUS_MET = "DTR_C19_JOINT_MOTION_CONFIDENCE_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C19_JOINT_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET"


@dataclass(frozen=True)
class DualEvidence:
    base: c11.SequenceEvidence
    early_score: np.ndarray
    consistent_score: np.ndarray
    empirical_min_entry_s: np.ndarray
    admitted_cells: int
    three_frame_supported_cells: int


def extract_c19(
    data: Any,
    matching_backend: str,
    mode_backend: str,
    route_backend: str,
) -> DualEvidence:
    base = c16.extract_sequence(data, route_backend)
    values = c14._r7_values(data)
    early_scores: list[float] = []
    consistent_scores: list[float] = []
    minimum: list[float] = []
    supported_cells = 0
    for index in range(len(data.frames)):
        recovered = c14._recover_frame(data, values, index, matching_backend)
        inputs = None if recovered is None else c16._mode_inputs(data, index, recovered)
        early_score, entry = c16._frame_score(inputs, mode_backend, route_backend)
        if inputs is None:
            consistent_score = 0.0
        else:
            confidence, supported = c18._three_frame_confidence(
                data, values, index, matching_backend
            )
            require(
                len(confidence) == len(inputs[3]),
                f"c19_confidence_cardinality:{data.sequence}:{int(data.frames[index])}",
            )
            consistent_inputs = (*inputs[:3], confidence, inputs[4])
            consistent_score, _consistent_entry = c16._frame_score(
                consistent_inputs, mode_backend, route_backend
            )
            supported_cells += supported
        early_scores.append(early_score)
        consistent_scores.append(consistent_score)
        minimum.append(entry)
    return DualEvidence(
        base=base,
        early_score=np.asarray(early_scores, dtype=np.float64),
        consistent_score=np.asarray(consistent_scores, dtype=np.float64),
        empirical_min_entry_s=np.asarray(minimum, dtype=np.float64),
        admitted_cells=int(data.ct_offsets[-1]),
        three_frame_supported_cells=supported_cells,
    )


def _fit_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == 2, "c19_fit_shape")
    params = np.asarray(
        [
            0.0,
            0.0,
            math.log((float(y.mean()) + 1e-6) / (1.0 - float(y.mean()) + 1e-6)),
        ],
        dtype=np.float64,
    )
    for _ in range(c11.PLATT_STEPS):
        logits = np.clip(x @ params[:2] + params[2], -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = probability - y
        weight = probability * (1.0 - probability)
        gradient = np.concatenate(
            [x.T @ residual + c11.PLATT_L2 * params[:2], [residual.sum()]]
        )
        hessian = np.empty((3, 3), dtype=np.float64)
        hessian[:2, :2] = x.T @ (weight[:, None] * x)
        hessian[:2, :2] += c11.PLATT_L2 * np.eye(2)
        hessian[:2, 2] = x.T @ weight
        hessian[2, :2] = hessian[:2, 2]
        hessian[2, 2] = weight.sum() + 1e-12
        params -= np.linalg.solve(hessian, gradient)
    return params


def _fit_torch(x: np.ndarray, y: np.ndarray, *, numpy_output: bool) -> Any:
    import torch

    tx = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    ty = torch.as_tensor(y, dtype=torch.float64, device="cuda")
    prevalence = torch.mean(ty)
    params = torch.stack(
        [
            torch.zeros((), dtype=torch.float64, device="cuda"),
            torch.zeros((), dtype=torch.float64, device="cuda"),
            torch.log((prevalence + 1e-6) / (1.0 - prevalence + 1e-6)),
        ]
    )
    identity = torch.eye(2, dtype=torch.float64, device="cuda")
    for _ in range(c11.PLATT_STEPS):
        logits = torch.clamp(tx @ params[:2] + params[2], -40.0, 40.0)
        probability = torch.sigmoid(logits)
        residual = probability - ty
        weight = probability * (1.0 - probability)
        gradient = torch.cat(
            [tx.T @ residual + c11.PLATT_L2 * params[:2], residual.sum()[None]]
        )
        hessian = torch.empty((3, 3), dtype=torch.float64, device="cuda")
        hessian[:2, :2] = tx.T @ (weight[:, None] * tx) + c11.PLATT_L2 * identity
        hessian[:2, 2] = tx.T @ weight
        hessian[2, :2] = hessian[:2, 2]
        hessian[2, 2] = weight.sum() + 1e-12
        params = params - torch.linalg.solve(hessian, gradient)
    return params.detach().cpu().numpy() if numpy_output else params


def _select_fit_backend(
    x: np.ndarray, y: np.ndarray, receipt: Path
) -> dict[str, Any]:
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
                raise BackendSelectionError("C19_FIT_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-newton-two-channel-logistic",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-newton-two-channel-logistic",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _fit(x: np.ndarray, y: np.ndarray, backend: str) -> np.ndarray:
    if backend == "torch-cuda-newton-two-channel-logistic":
        return _fit_torch(x, y, numpy_output=True)
    return _fit_numpy(x, y)


def _joint_probability(
    x: np.ndarray,
    params: Sequence[float],
    backend: str,
) -> np.ndarray:
    if backend == "torch-cuda-joint-motion-confidence-inference":
        import torch

        values = torch.as_tensor(x, dtype=torch.float64, device="cuda")
        weights = torch.as_tensor(params[:2], dtype=torch.float64, device="cuda")
        return torch.sigmoid(values @ weights + float(params[2])).cpu().numpy()
    logits = np.clip(x @ np.asarray(params[:2]) + float(params[2]), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _select_probability_backend(
    x: np.ndarray,
    reachable_score: np.ndarray,
    onset: Sequence[float],
    maintenance: Sequence[float],
    receipt: Path,
) -> dict[str, Any]:
    cache: dict[str, np.ndarray] = {}

    def cpu_probe() -> np.ndarray:
        cache["cpu"] = np.concatenate(
            [
                _joint_probability(x, onset, "numpy-joint-motion-confidence-inference"),
                c11._probability(
                    reachable_score,
                    maintenance,
                    "numpy-platt-inference",
                ),
            ]
        )
        return cache["cpu"]

    def gpu_probe() -> Any:
        import torch

        values = torch.as_tensor(x, dtype=torch.float64, device="cuda")
        reachable = torch.as_tensor(
            reachable_score, dtype=torch.float64, device="cuda"
        )
        weights = torch.as_tensor(onset[:2], dtype=torch.float64, device="cuda")
        return torch.cat(
            [
                torch.sigmoid(values @ weights + float(onset[2])),
                torch.sigmoid(
                    float(maintenance[0]) * reachable + float(maintenance[1])
                ),
            ]
        )

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = output.detach().cpu().numpy()
            if not np.allclose(cache["cpu"], gpu, atol=1e-12, rtol=1e-12):
                raise BackendSelectionError("C19_PROBABILITY_CPU_GPU_MISMATCH")
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-joint-motion-confidence-inference",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-joint-motion-confidence-inference",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _features(
    evidence: DualEvidence, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    raw = np.column_stack([evidence.early_score, evidence.consistent_score])
    return (raw - mean[None, :]) / scale[None, :]


def predict_c19(
    evidence: DualEvidence,
    onset_model: Sequence[float],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    c11_maintenance: Sequence[float],
    probability_backend: str,
) -> dict[str, Any]:
    onset_probability = _joint_probability(
        _features(evidence, feature_mean, feature_scale),
        onset_model,
        probability_backend,
    )
    maintenance_probability = c11._probability(
        evidence.base.reachable_score,
        c11_maintenance,
        "torch-cuda-platt-inference"
        if probability_backend == "torch-cuda-joint-motion-confidence-inference"
        else "numpy-platt-inference",
    )
    config = DTRConfig()
    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    guard = c16.HORIZON_S * FROZEN_R2_CONFIG.imminent_horizon_fraction
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
        probabilistic = bool(onset_probability[index] >= c11.PROBABILITY_THRESHOLD)
        onset = probabilistic or imminent
        maintain = bool(
            lifecycle.active
            and maintenance_probability[index] >= c11.PROBABILITY_THRESHOLD
        )
        raw = onset or maintain
        minimum = (
            float(evidence.empirical_min_entry_s[index])
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
        "three_frame_supported_cells": evidence.three_frame_supported_cells,
        "early_evidence_frames": int(np.count_nonzero(evidence.early_score)),
        "consistent_evidence_frames": int(
            np.count_nonzero(evidence.consistent_score)
        ),
        "joint_onset_frames": onset_frames,
        "imminent_geometry_origin_frames": imminent_origin_frames,
        "maintenance_only_frames": maintenance_frames,
        "active_alert_frames": len(output["active_alert_frames"]),
    }
    return output


def _score_group(
    rows: Sequence[Any],
    evidence: Mapping[str, DualEvidence],
    timelines: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    c11_onset: Sequence[float],
    c11_maintenance: Sequence[float],
    onset_model: Sequence[float],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    probability_backend: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    baseline_rows, candidate_rows, details = [], [], []
    for data in rows:
        item = evidence[data.sequence]
        baseline = c11.predict(
            item.base,
            c11_onset,
            c11_maintenance,
            probability_backend=(
                "torch-cuda-platt-inference"
                if probability_backend
                == "torch-cuda-joint-motion-confidence-inference"
                else "numpy-platt-inference"
            ),
        )
        candidate = predict_c19(
            item,
            onset_model,
            feature_mean,
            feature_scale,
            c11_maintenance,
            probability_backend,
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


def run(args: Any) -> dict[str, Any]:
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

    projected, current = c14._matching_probe(all_rows)
    matching_selection = c14._select_matching_backend(
        projected, current, args.matching_backend_receipt.resolve()
    )
    matching_backend = str(matching_selection["selected_backend"])
    probe = c16._mode_probe(all_rows, matching_backend)
    mode_selection = c16._select_mode_backend(
        probe[1], probe[2], args.mode_backend_receipt.resolve()
    )
    mode_backend = str(mode_selection["selected_backend"])
    observed_modes = c16._modes(probe[1], probe[2], mode_backend)
    route_selection = _select_backend(
        np.repeat(probe[0], 2, axis=0),
        (observed_modes - probe[4][None, None, :]).reshape(-1, 2),
        args.route_backend_receipt.resolve(),
    )
    route_backend = str(route_selection["selected_backend"])
    c18._CONFIDENCE_BACKEND = None
    c18._CONFIDENCE_SELECTION = None
    c18._CONFIDENCE_RECEIPT = args.confidence_backend_receipt.resolve()
    evidence = {
        data.sequence: extract_c19(
            data, matching_backend, mode_backend, route_backend
        )
        for data in all_rows
    }
    require(c18._CONFIDENCE_SELECTION is not None, "c19_confidence_backend_missing")

    train_timelines = _timelines(train_rows, train_timestamps, labels)
    validation_timelines = _timelines(
        validation_rows, validation_timestamps, labels
    )
    train_x, train_y, target_rows = [], [], {}
    for data in train_rows:
        eligible, target = _future_first_passage_target(
            train_timelines[data.sequence]
        )
        item = evidence[data.sequence]
        train_x.append(
            np.column_stack(
                [item.early_score[eligible], item.consistent_score[eligible]]
            )
        )
        train_y.append(target[eligible])
        target_rows[data.sequence] = {
            "eligible_frames": int(np.count_nonzero(eligible)),
            "positive_frames": int(target[eligible].sum()),
        }
    raw_x, y = np.concatenate(train_x), np.concatenate(train_y)
    require(bool(len(raw_x)) and 0 < float(y.mean()) < 1, "c19_target_degenerate")
    feature_mean = raw_x.mean(axis=0)
    feature_scale = np.maximum(raw_x.std(axis=0), 1e-6)
    x = (raw_x - feature_mean[None, :]) / feature_scale[None, :]
    fit_selection = _select_fit_backend(x, y, args.fit_backend_receipt.resolve())
    fit_backend = str(fit_selection["selected_backend"])
    development_model = _fit(x, y, fit_backend)

    all_features = np.concatenate(
        [_features(item, feature_mean, feature_scale) for item in evidence.values()]
    )
    all_reachable = np.concatenate(
        [item.base.reachable_score for item in evidence.values()]
    )
    probability_selection = _select_probability_backend(
        all_features,
        all_reachable,
        development_model,
        c11_maintenance,
        args.probability_backend_receipt.resolve(),
    )
    probability_backend = str(probability_selection["selected_backend"])
    baseline, candidate, details = _score_group(
        validation_rows,
        evidence,
        validation_timelines,
        c11_onset=c11_onset,
        c11_maintenance=c11_maintenance,
        onset_model=development_model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
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
    train_probability = _joint_probability(x, development_model, probability_backend)
    result = {
        "schema": SCHEMA,
        "status": STATUS_MET if passed else STATUS_NOT_MET,
        "question": (
            "Can a fixed training-only joint calibration preserve C16 early motion "
            "while using C18 temporal confidence to reject pseudo-motion?"
        ),
        "fixed_gate": gate,
        "development_model": {
            "early_motion_weight": float(development_model[0]),
            "three_frame_confidence_weight": float(development_model[1]),
            "intercept": float(development_model[2]),
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "decision_probability": c11.PROBABILITY_THRESHOLD,
        },
        "training_calibration": c11._calibration_metrics(train_probability, y),
        "development_validation": {
            C11_ARM: baseline,
            ARM: candidate,
            "per_sequence": details,
        },
        "feature": {
            "name": "joint early-motion and temporal-confidence calibration",
            "channels": [
                "C16 signed current/history mean route-entry mass",
                "C18 three-frame-confidence-weighted route-entry mass",
            ],
            "fusion": "two-feature L2-regularized logistic model fit on training ledgers only",
            "decision": "C19 onset replaces C11 probabilistic onset; p=0.5, imminent geometry, C11 maintenance, and lifecycle unchanged",
        },
        "backends": {
            "matching": matching_selection,
            "mode": mode_selection,
            "route": route_selection,
            "confidence": c18._CONFIDENCE_SELECTION,
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
            "Rummelhard et al.: separate static/dynamic predictions before parameterized fusion to avoid averaging bias",
            "DS-K3DOM: retain dynamic, static, free, and uncertainty evidence instead of collapsing ambiguous occupancy",
            "M-FUSE and DeltaFlow: preceding-frame and delta cues supply temporal motion information",
        ],
        "claim_limits": [
            "The four C11 confirmation sequences are consumed Development validation for C19.",
            "The joint model has two fixed causal inputs and is not trajectory forecasting.",
            "Future CONTACT truth enters training/scoring only and never inference.",
            "No algorithm-fresh sequence is opened unless this fixed gate passes.",
        ],
    }
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C19_JOINT_MOTION_CONFIDENCE_MODEL_FROZEN",
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


def parse_args() -> Any:
    args = c16.parse_args()
    root = c16.REPO / "artifacts.local" / "evidence" / "dtr-c19" / "joint-motion-confidence"
    args.matching_backend_receipt = root / "backend-matching.json"
    args.mode_backend_receipt = root / "backend-mode.json"
    args.route_backend_receipt = root / "backend-route.json"
    args.confidence_backend_receipt = root / "backend-confidence.json"
    args.fit_backend_receipt = root / "backend-fit.json"
    args.probability_backend_receipt = root / "backend-probability.json"
    args.output = root / "result.json"
    args.frozen_model = Path(__file__).resolve().with_name(
        "dtr_c19_joint_motion_confidence_model.json"
    )
    return args


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
