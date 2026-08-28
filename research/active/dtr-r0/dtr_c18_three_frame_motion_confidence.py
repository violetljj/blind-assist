"""Develop three-frame causal motion confidence over C16 route conflict.

C16 improved lead but admitted two false segments whose motion was not stable
across the preceding causal link.  C18 keeps C16's two signed velocity modes and
mean collision mass, replacing only the point confidence with the minimum of
the two frozen pair confidences and a frozen-scale velocity-delta consistency.
Missing second-level correspondence is UNKNOWN and contributes zero mass.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import dtr_c14_stochastic_route_conflict as c14
import dtr_c16_empirical_velocity_modes as c16
from dtr_c1_global_obb_cohort_admission import require, write_json
from dtr_c15_component_velocity_mixture import C15Evidence
from tools.research_backend import (
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


SCHEMA = "blindassist-dtr-c18-three-frame-motion-confidence-v1"
ARM = "M1_TFMC_GLOBAL"
STATUS_MET = "DTR_C18_THREE_FRAME_MOTION_CONFIDENCE_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C18_THREE_FRAME_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET"

_CONFIDENCE_BACKEND: str | None = None
_CONFIDENCE_SELECTION: dict[str, Any] | None = None
_CONFIDENCE_RECEIPT: Path | None = None


def _pair_state(
    data: Any,
    values: Mapping[str, np.ndarray],
    index: int,
    matching_backend: str,
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray, np.ndarray] | None:
    pair = c14._raw_pair(data, values, index)
    if pair is None:
        return None
    (
        _history,
        start,
        stop,
        previous_start,
        _current_local,
        _current_velocity_local,
        current_world,
        current_velocity_world,
        previous_velocity_world,
        projected,
        _span,
    ) = pair
    nearest, residual_world = c14._match(projected, current_world, matching_backend)
    valid = nearest >= 0
    confidence = np.zeros(stop - start, dtype=np.float64)
    velocity_delta_world = np.full((stop - start, 2), np.nan, dtype=np.float64)
    rows = np.nonzero(valid)[0]
    if len(rows):
        previous = nearest[rows]
        current_support = np.minimum(
            1.0,
            values["source_point_count"][start:stop][rows].astype(np.float64) / 3.0,
        )
        previous_support = np.minimum(
            1.0,
            values["source_point_count"][previous_start + previous].astype(np.float64)
            / 3.0,
        )
        support = np.minimum.reduce(
            [
                current_support,
                previous_support,
                values["flow_support"][start:stop][rows].astype(np.float64),
                values["flow_support"][previous_start + previous].astype(np.float64),
            ]
        )
        velocity_delta_world[rows] = (
            current_velocity_world[rows] - previous_velocity_world[previous]
        )
        position_confidence = np.exp(
            -0.5
            * (np.linalg.norm(residual_world[rows], axis=1) / c14.POSITION_SIGMA_M)
            ** 2
        )
        velocity_confidence = np.exp(
            -0.5
            * (
                np.linalg.norm(velocity_delta_world[rows], axis=1)
                / c14.VELOCITY_SIGMA_MPS
            )
            ** 2
        )
        confidence[rows] = np.minimum.reduce(
            [support, position_confidence, velocity_confidence]
        )
    return pair, nearest, confidence, velocity_delta_world


def _numpy_three_frame_confidence(
    current_pair_confidence: np.ndarray,
    previous_pair_confidence: np.ndarray,
    current_velocity_delta: np.ndarray,
    previous_velocity_delta: np.ndarray,
) -> np.ndarray:
    jerk_confidence = np.exp(
        -0.5
        * (
            np.linalg.norm(current_velocity_delta - previous_velocity_delta, axis=1)
            / c14.VELOCITY_SIGMA_MPS
        )
        ** 2
    )
    return np.minimum.reduce(
        [current_pair_confidence, previous_pair_confidence, jerk_confidence]
    )


def _torch_three_frame_confidence(
    current_pair_confidence: np.ndarray,
    previous_pair_confidence: np.ndarray,
    current_velocity_delta: np.ndarray,
    previous_velocity_delta: np.ndarray,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    q1 = torch.as_tensor(current_pair_confidence, dtype=torch.float64, device="cuda")
    q0 = torch.as_tensor(previous_pair_confidence, dtype=torch.float64, device="cuda")
    dv1 = torch.as_tensor(current_velocity_delta, dtype=torch.float64, device="cuda")
    dv0 = torch.as_tensor(previous_velocity_delta, dtype=torch.float64, device="cuda")
    jerk = torch.exp(
        -0.5 * (torch.linalg.vector_norm(dv1 - dv0, dim=1) / c14.VELOCITY_SIGMA_MPS) ** 2
    )
    output = torch.minimum(torch.minimum(q1, q0), jerk)
    return output.cpu().numpy() if numpy_output else output


def _select_confidence_backend(inputs: tuple[np.ndarray, ...]) -> dict[str, Any]:
    require(_CONFIDENCE_RECEIPT is not None, "c18_confidence_receipt_missing")
    probe = tuple(np.asarray(value[: c16.PROBE_CELLS]) for value in inputs)
    require(bool(len(probe[0])), "c18_confidence_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_three_frame_confidence(*probe)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_three_frame_confidence(
            *probe, numpy_output=False
        )
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_three_frame_confidence(*probe, numpy_output=True)
            require(
                np.allclose(cache["cpu"], gpu, atol=1e-12, rtol=1e-12),
                "c18_cpu_gpu_confidence_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-three-frame-motion-confidence",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-three-frame-motion-confidence",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=_CONFIDENCE_RECEIPT,
        warmups=1,
        repeats=3,
    )


def _three_frame_backend(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    global _CONFIDENCE_BACKEND, _CONFIDENCE_SELECTION
    if _CONFIDENCE_BACKEND is None:
        _CONFIDENCE_SELECTION = _select_confidence_backend(inputs)
        _CONFIDENCE_BACKEND = str(_CONFIDENCE_SELECTION["selected_backend"])
    if _CONFIDENCE_BACKEND == "torch-cuda-three-frame-motion-confidence":
        return _torch_three_frame_confidence(*inputs, numpy_output=True)
    return _numpy_three_frame_confidence(*inputs)


def _three_frame_confidence(
    data: Any,
    values: Mapping[str, np.ndarray],
    index: int,
    matching_backend: str,
) -> tuple[np.ndarray, int]:
    current = _pair_state(data, values, index, matching_backend)
    if current is None:
        return np.empty(0, dtype=np.float64), 0
    current_pair, current_nearest, current_confidence, current_delta = current
    admitted_rows = np.nonzero(current_confidence >= c14.CONFIDENCE_THRESHOLD)[0]
    output = np.zeros(len(admitted_rows), dtype=np.float64)
    if not len(admitted_rows):
        return output, 0
    previous = _pair_state(data, values, int(current_pair[0]), matching_backend)
    if previous is None:
        return output, 0
    _previous_pair, previous_nearest, previous_confidence, previous_delta = previous
    previous_rows = current_nearest[admitted_rows]
    valid = (previous_rows >= 0) & (previous_nearest[previous_rows] >= 0)
    if not np.any(valid):
        return output, 0
    rows = admitted_rows[valid]
    matched_previous = previous_rows[valid]
    inputs = (
        current_confidence[rows],
        previous_confidence[matched_previous],
        current_delta[rows],
        previous_delta[matched_previous],
    )
    output[valid] = _three_frame_backend(inputs)
    return output, int(np.count_nonzero(valid))


def extract_c18(
    data: Any,
    matching_backend: str,
    mode_backend: str,
    route_backend: str,
) -> C15Evidence:
    base = c16.extract_sequence(data, route_backend)
    values = c14._r7_values(data)
    scores, minimum = [], []
    supported_cells = 0
    for index in range(len(data.frames)):
        recovered = c14._recover_frame(data, values, index, matching_backend)
        inputs = None if recovered is None else c16._mode_inputs(data, index, recovered)
        if inputs is not None:
            confidence, supported = _three_frame_confidence(
                data, values, index, matching_backend
            )
            require(
                len(confidence) == len(inputs[3]),
                f"c18_confidence_cardinality:{data.sequence}:{int(data.frames[index])}",
            )
            inputs = (*inputs[:3], confidence, inputs[4])
            supported_cells += supported
        score, entry = c16._frame_score(inputs, mode_backend, route_backend)
        scores.append(score)
        minimum.append(entry)
    return C15Evidence(
        base=base,
        mixture_score=np.asarray(scores, dtype=np.float64),
        mixture_min_entry_s=np.asarray(minimum, dtype=np.float64),
        admitted_cells=int(data.ct_offsets[-1]),
        components=supported_cells,
        mean_mover_probability=float("nan"),
    )


def _arguments() -> Any:
    args = c16.parse_args()
    root = c16.REPO / "artifacts.local" / "evidence" / "dtr-c18" / "three-frame-motion-confidence"
    args.matching_backend_receipt = root / "backend-matching.json"
    args.mode_backend_receipt = root / "backend-mode.json"
    args.route_backend_receipt = root / "backend-route.json"
    args.fit_backend_receipt = root / "backend-fit.json"
    args.probability_backend_receipt = root / "backend-probability.json"
    args.confidence_backend_receipt = root / "backend-confidence.json"
    args.output = root / "result.json"
    args.frozen_model = Path(__file__).resolve().with_name(
        "dtr_c18_three_frame_motion_confidence_model.json"
    )
    return args


def run() -> dict[str, Any]:
    global _CONFIDENCE_RECEIPT
    args = _arguments()
    _CONFIDENCE_RECEIPT = args.confidence_backend_receipt.resolve()
    c16.SCHEMA = SCHEMA
    c16.ARM = ARM
    c16.extract_c16 = extract_c18
    result = c16.run(args)
    passed = bool(
        result["fixed_gate"]["recall_not_lower"]
        and result["fixed_gate"]["false_segments_not_higher"]
        and result["fixed_gate"]["observed_median_lead_gain_s"]
        >= c16.MINIMUM_LEAD_GAIN_S - c16.EPSILON
    )
    result["schema"] = SCHEMA
    result["status"] = STATUS_MET if passed else STATUS_NOT_MET
    result["question"] = (
        "Can frozen-scale three-frame point-motion confidence reject pseudo-motion "
        "without sacrificing C16's early route-conflict signal?"
    )
    result["feature"] = {
        "name": "three-frame causal motion confidence",
        "pair_confidence": "frozen M1 support, position residual, and velocity residual confidence",
        "chain_confidence": "min(current pair, previous pair, frozen-scale velocity-delta consistency)",
        "missing_chain": "UNKNOWN contributes zero onset mass",
        "collision_mass": "q3 * mean_mode[finite_entry * exp(-entry/3s)]",
        "decision": "C18 onset replaces C11 probabilistic onset; C16 velocity modes, imminent geometry, C11 maintenance, and lifecycle unchanged",
    }
    result["backends"]["confidence"] = _CONFIDENCE_SELECTION
    for row in result["development_validation"]["per_sequence"]:
        diagnostics = row["diagnostics"]
        diagnostics["three_frame_supported_cells"] = diagnostics.pop(
            "component_observations"
        )
    result["external_basis"] = [
        "RigidFlow++: spatial and temporal consistency define scene-flow confidence",
        "M-FUSE: preceding-frame motion adds useful temporal information to scene flow",
        "DeltaFlow: compact multi-frame delta cues improve motion consistency",
        "C18 uses a causal three-frame chain and no learned future trajectory",
    ]
    result["claim_limits"] = [
        "The four C11 confirmation sequences are consumed Development validation for C18.",
        "A missing second causal match is UNKNOWN for onset and not evidence of static occupancy.",
        "Future CONTACT truth enters fitting/scoring only and never inference.",
        "No algorithm-fresh sequence is opened unless this fixed gate passes.",
    ]
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C18_THREE_FRAME_MOTION_CONFIDENCE_MODEL_FROZEN",
                "model": result["development_model"],
                "feature": result["feature"],
                "development_validation": result["development_validation"],
                "sources": result["sources"],
                "claim_limits": result["claim_limits"],
            },
        )
    return result


def main() -> None:
    result = run()
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
