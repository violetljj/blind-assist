"""Develop point-wise local motion voting over C16 route conflict.

C19 showed that downstream linear fusion cannot recover information lost by
collapsing early motion and delayed three-frame confidence.  C20 instead acts
at the point-motion source: neighboring admitted cells cast continuous votes
for a shared velocity delta within the current causal pair.  The vote is a
parameter-free Gaussian agreement using only frozen M1 spatial/velocity scales;
there is no component threshold, duration gate, or route change.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import dtr_c14_stochastic_route_conflict as c14
import dtr_c16_empirical_velocity_modes as c16
import dtr_c18_three_frame_motion_confidence as c18
from dtr_c1_global_obb_cohort_admission import require, write_json
from dtr_c15_component_velocity_mixture import C15Evidence
from tools.research_backend import (
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


SCHEMA = "blindassist-dtr-c20-local-motion-voting-v1"
ARM = "M1_LMV_GLOBAL"
STATUS_MET = "DTR_C20_LOCAL_MOTION_VOTING_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C20_LOCAL_MOTION_VOTING_DEVELOPMENT_GATE_NOT_MET"

_VOTE_BACKEND: str | None = None
_VOTE_SELECTION: dict[str, Any] | None = None
_VOTE_RECEIPT: Path | None = None


def _numpy_vote(
    position: np.ndarray,
    velocity_delta: np.ndarray,
    pair_confidence: np.ndarray,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    velocity_delta = np.asarray(velocity_delta, dtype=np.float64)
    pair_confidence = np.asarray(pair_confidence, dtype=np.float64)
    if len(position) <= 1:
        return np.zeros(len(position), dtype=np.float64)
    dp = position[:, None, :] - position[None, :, :]
    dv = velocity_delta[:, None, :] - velocity_delta[None, :, :]
    spatial = np.exp(
        -0.5
        * np.sum(dp * dp, axis=2)
        / c16.FROZEN_FLOW_CONFIG.voxel_size_m**2
    )
    motion = np.exp(
        -0.5
        * np.sum(dv * dv, axis=2)
        / c14.VELOCITY_SIGMA_MPS**2
    )
    np.fill_diagonal(spatial, 0.0)
    denominator = spatial.sum(axis=1)
    agreement = np.divide(
        (spatial * motion).sum(axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(np.float64).tiny,
    )
    return pair_confidence * agreement


def _torch_vote(
    position: np.ndarray,
    velocity_delta: np.ndarray,
    pair_confidence: np.ndarray,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    p = torch.as_tensor(position, dtype=torch.float64, device="cuda")
    dv = torch.as_tensor(velocity_delta, dtype=torch.float64, device="cuda")
    confidence = torch.as_tensor(
        pair_confidence, dtype=torch.float64, device="cuda"
    )
    if len(position) <= 1:
        output = torch.zeros_like(confidence)
    else:
        dp_distance = torch.sum((p[:, None, :] - p[None, :, :]) ** 2, dim=2)
        dv_distance = torch.sum((dv[:, None, :] - dv[None, :, :]) ** 2, dim=2)
        spatial = torch.exp(
            -0.5
            * dp_distance
            / c16.FROZEN_FLOW_CONFIG.voxel_size_m**2
        )
        motion = torch.exp(-0.5 * dv_distance / c14.VELOCITY_SIGMA_MPS**2)
        spatial.fill_diagonal_(0.0)
        denominator = torch.sum(spatial, dim=1)
        agreement = torch.where(
            denominator > torch.finfo(torch.float64).tiny,
            torch.sum(spatial * motion, dim=1) / denominator,
            torch.zeros_like(denominator),
        )
        output = confidence * agreement
    return output.cpu().numpy() if numpy_output else output


def _select_vote_backend(inputs: tuple[np.ndarray, ...]) -> dict[str, Any]:
    require(_VOTE_RECEIPT is not None, "c20_vote_receipt_missing")
    probe = tuple(np.asarray(value[: c16.PROBE_CELLS]) for value in inputs)
    require(bool(len(probe[0])), "c20_vote_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_vote(*probe)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_vote(*probe, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_vote(*probe, numpy_output=True)
            require(
                np.allclose(cache["cpu"], gpu, atol=1e-12, rtol=1e-12),
                "c20_cpu_gpu_vote_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-local-motion-voting",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-local-motion-voting",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=_VOTE_RECEIPT,
        warmups=1,
        repeats=3,
    )


def _vote(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    global _VOTE_BACKEND, _VOTE_SELECTION
    if _VOTE_BACKEND is None:
        _VOTE_SELECTION = _select_vote_backend(inputs)
        _VOTE_BACKEND = str(_VOTE_SELECTION["selected_backend"])
    if _VOTE_BACKEND == "torch-cuda-local-motion-voting":
        return _torch_vote(*inputs, numpy_output=True)
    return _numpy_vote(*inputs)


def _local_confidence(
    data: Any,
    values: Mapping[str, np.ndarray],
    index: int,
    matching_backend: str,
) -> tuple[np.ndarray, int]:
    state = c18._pair_state(data, values, index, matching_backend)
    if state is None:
        return np.empty(0, dtype=np.float64), 0
    pair, _nearest, pair_confidence, velocity_delta = state
    admitted = pair_confidence >= c14.CONFIDENCE_THRESHOLD
    rows = np.nonzero(admitted)[0]
    if not len(rows):
        return np.empty(0, dtype=np.float64), 0
    current_local = np.asarray(pair[4], dtype=np.float64)[rows]
    inputs = (current_local, velocity_delta[rows], pair_confidence[rows])
    confidence = _vote(inputs)
    require(
        np.all(confidence <= pair_confidence[rows] + 1e-12),
        f"c20_confidence_not_lower_bound:{data.sequence}:{int(data.frames[index])}",
    )
    return confidence, int(np.count_nonzero(confidence))


def extract_c20(
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
            confidence, supported = _local_confidence(
                data, values, index, matching_backend
            )
            require(
                len(confidence) == len(inputs[3]),
                f"c20_confidence_cardinality:{data.sequence}:{int(data.frames[index])}",
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
    root = c16.REPO / "artifacts.local" / "evidence" / "dtr-c20" / "local-motion-voting"
    args.matching_backend_receipt = root / "backend-matching.json"
    args.mode_backend_receipt = root / "backend-mode.json"
    args.route_backend_receipt = root / "backend-route.json"
    args.vote_backend_receipt = root / "backend-vote.json"
    args.fit_backend_receipt = root / "backend-fit.json"
    args.probability_backend_receipt = root / "backend-probability.json"
    args.output = root / "result.json"
    args.frozen_model = Path(__file__).resolve().with_name(
        "dtr_c20_local_motion_voting_model.json"
    )
    return args


def run() -> dict[str, Any]:
    global _VOTE_RECEIPT
    args = _arguments()
    _VOTE_RECEIPT = args.vote_backend_receipt.resolve()
    c16.SCHEMA = SCHEMA
    c16.ARM = ARM
    c16.extract_c16 = extract_c20
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
        "Can first-pair point-wise local motion votes suppress pseudo-motion "
        "without waiting for the delayed three-frame chain?"
    )
    result["feature"] = {
        "name": "point-wise local motion voting",
        "spatial_vote": "Gaussian at frozen M1 voxel scale, self vote excluded",
        "motion_agreement": "Gaussian velocity-delta agreement at frozen M1 velocity scale",
        "confidence": "pair confidence multiplied by normalized neighboring motion vote",
        "collision_mass": "local confidence * mean observed-mode discounted route entry",
        "decision": "C20 onset replaces C11 probabilistic onset; p=0.5, imminent geometry, C11 maintenance, and lifecycle unchanged",
    }
    result["backends"]["vote"] = _VOTE_SELECTION
    for row in result["development_validation"]["per_sequence"]:
        diagnostics = row["diagnostics"]
        diagnostics["locally_supported_cells"] = diagnostics.pop(
            "component_observations"
        )
    result["external_basis"] = [
        "VoteFlow: neighboring pillars vote for a shared translation to encode local rigidity",
        "RigidFlow++: spatial proximity and motion consistency suppress unreliable correspondences",
        "C20 uses continuous local votes without learned scene-flow labels or object tracks",
    ]
    result["claim_limits"] = [
        "The four C11 confirmation sequences are consumed Development validation for C20.",
        "Local agreement is a short-interval rigidity cue, not object identity or trajectory forecasting.",
        "Future CONTACT truth enters fitting/scoring only and never inference.",
        "No algorithm-fresh sequence is opened unless this fixed gate passes.",
    ]
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C20_LOCAL_MOTION_VOTING_MODEL_FROZEN",
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
