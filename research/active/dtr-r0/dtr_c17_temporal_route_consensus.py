"""Develop causal route-entry consensus over C16's signed velocity modes.

C16 showed that retaining current and matched historical velocity improves lead,
but its arithmetic mean lets either mode create risk mass.  C17 admits mass only
when both causal observations predict route entry.  This is a parameter-free
lower envelope over observed motion, not a duration or route-threshold gate.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import dtr_c16_empirical_velocity_modes as c16
from dtr_c1_global_obb_cohort_admission import require, write_json
from tools.research_backend import (
    BackendCandidate,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)


SCHEMA = "blindassist-dtr-c17-temporal-route-consensus-v1"
ARM = "M1_TRC_GLOBAL"
STATUS_MET = "DTR_C17_TEMPORAL_ROUTE_CONSENSUS_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C17_TEMPORAL_ROUTE_CONSENSUS_DEVELOPMENT_GATE_NOT_MET"

_CONSENSUS_BACKEND: str | None = None
_CONSENSUS_SELECTION: dict[str, Any] | None = None
_CONSENSUS_RECEIPT: Path | None = None


def _numpy_consensus(discounted: np.ndarray) -> np.ndarray:
    return np.asarray(discounted, dtype=np.float64).min(axis=1)


def _torch_consensus(discounted: np.ndarray, *, numpy_output: bool) -> Any:
    import torch

    values = torch.as_tensor(discounted, dtype=torch.float64, device="cuda")
    output = values.min(dim=1).values
    return output.cpu().numpy() if numpy_output else output


def _select_consensus_backend(discounted: np.ndarray) -> dict[str, Any]:
    require(_CONSENSUS_RECEIPT is not None, "c17_consensus_receipt_missing")
    probe = np.asarray(discounted[: c16.PROBE_CELLS], dtype=np.float64)
    require(bool(len(probe)), "c17_consensus_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_consensus(probe)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_consensus(probe, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_consensus(probe, numpy_output=True)
            require(
                np.array_equal(cache["cpu"], gpu),
                "c17_cpu_gpu_consensus_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-temporal-route-consensus",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-temporal-route-consensus",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=_CONSENSUS_RECEIPT,
        warmups=1,
        repeats=3,
    )


def _consensus(discounted: np.ndarray) -> np.ndarray:
    global _CONSENSUS_BACKEND, _CONSENSUS_SELECTION
    if _CONSENSUS_BACKEND is None:
        _CONSENSUS_SELECTION = _select_consensus_backend(discounted)
        _CONSENSUS_BACKEND = str(_CONSENSUS_SELECTION["selected_backend"])
    if _CONSENSUS_BACKEND == "torch-cuda-temporal-route-consensus":
        return _torch_consensus(discounted, numpy_output=True)
    return _numpy_consensus(discounted)


def _frame_score(
    inputs: tuple[np.ndarray, ...] | None,
    mode_backend: str,
    route_backend: str,
) -> tuple[float, float]:
    if inputs is None:
        return 0.0, float("nan")
    position, current_velocity, velocity_delta, confidence, wearer_velocity = inputs
    observed_modes = c16._modes(current_velocity, velocity_delta, mode_backend)
    relative_modes = observed_modes - wearer_velocity[None, None, :]
    entries = c16._entries(
        np.repeat(position, 2, axis=0),
        relative_modes.reshape(-1, 2),
        route_backend,
    ).reshape(len(position), 2)
    discounted = np.where(
        np.isfinite(entries),
        np.exp(-np.nan_to_num(entries, nan=0.0) / c16.HORIZON_S),
        0.0,
    )
    mass = confidence * _consensus(discounted)
    cells = np.floor(position / c16.FROZEN_FLOW_CONFIG.voxel_size_m).astype(
        np.int64
    )
    collapsed: dict[tuple[int, int], float] = {}
    for cell, value in zip(cells, mass):
        key = (int(cell[0]), int(cell[1]))
        collapsed[key] = max(collapsed.get(key, 0.0), float(value))
    finite = entries[np.isfinite(entries)]
    score = np.log1p(
        c16.FROZEN_FLOW_CONFIG.voxel_size_m**2 * sum(collapsed.values())
    )
    return float(score), float(finite.min()) if len(finite) else float("nan")


def _arguments() -> Any:
    args = c16.parse_args()
    root = c16.REPO / "artifacts.local" / "evidence" / "dtr-c17" / "temporal-route-consensus"
    args.matching_backend_receipt = root / "backend-matching.json"
    args.mode_backend_receipt = root / "backend-mode.json"
    args.route_backend_receipt = root / "backend-route.json"
    args.fit_backend_receipt = root / "backend-fit.json"
    args.probability_backend_receipt = root / "backend-probability.json"
    args.consensus_backend_receipt = root / "backend-consensus.json"
    args.output = root / "result.json"
    args.frozen_model = Path(__file__).resolve().with_name(
        "dtr_c17_temporal_route_consensus_model.json"
    )
    return args


def run() -> dict[str, Any]:
    global _CONSENSUS_RECEIPT
    args = _arguments()
    _CONSENSUS_RECEIPT = args.consensus_backend_receipt.resolve()
    c16.SCHEMA = SCHEMA
    c16.ARM = ARM
    c16._frame_score = _frame_score
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
        "Does parameter-free agreement between current and matched historical "
        "route entry retain C16 lead while rejecting one-frame pseudo-motion?"
    )
    result["feature"] = {
        "name": "causal temporal route-entry consensus",
        "modes": [
            "current admitted M1 world velocity",
            "causally matched historical world velocity",
        ],
        "mode_aggregation": "minimum discounted collision mass",
        "collision_mass": "confidence * min_mode[finite_entry * exp(-entry/3s)]",
        "decision": "C17 onset replaces C11 probabilistic onset; imminent geometry and C11 maintenance unchanged",
    }
    result["backends"]["consensus"] = _CONSENSUS_SELECTION
    result["external_basis"] = [
        "Nuss et al.: dynamic occupancy cells retain velocity distributions rather than one mean",
        "M-FUSE: preceding-frame motion adds useful temporal information to scene flow",
        "DeltaFlow: compact multi-frame delta cues improve motion consistency",
        "C17 lower-envelope inference requires agreement between the two observed causal velocity hypotheses",
    ]
    result["claim_limits"] = [
        "The four C11 confirmation sequences are consumed Development validation for C17.",
        "Consensus spans two causal observations and is not a stable object track or learned trajectory.",
        "Future CONTACT truth enters fitting/scoring only and never inference.",
        "No algorithm-fresh sequence is opened unless this fixed gate passes.",
    ]
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C17_TEMPORAL_ROUTE_CONSENSUS_MODEL_FROZEN",
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
