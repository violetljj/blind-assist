"""Develop scene-common bias removal before C16 route conflict.

C20 showed that pseudo-motion is locally coherent, not isolated point noise.
C21 estimates the dominant velocity of every raw R7 cell in the current and
historical frames, then propagates only residual signed motion.  Coordinate-wise
median is fixed, robust, causal, and uses no route or speed threshold.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Mapping

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


SCHEMA = "blindassist-dtr-c21-scene-bias-residual-motion-v1"
ARM = "M1_SBRM_GLOBAL"
STATUS_MET = "DTR_C21_SCENE_BIAS_RESIDUAL_MOTION_DEVELOPMENT_GATE_MET"
STATUS_NOT_MET = "DTR_C21_SCENE_BIAS_RESIDUAL_MOTION_DEVELOPMENT_GATE_NOT_MET"

_BIAS_BACKEND: str | None = None
_BIAS_SELECTION: dict[str, Any] | None = None
_BIAS_RECEIPT: Path | None = None


def _numpy_bias(
    current_velocity_world: np.ndarray,
    previous_velocity_world: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.median(np.asarray(current_velocity_world, dtype=np.float64), axis=0),
            np.median(np.asarray(previous_velocity_world, dtype=np.float64), axis=0),
        ]
    )


def _torch_bias(
    current_velocity_world: np.ndarray,
    previous_velocity_world: np.ndarray,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    current = torch.as_tensor(
        current_velocity_world, dtype=torch.float64, device="cuda"
    )
    previous = torch.as_tensor(
        previous_velocity_world, dtype=torch.float64, device="cuda"
    )
    output = torch.stack(
        [torch.quantile(current, 0.5, dim=0), torch.quantile(previous, 0.5, dim=0)]
    )
    return output.cpu().numpy() if numpy_output else output


def _select_bias_backend(inputs: tuple[np.ndarray, ...]) -> dict[str, Any]:
    require(_BIAS_RECEIPT is not None, "c21_bias_receipt_missing")
    probe = tuple(np.asarray(value[: c16.PROBE_CELLS]) for value in inputs)
    require(bool(len(probe[0])) and bool(len(probe[1])), "c21_bias_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_bias(*probe)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_bias(*probe, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_bias(*probe, numpy_output=True)
            require(
                np.allclose(cache["cpu"], gpu, atol=1e-12, rtol=1e-12),
                "c21_cpu_gpu_bias_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-scene-velocity-median",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-scene-velocity-median",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=_BIAS_RECEIPT,
        warmups=1,
        repeats=3,
    )


def _bias(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    global _BIAS_BACKEND, _BIAS_SELECTION
    if _BIAS_BACKEND is None:
        _BIAS_SELECTION = _select_bias_backend(inputs)
        _BIAS_BACKEND = str(_BIAS_SELECTION["selected_backend"])
    if _BIAS_BACKEND == "torch-cuda-scene-velocity-median":
        return _torch_bias(*inputs, numpy_output=True)
    return _numpy_bias(*inputs)


def _residual_inputs(
    data: Any,
    index: int,
    recovered: tuple[np.ndarray, ...],
    values: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    inputs = c16._mode_inputs(data, index, recovered)
    pair = c14._raw_pair(data, values, index)
    require(pair is not None, f"c21_raw_pair_missing:{data.sequence}:{int(data.frames[index])}")
    bias_world = _bias((pair[7], pair[8]))
    bias_local = c16._rotate_world_to_local(
        bias_world, float(data.ego_yaw_rad[index])
    )
    current_velocity = inputs[1] - bias_local[0][None, :]
    velocity_delta = inputs[2] - bias_local[0][None, :] + bias_local[1][None, :]
    return inputs[0], current_velocity, velocity_delta, inputs[3], inputs[4]


def extract_c21(
    data: Any,
    matching_backend: str,
    mode_backend: str,
    route_backend: str,
) -> C15Evidence:
    base = c16.extract_sequence(data, route_backend)
    values = c14._r7_values(data)
    scores, minimum = [], []
    residual_cells = 0
    for index in range(len(data.frames)):
        recovered = c14._recover_frame(data, values, index, matching_backend)
        inputs = (
            None
            if recovered is None
            else _residual_inputs(data, index, recovered, values)
        )
        score, entry = c16._frame_score(inputs, mode_backend, route_backend)
        scores.append(score)
        minimum.append(entry)
        residual_cells += 0 if inputs is None else len(inputs[0])
    return C15Evidence(
        base=base,
        mixture_score=np.asarray(scores, dtype=np.float64),
        mixture_min_entry_s=np.asarray(minimum, dtype=np.float64),
        admitted_cells=int(data.ct_offsets[-1]),
        components=residual_cells,
        mean_mover_probability=float("nan"),
    )


def _arguments() -> Any:
    args = c16.parse_args()
    root = c16.REPO / "artifacts.local" / "evidence" / "dtr-c21" / "scene-bias-residual-motion"
    args.matching_backend_receipt = root / "backend-matching.json"
    args.mode_backend_receipt = root / "backend-mode.json"
    args.route_backend_receipt = root / "backend-route.json"
    args.bias_backend_receipt = root / "backend-bias.json"
    args.fit_backend_receipt = root / "backend-fit.json"
    args.probability_backend_receipt = root / "backend-probability.json"
    args.output = root / "result.json"
    args.frozen_model = Path(__file__).resolve().with_name(
        "dtr_c21_scene_bias_residual_motion_model.json"
    )
    return args


def run() -> dict[str, Any]:
    global _BIAS_RECEIPT
    args = _arguments()
    _BIAS_RECEIPT = args.bias_backend_receipt.resolve()
    c16.SCHEMA = SCHEMA
    c16.ARM = ARM
    c16.extract_c16 = extract_c21
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
        "Can robust scene-common velocity removal separate static pseudo-motion "
        "from independently moving route-conflict cells?"
    )
    result["feature"] = {
        "name": "scene-common bias residual motion",
        "bias": "coordinate-wise median velocity over every raw R7 cell in each causal frame",
        "modes": "current and historical signed velocities after frame-specific bias removal",
        "collision_mass": "M1 pair confidence * mean residual-mode discounted route entry",
        "decision": "C21 onset replaces C11 probabilistic onset; p=0.5, imminent geometry, C11 maintenance, and lifecycle unchanged",
    }
    result["backends"]["bias"] = _BIAS_SELECTION
    for row in result["development_validation"]["per_sequence"]:
        diagnostics = row["diagnostics"]
        diagnostics["residual_motion_cells"] = diagnostics.pop(
            "component_observations"
        )
    result["external_basis"] = [
        "SLIM: robust ego-motion and static-background aggregation improve scene flow and motion segmentation",
        "SCOPE: future occupancy explicitly compensates robot motion before dynamic prediction",
        "C21 estimates only a causal residual bias and does not use future labels or object tracks",
    ]
    result["claim_limits"] = [
        "The four C11 confirmation sequences are consumed Development validation for C21.",
        "Scene median assumes the dominant raw-cell motion is common bias; crowded co-motion can violate this.",
        "Future CONTACT truth enters fitting/scoring only and never inference.",
        "No algorithm-fresh sequence is opened unless this fixed gate passes.",
    ]
    write_json(args.output.resolve(), result)
    if passed:
        write_json(
            args.frozen_model.resolve(),
            {
                "schema": SCHEMA,
                "status": "DTR_C21_SCENE_BIAS_RESIDUAL_MOTION_MODEL_FROZEN",
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
