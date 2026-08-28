"""Develop signed empirical current/history velocity modes over frozen C11.

C14 generated symmetric covariance modes that were never observed, and C15's
frame-local component model classified nearly every component as moving.  C16
keeps only two causal, observed hypotheses for each admitted cell: its current
world velocity and the matched historical world velocity that already earned
M1 temporal confidence.  Their equally weighted collision mass is evaluated by
the unchanged continuous route geometry.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
from dtr_c15_component_velocity_mixture import C15Evidence, predict_c15  # noqa: E402
from dtr_r7_occupancy_flow_canary import FROZEN_FLOW_CONFIG, HORIZON_S  # noqa: E402


SCHEMA = "blindassist-dtr-c16-empirical-velocity-modes-v1"
ARM = "M1_EVM_GLOBAL"
C11_ARM = "M1_RROQ_GLOBAL"
PROBE_CELLS = 2048


def _numpy_modes(current_velocity: np.ndarray, velocity_delta: np.ndarray) -> np.ndarray:
    current_velocity = np.asarray(current_velocity, dtype=np.float64)
    velocity_delta = np.asarray(velocity_delta, dtype=np.float64)
    return np.stack([current_velocity, current_velocity - velocity_delta], axis=1)


def _torch_modes(
    current_velocity: np.ndarray,
    velocity_delta: np.ndarray,
    *,
    numpy_output: bool,
) -> Any:
    import torch

    current = torch.as_tensor(current_velocity, dtype=torch.float64, device="cuda")
    delta = torch.as_tensor(velocity_delta, dtype=torch.float64, device="cuda")
    output = torch.stack([current, current - delta], dim=1)
    return output.cpu().numpy() if numpy_output else output


def _select_mode_backend(
    current_velocity: np.ndarray,
    velocity_delta: np.ndarray,
    receipt: Path,
) -> dict[str, Any]:
    current_velocity = current_velocity[:PROBE_CELLS]
    velocity_delta = velocity_delta[:PROBE_CELLS]
    require(bool(len(current_velocity)), "c16_mode_probe_missing")
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _numpy_modes(current_velocity, velocity_delta)
        return cache["cpu"]

    def gpu_probe() -> Any:
        cache["gpu"] = _torch_modes(
            current_velocity, velocity_delta, numpy_output=False
        )
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu = _torch_modes(current_velocity, velocity_delta, numpy_output=True)
            require(
                np.array_equal(cache["cpu"], gpu),
                "c16_cpu_gpu_empirical_mode_mismatch",
            )
        return observation

    return select_backend(
        Workload.BATCH_TENSOR,
        cpu=BackendCandidate(
            "numpy-empirical-velocity-modes",
            "cpu",
            cpu_probe,
            lambda _output: DeviceObservation(
                "cpu",
                platform.processor() or "CPU",
                f"numpy-{np.__version__}",
            ),
        ),
        gpu=BackendCandidate(
            "torch-cuda-empirical-velocity-modes",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt,
        warmups=1,
        repeats=3,
    )


def _modes(
    current_velocity: np.ndarray,
    velocity_delta: np.ndarray,
    backend: str,
) -> np.ndarray:
    if backend == "torch-cuda-empirical-velocity-modes":
        return _torch_modes(current_velocity, velocity_delta, numpy_output=True)
    return _numpy_modes(current_velocity, velocity_delta)


def _mode_inputs(
    data: Any,
    index: int,
    recovered: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    world_position, world_velocity, confidence = _current_world_cells(data, index)
    current_velocity = _rotate_world_to_local(
        world_velocity, float(data.ego_yaw_rad[index])
    )
    wearer_velocity = _rotate_world_to_local(
        _wearer_velocity(data, index)[None, :], float(data.ego_yaw_rad[index])
    )[0]
    require(
        len(recovered[0])
        == len(current_velocity)
        == len(recovered[3])
        == len(confidence),
        f"c16_mode_cardinality:{data.sequence}:{int(data.frames[index])}",
    )
    return recovered[0], current_velocity, recovered[3], confidence, wearer_velocity


def _mode_probe(
    rows: Sequence[Any], matching_backend: str
) -> tuple[np.ndarray, ...]:
    for data in rows:
        values = _r7_values(data)
        for index in range(len(data.frames)):
            recovered = _recover_frame(data, values, index, matching_backend)
            if recovered is not None:
                return _mode_inputs(data, index, recovered)
    raise RuntimeError("c16_mode_probe_missing")


def _frame_score(
    inputs: tuple[np.ndarray, ...] | None,
    mode_backend: str,
    route_backend: str,
) -> tuple[float, float]:
    if inputs is None:
        return 0.0, float("nan")
    position, current_velocity, velocity_delta, confidence, wearer_velocity = inputs
    observed_modes = _modes(current_velocity, velocity_delta, mode_backend)
    relative_modes = observed_modes - wearer_velocity[None, None, :]
    entries = _entries(
        np.repeat(position, 2, axis=0),
        relative_modes.reshape(-1, 2),
        route_backend,
    ).reshape(len(position), 2)
    discounted = np.where(
        np.isfinite(entries),
        np.exp(-np.nan_to_num(entries, nan=0.0) / HORIZON_S),
        0.0,
    )
    mass = confidence * discounted.mean(axis=1)
    cells = np.floor(position / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    collapsed: dict[tuple[int, int], float] = {}
    for cell, value in zip(cells, mass):
        key = (int(cell[0]), int(cell[1]))
        collapsed[key] = max(collapsed.get(key, 0.0), float(value))
    finite = entries[np.isfinite(entries)]
    score = np.log1p(
        FROZEN_FLOW_CONFIG.voxel_size_m**2 * sum(collapsed.values())
    )
    return float(score), float(finite.min()) if len(finite) else float("nan")


def extract_c16(
    data: Any,
    matching_backend: str,
    mode_backend: str,
    route_backend: str,
) -> C15Evidence:
    base = extract_sequence(data, route_backend)
    values = _r7_values(data)
    scores, minimum = [], []
    for index in range(len(data.frames)):
        recovered = _recover_frame(data, values, index, matching_backend)
        inputs = None if recovered is None else _mode_inputs(data, index, recovered)
        score, entry = _frame_score(inputs, mode_backend, route_backend)
        scores.append(score)
        minimum.append(entry)
    return C15Evidence(
        base=base,
        mixture_score=np.asarray(scores, dtype=np.float64),
        mixture_min_entry_s=np.asarray(minimum, dtype=np.float64),
        admitted_cells=int(data.ct_offsets[-1]),
        components=0,
        mean_mover_probability=float("nan"),
    )


def _score_group(
    rows: Sequence[Any],
    evidence: Mapping[str, C15Evidence],
    timelines: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    c11_onset: Sequence[float],
    c11_maintenance: Sequence[float],
    c16_onset: Sequence[float],
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
            item, c16_onset, c11_maintenance, probability_backend
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
        diagnostics = {
            key: value
            for key, value in candidate["diagnostics"].items()
            if key != "mean_mover_probability"
        }
        diagnostics["observed_velocity_modes_per_cell"] = 2
        baseline_rows.append(baseline_score)
        candidate_rows.append(candidate_score)
        details.append(
            {
                "sequence": data.sequence,
                "scores": {C11_ARM: baseline_score, ARM: candidate_score},
                "diagnostics": diagnostics,
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
    probe = _mode_probe(all_rows, matching_backend)
    mode_selection = _select_mode_backend(
        probe[1], probe[2], args.mode_backend_receipt.resolve()
    )
    mode_backend = str(mode_selection["selected_backend"])
    observed_modes = _modes(probe[1], probe[2], mode_backend)
    route_selection = _select_backend(
        np.repeat(probe[0], 2, axis=0),
        (observed_modes - probe[4][None, None, :]).reshape(-1, 2),
        args.route_backend_receipt.resolve(),
    )
    route_backend = str(route_selection["selected_backend"])
    evidence = {
        data.sequence: extract_c16(
            data, matching_backend, mode_backend, route_backend
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
    require(bool(len(x)) and 0 < float(y.mean()) < 1, "c16_target_degenerate")
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
        c16_onset=development_model,
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
            "DTR_C16_EMPIRICAL_VELOCITY_MODES_DEVELOPMENT_GATE_MET"
            if passed
            else "DTR_C16_EMPIRICAL_VELOCITY_MODES_DEVELOPMENT_GATE_NOT_MET"
        ),
        "question": (
            "Can signed current/history velocity modes remove unobserved "
            "symmetric-risk mass while retaining early route conflict?"
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
            "name": "signed empirical current/history velocity modes",
            "modes": [
                "current admitted M1 world velocity",
                "causally matched historical world velocity",
            ],
            "mode_weight": 0.5,
            "collision_mass": "confidence * mean_mode[finite_entry * exp(-entry/3s)]",
            "decision": "C16 onset replaces C11 probabilistic onset; imminent geometry and C11 maintenance unchanged",
        },
        "backends": {
            "matching": matching_selection,
            "mode": mode_selection,
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
            "Nuss et al.: dynamic occupancy cells retain velocity distributions rather than one mean",
            "M-FUSE: preceding-frame motion adds useful temporal information to scene flow",
            "DeltaFlow: compact multi-frame delta cues improve motion consistency",
        ],
        "claim_limits": [
            "The four C11 confirmation sequences are consumed Development validation for C16.",
            "The two modes are observed velocities, not learned trajectories or stable object tracks.",
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
                "status": "DTR_C16_EMPIRICAL_VELOCITY_MODES_MODEL_FROZEN",
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
    output_root = evidence / "dtr-c16" / "empirical-velocity-modes"
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
        "--mode-backend-receipt",
        type=Path,
        default=output_root / "backend-mode.json",
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
            "dtr_c16_empirical_velocity_modes_model.json"
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
