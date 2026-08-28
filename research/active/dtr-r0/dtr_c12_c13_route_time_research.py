"""Develop C12/C13 route-time onset representations over frozen C11.

Each current M1-CT route-entry cell is advanced to its predicted world-space
conflict endpoint and absolute hit time.  Only observations repeated inside the
same spatial-temporal tubelet are treated as already occupied probability mass.
C12 calibrates only the remaining innovation entering the route-conflict set;
C13 instead calibrates the peak route-entry collision rate.  The frozen C11
onset, maintenance, route geometry, probability threshold, and lifecycle remain
unchanged; either candidate can only extend an alert earlier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    _load_sequence,
    _probe,
    _select_backend,
)
from dtr_c8_global_risk_belief_bridge import _current_absolute_entries  # noqa: E402
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
from dtr_r0 import DTRConfig  # noqa: E402
from dtr_r1 import RiskEventLifecycle  # noqa: E402
from dtr_r2 import FROZEN_R2_CONFIG  # noqa: E402
from dtr_r5_dropout_canary import ACTIVE_SIGNALS  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    HORIZON_S,
)


SCHEMA = "blindassist-dtr-c13-collision-probability-rate-v1"
ARM = "M1_CPR_GLOBAL"
C11_ARM = "M1_RROQ_GLOBAL"
MINIMUM_LEAD_GAIN_S = 0.3


@dataclass(frozen=True)
class TubeletEvidence:
    base: SequenceEvidence
    consistent_score: np.ndarray
    first_passage_score: np.ndarray
    collision_rate_score: np.ndarray
    repeated_tubelets: np.ndarray


def _tubelet_frame(
    data: Any,
    index: int,
    backend: str,
) -> tuple[dict[tuple[int, int, int], tuple[float, float]], float]:
    now_s = float(data.times_s[index])
    entries = _current_absolute_entries(data, index, backend)
    world_position, world_velocity, confidence = _current_world_cells(data, index)
    require(len(entries) == len(world_position), "tubelet_entry_cell_cardinality")
    finite = np.isfinite(entries)
    rows: dict[tuple[int, int, int], tuple[float, float]] = {}
    if not np.any(finite):
        return rows, 0.0
    endpoints = world_position[finite] + world_velocity[finite] * entries[finite, None]
    hit_times = now_s + entries[finite]
    weights = confidence[finite] * np.exp(-entries[finite] / HORIZON_S)
    spatial = np.floor(endpoints / FROZEN_FLOW_CONFIG.voxel_size_m).astype(np.int64)
    temporal = np.floor(hit_times / DTRConfig().clear_grace_s).astype(np.int64)
    for xy, time_bin, weight in zip(spatial, temporal, weights):
        key = (int(xy[0]), int(xy[1]), int(time_bin))
        previous = rows.get(key)
        if previous is None or float(weight) > previous[0]:
            rows[key] = (float(weight), now_s)
    relative_time_bins = np.floor(
        entries[finite] / DTRConfig().clear_grace_s
    ).astype(np.int64)
    rate_cells: dict[tuple[int, int, int], float] = {}
    for xy, time_bin, weight in zip(spatial, relative_time_bins, weights):
        key = (int(xy[0]), int(xy[1]), int(time_bin))
        rate_cells[key] = max(rate_cells.get(key, 0.0), float(weight))
    rate_mass: dict[int, float] = {}
    for (_x, _y, time_bin), weight in rate_cells.items():
        rate_mass[time_bin] = rate_mass.get(time_bin, 0.0) + weight
    cell_area_m2 = FROZEN_FLOW_CONFIG.voxel_size_m**2
    collision_rate_score = math.log1p(
        cell_area_m2 * max(rate_mass.values(), default=0.0)
    )
    return rows, collision_rate_score


def extract_tubelets(data: Any, backend: str) -> TubeletEvidence:
    base = extract_sequence(data, backend)
    cache: dict[tuple[int, int, int], tuple[float, float]] = {}
    scores = []
    rate_scores = []
    counts = []
    grace_s = DTRConfig().clear_grace_s
    cell_area_m2 = FROZEN_FLOW_CONFIG.voxel_size_m**2
    for index in range(len(data.frames)):
        now_s = float(data.times_s[index])
        cache = {
            key: value
            for key, value in cache.items()
            if now_s - value[1] <= grace_s + EPSILON
        }
        current, collision_rate_score = _tubelet_frame(data, index, backend)
        rate_scores.append(collision_rate_score)
        repeated = []
        for key, (weight, _time_s) in current.items():
            prior = cache.get(key)
            if prior is None:
                continue
            age_s = now_s - prior[1]
            repeated.append(min(weight, prior[0]) * math.exp(-age_s / grace_s))
        scores.append(math.log1p(cell_area_m2 * sum(repeated)))
        counts.append(len(repeated))
        for key, value in current.items():
            prior = cache.get(key)
            if prior is None or value[0] >= prior[0]:
                cache[key] = value
            else:
                cache[key] = (prior[0], value[1])
    repeated_score = np.asarray(scores, dtype=np.float64)
    current_intensity = np.expm1(base.current_score)
    repeated_intensity = np.expm1(repeated_score)
    first_passage_score = np.log1p(
        np.maximum(current_intensity - repeated_intensity, 0.0)
    )
    return TubeletEvidence(
        base=base,
        consistent_score=repeated_score,
        first_passage_score=first_passage_score,
        collision_rate_score=np.asarray(rate_scores, dtype=np.float64),
        repeated_tubelets=np.asarray(counts, dtype=np.int64),
    )


def _future_first_passage_target(
    timeline: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    labels = [str(row["label"]) for row in timeline]
    known_labels = {CONTACT, PROXIMITY, CLEAR}
    eligible = np.asarray([label in known_labels for label in labels], dtype=bool)
    target = np.asarray([label == CONTACT for label in labels], dtype=np.float64)
    return eligible, target


def predict_c12(
    evidence: TubeletEvidence,
    *,
    c11_onset: Sequence[float],
    c11_maintenance: Sequence[float],
    tubelet_onset: Sequence[float],
    probability_backend: str = "numpy-platt-inference",
) -> dict[str, Any]:
    base = evidence.base
    c11_onset_probability = _probability(
        base.current_score, c11_onset, probability_backend
    )
    maintenance_probability = _probability(
        base.reachable_score, c11_maintenance, probability_backend
    )
    collision_rate_probability = _probability(
        evidence.collision_rate_score, tubelet_onset, probability_backend
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
        "collision_rate_probability_by_frame": {},
    }
    c11_onset_frames = 0
    tubelet_origin_frames = 0
    maintenance_only_frames = 0
    for index, frame_value in enumerate(base.frames):
        frame = int(frame_value)
        current_evidence = bool(np.isfinite(base.current_min_entry_s[index]))
        imminent = bool(
            current_evidence and base.current_min_entry_s[index] <= guard + EPSILON
        )
        c11_onset_now = bool(
            c11_onset_probability[index] >= PROBABILITY_THRESHOLD or imminent
        )
        tubelet_origin = bool(
            evidence.collision_rate_score[index] > 0.0
            and collision_rate_probability[index] >= PROBABILITY_THRESHOLD
        )
        maintain = bool(
            lifecycle.active
            and maintenance_probability[index] >= PROBABILITY_THRESHOLD
        )
        raw = bool(c11_onset_now or tubelet_origin or maintain)
        c11_onset_frames += int(c11_onset_now)
        tubelet_origin_frames += int(tubelet_origin and not c11_onset_now)
        maintenance_only_frames += int(maintain and not c11_onset_now and not tubelet_origin)
        minimum = (
            float(base.current_min_entry_s[index])
            if c11_onset_now or tubelet_origin
            else float(base.reachable_min_entry_s[index])
            if maintain
            else float("nan")
        )
        urgent = bool(raw and np.isfinite(minimum) and minimum <= guard + EPSILON)
        signal = lifecycle.update(
            float(base.times_s[index] - base.times_s[0]), raw, urgent=urgent
        )
        output["route_region_probability_by_frame"][str(frame)] = float(
            max(
                c11_onset_probability[index],
                collision_rate_probability[index],
                maintenance_probability[index] if lifecycle.active else 0.0,
            )
        )
        output["collision_rate_probability_by_frame"][str(frame)] = float(
            collision_rate_probability[index]
        )
        if raw:
            output["raw_alert_frames"].append(frame)
            output["minimum_entry_s_by_frame"][str(frame)] = minimum
            output["risky_cells_by_frame"][str(frame)] = int(
                max(1, evidence.repeated_tubelets[index])
            )
        if urgent:
            output["urgent_frames"].append(frame)
        if signal in ACTIVE_SIGNALS:
            output["active_alert_frames"].append(frame)
    output["diagnostics"] = {
        "frames": len(base.frames),
        "c11_onset_frames": c11_onset_frames,
        "tubelet_evidence_frames": int(np.count_nonzero(evidence.repeated_tubelets)),
        "first_passage_evidence_frames": int(
            np.count_nonzero(evidence.first_passage_score)
        ),
        "collision_rate_evidence_frames": int(
            np.count_nonzero(evidence.collision_rate_score)
        ),
        "tubelet_origin_only_frames": tubelet_origin_frames,
        "maintenance_only_frames": maintenance_only_frames,
        "active_alert_frames": len(output["active_alert_frames"]),
    }
    return output


def _sequence_roots(roots: Sequence[Path]) -> list[tuple[str, Path]]:
    output: dict[str, Path] = {}
    for root_value in roots:
        root = root_value.resolve(strict=True)
        for path in root.iterdir():
            if not path.is_dir():
                continue
            require(path.name not in output, f"duplicate_sequence:{path.name}")
            output[path.name] = root
    require(bool(output), "no_sequences")
    return sorted(output.items())


def _load_group(
    roots: Sequence[Path],
    timestamps_path: Path,
) -> tuple[list[Any], dict[str, dict[int, float]]]:
    sequence_roots = _sequence_roots(roots)
    rows = []
    timestamps_by_sequence = {}
    with zipfile.ZipFile(timestamps_path) as archive:
        for sequence, root in sequence_roots:
            timestamps = _load_timestamps(archive, sequence)
            timestamps_by_sequence[sequence] = timestamps
            rows.append(
                _load_sequence(
                    sequence=sequence,
                    timestamps=timestamps,
                    c2_root=root,
                    c3_root=root,
                )
            )
    return rows, timestamps_by_sequence


def _timelines(
    data_rows: Sequence[Any],
    timestamps_by_sequence: Mapping[str, Mapping[int, float]],
    labels_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    with zipfile.ZipFile(labels_path) as labels:
        for data in data_rows:
            output[data.sequence] = global_truth_timeline(
                frames=data.frames.tolist(),
                timestamps=timestamps_by_sequence[data.sequence],
                boxes_by_frame=_load_boxes(labels, data.sequence),
            )
    return output


def _score_group(
    rows: Sequence[Any],
    evidence: Mapping[str, TubeletEvidence],
    timelines: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    c11_onset: Sequence[float],
    c11_maintenance: Sequence[float],
    tubelet_onset: Sequence[float],
    probability_backend: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    c11_scores = []
    c12_scores = []
    per_sequence = []
    for data in rows:
        item = evidence[data.sequence]
        c11 = predict_c11(
            item.base,
            c11_onset,
            c11_maintenance,
            probability_backend=probability_backend,
        )
        c12 = predict_c12(
            item,
            c11_onset=c11_onset,
            c11_maintenance=c11_maintenance,
            tubelet_onset=tubelet_onset,
            probability_backend=probability_backend,
        )
        timeline = timelines[data.sequence]
        c11_score = score_sequence(
            sequence=data.sequence,
            timeline=timeline,
            prediction_frames=_prediction_frames(item.base.frames.tolist(), c11),
        )
        c12_score = score_sequence(
            sequence=data.sequence,
            timeline=timeline,
            prediction_frames=_prediction_frames(item.base.frames.tolist(), c12),
        )
        c11_scores.append(c11_score)
        c12_scores.append(c12_score)
        per_sequence.append(
            {
                "sequence": data.sequence,
                "scores": {C11_ARM: c11_score, ARM: c12_score},
                "diagnostics": c12["diagnostics"],
            }
        )
    return aggregate_scores(c11_scores), aggregate_scores(c12_scores), per_sequence


def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamps_path = args.timestamps.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
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
    train_rows, train_timestamps = _load_group(args.training_ledger_roots, timestamps_path)
    validation_rows, validation_timestamps = _load_group(
        args.validation_ledger_roots, timestamps_path
    )
    all_rows = [*train_rows, *validation_rows]
    probe_position, probe_velocity = _probe(all_rows)
    route_receipt = args.route_backend_receipt.resolve()
    route_selection = _select_backend(probe_position, probe_velocity, route_receipt)
    route_backend = str(route_selection["selected_backend"])
    evidence = {
        data.sequence: extract_tubelets(data, route_backend) for data in all_rows
    }
    train_timelines = _timelines(train_rows, train_timestamps, labels_path)
    validation_timelines = _timelines(
        validation_rows, validation_timestamps, labels_path
    )
    train_score = []
    train_target = []
    target_rows = {}
    for data in train_rows:
        eligible, target = _future_first_passage_target(train_timelines[data.sequence])
        train_score.append(evidence[data.sequence].collision_rate_score[eligible])
        train_target.append(target[eligible])
        target_rows[data.sequence] = {
            "eligible_frames": int(np.count_nonzero(eligible)),
            "positive_frames": int(target[eligible].sum()),
        }
    x = np.concatenate(train_score)
    y = np.concatenate(train_target)
    require(
        bool(len(x)) and 0 < float(y.mean()) < 1,
        f"tubelet_target_degenerate:frames={len(x)}:positive={float(y.sum())}",
    )
    fit_receipt = args.fit_backend_receipt.resolve()
    fit_selection = _select_fit_backend(x, y, fit_receipt)
    fit_backend = str(fit_selection["selected_backend"])
    development_params = fit_platt(x, y, fit_backend)
    probability_receipt = args.probability_backend_receipt.resolve()
    probability_selection = _select_probability_backend(
        np.concatenate([item.base.current_score for item in evidence.values()]),
        np.concatenate([item.collision_rate_score for item in evidence.values()]),
        c11_onset,
        development_params.tolist(),
        probability_receipt,
    )
    probability_backend = str(probability_selection["selected_backend"])
    baseline, candidate, per_sequence = _score_group(
        validation_rows,
        evidence,
        validation_timelines,
        c11_onset=c11_onset,
        c11_maintenance=c11_maintenance,
        tubelet_onset=development_params,
        probability_backend=probability_backend,
    )
    lead_gain_s = float(candidate["median_first_alert_lead_s"]) - float(
        baseline["median_first_alert_lead_s"]
    )
    gate = {
        "recall_not_lower": candidate["bounded_contact_events_recalled"]
        >= baseline["bounded_contact_events_recalled"],
        "false_segments_not_higher": candidate["false_alert_segments"]
        <= baseline["false_alert_segments"],
        "median_lead_gain_at_least_s": MINIMUM_LEAD_GAIN_S,
        "observed_median_lead_gain_s": lead_gain_s,
    }
    passed = bool(
        gate["recall_not_lower"]
        and gate["false_segments_not_higher"]
        and lead_gain_s >= MINIMUM_LEAD_GAIN_S - EPSILON
    )
    final_params = None
    all_target_rows = dict(target_rows)
    if passed:
        all_x = [x]
        all_y = [y]
        for data in validation_rows:
            eligible, target = _future_first_passage_target(
                validation_timelines[data.sequence]
            )
            all_x.append(evidence[data.sequence].collision_rate_score[eligible])
            all_y.append(target[eligible])
            all_target_rows[data.sequence] = {
                "eligible_frames": int(np.count_nonzero(eligible)),
                "positive_frames": int(target[eligible].sum()),
            }
        final_params = fit_platt(np.concatenate(all_x), np.concatenate(all_y), fit_backend)
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_C13_COLLISION_PROBABILITY_RATE_DEVELOPMENT_GATE_MET"
            if passed
            else "DTR_C13_COLLISION_PROBABILITY_RATE_DEVELOPMENT_GATE_NOT_MET"
        ),
        "question": (
            "Can a concentrated route-entry collision probability rate extend C11 "
            "alerts earlier without losing recall or adding false segments?"
        ),
        "fixed_gate": gate,
        "development_model": {
            "slope": float(development_params[0]),
            "intercept": float(development_params[1]),
            "decision_probability": PROBABILITY_THRESHOLD,
        },
        "final_model": (
            None
            if final_params is None
            else {
                "slope": float(final_params[0]),
                "intercept": float(final_params[1]),
                "decision_probability": PROBABILITY_THRESHOLD,
            }
        ),
        "development_validation": {
            C11_ARM: baseline,
            ARM: candidate,
            "per_sequence": per_sequence,
        },
        "training_target": {
            "name": "native CONTACT: realized path intersection within the frozen future horizon",
            "negative_censoring": "UNKNOWN frames are excluded; CLEAR and PROXIMITY are negatives",
            "by_sequence": all_target_rows,
        },
        "feature": {
            "name": "route-entry collision probability rate",
            "space_bin_m": FROZEN_FLOW_CONFIG.voxel_size_m,
            "absolute_hit_time_bin_s": DTRConfig().clear_grace_s,
            "history_s": DTRConfig().clear_grace_s,
            "formula": (
                "log1p(voxel_area * max over 0.5s entry-time bins of "
                "sum confidence * exp(-entry/3s))"
            ),
            "authority": (
                "collision-rate score > 0 and calibrated future path-conflict probability >= 0.5; "
                "frozen C11 remains an independent onset"
            ),
        },
        "backends": {
            "route": route_selection,
            "fit": fit_selection,
            "probability": probability_selection,
        },
        "sources": {
            "c11_calibrator": str(calibrator_path),
            "c11_calibrator_sha256": sha256_file(calibrator_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "training_ledger_roots": [str(path.resolve()) for path in args.training_ledger_roots],
            "validation_ledger_roots": [
                str(path.resolve()) for path in args.validation_ledger_roots
            ],
        },
        "claim_limits": [
            "The four C11 confirmation sequences are consumed Development validation for C12/C13.",
            "The future CONTACT target is training/evaluation truth only and never enters inference.",
            "A separately frozen algorithm-fresh cohort is required before any C12/C13 performance claim.",
        ],
    }
    write_json(args.output.resolve(), result)
    if passed:
        frozen = {
            "schema": SCHEMA,
            "status": "DTR_C13_COLLISION_PROBABILITY_RATE_MODEL_FROZEN",
            "model": result["final_model"],
            "c11_model": calibrator["model"],
            "feature": result["feature"],
            "development_validation": {
                C11_ARM: baseline,
                ARM: candidate,
                "fixed_gate": gate,
            },
            "training": {
                "sequences": sorted(evidence),
                "target": result["training_target"],
            },
            "sources": result["sources"],
            "claim_limits": result["claim_limits"],
        }
        write_json(args.frozen_model.resolve(), frozen)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = repo / "artifacts.local" / "evidence"
    output_root = evidence / "dtr-c13" / "collision-probability-rate"
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
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument(
        "--route-backend-receipt", type=Path, default=output_root / "backend-route.json"
    )
    parser.add_argument(
        "--fit-backend-receipt", type=Path, default=output_root / "backend-fit.json"
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
            "dtr_c13_collision_probability_rate_model.json"
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
