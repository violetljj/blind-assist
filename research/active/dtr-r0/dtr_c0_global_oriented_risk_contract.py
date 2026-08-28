"""Replay frozen DTR arms under a global oriented-risk truth contract.

C0 changes only the evaluator contract and event unit.  It does not train,
tune, or alter R2, R3-C, R7-P, or M1-O predictions.  Realized future native
OBBs from every object are unioned into one wearer timeline:

    CONTACT   any future OBB intersects the 0.65 m route body;
    PROXIMITY no OBB contact, but the legacy circular envelope intersects;
    CLEAR     a full 3 s future has neither;
    UNKNOWN   no contact/proximity and the future is right-censored.

Per-target alert timelines are then unioned.  Component identity is retained
only to diagnose wrong-component/right-global-event contributions; it never
determines global alert correctness.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coda_static_ceiling import point_to_box_clearance
from dtr_m0_r7_error_attribution import _base_predictions
from dtr_m1_point_velocity_oracle import (
    ledger_paths as m1_ledger_paths,
    load_native_boxes,
    load_oracle_ledger,
)
from dtr_r0 import CausalFrame, DTRConfig, Prediction, Signal
from dtr_r3 import R3Arm, run_r3_arm
from dtr_r5_dropout_canary import (
    ACTIVE_SIGNALS,
    SegmentCase,
    cases_from_tracks,
    sample_pose,
    sensor_observation,
)
from dtr_r7_occupancy_flow_canary import (
    _causal_pose,
    ledger_paths as r7_ledger_paths,
    load_flow_ledger,
    run_flow_arm,
)
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    HORIZON_S,
    LAST_FRAME,
    ROUTE_HALF_WIDTH_M,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import load_truth_and_associate, read_jsonl, write_json


SCHEMA = "blindassist-dtr-c0-global-oriented-risk-contract-v1"
STATUS = "DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW"
CLAIM_CEILING = "READ_ONLY_RESCORING_ON_CONSUMED_M1_M3_JRDB_DEVELOPMENT_WINDOW"
CONTACT = "CONTACT"
PROXIMITY = "PROXIMITY"
CLEAR = "CLEAR"
UNKNOWN = "UNKNOWN"
ARMS = ("R2", "R3_C", "R7_P", "M1_O")


@dataclass(frozen=True)
class Interval:
    first_frame: int
    last_frame: int

    def overlaps(self, other: "Interval") -> bool:
        return self.first_frame <= other.last_frame and other.first_frame <= self.last_frame


def classify_future_contract(
    *,
    obb_hit: bool,
    circle_hit: bool,
    full_future: bool,
) -> str:
    if obb_hit:
        return CONTACT
    if circle_hit:
        return PROXIMITY
    return CLEAR if full_future else UNKNOWN


def _obb_clearance(box: Any) -> float:
    return point_to_box_clearance(
        0.0,
        0.0,
        float(box.center_forward_m),
        float(box.center_left_m),
        float(box.yaw_ego_rad),
        float(box.length_m),
        float(box.width_m),
    ) - ROUTE_HALF_WIDTH_M


def global_truth_timeline(
    *,
    boxes_by_frame: Mapping[int, Sequence[Any]],
    timestamps: Mapping[int, float],
) -> dict[int, dict[str, Any]]:
    final_time = float(timestamps[LAST_FRAME])
    output = {}
    for origin_frame in range(FIRST_FRAME, LAST_FRAME + 1):
        origin_time = float(timestamps[origin_frame])
        obb_rows = []
        circle_rows = []
        for future_frame in range(origin_frame, LAST_FRAME + 1):
            delta_s = float(timestamps[future_frame]) - origin_time
            if delta_s > HORIZON_S + 1e-9:
                break
            for box in boxes_by_frame.get(future_frame, ()):
                radius_m = max(0.15, 0.5 * max(float(box.width_m), float(box.length_m)))
                circle_clearance = math.hypot(
                    float(box.center_forward_m), float(box.center_left_m)
                ) - (ROUTE_HALF_WIDTH_M + radius_m)
                obb_clearance = _obb_clearance(box)
                row = {
                    "label_id": str(box.label_id),
                    "frame": future_frame,
                    "delta_s": delta_s,
                    "clearance_m": obb_clearance,
                }
                if obb_clearance <= 1e-9:
                    obb_rows.append(row)
                if circle_clearance <= 1e-9:
                    circle_rows.append({**row, "clearance_m": circle_clearance})
        full_future = final_time - origin_time >= HORIZON_S - 0.05
        label = classify_future_contract(
            obb_hit=bool(obb_rows),
            circle_hit=bool(circle_rows),
            full_future=full_future,
        )
        relevant = obb_rows if label == CONTACT else circle_rows if label == PROXIMITY else []
        obb_components = {str(row["label_id"]) for row in obb_rows}
        circle_components = {str(row["label_id"]) for row in circle_rows}
        first_delta = min((float(row["delta_s"]) for row in relevant), default=None)
        first_rows = (
            []
            if first_delta is None
            else [row for row in relevant if abs(float(row["delta_s"]) - first_delta) <= 1e-9]
        )
        output[origin_frame] = {
            "frame": origin_frame,
            "time_s": origin_time,
            "label": label,
            "full_future": full_future,
            "first_hit_delta_s": first_delta,
            "responsible_components": sorted({str(row["label_id"]) for row in first_rows}),
            "contact_components_in_horizon": sorted(
                obb_components
            ),
            "proximity_components_in_horizon": sorted(
                circle_components
            ),
            "circle_only_components_in_horizon": sorted(
                circle_components - obb_components
            ),
            "secondary_circle_only_proximity": bool(
                circle_components - obb_components
            ),
            "minimum_obb_clearance_m": min(
                (float(row["clearance_m"]) for row in obb_rows), default=None
            ),
            "minimum_circle_clearance_m": min(
                (float(row["clearance_m"]) for row in circle_rows), default=None
            ),
        }
    return output


def _intervals(frames: Sequence[int]) -> list[Interval]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    output = []
    first = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            output.append(Interval(first, previous))
            first = frame
        previous = frame
    output.append(Interval(first, previous))
    return output


def _match_intervals(
    predictions: Sequence[Interval], truths: Sequence[Interval]
) -> list[tuple[int, int]]:
    matches = []
    used_truth = set()
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if truth_index not in used_truth and prediction.overlaps(truth):
                matches.append((prediction_index, truth_index))
                used_truth.add(truth_index)
                break
    return matches


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _known_duration_s(
    truth: Mapping[int, dict[str, Any]], timestamps: Mapping[int, float]
) -> float:
    total = 0.0
    for frame in range(FIRST_FRAME, LAST_FRAME):
        if truth[frame]["label"] != UNKNOWN:
            total += float(timestamps[frame + 1]) - float(timestamps[frame])
    return total


def _known_non_contact_duration_s(
    truth: Mapping[int, dict[str, Any]], timestamps: Mapping[int, float]
) -> float:
    total = 0.0
    for frame in range(FIRST_FRAME, LAST_FRAME):
        if truth[frame]["label"] in {PROXIMITY, CLEAR}:
            total += float(timestamps[frame + 1]) - float(timestamps[frame])
    return total


def _contract_metrics_evaluable(truth: Mapping[int, dict[str, Any]]) -> bool:
    contact_frames = [
        frame for frame, row in truth.items() if row["label"] == CONTACT
    ]
    bounded_events = [
        event
        for event in _intervals(contact_frames)
        if event.first_frame > FIRST_FRAME and event.last_frame < LAST_FRAME
    ]
    has_known_non_contact = any(
        row["label"] in {PROXIMITY, CLEAR} for row in truth.values()
    )
    return bool(bounded_events) and has_known_non_contact


def _prediction_frames(
    case_predictions: Mapping[tuple[str, int], Sequence[Prediction]],
    cases_by_key: Mapping[tuple[str, int], SegmentCase],
) -> dict[int, dict[str, set[str]]]:
    output = {
        frame: {"active": set(), "raw": set()}
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    for key, predictions in case_predictions.items():
        case = cases_by_key[key]
        require(len(predictions) == len(case.samples), f"prediction_length:{key}")
        for sample, prediction in zip(case.samples, predictions):
            if prediction.signal in ACTIVE_SIGNALS:
                output[sample.frame_index]["active"].add(case.label_id)
            if prediction.raw_alert is True:
                output[sample.frame_index]["raw"].add(case.label_id)
    return output


def score_global_arm(
    *,
    arm: str,
    prediction_frames: Mapping[int, dict[str, set[str]]],
    truth: Mapping[int, dict[str, Any]],
    timestamps: Mapping[int, float],
) -> dict[str, Any]:
    contact_events = _intervals(
        [frame for frame, row in truth.items() if row["label"] == CONTACT]
    )
    bounded_contact_events = [
        event
        for event in contact_events
        if event.first_frame > FIRST_FRAME and event.last_frame < LAST_FRAME
    ]
    proximity_events = _intervals(
        [frame for frame, row in truth.items() if row["label"] == PROXIMITY]
    )
    alert_segments_all = _intervals(
        [frame for frame, row in prediction_frames.items() if row["active"]]
    )
    evaluable_segments = [
        segment
        for segment in alert_segments_all
        if any(
            truth[frame]["label"] != UNKNOWN
            for frame in range(segment.first_frame, segment.last_frame + 1)
        )
    ]
    matches = _match_intervals(evaluable_segments, contact_events)
    matched_predictions = {left for left, _right in matches}
    false_segments = [
        segment
        for index, segment in enumerate(evaluable_segments)
        if index not in matched_predictions
    ]
    proximity_only = [
        segment
        for segment in false_segments
        if any(segment.overlaps(event) for event in proximity_events)
    ]
    clear_false = [
        segment
        for segment in false_segments
        if segment not in proximity_only
        and any(
            truth[frame]["label"] == CLEAR
            for frame in range(segment.first_frame, segment.last_frame + 1)
        )
    ]
    descriptive_precision = _ratio(len(matches), len(evaluable_segments))
    descriptive_recall = _ratio(len(matches), len(contact_events))
    known_minutes = _known_duration_s(truth, timestamps) / 60.0
    known_non_contact_minutes = _known_non_contact_duration_s(truth, timestamps) / 60.0
    event_metrics_evaluable = bool(bounded_contact_events)
    false_rate_evaluable = known_non_contact_minutes > 0.0

    circle_only_contribution_frames = []
    for frame, prediction in prediction_frames.items():
        circle_only = set(truth[frame]["circle_only_components_in_horizon"])
        if prediction["active"] & circle_only:
            circle_only_contribution_frames.append(frame)
    circle_only_contribution_segments = _intervals(circle_only_contribution_frames)

    wrong_component_rows = []
    for prediction_index, truth_index in matches:
        prediction = evaluable_segments[prediction_index]
        event = contact_events[truth_index]
        overlap_first = max(prediction.first_frame, event.first_frame)
        overlap_last = min(prediction.last_frame, event.last_frame)
        truth_components = set()
        predicted_components = set()
        for frame in range(overlap_first, overlap_last + 1):
            truth_components.update(truth[frame]["contact_components_in_horizon"])
            predicted_components.update(prediction_frames[frame]["active"])
        for component in sorted(predicted_components - truth_components):
            wrong_component_rows.append(
                {
                    "alert_first_frame": prediction.first_frame,
                    "alert_last_frame": prediction.last_frame,
                    "contact_first_frame": event.first_frame,
                    "contact_last_frame": event.last_frame,
                    "predicted_component": component,
                    "truth_contact_components": sorted(truth_components),
                }
            )
    return {
        "arm": arm,
        "contact_events_descriptive": len(contact_events),
        "bounded_contact_events": len(bounded_contact_events),
        "contact_events_recalled_descriptive": len(matches),
        "event_metrics_evaluable": event_metrics_evaluable,
        "global_contact_event_recall": descriptive_recall if event_metrics_evaluable else None,
        "global_contact_event_precision": descriptive_precision if event_metrics_evaluable else None,
        "global_contact_event_f1": (
            _f1(descriptive_precision, descriptive_recall)
            if event_metrics_evaluable
            else None
        ),
        "descriptive_contact_event_recall": descriptive_recall,
        "descriptive_contact_event_precision": descriptive_precision,
        "descriptive_contact_event_f1": _f1(
            descriptive_precision, descriptive_recall
        ),
        "global_alert_segments": len(evaluable_segments),
        "global_false_segments_contact_contract": (
            len(false_segments) if false_rate_evaluable else None
        ),
        "global_false_segments_descriptive": len(false_segments),
        "false_rate_evaluable": false_rate_evaluable,
        "global_false_segments_per_wearer_minute": (
            _ratio(len(false_segments), known_non_contact_minutes)
            if false_rate_evaluable
            else None
        ),
        "global_clear_false_segments": len(clear_false) if false_rate_evaluable else None,
        "global_clear_false_segments_per_wearer_minute": (
            _ratio(len(clear_false), known_non_contact_minutes)
            if false_rate_evaluable
            else None
        ),
        "circle_only_proximity_alert_segments": len(
            circle_only_contribution_segments
        ),
        "circle_only_proximity_unmatched_global_segments_descriptive": len(
            proximity_only
        ),
        "wrong_component_but_right_global_event_contributions": len(
            wrong_component_rows
        ),
        "known_wearer_timeline_minutes": known_minutes,
        "known_non_contact_wearer_minutes": known_non_contact_minutes,
        "contact_event_ranges": [event.__dict__ for event in contact_events],
        "proximity_event_ranges": [event.__dict__ for event in proximity_events],
        "alert_segment_ranges": [segment.__dict__ for segment in evaluable_segments],
        "false_segment_ranges": [segment.__dict__ for segment in false_segments],
        "proximity_only_alert_ranges": [segment.__dict__ for segment in proximity_only],
        "circle_only_proximity_contribution_ranges": [
            segment.__dict__ for segment in circle_only_contribution_segments
        ],
        "clear_false_alert_ranges": [segment.__dict__ for segment in clear_false],
        "wrong_component_rows": wrong_component_rows,
    }


def _causal_frames(case: SegmentCase, dropped: set[int]) -> list[CausalFrame]:
    origin = case.samples[0].time_s
    frames = []
    for sample in case.samples:
        observation = sensor_observation(sample)
        if sample.frame_index in dropped:
            observation = None
        frames.append(
            CausalFrame(
                time_s=sample.time_s - origin,
                ego_pose=sample_pose(sample),
                observations=() if observation is None else (observation,),
                person_detection_count=int(observation is not None),
            )
        )
    return frames


def _run_r2(case: SegmentCase, dropped: set[int]) -> tuple[Prediction, ...]:
    if not dropped:
        return _base_predictions(case)
    from dtr_r2 import DTRR2Arm

    runner = DTRR2Arm(
        DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    )
    return tuple(runner.step(frame) for frame in _causal_frames(case, dropped))


def _run_r3(case: SegmentCase, dropped: set[int]) -> tuple[Prediction, ...]:
    frames = _causal_frames(case, dropped)
    return tuple(
        run_r3_arm(
            frames,
            R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED,
            r0_config=DTRConfig(
                route_horizon_s=HORIZON_S,
                route_half_width_m=ROUTE_HALF_WIDTH_M,
            ),
            guard_frames=frames,
        )
    )


def _run_case_arm(
    arm: str,
    case: SegmentCase,
    dropped: set[int],
    *,
    r7_ledger: Any,
    m1_ledger: Any,
) -> tuple[Prediction, ...]:
    if arm == "R2":
        return _run_r2(case, dropped)
    if arm == "R3_C":
        return _run_r3(case, dropped)
    if arm == "R7_P":
        return run_flow_arm(case, dropped, r7_ledger).predictions
    if arm == "M1_O":
        return run_flow_arm(case, dropped, m1_ledger).predictions
    raise ValueError(f"unknown_arm:{arm}")


def _target_obb_future_hit(
    *,
    label_id: str,
    origin_frames: Sequence[int],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    timestamps: Mapping[int, float],
) -> bool:
    for origin_frame in origin_frames:
        origin_time = float(timestamps[origin_frame])
        for future_frame in range(origin_frame, LAST_FRAME + 1):
            if float(timestamps[future_frame]) - origin_time > HORIZON_S + 1e-9:
                break
            for box in boxes_by_frame.get(future_frame, ()):
                if box.label_id == label_id and _obb_clearance(box) <= 1e-9:
                    return True
    return False


def score_dropout_contract(
    *,
    m1: dict[str, Any],
    baseline_predictions: Mapping[str, Mapping[tuple[str, int], Sequence[Prediction]]],
    cases_by_key: Mapping[tuple[str, int], SegmentCase],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    timestamps: Mapping[int, float],
    r7_ledger: Any,
    m1_ledger: Any,
) -> dict[str, Any]:
    trial_rows = []
    for duration, duration_result in m1["stress_by_duration_s"].items():
        for trial in duration_result["by_trial"]:
            target_key = next(
                key
                for key, case in cases_by_key.items()
                if case.label_id == trial["label_id"]
                and any(sample.frame_index == trial["contact_frame"] for sample in case.samples)
            )
            eligible = _target_obb_future_hit(
                label_id=trial["label_id"],
                origin_frames=trial["dropout_frames"],
                boxes_by_frame=boxes_by_frame,
                timestamps=timestamps,
            )
            arm_rows = {}
            for arm in ARMS:
                predictions = dict(baseline_predictions[arm])
                predictions[target_key] = _run_case_arm(
                    arm,
                    cases_by_key[target_key],
                    set(int(frame) for frame in trial["dropout_frames"]),
                    r7_ledger=r7_ledger,
                    m1_ledger=m1_ledger,
                )
                global_frames = _prediction_frames(predictions, cases_by_key)
                raw_components = sorted(
                    {
                        component
                        for frame in trial["dropout_frames"]
                        for component in global_frames[int(frame)]["raw"]
                    }
                )
                active_components = sorted(
                    {
                        component
                        for frame in trial["dropout_frames"]
                        for component in global_frames[int(frame)]["active"]
                    }
                )
                arm_rows[arm] = {
                    "raw_alerted": bool(raw_components),
                    "active_alerted": bool(active_components),
                    "target_component_raw_alerted": trial["label_id"]
                    in raw_components,
                    "target_component_active_alerted": trial["label_id"]
                    in active_components,
                    "raw_contributors": raw_components,
                    "active_contributors": active_components,
                }
            trial_rows.append(
                {
                    "duration_s": float(duration),
                    "label_id": trial["label_id"],
                    "category": trial["category"],
                    "legacy_circle_contact_frame": trial["contact_frame"],
                    "dropout_frames": trial["dropout_frames"],
                    "obb_contact_eligible": eligible,
                    "exclusion_reason": None
                    if eligible
                    else "LEGACY_CIRCLE_PROXIMITY_ONLY_NOT_OBB_CONTACT",
                    "arms": arm_rows,
                }
            )
    eligible_rows = [row for row in trial_rows if row["obb_contact_eligible"]]
    return {
        "legacy_trials": len(trial_rows),
        "obb_contact_trials": len(eligible_rows),
        "excluded_proximity_only_trials": len(trial_rows) - len(eligible_rows),
        "unique_obb_contact_targets": sorted({row["label_id"] for row in eligible_rows}),
        "by_arm": {
            arm: {
                "global_raw_alerted": sum(
                    bool(row["arms"][arm]["raw_alerted"]) for row in eligible_rows
                ),
                "target_component_contributed": sum(
                    bool(row["arms"][arm]["target_component_raw_alerted"])
                    for row in eligible_rows
                ),
                "total": len(eligible_rows),
                "global_raw_alert_rate": _ratio(
                    sum(bool(row["arms"][arm]["raw_alerted"]) for row in eligible_rows),
                    len(eligible_rows),
                ),
                "target_component_contribution_rate": _ratio(
                    sum(
                        bool(row["arms"][arm]["target_component_raw_alerted"])
                        for row in eligible_rows
                    ),
                    len(eligible_rows),
                ),
            }
            for arm in ARMS
        },
        "trials": trial_rows,
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8", newline="")
    os.replace(partial, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "arm",
        "contact_events_descriptive",
        "bounded_contact_events",
        "event_metrics_evaluable",
        "global_contact_event_recall",
        "global_contact_event_precision",
        "global_contact_event_f1",
        "global_alert_segments",
        "global_false_segments_contact_contract",
        "global_false_segments_per_wearer_minute",
        "global_clear_false_segments",
        "global_clear_false_segments_per_wearer_minute",
        "circle_only_proximity_alert_segments",
        "wrong_component_but_right_global_event_contributions",
        "known_wearer_timeline_minutes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _scorecard_svg(rows: Sequence[dict[str, Any]], stress: dict[str, Any]) -> str:
    width = 1060
    height = 155 + 72 * len(rows)
    columns = [
        ("global_contact_event_recall", "CONTACT recall"),
        ("global_contact_event_f1", "CONTACT F1"),
        ("global_false_segments_contact_contract", "False seg."),
        ("circle_only_proximity_alert_segments", "PROXIMITY seg."),
        ("wrong_component_but_right_global_event_contributions", "Wrong comp."),
    ]
    left = 205
    cell_width = 160
    values = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="21" font-weight="700">DTR-C0 global oriented-risk replay</text>',
        f'<text x="24" y="59" font-family="sans-serif" font-size="13" fill="#4b5563">CONTACT = realized future OBB union · secondary PROXIMITY = legacy circle-only · OBB dropout denominator {stress["obb_contact_trials"]}/{stress["legacy_trials"]}</text>',
    ]
    for index, (_key, label) in enumerate(columns):
        values.append(
            f'<text x="{left + index * cell_width + cell_width / 2:.1f}" y="96" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700">{html.escape(label)}</text>'
        )
    for row_index, row in enumerate(rows):
        y = 116 + row_index * 72
        values.append(
            f'<text x="24" y="{y + 29}" font-family="sans-serif" font-size="16" font-weight="700">{html.escape(row["arm"])}</text>'
        )
        for column_index, (key, _label) in enumerate(columns):
            raw = row[key]
            if isinstance(raw, float):
                text_value = f"{raw:.3f}"
            elif raw is None:
                text_value = "N/E"
            else:
                text_value = str(raw)
            x = left + column_index * cell_width + 10
            values.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 20}" height="45" rx="7" fill="#eff6ff" stroke="#93c5fd"/>'
            )
            values.append(
                f'<text x="{x + (cell_width - 20) / 2:.1f}" y="{y + 29}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="700" fill="#1e3a8a">{html.escape(text_value)}</text>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(values)
        + "</svg>"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    m3_path = args.m3_result.resolve(strict=True)
    output_path = args.output.resolve()
    m3 = json.loads(m3_path.read_text(encoding="utf-8"))
    require(
        m3.get("decision")
        == "DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O",
        "m3_terminal_drift",
    )
    m1_path = Path(m3["source"]["m1_result"]).resolve(strict=True)
    require(sha256_file(m1_path) == m3["source"]["m1_result_sha256"], "m3_m1_hash_drift")
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    source = m1["source"]
    r7_path = Path(source["r7_result"]).resolve(strict=True)
    known_tracks_path = Path(source["known_height_tracks"]).resolve(strict=True)
    labels_path = Path(source["labels"]).resolve(strict=True)
    timestamps_path = Path(source["timestamps"]).resolve(strict=True)
    bag_path = Path(source["bag"]).resolve(strict=True)
    require(sha256_file(r7_path) == source["r7_result_sha256"], "r7_hash_drift")
    require(sha256_file(known_tracks_path) == source["known_height_tracks_sha256"], "known_tracks_hash_drift")
    require(sha256_file(labels_path) == source["labels_sha256"], "labels_hash_drift")
    require(sha256_file(timestamps_path) == source["timestamps_sha256"], "timestamps_hash_drift")
    require(sha256_file(bag_path) == source["bag_sha256"], "bag_hash_drift")

    r7_npz, r7_manifest = r7_ledger_paths(r7_path)
    m1_npz, m1_manifest = m1_ledger_paths(m1_path)
    r7_ledger = load_flow_ledger(r7_npz, r7_manifest)
    m1_ledger = load_oracle_ledger(m1_npz, m1_manifest)
    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    causal_frame_poses = {
        frame: _causal_pose(poses, round(timestamps[frame] * 1e9))
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    boxes_by_frame = load_native_boxes(labels_path, timestamps, causal_frame_poses)
    truth = global_truth_timeline(boxes_by_frame=boxes_by_frame, timestamps=timestamps)
    context = {
        frame: {
            "image_time_s": timestamps[frame],
            "pose": interpolate_pose(poses, round(timestamps[frame] * 1e9)),
        }
        for frame in range(FIRST_FRAME, LAST_FRAME + 1)
    }
    tracks, geometry_quality = load_truth_and_associate(
        labels_path, read_jsonl(known_tracks_path), context
    )
    cases = cases_from_tracks(tracks)
    cases_by_key = {(case.label_id, case.segment_index): case for case in cases}
    baseline_predictions = {
        arm: {
            key: _run_case_arm(
                arm,
                case,
                set(),
                r7_ledger=r7_ledger,
                m1_ledger=m1_ledger,
            )
            for key, case in cases_by_key.items()
        }
        for arm in ARMS
    }
    arm_rows = [
        score_global_arm(
            arm=arm,
            prediction_frames=_prediction_frames(baseline_predictions[arm], cases_by_key),
            truth=truth,
            timestamps=timestamps,
        )
        for arm in ARMS
    ]
    stress = score_dropout_contract(
        m1=m1,
        baseline_predictions=baseline_predictions,
        cases_by_key=cases_by_key,
        boxes_by_frame=boxes_by_frame,
        timestamps=timestamps,
        r7_ledger=r7_ledger,
        m1_ledger=m1_ledger,
    )
    csv_path = output_path.with_name(output_path.stem + ".global-scorecard.csv")
    svg_path = output_path.with_name(output_path.stem + ".global-scorecard.svg")
    _write_csv(csv_path, arm_rows)
    _atomic_text(svg_path, _scorecard_svg(arm_rows, stress))
    truth_counts = Counter(row["label"] for row in truth.values())
    contact_saturated = truth_counts.get(CONTACT, 0) == len(truth)
    bounded_contact_events = _intervals(
        [frame for frame, row in truth.items() if row["label"] == CONTACT]
    )
    bounded_contact_events = [
        event
        for event in bounded_contact_events
        if event.first_frame > FIRST_FRAME and event.last_frame < LAST_FRAME
    ]
    contract_evaluable = _contract_metrics_evaluable(truth)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "How do frozen R2/R3-C/R7-P/M1-O score when physical CONTACT is the global "
            "wearer-level union of realized future OBBs and circle-only proximity is secondary?"
        ),
        "contract": {
            "primary_truth": "CONTACT: any realized future native OBB intersects the 0.65 m route body within 0-3 s",
            "secondary_truth": "PROXIMITY: legacy realized-center circle hits but no realized OBB hits",
            "clear": "full 3 s future with neither CONTACT nor PROXIMITY",
            "unknown": "right-censored future with neither observed CONTACT nor PROXIMITY",
            "event_unit": "one wearer-level global route-risk timeline; union over all native components",
            "component_identity": "diagnostic only; never participates in global alert correctness",
        },
        "frozen": {
            "arms": list(ARMS),
            "predictions_thresholds_lifecycle_and_motion_sources": "UNCHANGED_REPLAY",
            "training_or_tuning": False,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "horizon_s": HORIZON_S,
            "route_body_radius_m": ROUTE_HALF_WIDTH_M,
        },
        "truth_timeline": {
            "frame_counts": dict(sorted(truth_counts.items())),
            "contact_frame_rate": _ratio(truth_counts.get(CONTACT, 0), len(truth)),
            "contact_saturated": contact_saturated,
            "bounded_contact_events": len(bounded_contact_events),
            "contract_metrics_evaluable": contract_evaluable,
            "known_wearer_timeline_minutes": _known_duration_s(truth, timestamps) / 60.0,
            "known_non_contact_wearer_minutes": _known_non_contact_duration_s(
                truth, timestamps
            )
            / 60.0,
            "rows": [truth[frame] for frame in range(FIRST_FRAME, LAST_FRAME + 1)],
        },
        "global_replay": {row["arm"]: row for row in arm_rows},
        "dropout_contract_reset": stress,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": m1["source"]["sequence"],
            "m3_result": str(m3_path),
            "m3_result_sha256": sha256_file(m3_path),
            "m1_result": str(m1_path),
            "m1_result_sha256": sha256_file(m1_path),
            "r7_result": str(r7_path),
            "r7_result_sha256": sha256_file(r7_path),
            "r7_ledger": str(r7_npz),
            "r7_ledger_sha256": sha256_file(r7_npz),
            "m1_ledger": str(m1_npz),
            "m1_ledger_sha256": sha256_file(m1_npz),
            "known_height_tracks": str(known_tracks_path),
            "known_height_tracks_sha256": sha256_file(known_tracks_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "bag_authority": bag_authority,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "evaluator_firewall": {
            "future_native_boxes": "truth and responsibility diagnostics only; never enter predictions",
            "consumed_cohort": True,
            "geometry_quality": geometry_quality,
        },
        "decision": (
            "DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW"
            if not contract_evaluable
            else "DTR_C0_GLOBAL_ORIENTED_RISK_REPLAY_COMPLETE_FRESH_CONFIRMATION_REQUIRED"
        ),
        "next_authorized_step": (
            "Freeze a fresh global-OBB cohort with both bounded CONTACT events and known "
            "non-CONTACT wearer time before any deployable direct-motion estimator; forecasting, "
            "R8, and scene-flow competition remain closed."
        ),
        "artifacts": {
            "scorecard_csv": str(csv_path),
            "scorecard_csv_sha256": sha256_file(csv_path),
            "scorecard_svg": str(svg_path),
            "scorecard_svg_sha256": sha256_file(svg_path),
        },
        "limitations": [
            "C0 is retrospective evaluator-contract rescoring on the consumed JRDB Development window.",
            "The old and new numbers answer different truth and event-unit questions and are not directly comparable performance deltas.",
            "The nine legacy dropout trials reuse three events at three durations; the OBB denominator remains non-independent.",
            "Global union removes target-ID correctness but the replayed per-case prediction machinery remains privileged target-associated evaluation code.",
            "The global contract makes false segments per wearer minute well-defined on a suitable cohort, but this always-CONTACT window has zero non-CONTACT denominator and is NOT_EVALUABLE.",
            "In this window CONTACT is active on every frame because a left-censored native component remains in the route; primary event F1 and false/minute are NOT_EVALUABLE.",
            "No fresh generalization, deployable estimator, learned forecasting, Android, product, user-benefit, or safety claim follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    result = run(args)
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "global_replay": {
                    arm: {
                        key: result["global_replay"][arm][key]
                        for key in (
                            "global_contact_event_recall",
                            "global_contact_event_f1",
                            "global_false_segments_contact_contract",
                            "circle_only_proximity_alert_segments",
                            "wrong_component_but_right_global_event_contributions",
                        )
                    }
                    for arm in ARMS
                },
                "dropout_contract_reset": result["dropout_contract_reset"]["by_arm"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
