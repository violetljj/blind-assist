"""Replay frozen DTR motion arms on the fresh C1 global-CONTACT cohort.

C2 uses current/past native boxes as a privileged target observation ceiling,
current/past raw LiDAR for R7 occupancy flow, and current/past native-box point
velocity for M1-O.  Future boxes are opened only by the already frozen C1
global OBB evaluator.  The output emphasizes bounded CONTACT event recall,
lead time, and false alert segments per known non-CONTACT wearer minute.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dtr_c1_global_obb_cohort_admission import (
    CLEAR,
    CONTACT,
    PROXIMITY,
    ROSTER_SCHEMA,
    UNKNOWN,
    FrameInterval,
    _intervals,
    _load_boxes,
    _load_timestamps,
    bounded_contact_events,
    global_truth_timeline,
    require,
    sha256_file,
    write_json,
)
from dtr_c2_acquire_frozen_bags import SCHEMA as ACQUISITION_SCHEMA
from dtr_m0_r7_error_attribution import _base_predictions
from dtr_m1_point_velocity_oracle import (
    ledger_paths as oracle_ledger_paths,
    load_oracle_ledger,
    materialize_oracle_ledger,
)
from dtr_m1_confident_direct_velocity import (
    ledger_paths as confident_ledger_paths,
    load_ledger as load_confident_ledger,
    materialize as materialize_confident_ledger,
)
from dtr_r0 import Prediction, Signal
from dtr_r5_dropout_canary import (
    ACTIVE_SIGNALS,
    DROPOUT_DURATIONS_S,
    SegmentCase,
    cases_from_tracks,
)
from dtr_r7_occupancy_flow_canary import (
    _causal_pose,
    ledger_paths as flow_ledger_paths,
    load_flow_ledger,
    materialize_flow_ledger,
    run_flow_arm,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb
from jrdb_sensor_geometry_bridge import SensorSample
from dtr_c0_global_oriented_risk_contract import _run_r2, _run_r3


SCHEMA = "blindassist-dtr-c2-fresh-global-obb-replay-v1"
STATUS = "DTR_C2_FRESH_GLOBAL_OBB_FROZEN_REPLAY_COMPLETE"
DIRECT_ARMS = ("R2", "R3_C", "R7_P", "M1_O", "M1_CT")
ARMS = (*DIRECT_ARMS, "M1_CTB")


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else statistics.median(values)


def _tracks(
    *,
    boxes_by_frame: Mapping[int, Sequence[Any]],
    timestamps: Mapping[int, float],
    frame_poses: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[SensorSample]]:
    tracks: dict[str, list[SensorSample]] = {}
    for frame in sorted(timestamps):
        pose = frame_poses[frame]
        for box in boxes_by_frame.get(frame, ()):
            radius = max(0.15, 0.5 * max(float(box.length_m), float(box.width_m)))
            tracks.setdefault(str(box.label_id), []).append(
                SensorSample(
                    frame_index=frame,
                    time_s=float(timestamps[frame]),
                    ego_x_m=float(pose["x_m"]),
                    ego_y_m=float(pose["y_m"]),
                    ego_yaw_rad=float(pose["yaw_rad"]),
                    forward_m=float(box.center_forward_m),
                    left_m=float(box.center_left_m),
                    truth_radius_m=radius,
                    observed_radius_m=radius,
                    detector_track_id=str(box.label_id),
                    observed_forward_m=float(box.center_forward_m),
                    observed_left_m=float(box.center_left_m),
                )
            )
    return tracks


def _run_arm(
    arm: str,
    case: SegmentCase,
    dropout: set[int],
    *,
    r7: Any,
    m1: Any,
    m1_ct: Any,
) -> tuple[Prediction, ...]:
    if arm == "R2":
        return _run_r2(case, dropout)
    if arm == "R3_C":
        return _run_r3(case, dropout)
    if arm == "R7_P":
        return run_flow_arm(case, dropout, r7).predictions
    if arm == "M1_O":
        return run_flow_arm(case, dropout, m1).predictions
    if arm == "M1_CT":
        return run_flow_arm(case, dropout, m1_ct).predictions
    if arm == "M1_CTB":
        confident = run_flow_arm(case, dropout, m1_ct).predictions
        if not dropout:
            return confident
        bridge = run_flow_arm(case, dropout, r7).predictions
        return tuple(
            bridge_prediction if sample.frame_index in dropout else confident_prediction
            for sample, bridge_prediction, confident_prediction in zip(
                case.samples, bridge, confident
            )
        )
    raise ValueError(f"unknown_arm:{arm}")


def _prediction_frames(
    frames: Sequence[int],
    predictions: Mapping[tuple[str, int], Sequence[Prediction]],
    cases: Mapping[tuple[str, int], SegmentCase],
) -> dict[int, dict[str, set[str]]]:
    output = {int(frame): {"active": set(), "raw": set()} for frame in frames}
    for key, values in predictions.items():
        case = cases[key]
        require(len(values) == len(case.samples), f"prediction_length:{key}")
        for sample, prediction in zip(case.samples, values):
            if prediction.signal in ACTIVE_SIGNALS:
                output[sample.frame_index]["active"].add(case.label_id)
            if prediction.raw_alert is True:
                output[sample.frame_index]["raw"].add(case.label_id)
    return output


def _alert_intervals(
    frames: Sequence[int], prediction_frames: Mapping[int, Mapping[str, set[str]]]
) -> list[FrameInterval]:
    rows = [{"label": "ACTIVE" if prediction_frames[frame]["active"] else "NONE"} for frame in frames]
    return _intervals(rows, "ACTIVE")


def _overlap(left: FrameInterval, right: FrameInterval) -> bool:
    return left.first_index <= right.last_index and right.first_index <= left.last_index


def _match(
    predictions: Sequence[FrameInterval], truths: Sequence[FrameInterval]
) -> list[tuple[int, int]]:
    used: set[int] = set()
    output = []
    for prediction_index, prediction in enumerate(predictions):
        candidates = [
            index
            for index, truth in enumerate(truths)
            if index not in used and _overlap(prediction, truth)
        ]
        if candidates:
            truth_index = candidates[0]
            used.add(truth_index)
            output.append((prediction_index, truth_index))
    return output


def score_sequence(
    *,
    sequence: str,
    timeline: Sequence[Mapping[str, Any]],
    prediction_frames: Mapping[int, Mapping[str, set[str]]],
) -> dict[str, Any]:
    frames = [int(row["frame"]) for row in timeline]
    alerts = _alert_intervals(frames, prediction_frames)
    contact_intervals = _intervals(timeline, CONTACT)
    contact_matches = _match(alerts, contact_intervals)
    matched_alerts = {left for left, _right in contact_matches}
    false_alerts = [
        interval
        for index, interval in enumerate(alerts)
        if index not in matched_alerts
        and any(timeline[row]["label"] in {CLEAR, PROXIMITY} for row in range(interval.first_index, interval.last_index + 1))
    ]
    bounded = bounded_contact_events(timeline)
    bounded_intervals = [
        FrameInterval(int(event["first_index"]), int(event["last_index"])) for event in bounded
    ]
    bounded_matches = _match(alerts, bounded_intervals)
    matched_bounded = {right for _left, right in bounded_matches}
    lead_times = []
    event_rows = []
    for event_index, event in enumerate(bounded):
        matches = [left for left, right in bounded_matches if right == event_index]
        recalled = bool(matches)
        lead = None
        if recalled:
            alert = alerts[matches[0]]
            actual_contact_time = float(event["first_time_s"]) + float(event["onset_first_hit_delta_s"])
            lead = actual_contact_time - float(timeline[alert.first_index]["time_s"])
            lead_times.append(lead)
        event_rows.append(
            {
                "event_id": f"{sequence}:contact:{event_index + 1:03d}",
                "first_frame": int(event["first_frame"]),
                "last_frame": int(event["last_frame"]),
                "responsible_components": list(event["responsible_components"]),
                "recalled": recalled,
                "first_alert_lead_s": lead,
            }
        )
    durations: Counter[str] = Counter()
    for left, right in zip(timeline, timeline[1:]):
        durations[str(left["label"])] += float(right["time_s"]) - float(left["time_s"])
    non_contact_s = durations[CLEAR] + durations[PROXIMITY]
    precision = _ratio(len(matched_bounded), len(matched_bounded) + len(false_alerts))
    recall = _ratio(len(matched_bounded), len(bounded))
    return {
        "sequence": sequence,
        "bounded_contact_events": len(bounded),
        "bounded_contact_events_recalled": len(matched_bounded),
        "bounded_contact_event_recall": recall,
        "bounded_contact_event_precision": precision,
        "bounded_contact_event_f1": _f1(precision, recall),
        "alert_segments": len(alerts),
        "false_alert_segments": len(false_alerts),
        "known_non_contact_s": non_contact_s,
        "false_alert_segments_per_known_non_contact_minute": _ratio(len(false_alerts), non_contact_s / 60.0),
        "median_first_alert_lead_s": _median(lead_times),
        "lead_times_s": lead_times,
        "event_rows": event_rows,
        "false_alert_ranges": [
            {"first_frame": frames[row.first_index], "last_frame": frames[row.last_index]}
            for row in false_alerts
        ],
    }


def aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = sum(int(row["bounded_contact_events"]) for row in rows)
    recalled = sum(int(row["bounded_contact_events_recalled"]) for row in rows)
    false_alerts = sum(int(row["false_alert_segments"]) for row in rows)
    non_contact_s = sum(float(row["known_non_contact_s"]) for row in rows)
    leads = [float(value) for row in rows for value in row["lead_times_s"]]
    precision = _ratio(recalled, recalled + false_alerts)
    recall = _ratio(recalled, events)
    return {
        "sequences": len(rows),
        "bounded_contact_events": events,
        "bounded_contact_events_recalled": recalled,
        "bounded_contact_event_recall": recall,
        "bounded_contact_event_precision": precision,
        "bounded_contact_event_f1": _f1(precision, recall),
        "false_alert_segments": false_alerts,
        "known_non_contact_wearer_minutes": non_contact_s / 60.0,
        "false_alert_segments_per_known_non_contact_minute": _ratio(false_alerts, non_contact_s / 60.0),
        "median_first_alert_lead_s": _median(leads),
        "lead_times_s": leads,
    }


def _window_alerted(
    case: SegmentCase,
    predictions: Sequence[Prediction],
    dropped: set[int],
) -> bool:
    return any(
        sample.frame_index in dropped
        and prediction.signal in ACTIVE_SIGNALS
        and prediction.raw_alert is True
        for sample, prediction in zip(case.samples, predictions)
    )


def dropout_stress(
    *,
    roster_sequence: Mapping[str, Any],
    cases: Mapping[tuple[str, int], SegmentCase],
    r7: Any,
    m1: Any,
    m1_ct: Any,
) -> dict[str, Any]:
    rows = []
    for event_index, event in enumerate(roster_sequence["bounded_contact_event_details"], start=1):
        label_id = str(event["responsible_components"][0])
        contact_time = float(event["first_time_s"]) + float(event["onset_first_hit_delta_s"])
        candidates = [
            case
            for case in cases.values()
            if case.label_id == label_id
            and case.samples[0].time_s <= contact_time <= case.samples[-1].time_s
        ]
        if not candidates:
            rows.append(
                {
                    "event_index": event_index,
                    "label_id": label_id,
                    "status": "NOT_EVALUABLE_TARGET_CASE",
                }
            )
            continue
        case = min(candidates, key=lambda value: abs(value.samples[-1].time_s - contact_time))
        for duration_s in DROPOUT_DURATIONS_S:
            dropped = {
                sample.frame_index
                for sample in case.samples
                if contact_time - duration_s - 1e-9 <= sample.time_s <= contact_time + 1e-9
            }
            arms = {
                arm: _run_arm(arm, case, dropped, r7=r7, m1=m1, m1_ct=m1_ct)
                for arm in ("R2", "R7_P", "M1_O", "M1_CT")
            }
            alerted = {arm: _window_alerted(case, values, dropped) for arm, values in arms.items()}
            # M1-CTB admits raw R7 only inside an observable short track gap;
            # elsewhere it is exactly M1-CT.  The stress window is that gap.
            alerted["M1_CTB"] = alerted["R7_P"]
            rows.append(
                {
                    "event_index": event_index,
                    "label_id": label_id,
                    "duration_s": duration_s,
                    "dropped_frames": len(dropped),
                    "status": "EVALUATED" if dropped else "NOT_EVALUABLE_EMPTY_WINDOW",
                    "alerted": alerted,
                    "track_only_miss": not alerted["R2"],
                    "r7_recovered_track_only_miss": not alerted["R2"] and alerted["R7_P"],
                    "m1_recovered_track_only_miss": not alerted["R2"] and alerted["M1_O"],
                    "m1_ct_recovered_track_only_miss": not alerted["R2"] and alerted["M1_CT"],
                    "m1_ctb_recovered_track_only_miss": not alerted["R2"] and alerted["M1_CTB"],
                }
            )
    evaluated = [row for row in rows if row["status"] == "EVALUATED"]
    misses = [row for row in evaluated if row["track_only_miss"]]
    return {
        "trials": len(evaluated),
        "not_evaluable_events": sum(row["status"] != "EVALUATED" for row in rows),
        "track_only_window_misses": len(misses),
        "r7_recovered_track_only_window_misses": sum(row["r7_recovered_track_only_miss"] for row in misses),
        "m1_recovered_track_only_window_misses": sum(row["m1_recovered_track_only_miss"] for row in misses),
        "m1_ct_recovered_track_only_window_misses": sum(
            row["m1_ct_recovered_track_only_miss"] for row in misses
        ),
        "m1_ctb_recovered_track_only_window_misses": sum(
            row["m1_ctb_recovered_track_only_miss"] for row in misses
        ),
        "rows": rows,
    }


def _write_scorecard(path: Path, arms: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "arm",
        "bounded_contact_events_recalled",
        "bounded_contact_events",
        "bounded_contact_event_recall",
        "bounded_contact_event_precision",
        "bounded_contact_event_f1",
        "false_alert_segments",
        "known_non_contact_wearer_minutes",
        "false_alert_segments_per_known_non_contact_minute",
        "median_first_alert_lead_s",
    )
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            writer.writerow({"arm": arm, **{field: arms[arm][field] for field in fields if field != "arm"}})
    os.replace(partial, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve(strict=True)
    acquisition_path = args.acquisition.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(acquisition.get("schema") == ACQUISITION_SCHEMA, "acquisition_schema_drift")
    require(acquisition.get("roster_sha256") == sha256_file(roster_path), "acquisition_roster_drift")
    require(roster["source_authority"]["labels_sha256"] == sha256_file(labels_path), "labels_hash_drift")
    require(roster["source_authority"]["timestamps_sha256"] == sha256_file(timestamps_path), "timestamps_hash_drift")
    bag_rows = {str(row["sequence"]): row for row in acquisition["bags"]}
    selected_sequences = list(roster["selected_sequences"])
    if args.only_sequence is not None:
        selected_sequences = [
            row
            for row in selected_sequences
            if str(row["sequence"]) == args.only_sequence
        ]
        require(bool(selected_sequences), f"sequence_not_in_roster:{args.only_sequence}")
    output_path = args.output.resolve()
    ledger_root = output_path.parent / "ledgers"
    per_sequence = []
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    stress_rows = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps_zip:
        for sequence_index, roster_sequence in enumerate(selected_sequences, start=1):
            sequence = str(roster_sequence["sequence"])
            require(sequence in bag_rows, f"bag_missing:{sequence}")
            bag_path = Path(bag_rows[sequence]["bag"]).resolve(strict=True)
            require(sha256_file(bag_path) == bag_rows[sequence]["sha256"], f"bag_hash_drift:{sequence}")
            frame_timestamps = _load_timestamps(timestamps_zip, sequence)
            frames = sorted(frame_timestamps)
            boxes = _load_boxes(labels, sequence)
            poses, _rgb, bag_authority = read_bag_pose_and_rgb(bag_path)
            frame_poses = {
                frame: _causal_pose(poses, round(frame_timestamps[frame] * 1e9)) for frame in frames
            }
            cases_list = cases_from_tracks(
                _tracks(boxes_by_frame=boxes, timestamps=frame_timestamps, frame_poses=frame_poses)
            )
            cases = {(case.label_id, case.segment_index): case for case in cases_list}
            sequence_dir = ledger_root / sequence
            r7_npz, r7_manifest = flow_ledger_paths(sequence_dir / "r7.json")
            if not r7_npz.exists() or not r7_manifest.exists():
                materialize_flow_ledger(
                    bag_path=bag_path,
                    timestamps_path=timestamps_path,
                    calibration_dir=calibration_dir,
                    output_path=r7_npz,
                    manifest_path=r7_manifest,
                    sequence=sequence,
                    timestamps_override=frame_timestamps,
                )
            r7 = load_flow_ledger(
                r7_npz,
                r7_manifest,
                expected_sequence=sequence,
                expected_frames=frames,
            )
            m1_npz, m1_manifest = oracle_ledger_paths(sequence_dir / "m1.json")
            if not m1_npz.exists() or not m1_manifest.exists():
                materialize_oracle_ledger(
                    bag_path=bag_path,
                    timestamps_path=timestamps_path,
                    calibration_dir=calibration_dir,
                    labels_path=labels_path,
                    output_path=m1_npz,
                    manifest_path=m1_manifest,
                    sequence=sequence,
                    timestamps_override=frame_timestamps,
                )
            m1 = load_oracle_ledger(
                m1_npz,
                m1_manifest,
                expected_sequence=sequence,
                expected_frames=frames,
            )
            m1_ct_npz, m1_ct_manifest = confident_ledger_paths(sequence_dir / "m1-ct.json")
            if not m1_ct_npz.exists() or not m1_ct_manifest.exists():
                materialize_confident_ledger(
                    source_path=r7_npz,
                    source_manifest_path=r7_manifest,
                    output_path=m1_ct_npz,
                    manifest_path=m1_ct_manifest,
                )
            m1_ct = load_confident_ledger(
                m1_ct_npz,
                m1_ct_manifest,
                expected_sequence=sequence,
                expected_frames=frames,
            )
            timeline = global_truth_timeline(
                frames=frames,
                timestamps=frame_timestamps,
                boxes_by_frame=boxes,
            )
            sequence_scores = {}
            for arm in DIRECT_ARMS:
                predictions = {
                    key: _run_arm(arm, case, set(), r7=r7, m1=m1, m1_ct=m1_ct)
                    for key, case in cases.items()
                }
                score = score_sequence(
                    sequence=sequence,
                    timeline=timeline,
                    prediction_frames=_prediction_frames(frames, predictions, cases),
                )
                sequence_scores[arm] = score
                per_arm[arm].append(score)
            # With no detector gap in the natural replay, the bridge is closed
            # and M1-CTB is definitionally identical to M1-CT.
            sequence_scores["M1_CTB"] = dict(sequence_scores["M1_CT"])
            per_arm["M1_CTB"].append(sequence_scores["M1_CTB"])
            stress = dropout_stress(
                roster_sequence=roster_sequence,
                cases=cases,
                r7=r7,
                m1=m1,
                m1_ct=m1_ct,
            )
            stress_rows.append(stress)
            per_sequence.append(
                {
                    "sequence": sequence,
                    "frames": len(frames),
                    "cases": len(cases),
                    "scores": sequence_scores,
                    "dropout_stress": stress,
                    "bag_authority": bag_authority,
                    "ledgers": {
                        "r7": str(r7_npz),
                        "r7_sha256": sha256_file(r7_npz),
                        "m1": str(m1_npz),
                        "m1_sha256": sha256_file(m1_npz),
                        "m1_ct": str(m1_ct_npz),
                        "m1_ct_sha256": sha256_file(m1_ct_npz),
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "c2_sequence": sequence,
                        "index": sequence_index,
                        "total": len(selected_sequences),
                        "scores": {
                            arm: {
                                "recall": sequence_scores[arm]["bounded_contact_event_recall"],
                                "false": sequence_scores[arm]["false_alert_segments"],
                            }
                            for arm in ARMS
                        },
                    }
                ),
                flush=True,
            )
    aggregate = {arm: aggregate_scores(per_arm[arm]) for arm in ARMS}
    stress_misses = sum(int(row["track_only_window_misses"]) for row in stress_rows)
    stress = {
        "durations_s": list(DROPOUT_DURATIONS_S),
        "trials": sum(int(row["trials"]) for row in stress_rows),
        "track_only_window_misses": stress_misses,
        "r7_recovered_track_only_window_misses": sum(
            int(row["r7_recovered_track_only_window_misses"]) for row in stress_rows
        ),
        "m1_recovered_track_only_window_misses": sum(
            int(row["m1_recovered_track_only_window_misses"]) for row in stress_rows
        ),
        "m1_ct_recovered_track_only_window_misses": sum(
            int(row["m1_ct_recovered_track_only_window_misses"]) for row in stress_rows
        ),
        "m1_ctb_recovered_track_only_window_misses": sum(
            int(row["m1_ctb_recovered_track_only_window_misses"]) for row in stress_rows
        ),
        "r7_recovery_rate": _ratio(
            sum(int(row["r7_recovered_track_only_window_misses"]) for row in stress_rows),
            stress_misses,
        ),
        "m1_recovery_rate": _ratio(
            sum(int(row["m1_recovered_track_only_window_misses"]) for row in stress_rows),
            stress_misses,
        ),
        "m1_ct_recovery_rate": _ratio(
            sum(int(row["m1_ct_recovered_track_only_window_misses"]) for row in stress_rows),
            stress_misses,
        ),
        "m1_ctb_recovery_rate": _ratio(
            sum(int(row["m1_ctb_recovered_track_only_window_misses"]) for row in stress_rows),
            stress_misses,
        ),
    }
    scorecard_path = output_path.with_name(output_path.stem + ".scorecard.csv")
    _write_scorecard(scorecard_path, aggregate)
    result = {
        "schema": SCHEMA,
        "status": (
            STATUS
            if args.only_sequence is None
            else "DTR_C2_FRESH_GLOBAL_OBB_SEQUENCE_REPLAY_COMPLETE"
        ),
        "question": "Can confidence-aware direct scene motion suppress pseudo-motion while a track-gap bridge preserves dense-motion dropout recovery on fresh global OBB CONTACT replay?",
        "contract": roster["contract"],
        "frozen": {
            "roster": roster["selected_totals"],
            "arms": list(ARMS),
            "route_thresholds_lifecycle_and_motion_configs": "UNCHANGED",
            "training_or_tuning": False,
        },
        "aggregate": aggregate,
        "dropout_stress": stress,
        "per_sequence": per_sequence,
        "source": {
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "acquisition": str(acquisition_path),
            "acquisition_sha256": sha256_file(acquisition_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "calibration": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
        },
        "artifacts": {
            "scorecard_csv": str(scorecard_path),
            "scorecard_csv_sha256": sha256_file(scorecard_path),
        },
        "algorithm_increment": (
            "M1-CTB applies point/shape confidence plus identity-free ego-compensated forward-advection and velocity consistency before route-risk; raw dense motion may bypass that gate only inside an observable bounded gap of a previously tracked target, so it bridges occlusion but cannot originate an independent alert."
        ),
        "claim_limits": [
            "M1-CTB natural-replay scores equal M1-CT because the privileged target ceiling has no detector gaps; the dropout result measures the explicit bounded track-gap bridge.",
            "R2/R3-C and M1-O use privileged current/native past boxes; they are ceilings, not deployable sensor estimators.",
            "R7-P and M1-CT are truth-blind during temporal flow construction but native current boxes spatially attribute cells before global scoring.",
            "Future boxes are evaluator-only; this is curated public replay, not product, user-benefit, or safety evidence.",
        ],
    }
    write_json(output_path, result)
    return result


def merge_worker_results(args: argparse.Namespace) -> dict[str, Any]:
    """Merge sealed one-sequence workers without rerunning point-cloud work."""
    roster_path = args.roster.resolve(strict=True)
    acquisition_path = args.acquisition.resolve(strict=True)
    labels_path = args.labels.resolve(strict=True)
    timestamps_path = args.timestamps.resolve(strict=True)
    calibration_dir = args.calibration_dir.resolve(strict=True)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster_schema_drift")
    require(acquisition.get("schema") == ACQUISITION_SCHEMA, "acquisition_schema_drift")
    require(acquisition.get("roster_sha256") == sha256_file(roster_path), "acquisition_roster_drift")
    by_sequence: dict[str, dict[str, Any]] = {}
    for worker_path_value in args.merge_worker_results:
        worker_path = worker_path_value.resolve(strict=True)
        payload = json.loads(worker_path.read_text(encoding="utf-8"))
        require(payload.get("schema") == SCHEMA, f"worker_schema:{worker_path}")
        require(
            payload.get("status") == "DTR_C2_FRESH_GLOBAL_OBB_SEQUENCE_REPLAY_COMPLETE",
            f"worker_status:{worker_path}",
        )
        require(len(payload.get("per_sequence", [])) == 1, f"worker_cardinality:{worker_path}")
        row = payload["per_sequence"][0]
        sequence = str(row["sequence"])
        require(sequence not in by_sequence, f"worker_duplicate_sequence:{sequence}")
        for name in ("r7", "m1", "m1_ct"):
            ledger = Path(row["ledgers"][name]).resolve(strict=True)
            require(sha256_file(ledger) == row["ledgers"][f"{name}_sha256"], f"worker_ledger_drift:{sequence}:{name}")
        row["scores"]["M1_CTB"] = dict(row["scores"]["M1_CT"])
        stress = row["dropout_stress"]
        stress["m1_ctb_recovered_track_only_window_misses"] = int(
            stress["r7_recovered_track_only_window_misses"]
        )
        by_sequence[sequence] = row
    expected = [str(row["sequence"]) for row in roster["selected_sequences"]]
    require(set(by_sequence) == set(expected), "worker_sequence_coverage")
    per_sequence = [by_sequence[sequence] for sequence in expected]
    aggregate = {
        arm: aggregate_scores([row["scores"][arm] for row in per_sequence]) for arm in ARMS
    }
    stress_rows = [row["dropout_stress"] for row in per_sequence]
    stress_misses = sum(int(row["track_only_window_misses"]) for row in stress_rows)
    recovered = {
        name: sum(int(row[f"{name}_recovered_track_only_window_misses"]) for row in stress_rows)
        for name in ("r7", "m1", "m1_ct", "m1_ctb")
    }
    stress = {
        "durations_s": list(DROPOUT_DURATIONS_S),
        "trials": sum(int(row["trials"]) for row in stress_rows),
        "track_only_window_misses": stress_misses,
        **{
            f"{name}_recovered_track_only_window_misses": count
            for name, count in recovered.items()
        },
        **{
            f"{name}_recovery_rate": _ratio(count, stress_misses)
            for name, count in recovered.items()
        },
    }
    output_path = args.output.resolve()
    scorecard_path = output_path.with_name(output_path.stem + ".scorecard.csv")
    _write_scorecard(scorecard_path, aggregate)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "question": "Can confidence-aware direct scene motion suppress pseudo-motion while a track-gap bridge preserves dense-motion dropout recovery on fresh global OBB CONTACT replay?",
        "contract": roster["contract"],
        "frozen": {
            "roster": roster["selected_totals"],
            "arms": list(ARMS),
            "route_thresholds_lifecycle_and_motion_configs": "UNCHANGED",
            "training_or_tuning": False,
        },
        "aggregate": aggregate,
        "dropout_stress": stress,
        "per_sequence": per_sequence,
        "source": {
            "roster": str(roster_path),
            "roster_sha256": sha256_file(roster_path),
            "acquisition": str(acquisition_path),
            "acquisition_sha256": sha256_file(acquisition_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "calibration": str(calibration_dir),
            "calibration_sha256": sha256_file(calibration_dir / "lidars.yaml"),
            "worker_results": [str(path.resolve()) for path in args.merge_worker_results],
        },
        "artifacts": {
            "scorecard_csv": str(scorecard_path),
            "scorecard_csv_sha256": sha256_file(scorecard_path),
        },
        "algorithm_increment": (
            "M1-CTB uses confidence and identity-free temporal consistency before route-risk; raw dense motion may bypass that gate only inside an observable bounded gap of a previously tracked target, so it bridges occlusion but cannot originate an independent alert."
        ),
        "claim_limits": [
            "M1-CTB natural-replay scores equal M1-CT because the privileged target ceiling has no detector gaps; the dropout result measures the explicit bounded track-gap bridge.",
            "R2/R3-C and M1-O use privileged current/native past boxes; they are ceilings, not deployable sensor estimators.",
            "R7-P and M1-CT are truth-blind during temporal flow construction but native current boxes spatially attribute cells before global scoring.",
            "Future boxes are evaluator-only; this is curated public replay, not product, user-benefit, or safety evidence.",
        ],
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json"),
    )
    parser.add_argument(
        "--acquisition",
        type=Path,
        default=repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay" / "acquisition.json",
    )
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=repo
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
        / "calibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "artifacts.local" / "evidence" / "dtr-c2" / "fresh-global-obb-replay" / "result.json",
    )
    parser.add_argument("--only-sequence")
    parser.add_argument("--merge-worker-results", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_worker_results(args) if args.merge_worker_results else run(args)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "dropout": result["dropout_stress"]}))


if __name__ == "__main__":
    main()
