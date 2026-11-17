"""Compare online heading and position evidence against later annotated walk progress."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mobilio_turn_confirmation import write_json
from prepare_ping_turns import (
    DATA_SHA256,
    REVERSE,
    TURN_CUES,
    circular_center,
    deduplicate,
    route_vertices,
    wrap,
)
from turn_progress_metrics import retrospective_coverage


QUERY_MS = 3000
PAST_MS = 1000
FUTURE_MS = 2000
POSE_LAG_MS = 250
ANNOTATION_GAP_MS = 1500
CENTERLINE_M = 1.5
MIN_OUTBOUND_M = 1.0
MIN_PROGRESS_M = 0.6
HEADING_LIMIT_DEG = 25.0
MODELS = ("angle_only", "recent_progress", "last_position", "device_heading", "planned_direction", "estimated_motion")
FORECASTS = MODELS[2:]
WALK_HISTORY_MS = 2000
MAX_STATUS_MS = 12000
STATUS_PERIOD_MS = 500
STATUS_RULES = (
    "after_3s", "after_6s", "angle", "dwell_angle", "linear_angle",
    "angle_position", "angle_progress", "progress",
)


def unit_bearing(degrees):
    radians = np.radians(degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def annotated_position(participant, when, route, floor):
    """Interpolate only a short, same-route bracket in the annotated stream."""
    time = participant["gttime"]
    hi = np.searchsorted(time, when)
    if hi == len(time):
        return None
    lo = hi if time[hi] == when else hi - 1
    if (
        lo < 0 or time[hi] - time[lo] > ANNOTATION_GAP_MS
        or np.any(participant["gtroute"][lo:hi + 1] != route)
        or np.any(participant["gtfloor"][lo:hi + 1] != floor)
    ):
        return None
    a, b = participant["gtpos"][[lo, hi]]
    if lo == hi:
        return a
    return a + (b - a) * ((when - time[lo]) / (time[hi] - time[lo]))


def projection(start, end, origin, tangent):
    initial = start - origin
    final = end - origin
    return {
        "forward_m": float(final @ tangent),
        "cross_m": float(-final[0] * tangent[1] + final[1] * tangent[0]),
        "advance_m": float((final - initial) @ tangent),
    }


def leg_clearance(projected, length, lateral):
    """Signed distance to the tightest of the four progress conditions."""
    return min(
        projected["forward_m"] - MIN_OUTBOUND_M,
        length - projected["forward_m"],
        lateral - abs(projected["cross_m"]),
        projected["advance_m"] - MIN_PROGRESS_M,
    )


def compare_turn(clip, participant, vertices, calibration):
    cue_time = clip["cue_start_ms"]
    time = participant["logtime"]
    now = np.searchsorted(time, cue_time + QUERY_MS, side="right") - 1
    current = time[now]
    past = np.searchsorted(time, current - PAST_MS, side="right") - 1
    if (
        cue_time + QUERY_MS - current > POSE_LAG_MS or past < 0
        or current - time[past] > PAST_MS + POSE_LAG_MS
        or np.max(np.diff(time[past:now + 1])) > POSE_LAG_MS
    ):
        return None, "sparse_online_pose"
    if now == past or time[past] < cue_time:
        return None, "insufficient_post_cue_history"
    future = current + FUTURE_MS
    instructions = participant["instructions"]
    if np.any(
        (instructions["timestart"] > cue_time)
        & (instructions["timestart"] <= future)
        & np.isin(instructions["command"], list(TURN_CUES))
    ):
        return None, "another_directional_cue"
    route, floor = clip["route"], clip["floor"]
    annotated = (participant["gttime"] >= time[past]) & (participant["gttime"] <= future)
    if (
        np.any(participant["gtroute"][annotated] != route)
        or np.any(participant["gtfloor"][annotated] != floor)
    ):
        return None, "different_route_or_floor"
    before = annotated_position(participant, current, route, floor)
    after = annotated_position(participant, future, route, floor)
    if before is None or after is None:
        return None, "missing_annotation_bracket"

    origin, next_corner = vertices[clip["corner_index"]:clip["corner_index"] + 2]
    if not np.allclose(origin, clip["corner_xy_m"], atol=1e-8, rtol=0):
        raise ValueError("Turn clip does not refer to the expected map vertex")
    vector = next_corner - origin
    length = float(np.linalg.norm(vector))
    tangent = vector / length
    start = participant["logposition"][now]
    observed_velocity = (start - participant["logposition"][past]) / ((current - time[past]) / 1000)
    speed = np.linalg.norm(observed_velocity)
    mean_yaw = circular_center(calibration["offset_deg"] - participant["logorientation"][past:now + 1])
    direction_error = float(abs(wrap(mean_yaw - clip["target_heading_deg"])))
    displacements = {
        "last_position": np.zeros(2),
        "device_heading": (FUTURE_MS / 1000) * speed * unit_bearing(mean_yaw),
        "planned_direction": (FUTURE_MS / 1000) * speed * tangent,
        "estimated_motion": (FUTURE_MS / 1000) * observed_velocity,
    }
    actual = projection(before, after, origin, tangent)
    predictions = {model: projection(start, start + delta, origin, tangent) for model, delta in displacements.items()}
    observed = projection(participant["logposition"][past], start, origin, tangent)
    # Put past progress on the same displacement scale as the two-second requirement.
    observed["advance_m"] *= FUTURE_MS / (current - time[past])
    anchored = {
        model: projection(before, before + delta, origin, tangent) for model, delta in displacements.items()
    }
    row = {
        "person": clip["person"],
        "number": clip["number"],
        "route": route,
        "floor": floor,
        "corner_index": clip["corner_index"],
        "cue_s": float((current - cue_time) / 1000),
        "sample_time_ms": int(current),
        "leg_length_m": length,
        "estimated_speed_m_s": float(speed),
        "device_heading_error_deg": direction_error,
        "position_error_at_decision_m": float(np.linalg.norm(start - before)),
        "actual_displacement_m": float(np.linalg.norm(after - before)),
        "gt_x_now_m": float(before[0]),
        "gt_y_now_m": float(before[1]),
        "gt_x_future_m": float(after[0]),
        "gt_y_future_m": float(after[1]),
        "estimated_x_now_m": float(start[0]),
        "estimated_y_now_m": float(start[1]),
        "corner_x_m": float(origin[0]),
        "corner_y_m": float(origin[1]),
        "next_vertex_x_m": float(next_corner[0]),
        "next_vertex_y_m": float(next_corner[1]),
        **{f"actual_{field}": value for field, value in actual.items()},
    }
    for model, delta in displacements.items():
        row[f"{model}_displacement_error_m"] = float(np.linalg.norm(delta - (after - before)))
        row[f"{model}_end_error_m"] = float(np.linalg.norm(start + delta - after))
        for field, value in predictions[model].items():
            row[f"{model}_{field}"] = value
    for lateral in (1.0, CENTERLINE_M, 2.0):
        row[f"actual_clearance_{lateral}m"] = leg_clearance(actual, length, lateral)
        row[f"angle_only_clearance_{lateral}m"] = HEADING_LIMIT_DEG - direction_error
        row[f"recent_progress_clearance_{lateral}m"] = leg_clearance(observed, length, lateral)
        for model, estimate in predictions.items():
            row[f"{model}_clearance_{lateral}m"] = leg_clearance(estimate, length, lateral)
            row[f"{model}_annotated_start_clearance_{lateral}m"] = leg_clearance(anchored[model], length, lateral)
    return row, None


def ranked_pairs(label, score):
    pairs = score[label, None] - score[None, ~label]
    return float(np.sum((pairs > 0) + 0.5 * (pairs == 0))), pairs.size


def summarize(rows, lateral, models):
    label = np.array([row[f"actual_clearance_{lateral}m"] >= 0 for row in rows], dtype=bool)
    scores = {}
    for model in models:
        score = np.array([row[f"{model}_clearance_{lateral}m"] for row in rows])
        wins, pairs = ranked_pairs(label, score)
        prediction = score >= 0
        scores[model] = {
            "true_confirm": int(np.sum(prediction & label)),
            "false_confirm": int(np.sum(prediction & ~label)),
            "uncovered_progress": int(np.sum(~prediction & label)),
            "rank_auc": wins / pairs if pairs else None,
            "retrospective_correct_at_wrong_budget": retrospective_coverage(label, score),
        }
        if model in FORECASTS:
            for field in ("displacement_error_m", "end_error_m"):
                scores[model][f"median_{field}"] = float(np.median([row[f"{model}_{field}"] for row in rows]))
    return {"progress": int(np.sum(label)), "trials": len(rows), "models": scores}


def matched_corner_ranking(rows):
    """Count positive-negative comparisons within the same route, floor and corner."""
    groups = {}
    for row in rows:
        key = (row["route"], row["floor"], row["corner_index"])
        groups.setdefault(key, []).append(row)
    output = {}
    for model in ("angle_only", "estimated_motion", "estimated_motion_annotated_start"):
        wins = 0
        pairs = 0
        for group in groups.values():
            label = np.array([row[f"actual_clearance_{CENTERLINE_M}m"] >= 0 for row in group])
            score = np.array([row[f"{model}_clearance_{CENTERLINE_M}m"] for row in group])
            group_wins, group_pairs = ranked_pairs(label, score)
            wins += group_wins
            pairs += group_pairs
        output[model] = {"rank_auc": wins / pairs if pairs else None, "pairs": int(pairs)}
    return output


def status_events(clip, online, evaluated, vertices, calibration):
    """Run the prompts from online observations, then look up video labels."""
    origin, end = vertices[clip["corner_index"]:clip["corner_index"] + 2]
    length = float(np.linalg.norm(end - origin))
    tangent = (end - origin) / length
    cue = clip["cue_start_ms"]
    route, floor = clip["route"], clip["floor"]
    annotations = online["gttime"]
    near_cue = np.searchsorted(annotations, cue, side="right") - 1
    other_leg = np.flatnonzero(
        (online["gtroute"][near_cue:] != route) | (online["gtfloor"][near_cue:] != floor)
    )
    stop = cue + MAX_STATUS_MS + 1
    if len(other_leg):
        stop = min(stop, int(annotations[near_cue + other_leg[0]]))
    instructions = online["instructions"]
    for stamp, command in zip(instructions["timestart"], instructions["command"], strict=True):
        if cue < stamp < stop and command in TURN_CUES:
            stop = int(stamp)
            break

    time = online["logtime"]
    positions = online["logposition"]
    errors = wrap(clip["target_heading_deg"] - calibration["offset_deg"] + online["logorientation"])
    outputs = dict.fromkeys(STATUS_RULES)
    slots = []
    for stamp in range(cue + WALK_HISTORY_MS, stop, STATUS_PERIOD_MS):
        current = np.searchsorted(time, stamp, side="right") - 1
        previous = np.searchsorted(time, stamp - WALK_HISTORY_MS, side="right") - 1
        if (
            previous < 0 or current <= previous or stamp - time[current] > POSE_LAG_MS
            or stamp - WALK_HISTORY_MS - time[previous] > POSE_LAG_MS
        ):
            continue
        angle_error = errors[current]
        dwell_start = np.searchsorted(time, time[current] - 500, side="right") - 1
        slope_start = np.searchsorted(time, time[current] - 200, side="right") - 1
        enough_dwell = (
            dwell_start >= 0 and time[dwell_start] >= cue
            and np.max(np.diff(time[dwell_start:current + 1])) <= POSE_LAG_MS
        )
        enough_slope = (
            slope_start >= 0 and time[slope_start] >= cue
            and np.max(np.diff(time[slope_start:current + 1])) <= POSE_LAG_MS
        )
        projected_angle = (
            angle_error + 500 * wrap(angle_error - errors[slope_start]) / (time[current] - time[slope_start])
            if enough_slope else float("inf")
        )
        position = projection(positions[previous], positions[current], origin, tangent)
        in_leg = (
            MIN_OUTBOUND_M <= position["forward_m"] <= length and abs(position["cross_m"]) <= CENTERLINE_M
        )
        angle = abs(angle_error) <= HEADING_LIMIT_DEG
        progressing = in_leg and position["advance_m"] >= MIN_PROGRESS_M
        eligible = {
            "after_3s": stamp - cue >= QUERY_MS,
            "after_6s": stamp - cue >= 2 * QUERY_MS,
            "angle": angle,
            "dwell_angle": enough_dwell and np.all(np.abs(errors[dwell_start:current + 1]) <= HEADING_LIMIT_DEG),
            "linear_angle": angle and abs(projected_angle) <= HEADING_LIMIT_DEG,
            "angle_position": angle and in_leg,
            "angle_progress": angle and progressing,
            "progress": progressing,
        }
        before = annotated_position(evaluated, stamp - WALK_HISTORY_MS, route, floor)
        after = annotated_position(evaluated, stamp, route, floor)
        truth = None
        if before is not None and after is not None:
            truth = leg_clearance(projection(before, after, origin, tangent), length, CENTERLINE_M) >= 0
        event = {"time_s": (stamp - cue) / 1000, "correct": truth}
        slots.append(event)
        for rule, condition in eligible.items():
            if condition and outputs[rule] is None:
                outputs[rule] = event
    return {
        "slots": slots,
        "progress_opportunities": sum(slot["correct"] is True for slot in slots),
        "outputs": outputs,
    }


def summarize_status(records):
    summary = {}
    for rule in STATUS_RULES:
        issued = [record["outputs"][rule] for record in records if record["outputs"][rule] is not None]
        times = [row["time_s"] for row in issued if row["correct"] is True]
        paired = [
            record["outputs"][rule]["time_s"] - record["outputs"]["angle"]["time_s"]
            for record in records
            if all(record["outputs"][r] is not None and record["outputs"][r]["correct"] is True for r in (rule, "angle"))
        ]
        summary[rule] = {
            "correct": len(times),
            "early_or_unverified_progress": sum(row["correct"] is False for row in issued),
            "issued_without_video_label": sum(row["correct"] is None for row in issued),
            "silent": len(records) - len(issued),
            "silent_with_progress": sum(
                record["outputs"][rule] is None and record["progress_opportunities"] > 0 for record in records
            ),
            "median_correct_s": float(np.median(times)) if times else None,
            "correct_also_for_angle": len(paired),
            "median_paired_delay_vs_angle_s": float(np.median(paired)) if paired else None,
        }
    return {
        "clips": len(records),
        "clips_with_online_slots": sum(bool(record["slots"]) for record in records),
        "clips_with_progress": sum(record["progress_opportunities"] > 0 for record in records),
        "rules": summary,
    }


def compare_progress_to_wait(records):
    differences = []
    gained = lost = 0
    for record in records:
        wait = record["outputs"]["after_6s"]
        candidate = record["outputs"]["angle_progress"]
        wait_correct = wait is not None and wait["correct"] is True
        candidate_correct = candidate is not None and candidate["correct"] is True
        if wait_correct and candidate_correct:
            differences.append(candidate["time_s"] - wait["time_s"])
        gained += candidate_correct and not wait_correct
        lost += wait_correct and not candidate_correct
    return {
        "gained_correct": gained,
        "lost_correct": lost,
        "common_correct": len(differences),
        "median_paired_delay_s": float(np.median(differences)) if differences else None,
    }


def compare_schedule_only(records):
    """Transfer prompt times among different users at the same corner."""
    groups = {}
    for record in records:
        key = (record["route"], record["floor"], record["corner"])
        groups.setdefault(key, []).append(record)
    total = Counter()
    breakdown = {}
    included_groups = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        included_groups += 1
        for recipient in group:
            actual = recipient["outputs"]["angle_progress"]
            person = breakdown.setdefault(recipient["person"], Counter())
            for donor in group:
                if donor["person"] == recipient["person"]:
                    continue
                scheduled = donor["outputs"]["angle_progress"]
                transferred = None if scheduled is None else next(
                    (slot for slot in recipient["slots"] if slot["time_s"] >= scheduled["time_s"]), None
                )
                total["pairs"] += 1
                person["pairs"] += 1
                for name, event in (("motion_rule", actual), ("transferred_time", transferred)):
                    outcome = "silent" if event is None else {True: "correct", False: "early", None: "unknown"}[
                        event["correct"]
                    ]
                    total[f"{name}_{outcome}"] += 1
                    person[f"{name}_{outcome}"] += 1
                    if event is not None:
                        total[f"{name}_time_sum_s"] += event["time_s"]
    return {
        "matched_corners": included_groups,
        "pair_weighted_counts": dict(total),
        "mean_issued_s": {
            name: total[f"{name}_time_sum_s"] / (total["pairs"] - total[f"{name}_silent"])
            for name in ("motion_rule", "transferred_time")
            if total["pairs"] > total[f"{name}_silent"]
        },
        "by_person": breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--annotation-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    prepared = json.loads(args.clips.read_bytes())
    mask = json.loads(args.annotation_mask.read_bytes())
    if source_hash != DATA_SHA256 or prepared["receipt"]["sha256"] != source_hash or mask["ping_mat_sha256"] != source_hash:
        raise ValueError("Mismatched PING source, clips or pose-row identities")
    source = loadmat(args.source, simplify_cells=True, variable_names=["contextdata", "trajdata"])
    participants = {}
    online_participants = {}
    for index, item in enumerate(source["trajdata"], 1):
        person = f"P{index}"
        annotated = dict(item)
        indices = mask["people"][person]["unambiguous_pose_row_indices"]
        for field in ("gttime", "gtpos", "gtroute", "gtfloor"):
            annotated[field] = item[field][indices]
        if np.any(np.diff(annotated["gttime"]) <= 0):
            raise ValueError(f"Annotation pose indices do not match PING times for {person}")
        participants[person] = deduplicate(annotated)
        online_participants[person] = deduplicate(item)
    maps = {int(item["floorid"]): route_vertices(item["plannedpath"]) for item in source["contextdata"]}
    routes = {key: (maps[key[1]][::-1] if reverse else maps[key[1]]) for key, reverse in REVERSE.items()}
    rows = []
    excluded = []
    status_records = []
    for clip in prepared["clips"]:
        person = clip["person"]
        calibration = prepared["receipt"]["people"][person]["calibration"]
        route = routes[clip["route"], clip["floor"]]
        row, reason = compare_turn(clip, participants[person], route, calibration)
        status = status_events(clip, online_participants[person], participants[person], route, calibration)
        status_records.append({
            "person": person,
            "number": clip["number"],
            "route": clip["route"],
            "floor": clip["floor"],
            "corner": clip["corner_index"],
            **status,
        })
        if reason is None:
            rows.append(row)
        else:
            excluded.append({"person": person, "number": clip["number"], "reason": reason})
    if not rows:
        raise ValueError("No turn has both online pose and subsequent annotated position")
    primary = summarize(rows, CENTERLINE_M, MODELS)
    report = {
        "source_sha256": source_hash,
        "prepared_clips_sha256": hashlib.sha256(args.clips.read_bytes()).hexdigest(),
        "annotation_mask_sha256": hashlib.sha256(args.annotation_mask.read_bytes()).hexdigest(),
        "selected_major_turns": len(prepared["clips"]),
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "primary": primary,
        "by_person": {
            person: summarize(group, CENTERLINE_M, MODELS)
            for person in participants if (group := [row for row in rows if row["person"] == person])
        },
        "lateral_sensitivity": {str(lateral): summarize(rows, lateral, MODELS) for lateral in (1.0, 2.0)},
        "annotated_start_diagnostic": summarize(
            rows, CENTERLINE_M, tuple(f"{model}_annotated_start" for model in FORECASTS)
        ),
        "matched_corner_diagnostic": matched_corner_ranking(rows),
        "initial_position_error_median_m": float(np.median([row["position_error_at_decision_m"] for row in rows])),
        "actual_displacement_under_0_6m": sum(row["actual_displacement_m"] < MIN_PROGRESS_M for row in rows),
        "failed_progress_reasons": {
            "lateral_over_1_5m": sum(abs(row["actual_cross_m"]) > CENTERLINE_M for row in rows),
            "under_1m_after_corner": sum(row["actual_forward_m"] < MIN_OUTBOUND_M for row in rows),
            "beyond_next_vertex": sum(row["actual_forward_m"] > row["leg_length_m"] for row in rows),
            "under_0_6m_forward": sum(row["actual_advance_m"] < MIN_PROGRESS_M for row in rows),
        },
    }
    write_json(args.output_dir / "summary.json", report)
    write_json(args.output_dir / "excluded.json", excluded)
    write_json(args.output_dir / "status_replay.json", {
        "summary": summarize_status(status_records),
        "by_person": {
            person: summarize_status([record for record in status_records if record["person"] == person])
            for person in participants
        },
        "paired_against_6s_wait": compare_progress_to_wait(status_records),
        "same_corner_schedule_diagnostic": compare_schedule_only(status_records),
        "trials": status_records,
    })
    with (args.output_dir / "turn_progress.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "by_person"}, indent=2))


if __name__ == "__main__":
    main()
