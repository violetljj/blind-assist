
"""Audit Mobilio's outdoor segment boundaries and replay a 1 s to 2 s VIO forecast."""

import argparse
import ast
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

from mobilio_turn_confirmation import DEVELOPMENT, HOLDOUT, REVISION, wrap_degrees as wrap
from turn_progress_metrics import retrospective_coverage


MIN_MOVE_SQ = (0.3**2, 0.6**2)
QUERY_SECONDS = 3.0
PAST_SECONDS = 1.0
FUTURE_SECONDS = 2.0
MAX_SAMPLE_GAP = 0.25
TURN_DESCRIPTIVE_DEG = 30.0
LATERALS = (1.0, 1.5, 2.0)
PREDICTORS = ("stationary", "phone", "planned", "logged_goal", "past_vio")

def delta(a, b):
    return (b[0] - a[0], b[1] - a[1])


def length(vector):
    return math.hypot(*vector)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def bearing(vector):
    return math.degrees(math.atan2(*vector))


def unit(degrees):
    radians = math.radians(degrees)
    return (math.sin(radians), math.cos(radians))


def circular_mean(yaws):
    vectors = [unit(yaw) for yaw in yaws]
    return bearing((mean(v[0] for v in vectors), mean(v[1] for v in vectors)))


def scale(vector, multiplier):
    return (vector[0] * multiplier, vector[1] * multiplier)


def plus(a, b):
    return (a[0] + b[0], a[1] + b[1])


def read_log(archive, path):
    raw = archive.read(path)
    log = json.loads(raw)
    if not {"time", "user", "target", "localX", "localZ", "latSync2", "lngSync2", "filename"} <= log.keys():
        raise ValueError(f"Invalid field schema {path}")
    names = ("time", "user", "target", "localX", "localZ")
    n = len(log["time"])
    if any(len(log[key]) != n for key in names) or n == 0:
        raise ValueError(f"Inconsistent native rows {path}")
    if not all(math.isfinite(value) for key in names for value in log[key]):
        raise ValueError(f"Nonfinite native rows {path}")
    if any(b <= a for a, b in zip(log["time"], log["time"][1:])):
        raise ValueError(f"Nonincreasing sample clock in {path}")
    log["position"] = list(zip(log["localX"], log["localZ"]))
    log["goal"] = [wrap(yaw + error) for yaw, error in zip(log["user"], log["target"])]
    return log


def audit_boundary(person, run, step, points, previous, log, names, require_close_anchor=True):
    time = log["time"]
    angles = log["goal"]
    origin = points[step - 1]
    first_position = log["position"][0]
    start_to_anchor = length(delta(origin, first_position))
    nearest = min(range(len(points)), key=lambda k: length(delta(points[k], first_position)))
    first_five_end = bisect_right(time, time[0] + QUERY_SECONDS + FUTURE_SECONDS)
    gps_track_aligned = len(log["latSync2"]) == len(log["lngSync2"]) == len(time)
    row = {
        "person": person,
        "run": run,
        "step": step,
        "source": names[step],
        "embedded_id": log["filename"],
        "previous_embedded_id": previous["filename"],
        "samples": len(time),
        "duration_s": time[-1] - time[0],
        "median_gap_s": median(b - a for a, b in zip(time, time[1:])) if len(time) > 1 else None,
        "file_gap_s": time[0] - previous["time"][-1],
        "file_gap_pos_m": length(delta(previous["position"][-1], first_position)),
        "start_to_anchor_m": start_to_anchor,
        "expected_anchor_nearest": nearest == step - 1,
        "gps_offsets_aligned": gps_track_aligned,
        "has_outbound_anchor": step < len(points),
        "has_both_leg_tangents": 1 < step < len(points),
        "start_abs_relative_error_deg": abs(wrap(log["target"][0])),
        "logged_goal_step_abs_deg": abs(wrap(angles[0] - previous["goal"][-1])),
        "logged_goal_change_in_first_5s_abs_max_deg": max(abs(wrap(angle - angles[0])) for angle in angles[:first_five_end]),
    }

    if step < len(points):
        route = delta(origin, points[step])
        route_angle = bearing(route)
        row["outbound_tangent_deg"] = route_angle
        row["start_goal_vs_outbound_tangent_abs_deg"] = abs(wrap(angles[0] - route_angle))
    if 1 < step < len(points):
        row["planned_turn_abs_deg"] = abs(wrap(route_angle - bearing(delta(points[step - 2], origin))))

    if not row["has_outbound_anchor"]:
        return row, None, "outbound_endpoint_absent"
    if not gps_track_aligned:
        return row, None, "missing_aligned_gps_offsets"
    if row["duration_s"] < QUERY_SECONDS + FUTURE_SECONDS:
        return row, None, "segment_under_5s"
    if not (0 < row["file_gap_s"] <= MAX_SAMPLE_GAP and row["file_gap_pos_m"] <= 0.25):
        return row, None, "discontinuous_boundary_pose"
    if require_close_anchor and (not row["expected_anchor_nearest"] or row["start_to_anchor_m"] > 5.0):
        return row, None, "boundary_anchor_ambiguous"
    replay = forecast(log, row, origin, route, route_angle)
    return row, replay, None if replay else "insufficient_native_window"


def forecast(log, row, origin, route_vector, route_angle):
    time = log["time"]
    pos = log["position"]
    now = bisect_right(time, time[0] + QUERY_SECONDS) - 1
    past = bisect_right(time, time[now] - PAST_SECONDS) - 1
    if past < 0 or now <= past:
        return None
    when = time[now] + FUTURE_SECONDS
    future = bisect_left(time, when)
    if future >= len(time) or max(b - a for a, b in zip(time[past:future], time[past + 1:future + 1])) > MAX_SAMPLE_GAP:
        return None
    bracket = future - 1
    after = plus(pos[bracket], scale(delta(pos[bracket], pos[future]), (when - time[bracket]) / (time[future] - time[bracket])))
    current = pos[now]
    previous_delta = delta(pos[past], current)
    actual_delta = delta(current, after)
    scaled_delta = scale(previous_delta, FUTURE_SECONDS / (time[now] - time[past]))
    same_predicted_distance = length(scaled_delta)
    route_length = length(route_vector)
    tangent = scale(route_vector, 1 / route_length)
    normal = (tangent[1], -tangent[0])
    phone_direction = circular_mean(log["user"][past:now + 1])
    directions = {
        "phone": phone_direction,
        "planned": route_angle,
        "logged_goal": log["goal"][now],
        "past_vio": bearing(previous_delta),
    }
    predictions = {name: scale(unit(direction), same_predicted_distance) for name, direction in directions.items()}
    predictions["stationary"] = (0.0, 0.0)
    enough_motion = dot(previous_delta, previous_delta) >= MIN_MOVE_SQ[0] and dot(actual_delta, actual_delta) >= MIN_MOVE_SQ[1]
    result = {
        "person": row["person"],
        "run": row["run"],
        "step": row["step"],
        "source": row["source"],
        "planned_turn_abs_deg": row.get("planned_turn_abs_deg"),
        "logged_goal_step_abs_deg": row["logged_goal_step_abs_deg"],
        "start_to_anchor_m": row["start_to_anchor_m"],
        "query_seconds_after_boundary": time[now] - time[0],
        "history_s": time[now] - time[past],
        "route_length_m": route_length,
        "past_displacement_m": length(previous_delta),
        "actual_displacement_m": length(actual_delta),
        "enough_motion_for_angle": enough_motion,
        "phone_heading_error_deg": abs(wrap(phone_direction - route_angle)),
        "query_goal_vs_tangent_abs_deg": abs(wrap(log["goal"][now] - route_angle)),
        "query_goal_change_since_boundary_abs_deg": abs(wrap(log["goal"][now] - log["goal"][0])),
        "query_forward_m": dot(delta(origin, current), tangent),
        "query_cross_m": dot(delta(origin, current), normal),
        "recent_progress_forward_m": dot(delta(origin, current), tangent),
        "recent_progress_cross_m": dot(delta(origin, current), normal),
        "recent_progress_advance_m": dot(scaled_delta, tangent),
        "observed_forward_m": dot(delta(origin, after), tangent),
        "observed_cross_m": dot(delta(origin, after), normal),
        "observed_advance_m": dot(actual_delta, tangent),
    }
    for name, prediction in predictions.items():
        endpoint = delta(origin, plus(current, prediction))
        result[f"{name}_forward_m"] = dot(endpoint, tangent)
        result[f"{name}_cross_m"] = dot(endpoint, normal)
        result[f"{name}_advance_m"] = dot(prediction, tangent)
        result[f"{name}_endpoint_error_m"] = length(delta(prediction, actual_delta))
        result[f"{name}_actual_direction_error_deg"] = (
            abs(wrap(directions[name] - bearing(actual_delta))) if enough_motion and name != "stationary" else None
        )
    return result


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"min": min(values), "median": median(values), "max": max(values)}


def clearance(row, technique, lateral):
    forward = row[f"{technique}_forward_m"]
    return min(forward - 1.0, row["route_length_m"] - forward,
               lateral - abs(row[f"{technique}_cross_m"]), row[f"{technique}_advance_m"] - 0.6)


def summarize(rows, lateral=1.5):
    labels = [clearance(row, "observed", lateral) >= 0 for row in rows]
    scores = {}
    for name in ("angle_only", "recent_progress", *PREDICTORS):
        values = [25.0 - row["phone_heading_error_deg"] if name == "angle_only" else clearance(row, name, lateral) for row in rows]
        signs = [(a > b) + 0.5 * (a == b) for a, ya in zip(values, labels) if ya for b, yb in zip(values, labels) if not yb]
        scores[name] = {
            "correct": sum(score >= 0 and label for score, label in zip(values, labels)),
            "wrong": sum(score >= 0 and not label for score, label in zip(values, labels)),
            "missed": sum(score < 0 and label for score, label in zip(values, labels)),
            "rank_auc": mean(signs) if signs else None,
            "retrospective_correct_at_wrong_budget": retrospective_coverage(labels, values),
        }
        if name in PREDICTORS:
            distances = [row[f"{name}_endpoint_error_m"] for row in rows]
            scores[name]["endpoint_error_m"] = stats(distances)
            scores[name]["direction_error_deg_on_move_enough"] = stats(row[f"{name}_actual_direction_error_deg"] for row in rows)
            scores[name]["paired_vio_minus_comparator_error_m"] = stats(vio - other for vio, other in zip(
                (row["past_vio_endpoint_error_m"] for row in rows), distances
            ))
            if name != "past_vio":
                scores[name]["smaller_endpoint_error_count"] = sum(
                    row[f"{name}_endpoint_error_m"] < row["past_vio_endpoint_error_m"] for row in rows
                )
    return {
        "records": len(rows),
        "participants": len({row["person"] for row in rows}),
        "move_enough_for_angle": sum(row["enough_motion_for_angle"] for row in rows),
        "observed_progress": sum(labels),
        "observed_advance_m": stats(row["observed_advance_m"] for row in rows),
        "past_displacement_m": stats(row["past_displacement_m"] for row in rows),
        "actual_displacement_m": stats(row["actual_displacement_m"] for row in rows),
        "goal_vs_tangent_abs_deg_at_query": stats(row["query_goal_vs_tangent_abs_deg"] for row in rows),
        "goal_change_since_boundary_abs_deg": stats(row["query_goal_change_since_boundary_abs_deg"] for row in rows),
        "models": scores,
    }


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prefix = "Mobilio-" + REVISION
    boundaries = []
    forecasts = []
    far_anchor_forecasts = []
    rejected = Counter()
    file_count = 0
    source_runs = []
    with ZipFile(args.archive) as archive:
        if archive.comment.decode("ascii") != REVISION:
            raise ValueError("Archive's embedded commit comment does not match the pinned revision")
        people = sorted(DEVELOPMENT + HOLDOUT)
        for person in people:
            directory = f"{prefix}/data/{person}"
            manual_body = archive.read(f"{directory}/manual.txt")
            points_body = archive.read(f"{directory}/points.json")
            order = [field.split(maxsplit=1)[1] for field in ast.literal_eval(manual_body.decode().splitlines()[0])]
            run_anchors = json.loads(points_body)
            if sorted(order) != ["gmaps", "opt", "oracle"] or len(run_anchors) != 6:
                raise ValueError(f"Unexpected outdoor experiment metadata for {person}")
            # The three named conditions repeat in reverse order. VI03's field
            # files are physically indexed 64 positions before their trial IDs.
            offset = 64 if person == "VI03" else 0
            for run_index, condition in enumerate(order + order[::-1]):
                if condition != "opt":
                    continue
                first = 264 + run_index * 6
                names = [f"data/{person}/data/data_{number - offset}.json" for number in range(first, first + 6)]
                logs = [read_log(archive, f"{prefix}/{name}") for name in names]
                if [log["filename"] for log in logs] != [f"HEADING_{i}" for i in range(first, first + 6)]:
                    raise ValueError(f"Unexpected outdoor trial identities for {person}/{run_index + 1}")
                points = run_anchors[run_index]
                if len(points) != 5:
                    raise ValueError(f"Unexpected number of outdoor waypoints for {person}/{run_index + 1}")
                source_runs.append({
                    "person": person,
                    "run": run_index + 1,
                    "paths": names,
                    "manual_sha256": hashlib.sha256(manual_body).hexdigest(),
                    "points_sha256": hashlib.sha256(points_body).hexdigest(),
                })
                file_count += len(logs)
                for step in range(1, len(logs)):
                    boundary, record, excluded = audit_boundary(
                        person, run_index + 1, step, points,
                        logs[step - 1], logs[step], names,
                    )
                    boundary["replay_exclusion"] = excluded
                    boundaries.append(boundary)
                    if record is None:
                        rejected[excluded] += 1
                        if excluded == "boundary_anchor_ambiguous":
                            _, secondary_record, _ = audit_boundary(
                                person, run_index + 1, step, points,
                                logs[step - 1], logs[step], names, require_close_anchor=False,
                            )
                            if secondary_record is not None:
                                far_anchor_forecasts.append(secondary_record)
                    else:
                        forecasts.append(record)
    if len(boundaries) != 140 or file_count != 168 or len(forecasts) + sum(rejected.values()) != len(boundaries):
        raise ValueError("Unexpected number of Mobilio outdoor files or indexed boundaries")
    counts = {
        "boundaries": len(boundaries),
        "opt_source_files": file_count,
        "duration_5s_or_more": sum(row["duration_s"] >= 5 for row in boundaries),
        "outbound_tangent_known": sum(row["has_outbound_anchor"] for row in boundaries),
        "both_tangents_known": sum(row["has_both_leg_tangents"] for row in boundaries),
        "outbound_known_and_5s_or_more": sum(row["has_outbound_anchor"] and row["duration_s"] >= 5 for row in boundaries),
        "logged_goal_step_abs_over_30_deg": sum(row["logged_goal_step_abs_deg"] > TURN_DESCRIPTIVE_DEG for row in boundaries),
        "planned_turn_abs_over_30_deg": sum(row.get("planned_turn_abs_deg", 0) > TURN_DESCRIPTIVE_DEG for row in boundaries),
        "start_abs_error_over_30_deg": sum(row["start_abs_relative_error_deg"] > TURN_DESCRIPTIVE_DEG for row in boundaries),
        "expected_anchor_nearest": sum(row["expected_anchor_nearest"] for row in boundaries),
        "within_5m_of_expected_anchor": sum(row["start_to_anchor_m"] <= 5 for row in boundaries),
        "gps_offsets_aligned": sum(row["gps_offsets_aligned"] for row in boundaries),
        "replay_exclusions": dict(rejected),
    }
    with args.archive.open("rb") as archive_file:
        archive_hash = hashlib.file_digest(archive_file, "sha256").hexdigest()
    report = {
        "revision": REVISION,
        "source_archive_sha256": archive_hash,
        "source_runs": source_runs,
        "native_replays_are_descriptive": True,
        "pose_and_future_measurement_share_VIO": True,
        "counts": counts,
        "boundary_stats": {key: stats(row.get(key) for row in boundaries) for key in (
            "duration_s", "median_gap_s", "file_gap_s", "file_gap_pos_m", "start_to_anchor_m",
            "logged_goal_step_abs_deg", "start_goal_vs_outbound_tangent_abs_deg",
            "logged_goal_change_in_first_5s_abs_max_deg",
        )},
        "primary": summarize(forecasts),
        "without_close_anchor_requirement": summarize(forecasts + far_anchor_forecasts),
        "far_anchor_events_only": summarize(far_anchor_forecasts),
        "logged_goal_step_over_30_deg": summarize(
            [r for r in forecasts if r["logged_goal_step_abs_deg"] > TURN_DESCRIPTIVE_DEG]
        ),
        "planned_turn_over_30_deg": summarize(
            [r for r in forecasts if (r["planned_turn_abs_deg"] or 0) > TURN_DESCRIPTIVE_DEG]
        ),
        "lateral_sensitivity_m": {str(lateral): summarize(forecasts, lateral) for lateral in LATERALS},
        "by_person": {
            person: summarize([row for row in forecasts if row["person"] == person])
            for person in sorted({r["person"] for r in forecasts})
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_rows(args.output_dir / "boundary_audit.csv", boundaries)
    write_rows(args.output_dir / "progress_replay.csv", forecasts)
    write_rows(args.output_dir / "progress_replay_far_anchors.csv", far_anchor_forecasts)
    print(json.dumps({key: report[key] for key in ("counts", "boundary_stats", "primary")}, indent=2))


if __name__ == "__main__":
    main()
