"""Test whether an incoming-route position correction improves exit confirmation."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.io import loadmat

from mobilio_turn_confirmation import write_json
from ping_turn_progress import (
    CENTERLINE_M,
    MIN_PROGRESS_M,
    POSE_LAG_MS,
    WALK_HISTORY_MS,
    annotated_position,
    leg_clearance,
    projection,
    status_events,
    status_window_end,
)
from prepare_ping_turns import DATA_SHA256, REVERSE, deduplicate, route_vertices


PREFIX_MS = np.arange(-5000, -999, 500)
PREFIX_MIN_SAMPLES = 5
RULE = "angle_progress"


def estimate_shift(clip, online, vertices):
    """Use only previously emitted estimates and the finite incoming map leg."""
    corner_id = clip["corner_index"]
    previous_vertex, corner = vertices[corner_id - 1:corner_id + 1]
    run = corner - previous_vertex
    length = np.linalg.norm(run)
    tangent = run / length
    normal = np.array([-tangent[1], tangent[0]])
    stamps = clip["cue_start_ms"] + PREFIX_MS
    time = online["logtime"]
    indices = np.searchsorted(time, stamps, side="right") - 1
    fresh = (indices >= 0) & (stamps - time[indices] <= POSE_LAG_MS)
    points = online["logposition"][np.unique(indices[fresh])] - corner
    along = points @ tangent
    across = points[(-length <= along) & (along <= -MIN_PROGRESS_M)] @ normal
    if len(across) < PREFIX_MIN_SAMPLES:
        return None, len(across)
    return -float(np.median(across)) * normal, len(across)


def summarize(rows):
    counts = {}
    for arm in ("baseline", "corrected"):
        counts[arm] = {
            "correct": sum(row[f"{arm}_outcome"] == "correct" for row in rows),
            "early": sum(row[f"{arm}_outcome"] == "early" for row in rows),
            "unlabeled": sum(row[f"{arm}_outcome"] == "unlabeled" for row in rows),
            "silent": sum(row[f"{arm}_outcome"] == "silent" for row in rows),
            "silent_with_progress": sum(
                row[f"{arm}_outcome"] == "silent" and row["opportunities"] > 0 for row in rows
            ),
        }
    common = [
        row["corrected_s"] - row["baseline_s"] for row in rows
        if row["corrected_outcome"] == row["baseline_outcome"] == "correct"
    ]
    return {
        "turns": len(rows),
        "used_prefix": sum(row["shift_m"] is not None for row in rows),
        **counts,
        "common_correct": len(common),
        "median_common_delay_s": median(common) if common else None,
        "median_baseline_prompt_s": median([
            row["baseline_s"] for row in rows if row["baseline_s"] is not None
        ]) if counts["baseline"]["silent"] < len(rows) else None,
        "median_comparison_prompt_s": median([
            row["corrected_s"] for row in rows if row["corrected_s"] is not None
        ]) if counts["corrected"]["silent"] < len(rows) else None,
        "gained_correct": sum(
            row["baseline_outcome"] != "correct" and row["corrected_outcome"] == "correct" for row in rows
        ),
        "lost_correct": sum(
            row["baseline_outcome"] == "correct" and row["corrected_outcome"] != "correct" for row in rows
        ),
        "rescued_original_misses": [
            [row["person"], row["number"]] for row in rows
            if row["baseline_outcome"] == "silent" and row["opportunities"] > 0
            and row["corrected_outcome"] == "correct"
        ],
        "transitions": dict(Counter(f"{row['baseline_outcome']} → {row['corrected_outcome']}" for row in rows)),
    }


def outcome(event):
    if event is None:
        return "silent", None
    return {True: "correct", False: "early", None: "unlabeled"}[event["correct"]], event["time_s"]


def summarize_alternative(rows, name):
    return summarize([
        {**row, "corrected_outcome": row[f"{name}_outcome"], "corrected_s": row[f"{name}_s"]}
        for row in rows
    ])


def original_forward(clip, online, video, vertices, slots):
    """Score the first logged proceed/forward instruction at issuance and on the replay grid."""
    instructions = online["instructions"]
    cue = clip["cue_start_ms"]
    stop = status_window_end(clip, online)
    for stamp, command in zip(instructions["timestart"], instructions["command"], strict=True):
        if cue < stamp < stop and "Forward" in command.split():
            break
    else:
        return None, None, None
    origin, end = vertices[clip["corner_index"]:clip["corner_index"] + 2]
    direction = end - origin
    length = float(np.linalg.norm(direction))
    route, floor = clip["route"], clip["floor"]
    previous = annotated_position(video, stamp - WALK_HISTORY_MS, route, floor)
    current = annotated_position(video, stamp, route, floor)
    truth = None if previous is None or current is None else leg_clearance(
        projection(previous, current, origin, direction / length), length, CENTERLINE_M,
    ) >= 0
    when = (int(stamp) - cue) / 1000
    nearest_grid = next((slot for slot in slots if slot["time_s"] >= when), None)
    return {"time_s": when, "correct": truth}, nearest_grid, str(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True, help="Existing Mobilio/PING extension payload directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = args.prior / "sources/ping"
    paths = {
        "source": inputs / "data.mat",
        "clips": inputs / "turns.json",
        "mask": inputs / "annotation_pose_rows.json",
        "baseline": args.prior / "ping-progress/status_replay.json",
    }
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    clips = json.loads(paths["clips"].read_bytes())
    masks = json.loads(paths["mask"].read_bytes())
    if (
        source_hashes["source"] != DATA_SHA256
        or clips["receipt"]["sha256"] != DATA_SHA256 or masks["ping_mat_sha256"] != DATA_SHA256
    ):
        raise ValueError("Mismatched PING source identities")

    source = loadmat(paths["source"], simplify_cells=True, variable_names=["contextdata", "trajdata"])
    people = {}
    videos = {}
    for index, participant in enumerate(source["trajdata"], 1):
        person = f"P{index}"
        people[person] = deduplicate(participant)
        video = dict(participant)
        indices = masks["people"][person]["unambiguous_pose_row_indices"]
        for field in ("gttime", "gtpos", "gtroute", "gtfloor"):
            video[field] = participant[field][indices]
        videos[person] = deduplicate(video)
    maps = {int(floor["floorid"]): route_vertices(floor["plannedpath"]) for floor in source["contextdata"]}
    routes = {key: maps[key[1]][::-1] if reverse else maps[key[1]] for key, reverse in REVERSE.items()}
    prior = json.loads(paths["baseline"].read_bytes())
    expected = {(trial["person"], trial["number"]): trial for trial in prior["trials"]}
    alternatives = (
        ("both_positions", lambda at: min(at["raw_margin_m"], at["shifted_margin_m"]) >= 0),
        ("position_0_5m", lambda at: at["raw_margin_m"] >= 0.5),
        ("position_1m", lambda at: at["raw_margin_m"] >= 1.0),
    )

    rows = []
    for clip in clips["clips"]:
        person, number = clip["person"], clip["number"]
        route, floor = clip["route"], clip["floor"]
        data, video = people[person], videos[person]
        vertices = routes[route, floor]
        calibration = clips["receipt"]["people"][person]["calibration"]
        shift, support = estimate_shift(clip, data, vertices)
        baseline = status_events(clip, data, video, vertices, calibration)
        reference = expected[person, number]
        if any(baseline[key] != reference[key] for key in baseline):
            raise ValueError(f"Baseline did not reproduce the saved trace: {person}/{number}")
        candidate = status_events(
            clip, data, video, vertices, calibration, position_shift=shift, position_perturbation=True,
        )
        if baseline["slots"] != candidate["slots"]:
            raise ValueError(f"Candidate changed the evaluation window: {person}/{number}")
        base_outcome, base_time = outcome(baseline["outputs"][RULE])
        candidate_outcome, candidate_time = outcome(candidate["outputs"][RULE])
        forward_exact, forward_grid, forward_text = original_forward(clip, data, video, vertices, baseline["slots"])
        row = {
            "person": person,
            "number": number,
            "route": route,
            "floor": floor,
            "corner": clip["corner_index"],
            "prefix_samples": support,
            "shift_m": float(np.linalg.norm(shift)) if shift is not None else None,
            "shift_x_m": float(shift[0]) if shift is not None else None,
            "shift_y_m": float(shift[1]) if shift is not None else None,
            "opportunities": baseline["progress_opportunities"],
            "baseline_outcome": base_outcome,
            "baseline_s": base_time,
            "corrected_outcome": candidate_outcome,
            "corrected_s": candidate_time,
            "forward_command": forward_text,
        }
        for name, satisfies_position in alternatives:
            prompt = next((
                at["slot"] for at in candidate["placement"] if at["angle_and_advance"] and satisfies_position(at)
            ), None)
            row[f"{name}_outcome"], row[f"{name}_s"] = outcome(prompt)
        row["forward_exact_outcome"], row["forward_exact_s"] = outcome(forward_exact)
        row["forward_grid_outcome"], row["forward_grid_s"] = outcome(forward_grid)
        rows.append(row)
    if {(row["person"], row["number"]) for row in rows} != set(expected) or len(rows) != len(expected):
        raise ValueError("Candidate replay does not cover the exact saved baseline turns")
    report = {
        "sources_sha256": source_hashes,
        "all": summarize(rows),
        "used_prefix": summarize([row for row in rows if row["shift_m"] is not None]),
        "position_sensitivity": {
            name: summarize_alternative(rows, name) for name, _ in alternatives
        },
        "logged_forward_control": {
            name: summarize_alternative(rows, f"forward_{name}") for name in ("exact", "grid")
        },
        "correct_confirmation_after_logged_forward": {
            "turns_with_both": sum(
                row["forward_exact_s"] is not None and row["baseline_outcome"] == "correct"
                for row in rows
            ),
            "forward_started_first": sum(
                row["forward_exact_s"] is not None and row["baseline_outcome"] == "correct"
                and row["forward_exact_s"] < row["baseline_s"] for row in rows
            ),
            "forward_not_yet_satisfying_progress": sum(
                row["forward_exact_outcome"] == "early" and row["baseline_outcome"] == "correct"
                and row["forward_exact_s"] < row["baseline_s"] for row in rows
            ),
        },
        "by_person": {person: summarize([row for row in rows if row["person"] == person]) for person in people},
    }
    write_json(args.output_dir / "summary.json", report)
    with (args.output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
