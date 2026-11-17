"""Apply the externally calibrated position lead to fixed PING prompt replays."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.io import loadmat

from causal_position_compensation import compare_heading_lead, compensate
from evaluate_paired_turn_localization import metrics
from mobilio_turn_confirmation import write_json
from ping_incoming_line_replay import outcome
from ping_turn_progress import annotated_position, status_events, status_window_end
from prepare_ping_turns import DATA_SHA256, REVERSE, deduplicate, route_vertices
from turn_path_odometry import route_distance_corrections


RULE = "angle_progress"
ARMS = (
    "raw", "unguarded", "gated", "gated_with_heading_check", "heading_integration", "heading_along_only",
    "heading_aligned_history", "heading_along_aligned_history",
)
DWELL_RESCUE = "heading_dwell_rescue"
TIMED_RESCUE = "heading_exit_rescue"
CONTINUATION_RESCUE = "heading_straight_continuation_rescue"
SCORING_OUTCOMES = ("correct", "early", "unlabeled", "silent")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def continues_straight(vertices, turn, tangent):
    """Check whether the next mapped leg continues in the same direction."""
    if turn + 2 >= len(vertices):
        return False
    next_leg = vertices[turn + 2] - vertices[turn + 1]
    length = np.linalg.norm(next_leg)
    return length > 0 and np.allclose(next_leg / length, tangent, rtol=0, atol=1e-8)


def metrics_by_outcome(rows, arms):
    raw = {row["id"]: row for row in rows if row["arm"] == "raw"}
    outputs = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        counts = Counter(row["outcome"] for row in selected)
        compared = [(raw[row["id"]], row) for row in selected]
        common = [
            new["issue_s"] - old["issue_s"] for old, new in compared
            if old["outcome"] == new["outcome"] == "correct"
        ]
        misses = [row for row in selected if row["outcome"] == "silent" and row["opportunities"] > 0]
        outputs[arm] = {
            **{key: counts[key] for key in SCORING_OUTCOMES},
            "silent_with_progress": len(misses),
            "recovered_original_misses": [
                old["id"] for old, new in compared
                if old["outcome"] == "silent" and old["opportunities"] > 0 and new["outcome"] == "correct"
            ],
            "new_early": [
                old["id"] for old, new in compared if old["outcome"] != "early" and new["outcome"] == "early"
            ],
            "early_moved_earlier": [
                {"id": old["id"], "raw_issue_s": old["issue_s"], "issue_s": new["issue_s"]}
                for old, new in compared
                if old["outcome"] == new["outcome"] == "early" and new["issue_s"] < old["issue_s"]
            ],
            "removed_early": [
                old["id"] for old, new in compared if old["outcome"] == "early" and new["outcome"] != "early"
            ],
            "gained_correct": sum(old["outcome"] != "correct" and new["outcome"] == "correct" for old, new in compared),
            "lost_correct": sum(old["outcome"] == "correct" and new["outcome"] != "correct" for old, new in compared),
            "common_correct": len(common),
            "median_common_correct_delay_s": median(common) if common else None,
        }
    return outputs


def score_cohort(manifest, previous, people, video, maps, correction, *, arms, lead_s):
    baseline = {(row["person"], row["number"]): row for row in previous}
    cases = set()
    records = []
    # The rescue arm evaluates prompt eligibility only.
    rescue_arms = (DWELL_RESCUE, TIMED_RESCUE, CONTINUATION_RESCUE)
    pose_arms = [arm for arm in arms if arm not in rescue_arms]
    estimates = {arm: [] for arm in pose_arms}
    labels = []
    for clip in manifest:
        person = clip["person"]
        case = (person, clip["number"])
        if case in cases:
            raise ValueError(f"Duplicate PING turn: {case}")
        cases.add(case)
        route, floor = clip["route"], clip["floor"]
        vertices = maps[floor][::-1] if REVERSE[route, floor] else maps[floor]
        data, calibration = people[person]
        truth = video[person]
        baseline_status = status_events(clip, data, truth, vertices, calibration)
        if any(value != baseline[case][field] for field, value in baseline_status.items()):
            raise ValueError(f"Saved PING replay disagrees at {case}")
        corrections = dict(correction[person])
        turn = clip["corner_index"]
        tangent = vertices[turn + 1] - vertices[turn]
        tangent /= np.linalg.norm(tangent)
        corrections["heading_along_only"] = (corrections["heading_integration"] @ tangent)[:, None] * tangent
        corrections["heading_aligned_history"] = corrections["heading_integration"]
        corrections["heading_along_aligned_history"] = corrections["heading_along_only"]
        anchor_s = None
        if "observed_route_distance" in arms:
            times = (data["logtime"] - clip["cue_start_ms"]) / 1000
            stop = (status_window_end(clip, data) - clip["cue_start_ms"] - 1) / 1000
            position = data["logposition"]
            updated, anchor_s = route_distance_corrections(
                times, position, vertices[turn - 1:turn + 2], 0, stop,
            )
            odometry = np.zeros_like(position)
            for index, offset in updated:
                odometry[index] = offset
            corrections["observed_route_distance"] = odometry
        for slot in baseline_status["slots"]:
            stamp = clip["cue_start_ms"] + int(round(slot["time_s"] * 1000))
            label = annotated_position(truth, stamp, route, floor)
            if label is None:
                continue
            index = np.searchsorted(data["logtime"], stamp, side="right") - 1
            position = data["logposition"][index]
            labels.append(label)
            estimates["raw"].append(position)
            for arm in pose_arms[1:]:
                estimates[arm].append(position + corrections[arm][index])
        for arm in arms:
            status = baseline_status
            if arm != "raw":
                correction_arm = "heading_integration" if arm in rescue_arms else arm
                rescue = arm
                if arm == CONTINUATION_RESCUE:
                    rescue = DWELL_RESCUE if continues_straight(vertices, turn, tangent) else TIMED_RESCUE
                status = status_events(
                    clip, data, truth, vertices, calibration, position_correction=corrections[correction_arm],
                    correct_displacement=arm.endswith("_aligned_history"),
                    require_dwell_for_rescue=rescue == DWELL_RESCUE,
                    heading_exit_horizon_ms=1000 * lead_s if rescue == TIMED_RESCUE else None,
                )
                if status["slots"] != baseline_status["slots"]:
                    raise ValueError(f"Correction changed reference scoring slots at {case}")
            label, issue = outcome(status["outputs"][RULE])
            records.append({
                "id": f"{person}#{clip['number']}",
                "person": person,
                "arm": arm,
                "outcome": label,
                "issue_s": issue,
                "opportunities": status["progress_opportunities"],
                "odometry_anchor_s": anchor_s,
            })
    if cases != set(baseline):
        raise ValueError("PING replay cohort differs from its saved baseline")
    reference = np.asarray(labels).reshape(-1, 2)
    raw = np.asarray(estimates["raw"]).reshape(-1, 2)
    return {
        "turns": len(cases),
        "prompt_comparison": metrics_by_outcome(records, arms),
        "position_at_all_labeled_slots": {
            arm: metrics(np.asarray(points).reshape(-1, 2), reference, raw) for arm, points in estimates.items()
        },
        "by_person": {
            p: metrics_by_outcome([row for row in records if row["person"] == p], arms) for p in people
        },
    }, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--minor", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--odometry", action="store_true", help="Also replay measured distance from observed turns.")
    mode.add_argument("--dwell-output-dir", type=Path, help="Compare raw, heading integration and dwell-gated rescue.")
    mode.add_argument("--timing-output-dir", type=Path, help="Also test rescue when the heading window is closing.")
    mode.add_argument("--continuation-output-dir", type=Path, help="Also test rescue at straight-through waypoints.")
    args = parser.parse_args()
    ping = args.prior / "sources/ping"
    paths = {
        "data": ping / "data.mat",
        "major": ping / "turns.json",
        "mask": ping / "annotation_pose_rows.json",
        "previous_major": args.prior / "ping-progress/status_replay.json",
        "minor": args.minor,
        "calibration": args.run_dir / "external-replay/localization.json",
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    if hashes["data"] != DATA_SHA256:
        raise ValueError("Unexpected PING source")
    major = json.loads(paths["major"].read_bytes())
    mask = json.loads(paths["mask"].read_bytes())
    if major["receipt"]["sha256"] != DATA_SHA256 or mask["ping_mat_sha256"] != DATA_SHA256:
        raise ValueError("Mismatched PING clips or position labels")
    minor = json.loads(paths["minor"].read_bytes())
    fixed_lead_s = json.loads(paths["calibration"].read_bytes())["fixed_lead_s"]
    data = loadmat(paths["data"], simplify_cells=True, variable_names=["contextdata", "trajdata"])
    maps = {int(item["floorid"]): route_vertices(item["plannedpath"]) for item in data["contextdata"]}
    people = {}
    videos = {}
    correction = {}
    for i, participant in enumerate(data["trajdata"], 1):
        person = f"P{i}"
        online = deduplicate(participant)
        people[person] = online, major["receipt"]["people"][person]["calibration"]
        time_s = (online["logtime"] - online["logtime"][0]) / 1000
        gated, unguarded = compensate(time_s, online["logposition"], fixed_lead_s)
        yaw_deg = people[person][1]["offset_deg"] - online["logorientation"]
        same_gate, heading = compare_heading_lead(time_s, yaw_deg, gated, fixed_lead_s)
        correction[person] = {
            "gated": gated, "unguarded": unguarded, "gated_with_heading_check": same_gate,
            "heading_integration": heading,
        }
        take = mask["people"][person]["unambiguous_pose_row_indices"]
        videos[person] = {name: participant[name][take] for name in ("gttime", "gtpos", "gtroute", "gtfloor")}
    previous = json.loads(paths["previous_major"].read_bytes())["trials"]
    arms = ARMS + ("observed_route_distance",) if args.odometry else ARMS
    if args.dwell_output_dir is not None:
        arms = ("raw", "heading_integration", DWELL_RESCUE)
    if args.timing_output_dir is not None or args.continuation_output_dir is not None:
        arms = ("raw", "heading_integration", DWELL_RESCUE, TIMED_RESCUE)
    if args.continuation_output_dir is not None:
        arms += (CONTINUATION_RESCUE,)
    main, rows_main = score_cohort(
        major["clips"], previous, people, videos, maps, correction, arms=arms, lead_s=fixed_lead_s,
    )
    diagonal, rows_diag = score_cohort(
        minor["clips"], minor["status"], people, videos, maps, correction, arms=arms, lead_s=fixed_lead_s,
    )
    result = {"input_sha256": hashes, "lead_s": fixed_lead_s, "major": main, "minor": diagonal}
    output = args.continuation_output_dir or args.timing_output_dir or args.dwell_output_dir or args.run_dir / "ping-transfer"
    write_json(output / "summary.json", result)
    with (output / "paired.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["cohort", *rows_main[0]])
        writer.writeheader()
        for cohort, rows in (("major", rows_main), ("minor", rows_diag)):
            writer.writerows(dict(cohort=cohort, **row) for row in rows)
    for name, arm in (("major", main), ("minor", diagonal)):
        print(name, json.dumps(arm["prompt_comparison"]))


if __name__ == "__main__":
    main()
