"""Re-score saved PING prompt times after joining collinear map legs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mobilio_turn_confirmation import write_json
from ping_incoming_line_replay import outcome
from ping_position_transfer import CONTINUATION_RESCUE, continues_straight, digest, metrics_by_outcome
from ping_turn_progress import CENTERLINE_M, WALK_HISTORY_MS, annotated_position, leg_clearance, projection
from prepare_ping_turns import DATA_SHA256, REVERSE, route_vertices


ARMS = ("raw", CONTINUATION_RESCUE)


def read_json(path):
    return json.loads(path.read_bytes())


def merged_endpoint(vertices, corner, tangent):
    end = corner + 1
    while continues_straight(vertices, end - 1, tangent):
        end += 1
    return end


def score_slots(clip, slots, video, vertices):
    corner = clip["corner_index"]
    origin, end = vertices[corner:corner + 2]
    length = float(np.linalg.norm(end - origin))
    tangent = (end - origin) / length
    last = merged_endpoint(vertices, corner, tangent)
    merged_length = length if last == corner + 1 else float((vertices[last] - origin) @ tangent)
    original, merged = [], []
    for slot in slots:
        seconds = slot["time_s"]
        stamp = clip["cue_start_ms"] + round(seconds * 1000)
        before = annotated_position(video, stamp - WALK_HISTORY_MS, clip["route"], clip["floor"])
        after = annotated_position(video, stamp, clip["route"], clip["floor"])
        old_label = new_label = None
        if before is not None and after is not None:
            progress = projection(before, after, origin, tangent)
            old_label = leg_clearance(progress, length, CENTERLINE_M) >= 0
            new_label = leg_clearance(progress, merged_length, CENTERLINE_M) >= 0
        original.append({"time_s": seconds, "correct": old_label})
        merged.append({"time_s": seconds, "correct": new_label})
    if original != slots:
        raise ValueError(f"Original labels disagree for {clip['person']}#{clip['number']}")
    return length, merged_length, last - corner, merged


def score_cohort(name, clips, original_status, prompts, videos, maps):
    statuses = {(row["person"], row["number"]): row for row in original_status}
    records = []
    extended_turns = 0
    for clip in clips:
        person, number = clip["person"], clip["number"]
        ident = f"{person}#{number}"
        previous = statuses.pop((person, number))
        floor, route = clip["floor"], clip["route"]
        vertices = maps[floor][::-1] if REVERSE[route, floor] else maps[floor]
        length, merged_length, segments, slots = score_slots(clip, previous["slots"], videos[person], vertices)
        extended_turns += segments > 1
        old_slots = {slot["time_s"]: slot for slot in previous["slots"]}
        new_slots = {slot["time_s"]: slot for slot in slots}
        opportunities = sum(slot["correct"] is True for slot in slots)
        for arm in ARMS:
            saved = prompts.pop((name, ident, arm))
            issue = float(saved["issue_s"]) if saved["issue_s"] else None
            old_label, _ = outcome(old_slots[issue] if issue is not None else None)
            if old_label != saved["outcome"] or int(saved["opportunities"]) != previous["progress_opportunities"]:
                raise ValueError(f"Saved decision or coverage disagrees for {name} {ident} {arm}")
            label, _ = outcome(new_slots[issue] if issue is not None else None)
            records.append({
                "cohort": name,
                "id": ident,
                "person": person,
                "arm": arm,
                "issue_s": issue,
                "original_outcome": old_label,
                "outcome": label,
                "original_opportunities": previous["progress_opportunities"],
                "opportunities": opportunities,
                "original_leg_m": length,
                "merged_leg_m": merged_length,
                "merged_segments": segments,
            })
    if statuses:
        raise ValueError(f"Original status contains extra {name} trials")

    original_rows = [dict(row, outcome=row["original_outcome"], opportunities=row["original_opportunities"])
                     for row in records]
    raw = {row["id"]: row for row in records if row["arm"] == "raw"}
    shared_early = [row for row in records if row["arm"] == CONTINUATION_RESCUE
                    and row["outcome"] == raw[row["id"]]["outcome"] == "early"]
    return {
        "turns": len(clips),
        "extended_turns": extended_turns,
        "original": metrics_by_outcome(original_rows, ARMS),
        "merged": metrics_by_outcome(records, ARMS),
        "shared_early": len(shared_early),
        "shared_early_same_time": sum(row["issue_s"] == raw[row["id"]]["issue_s"] for row in shared_early),
    }, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--minor", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True, help="Saved continuation replay directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ping = args.prior / "sources/ping"
    paths = {
        "data": ping / "data.mat",
        "major": ping / "turns.json",
        "mask": ping / "annotation_pose_rows.json",
        "previous_major": args.prior / "ping-progress/status_replay.json",
        "minor": args.minor,
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    prior = read_json(args.replay / "summary.json")
    if hashes["data"] != DATA_SHA256 or any(prior["input_sha256"][key] != value for key, value in hashes.items()):
        raise ValueError("Rescoring inputs differ from the saved replay")
    saved_prompts = args.replay / "paired.csv"
    with saved_prompts.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["arm"] in ARMS]
    prompts = {(row["cohort"], row["id"], row["arm"]): row for row in rows}
    if len(prompts) != len(rows):
        raise ValueError("Duplicate saved prompt")

    data = loadmat(paths["data"], simplify_cells=True, variable_names=["contextdata", "trajdata"])
    maps = {int(item["floorid"]): route_vertices(item["plannedpath"]) for item in data["contextdata"]}
    mask = read_json(paths["mask"])["people"]
    videos = {}
    for i, participant in enumerate(data["trajdata"], 1):
        person = f"P{i}"
        take = mask[person]["unambiguous_pose_row_indices"]
        videos[person] = {key: participant[key][take] for key in ("gttime", "gtpos", "gtroute", "gtfloor")}
    major = read_json(paths["major"])
    minor = read_json(paths["minor"])
    cohorts = (("major", major["clips"], read_json(paths["previous_major"])["trials"]),
               ("minor", minor["clips"], minor["status"]))
    result = {"input_sha256": {**hashes, "saved_prompts": digest(saved_prompts)}}
    records = []
    for name, clips, status in cohorts:
        stats, scored = score_cohort(name, clips, status, prompts, videos, maps)
        expected = {arm: prior[name]["prompt_comparison"][arm] for arm in ARMS}
        if stats["original"] != expected:
            raise ValueError(f"Original outcome summary disagrees for {name}")
        result[name] = stats
        records.extend(scored)
    if prompts:
        raise ValueError("Saved prompts contain extra trials")
    write_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for name, _, _ in cohorts:
        print(name, json.dumps(result[name]))


if __name__ == "__main__":
    main()
