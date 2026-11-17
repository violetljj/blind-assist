"""Replay untouched PING diagonal turn events on the original cohort and time base."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from scipy.io import loadmat

from mobilio_turn_confirmation import write_json
from ping_incoming_line_replay import original_forward, outcome
from ping_turn_progress import compare_progress_to_wait, status_events, status_window_end, summarize_status
from prepare_ping_turns import (
    DATA_SHA256,
    REVERSE,
    SOURCE,
    deduplicate,
    prepare_person,
    route_vertices,
    turn_corners,
)


MINOR_CUES = {
    "TurnDiagonalLeft": 1,
    "TurnDiagonalRight": -1,
    "TurnLittleLeft": 1,
    "TurnLittleRight": -1,
}
SPEECH_ALIASES = {"Misc": "Approaching", "Forward TurRight": "Forward TurnRight"}
ENVFACTORS_SOURCE = "https://envfactors.github.io/code/CHI_code.zip"


def audit_speech(archive_path, participants):
    speeches = defaultdict(list)
    with ZipFile(archive_path) as archive:
        with archive.open("datasheet/trajectoryData.csv") as source:
            for row in csv.DictReader(TextIOWrapper(source, encoding="utf-8-sig")):
                if row["code"] == "speak":
                    speeches[row["id2"]].append((row["time2"], row["navigation_code"]))

    differences = []
    aliases = Counter()
    for person, participant in participants.items():
        log = participant["instructions"]
        for (stamp, category), milliseconds, command in zip(
            speeches[person], log["timestart"], log["command"], strict=True,
        ):
            if category != command:
                if SPEECH_ALIASES.get(category) != command:
                    raise ValueError(f"Unexpected EnvFactors and PING event labels for {person}: {category}/{command}")
                aliases[f"{category} → {command}"] += 1
            # Hourless and hour-prefixed representations both use minute:second suffixes.
            minutes, seconds = map(float, stamp.split(":")[-2:])
            csv_seconds = 60 * minutes + seconds
            differences.append((int(milliseconds) / 1000 - csv_seconds + 1800) % 3600 - 1800)
    return {
        "source_url": ENVFACTORS_SOURCE,
        "matched_logged_speech_records": len(differences),
        "exact_command_matches": len(differences) - sum(aliases.values()),
        "aliases": dict(aliases),
        "maximum_absolute_offset_s": max(map(abs, differences)),
        "time_basis": "Match ordered log records per participant. CSV times are rounded to a tenth of a second.",
    }


def run_minor_turns(participants, maps, previous, videos):
    routes = {key: maps[key[1]][::-1] if reverse else maps[key[1]] for key, reverse in REVERSE.items()}
    corner_sets = {
        key: [
            corner for corner in turn_corners(route, min_angle=22.5, max_angle=60.0)
            if abs(corner.turn_deg) > 22.5
        ]
        for key, route in routes.items()
    }
    prior_intervals = defaultdict(list)
    prior_corners = set()
    for clip in previous["clips"]:
        person = clip["person"]
        prior_intervals[person].append((clip["cue_start_ms"], status_window_end(clip, participants[person])))
        prior_corners.add((person, clip["route"], clip["floor"], clip["corner_index"]))

    selected = []
    rows = []
    status_records = []
    rejections = []
    for person, online in participants.items():
        calibration, clips, rejected = prepare_person(
            person, online, routes, corner_sets, cue_directions=MINOR_CUES,
        )
        if calibration != previous["receipt"]["people"][person]["calibration"]:
            raise ValueError(f"Heading registration changed for {person}")
        video = videos[person]
        rejections.extend(rejected)
        for clip in clips:
            cue = clip["cue_start_ms"]
            end = status_window_end(clip, online)
            key = (clip["route"], clip["floor"])
            reason = None
            if (person, *key, clip["corner_index"]) in prior_corners:
                reason = "same_corner_as_previously_used_turn"
            elif any(cue < old_end and old_start < end for old_start, old_end in prior_intervals[person]):
                reason = "overlapping_evaluation_interval"
            if reason:
                rejections.append({
                    "person": person,
                    "cue_index": clip["number"] - 1,
                    "cue": clip["cue"],
                    "timestart_ms": cue,
                    "reason": reason,
                })
                continue

            vertices = routes[key]
            status = status_events(clip, online, video, vertices, calibration)
            exact, grid, command = original_forward(clip, online, video, vertices, status["slots"])
            row = {
                "person": person,
                "number": clip["number"],
                "route": clip["route"],
                "floor": clip["floor"],
                "corner": clip["corner_index"],
                "turn_cue": clip["cue"],
                "logged_cue_start_ms": cue,
                "exclusive_window_end_ms": end,
                "forward_command": command,
                "opportunities": status["progress_opportunities"],
            }
            for arm, event in (
                ("baseline", status["outputs"]["angle_progress"]),
                ("wait_6s", status["outputs"]["after_6s"]),
                ("angle_position", status["outputs"]["angle_position"]),
                ("forward_exact", exact),
                ("forward_grid", grid),
            ):
                row[f"{arm}_outcome"], row[f"{arm}_s"] = outcome(event)
            rows.append(row)
            selected.append(clip)
            status_records.append({"person": person, "number": clip["number"], **status})
    return selected, rows, status_records, rejections


def summarize_forward(rows, timing):
    counts = Counter(row[f"forward_{timing}_outcome"] for row in rows)
    times = [row[f"forward_{timing}_s"] for row in rows if row[f"forward_{timing}_s"] is not None]
    return {
        "issued": len(times),
        "already_at_reference_progress": counts["correct"],
        "before_reference_progress": counts["early"],
        "unlabeled": counts["unlabeled"],
        "silent": counts["silent"],
        "median_issue_s": median(times) if times else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True, help="Existing Mobilio/PING payload directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.prior / "sources"
    ping = root / "ping"
    source_paths = {
        "mat": ping / "data.mat",
        "major_turn_manifest": ping / "turns.json",
        "pose_mask": ping / "annotation_pose_rows.json",
        "envfactors_zip": root / "envfactors/CHI_code.zip",
    }
    sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    major = json.loads(source_paths["major_turn_manifest"].read_bytes())
    mask = json.loads(source_paths["pose_mask"].read_bytes())
    if (
        sha256["mat"] != DATA_SHA256 or major["receipt"]["sha256"] != DATA_SHA256
        or mask["ping_mat_sha256"] != DATA_SHA256 or mask["ann_zip_sha256"] != sha256["envfactors_zip"]
    ):
        raise ValueError("Mismatched source, events, or independent position labels")

    data = loadmat(source_paths["mat"], simplify_cells=True, variable_names=["contextdata", "trajdata"])
    maps = {int(entry["floorid"]): route_vertices(entry["plannedpath"]) for entry in data["contextdata"]}
    participants = {}
    videos = {}
    for index, person in enumerate(data["trajdata"], 1):
        identity = f"P{index}"
        participants[identity] = deduplicate(person)
        indices = mask["people"][identity]["unambiguous_pose_row_indices"]
        videos[identity] = {field: person[field][indices] for field in ("gttime", "gtpos", "gtroute", "gtfloor")}

    selected, rows, status_records, excluded = run_minor_turns(participants, maps, major, videos)
    if not rows:
        raise ValueError("No new turn events with the selected geometry and source support")
    report = {
        "source_url": SOURCE,
        "source_sha256": sha256,
        "actual_app_logged_speech_only": True,
        "independent_users": False,
        "speech_alignment": audit_speech(source_paths["envfactors_zip"], participants),
        "candidate_events": len(excluded) + len(rows),
        "selected_by_cue": dict(Counter(clip["cue"] for clip in selected)),
        "reasons_excluded": dict(Counter(item["reason"] for item in excluded)),
        "events_with_labeled_device_slot": sum(
            any(slot["correct"] is not None for slot in item["slots"]) for item in status_records
        ),
        "status": summarize_status(status_records),
        "paired_against_6s_wait": compare_progress_to_wait(status_records),
        "logged_forward": {timing: summarize_forward(rows, timing) for timing in ("exact", "grid")},
        "by_person": {
            person: summarize_status([item for item in status_records if item["person"] == person])
            for person in participants
        },
    }
    write_json(args.output_dir / "summary.json", report)
    write_json(args.output_dir / "trials.json", {"clips": selected, "status": status_records, "excluded": excluded})
    with (args.output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "by_person"}, indent=2))


if __name__ == "__main__":
    main()
