"""Replay the fixed PING rules on clock-aligned poses, reference positions and recorded cue starts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from causal_position_compensation import compare_heading_lead, compensate
from mobilio_turn_confirmation import write_json
from ping_collinear_rescore import score_slots
from ping_incoming_line_replay import outcome
from ping_position_transfer import CONTINUATION_RESCUE, continues_straight, digest, metrics_by_outcome
from ping_turn_progress import MAX_STATUS_MS, status_events


LEAD_S = 0.75
ARMS = ("raw", CONTINUATION_RESCUE)
DEVICE_COLUMNS = ("time_ms", "x_m", "y_m", "yaw_deg")
REFERENCE_COLUMNS = ("time_ms", "x_m", "y_m", "route", "floor")


def read_numeric(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if tuple(stream.readline().strip().split(",")) != columns:
            raise ValueError(f"Unexpected columns in {path}")
        rows = np.loadtxt(stream, delimiter=",", ndmin=2)
    if not len(rows) or rows.shape[1] != len(columns) or not np.isfinite(rows).all():
        raise ValueError(f"Invalid numeric rows in {path}")
    times = rows[:, 0]
    if np.any(times != np.rint(times)) or np.any(np.diff(times) < 0):
        raise ValueError(f"Expected nondecreasing integer milliseconds in {path}")
    return rows


def read_session(folder, session):
    device_path = folder / session["device"]
    truth_path = folder / session["reference"]
    native = read_numeric(device_path, DEVICE_COLUMNS)
    # Use the last value received for a repeated timestamp, as in the PING input.
    device = native[np.r_[np.diff(native[:, 0]) > 0, True]]
    ref = read_numeric(truth_path, REFERENCE_COLUMNS)
    if np.any(ref[:, 3:] != np.rint(ref[:, 3:])):
        raise ValueError(f"Route and floor must be integer IDs in {truth_path}")
    time_s = (device[:, 0] - device[0, 0]) / 1000
    gated, _ = compensate(time_s, device[:, 1:3], LEAD_S)
    _, correction = compare_heading_lead(time_s, device[:, 3], gated, LEAD_S)
    online = {"logtime": device[:, 0].astype(np.int64), "logposition": device[:, 1:3], "logorientation": -device[:, 3]}
    video = {
        "gttime": ref[:, 0].astype(np.int64), "gtpos": ref[:, 1:3], "gtroute": ref[:, 3].astype(np.int64),
        "gtfloor": ref[:, 4].astype(np.int64),
    }
    receipt = {
        "device_rows": len(device), "device_duplicate_rows": len(native) - len(device),
        "reference_rows": len(ref), "device_sha256": digest(device_path), "reference_sha256": digest(truth_path),
        "median_native_interval_ms": float(np.median(np.diff(device[:, 0]))) if len(device) > 1 else None,
        "native_rows_with_heading_compensation": int(np.count_nonzero(np.any(correction != 0, axis=1))),
    }
    return online, video, correction, receipt


def replay_turn(turn, vertices, pose, reference, correction):
    corner = turn["corner_index"]
    if not 0 < corner < len(vertices) - 1:
        raise ValueError(f"Turn {turn['id']} requires incoming and outgoing map segments")
    cue = turn["cue_start_ms"]
    start, end = vertices[corner:corner + 2]
    length = float(np.linalg.norm(end - start))
    if length <= 0:
        raise ValueError(f"Empty outgoing map segment for {turn['id']}")
    tangent = (end - start) / length
    clip = {**turn, "target_heading_deg": float(np.degrees(np.arctan2(tangent[1], tangent[0])))}
    stop = turn.get("stop_ms", cue + MAX_STATUS_MS + 1)
    if not isinstance(stop, int) or stop <= cue:
        raise ValueError(f"Invalid observed window end for {turn['id']}")
    calibration = {"offset_deg": 0.0}
    straight = continues_straight(vertices, corner, tangent)
    raw = status_events(clip, pose, reference, vertices, calibration, window_end_ms=stop)
    fixed = status_events(
        clip, pose, reference, vertices, calibration, position_correction=correction, window_end_ms=stop,
        require_dwell_for_rescue=straight, heading_exit_horizon_ms=None if straight else 1000 * LEAD_S,
        position_perturbation=True,
    )
    if raw["slots"] != fixed["slots"]:
        raise ValueError("Raw and compensated rules must see the same scored times")
    length, merged_length, _, merged = score_slots(clip, raw["slots"], reference, vertices)
    original_slots = {event["time_s"]: event for event in raw["slots"]}
    merged_slots = {event["time_s"]: event for event in merged}
    records = []
    for arm, status in zip(ARMS, (raw, fixed), strict=True):
        first = status["outputs"]["angle_progress"]
        issue = first["time_s"] if first is not None else None
        original, _ = outcome(original_slots[issue] if issue is not None else None)
        final, _ = outcome(merged_slots[issue] if issue is not None else None)
        records.append({
            "id": turn["id"], "session": turn["session"], "group": turn["group"], "arm": arm,
            "cue_start_ms": cue, "issue_s": issue, "outcome": final, "original_outcome": original,
            "original_opportunities": raw["progress_opportunities"],
            "opportunities": sum(event["correct"] is True for event in merged),
            "scoring_slots": len(merged), "reference_slots": sum(event["correct"] is not None for event in merged),
            "original_leg_m": length, "merged_leg_m": merged_length,
            "straight_continuation": bool(straight),
            "position_margin_changed_slots": sum(
                point["raw_margin_m"] != point["shifted_margin_m"] for point in status.get("placement", ())
            ),
        })
    return records


def replay(manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    maps = {key: np.asarray(value, dtype=float) for key, value in manifest["maps"].items()}
    for key, vertices in maps.items():
        if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
            raise ValueError(f"Expected a finite XY map for {key}")
    sessions = {name: read_session(manifest_path.parent, spec) for name, spec in manifest["sessions"].items()}
    records = []
    seen = set()
    for turn in manifest["turns"]:
        case = (turn["group"], turn["id"])
        if case in seen or not isinstance(turn["cue_start_ms"], int):
            raise ValueError(f"Duplicate turn or invalid recorded cue time: {case}")
        seen.add(case)
        pose, reference, correction, _ = sessions[turn["session"]]
        records.extend(replay_turn(turn, maps[turn["map"]], pose, reference, correction))
    if not records:
        raise ValueError("No recorded cues to replay")
    summary = {
        "source": manifest["source"], "lead_s": LEAD_S, "manifest_sha256": digest(manifest_path),
        "sessions": {name: values[-1] for name, values in sessions.items()}, "groups": {},
    }
    for group in sorted({turn[0] for turn in seen}):
        rows = [row for row in records if row["group"] == group]
        raw_rows = [row for row in rows if row["arm"] == "raw"]
        summary["groups"][group] = {
            "turns": len(raw_rows), "scoring_slots": sum(row["scoring_slots"] for row in raw_rows),
            "reference_slots": sum(row["reference_slots"] for row in raw_rows),
            "turns_with_scoring_slots": sum(row["scoring_slots"] > 0 for row in raw_rows),
            "turns_with_reference_progress": sum(row["opportunities"] > 0 for row in raw_rows),
            "straight_continuation_turns": sum(row["straight_continuation"] for row in raw_rows),
            "prompts": metrics_by_outcome(rows, ARMS),
        }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, records = replay(args.manifest)
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for name, result in summary["groups"].items():
        print(name, json.dumps(result))


if __name__ == "__main__":
    main()
