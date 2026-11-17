"""Transfer the fixed PING confirmation rules to held-out ADVIO map-point turns."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from causal_position_compensation import compare_heading_lead, compensate
from mobilio_turn_confirmation import write_json
from ping_collinear_rescore import merged_endpoint, score_slots
from ping_incoming_line_replay import outcome
from ping_position_transfer import CONTINUATION_RESCUE, continues_straight, digest, metrics_by_outcome
from ping_turn_progress import WALK_HISTORY_MS, status_events


SEQUENCES = (1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19)
CALIBRATION_END_S = 25
MIN_CORNER_TIME_S = 32
MIN_LEG_M = 2.5
LEAD_S = 0.75
ARMS = ("raw", CONTINUATION_RESCUE)


def load_traces(device_path, reference_path):
    device = np.loadtxt(device_path, delimiter=",", ndmin=2)
    reference = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    if device.shape[1] != 8 or reference.shape[1] != 8:
        raise ValueError("Expected ADVIO timestamp, position and quaternion")
    if not (np.isfinite(device).all() and np.isfinite(reference).all()):
        raise ValueError("Unexpected non-finite pose data")
    if np.any(np.diff(device[:, 0]) < 0) or np.any(np.diff(reference[:, 0]) <= 0):
        raise ValueError("Unexpected pose timestamp order")

    ref_s = reference[:, 0]
    ref_xy = reference[:, [3, 1]] * [1, -1]
    paired = device[(device[:, 0] >= ref_s[0]) & (device[:, 0] <= ref_s[-1])]
    source_xy = paired[:, [1, 3]]
    prefix = paired[:, 0] <= CALIBRATION_END_S
    anchor = source_xy[prefix]
    target = np.column_stack([np.interp(paired[prefix, 0], ref_s, axis) for axis in ref_xy.T])
    if len(anchor) < 2 or np.linalg.norm(np.ptp(anchor, axis=0)) < MIN_LEG_M:
        raise ValueError("Insufficient movement in the fixed calibration prefix")
    left, _, right = np.linalg.svd((anchor - anchor.mean(axis=0)).T @ (target - target.mean(axis=0)))
    rotation = left @ np.diag([1, np.sign(np.linalg.det(left @ right))]) @ right
    translation = target.mean(axis=0) - anchor.mean(axis=0) @ rotation
    dev_xy = source_xy @ rotation + translation
    optical = Rotation.from_quat(paired[:, 4:8], scalar_first=True).apply([0, 0, -1])[:, [0, 2]] @ rotation
    yaw = np.degrees(np.arctan2(optical[:, 1], optical[:, 0]))

    dev_ms = np.rint(paired[:, 0] * 1000).astype(np.int64)
    keep = np.r_[np.diff(dev_ms) > 0, True]
    dev_ms = dev_ms[keep]
    dev_xy = dev_xy[keep]
    yaw = yaw[keep]
    dev_s = (dev_ms - dev_ms[0]) / 1000
    gated, _ = compensate(dev_s, dev_xy, LEAD_S)
    _, heading = compare_heading_lead(dev_s, yaw, gated, LEAD_S)

    ref_ms = np.rint(ref_s * 1000).astype(np.int64)
    ref_keep = np.r_[np.diff(ref_ms) > 0, True]
    ref_ms = ref_ms[ref_keep]
    video = {
        "gttime": ref_ms, "gtpos": ref_xy[ref_keep],
        "gtroute": np.ones(len(ref_ms), dtype=int), "gtfloor": np.zeros(len(ref_ms), dtype=int),
    }
    online = {
        "logtime": dev_ms, "logposition": dev_xy, "logorientation": -yaw,
        "gttime": ref_ms, "gtroute": video["gtroute"], "gtfloor": video["gtfloor"],
    }
    fit_m = np.linalg.norm(anchor @ rotation + translation - target, axis=1)
    return online, video, heading, {
        "native_device_rows": len(device), "paired_device_rows": len(dev_ms), "reference_rows": len(ref_ms),
        "device_duplicate_ms_rows": int((~keep).sum()),
        "fit_prefix_rmse_m": float(np.sqrt(np.mean(fit_m ** 2))),
        "device_median_interval_ms": float(np.median(np.diff(dev_ms))),
    }


def map_turns(fixes, exclusions):
    """Use authored points for the route and anchor, without reading outcome positions."""
    for corner in range(1, len(fixes) - 1):
        points = fixes[corner - 1:corner + 2, :4]
        if points[1, 0] < MIN_CORNER_TIME_S:
            exclusions["calibration_or_history"] += 1
            continue
        if not np.isfinite(points).all() or not np.all(points[:, 3] == points[1, 3]):
            exclusions["different_or_unknown_level"] += 1
            continue
        legs = np.diff(points[:, 1:3], axis=0)
        length = np.linalg.norm(legs, axis=1)
        if np.any(length < MIN_LEG_M) or np.any(np.diff(points[:, 0]) <= 0):
            exclusions["short_or_unordered_edges"] += 1
            continue
        incoming, outgoing = legs / length[:, None]
        bend = float(np.degrees(np.arccos(np.clip(incoming @ outgoing, -1, 1))))
        if not 22.5 < bend <= 120:
            exclusions["outside_angle_range"] += 1
            continue

        end = corner + 2
        while end < len(fixes) and fixes[end, 3] == points[1, 3] and np.isfinite(fixes[end, :4]).all():
            end += 1
        vertices = fixes[corner - 1:end, 1:3]
        last = corner - 1 + merged_endpoint(vertices, 1, outgoing)
        corner_ms = round(points[1, 0] * 1000)
        stop_ms = round(fixes[last, 0] * 1000)
        yield corner, bend, corner_ms, stop_ms, vertices, outgoing


def replay_sequence(folder):
    reference = folder / "ground-truth/pose.csv"
    if not reference.is_file():
        reference = folder / "ground-truth/poses.csv"
    device = folder / "pixel/arcore.csv"
    fixpoints = folder / "ground-truth/fixpoints.csv"
    online, video, heading, receipt = load_traces(device, reference)
    fixes = np.loadtxt(fixpoints, delimiter=",", ndmin=2)
    calibration = {"offset_deg": 0.0}
    exclusions = Counter()
    records = []
    for corner, bend, corner_ms, stop_ms, vertices, tangent in map_turns(fixes, exclusions):
        clip = {
            "person": folder.name, "number": corner, "route": 1, "floor": 0, "corner_index": 1,
            "cue_start_ms": corner_ms - WALK_HISTORY_MS,
            "target_heading_deg": float(np.degrees(np.arctan2(tangent[1], tangent[0]))),
        }
        timed_online = {**online, "instructions": {"timestart": [stop_ms], "command": ["TurnLeft"]}}
        straight = continues_straight(vertices, 1, tangent)
        raw = status_events(clip, timed_online, video, vertices, calibration)
        candidate = status_events(
            clip, timed_online, video, vertices, calibration, position_correction=heading,
            require_dwell_for_rescue=straight, heading_exit_horizon_ms=None if straight else 1000 * LEAD_S,
            position_perturbation=True,
        )
        if raw["slots"] != candidate["slots"]:
            raise ValueError("Compensation changed the shared evaluation grid")
        length, merged_length, _, truth_slots = score_slots(clip, raw["slots"], video, vertices)
        truth_by_time = {slot["time_s"]: slot for slot in truth_slots}
        margin_changes = sum(slot["raw_margin_m"] != slot["shifted_margin_m"] for slot in candidate["placement"])
        added_eligible_slots = sum(
            slot["raw_margin_m"] < 0 <= slot["shifted_margin_m"] and slot["angle_and_advance"]
            for slot in candidate["placement"]
        )
        position_only_possible = any(
            slot["correct"] is True and estimate["angle_and_advance"]
            for slot, estimate in zip(raw["slots"], candidate["placement"], strict=True)
        )
        for arm, status in zip(ARMS, (raw, candidate), strict=True):
            prompt = status["outputs"]["angle_progress"]
            label, issue = outcome(truth_by_time[prompt["time_s"]] if prompt is not None else None)
            records.append({
                "id": f"{folder.name}#{corner}", "sequence": folder.name, "arm": arm,
                "group": "major" if bend >= 60 else "minor", "angle_deg": bend,
                "corner_s": corner_ms / 1000, "stop_s": stop_ms / 1000, "issue_s": issue, "outcome": label,
                "opportunities": sum(slot["correct"] is True for slot in truth_slots),
                "scoring_slots": len(truth_slots), "labeled_slots": sum(slot["correct"] is not None for slot in truth_slots),
                "original_leg_m": length, "merged_leg_m": merged_length,
                "straight_continuation": bool(straight),
                "position_margin_changed_slots": margin_changes if arm == CONTINUATION_RESCUE else 0,
                "additional_eligible_slots": added_eligible_slots if arm == CONTINUATION_RESCUE else 0,
                "position_only_possible": position_only_possible,
            })
    receipt["input_sha256"] = {str(p.relative_to(folder)): digest(p) for p in (device, reference, fixpoints)}
    receipt["excluded_map_points"] = dict(exclusions)
    return receipt, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advio", type=Path, required=True, help="Extracted author data folders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {"lead_s": LEAD_S, "sequences": {}, "protocol": "ADVIO annotated corner time minus two seconds"}
    records = []
    for i in SEQUENCES:
        folder = args.advio / f"advio-{i:02d}"
        receipt, rows = replay_sequence(folder)
        result["sequences"][folder.name] = receipt
        records.extend(rows)
    for name in ("major", "minor"):
        group = [row for row in records if row["group"] == name]
        sample = [row for row in group if row["arm"] == CONTINUATION_RESCUE]
        original_misses = [row for row in group if row["arm"] == "raw" and row["outcome"] == "silent"
                           and row["opportunities"] > 0]
        result[name] = {
            "turns": len(sample),
            "turns_with_labels": sum(row["labeled_slots"] > 0 for row in sample),
            "turns_with_progress": sum(row["opportunities"] > 0 for row in sample),
            "scoring_slots": sum(row["scoring_slots"] for row in sample),
            "position_margin_changed_slots": sum(row["position_margin_changed_slots"] for row in sample),
            "additional_eligible_slots": sum(row["additional_eligible_slots"] for row in sample),
            "straight_continuation_turns": sum(row["straight_continuation"] for row in sample),
            "original_misses_with_position_only_opportunity": [
                row["id"] for row in original_misses if row["position_only_possible"]
            ],
            "prompts": metrics_by_outcome(group, ARMS),
        }
    write_json(args.output_dir / "summary.json", result)
    if records:
        with (args.output_dir / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for name in ("major", "minor"):
        print(name, json.dumps(result[name]))


if __name__ == "__main__":
    main()
