"""Prepare map-matched PING turns for a secondary, exploratory heading replay."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REVISION = "e4d07e10507fe5482e548009ea89a88c0bf9a771"
DATA_SHA256 = "45e36ddfb9a1c876b135233b9f3a215c61df133cba2e87116a9f3807eb26760e"
SOURCE = f"https://raw.githubusercontent.com/eshed1/PING/{REVISION}/data.mat"
STUDY = "https://proceedings.mlr.press/v87/ohnbar18a.html"
# Three supplied routes visit two floors each. Route 1 starts at the basement's
# far end and exits the lift on floor 3. Routes 2 and 3 return in order.
REVERSE = {
    (1, -1): False, (1, 3): False,
    (2, 3): True, (2, 1): False,
    (3, 1): True, (3, -1): True,
}
MAJOR_CUES = {"TurnLeft": 1, "TurnRight": -1}
TURN_CUES = {
    "TurnLeft", "TurnRight", "TurnDiagonalLeft", "TurnDiagonalRight",
    "TurnLittleLeft", "TurnLittleRight", "UTurn", "UTurn PoiLeft",
}
CLIP_MS = 5000
MIN_CLIP_MS = 3000
NEAR_CORNER_M = 4.0
AMBIGUITY_M = 1.5
MAX_POSE_LAG_MS = 250
MIN_LEG_M = 2.5
# Source preprocessing interpolates 500 points, inclusive, on each original edge.
INTERPOLATED_EDGE_STEPS = 499


@dataclass(frozen=True)
class Corner:
    id: int
    x: float
    y: float
    incoming_deg: float
    outgoing_deg: float
    turn_deg: float


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def bearings(vectors):
    return np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))


def route_vertices(points):
    """Restore the published polyline vertices, verifying source interpolation."""
    if (len(points) - 1) % INTERPOLATED_EDGE_STEPS != 0:
        raise ValueError("Unexpected PING planned-path interpolation")
    vertices = points[::INTERPOLATED_EDGE_STEPS]
    rebuilt = np.vstack([
        np.linspace(a, b, INTERPOLATED_EDGE_STEPS, endpoint=False)
        for a, b in zip(vertices, vertices[1:])
    ] + [vertices[-1:]])
    if not np.allclose(points, rebuilt, atol=5e-6, rtol=0.0):
        raise ValueError("PING planned path does not follow the expected straight interpolation")
    return vertices


def turn_corners(points, *, min_angle=60.0, max_angle=120.0):
    legs = np.diff(points, axis=0)
    angles = bearings(legs)
    lengths = np.linalg.norm(legs, axis=1)
    corners = []
    for index in range(1, len(points) - 1):
        turn = float(wrap(angles[index] - angles[index - 1]))
        if min_angle <= abs(turn) <= max_angle and min(lengths[index - 1:index + 1]) >= MIN_LEG_M:
            corners.append(Corner(
                index, *points[index],
                float(angles[index - 1]), float(angles[index]), turn,
            ))
    return corners


def circular_center(angle):
    radians = np.radians(angle)
    center = np.degrees(np.angle(np.mean(np.exp(1j * radians))))
    return float(wrap(center))


def basis_calibration(participant, first_route):
    """Use the middle of the first known straight, many metres before any turn."""
    start, end = first_route[:2]
    vector = end - start
    length = np.linalg.norm(vector)
    tangent = vector / length
    positions = participant["gtpos"] - start
    forward = positions @ tangent
    lateral = np.abs(positions[:, 0] * tangent[1] - positions[:, 1] * tangent[0])
    selected = np.flatnonzero(
        (participant["gtroute"] == 1) & (participant["gtfloor"] == -1)
        & (forward >= 0.25 * length) & (forward <= 0.60 * length)
        & (lateral <= 2.0)
    )
    if not len(selected) or np.any(np.diff(selected) != 1):
        raise ValueError("Calibration straight does not form one contiguous annotated passage")
    start_stamp, final_stamp = participant["gttime"][selected[[0, -1]]]
    sampled = (participant["logtime"] >= start_stamp) & (participant["logtime"] <= final_stamp)
    stamps = participant["logtime"][sampled]
    angles = participant["logorientation"][sampled]
    if len(stamps) < 15 or stamps[-1] - stamps[0] < 3000:
        raise ValueError("Initial route does not provide enough heading calibration samples")
    map_bearing = float(bearings(vector[None])[0])
    # Positive map turn is left; NavCog logging has clockwise-positive yaw.
    # Its arbitrary zero is registered to this disjoint straight.
    offset = circular_center(angles + map_bearing)
    return {
        "offset_deg": offset,
        "straight_map_heading_deg": map_bearing,
        "samples": len(stamps),
        "start_ms": int(stamps[0]),
        "end_ms": int(stamps[-1]),
        "p90_abs_residual_deg": float(np.percentile(np.abs(wrap(angles + map_bearing - offset)), 90)),
    }


def find_corner(corners, x, y, direction):
    compatible = [
        (np.hypot(x - c.x, y - c.y), c)
        for c in corners if np.sign(c.turn_deg) == direction
    ]
    compatible.sort(key=lambda item: item[0])
    if not compatible or compatible[0][0] > NEAR_CORNER_M:
        return None, "no_matching_corner_within_4m"
    if len(compatible) > 1 and compatible[1][0] - compatible[0][0] < AMBIGUITY_M:
        return None, "ambiguous_corner"
    return compatible[0], None


def validate_participant(participant):
    time = participant["logtime"]
    gt_time = participant["gttime"]
    starts = participant["instructions"]["timestart"]
    if any(np.any(np.diff(stamps) < 0) for stamps in (time, gt_time, starts)):
        raise ValueError("PING time stream decreases")
    if (
        len(participant["logorientation"]) != len(time)
        or participant["logposition"].shape != (len(time), 2)
        or len(participant["gtroute"]) != len(gt_time)
        or len(participant["gtfloor"]) != len(gt_time)
        or len(participant["instructions"]["command"]) != len(starts)
        or len(participant["instructions"]["timeend"]) != len(starts)
    ):
        raise ValueError("PING streams do not match their time axes")


def deduplicate(participant):
    """Use the last observation emitted at each millisecond timestamp."""
    participant = dict(participant)
    pose_unique = np.r_[np.diff(participant["logtime"]) != 0, True]
    annotation_unique = np.r_[np.diff(participant["gttime"]) != 0, True]
    for key in ("gtroute", "gtfloor"):
        if np.any(np.diff(participant[key])[~annotation_unique[:-1]] != 0):
            raise ValueError("Conflicting route or floor at the same timestamp")
    for key in ("logtime", "logposition", "logorientation"):
        participant[key] = participant[key][pose_unique]
    for key in ("gttime", "gtpos", "gtroute", "gtfloor"):
        participant[key] = participant[key][annotation_unique]
    return participant


def prepare_person(person, participant, routes, corners, *, cue_directions=None):
    if cue_directions is None:
        cue_directions = MAJOR_CUES
    validate_participant(participant)
    participant = deduplicate(participant)
    calibration = basis_calibration(participant, routes[1, -1])
    heading = wrap(calibration["offset_deg"] - participant["logorientation"])
    time = participant["logtime"]
    annotations = participant["gttime"]
    instructions = participant["instructions"]
    codes = instructions["command"]
    cue_times = instructions["timestart"]
    directional_cues = np.array(sorted(cue_times[np.isin(codes, list(TURN_CUES))]))
    rejections = []
    clips = []
    seen = set()
    for index in np.flatnonzero(np.isin(codes, list(cue_directions))):
        start = int(cue_times[index])
        cue = str(codes[index])
        rejection = {"person": person, "cue_index": int(index), "cue": cue, "timestart_ms": start}
        pose_index = np.searchsorted(time, start, side="right") - 1
        annotation_index = np.searchsorted(annotations, start, side="right") - 1
        if pose_index < 0 or start - int(time[pose_index]) > MAX_POSE_LAG_MS or annotation_index < 0:
            rejections.append(dict(rejection, reason="no_recent_pose_at_cue"))
            continue
        route = int(participant["gtroute"][annotation_index])
        floor = int(participant["gtfloor"][annotation_index])
        if (route, floor) not in corners:
            rejections.append(dict(rejection, reason="outside_published_route_floor"))
            continue
        if start <= calibration["end_ms"]:
            rejections.append(dict(rejection, reason="before_calibration_end"))
            continue
        x, y = participant["logposition"][pose_index]
        match, reason = find_corner(corners[(route, floor)], x, y, cue_directions[cue])
        if match is None:
            rejections.append(dict(rejection, reason=reason))
            continue
        distance, corner = match
        key = (route, floor, corner.id)
        if key in seen:
            rejections.append(dict(rejection, reason="duplicate_corner_cue"))
            continue
        # Censor the clip before a later directional command, a floor/route
        # change, or 5 s. Other speech stays as part of the logged interaction.
        later_turn = np.searchsorted(directional_cues, start, side="right")
        end_bound = start + CLIP_MS + 1
        if later_turn < len(directional_cues):
            end_bound = min(end_bound, int(directional_cues[later_turn]))
        subsequent_annotation = (participant["gtroute"][annotation_index:] != route) | (
            participant["gtfloor"][annotation_index:] != floor
        )
        departure = np.flatnonzero(subsequent_annotation)
        if len(departure):
            end_bound = min(end_bound, int(annotations[annotation_index + departure[0]]))
        first = np.searchsorted(time, start, side="left")
        last = np.searchsorted(time, end_bound, side="left")
        stamps = time[first:last]
        seen.add(key)
        if len(stamps) < 2 or stamps[-1] - start < MIN_CLIP_MS:
            rejections.append(dict(rejection, reason="under_3s_before_next_turn_or_route_boundary"))
            continue
        relative_s = (stamps - start) / 1000.0
        yaw = heading[first:last]
        clips.append({
            "person": person,
            "number": int(index + 1),
            "cue": cue,
            "route": route,
            "floor": floor,
            "corner_index": corner.id,
            "corner_xy_m": [corner.x, corner.y],
            "cue_pose_xy_m": [float(x), float(y)],
            "cue_pose_to_corner_m": float(distance),
            "cue_start_ms": start,
            "cue_end_s": (int(instructions["timeend"][index]) - start) / 1000.0,
            "planned_turn_deg": corner.turn_deg,
            "planned_incoming_deg": corner.incoming_deg,
            "target_heading_deg": corner.outgoing_deg,
            "time_s": relative_s.tolist(),
            "heading_deg": yaw.tolist(),
            "error_deg": wrap(corner.outgoing_deg - yaw).tolist(),
            "max_sample_gap_s": float(np.max(np.diff(relative_s))),
        })
    return calibration, clips, rejections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("data.mat"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("turns.json"))
    args = parser.parse_args()
    body = args.source.read_bytes()
    if sha256(body).hexdigest() != DATA_SHA256:
        raise ValueError("Unexpected PING source bytes")
    data = loadmat(args.source, simplify_cells=True, variable_names=["contextdata", "trajdata"])
    if len(data["trajdata"]) != 9:
        raise ValueError("Expected nine released PING participants")
    maps = {int(f["floorid"]): route_vertices(f["plannedpath"]) for f in data["contextdata"]}
    routes = {key: (maps[key[1]][::-1] if reverse else maps[key[1]]) for key, reverse in REVERSE.items()}
    corners = {key: turn_corners(vertices) for key, vertices in routes.items()}
    people = {}
    clips = []
    rejections = []
    vocabulary = Counter()
    gaps_ms = []
    for index, participant in enumerate(data["trajdata"], 1):
        person = f"P{index}"
        vocabulary.update(map(str, participant["instructions"]["command"]))
        gaps_ms.extend(np.diff(participant["logtime"]).tolist())
        calibration, examples, reasons = prepare_person(person, participant, routes, corners)
        people[person] = {
            "calibration": calibration,
            "eligible": len(examples),
            "exclusion_counts": dict(Counter(row["reason"] for row in reasons)),
        }
        clips.extend(examples)
        rejections.extend(reasons)
    receipt = {
        "source_url": SOURCE,
        "paper": STUDY,
        "sha256": DATA_SHA256,
        "data_mat_bytes": len(body),
        "repository_license": "MIT",
        "participants": len(people),
        "full_release_instruction_counts": dict(vocabulary),
        "full_release_pose_sample_count": len(gaps_ms) + len(people),
        "full_release_pose_gap_ms_percentiles_1_10_50_90_95_99": np.percentile(
            gaps_ms, [1, 10, 50, 90, 95, 99],
        ).tolist(),
        "geometry": {
            f"R{route}/F{floor}": [vars(corner) for corner in values]
            for (route, floor), values in corners.items()
        },
        "candidate_major_cues": len(clips) + len(rejections),
        "selected_major_turns": len(clips),
        "exclusion_counts": dict(Counter(row["reason"] for row in rejections)),
        "people": people,
        "clips_with_all_sample_gaps_le_250ms": sum(row["max_sample_gap_s"] <= 0.25 for row in clips),
        "semantics": (
            "Estimated NavCog device-heading versus outgoing fixed route leg. Pose gaps are "
            "native; censor labels with incomplete future coverage or long intervening gaps. "
            "No independently measured body heading and no public heading-feedback timestamps."
        ),
    }
    args.out.write_text(json.dumps({"receipt": receipt, "clips": clips, "rejections": rejections}, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({
        "candidate_major_cues": receipt["candidate_major_cues"],
        "selected_major_turns": receipt["selected_major_turns"],
        "exclusion_counts": receipt["exclusion_counts"],
        "full_release_pose_count": receipt["full_release_pose_sample_count"],
        "whole_clip_le_250ms_gaps": receipt["clips_with_all_sample_gaps_le_250ms"],
        "people": people,
    }, indent=2))


if __name__ == "__main__":
    main()
