"""Replay fixed history settings on mixed pedestrian and static-obstacle clips."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from real_multiframe_person_20261117 import (
    PERSON_HISTORY_S, TUM_DILATION_PX, dilate_person, history_support, infer_person,
)
from real_multiframe_pilot_20261117 import HISTORY_S, MAX_GAP_S, corridor_bands, replay, save
from real_multiframe_tum_20261117 import FR3_RGB, tum_frames


METHODS = ("single", "memory", "person", "expanded")


def restrict(frames, flags):
    # Reference labels are used here for attribution only. The full replay sees every depth point.
    return [replace(frame, cloud=frame.cloud[keep], weights=frame.weights[keep], person=frame.person[keep])
            for frame, keep in zip(frames, flags, strict=True)]


def replay_arms(frames, expanded):
    reference = replay(frames, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    widened = replay(expanded, keep_trace=True, person_history_s=PERSON_HISTORY_S)
    if reference.keys() != widened.keys():
        raise AssertionError("Expanded masks changed which frames were scored")
    for row, item in reference.items():
        for key in ("raw", "alarm"):
            new = widened[row][key]
            if item[key][:3] != new[:3] or np.any(np.array(new[3]) & ~np.array(item[key][3])):
                raise AssertionError(f"Mask dilation changed the controls or added old support at row {row}")
            item[key] = [item[key][0], item[key][1], item[key][3], new[3]]
    return reference


def sees_target(frame):
    return any(band.any() for band in corridor_bands(frame.cloud, frame.origin, frame.plane))


def static_ring_hits(frames, expanded, labels, rows):
    """Count labeled obstacle samples newly expired by the ring in the scored corridor."""
    known = {frame.row: (frame, added.person & ~frame.person & mask)
             for frame, added, mask in zip(frames, expanded, labels, strict=True)}
    hits = set()
    for row in rows:
        query, _ = known[row]
        newer = query
        while (previous := known.get(newer.row - 1)) is not None:
            past, ring = previous
            age = query.time - past.time
            if (age > HISTORY_S + 1e-8 or past.component != newer.component
                    or not 0 < newer.time - past.time <= MAX_GAP_S):
                break
            newer = past
            if age > PERSON_HISTORY_S + 1e-8 and ring.any():
                points = np.flatnonzero(ring)
                bands = corridor_bands(past.cloud[points], query.origin, query.plane)
                hits.update((past.row, int(point)) for point in points[np.logical_or.reduce(bands)])
    return len(hits)


def score_clip(frames, expanded, person_reference, obstacle_reference, episodes):
    for frame, new, human, obstacle in zip(frames, expanded, person_reference, obstacle_reference, strict=True):
        if (human.shape != frame.weights.shape or obstacle.shape != frame.weights.shape or np.any(human & obstacle)
                or not np.array_equal(frame.cloud, new.cloud) or not np.array_equal(frame.weights, new.weights)
                or np.any(frame.person & ~new.person)):
            raise ValueError(f"Reference labels or expanded masks do not align with the original geometry at {frame.row}")
    full = replay_arms(frames, expanded)
    people_frames = restrict(frames, person_reference)
    obstacle_frames = restrict(frames, obstacle_reference)
    expanded_obstacle = restrict(expanded, obstacle_reference)
    people = replay_arms(people_frames, restrict(expanded, person_reference))
    obstacle = replay_arms(obstacle_frames, expanded_obstacle)
    people_visible = {frame.row: sees_target(frame) for frame in people_frames}
    obstacle_visible = {frame.row: sees_target(frame) for frame in obstacle_frames}
    result = []
    for event in episodes:
        start, exit_row, end = (event[key] for key in ("start", "exit", "end"))
        if not 0 <= start < exit_row < end:
            raise ValueError("Invalid annotated departure interval")
        rows = list(range(exit_row, end + 1))
        post = {key: np.array([signal[row]["alarm"] for row in rows]).any(axis=2)
                for key, signal in (("full", full), ("people", people), ("obstacle", obstacle))}
        times = np.array([full[row]["time"] for row in rows])
        if any(people_visible[row] for row in rows):
            raise ValueError(f"Person reference is still in the corridor after claimed exit at row {exit_row}")
        if not any(people_visible[row] for row in range(start, exit_row)):
            raise ValueError(f"Person reference never reaches the corridor before exit at row {exit_row}")
        observed_static_rows = [row for row in rows if obstacle_visible[row]]
        hidden_rows = event.get("hidden_rows")
        if hidden_rows is not None:
            hidden = set(hidden_rows)
            if not hidden or not hidden.issubset(rows) or not hidden.isdisjoint(observed_static_rows):
                raise ValueError(f"Hidden obstacle rows must be after departure and have no target depth at {exit_row}")
            if not any(obstacle_visible[row] for row in range(start, exit_row)):
                raise ValueError(f"No in-corridor obstacle observation before departure at {exit_row}")
        elif len(observed_static_rows) < 2:
            raise ValueError(f"Fewer than two post-departure frames observe the target obstacle from row {exit_row}")

        exposure = {}
        for i, method in enumerate(METHODS):
            quiet = np.flatnonzero(~post["people"][:, i])
            exposure[method] = dict(person_alarm_frames=int(post["people"][:, i].sum()),
                                    person_tail_s=float(times[quiet[0]] - times[0]) if len(quiet) else None,
                                    obstacle_alarm_frames=int(post["obstacle"][:, i].sum()),
                                    full_alarm_frames=int(post["full"][:, i].sum()))
        lost_static = {
            baseline: [row for row, lost in zip(rows, post["obstacle"][:, i] & ~post["obstacle"][:, 3]) if lost]
            for i, baseline in ((1, "memory"), (2, "person"))
        }
        memory_only = post["obstacle"][:, 1] & ~post["obstacle"][:, 0]
        summary = dict(event, rows=rows, times=times.tolist(), counts=exposure, lost_static_rows=lost_static,
                       observed_static_rows=observed_static_rows,
                       newly_expired_static_points=static_ring_hits(frames, expanded, obstacle_reference, rows),
                       static_memory_only_rows=[row for row, used in zip(rows, memory_only, strict=True) if used])
        if hidden_rows is not None:
            eligible = [row for row in summary["static_memory_only_rows"] if row in hidden]
            trace_rows = {vote for row in hidden for vote in range(row - 2, row + 1)}
            source_votes = history_support(expanded_obstacle, trace_rows)
            for vote in source_votes:
                del vote["in_view_retained_sr"]  # No query-view projection was supplied to this trace.
            summary["hidden_obstacle"] = dict(
                memory_only_rows=eligible,
                retained_rows=[row for row in eligible if any(obstacle[row]["alarm"][3])],
                lost_rows=[row for row in eligible if not any(obstacle[row]["alarm"][3])],
                source_votes=source_votes,
            )
        result.append(summary)
    return dict(events=result, full=full, people=people, obstacle=obstacle)


def read_bundle(source, model, cached_masks=None):
    """NumPy observation bundle plus independent boolean person and obstacle pixel masks."""
    with np.load(source / "frames.npz", allow_pickle=False) as data, \
            np.load(source / "references.npz", allow_pickle=False) as reference:
        intrinsics = data["intrinsics"] if "intrinsics" in data else FR3_RGB
        planes = data["planes"] if "planes" in data else None
        scale = data["depth_scale"].item() if "depth_scale" in data else 5000.
        radius = max(1, round(TUM_DILATION_PX * intrinsics[0] / FR3_RGB[0]))
        human, obstacle = (reference[key].astype(bool) for key in ("person_reference", "static_reference"))
        color = data["bgr"]
        if cached_masks is None:
            masks = infer_person(color, model)
        else:
            with np.load(cached_masks, allow_pickle=False) as cache:
                if cache["snapshot"].item() != model.name:
                    raise ValueError("Cached masks are from a different model snapshot")
                stamps = data["rgb_timestamp"]
                rows = np.searchsorted(cache["rgb_timestamp"], stamps)
                if np.any(rows == len(cache["rgb_timestamp"])) or not np.array_equal(cache["rgb_timestamp"][rows], stamps):
                    raise ValueError("Cached masks are not for the requested color frames")
                masks = cache["mask"][rows]
        if "rgb_timestamp" in data and "depth_timestamp" in data:
            if np.any(data["rgb_timestamp"][:-1] > data["depth_timestamp"][1:]):
                raise ValueError("A history mask uses color captured after its query frame")
        if human.shape != color.shape[:3] or obstacle.shape != human.shape or masks.shape != human.shape:
            raise ValueError("Images, reference and person-mask sizes must match")
        options = dict(intrinsics=intrinsics, stride=radius, depth_scale=scale, planes=planes)
        frames = list(tum_frames(data, masks, **options))
        expanded = list(tum_frames(data, dilate_person(masks, radius), **options))
        # Reuse the observation adapter so annotation pixels follow the very same sampled/valid rays.
        person_reference = [f.person for f in tum_frames(data, human, **options)]
        static_reference = [f.person for f in tum_frames(data, obstacle, **options)]
    return radius, frames, expanded, person_reference, static_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--cached-masks", type=Path)
    args = parser.parse_args()
    radius, *inputs = read_bundle(args.source, args.snapshot, args.cached_masks)
    episodes = json.loads((args.source / "episodes.json").read_text(encoding="utf-8"))
    result = score_clip(*inputs, episodes)
    result["mask_dilation_px"] = radius
    save(args.source / "comparison.json", result)
    print(json.dumps(result["events"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
