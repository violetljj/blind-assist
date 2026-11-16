"""Try corroborated static body history on the natural table before scoring the crossing."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_multiframe_dense_association_20261117 import verify_previous
from real_multiframe_foreground_20261117 import (
    DEPTH_TOLERANCE_M, METHODS, backproject, confirm_foreground, observations, project,
)
from real_multiframe_mixed_20261117 import read_bundle, restrict, sees_target
from real_multiframe_motion_20261117 import compare
from real_multiframe_motion_transfer_20261117 import BANDS, band_tails, score_event
from real_multiframe_person_20261117 import PERSON_HISTORY_S
from real_multiframe_pilot_20261117 import BODY_BAND, MIN_SOLID_ANGLE, corridor_bands, follows, replay, save


CANDIDATE_METHODS = ("single", "memory", "cleared", "static_body_history")


def corroborated_at(frames, views, intrinsics):
    """A sampled point needs positive registered depth in its own and an adjacent frame."""
    ready = {frame.row: np.full(len(frame.cloud), np.inf) for frame in frames}
    stable = [backproject(view, project(view, frame.cloud, intrinsics), intrinsics)[2]
              for frame, view in zip(frames, views, strict=True)]
    for right in range(1, len(frames)):
        left = right - 1
        if not follows(frames[left], frames[right]):
            continue
        stamp = max(views[left].available_at, views[right].available_at)
        for source, other in ((left, right), (right, left)):
            frame, witness = frames[source], views[other]
            points = frame.cloud
            camera_z = (points - witness.origin) @ witness.rotation[:, 2]
            measured, _, valid = backproject(witness, project(witness, points, intrinsics), intrinsics)
            match = stable[source] & valid & (camera_z > .1)
            match &= np.linalg.norm(measured - points, axis=1) <= DEPTH_TOLERANCE_M
            ready[frame.row][match] = np.minimum(ready[frame.row][match], stamp)
    return ready


def candidate(frames, ready, history_only):
    def allowed(source, query):
        if query.row in history_only:
            return (ready[source.row] < query.time) & (source.person_motion_at > query.time)
        return None

    return replay(frames, keep_trace=True, person_history_s=PERSON_HISTORY_S, body_history=allowed)


def verify_candidate(before, after, history_only):
    if before.keys() != after.keys():
        raise AssertionError("Candidate changed replay coverage")
    for row, prior in before.items():
        checked = after[row]
        if any(prior[key] != checked[key] for key in ("time", "history_frames")):
            raise AssertionError(f"Candidate changed observation timing at row {row}")
        for key in ("raw", "alarm"):
            previous, current = prior[key], checked[key]
            if previous[:2] != current[:2] or previous[-1][0] != current[-1][0]:
                raise AssertionError(f"Candidate changed a common control or head band at row {row}")
            if current[-1][BODY_BAND] and not previous[-1][BODY_BAND]:
                raise AssertionError(f"Candidate created new body support at row {row}")
        if row not in history_only and prior["raw"][-1] != checked["raw"][-1]:
            raise AssertionError(f"Candidate changed a direct-vote query at row {row}")


def table_score(before, after, row):
    details = {}
    for label in ("full", "obstacle"):
        first, second = before[label][row], after[label][row]
        if first["alarm"][0][BODY_BAND] or not first["alarm"][-1][BODY_BAND]:
            raise AssertionError(f"The frozen table query is no longer memory dependent for {label}")
        details[label] = {
            "before_confirmation": first["alarm"][-1],
            "after_confirmation": second["alarm"][-1],
            "preceding_votes_before": [before[label][i]["raw"][-1] for i in range(row - 2, row + 1)],
            "preceding_votes_after": [after[label][i]["raw"][-1] for i in range(row - 2, row + 1)],
            "preceding_body_sources": [after[label][i]["body_history_sources"] for i in range(row - 2, row + 1)],
        }
    passed = all(item["after_confirmation"][BODY_BAND] for item in details.values())
    return dict(passed=passed, row=row, detail=details, crossing="AUTHORIZED" if passed else "NOT_RUN")


def crossing_score(folder, before, after, groups, source_indices):
    person_present = [sees_target(frame) for frame in groups["people"]]
    static_present = [sees_target(frame) for frame in groups["obstacle"]]
    events = json.loads((folder / "episodes.json").read_text(encoding="utf-8"))
    scores = []
    for event in events:
        result = {}
        for label, signals, methods in (("before", before, METHODS), ("after", after, CANDIDATE_METHODS)):
            summary = score_event(event, groups["full"], signals, person_present, static_present, methods=methods)
            rows, times = summary["rows"], np.array(summary["times_s"])
            whole = band_tails(signals["full"], rows, times, methods=methods)
            result[label] = {
                "rows": [source_indices[row] for row in rows],
                "times_s": times.tolist(),
                **{name: {
                    band: dict(tails[band]["methods"][methods[-1]],
                               frames=[source_indices[row] for row in rows
                                       if signals[name][row]["alarm"][-1][i]])
                    for i, band in enumerate(BANDS)}
                   for name, tails in (("full", whole), ("people", summary["person"]))},
            }
        scores.append(result)
    return scores


def measure(case, model, pose_times):
    folder = Path(case["source"])
    stride, _, frames, humans, obstacles, masks = read_bundle(folder, model, include_pixel_masks=True)
    with np.load(folder / "frames.npz", allow_pickle=False) as data:
        indices, times = data["source_indices"].tolist(), data["depth_timestamp"]
        if not np.array_equal(times, data["rgb_timestamp"]) or np.any(np.diff(times) <= 0.):
            raise ValueError("Expected distinct original paired ETH3D observations")
        following = np.searchsorted(pose_times, times)
        if np.any((following == 0) | (following == len(pose_times))):
            raise ValueError("An observation falls outside Vicon coverage")
        pose_spans = pose_times[following] - pose_times[following - 1]
        if np.any(pose_spans > .020):
            raise ValueError("An observation crosses the original ETH3D pose-gap limit")
        intrinsics = data["intrinsics"]
        views = list(observations(data, frames, masks, stride))
    history_only = {frame.row for frame in frames
                    if frame.weights[corridor_bands(frame.cloud, frame.origin, frame.plane)[BODY_BAND]].sum()
                    < MIN_SOLID_ANGLE}
    marked, _ = confirm_foreground(frames, views, intrinsics)
    ready = corroborated_at(frames, views, intrinsics)
    # Reference masks enter only after evidence and gate decisions have been computed.
    labels = {"obstacle": obstacles}
    if "exit" in case:
        labels["people"] = humans
    raw = dict(full=frames, **{label: restrict(frames, mask) for label, mask in labels.items()})
    groups = dict(full=marked, **{label: restrict(marked, mask) for label, mask in labels.items()})
    before = {label: compare(raw[label], stream) for label, stream in groups.items()}
    verify_previous(before, case)
    after = {"full": candidate(marked, ready, history_only)}
    for label, flags in labels.items():
        sliced = {f.row: ready[f.row][keep] for f, keep in zip(frames, flags, strict=True)}
        after[label] = candidate(groups[label], sliced, history_only)
    for label in before:
        verify_candidate(before[label], after[label], history_only)

    cutoff = case["exit"] if "exit" in case else case["score_row"]
    prefix, _ = confirm_foreground(frames[:cutoff + 1], views[:cutoff + 1], intrinsics)
    prefix_ready = corroborated_at(prefix, views[:cutoff + 1], intrinsics)
    partial = candidate(prefix, prefix_ready, history_only)
    if any(item != after["full"][row] for row, item in partial.items()):
        raise AssertionError("Later observations changed the candidate prefix")

    if "exit" in case:
        summary = crossing_score(folder, before, after, groups, indices)
    else:
        if indices[cutoff] != 1101 or cutoff not in history_only:
            raise AssertionError("The natural table gate is no longer the intended query")
        summary = table_score(before, after, cutoff)
    return dict(n_frames=len(frames), source_indices=indices, pose_spans_s=pose_spans.tolist(),
                gate_queries=[indices[row] for row in sorted(history_only)],
                summary=summary, before=before, after=after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("vicon", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    times = np.loadtxt(args.vicon, usecols=0)
    table = measure(cases["table_40_clean"], args.snapshot, times)
    save(args.output / "table.json", table)
    print("table", json.dumps(table["summary"], ensure_ascii=False), flush=True)
    if table["summary"]["passed"]:
        cross = measure(cases["eth3d_crossing"], args.snapshot, times)
        save(args.output / "crossing.json", cross)
        print("crossing", json.dumps(cross["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
