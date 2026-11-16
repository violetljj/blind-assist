"""Use visible RGB-D foreground tracks to assign short lifetimes to moving old points."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from real_multiframe_mixed_20261117 import read_bundle, restrict, sees_target
from real_multiframe_motion_20261117 import compare
from real_multiframe_motion_transfer_20261117 import score_event
from real_multiframe_pilot_20261117 import HISTORY_S, follows, save
from real_multiframe_tum_20261117 import FR3_RGB


DEPTH_TOLERANCE_M = .08
MIN_ANCHORS = 3
LK = dict(winSize=(21, 21), maxLevel=3)
METHODS = ("single", "memory", "fixed_person", "visible_foreground")


@dataclass
class Observation:
    image: np.ndarray
    depth: np.ndarray
    rotation: np.ndarray
    origin: np.ndarray
    labels: np.ndarray
    sampled_labels: np.ndarray
    centers: dict
    anchors: dict
    available_at: float


def foreground_components(depth, mask):
    valid = (depth >= .35) & (depth <= 5.46) & mask
    nodes = np.full(depth.shape, -1, dtype=np.int32)
    size = int(valid.sum())
    nodes[valid] = np.arange(size)
    starts, ends = [], []
    # Add only four-neighbor depth-continuous edges inside the original expanded mask.
    for axis in (0, 1):
        first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        second = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        keep = valid[first] & valid[second] & (np.abs(depth[first] - depth[second]) <= DEPTH_TOLERANCE_M)
        starts.append(nodes[first][keep])
        ends.append(nodes[second][keep])
    edge = np.concatenate(starts), np.concatenate(ends)
    graph = coo_matrix((np.ones(len(edge[0]), dtype=bool), edge), shape=(size, size)).tocsr()
    _, component = connected_components(graph, directed=False)
    nodes[valid] = component
    return nodes


def observations(data, frames, masks, stride):
    scale = data["depth_scale"].item() if "depth_scale" in data else 5000.
    rgb_lag = np.maximum(0., data["rgb_timestamp"] - data["depth_timestamp"])
    for frame, rgb, raw, mask, rotation, lag in zip(frames, data["bgr"], data["depth_png"],
                                                   masks, data["rotations"], rgb_lag, strict=True):
        depth = raw.astype(np.float32) / scale
        labels = foreground_components(depth, mask)
        subsampled_depth = depth[::stride, ::stride]
        valid = (subsampled_depth >= .35) & (subsampled_depth <= 5.46)
        samples = labels[::stride, ::stride][valid]
        if not np.array_equal(samples >= 0, frame.person):
            raise ValueError("Visible foreground and history rays disagree")
        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        centers, anchors = {}, {}
        ids, sizes = np.unique(samples[samples >= 0], return_counts=True)
        for label in ids[sizes >= MIN_ANCHORS]:
            centers[int(label)] = np.median(frame.cloud[samples == label], axis=0)
            corners = cv2.goodFeaturesToTrack(image, 250, .01, stride, mask=(labels == label).astype(np.uint8),
                                              blockSize=7)
            if corners is not None and len(corners) >= MIN_ANCHORS:
                anchors[int(label)] = corners[:, 0]
        yield Observation(image, depth, rotation, frame.origin, labels, samples, centers, anchors,
                          frame.time + float(lag))


def backproject(view, pixels, intrinsics):
    fx, fy, cx, cy = intrinsics
    h, w = view.depth.shape
    x, y = np.rint(pixels).astype(int).T
    inside = (1 <= x) & (x < w - 1) & (1 <= y) & (y < h - 1)
    x, y = x.clip(1, w - 2), y.clip(1, h - 2)
    depth = view.depth[y, x]
    patch = np.stack([view.depth[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
    smooth = ((patch >= .35) & (patch <= 5.46)
              & (np.abs(patch - depth[:, None]) <= DEPTH_TOLERANCE_M)).all(axis=1)
    ray = np.column_stack(((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))))
    return (ray * depth[:, None]) @ view.rotation.T + view.origin, view.labels[y, x], inside & smooth


def project(view, points, intrinsics):
    fx, fy, cx, cy = intrinsics
    x, y, z = ((points - view.origin) @ view.rotation).T
    z = np.maximum(z, .1)
    return np.column_stack((fx * x / z + cx, fy * y / z + cy))


def match(source, target, pixels, seed, intrinsics):
    points = np.ascontiguousarray(pixels[:, None], dtype=np.float32)
    initial = None if seed is None else np.ascontiguousarray(seed[:, None], dtype=np.float32)
    flag = 0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW
    next_points, good, _ = cv2.calcOpticalFlowPyrLK(source.image, target.image, points, initial, flags=flag, **LK)
    usable = (good[:, 0] != 0) & np.isfinite(next_points[:, 0]).all(axis=1)
    if not usable.any():
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3))
    pixels, next_points = pixels[usable], next_points[usable]
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(target.image, source.image, next_points, None, **LK)
    paired = (reverse[:, 0] != 0) & (np.linalg.norm(back[:, 0] - pixels, axis=1) <= 1.)
    left, left_label, left_good = backproject(source, pixels[paired], intrinsics)
    right, right_label, right_good = backproject(target, next_points[paired, 0], intrinsics)
    valid = left_good & right_good & (left_label >= 0) & (right_label >= 0)
    return left_label[valid], right_label[valid], right[valid] - left[valid]


def propose_links(source, target, intrinsics):
    if not target.centers:
        return {}
    collected = {}
    for label, points in source.anchors.items():
        world, _, stable = backproject(source, points, intrinsics)
        if int(stable.sum()) < MIN_ANCHORS:
            continue
        points, world = points[stable], world[stable]
        static_seed = project(target, world, intrinsics)
        center = source.centers[label]
        nearest = sorted(target.centers, key=lambda other: np.linalg.norm(target.centers[other] - center))[:3]
        shifts = [project(target, np.array([target.centers[other], center]), intrinsics) for other in nearest]
        seeds = [None, static_seed, *(static_seed + pair[0] - pair[1] for pair in shifts)]
        for seed in seeds:
            left_label, right_label, vectors = match(source, target, points, seed, intrinsics)
            for dest in np.unique(right_label[left_label == label]):
                if dest not in target.centers:
                    continue
                values = vectors[(left_label == label) & (right_label == dest)]
                if len(values) < MIN_ANCHORS:
                    continue
                moved = np.linalg.norm(values, axis=1) > DEPTH_TOLERANCE_M
                motion_support = 0
                if int(moved.sum()) >= MIN_ANCHORS:
                    displacement = np.median(values[moved], axis=0)
                    agrees = np.linalg.norm(values - displacement, axis=1) <= DEPTH_TOLERANCE_M
                    if np.linalg.norm(displacement) > DEPTH_TOLERANCE_M and int(agrees.sum()) * 2 > len(values):
                        motion_support = int((moved & agrees).sum())
                score = (len(values), motion_support)
                key = (label, int(dest))
                if score > collected.get(key, (0, 0)):
                    collected[key] = score
    return collected


def associate(left, right, intrinsics):
    proposals = propose_links(left, right, intrinsics)
    for (dest, source), score in propose_links(right, left, intrinsics).items():
        key = (source, dest)
        proposals[key] = max(score, proposals.get(key, (0, 0)))
    used_left, used_right = set(), set()
    for (source, target), (matched, motion) in sorted(proposals.items(), key=lambda item: item[1], reverse=True):
        if source not in used_left and target not in used_right:
            used_left.add(source)
            used_right.add(target)
            yield source, target, matched, motion


def direct_links(left, right, intrinsics):
    available_at = max(left.available_at, right.available_at)
    return [(*pair, available_at) for pair in associate(left, right, intrinsics)]


def confirm_foreground(frames, views, intrinsics, *, links_by_end=None):
    marked = [np.full(len(frame.cloud), np.inf) for frame in frames]
    tracks, evidence = {}, []
    for i in range(1, len(frames)):
        previous, current = frames[i - 1], frames[i]
        updated = {}
        if follows(previous, current):
            links = (direct_links(views[i - 1], views[i], intrinsics)
                     if links_by_end is None else links_by_end.get(i, ()))
            for source, target, matched, motion, available_at in links:
                members, last_moved = tracks.get(source, ([(i - 1, source)], -np.inf))
                members = [(j, group) for j, group in members if current.time - frames[j].time <= HISTORY_S + 1e-8]
                members.append((i, target))
                if motion >= MIN_ANCHORS:
                    last_moved = available_at
                if available_at - last_moved <= HISTORY_S + 1e-8:
                    for j, group in members:
                        points = views[j].sampled_labels == group
                        marked[j][points] = np.minimum(marked[j][points], available_at)
                updated[target] = (members, last_moved)
                evidence.append(dict(source_row=previous.row, target_row=current.row, source_component=source,
                                     target_component=target, matched=matched, moving_support=motion,
                                     available_at_s=available_at))
        tracks = updated
    return [replace(f, person_motion_at=t) for f, t in zip(frames, marked, strict=True)], evidence


def case_result(case, snapshot):
    clip = Path(case["source"])
    cache = Path(case["cached_masks"]) if "cached_masks" in case else None
    radius, _, frames, people, obstacles, masks = read_bundle(clip, snapshot, cache, include_pixel_masks=True)
    with np.load(clip / "frames.npz", allow_pickle=False) as data:
        intrinsics = data["intrinsics"] if "intrinsics" in data else FR3_RGB
        views = list(observations(data, frames, masks, radius))
    marked, evidence = confirm_foreground(frames, views, intrinsics)
    cutoff = case.get("exit", len(frames) // 2) + 1
    prefix, _ = confirm_foreground(frames[:cutoff], views[:cutoff], intrinsics)
    cutoff_s = prefix[-1].time
    for earlier, full in zip(prefix, marked[:cutoff], strict=True):
        if not np.array_equal(earlier.person_motion_at <= cutoff_s, full.person_motion_at <= cutoff_s):
            raise AssertionError("A not-yet-observed foreground match changed a prior frame")

    person, target = restrict(frames, people), restrict(frames, obstacles)
    signals = dict(full=compare(frames, marked), people=compare(person, restrict(marked, people)),
                   obstacle=compare(target, restrict(marked, obstacles)))
    prefix_signal = compare(frames[:cutoff], prefix)
    if any(item != signals["full"][row] for row, item in prefix_signal.items()):
        raise AssertionError("The full-clip foreground matches changed the prefix alarms")
    prior = json.loads(Path(case["previous"]).read_text(encoding="utf-8"))[case["previous_key"]]["signals"]
    for kind, old in prior.items():
        now = signals[kind]
        if {str(row) for row in now} != old.keys():
            raise AssertionError("Expected the same windows as the previous comparison")
        for row, item in now.items():
            expected = old[str(row)]
            if (item["time"] != expected["time"] or item["history_frames"] != expected["history_frames"]
                    or any(item[key][:3] != expected[key][:3] for key in ("raw", "alarm"))):
                raise AssertionError(f"An unchanged arm differed in {kind} at row {row}")
    if "exit" in case:
        events = json.loads((clip / "episodes.json").read_text(encoding="utf-8"))
        person_present = [sees_target(f) for f in person]
        static_present = [sees_target(f) for f in target]
        summary = [score_event(e, frames, signals, person_present, static_present, methods=METHODS)
                   for e in events]
    else:
        summary = {row: {key: signal[row]["alarm"] for key, signal in signals.items()} for row in case["score_rows"]}
    marked_points = [[int(np.isfinite(f.person_motion_at[mask]).sum()) for mask in (f.person, h, s)]
                     for f, h, s in zip(marked, people, obstacles, strict=True)]
    return dict(methods=list(METHODS), summary=summary, signals=signals, evidence=evidence, marked_counts=marked_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {}
    for name, case in cases.items():
        result[name] = case_result(case, args.snapshot)
        print(name, json.dumps(result[name]["summary"], ensure_ascii=False), flush=True)
    save(args.output, result)


if __name__ == "__main__":
    main()
