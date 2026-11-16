"""Check native ToF against the NavVis reference and replay a fixed causal veto."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from arkit_selective_heading_dev_20261116 import (
    HALF_WIDTH, MARGIN, MIN_SOLID_ANGLE, angle_deg, in_corridor, samples,
)
from lamar_navvis_corridor_dev_20261116 import MAX_RANGE, NavVis


class DepthSensor:
    def __init__(self, lut, head_from_depth):
        self.lut = lut
        self.head_from_depth = head_from_depth
        z1 = np.dstack((lut, np.ones(lut.shape[:2])))
        self.rays = z1 / np.linalg.norm(z1, axis=-1, keepdims=True)
        du, dv = np.gradient(self.rays, axis=(1, 0))
        self.weights = np.linalg.norm(np.cross(du, dv), axis=-1)
        self.valid_pixel = np.isfinite(lut).all(axis=2)
        self.tree = cKDTree(lut[self.valid_pixel])

    def valid(self, depth):
        return self.valid_pixel & (depth >= .5) & (depth <= np.float32(5.46))

    def cloud(self, depth, pose):
        mask = (self.valid(depth) & np.isfinite(self.weights))[::2, ::2]
        xyz = self.rays[::2, ::2][mask] * depth[::2, ::2][mask, None]
        return xyz @ pose[:3, :3].T + pose[:3, 3], 4. * self.weights[::2, ::2][mask]

    def clear_mask(self, points, depth, pose):
        coords = (points - pose[:3, 3]) @ pose[:3, :3]
        ratio = coords[..., :2] / np.maximum(coords[..., 2:], .1)
        mismatch, nearest = self.tree.query(ratio)
        found = depth[self.valid_pixel][nearest]
        return ((coords[..., 2] > .1) & (mismatch <= .008) & (found >= .5)
                & (found <= np.float32(5.46)) & (found > np.linalg.norm(coords, axis=-1) + .08))

    def clear_fraction(self, points, depth, pose):
        return self.clear_mask(points, depth, pose).mean(axis=1)


def summarize_errors(differences):
    if not len(differences):
        return dict(pairs=0)
    return dict(pairs=len(differences), median_m=round(float(np.median(differences)), 4),
                median_abs_m=round(float(np.median(np.abs(differences))), 4),
                p90_abs_m=round(float(np.quantile(np.abs(differences), .9)), 4),
                within_10cm=round(float((np.abs(differences) <= .10).mean()), 4))


def tof_mesh_check(args):
    with np.load(args.crop, allow_pickle=False) as mesh:
        scan = NavVis(mesh["vertices"], mesh["faces"])
    sources = json.loads(args.samples.read_text(encoding="utf-8"))["sources"]
    result = {}
    for session in sources:
        source = session["source"]
        with np.load(args.prepared / f"{source}.npz", allow_pickle=False) as data:
            poses, stamps = data["T_wc"], data["timestamp_us"]
            sensor = DepthSensor(data["depth_lut"], data["T_depth_to_head"])
        per_sample, all_errors, near_errors = [], [], []
        for item in session["samples"]:
            p = item["pose_row"]
            if stamps[p] != item["timestamp_us"] or not np.isfinite(poses[p]).all():
                raise ValueError(f"Native audit did not match pose at {source}/{item['timestamp_us']}")
            image = cv2.imread(item["path"], cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint16 or image.shape != sensor.lut.shape[:2]:
                raise ValueError(f"Invalid ToF sample at {item['path']}")
            native = image.astype(np.float32) * .001
            pose = poses[p] @ sensor.head_from_depth
            rays = sensor.rays[::3, ::3]
            ray_range = scan.distance(pose[:3, 3], rays.reshape(-1, 3) @ pose[:3, :3].T)
            ray_range = ray_range.reshape(rays.shape[:2])
            d = native[::3, ::3]
            valid = sensor.valid(native)[::3, ::3]
            paired = valid & (ray_range >= .30) & (ray_range < MAX_RANGE)
            near = d <= 2.
            residual = d - ray_range
            all_errors.append(residual[paired])
            near_errors.append(residual[paired & near])
            per_sample.append(dict(timestamp_us=item["timestamp_us"], native_valid=int(valid.sum()),
                                   native_near=int((valid & near).sum()), map_hits=int((ray_range < MAX_RANGE).sum()),
                                   paired=summarize_errors(all_errors[-1]), near=summarize_errors(near_errors[-1])))
        info = dict(paired=summarize_errors(np.concatenate(all_errors)),
                    near=summarize_errors(np.concatenate(near_errors)),
                    native_near=sum(row["native_near"] for row in per_sample), samples=per_sample)
        result[source] = info
        print(source, info["native_near"], info["near"], info["paired"], flush=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)


def walking_plane(data, index):
    poses, times, down, component = (data[k] for k in ("T_local", "ts", "local_down", "component"))
    last = slice(max(0, index - 20), index + 1)
    usable = np.flatnonzero(np.isfinite(down[last]).all(axis=1) & (component[last] == component[index])
                            & (times[last] >= times[index] - .10))
    if not len(usable):
        return None
    up = -down[usable[-1] + last.start].copy()
    magnitude = np.linalg.norm(up)
    if not .5 <= magnitude <= 1.5:
        return None
    up /= magnitude
    forward = poses[index, :3, 2] - up * (poses[index, :3, 2] @ up)
    norm = np.linalg.norm(forward)
    if norm < .5:
        return None
    forward /= norm
    return np.array([forward, np.cross(up, forward), up])


def reliable_direction(data, index, plane):
    times, poses, components = (data[key] for key in ("ts", "T_local", "component"))
    targets = times[index] - np.array([.8, .4])
    past = np.searchsorted(times[:index], targets, side="right") - 1
    if (past < 0).any() or (targets - times[past] > .07).any():
        return None
    if (components[past[0]:index + 1] != components[index]).any() or np.diff(times[past[0]:index + 1]).max() > .22:
        return None
    picks = np.r_[past, index]
    delta = np.diff(poses[picks, :3, 3], axis=0) @ plane[:2].T
    if (np.linalg.norm(delta, axis=1) / np.diff(times[picks]) < .25).any():
        return None
    total = delta.sum(axis=0)
    head = np.array([1., 0.])
    if not 7.5 <= angle_deg(head, total) <= 30. or angle_deg(*delta) > 15.:
        return None
    return total / np.linalg.norm(total)


def one_observation(data, sensor, row, image):
    index = data["depth_index"][row]
    if index < 0:
        return None
    plane = walking_plane(data, index)
    if plane is None:
        return None
    depth = image.astype(np.float32) * .001
    head_pose = np.eye(4)
    head_pose[:3, :3] = plane @ data["T_local"][index, :3, :3]
    sensor_pose = head_pose @ sensor.head_from_depth
    cloud = sensor.cloud(depth, sensor_pose)
    origin, head = np.zeros(3), np.array([1., 0.])
    occupied, near = in_corridor(cloud, origin, head, HALF_WIDTH)
    head_hit = occupied >= MIN_SOLID_ANGLE
    can_veto = head_hit & (near < MIN_SOLID_ANGLE)
    veto = np.zeros(2, dtype=bool)
    stage = "no_current_distant_hit"
    if can_veto.any():
        stage = "motion_gate"
        direction = reliable_direction(data, index, plane)
        if direction is not None:
            rotated, _ = in_corridor(cloud, origin, direction, HALF_WIDTH + MARGIN)
            coverage = sensor.clear_fraction(samples(origin, direction, HALF_WIDTH + MARGIN), depth, sensor_pose)
            unobstructed = can_veto & (rotated == 0)
            veto = unobstructed & (coverage >= .95)
            stage = "veto" if veto.any() else "coverage_gate" if unobstructed.any() else "observed_occupancy"
    return head_hit, veto, stage


def predictions(prepared, native):
    keys = ("T_local", "ts", "local_down", "component", "depth_index", "depth_timestamp_us")
    with np.load(prepared, allow_pickle=False) as data:
        local = {key: data[key] for key in keys}
        sensor = DepthSensor(data["depth_lut"], data["T_depth_to_head"])
    with np.load(native, allow_pickle=False) as recorded:
        rows, stamps, images = recorded["depth_rows"], recorded["timestamp_us"], recorded["depth_mm"]
        if not np.array_equal(local["depth_timestamp_us"][rows], stamps) or images.shape != (len(rows), *sensor.lut.shape[:2]):
            raise ValueError(f"Depth was not exactly paired for {prepared.stem}")
        obs = {int(row): one_observation(local, sensor, int(row), img) for row, img in zip(rows, images, strict=True)}
    outputs = {}
    for row, now in obs.items():
        latest = [obs.get(row - step) for step in (2, 1, 0)]
        if row < 2 or any(v is None for v in latest):
            continue
        timeline = local["depth_timestamp_us"][row - 2:row + 1]
        if np.diff(timeline).max() > 600_000:
            continue
        pose_rows = local["depth_index"][row - 2:row + 1]
        if len(np.unique(local["component"][pose_rows])) != 1:
            continue
        base = np.stack([reading[0] for reading in latest]).sum(axis=0) >= 2
        outputs[row] = base, base & ~now[1], now[2]
    return outputs


def episode_pairs(rows, kind):
    episodes = []
    last = -np.inf
    for row in rows:
        if row["category"] != kind:
            last = -np.inf
            continue
        if row["time"] - last > 1.0:
            episodes.append([])
        episodes[-1].append(row)
        last = row["time"]
    return episodes


def evaluate_pass(geometry, scored):
    by_episode_type = {}
    for kind in ("head_occupied_travel_empty", "travel_occupied", "other_clear"):
        positive = kind == "travel_occupied"
        counts = dict(episodes_total=0, episodes_paired=0, head_alert=0, veto_alert=0, lost_alert=0)
        for visit in geometry["visits"]:
            pred = scored.get(visit["visit"], {})
            for episode in episode_pairs(visit["rows"], kind):
                counts["episodes_total"] += 1
                if not all(row["depth_row"] in pred for row in episode):
                    continue
                counts["episodes_paired"] += 1
                pair = np.zeros(2, dtype=bool)
                for row in episode:
                    base, candidate, _ = pred[row["depth_row"]]
                    mask = np.asarray(row["travel"]) == 1 if positive else np.ones(2, dtype=bool)
                    pair |= np.array([(base & mask).any(), (candidate & mask).any()])
                counts["head_alert"] += int(pair[0])
                counts["veto_alert"] += int(pair[1])
                counts["lost_alert"] += int(pair[0] and not pair[1])
        by_episode_type[kind] = counts
    return by_episode_type


def causal_check():
    times = np.arange(64) / 30.
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, 2] = [1., 0., 0.]
    poses[:, 0, 3], poses[:, 1, 3] = times, .23 * times
    data = dict(ts=times, T_local=poses, local_down=np.tile([0., 0., -1.], (len(times), 1)),
                component=np.zeros(len(times), dtype=int))
    current = 31
    plane = walking_plane(data, current)
    direction = reliable_direction(data, current, plane)
    assert direction is not None
    altered = poses.copy()
    altered[current + 1:] *= -100.
    changed = dict(data, T_local=altered)
    np.testing.assert_array_equal(plane, walking_plane(changed, current))
    np.testing.assert_array_equal(direction, reliable_direction(changed, current, plane))


def replay(args):
    causal_check()
    geom = json.loads(args.geometry.read_text(encoding="utf-8"))
    scored = {}
    for path in args.native:
        scored[path.stem] = predictions(args.prepared / f"{path.stem}.npz", path)
    result = evaluate_pass(geom, scored)
    robust = dict(visits=[dict(visit, rows=[dict(row, category=row["robust_category"], travel=row["travel_robust"])
                                          for row in visit["rows"]]) for visit in geom["visits"]])
    robust_results = evaluate_pass(robust, scored)
    changes = dict(known_positive_band_removed=0, known_negative_band_removed=0, unknown_band_removed=0)
    false_alert_gates = Counter()
    decisions = []
    for visit in geom["visits"]:
        for row in visit["rows"]:
            pred = scored.get(visit["visit"], {}).get(row["depth_row"])
            if pred is None:
                continue
            base, veto, stage = pred
            removed = base & ~veto
            for label, key in ((1, "known_positive_band_removed"), (0, "known_negative_band_removed"), (-1, "unknown_band_removed")):
                changes[key] += int((removed & (np.asarray(row["travel"]) == label)).sum())
            if base.any() and row["category"] == "head_occupied_travel_empty":
                false_alert_gates[stage] += 1
            decisions.append(dict(session=visit["visit"], timestamp_us=row["timestamp"],
                                  baseline=base.tolist(), candidate=veto.tolist(), gate=stage))
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(dict(episodes=result, robust_episodes=robust_results, band_changes=changes,
                       false_alert_gates=false_alert_gates, predictions=decisions), out, indent=2)
    print(result, robust_results, changes, false_alert_gates, flush=True)


def main():
    main_parser = argparse.ArgumentParser(description=__doc__)
    runs = main_parser.add_subparsers(dest="mode", required=True)
    audit = runs.add_parser("audit")
    audit.add_argument("prepared", type=Path)
    audit.add_argument("samples", type=Path)
    audit.add_argument("crop", type=Path)
    audit.add_argument("output", type=Path)
    compare = runs.add_parser("compare")
    compare.add_argument("prepared", type=Path)
    compare.add_argument("geometry", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("native", nargs="+", type=Path)
    args = main_parser.parse_args()
    if args.mode == "audit":
        tof_mesh_check(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
