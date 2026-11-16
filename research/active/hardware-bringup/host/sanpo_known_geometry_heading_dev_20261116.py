"""Analytic obstacle pairs on public SANPO camera trajectories."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from arkit_selective_heading_dev_20261116 import (
    CONFIGS, FAR, HALF_WIDTH, NEAR, confusion, make_cloud, predict, self_check,
)


def render(poses, ray, head_frame, center, origin):
    """Depth of the frontmost cuboid or room surface along each camera ray."""
    half_size = np.array([.06, .045, .39])
    depth = []
    for pose in poses:
        rays = ray @ pose[:3, :3].T
        direction = rays @ head_frame
        offset = (pose[:3, 3] - center) @ head_frame
        safe = np.where(np.abs(direction) > 1e-10, direction, 1e-10)
        face_one = (-half_size - offset) / safe
        face_two = (half_size - offset) / safe
        start = np.minimum(face_one, face_two).max(axis=-1)
        stop = np.maximum(face_one, face_two).min(axis=-1)
        thing = np.where((start > .1) & (stop >= start), start, np.inf)
        ground_axis = rays[..., 2]
        ground = (origin[2] - 1.5 - pose[2, 3]) / np.minimum(ground_axis, -1e-10)
        ground = np.where((ground_axis < 0) & (ground > .1), ground, np.inf)
        front = head_frame[:, 1]
        den = rays @ front
        wall = (4. - (pose[:3, 3] - origin) @ front) / np.maximum(den, 1e-10)
        wall = np.where((den > 0) & (wall > .1), wall, np.inf)
        depth.append(np.minimum(np.minimum(thing, ground), wall))
    depth = np.array(depth, dtype=np.float32)
    return depth, ((depth > .1) & (depth < 8)).astype(np.uint8) * 2


def reference(center, origin, frame, direction):
    lateral = np.array([direction[1], -direction[0]])
    corners = np.array([center + a * .06 * frame[:, 0] + b * .045 * frame[:, 1]
                        for a in (-1, 1) for b in (-1, 1)])[:, :2] - origin[:2]
    x, z = corners @ lateral, corners @ direction
    if x.min() > HALF_WIDTH or x.max() < -HALF_WIDTH or z.min() > FAR or z.max() < NEAR:
        return 0
    if np.max(np.abs(x)) <= HALF_WIDTH and z.min() >= NEAR and z.max() <= FAR:
        return 1
    return -1


def anchors(poses, times, lag):
    for i in range(2 * lag, len(poses) - 2 * lag):
        a, b = i - 2 * lag, i + 2 * lag
        delta = poses[b, :2, 3] - poses[i, :2, 3]
        length = np.linalg.norm(delta)
        if np.diff(times[a:b + 1]).max() <= .60 and length / (times[b] - times[i]) >= .30:
            yield i, delta / length


def capture(path):
    with np.load(path, allow_pickle=False) as data:
        poses, times = data["T_pose_raw"], data["ts"]
        fx, fy, cx, cy, native_w, native_h = data["K"]
    lag = max(1, int(round(.4 / np.median(np.diff(times)))))
    w, h = 192, 108
    fx, cx = fx * w / native_w, (cx + .5) * w / native_w - .5
    fy, cy = fy * h / native_h, (cy + .5) * h / native_h - .5
    v, u = np.mgrid[:h, :w]
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    ray_half = ray[::2, ::2].astype(np.float32)
    weights = 4. / (fx * fy) / np.power((ray_half**2).sum(axis=-1), 1.5)
    strict = CONFIGS.index((15, .95))
    rows = []
    for i, actual_direction in list(anchors(poses, times, lag))[:3]:
        o = poses[i, :3, 3]
        f = poses[i, :3, 2].copy()
        f[2] = 0
        norm = np.linalg.norm(f)
        if norm < .30:
            continue
        f /= norm
        lateral = np.array([f[1], -f[0], 0.])
        axes = np.column_stack([lateral, f, [0., 0., 1.]])
        past = slice(i - 2 * lag, i + 1)
        for distance in (1.25, 1.65):
            for side in (-.24, 0., .24):
                center = o + distance * f + side * lateral + [0., 0., -.33]
                depth, confidence = render(poses[past], ray, axes, center, o)
                clouds = [make_cloud(d, c, p, ray_half, weights)
                          for d, c, p in zip(depth, confidence, poses[past])]
                head, rotate, withdraw, _ = predict(
                    depth, confidence, poses[past], times[past], (fx, fy, cx, cy), clouds, lag, .6)
                rows.append(dict(frame=i, distance=distance, lateral=side,
                                 label=reference(center, o, axes, actual_direction),
                                 head=head[-1].tolist(), rotate=rotate[-1].tolist(),
                                 gated=(head[-1] & ~withdraw[strict, -1]).tolist()))
    return rows


def stats(rows):
    if not rows:
        return {"cases": 0}
    truth = np.repeat(np.array([r["label"] for r in rows])[:, None], 2, axis=1)
    base = np.array([r["head"] for r in rows])
    return dict(cases=len(rows), positives=int((truth == 1).sum()),
                negatives=int((truth == 0).sum()), unknown=int((truth < 0).sum()),
                **{arm: confusion(truth, np.array([r[arm] for r in rows]), base)
                   for arm in ("head", "rotate", "gated")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    self_check()
    start = perf_counter()
    paths = sorted(args.poses.glob("*.npz"))
    if len(paths) != 8:
        raise ValueError("Expected the frozen set of eight public pose windows.")
    per_window = {p.stem: capture(p) for p in paths}
    flat = [r for rows in per_window.values() for r in rows]
    report = dict(summary=stats(flat), per_window={p: stats(r) for p, r in per_window.items()},
                  rows=per_window, wall_seconds=round(perf_counter() - start, 3))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wall seconds: {report['wall_seconds']}")


if __name__ == "__main__":
    main()
