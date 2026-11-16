"""Causal corridor adjustment on public depth and camera-pose clips."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np


CONFIGS = list(product((15, 25), (0.65, 0.80, 0.95)))
BANDS = ((-0.20, 0.42), (0.42, 0.90))
MIN_SOLID_ANGLE = 8.3e-4
NEAR, FAR = 0.30, 2.00
HALF_WIDTH = 0.30
MARGIN = 0.10


def cross_xy(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def angle_deg(a, b):
    return np.degrees(np.arctan2(abs(cross_xy(a, b)), a @ b))


def motion_votes(poses, times, lag, max_gap):
    positions = poses[:, :2, 3]
    head_xy = poses[:, :2, 2]
    norms = np.linalg.norm(head_xy, axis=1)
    head = head_xy / np.maximum(norms[:, None], 1e-9)
    direction = head.copy()
    agreement = np.full(len(times), np.inf)
    for i in range(2 * lag, len(times)):
        a, b = i - 2 * lag, i - lag
        if norms[i] < .30 or np.max(np.diff(times[a:i + 1])) > max_gap:
            continue
        first, second = positions[b] - positions[a], positions[i] - positions[b]
        if min(np.linalg.norm(first) / (times[b] - times[a]),
               np.linalg.norm(second) / (times[i] - times[b])) < .25:
            continue
        delta = first + second
        if np.linalg.norm(delta) == 0 or not 7.5 <= angle_deg(delta, head[i]) <= 30:
            continue
        agreement[i] = angle_deg(first, second)
        direction[i] = delta / np.linalg.norm(delta)
    return head, direction, agreement


def make_cloud(depth, confidence, pose, ray, weights):
    depth = depth[::2, ::2]
    valid = (confidence[::2, ::2] >= 2) & np.isfinite(depth) & (depth > .1) & (depth < 8)
    points = ray[valid] * depth[valid, None]
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32), weights[valid]


def in_corridor(cloud, origin, forward, half_width):
    xyz, weights = cloud
    xy = xyz[:, :2] - origin[:2]
    z = xy @ forward
    x = xy @ np.array([forward[1], -forward[0]])
    y = origin[2] - xyz[:, 2]
    planar = (np.abs(x) <= half_width) & (z >= NEAR) & (z <= FAR)
    inside = [planar & (y >= lo) & (y < hi) for lo, hi in BANDS]
    return np.array([weights[m].sum() for m in inside]), np.array(
        [weights[m & (z <= .65)].sum() for m in inside])


def samples(origin, forward, half_width):
    right = np.array([forward[1], -forward[0]])
    x, z, h = np.meshgrid(
        np.linspace(-.8 * half_width, .8 * half_width, 3),
        [.46, .85, 1.25, 1.75], [.2, .5, .8], indexing="ij")
    xy = origin[:2] + x.ravel()[:, None] * right + z.ravel()[:, None] * forward
    return np.stack([
        np.column_stack([xy, origin[2] - (lo + h.ravel() * (hi - lo))])
        for lo, hi in BANDS
    ])


def free_rays(target, depth, confidence, pose, intrinsics):
    fx, fy, cx, cy = intrinsics
    q = (target - pose[:3, 3]) @ pose[:3, :3]
    z = q[..., 2]
    scale = np.where(z > .1, z, 1.)
    u = np.rint(fx * q[..., 0] / scale + cx).astype(int)
    v = np.rint(fy * q[..., 1] / scale + cy).astype(int)
    h, w = depth.shape
    inside = (z > .1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    observed = depth[v, u]
    return inside & (confidence[v, u] >= 2) & (observed > z + .08) & (observed < 8)


def confirmed(raw, times, max_gap):
    output = np.zeros_like(raw)
    good = np.diff(times) <= max_gap
    hits = raw[:-2].astype(np.uint8) + raw[1:-1] + raw[2:]
    output[2:] = (hits >= 2) & (good[:-1] & good[1:])[:, None]
    return output


def predict(depth, confidence, poses, times, intrinsics, clouds, lag, max_gap):
    head, directions, agreement = motion_votes(poses, times, lag, max_gap)
    n = len(depth)
    head_raw = np.zeros((n, 2), dtype=bool)
    close = np.zeros_like(head_raw)
    rotated_raw = np.zeros_like(head_raw)
    for i, cloud in enumerate(clouds):
        origin = poses[i, :3, 3]
        occupied, near = in_corridor(cloud, origin, head[i], HALF_WIDTH)
        head_raw[i] = occupied >= MIN_SOLID_ANGLE
        close[i] = near >= MIN_SOLID_ANGLE
        if np.isfinite(agreement[i]):
            occupied, _ = in_corridor(cloud, origin, directions[i], HALF_WIDTH)
            rotated_raw[i] = occupied >= MIN_SOLID_ANGLE
        else:
            rotated_raw[i] = head_raw[i]
    base = confirmed(head_raw, times, max_gap)
    rotate = confirmed(rotated_raw, times, max_gap)
    withdraw = np.zeros((len(CONFIGS), n, 2), dtype=bool)
    for i in np.flatnonzero((agreement <= 25) & (base & head_raw & ~close).any(axis=1)):
        origin, direction = poses[i, :3, 3], directions[i]
        occupied, _ = in_corridor(clouds[i], origin, direction, HALF_WIDTH + MARGIN)
        if not np.any(occupied == 0):
            continue
        targets = samples(origin, direction, HALF_WIDTH + MARGIN)
        coverage = free_rays(targets, depth[i], confidence[i], poses[i], intrinsics).mean(axis=1)
        for j, (max_disagreement, min_coverage) in enumerate(CONFIGS):
            withdraw[j, i] = ((agreement[i] <= max_disagreement)
                              & base[i] & head_raw[i] & ~close[i]
                              & (occupied == 0) & (coverage >= min_coverage))
    return base, rotate, withdraw, agreement


def future_reference(depth, confidence, poses, times, intrinsics, clouds, lag, max_gap):
    labels = np.full((len(depth), 2), -1, dtype=np.int8)
    eligible = np.zeros(len(depth), dtype=bool)
    for i in range(len(depth) - 3 * lag):
        j, k = i + 2 * lag, i + 3 * lag
        if np.max(np.diff(times[i:k + 1])) > max_gap:
            continue
        origin = poses[i, :3, 3]
        delta = poses[j, :2, 3] - origin[:2]
        distance = np.linalg.norm(delta)
        if distance / (times[j] - times[i]) < .30:
            continue
        eligible[i] = True
        direction = delta / distance
        occ_j, _ = in_corridor(clouds[j], origin, direction, HALF_WIDTH)
        occ_k, _ = in_corridor(clouds[k], origin, direction, HALF_WIDTH)
        positive = (occ_j >= MIN_SOLID_ANGLE) & (occ_k >= MIN_SOLID_ANGLE)
        labels[i, positive] = 1
        absent = (occ_j == 0) & (occ_k == 0)
        if absent.any():
            target = samples(origin, direction, HALF_WIDTH)
            seen = free_rays(target, depth[j], confidence[j], poses[j], intrinsics)
            seen |= free_rays(target, depth[k], confidence[k], poses[k], intrinsics)
            labels[i, absent & (seen.mean(axis=1) >= .75)] = 0
    return labels, eligible


def self_check():
    # Camera facing world +X, right -Y, image down -Z.
    pose = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                     [0., -1., 0., 1.5], [0., 0., 0., 1.]])
    poses = np.tile(pose, (10, 1, 1))
    poses[:, 0, 3] = np.arange(10) * .20
    poses[:, 1, 3] = np.arange(10) * .04
    times = np.arange(10) * .40
    a = motion_votes(poses, times, 1, .6)
    perturb = poses.copy()
    perturb[6:, :2, 3] *= -10
    for original, modified in zip(a, motion_votes(perturb, times, 1, .6)):
        np.testing.assert_array_equal(original[:6], modified[:6])
    assert np.isfinite(a[2][2:]).all()
    assert (a[1][2:, 1] > 0).all()

    point = np.array([[1.35, -.45, 1.4]])
    angle = np.arctan2(.45, 1.35)
    assert in_corridor((point, np.array([1.])), pose[:3, 3], np.array([1., 0.]), .3)[0][0] == 0
    turned = np.array([np.cos(angle), -np.sin(angle)])
    assert in_corridor((point, np.array([1.])), pose[:3, 3], turned, .3)[0][0] == 1

    target = np.array([[[1., 0., 1.5]]])
    values = np.ones((50, 50)) * 2
    confidence = np.full((50, 50), 2)
    assert free_rays(target, values, confidence, pose, (40, 40, 25, 25)).item()
    assert not free_rays(target, values * .25, confidence, pose, (40, 40, 25, 25)).item()
    assert not free_rays(target, values, confidence * 0, pose, (40, 40, 25, 25)).item()

    # A side obstacle can be removed only when the new corridor is observed.
    depth = np.full((len(poses), 96, 120), 4.0, dtype=np.float32)
    high = np.full(depth.shape, 2, dtype=np.uint8)
    clouds = [(p[:3, 3][None, :] + [1.35, -.27, -.10], np.array([.001]))
              for p in poses]
    common = (poses, times, (30, 30, 60, 48), clouds, 1, .6)
    head, _, withdrawal, _ = predict(depth, high, *common)
    assert head[2:, 0].all() and withdrawal[2, 2:, 0].all()
    _, _, not_observed, _ = predict(depth, high * 0, *common)
    assert not not_observed.any()
    changed_depth = depth.copy()
    changed_depth[6:] = 0
    _, _, changed_withdrawal, _ = predict(changed_depth, high, *common)
    np.testing.assert_array_equal(changed_withdrawal[:, :6], withdrawal[:, :6])


def score_capture(path, sanpo=False):
    with np.load(path, allow_pickle=False) as data:
        depth, times = data["depth"], data["ts"]
        if sanpo:
            confidence = data["valid"].astype(np.uint8) * 2
            poses = data["T_pose_raw"]
            fx, fy, cx, cy, width, height = data["K"]
            intrinsics = (fx, fy, cx, cy)
        else:
            confidence, poses = data["conf"], data["T_wc"]
            width, height, *intrinsics = data["K"]
    if depth.shape != confidence.shape or depth.shape != (len(times), int(height), int(width)):
        raise ValueError(f"Bad depth/pose contract in {path}.")
    if poses.shape != (len(times), 4, 4) or not np.isfinite(poses).all() or not (np.diff(times) > 0).all():
        raise ValueError(f"Bad pose sequence in {path}.")
    depth = depth.astype(np.float32) * .001 if depth.dtype.kind in "ui" else depth.astype(np.float32)
    lag = max(1, int(round(.4 / np.median(np.diff(times)))))
    max_gap = .60
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[:int(height):2, :int(width):2]
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1).astype(np.float32)
    weights = (4. / (fx * fy) / np.power((ray ** 2).sum(axis=-1), 1.5)).astype(np.float32)
    clouds = [make_cloud(d, c, p, ray, weights) for d, c, p in zip(depth, confidence, poses)]
    head, rotate, withdraw, agreement = predict(depth, confidence, poses, times, intrinsics, clouds, lag, max_gap)
    labels, eligible = future_reference(depth, confidence, poses, times, intrinsics, clouds, lag, max_gap)
    assert not (withdraw & ~head).any()
    return dict(head=head, rotate=rotate, withdraw=withdraw, labels=labels,
                eligible=eligible, stable=(agreement <= 25))


def confusion(labels, pred, head):
    return {
        "fp": int((pred & (labels == 0)).sum()),
        "fn": int((~pred & (labels == 1)).sum()),
        "fp_removed": int((head & ~pred & (labels == 0)).sum()),
        "fn_added": int((head & ~pred & (labels == 1)).sum()),
        "unknown_removed": int((head & ~pred & (labels < 0)).sum()),
    }


def summarize(out):
    head, labels = out["head"], out["labels"]
    choices = [confusion(labels, head & ~arm, head) for arm in out["withdraw"]]
    return {
        "frames": len(head),
        "future_moving": int(out["eligible"].sum()),
        "stable": int(out["stable"].sum()),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "unknown_moving": int((labels[out["eligible"]] < 0).sum()),
        "head": confusion(labels, head, head),
        "rotate": confusion(labels, out["rotate"], head),
        "choices": choices,
    }


def combine(outputs):
    return {name: np.concatenate([x[name] for x in outputs], axis=1 if name == "withdraw" else 0)
            for name in outputs[0]}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("frames", type=Path)
    args.add_argument("split", type=Path)
    args.add_argument("output", type=Path)
    args.add_argument("--sanpo", action="store_true")
    cfg = args.parse_args()
    if cfg.sanpo:
        # This separate source is evaluated only with the fixed strict setting.
        if set(json.loads(cfg.split.read_text(encoding="utf-8"))) != {"evaluation"}:
            args.error("SANPO requires an evaluation-only split.")
    summary_path = cfg.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    self_check()
    start = perf_counter()
    split = json.loads(cfg.split.read_text(encoding="utf-8"))
    reports = {}
    for role, rows in split.items():
        already = set()
        by_visit, results = {}, []
        for row in rows:
            if row["visit"] in already:
                raise ValueError("Multiple captures from a single visit")
            already.add(row["visit"])
            out = score_capture(cfg.frames / f'{row["capture"]}.npz', sanpo=cfg.sanpo)
            by_visit[row["visit"]] = summarize(out)
            results.append(out)
        combined = combine(results)
        reports[role] = dict(summary=summarize(combined), visits=by_visit)
        with (cfg.output / f"{role}_decisions.npz").open("xb") as stream:
            np.savez_compressed(stream, **combined)
    options = reports.get("calibration", {}).get("summary", {}).get("choices", [])
    keep = [i for i, m in enumerate(options) if m["fn_added"] == 0 and m["fp_removed"] > 0]
    selected = max(keep, key=lambda i: (options[i]["fp_removed"], -options[i]["unknown_removed"],
                                       CONFIGS[i][1], -CONFIGS[i][0])) if keep else CONFIGS.index((15, .95))
    report = dict(configuration=CONFIGS, selected=selected, calibration_passed=bool(keep),
                  reports=reports, wall_seconds=round(perf_counter() - start, 3))
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({role: r["summary"] for role, r in reports.items()}, indent=2))
    print(f"Calibration supported: {bool(keep)}, selected {CONFIGS[selected]}, {report['wall_seconds']}s")


if __name__ == "__main__":
    main()
