"""Conditional replay of wall-patch occupancy attribution in previously scored HoloSet frames.

Inputs are the frozen cohort, completed D8 scores/cached depth, and score-blind opaque
plain-wall polygons in native PV pixels. See head/attribution/PLAN.md for selection.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

import real_wall_holoset_score_20261118 as original


def select_targets(cohort, scores, polygons):
    by_id = {(r["seq"], r["depth_ts"]): r for r in polygons}
    first = {}
    for row in sorted(cohort, key=lambda r: (r["seq"], r["depth_ts"])):
        seq, ts = row["seq"], row["depth_ts"]
        key = (seq, ts)
        if row["stratum"] == "low" and key in by_id and scores[key]["alert"]:
            first.setdefault(seq, by_id[key])
    return list(first.values())[:8]


def cached_frame(cap, row, path, max_m):
    pose, center, rays, maps, ref = cap.registration(row)
    mono = np.load(path)
    valid = (ref > 0) & (ref < max_m) & np.isfinite(mono) & (mono > 0)
    if valid.sum() <= 100:
        raise ValueError(f"Unusable reference at {path}")
    scale = float(np.median(ref[valid] / mono[valid]))
    depth = np.where(ref > 0, mono * scale, 0.)
    ok = (depth > 0) & (depth < max_m)
    def world(z):
        return ((rays[ok] * z[ok, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return dict(ts=row["depth_ts"], pose=pose, center=center, depth=depth,
                points=world(depth), measured=world(ref), ref=ref, maps=maps)


def polygon_on_grid(cur, rec, cap):
    points = np.asarray(rec["polygon"], dtype=np.int32)
    if not (points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3):
        raise ValueError(f"Invalid polygon {rec}")
    if np.any(points < 0) or np.any(points >= [cap.width, cap.height]):
        raise ValueError("Wall polygon is outside native PV")
    mask = np.zeros((cap.height, cap.width), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.remap(mask, *cur["maps"], cv2.INTER_NEAREST).astype(bool)


def inside_volume(points, origin, right, fwd, floor, xmin, xmax, sign):
    d = points - origin
    x, y, z = d @ right, d @ fwd, points[:, 2] - floor
    lateral = abs(x) if sign == 0 else sign * x
    return (lateral >= xmin) & (lateral <= xmax) & (y >= .30) & (y <= 2.) & (z >= .60) & (z <= 1.90)


def replay_target(data, score_root, polygon, score, model, torch, A, V):
    seq, ts = polygon["seq"], polygon["depth_ts"]
    cap = original.Recording(data / seq)
    rows = sorted(original.records(cap.directory / "geometry.jsonl"), key=lambda r: r["depth_ts"])
    rows = [r for r in rows if ts-original.WINDOW <= r["depth_ts"] <= ts and
            r["tag"] not in ("no_depth_pose", "no_rgb_match") and
            original.rigid(cap.rig[r["depth_ts"]]) and original.rigid(cap.pv[r["rgb_ts"]][2])]
    if not rows or rows[-1]["depth_ts"] != ts or len(rows) != score["history"]:
        raise ValueError("Original temporal support does not reproduce")
    cache = score_root / seq / "predicted"
    history = [cached_frame(cap, row, cache / f"{row['depth_ts']}.npy", A.MAX_MM * .001) for row in rows]
    cur = history[-1]
    floor = rows[-1]["floor_y"]
    x = original.features(history, floor, V)
    def infer(features):
        with torch.inference_mode():
            t = torch.from_numpy(features[None]).float().to(next(model.parameters()).device)
            return float(torch.sigmoid(model(V.pad(t))).item())
    before = infer(x)
    if abs(before - score["score"]) > 1e-5:
        raise ValueError(f"Original frozen D8 score mismatch {before} vs {score}")
    region = polygon_on_grid(cur, polygon, cap)
    pose = cur["pose"]
    origin = pose[:3, 3]
    forward = pose[:3, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    side_sign = {"right": 1, "left": -1}[polygon["side"]]
    hp, wp = cur["ref"].shape
    changed = []
    dropped = []
    support_counts = []
    for entry in history:
        truth = entry["measured"]
        pred = entry["points"]
        match = inside_volume(truth, origin, right, forward, floor, .35, 1.20, side_sign)
        candidate = np.flatnonzero(match)
        projected = (truth[candidate] - origin) @ pose[:3, :3]
        z = projected[:, 2]
        coords = original.FOCAL * projected[:, :2] / np.maximum(z[:, None], .01) + cur["center"]
        u, v = np.rint(coords / original.STEP).astype(np.int32).T
        visible = (z > .05) & (u >= 0) & (u < wp) & (v >= 0) & (v < hp)
        candidate, v, u, z = candidate[visible], v[visible], u[visible], z[visible]
        observed = cur["ref"][v, u]
        same_patch = region[v, u] & (observed > 0) & (abs(observed - z) <= .05)
        verified = candidate[same_patch]
        support_counts.append(len(verified))
        remove = verified[inside_volume(pred[verified], origin, right, forward, floor, 0., .25, 0)]
        keep = np.ones(len(pred), bool)
        keep[remove] = False
        changed.append({**entry, "points": pred[keep]})
        dropped.append(len(remove))
    after_x = original.features(changed, floor, V)
    if not np.array_equal(x[2:], after_x[2:]) or (after_x[:2] > x[:2]).any():
        raise ValueError("Intervention changed non-occupancy channels or added hits")
    after = infer(after_x)
    return dict(seq=seq, depth_ts=ts, rgb_ts=rows[-1]["rgb_ts"], side=polygon["side"],
                polygon=polygon["polygon"], primary_score=score["score"], replay=before,
                after=after, alert_after=after > .5367, dropped_current=dropped[-1],
                dropped_history=sum(dropped[:-1]), dropped_per_frame=dropped,
                matched_wall_pixels_current=support_counts[-1],
                matched_wall_observations_history=sum(support_counts[:-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("data","cohort","scores","polygons","out"):
        ap.add_argument(key,type=Path)
    ap.add_argument("--posthoc-all-low-alerts", action="store_true",
                    help="Explore every low-tercile alert with a reviewed wall polygon")
    ap.add_argument("--device", default="cuda", choices=("cpu","cuda"))
    a = ap.parse_args()
    cohort = original.records(a.cohort)
    digest = sha256(a.cohort.read_bytes()).hexdigest()
    lookup = {}
    for seq in sorted({r["seq"] for r in cohort}):
        item = json.loads((a.scores/seq/"scores.json").read_text())
        if item["cohort_sha256"] != digest:
            raise ValueError("Expected completed score with same cohort")
        lookup.update({(seq,r["depth_ts"]):r for r in item["scores"]})
    if set(lookup) != {(r["seq"],r["depth_ts"]) for r in cohort}:
        raise ValueError("Incomplete primary scores")
    polys = original.records(a.polygons)
    if len({(r["seq"], r["depth_ts"]) for r in polys}) != len(polys):
        raise ValueError("More than one polygon row per case")
    low_alerts = sorted((r["seq"], r["depth_ts"]) for r in cohort
                        if r["stratum"] == "low" and lookup[r["seq"], r["depth_ts"]]["alert"])
    if a.posthoc_all_low_alerts:
        by_id = {(r["seq"], r["depth_ts"]): r for r in polys}
        if not by_id.keys() <= set(low_alerts):
            raise ValueError("An exploratory polygon does not belong to a low-tercile alert")
        picked = [by_id[key] for key in low_alerts if key in by_id]
    else:
        picked = select_targets(cohort, lookup, polys)
    results = []
    if picked:
        import torch
        from spires_insert_dev_20261117 import A, I, V
        weights = a.data.parents[1]/"ase-cane-reach-pilot-20261015"/"decision_head8_all.pt"
        torch.set_num_threads(2)
        net = I.D.make_head(8)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        net.to(a.device).eval()
        for poly in picked:
            key = poly["seq"],poly["depth_ts"]
            results.append(replay_target(a.data,a.scores,poly,lookup[key],net,torch,A,V))
    receipt = dict(cohort_sha256=digest, polygon_rows=len(polys), targets=len(picked), results=results)
    if a.posthoc_all_low_alerts:
        receipt.update(selection="post-hoc census of low-tercile alerts", low_alerts=len(low_alerts),
                       polygon_sha256=sha256(a.polygons.read_bytes()).hexdigest(),
                       without_confirmed_wall_roi=[
                           dict(seq=seq, depth_ts=ts, score=lookup[seq, ts]["score"],
                                reason="No visually confirmed opaque plain wall interior ROI; see case review")
                           for seq, ts in low_alerts if (seq, ts) not in by_id])
    a.out.write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(json.dumps(receipt,allow_nan=False))


if __name__ == "__main__":
    main()
