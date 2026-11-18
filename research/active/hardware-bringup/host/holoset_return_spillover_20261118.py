"""Replay cached image edits with the tube always unknown in the two reverse alerts.

Run from the BlindAssist checkout. Raw-depth-only and global-scale-only substitutions
use the same original active range mask and per-exposure full-frame calibration rule.
"""
import argparse
import json

import cv2
import numpy as np
import torch

import holoset_tube_rgb_20261118 as rgb
from spires_insert_dev_20261117 import A, I, V


SEQ = "research-center-seq4"
OUT = rgb.BASE / "holoset-return-20261118/spillover"
MAX_M = A.MAX_MM * .001


def read_exposures():
    cap = rgb.replay.Recording(rgb.HEAD / SEQ)
    timeline = {r["depth_ts"]: r for r in rgb.replay.records(cap.directory / "geometry.jsonl")}
    frames = []
    for entry in rgb.histories()[SEQ]:
        ts, rt = entry["depth_ts"], entry["rgb_ts"]
        row = timeline[ts]
        pose, center, dirs, maps, ref = cap.registration(row)
        outer = rgb.read(rgb.MASKS / "outer" / SEQ / f"{rt}.png", cv2.IMREAD_GRAYSCALE)
        control = rgb.read(rgb.MASKS / "control" / SEQ / f"{rt}.png", cv2.IMREAD_GRAYSCALE)
        near = cv2.distanceTransform((outer == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        edge = cv2.morphologyEx(control, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        control_dist = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        near = cv2.remap(near, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        control_dist = cv2.remap(control_dist, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        outer = cv2.remap(outer, *maps, cv2.INTER_NEAREST) > 0
        raw, scales = {}, {}
        for arm in rgb.ARMS:
            depth = np.load(rgb.OUT / "predicted" / SEQ / arm / f"{ts}.npy")
            valid = (ref > 0) & (ref < MAX_M) & np.isfinite(depth) & (depth > 0)
            if valid.sum() <= 100:
                raise ValueError("Insufficient active depth for global alignment")
            raw[arm] = depth
            scales[arm] = float(np.median(ref[valid] / depth[valid]))
        archived = np.load(rgb.SCORES / SEQ / "predicted" / f"{ts}.npy")
        if not np.array_equal(archived, raw["original"]):
            raise ValueError(f"Original predictions differ at {ts}")
        frames.append(dict(ts=ts, rt=rt, pose=pose, center=center, dirs=dirs, ref=ref, outer=outer, maps=maps,
                           near=near, control_dist=control_dist, raw=raw, scales=scales, floor=row.get("floor_y")))
    return frames


def projected(frame, depth, exclude=True):
    used = frame["ref"] > 0
    if exclude:
        used &= ~frame["outer"]
    depth = np.where(used, depth, 0.)
    good = (depth > 0) & (depth < MAX_M)
    pose = frame["pose"]
    points = ((frame["dirs"][good] * depth[good, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return dict(ts=frame["ts"], pose=pose, center=frame["center"], depth=depth, points=points)


def features(history, arm, scale_arm, exclude=True):
    entries = [projected(f, f["raw"][arm] * f["scales"][scale_arm], exclude) for f in history]
    return rgb.replay.features(entries, history[-1]["floor"], V)


def time_scores(net, base, alt):
    # The scorer stores independent current and past channels on the same target-frame grid.
    if not np.array_equal(base[6:], alt[6:]):
        raise ValueError("Static coordinates changed")
    scores = {}
    for label, channels in (("current", [0, 2, 4]), ("past", [1, 3, 5]), ("all", list(range(6)))):
        x = base.copy()
        x[channels] = alt[channels]
        with torch.inference_mode():
            scores[label] = float(torch.sigmoid(net(V.pad(torch.from_numpy(x[None]).float()))).item())
    return scores


def margin_scores(net, hist, base):
    results = []
    for radius in (3, 8, 16, 32, 64):
        changed = {"delete": [], "measured": [], "control_delete": [], "control_measured": []}
        counts = []
        for frame in hist:
            valid = (frame["ref"] > 0) & ~frame["outer"]
            ring = valid & (frame["near"] <= radius) if frame["outer"].any() else np.zeros_like(valid)
            count = int(ring.sum())
            counts.append(count)
            candidates = np.flatnonzero(valid & ~ring)
            if len(candidates) < count:
                raise ValueError("Insufficient rays for disjoint matched control")
            order = np.argsort(frame["control_dist"].ravel()[candidates], kind="stable")
            control = np.zeros_like(valid)
            control.ravel()[candidates[order[:count]]] = True
            original = frame["raw"]["original"] * frame["scales"]["original"]
            for prefix, region in (("", ring), ("control_", control)):
                changed[prefix+"delete"].append(projected(frame, np.where(region, 0., original)))
                changed[prefix+"measured"].append(projected(frame, np.where(region, frame["ref"], original)))
        results.append(dict(radius_px=radius, valid_ray_counts=counts, scores={
            label: time_scores(net, base, rgb.replay.features(entries, hist[-1]["floor"], V))
            for label, entries in changed.items()
        }))
    return results


def model():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net = I.D.make_head(8)
    weights = rgb.BASE / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    return net, weights


def main():
    net, weights = model()
    exposures = read_exposures()
    ablated = json.loads((rgb.MASKS.parent / "d8/attribution.json").read_bytes())["results"]
    paired = {r["depth_ts"]: r["scores"] for r in json.loads((rgb.OUT / "scores.json").read_bytes()) if r["seq"] == SEQ}
    results = []
    for target in (r for r in ablated if r["seq"] == SEQ):
        ts = target["depth_ts"]
        hist = [f for f in exposures if ts - rgb.replay.WINDOW <= f["ts"] <= ts]
        if len(hist) != len(target["arms"]["outer"]["frames"]) or hist[-1]["ts"] != ts:
            raise ValueError("Original chronology differs")
        base = features(hist, "original", "original")
        baseline = time_scores(net, base, base)["all"]
        if abs(baseline - target["arms"]["outer"]["scores"]["all"]) > 1e-4:
            raise ValueError("Tube-out score not reproduced")
        result = dict(case=target["case"], depth_ts=ts, baseline=baseline, arms={})
        for arm in rgb.ARMS:
            full = features(hist, arm, arm, exclude=False)
            score = time_scores(net, full, full)["all"]
            if abs(score - paired[ts][arm]) > 1e-4:
                raise ValueError(f"Tube-in source replay differs: {arm}")
            arms = {}
            for name, raw, scale in (("fixed_scale", arm, "original"),
                                     ("scale_only", "original", arm), ("recalibrated", arm, arm)):
                alt = features(hist, raw, scale)
                arms[name] = time_scores(net, base, alt)
            result["arms"][arm] = arms
        measured = rgb.replay.features([projected(f, f["ref"]) for f in hist], hist[-1]["floor"], V)
        result["active_depth_outside_tube"] = time_scores(net, base, measured)
        result["native_margin_sensitivity"] = margin_scores(net, hist, base)
        results.append(result)
    if len(results) != 2:
        raise ValueError("Expected two return alerts")
    payload = dict(threshold=I.THETA, weight_sha256=rgb.replay.file_digest(weights),
                   frames=[dict(depth_ts=f["ts"], rgb_ts=f["rt"], scale=f["scales"]) for f in exposures],
                   results=results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for r in results:
        print(r["case"], r["baseline"], r["arms"]["tube_column"], r["active_depth_outside_tube"])


def localized(include_scene):
    net, _ = model()
    exposures = read_exposures()
    groups = {"right_top": ()}
    if include_scene:
        manifest = OUT.parent / "scene/manifest.json"
        manifest_sha = rgb.replay.file_digest(manifest)
        label_sha = {r["rgb_ts"]: r["label_sha256"] for r in json.loads(manifest.read_bytes())["frames"]}
        groups.update({"label15": (15,), "label8": (8,), "label10": (10,), "label16": (16,),
                       "label17": (17,), "label0": (0,), "labels15_17": (15, 17),
                       "labels15_17_8": (15, 17, 8),
                       "labels15_8": (15, 8), "labels15_8_10": (15, 8, 10),
                       "labels15_8_10_16": (15, 8, 10, 16)})
    for frame in exposures:
        u, v = (np.rint(a).astype(int) for a in frame["maps"])
        frame["regions"] = {"right_top": (u >= 380) & (u < 760) & (v >= 0) & (v < 107)}
        if include_scene:
            path = OUT.parent / "scene/labels" / f"{frame['rt']}.png"
            if rgb.replay.file_digest(path) != label_sha[frame["rt"]]:
                raise ValueError("Scene map differs from its frozen manifest")
            label = cv2.remap(rgb.read(path, cv2.IMREAD_UNCHANGED), *frame["maps"], cv2.INTER_NEAREST)
            frame["regions"].update({k: np.isin(label, ids) for k, ids in groups.items() if ids})
            top = frame["regions"]["right_top"]
            main_surfaces = np.isin(label, (15, 17, 8))
            frame["regions"]["right_top_15_17_8"] = top & main_surfaces
            frame["regions"]["right_top_others"] = top & ~main_surfaces
            frame["regions"]["right_top_10_11_14"] = top & np.isin(label, (10, 11, 14))
    prior = json.loads((OUT / "result.json").read_bytes())
    output = []
    for target in prior["results"]:
        ts = target["depth_ts"]
        hist = [f for f in exposures if ts - rgb.replay.WINDOW <= f["ts"] <= ts]
        base = features(hist, "original", "original")
        case = dict(case=target["case"], depth_ts=ts, arms={})
        if include_scene:
            case["scene_manifest_sha256"] = manifest_sha
        for arm in ("tube_column", "control_column"):
            overall = features(hist, arm, "original")
            arm_result = {}
            for region in hist[-1]["regions"]:
                by_method = {}
                for method, background, foreground, before in (("replace", "original", arm, base),
                                                               ("restore", arm, "original", overall)):
                    mixed, recent = [], []
                    for f in hist:
                        mask = f["regions"][region]
                        scale = f["scales"]["original"]
                        raw = f["raw"]
                        changed = projected(f, np.where(mask, raw[foreground], raw[background]) * scale)
                        mixed.append(changed)
                        if 0 < ts - f["ts"] <= 10_010_000:
                            recent.append(changed)
                        else:
                            recent.append(projected(f, raw[background] * scale))
                    x = rgb.replay.features(mixed, hist[-1]["floor"], V)
                    scores = time_scores(net, before, x)
                    xr = rgb.replay.features(recent, hist[-1]["floor"], V)
                    scores["past_1s"] = time_scores(net, before, xr)["all"]
                    by_method[method] = scores
                arm_result[region] = by_method
            case["arms"][arm] = arm_result
        output.append(case)
    name = "localized.json" if include_scene else "localized-rect.json"
    (OUT / name).write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    for r in output:
        print(r["case"], r["arms"]["tube_column"]["right_top"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--localized", choices=("rect", "scene"))
    args = ap.parse_args()
    if args.localized:
        localized(include_scene=args.localized == "scene")
    else:
        main()
