"""Ablate the original D8 evidence at the visually annotated tube in seven HoloSet alerts.

Run from the repository root. Arguments are local artifact directories. Masks are per
native PV exposure, including the individual frames that make up the three-second history.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

import real_wall_holoset_attribution_20261118 as replay
import real_wall_holoset_score_20261118 as original


def matched_surrounding(mask, outer, valid):
    """Use the same number of measured rays, nearest the masked area but outside the outline."""
    n = np.count_nonzero(mask & valid)
    result = np.zeros(mask.shape, bool)
    if not n:
        return result
    candidates = np.flatnonzero(~outer & valid)
    if len(candidates) < n:
        return None
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    order = np.argsort(distance.ravel()[candidates], kind="stable")
    result.ravel()[candidates[order[:n]]] = True
    return result


def boundary_regions(outer, core):
    """Vary the traced edge separately from the artificial edge where the tube leaves the image."""
    masks = {"outer_minus_core": outer & ~core}
    for radius in (3, 6, 10):
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        inset = cv2.erode(outer, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        masks[f"inset{radius}_keep_image_edge"] = inset
    interior = masks["inset10_keep_image_edge"]
    if np.any(core & ~interior):
        raise ValueError("The annotated core extends outside the replicated erosion")
    masks["traced_edge_band"] = outer & ~interior
    masks["image_edge_band"] = interior & ~core
    return masks


def load_masks(frame, cap, folder, seq, rt, boundary_check):
    regions = {}
    for name in ("core", "outer", "control"):
        path = folder / name / seq / f"{rt}.png"
        source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if source is None or source.shape != (cap.height, cap.width):
            raise ValueError(f"Missing or invalid source mask: {path}")
        regions[name] = source
    if boundary_check:
        regions = boundary_regions(regions["outer"], regions["core"])
    regions = {name: cv2.remap(source, *frame["maps"], cv2.INTER_NEAREST) > 0
               for name, source in regions.items()}
    if boundary_check:
        return regions
    if np.any(regions["core"] & ~regions["outer"]):
        raise ValueError(f"Core extends outside tube silhouette at {rt}")
    regions["control"] &= ~regions["outer"]
    for name in ("core", "outer"):
        regions[name + "_matched"] = matched_surrounding(regions[name], regions["outer"], frame["ref"] > 0)
    return regions


def inspect_points(points, pose, floor, V):
    delta = points - pose[:3, 3]
    forward = pose[:3, 2].copy()
    forward[2] = 0.
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    x, z, h = delta @ right, delta @ forward, points[:, 2] - floor
    grid = (abs(x) <= V.X_HALF) & (z >= 0) & (z < V.CELL * V.NF) & (h >= 0) & (h < V.CELL * V.NH)
    body = (abs(x) <= .25) & (z >= .30) & (z <= 2.0) & (h >= .60) & (h <= 1.90)
    return int(grid.sum()), int(body.sum())


def ablate(history, masks, floor, V, max_m):
    counts = []
    changed = []
    for frame, mask in zip(history, masks):
        if mask is None:
            return None, None
        depth = frame["depth"]
        in_range = (depth > 0) & (depth < max_m)
        selected = mask[in_range]
        grid, body = inspect_points(frame["points"][selected], history[-1]["pose"], floor, V)
        _, measured_body = inspect_points(frame["measured"][selected], history[-1]["pose"], floor, V)
        nr = int(np.count_nonzero(mask & (frame["ref"] > 0)))
        counts.append(dict(ts=frame["ts"], image_grid_pixels=int(mask.sum()), measured_rays=nr,
                           removed_points=int(selected.sum()), grid=grid, body=body,
                           measured_body=measured_body))
        changed.append({**frame, "points": frame["points"][~selected], "depth": np.where(mask, 0., depth)})
    return original.features(changed, floor, V), counts


def reference_replacement(history, masks, floor, V, max_m):
    filled = []
    for frame, mask in zip(history, masks):
        original_z = frame["depth"]
        depth = np.where(mask, frame["ref"], original_z)
        good = (depth > 0) & (depth < max_m)
        v, u = np.indices(depth.shape, dtype=np.float32) * original.STEP
        cx, cy = frame["center"]
        directions = np.stack([(u-cx)/original.FOCAL, (v-cy)/original.FOCAL, np.ones_like(u)], axis=-1)
        pose = frame["pose"]
        points = ((directions[good] * depth[good, None]) @ pose[:3,:3].T + pose[:3,3]).astype(np.float32)
        filled.append({**frame, "points": points, "depth": depth})
    return original.features(filled, floor, V)


def branch_scores(original_x, changed_x, model, torch, V, device):
    arms = dict(occupancy=[0, 1], visibility_free=[2, 3, 4, 5], current=[0, 2, 4],
                history=[1, 3, 5], all=list(range(6)), restore=[])
    scores = {}
    if not np.array_equal(original_x[6:], changed_x[6:]):
        raise ValueError("Coordinate channels changed")
    for name, indices in arms.items():
        features = original_x.copy()
        features[indices] = changed_x[indices]
        with torch.inference_mode():
            data = torch.from_numpy(features[None]).float().to(device)
            scores[name] = float(torch.sigmoid(model(V.pad(data))).item())
    return scores


def run(args):
    import torch
    from spires_insert_dev_20261117 import A, I, V
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = args.device
    net = I.D.make_head(8)
    weight_file = args.data.parents[1] / "ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    net.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True))
    net.to(device).eval()
    digest = sha256((args.data / "cohort.jsonl").read_bytes()).hexdigest()
    cases = json.loads((args.data / "alert-review/cases.json").read_bytes())
    cases = [r for r in cases if r["seq"] in ("research-center-seq3", "research-center-seq4")]
    results = []
    for seq in sorted({r["seq"] for r in cases}):
        cap = original.Recording(args.data / seq)
        base = json.loads((args.scores / seq / "scores.json").read_bytes())
        if base["cohort_sha256"] != digest:
            raise ValueError(f"{seq}: incorrect source cohort")
        scores = {r["depth_ts"]: r for r in base["scores"]}
        rows = [r for r in original.records(cap.directory / "geometry.jsonl")
                if r["tag"] not in ("no_depth_pose", "no_rgb_match")
                and original.rigid(cap.rig[r["depth_ts"]]) and original.rigid(cap.pv[r["rgb_ts"]][2])]
        rows.sort(key=lambda r: r["depth_ts"])
        cache = {}
        for case in (r for r in cases if r["seq"] == seq):
            ts = case["depth_ts"]
            selected = [r for r in rows if ts - original.WINDOW <= r["depth_ts"] <= ts]
            if selected[-1]["depth_ts"] != ts or len(selected) != scores[ts]["history"]:
                raise ValueError(f"{seq}:{ts}: original window differs")
            history, annotations = [], []
            for row in selected:
                dt = row["depth_ts"]
                if dt not in cache:
                    path = args.scores / seq / "predicted" / f"{dt}.npy"
                    frame = replay.cached_frame(cap, row, path, A.MAX_MM * .001)
                    masks = load_masks(frame, cap, args.masks, seq, row["rgb_ts"], args.boundary_check)
                    cache[dt] = (frame, masks)
                frame, masks = cache[dt]
                history.append(frame)
                annotations.append(masks)
            floor = selected[-1]["floor_y"]
            x = original.features(history, floor, V)
            zero = [np.zeros(e["depth"].shape, bool) for e in history]
            if not np.array_equal(x, reference_replacement(history, zero, floor, V, A.MAX_MM * .001)):
                raise ValueError("Reference-replacement null changed the original features")
            baseline = branch_scores(x, x, net, torch, V, device)["restore"]
            if abs(baseline - scores[ts]["score"]) > .0001:
                raise ValueError(f"{seq}:{ts}: baseline mismatch {baseline} vs {scores[ts]['score']}")
            arms = {}
            for name in annotations[-1]:
                masks = [data[name] for data in annotations]
                x2, counts = ablate(history, masks, floor, V, A.MAX_MM * .001)
                if counts is None:
                    arms[name] = {"status": "insufficient_non_tube_rays_for_matched_count"}
                    continue
                output = branch_scores(x, x2, net, torch, V, device)
                if output["restore"] != baseline:
                    raise ValueError("Restoration did not reproduce local baseline")
                aggregate = {k: sum(rec[k] for rec in counts) for k in counts[0] if k != "ts"}
                filled = reference_replacement(history, masks, floor, V, A.MAX_MM * .001)
                filled_scores = branch_scores(x, filled, net, torch, V, device)
                arms[name] = dict(scores=output, reference_fill=filled_scores, totals=aggregate, current=counts[-1], frames=counts,
                                  changed_cells=[int(np.count_nonzero(a != b)) for a,b in zip(x,x2)])
            results.append(dict(seq=seq, depth_ts=ts, rgb_ts=selected[-1]["rgb_ts"], case=case["case"],
                                baseline=baseline, archived=scores[ts]["score"], arms=arms))
            print(seq, case["case"], baseline, {k:v.get("scores", {}).get("all") for k,v in arms.items()}, flush=True)
    if len(results) != 7:
        raise ValueError("Expected seven alerted times")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(cohort_sha256=digest, model_sha256=original.file_digest(weight_file),
                                        mask_manifest_sha256=original.file_digest(args.masks / "manifest.json"),
                                        boundary_check=args.boundary_check, threshold=.5367, torch=torch.__version__,
                                        device=device, results=results),
                                   indent=1, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "scores", "masks", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    parser.add_argument("--boundary-check", action="store_true", help="Run post hoc edge sensitivity masks")
    run(parser.parse_args())
