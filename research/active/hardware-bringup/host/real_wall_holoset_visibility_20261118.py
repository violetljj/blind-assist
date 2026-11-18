"""HoloSet: audit the observed portion of current-frame empty body boxes with native neighbouring depth views."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

DEFAULT = Path("artifacts.local/work/real-wall-search-20261118/head")
FORWARD = np.array([.35, .60, 1., 1.5, 1.95])
LATERAL = np.array([-.20, 0, .20])
HEIGHT = np.array([.65, .95, 1.25, 1.55, 1.85])
F, L, H = np.meshgrid(FORWARD, LATERAL, HEIGHT, indexing="ij")


def posefile(file, skip=0):
    lines = file.read_text().splitlines()[skip:]
    ts = np.array([int(x.split(",", 1)[0]) for x in lines], dtype=np.int64)
    data = np.loadtxt(file, delimiter=",", skiprows=skip, ndmin=2)
    return ts, data[:, 1:]


def analyze(folder):
    rows = [json.loads(x) for x in (folder / "geometry.jsonl").read_text().splitlines()]
    if not rows:
        return []
    depth_ts, dmat = posefile(folder / "Depth Long Throw_rig2world.txt")
    depth_poses = dmat.reshape(-1, 4, 4) @ np.linalg.inv(np.loadtxt(folder / "Depth Long Throw_extrinsics.txt", delimiter=",").reshape(4, 4))
    pvfile = next(folder.glob("*pv.txt"))
    pv_ts, pvdata = posefile(pvfile, 1)
    cx, cy, width, height = map(float, pvfile.read_text().splitlines()[0].split(","))
    pv_ix = {int(t): i for i, t in enumerate(pv_ts)}
    lut = np.fromfile(folder / "Depth Long Throw_lut.bin", np.float32).reshape(288, 320, 3)
    good_lut = np.where(np.isfinite(lut).all(axis=2).ravel() & (lut[:, :, 2].ravel() > 0))[0]
    tree = cKDTree(lut.reshape(-1, 3)[good_lut])
    exist = np.array([int(p.stem) for p in sorted((folder/"depth").glob("*.pgm"))])
    valid = np.isin(depth_ts, exist)
    depth_ts, depth_poses = depth_ts[valid], depth_poses[valid]
    order = np.argsort(depth_ts)
    depth_ts, depth_poses = depth_ts[order], depth_poses[order]

    @lru_cache(maxsize=12)
    def view(i):
        T = depth_poses[i]
        im = cv2.imread(str(folder/"depth"/f"{depth_ts[i]}.pgm"), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(depth_ts[i])
        raw = im[::4, ::4]
        mask = (raw > 0) & (lut[::4, ::4, 2] > 0)
        pts = (lut[::4, ::4][mask] * (raw[mask, None] * .001)) @ T[:3, :3].T + T[:3, 3]
        return im, pts

    result = []
    for row in rows:
        if row["tag"] not in ("side", "none"):
            continue
        ts, exposure = row["depth_ts"], row["rgb_ts"]
        j = pv_ix[exposure]
        fx, fy = pvdata[j, :2]
        P = pvdata[j, 2:].reshape(4, 4)
        forward = -P[:3, 2].copy()
        forward[1] = 0
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        origin = P[:3, 3].copy()
        origin[1] = row["floor_y"]
        probes = (origin + F[..., None]*forward + L[..., None]*right + H[..., None]*[0, 1, 0]).reshape(-1, 3)
        ids = np.arange(np.searchsorted(depth_ts, exposure-6000000), np.searchsorted(depth_ts, exposure+6000000, side="right"))
        current_hits = np.zeros(len(probes), bool)
        union = current_hits.copy()
        voxels_seen = Counter()
        has_current = False
        for i in ids:
            T = depth_poses[i]
            image, points = view(int(i))
            local = (probes - T[:3, 3]) @ T[:3, :3]
            distance = np.linalg.norm(local, axis=1)
            chord, ix = tree.query(local / np.maximum(distance[:, None], 1.e-9))
            # Exclude off-image nearest-edge extrapolation. Central beam spacing is ~0.004 rad.
            measured = image.ravel()[good_lut[ix]]*.001
            hits = (chord <= .0045) & (measured > distance+.08)
            union |= hits
            if depth_ts[i] == ts:
                current_hits = hits
                has_current = True
                continue
            delta = points - origin
            along, side = delta @ forward, delta @ right
            above = delta[:, 1]
            mask = (along >= .3) & (along <= 2) & (abs(side) <= .25) & (above >= .6) & (above <= 1.9)
            q = (points[mask]-P[:3, 3]) @ P[:3, :3]
            zz = np.minimum(q[:, 2], -.001)
            u, v = cx-fx*q[:, 0]/zz, cy+fy*q[:, 1]/zz
            seen = (q[:, 2] < -.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            voxel = np.floor(np.stack([along[mask][seen]-.3, side[mask][seen]+.25, above[mask][seen]-.6], 1)/.05).astype(int)
            voxels_seen.update(set(map(tuple, voxel)))
        assert has_current, (folder, ts)
        ugrid = union.reshape(F.shape)
        repeated = sum(n >= 2 for n in voxels_seen.values())
        result.append(dict(seq=row["seq"], depth_ts=ts, rgb_ts=exposure, group=row["tag"],
                           views=len(ids), current_sites=int(sum(current_hits)), union_sites=int(sum(union)),
                           near_union_sites=int(ugrid[0].sum()), min_lateral_sites=int(ugrid.sum(axis=1).min()),
                           distributed_coverage=bool(np.all(ugrid.sum(axis=1) >= 2)), repeated_body_voxels=repeated))
    (folder / "visibility.jsonl").write_text("".join(json.dumps(r)+"\n" for r in result))
    print(folder.name, len(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequences", nargs="*")
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--rgb-sheet", action="store_true")
    args = p.parse_args()
    folders = [args.root/s for s in args.sequences] if args.sequences else sorted(f.parent for f in args.root.glob("*/geometry.jsonl"))
    if args.summarize or args.rgb_sheet:
        rows = [json.loads(line) for d in folders for line in (d/"visibility.jsonl").read_text().splitlines()]
        if args.summarize:
            summary = {}
            for g in ("side", "none"):
                group = [r for r in rows if r["group"] == g]
                summary[g] = dict(frames=len(group), sequences=len({r["seq"] for r in group}),
                                  current_median=float(np.median([r["current_sites"] for r in group])),
                                  union_median=float(np.median([r["union_sites"] for r in group])),
                                  near_union_median=float(np.median([r["near_union_sites"] for r in group])),
                                  enhanced_frames=sum(r["union_sites"] > r["current_sites"] for r in group),
                                  distributed_coverage=sum(r["distributed_coverage"] for r in group),
                                  repeated_neighbour_points=sum(r["repeated_body_voxels"] >= 3 for r in group))
            (args.root/"visibility_summary.json").write_text(json.dumps(summary, indent=2))
        if args.rgb_sheet:
            draw_sample(args.root, rows)
        return
    for folder in folders:
        analyze(folder)


def draw_sample(root, rows):
    rng = np.random.default_rng(20261118)
    selected = []
    for seq in sorted({r["seq"] for r in rows}):
        controls = [r for r in rows if r["seq"] == seq and r["group"] == "none"]
        for half in np.array_split(np.arange(len(controls)), 2):
            selected.append(controls[int(rng.choice(half))])
        selected.extend(r for r in rows if r["seq"] == seq and r["repeated_body_voxels"] >= 3)
    tiles = []
    for k, r in enumerate(selected, 1):
        path = root/r["seq"]/"rgb"/f"{r['rgb_ts']}.png"
        im = cv2.imread(str(path)) if path.is_file() else None
        if im is None:
            print("Missing", k, path)
            continue
        h, w = im.shape[:2]
        im = cv2.resize(im, (400, int(round(400*h/w))), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(im, f"{k}: {r['seq']}   {r['group']}", (3, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(im)
    if tiles:
        filename = "visibility_rgb_review.jpg" if len(tiles) == len(selected) else "visibility_rgb_partial.jpg"
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(root/filename),
                    np.vstack([np.hstack(tiles[n:n+3]) for n in range(0, len(tiles), 3)]))


if __name__ == "__main__":
    main()
