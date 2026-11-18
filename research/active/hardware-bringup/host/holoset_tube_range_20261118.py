"""Check whether actual range returns inside the tube's image follow it across viewpoints."""
import json
from pathlib import Path

import cv2
import numpy as np

from real_wall_holoset_score_20261118 import Recording, records


WORK = Path("artifacts.local/work")
HEAD = WORK / "real-wall-search-20261118/head"
STUDY = WORK / "holoset-tube-20261118"


def project(cap, rt, world):
    fx, fy, pose = cap.pv[rt]
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    z = -local[:, 2]
    u = np.rint(cap.cx + fx * local[:, 0] / np.maximum(z, .05)).astype(int)
    v = np.rint(cap.cy - fy * local[:, 1] / np.maximum(z, .05)).astype(int)
    valid = (z > .05) & (u >= 0) & (u < cap.width) & (v >= 0) & (v < cap.height)
    return u.clip(0, cap.width - 1), v.clip(0, cap.height - 1), z, valid


def read_mask(kind, seq, ts):
    path = STUDY / "masks" / kind / seq / f"{ts}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (428, 760):
        raise ValueError(f"Missing or invalid mask: {path}")
    return image != 0


def examine(row, cap, context, floor, score):
    dt, rt, seq = row["depth_ts"], row["rgb_ts"], row["seq"]
    depth = cv2.imread(str(cap.directory / "depth" / f"{dt}.pgm"), cv2.IMREAD_UNCHANGED)
    ok = cap.valid & (depth > 0)
    sensor = cap.rig[dt] @ cap.extr
    world = (cap.native_rays[ok] * (depth[ok, None] * .001)) @ sensor[:3, :3].T + sensor[:3, 3]
    u, v, z, inside = project(cap, rt, world)
    core = read_mask("core", seq, rt)
    _, _, _, maps, ref = cap.registration(row)
    pred_dir = WORK / "real-wall-search-20261118/holoset-d8/remote/out" / seq / "predicted"
    pred = np.load(pred_dir / f"{dt}.npy") * score["scale"]
    grid_core = cv2.remap(core.astype(np.uint8), *maps, cv2.INTER_NEAREST) != 0
    paired = grid_core & (ref > 0) & (ref < 8) & (pred > 0) & (pred < 8)
    grid_depth = dict(pairs=int(paired.sum()), ref_median_m=float(np.median(ref[paired])),
                      dav2_median_m=float(np.median(pred[paired])),
                      paired_dav2_minus_ref_median_m=float(np.median((pred-ref)[paired])))
    use = inside & core[v, u]
    world, z = world[use], z[use]
    pose = cap.pv[rt][2]
    forward = -pose[:3, 2].copy()
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    relative = world - pose[:3, 3]
    longitudinal = relative @ forward
    lateral = relative @ np.cross(forward, [0, 1, 0])
    height = world[:, 1] - floor
    body = (longitudinal >= .30) & (longitudinal <= 2.) & (abs(lateral) <= .25) & (height >= .60) & (height <= 1.90)
    n_visible = np.zeros(len(z), int)
    n_outside = np.zeros(len(z), int)
    for other in context:
        r = other["rgb_ts"]
        baseline = np.linalg.norm(cap.pv[r][2][:3, 3] - pose[:3, 3])
        if baseline < .35:
            continue
        outer = read_mask("outer", seq, r)
        if not outer.any():
            continue
        # Allow additional annotation/pose slack when ruling out an image location.
        outer = cv2.dilate(outer.astype(np.uint8), np.ones((11, 11), np.uint8))
        u, v, _, inside = project(cap, r, world)
        n_visible += inside
        n_outside += inside & (outer[v, u] == 0)
    return dict(case=row["case"], seq=seq, depth_ts=dt, rgb_ts=rt, rgb_depth_offset_ms=(rt-dt)/1e4,
                core_pixels=int(core.sum()), native_returns=len(z), body_returns=int(body.sum()), model_grid=grid_depth,
                pv_z_p10_p50_p90_m=np.percentile(z, [10, 50, 90]).tolist() if len(z) else [],
                visible_in_two_other_views=int((n_visible >= 2).sum()),
                outside_expanded_tube_in_two=int((n_outside >= 2).sum()))


def main():
    cases = json.loads((HEAD / "alert-review/cases.json").read_bytes())
    histories = json.loads((STUDY / "masks/histories.json").read_bytes())
    results = []
    for seq, context in histories.items():
        cap = Recording(HEAD / seq)
        scored = WORK / "real-wall-search-20261118/holoset-d8/remote/out" / seq / "scores.json"
        lookup = {r["depth_ts"]: r for r in json.loads(scored.read_bytes())["scores"]}
        floor = {r["depth_ts"]: r.get("floor_y") for r in records(cap.directory / "geometry.jsonl")}
        for row in cases:
            if row["seq"] == seq:
                results.append(examine(row, cap, context, floor[row["depth_ts"]], lookup[row["depth_ts"]]))
    if len(results) != 7:
        raise ValueError("Expected seven alarm frames")
    path = STUDY / "sensor/range.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print([(r["case"], r["native_returns"], r["body_returns"], r["outside_expanded_tube_in_two"]) for r in results])


if __name__ == "__main__":
    main()
