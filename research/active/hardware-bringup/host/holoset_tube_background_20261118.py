"""Measure how much of the tube can be replaced by verified background seen after passing."""
import json

import cv2
import numpy as np

import holoset_tube_rgb_20261118 as rgb


# Native range exposures paired with the visually reviewed post-passage RGB donors.
DONORS = {
    "research-center-seq3": [
        133070577021200219, 133070577025200304, 133070577031200429,
        133070577033200470, 133070577037200555, 133070577041200638,
        133070577049200806, 133070577055200931, 133070577059201015,
    ],
    "research-center-seq4": [
        133070611289817290, 133070611295817413, 133070611299817496,
        133070611303817578, 133070611305817621, 133070611311817746,
        133070611313817786, 133070611321817953, 133070611323817994,
        133070611329818120, 133070611331818162, 133070611333818203,
    ],
}


def exposure(cap, row):
    pose, _, _, _, ref_grid = cap.registration(row)
    rt = row["rgb_ts"]
    im = rgb.read(cap.directory / "rgb" / f"{rt}.png")
    fx, fy, _ = cap.pv[rt]
    v, u = np.indices(im.shape[:2], dtype=np.float32)
    ref = cv2.remap(ref_grid, u * rgb.replay.FOCAL / (fx * rgb.replay.STEP),
                    v * rgb.replay.FOCAL / (fy * rgb.replay.STEP), cv2.INTER_NEAREST)
    frame = dict(pose=pose, ref=ref, K=(fx, fy, cap.cx, cap.cy), rt=rt, im=im)
    return frame, u, v


def main():
    results = {}
    cases = json.loads((rgb.HEAD / "alert-review/cases.json").read_bytes())
    for seq, rows in rgb.histories().items():
        cap = rgb.replay.Recording(rgb.HEAD / seq)
        timeline = {r["depth_ts"]: r for r in rgb.replay.records(cap.directory / "geometry.jsonl")}
        first = min(r["rgb_ts"] for r in cases if r["seq"] == seq)
        donors, identities = [], []
        for dt in DONORS[seq]:
            frame, u, v = exposure(cap, timeline[dt])
            # A piece of duct is visible in the upper left in part of the first pass.
            frame["tube_guard"] = (u < .40 * cap.width) & (v < .45 * cap.height) if seq.endswith("3") else np.zeros_like(u, bool)
            donors.append(frame)
            identities.append(dict(rgb_ts=frame["rt"], depth_ts=dt, offset_seconds=(frame["rt"]-first)/1e7))
        counts = []
        for row in rows:
            frame, u, v = exposure(cap, row)
            fx, fy, cx, cy = frame["K"]
            dirs = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones_like(u)), axis=-1)
            ref, pose = frame["ref"], frame["pose"]
            xyz = ((dirs * ref[:, :, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
            xyz[ref <= 0] = np.nan
            mask = rgb.read(rgb.MASKS / "outer" / seq / f"{frame['rt']}.png", cv2.IMREAD_GRAYSCALE) > 0
            _, count = rgb.transfer([dict(**frame, xyz=xyz, tube=mask), *donors], 0, "tube")
            counts.append(dict(rgb_ts=frame["rt"], pixels=int(mask.sum()),
                               target_reference_pixels=int((mask & (ref > 0)).sum()), two_donor_pixels=count))
        area = sum(r["pixels"] for r in counts)
        accepted = sum(r["two_donor_pixels"] for r in counts)
        results[seq] = dict(donors=identities, frames=counts, area=area,
                            transferred=accepted, fraction=accepted/area)
    rgb.OUT.mkdir(parents=True, exist_ok=True)
    (rgb.OUT / "clean_donor_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
