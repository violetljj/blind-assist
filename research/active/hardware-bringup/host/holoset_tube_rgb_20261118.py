"""Paired RGB sensitivity tests on the two real HoloSet duct passages.

Run from the checkout root: python holoset_tube_rgb_20261118.py prepare|infer|score.
The local PLAN.md defines the edits. Source depth, poses, weights and D8 are held fixed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

import real_wall_holoset_score_20261118 as replay


BASE = Path("artifacts.local/work")
ORIGINAL = BASE / "real-wall-search-20261118"
HEAD = ORIGINAL / "head"
SCORES = ORIGINAL / "holoset-d8/remote/out"
MASKS = BASE / "holoset-tube-20261118/masks"
OUT = MASKS.parent / "rgb"
ARMS = ("original", "tube_telea", "control_telea", "tube_column", "control_column",
        "tube_transfer", "control_transfer")


def read(path, flag=cv2.IMREAD_COLOR):
    im = cv2.imread(str(path), flag)
    if im is None:
        raise FileNotFoundError(path)
    return im


def column_fill(im, mask):
    out = im.copy()
    for x in np.flatnonzero(mask.any(axis=0)):
        rows = np.flatnonzero(mask[:, x])
        visible = np.flatnonzero(~mask[:, x])
        if not len(visible):
            raise ValueError("Column fill has no background in this column")
        for c in range(3):
            out[rows, x, c] = np.rint(np.interp(rows, visible, im[visible, x, c])).astype(np.uint8)
    return out


def transfer(frames, i, region):
    target = frames[i]
    mask = target[region]
    v, u = np.nonzero(mask)
    base = target["im"].copy()
    if not len(u):
        return base, 0
    xyz = target["xyz"][v, u]
    usable = np.isfinite(xyz).all(1)
    first = np.zeros((len(u), 3), np.int16)
    source = np.full(len(u), -1)
    done = np.zeros(len(u), bool)
    donors = sorted(range(len(frames)), key=lambda j: abs(frames[j]["rt"] - target["rt"]))
    for j in donors:
        donor = frames[j]
        origin = donor["pose"][:3, 3]
        if np.linalg.norm(target["pose"][:3, 3] - origin) < .1:
            continue
        q = (xyz - origin) @ donor["pose"][:3, :3]
        z = q[:, 2]
        fx, fy, cx, cy = donor["K"]
        m = usable & ~done & (z > .05)
        ids = np.flatnonzero(m)
        uv = q[ids, :2] / z[ids, None] * [fx, fy] + [cx, cy]
        uu, vv = np.rint(uv).astype(int).T
        h, w = mask.shape
        inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        ids, uu, vv = ids[inside], uu[inside], vv[inside]
        ref = donor["ref"][vv, uu]
        consistent = (ref > 0) & (abs(ref - z[ids]) <= np.maximum(.07, .02 * z[ids]))
        consistent &= ~donor["tube_guard"][vv, uu]
        if region == "control":
            consistent &= ~donor["control_guard"][vv, uu]
        ids, uu, vv = ids[consistent], uu[consistent], vv[consistent]
        color = donor["im"][vv, uu].astype(np.int16)
        prev = source[ids]
        for k in np.unique(prev[prev >= 0]):
            if np.linalg.norm(frames[k]["pose"][:3, 3] - origin) < .1:
                continue
            sel = (prev == k) & (np.max(abs(color - first[ids]), 1) <= 25)
            match = ids[sel]
            base[v[match], u[match]] = (first[match] + color[sel] + 1) // 2
            done[match] = True
        new = prev < 0
        first[ids[new]] = color[new]
        source[ids[new]] = j
    return base, int(done.sum())


def histories():
    return json.loads((MASKS / "histories.json").read_bytes())


def prepare():
    counts = []
    for seq, rows in histories().items():
        cap = replay.Recording(HEAD / seq)
        frames = []
        for r in rows:
            rt = r["rgb_ts"]
            pose, _, _, _, ref_grid = cap.registration(r)
            im = read(HEAD / seq / "rgb" / f"{rt}.png")
            fx, fy, _ = cap.pv[rt]
            v, u = np.indices(im.shape[:2], dtype=np.float32)
            uu = u * replay.FOCAL / (fx * replay.STEP)
            vv = v * replay.FOCAL / (fy * replay.STEP)
            ref = cv2.remap(ref_grid, uu, vv, cv2.INTER_NEAREST)
            dirs = np.stack(((u-cap.cx)/fx, (v-cap.cy)/fy, np.ones_like(u)), -1)
            xyz = ((dirs * ref[:, :, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
            xyz[ref <= 0] = np.nan
            masks = {}
            for name, path in (("tube", "outer"), ("control", "control")):
                mask = read(MASKS / path / seq / f"{rt}.png", cv2.IMREAD_GRAYSCALE)
                masks[name] = mask > 0
                masks[name+"_guard"] = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
            if masks["tube"].sum() != masks["control"].sum() or (masks["tube"] & masks["control"]).any():
                raise ValueError(f"Control does not match {seq} {rt}")
            frames.append(dict(im=im, xyz=xyz, pose=pose, ref=ref, K=(fx, fy, cap.cx, cap.cy),
                               rt=rt, **masks))
        for i, frame in enumerate(frames):
            rt, im = frame["rt"], frame["im"]
            row = dict(seq=seq, rgb_ts=rt, area=int(frame["tube"].sum()), image_pixels=im.shape[0]*im.shape[1])
            for region in ("tube", "control"):
                mask = frame[region]
                transferred, n = transfer(frames, i, region)
                row[region+"_transfer_pixels"] = n
                for method, edited in (("telea", cv2.inpaint(im, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)),
                                       ("column", column_fill(im, mask)), ("transfer", transferred)):
                    if not np.array_equal(edited[~mask], im[~mask]):
                        raise ValueError("Non-local image edit")
                    folder = OUT / "img" / seq / f"{region}_{method}"
                    folder.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(folder / f"{rt}.png"), edited):
                        raise IOError(folder)
            counts.append(row)
        print("Prepared", seq, len(rows), flush=True)
        del frames
    (OUT / "masks_and_transfer.json").write_text(json.dumps(counts, indent=1))


def infer(arms=ARMS):
    import psutil
    import torch
    import torch.nn.functional as fn
    import spires_insert_dev_20261117 as source
    gib = 1024**3
    if psutil.virtual_memory().available < 3*gib:
        raise RuntimeError("Insufficient host RAM; retain completed predictions")
    torch.cuda.set_per_process_memory_fraction(.70)
    torch.set_num_threads(2)
    model = source._dav2s()
    for seq, rows in histories().items():
        cap = replay.Recording(HEAD / seq)
        for r in rows:
            _, _, _, maps, _ = cap.registration(r)
            for arm in arms:
                target = OUT/"predicted"/seq/arm/f"{r['depth_ts']}.npy"
                if target.exists():
                    continue
                if psutil.virtual_memory().available < 2*gib:
                    raise RuntimeError("Free host RAM below 2 GiB; resume from cache")
                image_root = HEAD/seq/"rgb" if arm == "original" else OUT/"img"/seq/arm
                im = read(image_root/f"{r['rgb_ts']}.png")
                t = model.image2tensor(im, 518)[0]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    d = model(t)
                dense = fn.interpolate(d[:, None].float(), im.shape[:2], mode="bilinear",
                                       align_corners=True)[0, 0].cpu().numpy()
                mono = cv2.remap(dense, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                target.parent.mkdir(exist_ok=True, parents=True)
                tmp = target.with_suffix(".part")
                with tmp.open("wb") as f:
                    np.save(f, mono)
                tmp.replace(target)
        print("Predicted", seq, len(rows), "GPU GiB", torch.cuda.max_memory_reserved()/gib, flush=True)


def score():
    import torch
    from spires_insert_dev_20261117 import A, I, V
    torch.set_num_threads(2)
    net = I.D.make_head(8)
    net.load_state_dict(torch.load(BASE/"ase-cane-reach-pilot-20261015/decision_head8_all.pt",
                                   map_location="cpu", weights_only=True))
    net.eval()
    cases = json.loads((HEAD/"alert-review/cases.json").read_bytes())
    results = []
    for seq, inputs in histories().items():
        cap = replay.Recording(HEAD / seq)
        rows = {r["depth_ts"]: r for r in replay.records(cap.directory/"geometry.jsonl")}
        scores = {r["depth_ts"]:r for r in json.loads((SCORES/seq/"scores.json").read_bytes())["scores"]}
        targets = {c["depth_ts"]: c for c in cases if c["seq"] == seq}
        per_target = defaultdict(dict)
        for arm in ARMS:
            hist = []
            for inp in inputs:
                ts = inp["depth_ts"]
                r = rows[ts]
                pose, center, dirs, _, ref = cap.registration(r)
                mono = np.load(OUT/"predicted"/seq/arm/f"{ts}.npy")
                valid = (ref > 0) & (ref < A.MAX_MM*.001) & np.isfinite(mono) & (mono > 0)
                if valid.sum() <= 100:
                    raise ValueError("Insufficient reference depth")
                scale = float(np.median(ref[valid]/mono[valid]))
                zz = np.where(ref > 0, mono*scale, 0.)
                ok = (zz > 0) & (zz < A.MAX_MM*.001)
                points = ((dirs[ok]*zz[ok, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
                hist = [e for e in hist if ts-e["ts"] <= replay.WINDOW]
                hist.append(dict(ts=ts, pose=pose, center=center, depth=zz, points=points))
                if ts not in targets:
                    continue
                if len(hist) != scores[ts]["history"]:
                    raise ValueError("Incomplete source history")
                grid = replay.features(hist, r["floor_y"], V)
                with torch.inference_mode():
                    value = torch.sigmoid(net(V.pad(torch.from_numpy(grid[None]).float()))).item()
                per_target[ts][arm] = value
        for ts, record in per_target.items():
            source_score = scores[ts]["score"]
            if abs(record["original"] - source_score) > .02 or (record["original"] > I.THETA) != scores[ts]["alert"]:
                raise ValueError(f"Unstable same-device baseline for {seq} {ts}: {record['original']} vs {source_score}")
            results.append(dict(case=targets[ts]["case"], seq=seq, depth_ts=ts,
                                published=source_score, scores=record))
    (OUT/"scores.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "score"])
    parser.add_argument("--original-only", action="store_true", help="Cache the unmodified RGB predictions first")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "infer":
        infer(("original",) if args.original_only else ARMS)
    else:
        {"prepare": prepare, "score": score}[args.stage]()
