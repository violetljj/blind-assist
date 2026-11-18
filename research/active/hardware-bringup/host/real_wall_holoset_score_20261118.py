"""Stream HoloSet through the frozen ASE D8 head. Use the pre-score cohort of distinct depth frames.

check DATA SEQUENCE verifies the camera transforms and reference coverage without a model.
score DATA COHORT OUTPUT [--sequence NAME] processes the whole cohort or one sequence.
COHORT is a JSONL of {seq, depth_ts, stratum} with stratum low/mid/high/none.
All time/coverage/texture decisions are evaluator-only. Reference depth is used to align
the metric monocular depth as in the source experiment. Keep predicted grids locally for replay.
"""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import os

import cv2
import numpy as np


FOCAL = 425.0
STEP = 2
WINDOW = 30_000_000  # 100 ns HoloLens timestamp units
ZUP = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
CV_AXES = np.diag([1., -1., -1.])


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def file_digest(path):
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transforms(path):
    return {int(fields[0]): np.asarray(fields[1:], float).reshape(4, 4)
            for fields in (line.split(",") for line in path.read_text().splitlines() if line.strip())}

def rigid(matrix):
    return (np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=3e-5)
            and abs(np.linalg.det(matrix[:3, :3]) - 1.) < 5e-5)


class Recording:
    def __init__(self, directory):
        self.directory = directory
        lines = next(directory.glob("*pv.txt")).read_text().splitlines()
        self.cx, self.cy, width, height = map(float, lines[0].split(","))
        self.width, self.height = int(width), int(height)
        self.pv = {int(row[0]): (float(row[1]), float(row[2]), np.array(row[3:], float).reshape(4, 4))
                   for row in (line.split(",") for line in lines[1:] if line.strip())}
        self.rig = transforms(directory / "Depth Long Throw_rig2world.txt")
        self.extr = np.linalg.inv(np.loadtxt(directory / "Depth Long Throw_extrinsics.txt", delimiter=",").reshape(4, 4))
        lut = np.fromfile(directory / "Depth Long Throw_lut.bin", np.float32).reshape(288, 320, 3)
        self.valid = np.isfinite(lut).all(axis=-1) & (np.abs(np.linalg.norm(lut, axis=-1) - 1.) < .02)
        self.native_rays = lut
        # Subdivide each valid detector pixel into four directions. Preserve its own
        # measured range at each subpixel; never interpolate ranges between objects.
        rays = cv2.resize(np.where(self.valid[:, :, None], lut, 0.), None, fx=2, fy=2)
        norm = np.linalg.norm(rays, axis=2, keepdims=True)
        self.rays = rays / np.maximum(norm, 1e-6)

    def registration(self, row, native=False):
        dt, rt = row["depth_ts"], row["rgb_ts"]
        if abs(dt - rt) > 750_000:
            raise ValueError("PV and depth outside predeclared tolerance")
        fx, fy, pv = self.pv[rt]
        pose = np.eye(4)
        pose[:3, :3] = ZUP @ pv[:3, :3] @ CV_AXES
        pose[:3, 3] = ZUP @ pv[:3, 3]
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=3e-5):
            raise ValueError("Invalid PV pose")
        # Full PV angular field with the original ASE rays per steradian.
        sx, sy = FOCAL / fx, FOCAL / fy
        width, height = round(self.width * sx), round(self.height * sy)
        cx, cy = self.cx * sx, self.cy * sy
        grid_v, grid_u = np.mgrid[0:height:STEP, 0:width:STEP].astype(np.float32)
        directions = np.stack([(grid_u - cx) / FOCAL, (grid_v - cy) / FOCAL, np.ones_like(grid_u)], axis=-1)
        img = cv2.imread(str(self.directory / "depth" / f"{dt}.pgm"), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16 or img.shape != self.valid.shape:
            raise ValueError(f"Invalid depth frame {dt}")
        depth_to_pv = np.linalg.inv(pv) @ self.rig[dt] @ self.extr
        ranges = np.where(self.valid, img, 0)
        rays = self.native_rays if native else self.rays
        if not native:
            ranges = np.repeat(np.repeat(ranges, 2, axis=1), 2, axis=0)
        ok = ranges > 0
        pts = (rays[ok] * (ranges[ok, None] * .001)) @ depth_to_pv[:3, :3].T + depth_to_pv[:3, 3]
        pts = pts[pts[:, 2] < -.05]
        axial = -pts[:, 2]
        u = np.rint((cx + FOCAL * pts[:, 0] / axial) / STEP).astype(int)
        v = np.rint((cy - FOCAL * pts[:, 1] / axial) / STEP).astype(int)
        h, w = grid_u.shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ref = np.full(h * w, np.inf, np.float32)
        np.minimum.at(ref, v[inside] * w + u[inside], axial[inside])
        ref[~np.isfinite(ref)] = 0.
        return pose, (cx, cy), directions, (grid_u / sx, grid_v / sy), ref.reshape(h, w)


def features(history, floor, V):
    cur = history[-1]
    pose = cur["pose"]
    origin = pose[:3, 3]
    up = np.asarray([0., 0., 1.])
    forward = pose[:3, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    h, f = (np.arange(V.NH) + .5) * V.CELL, (np.arange(V.NF) + .5) * V.CELL
    L, F, H = np.meshgrid([-.2, 0., .2], f, h, indexing="ij")
    world = (origin * (1 - up) + L[..., None] * right + F[..., None] * forward + (floor + H)[..., None] * up).reshape(-1, 3)
    hits = np.zeros((2, V.NH, V.NF))
    seen, free = np.zeros_like(hits), np.zeros_like(hits)
    for e in history:
        arm = int(e is not cur)
        points = e["points"]
        d = points - origin
        x, z, y = d @ right, d @ forward, points[:, 2] - floor
        m = (abs(x) <= V.X_HALF) & (z >= 0) & (z < V.NF * V.CELL) & (y >= 0) & (y < V.NH * V.CELL)
        ids = (y[m] // V.CELL).astype(int) * V.NF + (z[m] // V.CELL).astype(int)
        hits[arm] += np.bincount(ids, minlength=V.NH * V.NF).reshape(V.NH, V.NF)
        q = (world - e["pose"][:3, 3]) @ e["pose"][:3, :3]
        axial = q[:, 2]
        proj = FOCAL * q[:, :2] / np.where(axial > .05, axial, 1.)[:, None]
        u, v = ((proj + e["center"]) / STEP).T
        depth = e["depth"]
        ins = (axial > .05) & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        zm = np.zeros(len(q))
        zm[ins] = depth[v[ins].astype(int), u[ins].astype(int)]
        vis = ins & (zm > 0)
        seen[arm] += vis.reshape(3, V.NF, V.NH).sum(0).T
        free[arm] += (vis & (axial < zm - .1)).reshape(3, V.NF, V.NH).sum(0).T
    gf, gh = np.meshgrid(f / (V.NF * V.CELL), h / (V.NH * V.CELL))
    return np.stack([np.log1p(hits[0]), np.log1p(hits[1]), free[0] / 3, np.log1p(free[1]),
                     seen[0] / 3, np.log1p(seen[1]), gf, gh]).astype(np.float16)


def score(data, cohort, output, sequence=None):
    import psutil
    gib = 1024**3
    if psutil.virtual_memory().available < 3 * gib:
        raise RuntimeError("Wait for at least 3 GiB of available host RAM before loading the models")
    import torch
    import torch.nn.functional as fn
    import spires_insert_dev_20261117 as source
    from spires_insert_dev_20261117 import A, I, V
    targets = records(cohort)
    groups = {r["stratum"] for r in targets}
    if not groups <= {"low", "mid", "high", "none"}:
        raise ValueError(f"Unknown strata {groups}")
    for group in ("low", "none"):
        rows = [r for r in targets if r["stratum"] == group]
        if len(rows) < 100 or len({r["seq"] for r in rows}) < 4:
            raise ValueError("Cohort failed the predeclared coverage gate")
    target_ids = {(r["seq"], r["depth_ts"]) for r in targets}
    if len(target_ids) != len(targets):
        raise ValueError("Duplicate scoring targets")
    torch.cuda.set_per_process_memory_fraction(.70)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = source._dav2s()
    net = I.D.make_head(8).cuda()
    weights = Path(__file__).resolve().parents[4] / "artifacts.local/work/ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.eval()
    digest = sha256(cohort.read_bytes()).hexdigest()
    backbone = weights.parents[2] / "models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/depth_anything_v2_metric_hypersim_vits.pth"
    metadata = dict(cohort_sha256=digest, scorer_sha256=file_digest(Path(__file__)),
                    d8_sha256=file_digest(weights), dav2_sha256=file_digest(backbone),
                    torch_version=str(torch.__version__), device=torch.cuda.get_device_name(),
                    focal=FOCAL, step=STEP, seconds=WINDOW/1e7, threshold=I.THETA)
    output.mkdir(parents=True, exist_ok=True)
    run_record = output / "run.json"
    if run_record.exists() and json.loads(run_record.read_text()) != metadata:
        raise ValueError("Run metadata differs; choose a separate output directory")
    if not run_record.exists():
        run_record.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    for name in sorted({r["seq"] for r in targets}):
        if sequence and sequence != name:
            continue
        cap = Recording(data / name)
        geom = records(cap.directory / "geometry.jsonl")
        ids = {ts for n, ts in target_ids if n == name}
        invalid_poses = {r["depth_ts"] for r in geom if r["tag"] not in ("no_depth_pose", "no_rgb_match")
                         and not (rigid(cap.rig[r["depth_ts"]]) and rigid(cap.pv[r["rgb_ts"]][2]))}
        if ids & invalid_poses:
            raise ValueError("A scoring target has an invalid pose")
        out = output / name
        cache = out / "predicted"
        cache.mkdir(exist_ok=True, parents=True)
        final = out / "scores.json"
        if final.exists():
            receipt = json.loads(final.read_text())
            if receipt["cohort_sha256"] != digest or {r["depth_ts"] for r in receipt["scores"]} != ids:
                raise ValueError(f"Result belongs to a different cohort: {final}")
            continue
        checkpoint = out / "partial.json"
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text())
            if prior["cohort_sha256"] != digest:
                raise ValueError("Partial scores belong to a different cohort")
            rows = prior["scores"]
        else:
            rows = []
        written = {r["depth_ts"] for r in rows}
        needed = sorted(ids - written)
        hist = deque()
        def save_partial():
            tmp = checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(cohort_sha256=digest, scores=rows)), encoding="utf8")
            os.replace(tmp, checkpoint)
        for frame_index, r in enumerate(sorted(geom, key=lambda row: row["depth_ts"])):
            ts, rt = r["depth_ts"], r["rgb_ts"]
            i = np.searchsorted(needed, ts)
            if i == len(needed) or needed[i] - ts > WINDOW:
                continue
            if r["tag"] in ("no_depth_pose", "no_rgb_match") or ts in invalid_poses:
                continue
            if frame_index % 16 == 0:
                available = psutil.virtual_memory().available / gib
                if available < 2:
                    save_partial()
                    raise RuntimeError(f"Host available RAM {available:.2f} GiB; resume from cache when free")
                if frame_index % 128 == 0:
                    print(name, frame_index, len(rows), "available/RSS/reserved GiB",
                          round(available, 2), round(psutil.Process().memory_info().rss / gib, 2),
                          round(torch.cuda.max_memory_reserved() / gib, 2), flush=True)
            pv_path = cap.directory / "rgb" / f"{rt}.png"
            if not pv_path.is_file():
                raise FileNotFoundError(pv_path)
            pose, center, dirs, maps, ref = cap.registration(r)
            pred_path = cache / f"{ts}.npy"
            if pred_path.exists():
                mono = np.load(pred_path)
            else:
                im = cv2.imread(str(pv_path))
                if im is None or im.shape[:2] != (cap.height, cap.width):
                    raise ValueError(pv_path)
                t = model.image2tensor(im, 518)[0]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    d = model(t)
                dense = fn.interpolate(d[:, None].float(), im.shape[:2], mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
                mono = cv2.remap(dense, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                with pred_path.with_suffix(".tmp").open("wb") as f:
                    np.save(f, mono)
                os.replace(pred_path.with_suffix(".tmp"), pred_path)
            valid = (ref > 0) & (ref < A.MAX_MM * .001) & np.isfinite(mono) & (mono > 0)
            if valid.sum() <= 100:
                raise ValueError(f"Insufficient depth for metric scaling: {name} {ts}")
            scale = float(np.median(ref[valid] / mono[valid]))
            zz = np.where(ref > 0, mono * scale, 0.)
            ok = (zz > 0) & (zz < A.MAX_MM * .001)
            points = ((dirs[ok] * zz[ok, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
            hist.append(dict(ts=ts, rgb_ts=rt, pose=pose, center=center, depth=zz, points=points))
            while hist[0]["ts"] < ts - WINDOW:
                hist.popleft()
            if ts not in ids or ts in written:
                continue
            x = features(hist, r["floor_y"], V)
            with torch.inference_mode():
                s = torch.sigmoid(net(V.pad(torch.from_numpy(x[None]).float().cuda()))).item()
            if not np.isfinite(s):
                raise ValueError("non-finite D8 score")
            gaps = np.diff([ts - WINDOW, *(e["ts"] for e in hist), ts])
            rows.append(dict(depth_ts=ts, rgb_ts=rt, score=s, alert=s>I.THETA, scale=scale,
                             valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()), history=len(hist),
                             history_rgb_count=len({e["rgb_ts"] for e in hist}),
                             max_window_gap_s=float(gaps.max())/1e7,
                             history_span_s=(ts-hist[0]["ts"])/1e7))
            written.add(ts)
            if len(rows) % 48 == 0:
                save_partial()
        if written != ids:
            raise ValueError(f"{name}: {len(ids-written)} targets missing")
        tmp = out / "scores.tmp"
        tmp.write_text(json.dumps(dict(cohort_sha256=digest, sequence=name, invalid_history_pose_ids=sorted(invalid_poses),
                                       scores=sorted(rows, key=lambda r:r["depth_ts"])),
                                   allow_nan=False), encoding="utf8")
        os.replace(tmp, checkpoint)
        os.replace(checkpoint, final)
        print(name, len(rows), "scored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("data", type=Path)
    check.add_argument("sequence")
    run = sub.add_parser("score")
    for arg in ("data", "cohort", "output"):
        run.add_argument(arg, type=Path)
    run.add_argument("--sequence")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.command == "score":
        score(args.data, args.cohort, args.output, args.sequence)
    else:
        cap = Recording(args.data / args.sequence)
        rows = records(cap.directory / "geometry.jsonl")
        sample = next(r for r in rows if r["tag"] in {"side", "none"} and (cap.directory / "rgb" / f"{r['rgb_ts']}.png").exists())
        pose, center, directions, maps, ref = cap.registration(sample)
        native = cap.registration(sample, native=True)[-1]
        print(json.dumps(dict(seq=args.sequence, depth_ts=sample["depth_ts"], rgb_dt=sample["rgb_dt"],
                              grid=list(ref.shape), f425_native=float(np.mean(native>0)),
                              f425_footprint=float(np.mean(ref>0)), max_rgb_sync_s=.075)))
