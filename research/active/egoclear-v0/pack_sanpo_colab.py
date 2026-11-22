"""Phase 0 CPU Colab packer for EgoClear-v0 on the 287 SANPO-Synthetic dev sessions.

Downloads RGB + depth + segmentation + poses from GCS (official train split only;
official test_session_ids.txt is never read), verifies per-file size & MD5 against GCS,
resizes RGB to 518x294 (exact 14x multiples for ViT-S/14) JPEG q92, keeps stride-8
depth/seg for bit-for-bit walk-alert-r0 v2 evaluator parity, precomputes egocentric
corridor BEV static/dynamic GT tensors, packs into tar shards on /content, copies to
/content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_287_v0/, writes manifest_sha256.json,
and calls drive.flush_and_unmount() before exit.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
BASE = "https://storage.googleapis.com/gresearch/"
PRE = "sanpo_dataset/v0/sanpo-synthetic/"

# Exact walk-alert-r0 v2 evaluator constants
WALK = [1, 2, 3, 5, 6, 17, 30]
SKY = 27
DYN = [12, 13, 14, 21]
S = 8
CELL = 0.25
VOX = 0.05
X_HALF, FWD0, H, HGT = 0.30, 0.30, 4.0, (0.15, 1.90)
MIN_PTS = 21
MIN_VOX_V2 = 8

# RGB resize target (exact 14x multiples, 16:9)
RGB_W, RGB_H = 518, 294

# Egocentric BEV corridor grid (walker causal heading + gravity Z)
BEV_X_MIN, BEV_X_MAX, BEV_DX = -1.50, 1.50, 0.20  # 15 bins: central 3 = [-0.30, +0.30]
BEV_Z_MIN, BEV_Z_MAX, BEV_DZ = 0.30, 5.10, 0.20   # 24 bins: [0.30, 5.10]
BEV_NX = 15
BEV_NZ = 24
BEV_HBANDS = [(0.15, 0.65), (0.65, 1.35), (1.35, 1.90)]  # LOW, BODY, HEAD
SHARD_SIZE = 30


def http_get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(2 ** k, 15))
    raise RuntimeError("unreachable")


def gcs_list(prefix: str) -> list[dict]:
    out, tok = [], None
    while True:
        q = {
            "prefix": prefix,
            "maxResults": 1000,
            "fields": "items(name,size,md5Hash),nextPageToken",
        }
        if tok:
            q["pageToken"] = tok
        d = json.loads(http_get(API + "?" + urllib.parse.urlencode(q)))
        out.extend(d.get("items", []))
        tok = d.get("nextPageToken")
        if not tok:
            return out


def fetch_verified(item: dict) -> bytes:
    want_sz = int(item["size"])
    want_md5 = base64.b64decode(item["md5Hash"]).hex()
    data = http_get(BASE + urllib.parse.quote(item["name"]))
    if len(data) != want_sz or hashlib.md5(data).hexdigest() != want_md5:
        raise ValueError(f"checksum mismatch: {item['name']}")
    return data


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def corridor_d(z: np.ndarray, n_min: int) -> float:
    if len(z) < n_min:
        return float(np.inf)
    z = np.sort(z)
    cnt = np.searchsorted(z, z + 0.30, side="right") - np.arange(len(z))
    ok = np.nonzero(cnt >= n_min)[0]
    return float(z[ok[0]]) if len(ok) else float(np.inf)


def voxelize_bev(pts_xyz_h: np.ndarray, min_count: int = 2) -> np.ndarray:
    """pts_xyz_h: (N, 3) with columns (x_right, z_fwd, h_ground) -> (3, 24, 15) uint8."""
    grid = np.zeros((3, BEV_NZ, BEV_NX), dtype=np.uint16)
    if len(pts_xyz_h) == 0:
        return (grid >= min_count).astype(np.uint8)
    x, z, h = pts_xyz_h[:, 0], pts_xyz_h[:, 1], pts_xyz_h[:, 2]
    m = (x >= BEV_X_MIN) & (x < BEV_X_MAX) & (z >= BEV_Z_MIN) & (z < BEV_Z_MAX) & np.isfinite(h)
    if not np.any(m):
        return (grid >= min_count).astype(np.uint8)
    x, z, h = x[m], z[m], h[m]
    ix = np.floor((x - BEV_X_MIN) / BEV_DX).astype(np.int32)
    iz = np.floor((z - BEV_Z_MIN) / BEV_DZ).astype(np.int32)
    ix = np.clip(ix, 0, BEV_NX - 1)
    iz = np.clip(iz, 0, BEV_NZ - 1)
    for b_idx, (hlo, hhi) in enumerate(BEV_HBANDS):
        mb = (h >= hlo) & (h < hhi if b_idx < 2 else h <= hhi)
        if np.any(mb):
            np.add.at(grid[b_idx], (iz[mb], ix[mb]), 1)
    return (grid >= min_count).astype(np.uint8)


def process_session(scene: str, split: str, dl_pool: ThreadPoolExecutor) -> tuple[bytes, dict]:
    items = gcs_list(PRE + scene + "/")
    by_rel = {}
    cam_name = None
    for it in items:
        name = it["name"]
        if name.endswith("_$folder$") or int(it["size"]) == 0 or "/left/temp/" in name:
            continue
        rel = name[len(PRE + scene + "/") :]
        by_rel[rel] = it
        if cam_name is None and rel.startswith("camera_"):
            cam_name = rel.split("/")[0]
    if cam_name is None:
        raise RuntimeError(f"No camera_* dir in {scene}")

    # Identify matching frame indices across depth, seg, and rgb
    depth_pref = f"{cam_name}/left/depth_maps/"
    seg_pref = f"{cam_name}/left/segmentation_masks/"
    rgb_pref = f"{cam_name}/left/video_frames/"

    depth_keys = sorted(k for k in by_rel if k.startswith(depth_pref) and k.endswith(".float16.gz"))
    frame_ids = [int(Path(k).name[:6]) for k in depth_keys]

    needed = [by_rel["description.json"], by_rel[f"{cam_name}/camera_poses.csv"]]
    for k in frame_ids:
        needed.append(by_rel[f"{depth_pref}{k:06d}.float16.gz"])
        needed.append(by_rel[f"{seg_pref}{k:06d}.png"])
        needed.append(by_rel[f"{rgb_pref}{k:06d}.png"])

    raws = list(dl_pool.map(fetch_verified, needed))
    meta = json.loads(raws[0].decode("utf-8"))["session_camera_details"][0]
    p = meta["left_camera_params"]
    Hh, Ww = int(p["image_height"]), int(p["image_width"])
    poses_rows = list(csv.DictReader(io.StringIO(raws[1].decode("utf-8"))))

    vv, uu = np.mgrid[0:Hh:S, 0:Ww:S]
    s8_h, s8_w = vv.shape
    dirs = np.stack([(uu - p["cx"]) / p["fx"], (vv - p["cy"]) / p["fy"], np.ones(uu.shape)], -1).reshape(-1, 3)

    n = len(frame_ids)
    rgb_jpegs = []
    depth_s8 = np.empty((n, s8_h, s8_w), dtype=np.float16)
    seg_s8 = np.empty((n, s8_h, s8_w), dtype=np.uint8)
    poses_R = np.empty((n, 3, 3), dtype=np.float32)
    poses_t = np.empty((n, 3), dtype=np.float32)
    P_all, L_all = [], []

    def decode_frame(idx_k):
        idx, k = idx_k
        raw_d = raws[2 + 3 * idx]
        raw_s = raws[2 + 3 * idx + 1]
        raw_rgb = raws[2 + 3 * idx + 2]

        d_full = np.frombuffer(gzip.decompress(raw_d), "<f2")[2:].reshape(Hh, Ww)
        d_sub = np.array(d_full[::S, ::S], copy=True)

        s_full = cv2.imdecode(np.frombuffer(raw_s, np.uint8), cv2.IMREAD_UNCHANGED)
        s_sub = np.array(s_full[::S, ::S, 2], copy=True)

        img = cv2.imdecode(np.frombuffer(raw_rgb, np.uint8), cv2.IMREAD_COLOR)
        img_rs = cv2.resize(img, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
        ok_enc, jpg = cv2.imencode(".jpg", img_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            raise RuntimeError(f"JPEG encode failed {scene} frame {k}")
        return idx, k, d_sub, s_sub, jpg.tobytes()

    decoded = [decode_frame(ik) for ik in enumerate(frame_ids)]

    for idx, k, d_sub, s_sub, jpg_bytes in decoded:
        depth_s8[idx] = d_sub
        seg_s8[idx] = s_sub
        rgb_jpegs.append(jpg_bytes)

        r = poses_rows[k]
        R = quat_to_R(float(r["q_x"]), float(r["q_y"]), float(r["q_z"]), float(r["q_w"]))
        t = np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])], dtype=np.float64)
        poses_R[idx] = R.astype(np.float32)
        poses_t[idx] = t.astype(np.float32)

        d_flat = d_sub.astype(np.float32).reshape(-1)
        s_flat = s_sub.reshape(-1)
        ok = np.isfinite(d_flat) & (d_flat > 0.1) & (d_flat < 30.0)
        ray = dirs[ok] * d_flat[ok, None]
        P_all.append((R @ ray.T).T + t)
        L_all.append(s_flat[ok])

    # Pack variable-length JPEG buffers into a single contiguous uint8 array + int64 offsets
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, b in enumerate(rgb_jpegs):
        offsets[i + 1] = offsets[i] + len(b)
    rgb_blob = np.frombuffer(b"".join(rgb_jpegs), dtype=np.uint8)

    ps = poses_t.astype(np.float64)
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

    # Evaluator-only ground truth (exact walk-alert-r0 v2 logic)
    eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
    gt_single_d = np.full(n, np.nan, dtype=np.float32)
    gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    causal_fwd = np.zeros((n, 3), dtype=np.float32)
    valid_heading = np.zeros(n, dtype=np.uint8)

    if n >= 5:
        G = np.concatenate([P[np.isin(L, WALK)] for P, L in zip(P_all, L_all)])
        keys, inv = np.unique(np.floor(G[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="mergesort")
        z_sorted = G[order, 2]
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        gz = np.array([np.median(g) for g in np.split(z_sorted, cuts)], dtype=np.float64)
        gtree = cKDTree((keys + 0.5) * CELL)
        cellset = {tuple(k) for k in keys}

        def height(Q: np.ndarray) -> np.ndarray:
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.0)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            return h

        stat = np.concatenate([P[~np.isin(L, WALK + [SKY] + DYN)] for P, L in zip(P_all, L_all)])
        hs = height(stat)
        stat = stat[(hs >= HGT[0]) & (hs <= HGT[1])]
        vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
        vc = (vox + 0.5) * VOX
        vc_h = height(vc)
        for k_c in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
            cellset.add(tuple(k_c))

        for i in range(n):
            a = max(0, i - 2)
            b = i if i >= 2 else min(n - 1, 2)
            v = (ps[b] - ps[a])[:2]
            vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]

            # Causal forward for model (fallback to camera horizontal look if standing)
            if np.linalg.norm(v) >= 0.2:
                fwd = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v)
                v_ok = True
            else:
                cam_z = poses_R[i, :, 2].astype(np.float64)
                cam_z[2] = 0.0
                nz = np.linalg.norm(cam_z)
                fwd = cam_z / nz if nz > 1e-6 else np.array([1.0, 0.0, 0.0])
                v_ok = False
            right = np.array([fwd[1], -fwd[0], 0.0])
            causal_fwd[i] = fwd.astype(np.float32)

            # BEV static GT in causal walker frame
            rel_v = vc - ps[i]
            vx = rel_v @ right
            vz = rel_v @ fwd
            gt_bev_static[i] = voxelize_bev(np.column_stack([vx, vz, vc_h]), min_count=2)

            # BEV dynamic GT (current frame only) in causal walker frame
            P0, L0 = P_all[i], L_all[i]
            mdyn = np.isin(L0, DYN)
            if np.any(mdyn):
                Pd = P0[mdyn]
                hd = height(Pd)
                reld = Pd - ps[i]
                gt_bev_dynamic[i] = voxelize_bev(np.column_stack([reld @ right, reld @ fwd, hd]), min_count=2)

            if not v_ok or np.linalg.norm(vt) < 0.2:
                continue
            valid_heading[i] = 1
            tf = np.array([vt[0], vt[1], 0.0]) / np.linalg.norm(vt)
            tr_ = np.array([tf[1], -tf[0], 0.0])
            x_t, z_t = rel_v @ tr_, rel_v @ tf
            inc = (np.abs(x_t) <= X_HALF) & (z_t >= FWD0) & (z_t <= H)
            D = corridor_d(z_t[inc], MIN_VOX_V2)
            reach = min(D, H)
            samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
            cover = float(np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp])) if len(samp) else 1.0

            h0 = height(P0)
            rel0 = P0 - ps[i]
            x0, z0 = rel0 @ tr_, rel0 @ tf
            dyn_flag = float(
                ((np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H) & np.isin(L0, DYN) & (h0 >= HGT[0]) & (h0 <= HGT[1])).sum()
                >= MIN_PTS
            )
            eval_rows[i] = [D, cover, dyn_flag, seg_len[i]]

            # Exact single-frame GT-depth corridor distance (walk-alert-r0 arm 0 parity check)
            xq, zq = rel0 @ right, rel0 @ fwd
            strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
            g_est = np.percentile(P0[strip, 2], 5) if strip.sum() >= 50 else ps[i][2] - 1.5
            hq = P0[:, 2] - g_est
            c_m = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT[0]) & (hq <= HGT[1])
            gt_single_d[i] = corridor_d(zq[c_m], MIN_PTS)

    buf = io.BytesIO()
    np.savez(
        buf,
        scene=np.array(scene),
        split=np.array(split),
        cam=np.array(cam_name),
        frame_ids=np.array(frame_ids, dtype=np.int32),
        orig_hw=np.array([Hh, Ww], dtype=np.int32),
        rgb_hw=np.array([RGB_H, RGB_W], dtype=np.int32),
        K_orig=np.array([p["fx"], p["fy"], p["cx"], p["cy"]], dtype=np.float32),
        rgb_blob=rgb_blob,
        rgb_offsets=offsets,
        depth_s8=depth_s8,
        seg_s8=seg_s8,
        poses_R=poses_R,
        poses_t=poses_t,
        causal_fwd=causal_fwd,
        valid_heading=valid_heading,
        eval_rows=eval_rows,
        gt_single_d=gt_single_d,
        gt_bev_static=gt_bev_static,
        gt_bev_dynamic=gt_bev_dynamic,
    )
    npz_bytes = buf.getvalue()
    info = {
        "scene": scene,
        "split": split,
        "cam": cam_name,
        "n_frames": n,
        "npz_bytes": len(npz_bytes),
        "raw_bytes": sum(int(x["size"]) for x in needed),
    }
    return npz_bytes, info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_one_worker(args: tuple[str, str, str]) -> dict:
    cv2.setNumThreads(1)
    scene, split, scenes_dir_str = args
    dst = Path(scenes_dir_str) / f"{scene}.npz"
    with ThreadPoolExecutor(max_workers=12) as dl_pool:
        npz_bytes, info = process_session(scene, split, dl_pool)
    dst.write_bytes(npz_bytes)
    return info


def main():
    manifest_path = Path("/content/sanpo_287_manifest.json")
    drive_out = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_287_v0")
    local_root = Path("/content/sanpo_287")
    scenes_dir = local_root / "scenes"
    local_shards = local_root / "shards"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    local_shards.mkdir(parents=True, exist_ok=True)
    drive_out.mkdir(parents=True, exist_ok=True)

    m = json.loads(manifest_path.read_text())
    sessions = m["sessions"]
    splits = m["splits"]
    ncpu = os.cpu_count() or 2
    n_workers = min(20, max(2, ncpu - 2)) if ncpu >= 4 else 2
    print(f"[pack] starting {len(sessions)} sessions on {ncpu} vCPUs ({n_workers} workers) -> {drive_out}", flush=True)

    t0 = time.time()
    info_by_scene: dict[str, dict] = {}
    tasks = [(sc, splits[sc], str(scenes_dir)) for sc in sessions]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(pack_one_worker, t) for t in tasks]
        done_n = 0
        for fut in as_completed(futs):
            info = fut.result()
            info_by_scene[info["scene"]] = info
            done_n += 1
            if done_n <= 5 or done_n % 15 == 0 or done_n == len(sessions):
                el = time.time() - t0
                rate = el / done_n
                eta = rate * (len(sessions) - done_n)
                print(
                    f"  [pack] {done_n}/{len(sessions)} scenes "
                    f"({info['n_frames']}fr, avg={rate:.2f}s/sc, ETA={eta/60:.1f}m)",
                    flush=True,
                )

    shard_records = []
    session_records = []
    for s_idx in range(0, len(sessions), SHARD_SIZE):
        shard_id = s_idx // SHARD_SIZE
        shard_name = f"shard_{shard_id:03d}.tar"
        batch = sessions[s_idx : s_idx + SHARD_SIZE]
        local_shard = local_shards / shard_name
        t_sh = time.time()
        with tarfile.open(local_shard, "w") as tf:
            for sc in batch:
                tf.add(scenes_dir / f"{sc}.npz", arcname=f"{sc}.npz")
                inf = dict(info_by_scene[sc])
                inf["shard"] = shard_name
                session_records.append(inf)
        sh_sha = sha256_file(local_shard)
        sh_bytes = local_shard.stat().st_size
        drive_shard = drive_out / shard_name
        if drive_shard.exists():
            drive_shard.unlink()
        shutil.copy2(local_shard, drive_shard)
        local_shard.unlink()
        shard_records.append(
            {
                "file": shard_name,
                "bytes": sh_bytes,
                "sha256": sh_sha,
                "n_sessions": len(batch),
                "scenes": batch,
            }
        )
        print(
            f"[pack] wrote {shard_name} ({sh_bytes/1e6:.1f} MB, sha={sh_sha[:12]}) in {time.time()-t_sh:.1f}s",
            flush=True,
        )

    manifest_out = {
        "dataset": "sanpo_synth_287_v0",
        "source": BASE + PRE,
        "license": "CC BY 4.0 (SANPO); research use only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rgb_hw": [RGB_H, RGB_W],
        "bev_spec": {
            "x_range": [BEV_X_MIN, BEV_X_MAX, BEV_DX, BEV_NX],
            "z_range": [BEV_Z_MIN, BEV_Z_MAX, BEV_DZ, BEV_NZ],
            "h_bands": BEV_HBANDS,
        },
        "n_sessions": len(session_records),
        "n_frames": sum(r["n_frames"] for r in session_records),
        "total_shard_bytes": sum(r["bytes"] for r in shard_records),
        "total_raw_downloaded_bytes": sum(r["raw_bytes"] for r in session_records),
        "bw_speeds": m["bw_speeds"],
        "splits": splits,
        "shards": shard_records,
        "sessions": session_records,
    }
    man_file = drive_out / "manifest_sha256.json"
    if man_file.exists():
        man_file.unlink()
    man_file.write_text(json.dumps(manifest_out, indent=2))
    (local_root / ".extracted_ok").write_text("ok")
    print(
        f"[pack] DONE: {manifest_out['n_sessions']} sessions, {manifest_out['n_frames']} frames, "
        f"{manifest_out['total_shard_bytes']/1e9:.2f} GB packed from "
        f"{manifest_out['total_raw_downloaded_bytes']/1e9:.2f} GB raw in {time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
