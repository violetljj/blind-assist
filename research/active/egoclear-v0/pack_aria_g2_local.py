"""Pack Aria Gen 2 Pilot (walk_0, walk_1) locally at 5 Hz into 20 evaluation clips for zero-shot glasses benchmark.
CC BY-NC 4.0: private evaluation cache only, never redistributed; only aggregate metrics are reported.
"""
from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DATA = Path("artifacts.local/downloads/aria-gen2-pilot-20261117")
OUT_DIR = Path("artifacts.local/work/egoclear-v0-phase4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEQS = ("walk_0", "walk_1")
STRIDE = 6       # 30 fps -> 5 Hz
CLIP_FR = 150    # 30 seconds per clip -> 10 clips per walk = 20 clips total
RGB_W, RGB_H = 518, 294
S_GRID = 160     # 160x160 depth subsample (~25.6k pts vs 512x512, matching SANPO S=8 density ~43k -> scale MIN_PTS)
X_HALF, FWD0, H, HGT = 0.30, 0.30, 4.0, (0.15, 1.90)
CELL, VOX = 0.25, 0.05
MIN_PTS_ARIA = 14
MIN_VOX_V2 = 8


def corridor_d(z: np.ndarray, n_min: int) -> float:
    if len(z) < n_min:
        return float(np.inf)
    z = np.sort(z)
    cnt = np.searchsorted(z, z + 0.30, side="right") - np.arange(len(z))
    ok = np.nonzero(cnt >= n_min)[0]
    return float(z[ok[0]]) if len(ok) else float(np.inf)


def fit_ground_plane_cam(pts_w: np.ndarray, cam_z_pos: float, rng: np.random.Generator) -> float:
    """Estimate local ground Z in world (gravity-aligned Z up) from points below camera."""
    below = pts_w[pts_w[:, 2] < cam_z_pos - 0.5]
    if len(below) < 200:
        return cam_z_pos - 1.53
    # Ground is a dense horizontal sheet near the bottom: take 10th percentile of Z within 0.5..4m
    return float(np.percentile(below[:, 2], 10))


def main():
    t0 = time.time()
    tar_path = OUT_DIR / "aria_g2_20clips.tar"
    clips_meta = []
    vv, uu = np.mgrid[0:S_GRID, 0:S_GRID]

    with tarfile.open(tar_path, "w") as tarf:
        for seq in SEQS:
            base = DATA / seq / "x/depth"
            params = json.loads((base / "pinhole_camera_parameters.json").read_text())
            sub = params[::STRIDE]
            n_clips = len(sub) // CLIP_FR
            for ci in range(n_clips):
                chunk = sub[ci * CLIP_FR : (ci + 1) * CLIP_FR]
                n = len(chunk)
                rgb_jpegs = []
                poses_R = np.empty((n, 3, 3), dtype=np.float32)
                poses_t = np.empty((n, 3), dtype=np.float32)
                K_per_f = np.empty((n, 4), dtype=np.float32)
                P_all = []
                for idx, p in enumerate(chunk):
                    fi = p["index"]
                    fu, fv, u0, v0 = p["camera"]["Parameters"]
                    K_per_f[idx] = [fu, fv, u0, v0]
                    twc = p["T_world_camera"]
                    R = Rotation.from_quat(twc["QuaternionXYZW"]).as_matrix()
                    t = np.array(twc["Translation"], dtype=np.float64)
                    poses_R[idx] = R.astype(np.float32)
                    poses_t[idx] = t.astype(np.float32)

                    gray = cv2.imread(str(base / "rectified_images" / f"image_{fi:08d}.png"), cv2.IMREAD_UNCHANGED)
                    bgr = np.repeat(gray[:, :, None], 3, 2)
                    # Resize 512x512 -> 518x294 (with exact K_s scaling handled via orig_hw=[512,512]!)
                    img_rs = cv2.resize(bgr, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
                    _, jpg = cv2.imencode(".jpg", img_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    rgb_jpegs.append(jpg.tobytes())

                    d_mm = cv2.imread(str(base / "depth" / f"depth_{fi:08d}.png"), cv2.IMREAD_UNCHANGED)
                    d_rs = cv2.resize(d_mm, (S_GRID, S_GRID), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 1000.0
                    s_x = S_GRID / 512.0
                    d_flat = d_rs.reshape(-1)
                    ok = (d_flat > 0.3) & (d_flat < 12.0)
                    u_k = uu.reshape(-1)[ok]
                    v_k = vv.reshape(-1)[ok]
                    dk = d_flat[ok]
                    pc = np.stack([(u_k - u0 * s_x) / (fu * s_x) * dk, (v_k - v0 * s_x) / (fv * s_x) * dk, dk], -1)
                    P_all.append(((R @ pc.T).T + t).astype(np.float32))

                offsets = np.zeros(n + 1, dtype=np.int64)
                for i, b in enumerate(rgb_jpegs):
                    offsets[i + 1] = offsets[i] + len(b)
                rgb_blob = np.frombuffer(b"".join(rgb_jpegs), dtype=np.uint8)

                ps = poses_t.astype(np.float64)
                seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

                # Local-window (+/-4 frames = +/-0.8s) static FoundationStereo corridor GT (clean, drift-free!)
                eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
                gt_single_d = np.full(n, np.inf, dtype=np.float32)
                causal_fwd = np.zeros((n, 3), dtype=np.float32)
                valid_heading = np.zeros(n, dtype=np.uint8)
                rng = np.random.default_rng(ci)

                for i in range(n):
                    a = max(0, i - 2)
                    b = i if i >= 2 else min(n - 1, 2)
                    v = (ps[b] - ps[a])[:2]
                    vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
                    if np.linalg.norm(v) >= 0.15:
                        fwd = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v)
                        v_ok = True
                    else:
                        cz = poses_R[i, :, 2].astype(np.float64)
                        cz[2] = 0.0
                        nz = np.linalg.norm(cz)
                        fwd = cz / nz if nz > 1e-6 else np.array([1.0, 0.0, 0.0])
                        v_ok = False
                    causal_fwd[i] = fwd.astype(np.float32)
                    if not v_ok or np.linalg.norm(vt) < 0.15:
                        continue
                    valid_heading[i] = 1
                    tf = np.array([vt[0], vt[1], 0.0]) / np.linalg.norm(vt)
                    tr_ = np.array([tf[1], -tf[0], 0.0])
                    right = np.array([fwd[1], -fwd[0], 0.0])

                    # Local window ground & obstacle voxels
                    lo, hi = max(0, i - 4), min(n, i + 5)
                    Pw = np.concatenate(P_all[lo:hi])
                    rel_w = Pw - ps[i]
                    xw, zw = rel_w @ tr_, rel_w @ tf
                    # Estimate local ground surface z_g per 0.25m cell via 5th percentile of z in lower band
                    below = Pw[Pw[:, 2] < ps[i, 2] - 0.8]
                    if len(below) < 200:
                        continue
                    keys, inv = np.unique(np.floor(below[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
                    inv = inv.ravel()
                    order = np.argsort(inv, kind="mergesort")
                    zs_b = below[order, 2]
                    cuts = np.flatnonzero(np.diff(inv[order])) + 1
                    gz = np.array([np.percentile(g, 10) for g in np.split(zs_b, cuts)], dtype=np.float64)
                    gtree = cKDTree((keys + 0.5) * CELL)
                    cellset = {tuple(k) for k in keys}

                    dist_g, idx_g = gtree.query(Pw[:, :2], distance_upper_bound=0.8)
                    okg = np.isfinite(dist_g)
                    hw = np.full(len(Pw), np.nan)
                    hw[okg] = Pw[okg, 2] - gz[idx_g[okg]]
                    obs = Pw[(hw >= HGT[0]) & (hw <= HGT[1])]
                    if len(obs) > 0:
                        vox = np.unique(np.floor(obs / VOX).astype(np.int64), axis=0)
                        vc = (vox + 0.5) * VOX
                        rv = vc - ps[i]
                        xt, zt = rv @ tr_, rv @ tf
                        inc = (np.abs(xt) <= X_HALF) & (zt >= FWD0) & (zt <= H)
                        D = corridor_d(zt[inc], MIN_VOX_V2)
                    else:
                        D = float(np.inf)
                    reach = min(D, H)
                    samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
                    cover = float(np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp])) if len(samp) else 1.0
                    eval_rows[i] = [D, cover, 0.0, seg_len[i]]

                    # Single-frame reference FoundationStereo distance
                    P0 = P_all[i]
                    r0 = P0 - ps[i]
                    x0, z0 = r0 @ right, r0 @ fwd
                    strip = (np.abs(x0) <= 1.0) & (z0 >= 1.0) & (z0 <= H)
                    g0 = np.percentile(P0[strip, 2], 5) if strip.sum() >= 50 else ps[i, 2] - 1.53
                    h0 = P0[:, 2] - g0
                    cm = (np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H) & (h0 >= HGT[0]) & (h0 <= HGT[1])
                    gt_single_d[i] = corridor_d(z0[cm], MIN_PTS_ARIA)

                cname = f"aria_{seq}_c{ci:02d}"
                npz_p = OUT_DIR / f"{cname}.npz"
                np.savez(
                    npz_p,
                    scene=np.array(cname),
                    split=np.array("aria_zeroshot"),
                    cam=np.array("camera_head"),
                    orig_hw=np.array([512, 512], dtype=np.int32),
                    rgb_hw=np.array([RGB_H, RGB_W], dtype=np.int32),
                    K_orig=np.median(K_per_f, axis=0).astype(np.float32),
                    rgb_blob=rgb_blob,
                    rgb_offsets=offsets,
                    poses_R=poses_R,
                    poses_t=poses_t,
                    causal_fwd=causal_fwd,
                    valid_heading=valid_heading,
                    eval_rows=eval_rows,
                    gt_single_d=gt_single_d,
                    gt_bev_static=np.zeros((n, 3, 24, 15), dtype=np.uint8),
                    gt_bev_dynamic=np.zeros((n, 3, 24, 15), dtype=np.uint8),
                )
                tarf.add(npz_p, arcname=f"{cname}.npz")
                npz_p.unlink()
                nhaz = int(np.sum(np.isfinite(eval_rows[:, 0]) & (eval_rows[:, 0] <= H)))
                clips_meta.append({"clip": cname, "frames": n, "haz_frames": nhaz})
                print(f"  packed {cname}: {n} fr, haz<=4m={nhaz}", flush=True)

    print(f"[aria] DONE {len(clips_meta)} clips -> {tar_path} ({tar_path.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
