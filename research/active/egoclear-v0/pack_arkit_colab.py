"""ARKitScenes Real Indoor Corridor/Room 48-vCPU Packer for EgoClear Phase 5.

Downloads official Apple CDN lowres_wide / lowres_depth / lowres_wide_intrinsics / lowres_wide.traj,
strictly upright-rectifies portrait/landscape captures (exact RGB + Depth + K + R_c2w rotation),
subsamples to ~5 Hz (0.2s step, up to 120 frames/capture), builds drift-free +/-0.8s local-window
LiDAR floor + static obstacle BEV/corridor GT matching SANPO/Aria contracts, and writes .tar shards
+ manifest_sha256.json to Google Drive.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from pack_sanpo_colab import CELL, FWD0, HGT, MIN_PTS, MIN_VOX_V2, VOX, X_HALF, corridor_d, voxelize_bev

H_EVAL = 4.0
ARKIT_BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"


def fetch_bytes(url: str, timeout: int = 45, retries: int = 3) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-Research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def detect_upright_rot_code(R_c2w_seq: np.ndarray) -> int:
    """Determine discrete 90-deg rotation so image top points toward world +Z (up).

    Returns:
      0: no rotation (v_up <= -0.5)
      1: cv2.ROTATE_90_CLOCKWISE (u_up < -0.5)
      2: cv2.ROTATE_90_COUNTERCLOCKWISE (u_up > 0.5)
      3: cv2.ROTATE_180 (v_up > 0.5)
    """
    # World +Z expressed in camera frame is row 2 of R_c2w: [u_up, v_up, z_up]
    u_up = float(np.median(R_c2w_seq[:, 2, 0]))
    v_up = float(np.median(R_c2w_seq[:, 2, 1]))
    if abs(v_up) >= abs(u_up):
        return 0 if v_up <= 0.0 else 3
    return 1 if u_up < 0.0 else 2


def apply_upright_rotation(
    img: np.ndarray,
    depth: np.ndarray,
    K_vec: np.ndarray,  # [fx, fy, cx, cy]
    hw: tuple[int, int],  # (H, W)
    R_c2w: np.ndarray,
    rot_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    if rot_code == 0:
        return img, depth, K_vec, hw, R_c2w
    Hh, Ww = hw
    fx, fy, cx, cy = [float(x) for x in K_vec]
    if rot_code == 1:  # ROTATE_90_CLOCKWISE: u'=H-1-v, v'=u
        img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        dep_r = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        K_r = np.array([fy, fx, (Hh - 1.0) - cy, cx], dtype=np.float64)
        R_old_from_new = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return img_r, dep_r, K_r, (Ww, Hh), R_c2w @ R_old_from_new
    if rot_code == 2:  # ROTATE_90_COUNTERCLOCKWISE: u'=v, v'=W-1-u
        img_r = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dep_r = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        K_r = np.array([fy, fx, cy, (Ww - 1.0) - cx], dtype=np.float64)
        R_old_from_new = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return img_r, dep_r, K_r, (Ww, Hh), R_c2w @ R_old_from_new
    # rot_code == 3: ROTATE_180
    img_r = cv2.rotate(img, cv2.ROTATE_180)
    dep_r = cv2.rotate(depth, cv2.ROTATE_180)
    K_r = np.array([fx, fy, (Ww - 1.0) - cx, (Hh - 1.0) - cy], dtype=np.float64)
    R_old_from_new = np.diag([-1.0, -1.0, 1.0])
    return img_r, dep_r, K_r, (Hh, Ww), R_c2w @ R_old_from_new


def pack_arkit_worker(args: tuple[str, str, str, int]) -> dict | None:
    """Download & pack one ARKitScenes indoor capture into EgoClear .npz format."""
    video_id, fold, out_dir_str, min_valid_frames = args
    cv2.setNumThreads(1)
    out_dir = Path(out_dir_str)
    sc_name = f"arkit_{video_id}"
    out_npz = out_dir / f"{sc_name}.npz"
    if out_npz.exists():
        d = np.load(out_npz, allow_pickle=False)
        vh_cnt = int((d["valid_heading"] > 0).sum())
        return {"scene": sc_name, "frames": len(d["eval_rows"]), "valid_frames": vh_cnt, "cached": True}

    base = f"{ARKIT_BASE}/{fold}/{video_id}"
    traj_raw = fetch_bytes(f"{base}/lowres_wide.traj").decode("utf-8").strip().splitlines()
    if len(traj_raw) < 25:
        return None

    t_traj, R_traj, p_traj = [], [], []
    for line in traj_raw:
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = [float(x) for x in parts[:7]]
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = Rotation.from_rotvec(vals[1:4]).as_matrix()
        ext[:3, 3] = vals[4:7]
        c2w = np.linalg.inv(ext)
        t_traj.append(vals[0])
        R_traj.append(c2w[:3, :3])
        p_traj.append(c2w[:3, 3])
    t_traj = np.array(t_traj, dtype=np.float64)
    R_traj = np.array(R_traj, dtype=np.float64)
    p_traj = np.array(p_traj, dtype=np.float64)

    # Quick pre-filter on trajectory path length before downloading zips (saves bandwidth!)
    path_xy = float(np.linalg.norm(np.diff(p_traj[:, :2], axis=0), axis=1).sum())
    if path_xy < 4.0:
        return None

    # Download RGB, LiDAR depth, and intrinsics zips
    rgb_zip = zipfile.ZipFile(io.BytesIO(fetch_bytes(f"{base}/lowres_wide.zip", timeout=60)))
    dep_zip = zipfile.ZipFile(io.BytesIO(fetch_bytes(f"{base}/lowres_depth.zip", timeout=60)))
    k_zip = zipfile.ZipFile(io.BytesIO(fetch_bytes(f"{base}/lowres_wide_intrinsics.zip", timeout=45)))

    rgb_idx = {Path(n).stem: n for n in rgb_zip.namelist() if n.endswith(".png")}
    dep_idx = {Path(n).stem: n for n in dep_zip.namelist() if n.endswith(".png")}
    k_idx = {Path(n).stem: n for n in k_zip.namelist() if n.endswith(".pincam")}
    common = sorted(set(rgb_idx) & set(dep_idx) & set(k_idx), key=lambda s: float(s.rsplit("_", 1)[1]))
    if len(common) < 20:
        return None

    # Subsample to ~5 Hz (dt >= 0.19s), cap at 120 frames
    sel_stems = []
    sel_ti = []
    last_t = -1e9
    for st in common:
        ts = float(st.rsplit("_", 1)[1])
        if ts - last_t < 0.19:
            continue
        ti = int(np.argmin(np.abs(t_traj - ts)))
        if abs(t_traj[ti] - ts) > 0.05:
            continue
        sel_stems.append(st)
        sel_ti.append(ti)
        last_t = ts
        if len(sel_stems) >= 120:
            break

    n = len(sel_stems)
    if n < 20:
        return None

    Rs_raw = R_traj[sel_ti].copy()
    ps = p_traj[sel_ti].copy()
    rot_code = detect_upright_rot_code(Rs_raw)

    # Decode frames, upright-rectify RGB/Depth/K/R_c2w, and encode 518x294 RGB JPEGs
    Rs = np.empty((n, 3, 3), dtype=np.float64)
    Ks_all = np.empty((n, 4), dtype=np.float64)
    depths_all = []
    jpegs = []
    orig_hw = None

    for i, st in enumerate(sel_stems):
        rgb_arr = cv2.imdecode(np.frombuffer(rgb_zip.read(rgb_idx[st]), np.uint8), cv2.IMREAD_COLOR)
        dep_u16 = cv2.imdecode(np.frombuffer(dep_zip.read(dep_idx[st]), np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb_arr is None or dep_u16 is None:
            return None
        w_k, h_k, fx, fy, cx, cy = [float(x) for x in k_zip.read(k_idx[st]).decode("utf-8").split()[:6]]
        Hh, Ww = rgb_arr.shape[:2]
        K_v = np.array([fx * Ww / w_k, fy * Hh / h_k, cx * Ww / w_k, cy * Hh / h_k], dtype=np.float64)
        rgb_u, dep_u, K_u, hw_u, R_u = apply_upright_rotation(rgb_arr, dep_u16, K_v, (Hh, Ww), Rs_raw[i], rot_code)
        if orig_hw is None:
            orig_hw = hw_u
        elif hw_u != orig_hw:
            rgb_u = cv2.resize(rgb_u, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
            dep_u = cv2.resize(dep_u, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)

        Rs[i] = R_u
        Ks_all[i] = K_u
        depths_all.append(dep_u.astype(np.float32) * 0.001)  # mm -> meters
        rgb_rs = cv2.resize(rgb_u, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
        ok_enc, enc = cv2.imencode(".jpg", rgb_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            return None
        jpegs.append(enc.tobytes())

    K_orig = np.median(Ks_all, axis=0)
    dh, dw = depths_all[0].shape
    # Unproject LiDAR points per frame (native 192x256 or 256x192 -> fast!)
    vv, uu = np.mgrid[0:dh, 0:dw]
    P_all = []
    for i in range(n):
        fx, fy, cx, cy = Ks_all[i]
        df = depths_all[i]
        ok = np.isfinite(df) & (df > 0.18) & (df < 8.5)
        if ok.sum() < 50:
            P_all.append(np.zeros((0, 3), dtype=np.float32))
            continue
        z = df[ok]
        x = (uu[ok] - cx) * z / fx
        y = (vv[ok] - cy) * z / fy
        pc = np.stack([x, y, z], axis=-1)
        pw = (Rs[i] @ pc.T).T + ps[i]
        P_all.append(pw.astype(np.float32))

    # Causal walking heading (matching SANPO/Aria) + forward-facing filter (excludes panning sideways/backwards)
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
    causal_fwd = np.zeros((n, 3), dtype=np.float64)
    valid_heading = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        cam_f = Rs[i, :2, 2]
        cam_fn = float(np.linalg.norm(cam_f))
        cam_f2 = (cam_f / cam_fn) if cam_fn > 1e-4 else np.array([1.0, 0.0])

        # Look back up to 5 frames (1.0 s) to find causal displacement >= 0.18 m
        v_xy = None
        for back in range(2, min(6, i + 1)):
            dv = (ps[i] - ps[i - back])[:2]
            if np.linalg.norm(dv) >= 0.18:
                v_xy = dv
                break
        if v_xy is not None:
            f2 = v_xy / np.linalg.norm(v_xy)
            cos_align = float(np.dot(f2, cam_f2))
            causal_fwd[i] = [f2[0], f2[1], 0.0]
            # Require forward-ish walking (cos > 0.25, i.e., within ~75 deg of optical axis)
            if cos_align >= 0.25:
                valid_heading[i] = 1
        else:
            causal_fwd[i] = [cam_f2[0], cam_f2[1], 0.0]

    n_valid_h = int(valid_heading.sum())
    if n_valid_h < min_valid_frames:
        return None

    # Build drift-free +/-0.8s local-window floor + static obstacle BEV/corridor GT
    eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
    gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_single_d = np.full(n, np.inf, dtype=np.float32)

    # Scale MIN_PTS to native depth grid (192x256 = 49k pixels vs SANPO s8 156x276 = 43k pixels -> nearly identical!)
    min_pts_indoor = max(8, int(round(MIN_PTS * (dh * dw) / (156.0 * 276.0))))

    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        non_empty = [P_all[k] for k in range(lo, hi) if len(P_all[k]) > 0]
        if not non_empty:
            continue
        Pw_win = np.concatenate(non_empty, axis=0)
        # Robust indoor floor estimate: 5th percentile of points below camera
        below = Pw_win[Pw_win[:, 2] < ps[i, 2] - 0.35]
        if len(below) < 150:
            continue
        z_floor = float(np.percentile(below[:, 2], 5))
        floor_mask = np.abs(Pw_win[:, 2] - z_floor) <= 0.12
        Gw = Pw_win[floor_mask]
        if len(Gw) < 80:
            continue

        keys, inv = np.unique(np.floor(Gw[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="mergesort")
        z_s = Gw[order, 2]
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        gz = np.array([np.median(g) for g in np.split(z_s, cuts)], dtype=np.float64)
        gtree = cKDTree((keys + 0.5) * CELL)
        cellset = {tuple(k) for k in keys}

        def height_loc(Q: np.ndarray) -> np.ndarray:
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.2)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            # Fallback to global local floor if slightly outside observed floor patch
            miss = ~okg
            if np.any(miss):
                h[miss] = Q[miss, 2] - z_floor
            return h

        h_win = height_loc(Pw_win)
        stat = Pw_win[(h_win >= HGT[0]) & (h_win <= HGT[1])]
        fwd = causal_fwd[i]
        right = np.array([fwd[1], -fwd[0], 0.0])

        if len(stat) > 0:
            vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
            vc = (vox + 0.5) * VOX
            vc_h = height_loc(vc)
            rel_v = vc - ps[i]
            gt_bev_static[i] = voxelize_bev(np.column_stack([rel_v @ right, rel_v @ fwd, vc_h]), min_count=2)
            for kc in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
                cellset.add(tuple(kc))
        else:
            vc = np.zeros((0, 3), dtype=np.float64)

        # Single-frame LiDAR reference distance
        P0 = P_all[i]
        if len(P0) > 0:
            rel0 = P0 - ps[i]
            x0, z0 = rel0 @ right, rel0 @ fwd
            h0 = P0[:, 2] - z_floor
            cm0 = (np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H_EVAL) & (h0 >= HGT[0]) & (h0 <= HGT[1])
            gt_single_d[i] = corridor_d(z0[cm0], min_pts_indoor)

        if not valid_heading[i]:
            continue

        # Centered truth heading for closed-loop evaluation (matches SANPO/Aria v2 exact protocol!)
        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
        if np.linalg.norm(vt) < 0.15:
            continue
        tf = np.array([vt[0], vt[1], 0.0]) / np.linalg.norm(vt)
        tr_ = np.array([tf[1], -tf[0], 0.0])
        if len(vc) > 0:
            rel_v = vc - ps[i]
            x_t, z_t = rel_v @ tr_, rel_v @ tf
            inc = (np.abs(x_t) <= X_HALF) & (z_t >= FWD0) & (z_t <= H_EVAL)
            D = corridor_d(z_t[inc], MIN_VOX_V2)
        else:
            D = float(np.inf)

        reach = min(D, H_EVAL)
        samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
        cover = float(np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp])) if len(samp) else 1.0
        eval_rows[i] = [D, cover, 0.0, seg_len[i]]

    n_eval_ok = int(np.isfinite(eval_rows[:, 1]).sum())
    if n_eval_ok < min_valid_frames:
        return None

    # Pack variable-length JPEGs cleanly
    jpeg_buf = b"".join(jpegs)
    jpeg_off = np.zeros(n + 1, dtype=np.int64)
    cur = 0
    for idx_j, jb in enumerate(jpegs):
        cur += len(jb)
        jpeg_off[idx_j + 1] = cur

    np.savez(
        out_npz,
        rgb_blob=np.frombuffer(jpeg_buf, dtype=np.uint8),
        rgb_offsets=jpeg_off,
        K_orig=K_orig.astype(np.float32),
        orig_hw=np.array(orig_hw, dtype=np.int32),
        poses_R=Rs.astype(np.float32),
        poses_t=ps.astype(np.float32),
        causal_fwd=causal_fwd.astype(np.float32),
        valid_heading=valid_heading,
        eval_rows=eval_rows,
        gt_single_d=gt_single_d,
        gt_bev_static=gt_bev_static,
        gt_bev_dynamic=gt_bev_dynamic,
    )
    haz_fr = int((np.isfinite(eval_rows[:, 0]) & (eval_rows[:, 0] <= H_EVAL)).sum())
    return {
        "scene": sc_name,
        "video_id": video_id,
        "fold": fold,
        "rot_code": rot_code,
        "frames": n,
        "valid_frames": n_eval_ok,
        "haz_frames": haz_fr,
        "path_m": round(float(seg_len.sum()), 2),
        "z_up_check": round(float(np.median(Rs[:, 2, 1])), 3),
    }


def write_arkit_shards(local_dir: Path, stems: list[str], drive_dir: Path, prefix: str, meta: dict) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 30
    shards_meta = []
    for si in range(0, len(stems), shard_size):
        chunk = stems[si : si + shard_size]
        tname = f"{prefix}_shard{si // shard_size:02d}.tar"
        tpath = drive_dir / tname
        if tpath.exists():
            tpath.unlink()
        with tarfile.open(tpath, "w") as tf:
            for st in chunk:
                p = local_dir / f"{st}.npz"
                tf.add(p, arcname=p.name)
        h = hashlib.sha256()
        with open(tpath, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        shards_meta.append({"file": tname, "bytes": tpath.stat().st_size, "sha256": h.hexdigest(), "scenes": chunk})
    man = {"prefix": prefix, "meta": meta, "shards": shards_meta}
    mpath = drive_dir / "manifest_sha256.json"
    if mpath.exists():
        mpath.unlink()
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")


def pack_arkit_split(
    pool: list[dict],
    target_n: int,
    local_dir: Path,
    drive_dir: Path,
    prefix: str,
    min_valid_frames: int = 20,
    max_workers: int = 20,
) -> tuple[list[str], list[dict]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    # Restore from Drive cache if already packed
    mpath = drive_dir / "manifest_sha256.json"
    if len(list(local_dir.glob("*.npz"))) < target_n and mpath.exists():
        print(f"[arkit-pack] restoring {prefix} from Drive cache {drive_dir}...", flush=True)
        man = json.loads(mpath.read_text())
        for sh in man["shards"]:
            with tarfile.open(drive_dir / sh["file"], "r") as tf:
                tf.extractall(local_dir)

    existing = sorted(p.stem for p in local_dir.glob("*.npz"))
    if len(existing) >= target_n:
        print(f"[arkit-pack] {prefix} ready locally ({len(existing)} scenes)", flush=True)
        return existing[:target_n], []

    t0 = time.time()
    print(f"[arkit-pack] packing {prefix}: target={target_n} from pool={len(pool)} (workers={max_workers})...", flush=True)
    infos = []
    # Process in deterministic batches so we stop promptly once target_n valid walking indoor captures are ready
    idx = 0
    while len(infos) < target_n and idx < len(pool):
        need = target_n - len(infos)
        batch_sz = min(len(pool) - idx, max(need + 12, int(need * 1.4)))
        batch = pool[idx : idx + batch_sz]
        idx += batch_sz
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(pack_arkit_worker, (r["video_id"], r["fold"], str(local_dir), min_valid_frames)): r
                for r in batch
            }
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    inf = fut.result()
                    if inf is not None:
                        infos.append(inf)
                except Exception as e:
                    print(f"  [warn] arkit {r['video_id']} failed: {e}", flush=True)

    # Sort deterministically by scene name and trim to exact target_n
    infos.sort(key=lambda x: x["scene"])
    if len(infos) > target_n:
        for drop in infos[target_n:]:
            p = local_dir / f"{drop['scene']}.npz"
            if p.exists():
                p.unlink()
        infos = infos[:target_n]

    stems = [x["scene"] for x in infos]
    write_arkit_shards(local_dir, stems, drive_dir, prefix, {"n_ok": len(stems), "infos": infos[:5]})
    z_chk = [x["z_up_check"] for x in infos if "z_up_check" in x]
    hz_tot = sum(x.get("haz_frames", 0) for x in infos)
    vf_tot = sum(x.get("valid_frames", 0) for x in infos)
    print(
        f"[arkit-pack] {prefix} DONE: {len(stems)} scenes in {time.time()-t0:.1f}s | "
        f"valid_fr={vf_tot}, haz_fr={hz_tot} ({100*hz_tot/max(vf_tot,1):.1f}%) | "
        f"median z_up_check={np.median(z_chk) if z_chk else 'cached'}",
        flush=True,
    )
    return stems, infos
