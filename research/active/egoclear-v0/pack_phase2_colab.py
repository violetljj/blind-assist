"""Phase 2 high-speed 48-vCPU packer on G4:
1) Extra 450 disjoint SANPO-Synthetic train sessions -> Drive sanpo_synth_extra450_v0/ + local /content/sanpo_p2/extra_scenes/
2) 40 real-world SANPO-Real zero-shot eval sessions (5 Hz, READY ZED VIO, session-level gravity alignment)
   -> Drive sanpo_real_zeroshot_40_v0/ + local /content/sanpo_p2/real_scenes/
3) Copies existing 287 SANPO-Synthetic shards from Drive -> local /content/sanpo_287/scenes/ in parallel.
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

sys.path.insert(0, "/content")
from pack_sanpo_colab import (
    API,
    BASE,
    BEV_DX,
    BEV_DZ,
    BEV_HBANDS,
    BEV_NX,
    BEV_NZ,
    BEV_X_MAX,
    BEV_X_MIN,
    BEV_Z_MAX,
    BEV_Z_MIN,
    CELL,
    DYN,
    FWD0,
    H,
    HGT,
    MIN_PTS,
    MIN_VOX_V2,
    RGB_H,
    RGB_W,
    S,
    SKY,
    VOX,
    WALK,
    X_HALF,
    corridor_d,
    fetch_verified,
    gcs_list,
    process_session,
    quat_to_R,
    sha256_file,
    voxelize_bev,
)

PRE_REAL = "sanpo_dataset/v0/sanpo-real/"
SHARD_SIZE = 30
REAL_STRIDE = 3  # 15 fps -> 5 Hz


def rotation_align_z(normal: np.ndarray) -> np.ndarray:
    """Compute rigid 3x3 rotation R_align mapping unit `normal` to [0, 0, 1]."""
    n = normal / max(np.linalg.norm(normal), 1e-9)
    if n[2] < 0:
        n = -n
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = float(np.dot(n, z))
    if s < 1e-8:
        return np.eye(3, dtype=np.float64) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def fit_plane_normal_ransac(pts: np.ndarray, iters: int = 80, thr: float = 0.08) -> np.ndarray:
    if len(pts) < 100:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(42)
    sub = pts[rng.choice(len(pts), min(len(pts), 8000), replace=False)]
    best_n, best_in = np.array([0.0, 0.0, 1.0]), -1
    for _ in range(iters):
        idx = rng.choice(len(sub), 3, replace=False)
        p0, p1, p2 = sub[idx]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-6:
            continue
        n = n / ln
        d = np.abs((sub - p0) @ n)
        nin = int((d <= thr).sum())
        if nin > best_in:
            best_in, best_n = nin, n
    if best_n[2] < 0:
        best_n = -best_n
    return best_n


def pack_extra_synth_worker(args: tuple[str, str]) -> dict:
    cv2.setNumThreads(1)
    scene, dst_dir = args
    dst = Path(dst_dir) / f"{scene}.npz"
    if dst.exists() and dst.stat().st_size > 1000:
        return {"scene": scene, "skipped": True}
    with ThreadPoolExecutor(max_workers=12) as dl_pool:
        npz_bytes, info = process_session(scene, "train", dl_pool)
    dst.write_bytes(npz_bytes)
    return info


def pack_real_worker(args: tuple[str, str, str]) -> dict | None:
    cv2.setNumThreads(1)
    scene, cam_name, dst_dir = args
    dst = Path(dst_dir) / f"{scene}_{cam_name}.npz"
    items = gcs_list(PRE_REAL + scene + "/")
    by_rel = {}
    for it in items:
        name = it["name"]
        if name.endswith("_$folder$") or int(it["size"]) == 0 or "/left/temp/" in name:
            continue
        rel = name[len(PRE_REAL + scene + "/") :]
        by_rel[rel] = it

    pose_rel = f"{cam_name}/fixed_camera_poses.csv"
    if pose_rel not in by_rel:
        pose_rel = f"{cam_name}/camera_poses.csv"
    if "description.json" not in by_rel or pose_rel not in by_rel:
        return None

    depth_pref = f"{cam_name}/left/depth_maps/"
    seg_pref = f"{cam_name}/left/segmentation_masks/"
    rgb_pref = f"{cam_name}/left/video_frames/"

    d_ids = {int(Path(k).name[:6]) for k in by_rel if k.startswith(depth_pref) and k.endswith(".float16.gz")}
    s_ids = {int(Path(k).name[:6]) for k in by_rel if k.startswith(seg_pref) and k.endswith(".png")}
    r_ids = {int(Path(k).name[:6]) for k in by_rel if k.startswith(rgb_pref) and k.endswith(".png")}
    common_ids = sorted(d_ids & s_ids & r_ids)
    if len(common_ids) < 20:
        return None

    with ThreadPoolExecutor(max_workers=12) as dl_pool:
        meta_raw, pose_raw = list(dl_pool.map(fetch_verified, [by_rel["description.json"], by_rel[pose_rel]]))
        meta_all = json.loads(meta_raw.decode("utf-8"))["session_camera_details"]
        # Pick matching camera params: ZED_M for camera_head, ZED_2I for camera_chest
        want_key = "ZED_M" if cam_name == "camera_head" else "ZED_2I"
        cam_meta = next((c for c in meta_all if want_key in str(c.get("model", ""))), meta_all[0])
        p = cam_meta["left_camera_params"]
        Hh, Ww = int(p["image_height"]), int(p["image_width"])
        poses_rows = list(csv.DictReader(io.StringIO(pose_raw.decode("utf-8"))))

        # Filter to READY poses and subsample to ~5 Hz (stride 3), cap at 120 frames per session for fast uniform eval
        ready_ids = [
            k for k in common_ids
            if k < len(poses_rows) and "READY" in str(poses_rows[k].get("tracking_state", "READY"))
            and "NOT_READY" not in str(poses_rows[k].get("tracking_state", "READY"))
        ]
        frame_ids = ready_ids[::REAL_STRIDE][:120]
        if len(frame_ids) < 15:
            return None

        needed = []
        for k in frame_ids:
            needed.append(by_rel[f"{depth_pref}{k:06d}.float16.gz"])
            needed.append(by_rel[f"{seg_pref}{k:06d}.png"])
            needed.append(by_rel[f"{rgb_pref}{k:06d}.png"])
        raws = list(dl_pool.map(fetch_verified, needed))

    # Target stride-8 equivalent grid (156x276) matching SANPO-Synthetic density for exact MIN_PTS parity
    s8_h, s8_w = 156, 276
    fx_s8 = p["fx"] * (s8_w / float(Ww))
    fy_s8 = p["fy"] * (s8_h / float(Hh))
    cx_s8 = p["cx"] * (s8_w / float(Ww))
    cy_s8 = p["cy"] * (s8_h / float(Hh))
    vv, uu = np.mgrid[0:s8_h, 0:s8_w]
    dirs = np.stack([(uu - cx_s8) / fx_s8, (vv - cy_s8) / fy_s8, np.ones(uu.shape)], -1).reshape(-1, 3)

    n = len(frame_ids)
    rgb_jpegs = []
    depth_s8 = np.empty((n, s8_h, s8_w), dtype=np.float16)
    seg_s8 = np.empty((n, s8_h, s8_w), dtype=np.uint8)
    poses_R_raw = np.empty((n, 3, 3), dtype=np.float64)
    poses_t_raw = np.empty((n, 3), dtype=np.float64)
    P_raw, L_all = [], []

    for idx, k in enumerate(frame_ids):
        d_raw = np.frombuffer(gzip.decompress(raws[3 * idx]), "<f2")
        dh_f, dw_f = int(d_raw[0]), int(d_raw[1])
        d_body = d_raw[2:]
        if dh_f * dw_f == len(d_body):
            d_full = d_body.reshape(dh_f, dw_f)
        elif len(d_body) == 720 * 1280:
            d_full = d_body.reshape(720, 1280)
        elif len(d_body) == Hh * Ww:
            d_full = d_body.reshape(Hh, Ww)
        else:
            raise ValueError(f"unexpected depth size {len(d_body)} hdr=({dh_f},{dw_f})")
        d_sub = cv2.resize(d_full.astype(np.float32), (s8_w, s8_h), interpolation=cv2.INTER_NEAREST).astype(np.float16)
        depth_s8[idx] = d_sub

        s_full = cv2.imdecode(np.frombuffer(raws[3 * idx + 1], np.uint8), cv2.IMREAD_UNCHANGED)
        s_ch = s_full[:, :, 2] if s_full.ndim == 3 else s_full
        s_sub = cv2.resize(s_ch, (s8_w, s8_h), interpolation=cv2.INTER_NEAREST)
        seg_s8[idx] = s_sub

        img = cv2.imdecode(np.frombuffer(raws[3 * idx + 2], np.uint8), cv2.IMREAD_COLOR)
        img_rs = cv2.resize(img, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
        ok_enc, jpg = cv2.imencode(".jpg", img_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            return None
        rgb_jpegs.append(jpg.tobytes())

        r = poses_rows[k]
        R = quat_to_R(float(r["q_x"]), float(r["q_y"]), float(r["q_z"]), float(r["q_w"]))
        t = np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])], dtype=np.float64)
        poses_R_raw[idx] = R
        poses_t_raw[idx] = t

        d_flat = d_sub.astype(np.float32).reshape(-1)
        s_flat = s_sub.reshape(-1)
        # Mask out SKY class (27) in real CREStereo pseudo-GT depth!
        ok = np.isfinite(d_flat) & (d_flat > 0.2) & (d_flat < 25.0) & (s_flat != SKY)
        ray = dirs[ok] * d_flat[ok, None]
        P_raw.append((R @ ray.T).T + t)
        L_all.append(s_flat[ok])

    # Single constant session-level rigid rotation R_align mapping ground normal -> world +Z
    G_raw = np.concatenate([P[np.isin(L, WALK)] for P, L in zip(P_raw, L_all)])
    if len(G_raw) < 500:
        return None
    gnorm = fit_plane_normal_ransac(G_raw)
    R_align = rotation_align_z(gnorm)
    align_deg = float(np.degrees(np.arccos(np.clip(gnorm[2], -1.0, 1.0))))

    poses_R = np.einsum("ij,njk->nik", R_align, poses_R_raw).astype(np.float32)
    poses_t = (poses_t_raw @ R_align.T).astype(np.float32)
    P_all = [P @ R_align.T for P in P_raw]

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, b in enumerate(rgb_jpegs):
        offsets[i + 1] = offsets[i] + len(b)
    rgb_blob = np.frombuffer(b"".join(rgb_jpegs), dtype=np.uint8)

    ps = poses_t.astype(np.float64)
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

    eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
    gt_single_d = np.full(n, np.nan, dtype=np.float32)
    gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    causal_fwd = np.zeros((n, 3), dtype=np.float32)
    valid_heading = np.zeros(n, dtype=np.uint8)

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
    if len(stat) > 0:
        vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
        vc = (vox + 0.5) * VOX
        vc_h = height(vc)
        for k_c in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
            cellset.add(tuple(k_c))
    else:
        vc = np.zeros((0, 3), dtype=np.float64)
        vc_h = np.zeros(0, dtype=np.float64)

    for i in range(n):
        a = max(0, i - 2)
        b = i if i >= 2 else min(n - 1, 2)
        v = (ps[b] - ps[a])[:2]
        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
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

        if len(vc) > 0:
            rel_v = vc - ps[i]
            gt_bev_static[i] = voxelize_bev(np.column_stack([rel_v @ right, rel_v @ fwd, vc_h]), min_count=2)

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
        if len(vc) > 0:
            rel_v = vc - ps[i]
            x_t, z_t = rel_v @ tr_, rel_v @ tf
            inc = (np.abs(x_t) <= X_HALF) & (z_t >= FWD0) & (z_t <= H)
            D = corridor_d(z_t[inc], MIN_VOX_V2)
        else:
            D = float(np.inf)
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

        xq, zq = rel0 @ right, rel0 @ fwd
        strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
        g_est = np.percentile(P0[strip, 2], 5) if strip.sum() >= 50 else ps[i][2] - 1.5
        hq = P0[:, 2] - g_est
        cm = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT[0]) & (hq <= HGT[1])
        gt_single_d[i] = corridor_d(zq[cm], MIN_PTS)

    buf = io.BytesIO()
    np.savez(
        buf,
        scene=np.array(f"{scene}_{cam_name}"),
        split=np.array("real_zeroshot"),
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
        align_deg=np.array(align_deg, dtype=np.float32),
    )
    npz_bytes = buf.getvalue()
    dst.write_bytes(npz_bytes)
    return {
        "scene": f"{scene}_{cam_name}",
        "raw_scene": scene,
        "cam": cam_name,
        "n_frames": n,
        "npz_bytes": len(npz_bytes),
        "align_deg": round(align_deg, 2),
        "fx_orig": round(float(p["fx"]), 1),
    }


def write_shards_to_drive(scenes_dir: Path, file_stems: list[str], drive_dir: Path, dataset_name: str, extra_meta: dict) -> dict:
    drive_dir.mkdir(parents=True, exist_ok=True)
    local_sh = scenes_dir.parent / f"shards_{dataset_name}"
    local_sh.mkdir(parents=True, exist_ok=True)
    shards = []
    for s_idx in range(0, len(file_stems), SHARD_SIZE):
        sid = s_idx // SHARD_SIZE
        sname = f"shard_{sid:03d}.tar"
        batch = file_stems[s_idx : s_idx + SHARD_SIZE]
        lsh = local_sh / sname
        with tarfile.open(lsh, "w") as tf:
            for st in batch:
                tf.add(scenes_dir / f"{st}.npz", arcname=f"{st}.npz")
        sh_sha = sha256_file(lsh)
        sh_bytes = lsh.stat().st_size
        dsh = drive_dir / sname
        if dsh.exists():
            dsh.unlink()
        shutil.copy2(lsh, dsh)
        lsh.unlink()
        shards.append({"file": sname, "bytes": sh_bytes, "sha256": sh_sha, "n_sessions": len(batch), "scenes": batch})
    man = {
        "dataset": dataset_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sessions": len(file_stems),
        "total_shard_bytes": sum(s["bytes"] for s in shards),
        "shards": shards,
        **extra_meta,
    }
    mf = drive_dir / "manifest_sha256.json"
    if mf.exists():
        mf.unlink()
    mf.write_text(json.dumps(man, indent=2))
    return man


def extract_existing_287():
    drive_287 = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_287_v0")
    local_287 = Path("/content/sanpo_287")
    scenes_dir = local_287 / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    marker = local_287 / ".extracted_ok"
    if marker.exists() and len(list(scenes_dir.glob("*.npz"))) >= 287:
        print("[p2-pack] sanpo_287 already extracted locally", flush=True)
        return
    t0 = time.time()
    man = json.loads((drive_287 / "manifest_sha256.json").read_text())
    for sh in man["shards"]:
        dst = local_287 / sh["file"]
        shutil.copy2(drive_287 / sh["file"], dst)
        if sha256_file(dst) != sh["sha256"]:
            raise RuntimeError(f"SHA256 mismatch on {sh['file']}")
        with tarfile.open(dst, "r") as tf:
            tf.extractall(scenes_dir)
        dst.unlink()
    marker.write_text("ok")
    print(f"[p2-pack] extracted existing 287 scenes in {time.time()-t0:.1f}s", flush=True)


def main():
    t0 = time.time()
    ncpu = os.cpu_count() or 8
    n_workers = min(24, max(4, ncpu - 2))
    p2_man = json.loads(Path("/content/phase2_manifest.json").read_text())

    # 1. Extract existing 287 scenes from Drive
    extract_existing_287()

    # 2. Pack extra 450 SANPO-Synthetic train scenes
    extra_dir = Path("/content/sanpo_p2/extra_scenes")
    extra_dir.mkdir(parents=True, exist_ok=True)
    drive_extra = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_extra450_v0")
    extra_ids = p2_man["extra_synth_train_450"]

    if len(list(extra_dir.glob("*.npz"))) >= len(extra_ids):
        print(f"[p2-pack] extra450 already present locally ({len(list(extra_dir.glob('*.npz')))} files)", flush=True)
    elif (drive_extra / "manifest_sha256.json").exists():
        print("[p2-pack] extra450 already on Drive, extracting locally...", flush=True)
        man_e = json.loads((drive_extra / "manifest_sha256.json").read_text())
        for sh in man_e["shards"]:
            dst = extra_dir.parent / sh["file"]
            shutil.copy2(drive_extra / sh["file"], dst)
            with tarfile.open(dst, "r") as tf:
                tf.extractall(extra_dir)
            dst.unlink()
    else:
        print(f"[p2-pack] packing {len(extra_ids)} extra SANPO-Synthetic train scenes on {ncpu} vCPUs ({n_workers}w)...", flush=True)
        t_ex = time.time()
        ok_extra = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(pack_extra_synth_worker, (sc, str(extra_dir))): sc for sc in extra_ids}
            done = 0
            for fut in as_completed(futs):
                sc = futs[fut]
                try:
                    inf = fut.result()
                    ok_extra.append(sc)
                except Exception as e:
                    print(f"  [warn] extra synth {sc} failed: {e}", flush=True)
                done += 1
                if done % 50 == 0 or done == len(extra_ids):
                    el = time.time() - t_ex
                    print(f"  [extra450] {done}/{len(extra_ids)} ({el/done:.2f}s/sc, elapsed={el:.0f}s)", flush=True)
        ok_extra = [sc for sc in extra_ids if sc in set(ok_extra)]
        man_e = write_shards_to_drive(extra_dir, ok_extra, drive_extra, "sanpo_synth_extra450_v0", {"split": "train"})
        print(f"[p2-pack] extra450 DONE: {len(ok_extra)} scenes, {man_e['total_shard_bytes']/1e9:.2f} GB in {time.time()-t_ex:.0f}s", flush=True)

    # 3. Pack 40 SANPO-Real zero-shot real outdoor walking scenes
    real_dir = Path("/content/sanpo_p2/real_scenes")
    real_dir.mkdir(parents=True, exist_ok=True)
    drive_real = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_real_zeroshot_40_v0")
    real_specs = p2_man["sanpo_real_zeroshot_40"]

    real_manifest_ok = False
    if (drive_real / "manifest_sha256.json").exists():
        try:
            real_manifest_ok = json.loads((drive_real / "manifest_sha256.json").read_text()).get("n_sessions", 0) >= 15
        except Exception:
            real_manifest_ok = False
    if len(list(real_dir.glob("*.npz"))) >= 15:
        print(f"[p2-pack] real_zeroshot already present locally ({len(list(real_dir.glob('*.npz')))} files)", flush=True)
    elif real_manifest_ok:
        print("[p2-pack] real_zeroshot_40 already on Drive, extracting locally...", flush=True)
        man_r = json.loads((drive_real / "manifest_sha256.json").read_text())
        for sh in man_r["shards"]:
            dst = real_dir.parent / sh["file"]
            shutil.copy2(drive_real / sh["file"], dst)
            with tarfile.open(dst, "r") as tf:
                tf.extractall(real_dir)
            dst.unlink()
    else:
        print(f"[p2-pack] packing {len(real_specs)} SANPO-Real zero-shot sessions (5 Hz)...", flush=True)
        t_re = time.time()
        real_infos = []
        with ProcessPoolExecutor(max_workers=min(16, n_workers)) as pool:
            futs = [pool.submit(pack_real_worker, (r["scene"], r["cam"], str(real_dir))) for r in real_specs]
            for fut in as_completed(futs):
                try:
                    inf = fut.result()
                    if inf is not None:
                        real_infos.append(inf)
                except Exception as e:
                    print(f"  [warn] real session failed: {e}", flush=True)
        real_stems = sorted(r["scene"] for r in real_infos)
        man_r = write_shards_to_drive(
            real_dir, real_stems, drive_real, "sanpo_real_zeroshot_40_v0", {"sessions": real_infos}
        )
        print(
            f"[p2-pack] real_zeroshot DONE: {len(real_stems)}/{len(real_specs)} sessions, "
            f"{sum(r['n_frames'] for r in real_infos)} real frames, "
            f"{man_r['total_shard_bytes']/1e6:.1f} MB in {time.time()-t_re:.0f}s",
            flush=True,
        )

    print(f"[p2-pack] ALL DATA READY in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
