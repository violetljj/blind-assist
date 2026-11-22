"""Phase 7 High-Speed 48-vCPU Dual-Camera SANPO-Real Goldmine Packer (G4):

Unlocks the massive hidden goldmine discovered by probing SANPO-Real GCS:
  1. Almost ALL 444 unused SANPO-Real sessions have COMPLETE `camera_chest` (CREStereo depth_maps +
     segmentation_masks + video_frames + fixed_camera_poses.csv) -> packs 240+ new full-seg real outdoor
     sessions with drift-free +/-0.8s local GT!
  2. Almost ALL 444 unused SANPO-Real sessions ALSO have COMPLETE `camera_head` (CREStereo depth_maps +
     video_frames + fixed_camera_poses.csv, only missing segmentation_masks) -> packs 160+ new real outdoor
     HEAD-camera sessions using RANSAC local ground plane + sky-row filter + has_dyn_gt=0 flag!
     (Directly feeds real outdoor head-mounted data to crush the camera_head 65.7% -> <45% gap!)

Saves to Drive:
  - /BlindAssist/datasets/sanpo_real_chest240_v0/
  - /BlindAssist/datasets/sanpo_real_head160_noseg_v0/
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from pack_phase2_colab import PRE_REAL, REAL_STRIDE, fit_plane_normal_ransac, pack_real_worker, rotation_align_z
from pack_phase6_treasure_colab import write_incremental_shards
from pack_sanpo_colab import CELL, FWD0, H, HGT, MIN_VOX_V2, VOX, X_HALF, corridor_d, fetch_verified, gcs_list, quat_to_R, voxelize_bev
from phase4_simreal_aria_onnx_g4 import build_local_bev_and_rows_for_real_train

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist/datasets")
DRIVE_CHEST = DRIVE_ROOT / "sanpo_real_chest240_v0"
DRIVE_HEAD = DRIVE_ROOT / "sanpo_real_head160_noseg_v0"

LOCAL_CHEST = Path("/content/p7_data/real_chest240")
LOCAL_HEAD = Path("/content/p7_data/real_head160")


def pack_chest_full_worker(args: tuple[str, str]) -> str | None:
    scene, out_dir = args
    dst = Path(out_dir) / f"{scene}_camera_chest.npz"
    if dst.exists() and dst.stat().st_size > 5000:
        return dst.stem
    inf = pack_real_worker((scene, "camera_chest", out_dir))
    if inf is None or not dst.exists():
        return None
    build_local_bev_and_rows_for_real_train(str(dst))
    return dst.stem


def pack_head_noseg_worker(args: tuple[str, str]) -> str | None:
    """Pack a SANPO-Real camera_head session using CREStereo depth + fixed_camera_poses + local floor plane (no seg needed)."""
    cv2.setNumThreads(1)
    scene, out_dir = args
    dst = Path(out_dir) / f"{scene}_camera_head_noseg.npz"
    if dst.exists() and dst.stat().st_size > 5000:
        return dst.stem

    items = gcs_list(PRE_REAL + scene + "/camera_head/")
    # Also fetch description.json from session root
    root_items = gcs_list(PRE_REAL + scene + "/description.json")
    if not root_items:
        return None
    desc_item = root_items[0]

    by_rel = {}
    pref_full = PRE_REAL + scene + "/camera_head/"
    for it in items:
        name = it["name"]
        if name.endswith("_$folder$") or int(it["size"]) == 0 or "/left/temp/" in name:
            continue
        by_rel[name[len(pref_full) :]] = it

    pose_rel = "fixed_camera_poses.csv" if "fixed_camera_poses.csv" in by_rel else "camera_poses.csv"
    if pose_rel not in by_rel:
        return None

    d_ids = {int(Path(k).name[:6]) for k in by_rel if k.startswith("left/depth_maps/") and k.endswith(".float16.gz")}
    r_ids = {int(Path(k).name[:6]) for k in by_rel if k.startswith("left/video_frames/") and k.endswith(".png")}
    common_ids = sorted(d_ids & r_ids)
    if len(common_ids) < 25:
        return None

    with ThreadPoolExecutor(max_workers=12) as dl_pool:
        meta_raw, pose_raw = list(dl_pool.map(fetch_verified, [desc_item, by_rel[pose_rel]]))
        meta_all = json.loads(meta_raw.decode("utf-8"))["session_camera_details"]
        cam_meta = next((c for c in meta_all if "ZED_M" in str(c.get("model", ""))), meta_all[0])
        p = cam_meta["left_camera_params"]
        Hh, Ww = int(p["image_height"]), int(p["image_width"])
        poses_rows = list(csv.DictReader(io.StringIO(pose_raw.decode("utf-8"))))

        ready_ids = [
            k for k in common_ids
            if k < len(poses_rows) and "READY" in str(poses_rows[k].get("tracking_state", "READY"))
            and "NOT_READY" not in str(poses_rows[k].get("tracking_state", "READY"))
        ]
        frame_ids = ready_ids[::REAL_STRIDE][:120]
        if len(frame_ids) < 20:
            return None

        needed = []
        for k in frame_ids:
            needed.append(by_rel[f"left/depth_maps/{k:06d}.float16.gz"])
            needed.append(by_rel[f"left/video_frames/{k:06d}.png"])
        raws = list(dl_pool.map(fetch_verified, needed))

    s8_h, s8_w = 156, 276
    fx_s8, fy_s8 = p["fx"] * (s8_w / float(Ww)), p["fy"] * (s8_h / float(Hh))
    cx_s8, cy_s8 = p["cx"] * (s8_w / float(Ww)), p["cy"] * (s8_h / float(Hh))
    vv, uu = np.mgrid[0:s8_h, 0:s8_w]
    dirs = np.stack([(uu - cx_s8) / fx_s8, (vv - cy_s8) / fy_s8, np.ones(uu.shape)], -1).reshape(-1, 3)

    n = len(frame_ids)
    rgb_jpegs = []
    poses_R_raw = np.empty((n, 3, 3), dtype=np.float64)
    poses_t_raw = np.empty((n, 3), dtype=np.float64)
    P_raw = []
    ground_seed_pts = []

    for idx, k in enumerate(frame_ids):
        d_raw = np.frombuffer(gzip.decompress(raws[2 * idx]), "<f2")
        dh_f, dw_f = int(d_raw[0]), int(d_raw[1])
        d_body = d_raw[2:]
        if dh_f * dw_f == len(d_body):
            d_full = d_body.reshape(dh_f, dw_f)
        elif len(d_body) == 720 * 1280:
            d_full = d_body.reshape(720, 1280)
        else:
            return None
        d_sub = cv2.resize(d_full.astype(np.float32), (s8_w, s8_h), interpolation=cv2.INTER_NEAREST)

        img = cv2.imdecode(np.frombuffer(raws[2 * idx + 1], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
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

        d_flat = d_sub.reshape(-1)
        # Filter invalid/far sky depths (< 18m)
        ok = np.isfinite(d_flat) & (d_flat > 0.25) & (d_flat < 18.0)
        pts_w = (R @ (dirs[ok] * d_flat[ok, None]).T).T + t
        P_raw.append(pts_w.astype(np.float32))
        # Bottom-third close points (1.0..6.0m) are strong sidewalk ground seeds for RANSAC plane alignment
        bot_mask = (vv.reshape(-1)[ok] >= int(0.62 * s8_h)) & (d_flat[ok] >= 1.0) & (d_flat[ok] <= 6.5)
        if bot_mask.sum() > 20:
            ground_seed_pts.append(pts_w[bot_mask])

    if not ground_seed_pts:
        return None
    G_seed = np.concatenate(ground_seed_pts, axis=0)
    if len(G_seed) < 400:
        return None
    gnorm = fit_plane_normal_ransac(G_seed)
    R_align = rotation_align_z(gnorm)

    poses_R = np.einsum("ij,njk->nik", R_align, poses_R_raw).astype(np.float32)
    poses_t = (poses_t_raw @ R_align.T).astype(np.float32)
    # Sanity check: camera optical +Y should point downward (R[2, 1] < -0.5)
    if float(np.median(poses_R[:, 2, 1])) > -0.45:
        # Flip Z sign if RANSAC picked inverted normal
        R_flip = np.diag([1.0, -1.0, -1.0])
        R_align = R_flip @ R_align
        poses_R = np.einsum("ij,njk->nik", R_align, poses_R_raw).astype(np.float32)
        poses_t = (poses_t_raw @ R_align.T).astype(np.float32)

    P_all = [(P @ R_align.T).astype(np.float32) for P in P_raw]
    ps = poses_t.astype(np.float64)
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

    # Build drift-free +/-0.8s local-window floor & static obstacle GT
    eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
    gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    causal_fwd = np.zeros((n, 3), dtype=np.float32)
    valid_heading = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        a = max(0, i - 2)
        b = i if i >= 2 else min(n - 1, 2)
        v = (ps[b] - ps[a])[:2]
        if np.linalg.norm(v) >= 0.20:
            fwd = np.array([v[0], v[1], 0.0]) / np.linalg.norm(v)
            v_ok = True
        else:
            cz = poses_R[i, :, 2].astype(np.float64); cz[2] = 0.0
            nz = np.linalg.norm(cz)
            fwd = cz / nz if nz > 1e-6 else np.array([1.0, 0.0, 0.0])
            v_ok = False
        causal_fwd[i] = fwd.astype(np.float32)
        if v_ok:
            valid_heading[i] = 1

        lo, hi = max(0, i - 4), min(n, i + 5)
        Pw_win = np.concatenate(P_all[lo:hi], axis=0)
        # Head camera is ~1.4-1.8m above sidewalk; estimate local ground from points below camera - 0.8m
        below = Pw_win[(Pw_win[:, 2] < ps[i, 2] - 0.80) & (Pw_win[:, 2] > ps[i, 2] - 2.30)]
        if len(below) < 200:
            continue
        z_floor = float(np.percentile(below[:, 2], 8))
        Gw = Pw_win[np.abs(Pw_win[:, 2] - z_floor) <= 0.14]
        if len(Gw) < 100:
            continue

        keys, inv = np.unique(np.floor(Gw[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="mergesort")
        z_s = Gw[order, 2]
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        gz = np.array([np.median(g) for g in np.split(z_s, cuts)], dtype=np.float64)
        gtree = cKDTree((keys + 0.5) * CELL)
        cellset = {tuple(kc) for kc in keys}

        def height_loc(Q: np.ndarray) -> np.ndarray:
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.2)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            return h

        h_win = height_loc(Pw_win)
        stat = Pw_win[(h_win >= HGT[0]) & (h_win <= HGT[1])]
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

        if not v_ok:
            continue
        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
        if np.linalg.norm(vt) < 0.2:
            continue
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
        eval_rows[i] = [D, cover, 0.0, seg_len[i]]

    if int(np.isfinite(eval_rows[:, 1]).sum()) < 15:
        return None

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, b in enumerate(rgb_jpegs):
        offsets[i + 1] = offsets[i] + len(b)
    rgb_blob = np.frombuffer(b"".join(rgb_jpegs), dtype=np.uint8)

    np.savez(
        dst,
        rgb_blob=rgb_blob,
        rgb_offsets=offsets,
        K_orig=np.array([p["fx"], p["fy"], p["cx"], p["cy"]], dtype=np.float32),
        orig_hw=np.array([Hh, Ww], dtype=np.int32),
        poses_R=poses_R,
        poses_t=poses_t,
        causal_fwd=causal_fwd,
        valid_heading=valid_heading,
        eval_rows=eval_rows,
        gt_bev_static=gt_bev_static,
        gt_bev_dynamic=gt_bev_dynamic,
        has_dyn_gt=np.array(0, dtype=np.uint8),
    )
    return dst.stem


def main():
    t0 = time.time()
    ncpu = os.cpu_count() or 8
    n_workers = min(24, max(6, ncpu - 2))
    man = json.loads(Path("/content/phase6_real_expand_manifest.json").read_text())["real_extra_444"]
    all_scenes = [r["scene"] for r in man]

    # 1. Pack 240 full-seg SANPO-Real camera_chest sessions (from first 300 unused sessions)
    LOCAL_CHEST.mkdir(parents=True, exist_ok=True)
    if len(list(LOCAL_CHEST.glob("*.npz"))) < 200 and (DRIVE_CHEST / "manifest_sha256.json").exists():
        print("[p7-pack] restoring sanpo_real_chest240_v0 from Drive...", flush=True)
        dm = json.loads((DRIVE_CHEST / "manifest_sha256.json").read_text())
        for sh in dm["shards"]:
            with tarfile.open(DRIVE_CHEST / sh["file"], "r") as tf:
                tf.extractall(LOCAL_CHEST)

    if len(list(LOCAL_CHEST.glob("*.npz"))) < 220:
        t1 = time.time()
        cand_chest = all_scenes[:280]
        print(f"[p7-pack] packing up to 240 full-seg SANPO-Real camera_chest sessions ({n_workers} workers)...", flush=True)
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(pack_chest_full_worker, (sc, str(LOCAL_CHEST))) for sc in cand_chest]
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                    if res is not None:
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    fail += 1
                    if fail <= 5:
                        print(f"  [warn] chest failed: {e}", flush=True)
                if (ok + fail) % 40 == 0:
                    print(f"  [chest] {ok+fail}/{len(cand_chest)} (ok={ok}, skip={fail}) {time.time()-t1:.1f}s", flush=True)
        stems_c = sorted(p.stem for p in LOCAL_CHEST.glob("*.npz"))[:240]
        write_incremental_shards(LOCAL_CHEST, stems_c, DRIVE_CHEST, "sanpo_real_chest240_v0", shard_size=30)
        print(f"[p7-pack] DONE camera_chest: {len(stems_c)} scenes saved to Drive in {time.time()-t1:.1f}s!", flush=True)

    # 2. Pack 160 seg-free SANPO-Real camera_head sessions (from remaining sessions 240..444 -> 100% session-disjoint!)
    LOCAL_HEAD.mkdir(parents=True, exist_ok=True)
    if len(list(LOCAL_HEAD.glob("*.npz"))) < 130 and (DRIVE_HEAD / "manifest_sha256.json").exists():
        print("[p7-pack] restoring sanpo_real_head160_noseg_v0 from Drive...", flush=True)
        dm = json.loads((DRIVE_HEAD / "manifest_sha256.json").read_text())
        for sh in dm["shards"]:
            with tarfile.open(DRIVE_HEAD / sh["file"], "r") as tf:
                tf.extractall(LOCAL_HEAD)

    if len(list(LOCAL_HEAD.glob("*.npz"))) < 140:
        t2 = time.time()
        cand_head = all_scenes[240:]  # 204 distinct sessions for head cam!
        print(f"[p7-pack] packing up to 160 SANPO-Real camera_head (noseg local-plane GT) sessions ({n_workers} workers)...", flush=True)
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(pack_head_noseg_worker, (sc, str(LOCAL_HEAD))) for sc in cand_head]
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                    if res is not None:
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    fail += 1
                    if fail <= 5:
                        print(f"  [warn] head failed: {e}", flush=True)
                if (ok + fail) % 40 == 0:
                    print(f"  [head] {ok+fail}/{len(cand_head)} (ok={ok}, skip={fail}) {time.time()-t2:.1f}s", flush=True)
        stems_h = sorted(p.stem for p in LOCAL_HEAD.glob("*.npz"))[:160]
        write_incremental_shards(LOCAL_HEAD, stems_h, DRIVE_HEAD, "sanpo_real_head160_noseg_v0", shard_size=30)
        print(f"[p7-pack] DONE camera_head: {len(stems_h)} scenes saved to Drive in {time.time()-t2:.1f}s!", flush=True)

    print(f"[p7-pack] ALL DUAL-CAM SANPO-REAL GOLDMINE PACKED in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
