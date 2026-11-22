"""Phase 13 48-vCPU Turbo Mega-Data Factory (Colab G4 -> 20 TB Google Drive):

Massive strictly-disjoint multi-domain expansion (ZERO overlap with eval sets or official protected test splits!):
  Stage A1: ALL remaining unused full-semantic SANPO-Real train sessions (camera_chest + camera_head)
            -> Drive /BlindAssist/datasets/sanpo_real_p13_extra_v0/
  Stage A2: GPU DeepLabV3 dynamic-pedestrian cleaning on the 100 existing SANPO-Real camera_head sessions
            -> Drive /BlindAssist/datasets/sanpo_real_head100_autoseg_v0/
  Stage B:  +500 strictly visit-disjoint ARKitScenes real indoor LiDAR captures (300 -> 800 total!)
            -> Drive /BlindAssist/datasets/arkit_indoor_p13_500_v0/
  Stage C:  TartanAir 90-deg wide-FOV indoor/alley/urban/pitch-dark-night (8 envs, exact NED->Z-up + depth_s8!)
            -> Drive /BlindAssist/datasets/tartanair_widefov_p13_v0/
  Stage D:  +450 unused official SANPO-Synthetic train sessions (627 -> 1,077 total!)
            -> Drive /BlindAssist/datasets/sanpo_synth_p13_450_v0/

Every stage writes SHA256-verified .tar shards + manifest_sha256.json to Drive and keeps local .npz under
/content/p13_data/ so EgoClear-v10 can immediately start training on the exact same G4 session!
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from pack_arkit_colab import pack_arkit_worker, write_arkit_shards
from pack_phase2_colab import PRE_REAL, pack_real_worker
from pack_phase6_treasure_colab import write_incremental_shards
from pack_sanpo_colab import CELL, FWD0, H, HGT, MIN_VOX_V2, VOX, X_HALF, corridor_d, process_session, voxelize_bev
from phase4_simreal_aria_onnx_g4 import build_local_bev_and_rows_for_real_train
from train_and_eval_g4 import decode_rgb_batch

DRIVE_DATA = Path("/content/drive/MyDrive/BlindAssist/datasets")
LOCAL_P13 = Path("/content/p13_data")

# Exact NED -> EgoClear Z-up conversion (det = +1 proper rotations!)
M_W_NED2ZUP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
M_C_OPT2NED = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-Research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _pack_real_one(args: tuple[str, str, str]) -> str | None:
    sc, cam, out_dir = args
    dst1 = Path(out_dir) / f"{sc}_{cam}.npz"
    dst2 = Path(out_dir) / f"{sc}__{cam}.npz"
    dst3 = Path(out_dir) / f"{sc}.npz"
    for d in (dst1, dst2, dst3):
        if d.exists() and d.stat().st_size > 5000:
            return d.stem
    try:
        inf = pack_real_worker((sc, cam, out_dir))
    except Exception:
        return None
    if inf is None:
        return None
    for d in (dst1, dst2, dst3):
        if d.exists() and d.stat().st_size > 5000:
            build_local_bev_and_rows_for_real_train(str(d))
            return d.stem
    return None


def stage_a1_sanpo_real_remaining(n_workers: int) -> Path:
    drive_dir = DRIVE_DATA / "sanpo_real_p13_extra_v0"
    local_dir = LOCAL_P13 / "sanpo_real_extra"
    local_dir.mkdir(parents=True, exist_ok=True)
    if (drive_dir / "manifest_sha256.json").exists():
        if len(list(local_dir.glob("*.npz"))) < 5:
            man = json.loads((drive_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                with tarfile.open(drive_dir / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        print(f"[Stage A1] SANPO-Real extra already ready ({len(list(local_dir.glob('*.npz')))} scenes)", flush=True)
        return local_dir

    t0 = time.time()
    train_ids = _fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/splits/train_session_ids.txt").split()
    test_ids = set(_fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/splits/test_session_ids.txt").split())

    used_pairs = set()
    used_eval40 = set()
    for p in Path("/content/sanpo_p2/real_scenes").glob("*.npz"):
        used_eval40.add(p.stem.split("_camera_")[0].split("__")[0])
    for ldir in ["/content/sanpo_p4/real_train_scenes", "/content/p6_data/real_extra444", "/content/p7_data/real_chest240"]:
        lp = Path(ldir)
        if lp.exists():
            for p in lp.glob("*.npz"):
                st = p.stem
                cam = "camera_head" if "camera_head" in st else "camera_chest"
                sc = st.replace("_camera_chest", "").replace("_camera_head", "").replace("__camera_chest", "").replace("__camera_head", "")
                used_pairs.add((sc, cam))

    tasks = []
    for sc in train_ids:
        if sc in test_ids or sc in used_eval40:
            continue
        for cam in ("camera_chest", "camera_head"):
            if (sc, cam) not in used_pairs:
                tasks.append((sc, cam, str(local_dir)))

    print(f"[Stage A1] Scanning & packing up to {len(tasks)} remaining SANPO-Real (scene,cam) pairs ({n_workers} workers)...", flush=True)
    ok = skip = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_pack_real_one, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                ok += 1
            else:
                skip += 1
            if (ok + skip) % 80 == 0 or (ok + skip) == len(tasks):
                print(f"  [A1 SANPO-Real] {ok+skip}/{len(tasks)} ok={ok} skip={skip} ({time.time()-t0:.1f}s)", flush=True)

    stems = sorted(p.stem for p in local_dir.glob("*.npz"))
    if stems:
        write_incremental_shards(local_dir, stems, drive_dir, "sanpo_real_p13_extra_v0", shard_size=30)
    print(f"[Stage A1] DONE SANPO-Real extra: {len(stems)} new full-seg sessions in {time.time()-t0:.1f}s -> {drive_dir}", flush=True)
    return local_dir


def stage_a2_autoseg_head100() -> Path:
    """Use GPU DeepLabV3-ResNet50 (COCO) to auto-detect dynamic pedestrians/vehicles on the 100 real camera_head sessions!"""
    import torch
    import torchvision

    drive_dir = DRIVE_DATA / "sanpo_real_head100_autoseg_v0"
    local_dir = LOCAL_P13 / "sanpo_real_head100_autoseg"
    local_dir.mkdir(parents=True, exist_ok=True)
    if (drive_dir / "manifest_sha256.json").exists():
        if len(list(local_dir.glob("*.npz"))) < 50:
            man = json.loads((drive_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                with tarfile.open(drive_dir / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        print(f"[Stage A2] SANPO-Real head100 autoseg already ready ({len(list(local_dir.glob('*.npz')))} scenes)", flush=True)
        return local_dir

    src_dir = Path("/content/p7_data/real_head160")
    if not src_dir.exists() or len(list(src_dir.glob("*.npz"))) < 10:
        print("[Stage A2] Skipping: /content/p7_data/real_head160 not found", flush=True)
        return local_dir

    t0 = time.time()
    device = torch.device("cuda")
    seg_m = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").to(device).eval()
    # COCO/VOC 21 classes: 2=bicycle, 6=bus, 7=car, 8=cat, 12=dog, 14=motorbike, 15=person
    dyn_cls = torch.tensor([2, 6, 7, 8, 12, 14, 15], device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    eval40_ids = {p.stem.split("_camera_")[0].split("__")[0] for p in Path("/content/sanpo_p2/real_scenes").glob("*.npz")}
    files = sorted(src_dir.glob("*.npz"))
    n_ok = n_dyn_marked = 0
    for f in files:
        sc_base = f.stem.split("_camera_")[0]
        if sc_base in eval40_ids:
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["eval_rows"])
        rgb = decode_rgb_batch(d, np.arange(n))  # (n, 3, H, W) [0,1]
        dyn_frac = np.zeros(n, dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, n, 32):
                x = (torch.from_numpy(rgb[i : i + 32]).to(device) - mean) / std
                pred = seg_m(x)["out"].argmax(dim=1)  # (b, H, W)
                is_dyn = torch.isin(pred, dyn_cls)
                # Central walking corridor region of image: middle 50% width, lower 65% height
                c_dyn = is_dyn[:, int(0.25 * RGB_H) :, int(0.25 * RGB_W) : int(0.75 * RGB_W)].float().mean(dim=(1, 2))
                dyn_frac[i : i + 32] = c_dyn.cpu().numpy()

        er = d["eval_rows"].copy()
        gt_st = d["gt_bev_static"].copy()
        gt_dy = d["gt_bev_dynamic"].copy()
        # Any frame with >= 2.5% dynamic person/vehicle in central corridor is flagged dynamic & its static hazard masked!
        is_d_fr = dyn_frac >= 0.025
        er[is_d_fr, 2] = 1.0
        # If dynamic obstacle is right in front and could have caused false static corridor label, mark cover=NaN so loss ignores static label on dyn-ambiguous frames!
        ambig = is_d_fr & np.isfinite(er[:, 0]) & (er[:, 0] <= H)
        er[ambig, 1] = np.nan
        gt_dy[is_d_fr] = np.maximum(gt_dy[is_d_fr], gt_st[is_d_fr])
        gt_st[ambig] = 0
        n_dyn_marked += int(is_d_fr.sum())

        d["eval_rows"] = er
        d["gt_bev_static"] = gt_st
        d["gt_bev_dynamic"] = gt_dy
        d["has_dyn_gt"] = np.array(1, dtype=np.uint8)
        np.savez(local_dir / f.name, **d)
        n_ok += 1

    del seg_m
    torch.cuda.empty_cache()
    stems = sorted(p.stem for p in local_dir.glob("*.npz"))
    if stems:
        write_incremental_shards(local_dir, stems, drive_dir, "sanpo_real_head100_autoseg_v0", shard_size=30)
    print(f"[Stage A2] DONE head100 DeepLabV3 auto-seg: {len(stems)} scenes, {n_dyn_marked} dyn frames cleaned in {time.time()-t0:.1f}s!", flush=True)
    return local_dir


def stage_b_arkit_500(n_workers: int, target_n: int = 500) -> Path:
    drive_dir = DRIVE_DATA / "arkit_indoor_p13_500_v0"
    local_dir = LOCAL_P13 / "arkit_500"
    local_dir.mkdir(parents=True, exist_ok=True)
    if (drive_dir / "manifest_sha256.json").exists():
        if len(list(local_dir.glob("*.npz"))) < int(target_n * 0.85):
            man = json.loads((drive_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                with tarfile.open(drive_dir / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        print(f"[Stage B] ARKitScenes +500 already ready ({len(list(local_dir.glob('*.npz')))} scenes)", flush=True)
        return local_dir

    t0 = time.time()
    csv_txt = _fetch_text("https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv")
    rows = list(csv.DictReader(io.StringIO(csv_txt)))

    used_vids = set()
    for ldir in ["/content/arkit_p5/train_scenes", "/content/p6_data/arkit_extra210", "/content/arkit_p5/val_scenes"]:
        lp = Path(ldir)
        if lp.exists():
            for p in lp.glob("*.npz"):
                used_vids.add(p.stem.replace("arkit_", ""))

    # Strictly visit-disjoint from val AND already-used train!
    val_visits = {r["visit_id"] for r in rows if r["fold"] != "Training" and r["visit_id"] != "NA"}
    used_visits = {r["visit_id"] for r in rows if r["video_id"] in used_vids and r["visit_id"] != "NA"}
    ban_visits = val_visits | used_visits

    cand = []
    seen_visits = set()
    for r in rows:
        if r["fold"] != "Training":
            continue
        vid, vis = r["video_id"], r["visit_id"]
        if vid in used_vids:
            continue
        if vis != "NA" and (vis in ban_visits or vis in seen_visits):
            continue
        if vis != "NA":
            seen_visits.add(vis)
        cand.append({"video_id": vid, "fold": "Training"})

    # Deterministic shuffle for diverse indoor scenes
    rng = np.random.default_rng(20261121)
    perm = rng.permutation(len(cand))
    pool = [cand[i] for i in perm[: min(len(cand), target_n * 3)]]

    existing = sorted(p.stem for p in local_dir.glob("*.npz"))
    print(f"[Stage B] Existing cached ARKitScenes in {local_dir}: {len(existing)}/{target_n}", flush=True)
    idx = 0
    done_vids = {st.replace("arkit_", "") for st in existing}
    pool = [r for r in pool if r["video_id"] not in done_vids]
    n_ok = len(existing)
    n_skip = 0

    while n_ok < target_n and idx < len(pool):
        need = target_n - n_ok
        bsz = min(len(pool) - idx, max(40, min(90, int(need * 1.35))))
        chunk = pool[idx : idx + bsz]
        idx += bsz
        ex = ProcessPoolExecutor(max_workers=min(n_workers, 24), mp_context=mp.get_context("spawn"))
        futs = {ex.submit(pack_arkit_worker, (r["video_id"], r["fold"], str(local_dir), 20)): r for r in chunk}
        try:
            for fut in as_completed(futs, timeout=240):
                try:
                    inf = fut.result(timeout=5)
                    if inf is not None:
                        n_ok += 1
                    else:
                        n_skip += 1
                except Exception:
                    n_skip += 1
                if (n_ok + n_skip) % 25 == 0 or n_ok >= target_n:
                    print(f"  [B ARKitScenes] ok={n_ok}/{target_n} skip={n_skip} elapsed={time.time()-t0:.1f}s", flush=True)
                if n_ok >= target_n:
                    break
        except TimeoutError:
            print(f"  [B ARKitScenes] batch timeout after 240s (ok={n_ok}/{target_n}); moving on cleanly!", flush=True)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    stems = sorted(p.stem for p in local_dir.glob("*.npz"))[:target_n]
    write_arkit_shards(local_dir, stems, drive_dir, "arkit_indoor_p13_500_v0", {"n_ok": len(stems)})
    print(f"[Stage B] DONE ARKitScenes +500: {len(stems)} scenes in {time.time()-t0:.1f}s!", flush=True)
    return local_dir


def _pack_tartan_traj_worker(args: tuple[str, str, str, str, str]) -> list[str]:
    """Pack one TartanAir trajectory (P00x) into 1..3 clean 5Hz (stride=2) 120-frame clips with exact Z-up & depth_s8!"""
    cv2.setNumThreads(1)
    env, diff, traj, env_dir_s, out_dir_s = args
    env_dir = Path(env_dir_s)
    out_dir = Path(out_dir_s)
    t_dir = env_dir / env / diff / traj
    pose_p = t_dir / "pose_left.txt"
    img_dir = t_dir / "image_left"
    dep_dir = t_dir / "depth_left"
    if not (pose_p.exists() and img_dir.exists() and dep_dir.exists()):
        return []

    raw_poses = np.loadtxt(str(pose_p), dtype=np.float64)
    if raw_poses.ndim != 2 or len(raw_poses) < 60:
        return []

    # Stride=2 (10Hz -> 5Hz), split into <=120-frame clips (max 3 clips/traj for diversity)
    idx_all = np.arange(0, len(raw_poses), 2)
    clips_out = []
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    Hh, Ww = 480, 640
    S_T = 4  # 120x160 dense grid for exact BEV voxelization (~19.2k pts/fr)
    sd_h, sd_w = Hh // S_T, Ww // S_T
    vv, uu = np.mgrid[0:sd_h, 0:sd_w]
    u_s = (uu * S_T + 0.5 * S_T).reshape(-1).astype(np.float64)
    v_s = (vv * S_T + 0.5 * S_T).reshape(-1).astype(np.float64)
    ray_x = (u_s - cx) / fx
    ray_y = (v_s - cy) / fy

    for ci in range(min(3, len(idx_all) // 60)):
        sl = idx_all[ci * 110 : ci * 110 + 120]
        if len(sl) < 40:
            continue
        cname = f"tartan_{env}_{diff}_{traj}_c{ci:02d}"
        out_p = out_dir / f"{cname}.npz"
        if out_p.exists() and out_p.stat().st_size > 5000:
            clips_out.append(cname)
            continue

        n = len(sl)
        Rs = np.empty((n, 3, 3), dtype=np.float64)
        ps = np.empty((n, 3), dtype=np.float64)
        jpegs = []
        depth_s8 = np.zeros((n, 60, 80), dtype=np.float16)
        P_all = []

        valid_clip = True
        for k, fi in enumerate(sl):
            ip = img_dir / f"{fi:06d}_left.png"
            dp = dep_dir / f"{fi:06d}_left_depth.npy"
            if not (ip.exists() and dp.exists()):
                valid_clip = False
                break
            row = raw_poses[fi]
            t_ned = row[:3]
            R_ned = Rotation.from_quat(row[3:7]).as_matrix()  # xyzw
            R_w = M_W_NED2ZUP @ R_ned @ M_C_OPT2NED
            t_w = M_W_NED2ZUP @ t_ned
            Rs[k] = R_w
            ps[k] = t_w

            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if bgr is None:
                valid_clip = False
                break
            bgr_rs = cv2.resize(bgr, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
            _, enc = cv2.imencode(".jpg", bgr_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            jpegs.append(enc.tobytes())

            dep = np.load(str(dp)).astype(np.float32)
            depth_s8[k] = cv2.resize(dep, (80, 60), interpolation=cv2.INTER_NEAREST).astype(np.float16)
            d_sub = dep[::S_T, ::S_T].reshape(-1).astype(np.float64)
            ok = np.isfinite(d_sub) & (d_sub > 0.25) & (d_sub < 14.0)
            if ok.sum() > 50:
                dk = d_sub[ok]
                pc = np.stack([ray_x[ok] * dk, ray_y[ok] * dk, dk], axis=-1)
                P_all.append(((R_w @ pc.T).T + t_w).astype(np.float32))
            else:
                P_all.append(np.zeros((0, 3), dtype=np.float32))

        if not valid_clip:
            continue

        # Sanity check: camera roughly upright (R_c2w[2,1] < -0.5) & mostly horizontal motion (filter pure vertical drone climbs)
        if float(np.median(Rs[:, 2, 1])) > -0.5:
            continue
        dxyz = np.diff(ps, axis=0)
        dxy = np.linalg.norm(dxyz[:, :2], axis=1)
        dz = np.abs(dxyz[:, 2])
        if float(dxy.sum()) < 3.0 or float(dz.sum()) / max(float(dxy.sum()), 1e-3) > 0.45:
            continue

        seg_len = np.r_[0.0, dxy]
        causal_fwd = np.zeros((n, 3), dtype=np.float32)
        valid_heading = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            a = max(0, i - 2)
            b = i if i >= 2 else min(n - 1, 2)
            v = (ps[b] - ps[a])[:2]
            if np.linalg.norm(v) >= 0.12:
                causal_fwd[i] = np.array([v[0], v[1], 0.0], dtype=np.float32) / float(np.linalg.norm(v))
                valid_heading[i] = 1
            else:
                cz = Rs[i, :, 2].copy(); cz[2] = 0.0
                nz = float(np.linalg.norm(cz))
                causal_fwd[i] = (cz / nz).astype(np.float32) if nz > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
        gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
        gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)

        for i in range(n):
            lo, hi = max(0, i - 4), min(n, i + 5)
            pts_l = [P_all[j] for j in range(lo, hi) if len(P_all[j]) > 0]
            if not pts_l:
                continue
            Pw = np.concatenate(pts_l, axis=0).astype(np.float64)
            # Estimate local floor below camera (within 0.4m..2.5m below camera -> realistic walking height!)
            below = Pw[(Pw[:, 2] < ps[i, 2] - 0.35) & (Pw[:, 2] > ps[i, 2] - 2.6)]
            if len(below) < 150:
                continue
            z_floor = float(np.percentile(below[:, 2], 8))
            keys, inv = np.unique(np.floor(below[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
            inv = inv.ravel()
            order = np.argsort(inv, kind="mergesort")
            cuts = np.flatnonzero(np.diff(inv[order])) + 1
            gz = np.array([np.percentile(g, 10) for g in np.split(below[order, 2], cuts)], dtype=np.float64)
            gtree = cKDTree((keys + 0.5) * CELL)
            cellset = {tuple(k) for k in keys}

            def h_loc(Q: np.ndarray) -> np.ndarray:
                dg, ig = gtree.query(Q[:, :2], distance_upper_bound=0.8)
                okg = np.isfinite(dg)
                hh = np.full(len(Q), np.nan)
                hh[okg] = Q[okg, 2] - gz[ig[okg]]
                if np.any(~okg):
                    hh[~okg] = Q[~okg, 2] - z_floor
                return hh

            hw = h_loc(Pw)
            stat = Pw[(hw >= HGT[0]) & (hw <= HGT[1])]
            fwd = causal_fwd[i].astype(np.float64)
            right = np.array([fwd[1], -fwd[0], 0.0])
            if len(stat) > 0:
                vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
                vc = (vox + 0.5) * VOX
                vc_h = h_loc(vc)
                rel_v = vc - ps[i]
                gt_bev_static[i] = voxelize_bev(np.column_stack([rel_v @ right, rel_v @ fwd, vc_h]), min_count=2)
                for kc in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
                    cellset.add(tuple(kc))
            else:
                vc = np.zeros((0, 3), dtype=np.float64)

            if not valid_heading[i]:
                continue
            vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
            if np.linalg.norm(vt) < 0.12:
                continue
            tf = np.array([vt[0], vt[1], 0.0]) / np.linalg.norm(vt)
            tr_ = np.array([tf[1], -tf[0], 0.0])
            if len(vc) > 0:
                rel_v = vc - ps[i]
                xt, zt = rel_v @ tr_, rel_v @ tf
                inc = (np.abs(xt) <= X_HALF) & (zt >= FWD0) & (zt <= H)
                D = corridor_d(zt[inc], MIN_VOX_V2)
            else:
                D = float(np.inf)
            reach = min(D, H)
            samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
            cover = float(np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp])) if len(samp) else 1.0
            eval_rows[i] = [D, cover, 0.0, seg_len[i]]

        if int(np.isfinite(eval_rows[:, 1]).sum()) < 20:
            continue

        offs = np.zeros(n + 1, dtype=np.int64)
        for idx_j, jb in enumerate(jpegs):
            offs[idx_j + 1] = offs[idx_j] + len(jb)
        np.savez(
            out_p,
            rgb_blob=np.frombuffer(b"".join(jpegs), dtype=np.uint8),
            rgb_offsets=offs,
            K_orig=np.array([fx, fy, cx, cy], dtype=np.float32),
            orig_hw=np.array([Hh, Ww], dtype=np.int32),
            poses_R=Rs.astype(np.float32),
            poses_t=ps.astype(np.float32),
            causal_fwd=causal_fwd,
            valid_heading=valid_heading,
            eval_rows=eval_rows,
            gt_bev_static=gt_bev_static,
            gt_bev_dynamic=gt_bev_dynamic,
            depth_s8=depth_s8,
        )
        clips_out.append(cname)
    return clips_out


def stage_c_tartanair(n_workers: int) -> Path:
    drive_dir = DRIVE_DATA / "tartanair_widefov_p13_v0"
    local_dir = LOCAL_P13 / "tartanair_widefov"
    local_dir.mkdir(parents=True, exist_ok=True)
    if (drive_dir / "manifest_sha256.json").exists():
        if len(list(local_dir.glob("*.npz"))) < 20:
            man = json.loads((drive_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                with tarfile.open(drive_dir / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        print(f"[Stage C] TartanAir wide-FOV already ready ({len(list(local_dir.glob('*.npz')))} clips)", flush=True)
        return local_dir

    t0 = time.time()
    envs = [
        ("japanesealley", "Easy"),
        ("office", "Easy"),
        ("office2", "Easy"),
        ("oldtown", "Easy"),
        ("seasidetown", "Easy"),
        ("abandonedfactory_night", "Easy"),
        ("hospital", "Easy"),
        ("soulcity", "Easy"),
    ]
    tmp_root = Path("/content/tmp_tartan")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for env, diff in envs:
        te = time.time()
        env_work = tmp_root / f"{env}_{diff}"
        if env_work.exists():
            shutil.rmtree(env_work, ignore_errors=True)
        env_work.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env}/{diff}"
        img_zip = env_work / "image_left.zip"
        dep_zip = env_work / "depth_left.zip"
        print(f"  [TartanAir] downloading {env}/{diff} (image_left + depth_left in parallel)...", flush=True)

        def _dl(pair):
            u, dst = pair
            subprocess.check_call(["curl", "-sSL", "--retry", "3", "-o", str(dst), u])

        try:
            with ThreadPoolExecutor(max_workers=2) as dl_ex:
                list(dl_ex.map(_dl, [(f"{base_url}/image_left.zip", img_zip), (f"{base_url}/depth_left.zip", dep_zip)]))
            with ThreadPoolExecutor(max_workers=2) as uz_ex:
                def _uz(zp):
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(env_work)
                    zp.unlink(missing_ok=True)
                list(uz_ex.map(_uz, [img_zip, dep_zip]))

            trajs = sorted(p.name for p in (env_work / env / diff).glob("P*") if p.is_dir())
            n_before = len(list(local_dir.glob("*.npz")))
            with ProcessPoolExecutor(max_workers=min(n_workers, max(1, len(trajs))), mp_context=mp.get_context("spawn")) as pool:
                futs = [pool.submit(_pack_tartan_traj_worker, (env, diff, tr, str(env_work), str(local_dir))) for tr in trajs]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"    [warn] {env}/{tr} failed: {e}", flush=True)
            n_after = len(list(local_dir.glob("*.npz")))
            print(f"  [TartanAir] {env}/{diff}: +{n_after - n_before} clips (total={n_after}) in {time.time()-te:.1f}s", flush=True)
        except Exception as e:
            print(f"  [warn] TartanAir {env}/{diff} failed: {e}", flush=True)
        finally:
            shutil.rmtree(env_work, ignore_errors=True)

    stems = sorted(p.stem for p in local_dir.glob("*.npz"))
    if stems:
        write_incremental_shards(local_dir, stems, drive_dir, "tartanair_widefov_p13_v0", shard_size=30)
    print(f"[Stage C] DONE TartanAir wide-FOV: {len(stems)} clips in {time.time()-t0:.1f}s -> {drive_dir}", flush=True)
    return local_dir


def _pack_synth_one(args: tuple[str, str]) -> str | None:
    cv2.setNumThreads(1)
    sc, out_dir = args
    dst = Path(out_dir) / f"{sc}.npz"
    if dst.exists() and dst.stat().st_size > 5000:
        return sc
    try:
        with ThreadPoolExecutor(max_workers=8) as dl_pool:
            raw_bytes, _meta = process_session(sc, "train", dl_pool)
        dst.write_bytes(raw_bytes)
        return sc if dst.stat().st_size > 5000 else None
    except Exception:
        return None


def stage_d_sanpo_synth_450(n_workers: int, target_n: int = 450) -> Path:
    drive_dir = DRIVE_DATA / "sanpo_synth_p13_450_v0"
    local_dir = LOCAL_P13 / "sanpo_synth_450"
    local_dir.mkdir(parents=True, exist_ok=True)
    if (drive_dir / "manifest_sha256.json").exists():
        if len(list(local_dir.glob("*.npz"))) < int(target_n * 0.85):
            man = json.loads((drive_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                with tarfile.open(drive_dir / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        print(f"[Stage D] SANPO-Synth +450 already ready ({len(list(local_dir.glob('*.npz')))} scenes)", flush=True)
        return local_dir

    t0 = time.time()
    train_ids = _fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-synthetic/splits/train_session_ids.txt").split()
    test_ids = set(_fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-synthetic/splits/test_session_ids.txt").split())

    used = set()
    for ldir in ["/content/sanpo_287/scenes", "/content/sanpo_p2/extra_scenes"]:
        lp = Path(ldir)
        if lp.exists():
            for p in lp.glob("*.npz"):
                used.add(p.stem)
    for dsub in ["sanpo_synth_287_v0", "sanpo_synth_extra450_v0"]:
        mp_d = DRIVE_DATA / dsub / "manifest_sha256.json"
        if mp_d.exists():
            mj = json.loads(mp_d.read_text())
            for sh in mj.get("shards", []):
                for sc_n in sh.get("scenes", []):
                    used.add(str(sc_n).replace(".npz", ""))
    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    used.update(m287["splits"].keys())

    avail = [s for s in train_ids if s not in used and s not in test_ids]
    rng = np.random.default_rng(20261121)
    perm = rng.permutation(len(avail))
    todo = [avail[i] for i in perm[: min(len(avail), target_n + 60)]]
    print(f"[Stage D] Packing +{target_n} unused SANPO-Synthetic train sessions from {len(avail)} available ({n_workers} workers)...", flush=True)

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_pack_synth_one, (sc, str(local_dir))) for sc in todo]
        for fut in as_completed(futs):
            if fut.result() is not None:
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 50 == 0 or ok >= target_n:
                print(f"  [D SANPO-Synth] ok={ok}/{target_n} fail={fail} ({time.time()-t0:.1f}s)", flush=True)
            if ok >= target_n:
                break

    stems = sorted(p.stem for p in local_dir.glob("*.npz"))[:target_n]
    if stems:
        write_incremental_shards(local_dir, stems, drive_dir, "sanpo_synth_p13_450_v0", shard_size=30)
    print(f"[Stage D] DONE SANPO-Synth +450: {len(stems)} scenes in {time.time()-t0:.1f}s -> {drive_dir}", flush=True)
    return local_dir


def main():
    t_all = time.time()
    ncpu = os.cpu_count() or 8
    n_workers = min(28, max(8, ncpu - 2))
    print(f"=== [Phase 13 Turbo Mega-Data Factory] ncpu={ncpu}, workers={n_workers} ===", flush=True)

    p_a1 = stage_a1_sanpo_real_remaining(n_workers)
    p_a2 = stage_a2_autoseg_head100()
    p_b = stage_b_arkit_500(n_workers, target_n=500)
    p_c = stage_c_tartanair(n_workers)
    p_d = stage_d_sanpo_synth_450(n_workers, target_n=450)

    summary = {
        "sanpo_real_p13_extra": len(list(p_a1.glob("*.npz"))),
        "sanpo_real_head100_autoseg": len(list(p_a2.glob("*.npz"))),
        "arkit_indoor_p13_500": len(list(p_b.glob("*.npz"))),
        "tartanair_widefov_p13": len(list(p_c.glob("*.npz"))),
        "sanpo_synth_p13_450": len(list(p_d.glob("*.npz"))),
        "wall_sec": round(time.time() - t_all, 1),
    }
    print(f"\n=== [Phase 13 Mega-Data Factory ALL DONE] {summary} ===", flush=True)
    (DRIVE_DATA / "phase13_megapack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
