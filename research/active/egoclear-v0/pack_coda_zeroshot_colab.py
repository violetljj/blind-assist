"""Phase 9: Zero-Download-Overhead UT Austin CODa Zero-Shot External Benchmark Packer (ZIP64 HTTP Range).

Streams ONLY selected 5 Hz contiguous walking clips (cam0 rectified RGB + OS1-128 motion-compensated LiDAR
+ dense_global SLAM poses + calibrations + overlapping 3D bounding boxes) directly from UT Austin TACC
via 16-thread ZIP64 HTTP Range requests (~4-5 GB transferred instead of 350+ GB full zips!).

Covers 6 diverse unseen campus sequences (Sunny, Cloudy, Sunset->Dark, Heavy-Traffic Rainy):
  seq 0 (GDC-F Cloudy), seq 3 (GDC-B Sunny), seq 5 (GDC-F Sunset->Dark),
  seq 6 (WCP-B Sunny), seq 10 (WCP-F Cloudy), seq 13 (WCP-F Heavy-Traffic Cloudy/Rainy).

Builds exact walk-alert-r0 v2 drift-free +/-0.8s local ground-plane + multi-scan static-persistence GT
with dynamic-object exclusion (3D bbox + temporal persistence) -> outputs EgoClear .npz schema + Drive .tar!
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import tarfile
import time
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from pack_sanpo_colab import CELL, FWD0, H as H_EVAL, HGT, MIN_VOX_V2, VOX, X_HALF, corridor_d, voxelize_bev

TACC_BASE = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences"
DRIVE_OUT = Path("/content/drive/MyDrive/BlindAssist/datasets/coda_zeroshot_v0")
LOCAL_OUT = Path("/content/coda_zeroshot_v0")

DYNAMIC_CLASSES = {
    "Car", "Pedestrian", "Bike", "Motorcycle", "Golf Cart", "Truck", "Scooter",
    "Dog", "Skateboard", "Wheelchair", "Segway", "Delivery Robot", "Service Vehicle",
    "Bus", "Pickup Truck", "Person", "Cyclist",
}

SEQ_META = {
    0: "GDC_F_Cloudy_Light",
    3: "GDC_B_Sunny_Med",
    5: "GDC_F_SunsetDark_Light",
    6: "WCP_B_Sunny_Med",
    10: "WCP_F_Cloudy_Med",
    13: "WCP_F_Rainy_Heavy",
}


class RemoteZip64:
    """Zero-full-download ZIP64 reader using HTTP Range requests."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            self.total_size = int(resp.headers["Content-Length"])
        self.entries: dict[str, tuple[int, int, int, int]] = {}  # name -> (offset, comp_sz, uncomp_sz, method)
        self._load_central_directory()

    def _range_get(self, start: int, end: int, retries: int = 4) -> bytes:
        end = min(end, self.total_size - 1)
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Range GET {self.url} [{start}-{end}] failed: {last_err}")

    def _load_central_directory(self) -> None:
        # Read tail to locate EOCD + ZIP64 EOCD Locator / Record
        tail_sz = min(self.total_size, 131072)
        tail = self._range_get(self.total_size - tail_sz, self.total_size - 1)
        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise RuntimeError(f"No EOCD in {self.url}")
        cd_sz, cd_off = struct.unpack("<II", tail[eocd_pos + 12 : eocd_pos + 20])
        if cd_off == 0xFFFFFFFF or cd_sz == 0xFFFFFFFF:
            loc_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
            if loc_pos < 0:
                raise RuntimeError(f"ZIP64 without locator in {self.url}")
            z64_eocd_off = struct.unpack("<Q", tail[loc_pos + 8 : loc_pos + 16])[0]
            z64_rec = self._range_get(z64_eocd_off, z64_eocd_off + 64)
            assert z64_rec[:4] == b"PK\x06\x06"
            cd_sz, cd_off = struct.unpack("<QQ", z64_rec[40:56])

        cd_bytes = self._range_get(cd_off, cd_off + cd_sz - 1)
        pos = 0
        n_bytes = len(cd_bytes)
        while pos + 46 <= n_bytes:
            if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
                break
            method = struct.unpack("<H", cd_bytes[pos + 10 : pos + 12])[0]
            csz, usz = struct.unpack("<II", cd_bytes[pos + 20 : pos + 28])
            fn_len, ex_len, cm_len = struct.unpack("<HHH", cd_bytes[pos + 28 : pos + 34])
            lh_off = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
            fn = cd_bytes[pos + 46 : pos + 46 + fn_len].decode("utf-8", "ignore")
            extra = cd_bytes[pos + 46 + fn_len : pos + 46 + fn_len + ex_len]
            pos += 46 + fn_len + ex_len + cm_len

            if csz == 0xFFFFFFFF or usz == 0xFFFFFFFF or lh_off == 0xFFFFFFFF:
                ep = 0
                while ep + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[ep : ep + 4])
                    ep += 4
                    if hid == 0x0001:
                        cur = ep
                        if usz == 0xFFFFFFFF:
                            usz = struct.unpack("<Q", extra[cur : cur + 8])[0]; cur += 8
                        if csz == 0xFFFFFFFF:
                            csz = struct.unpack("<Q", extra[cur : cur + 8])[0]; cur += 8
                        if lh_off == 0xFFFFFFFF:
                            lh_off = struct.unpack("<Q", extra[cur : cur + 8])[0]; cur += 8
                        break
                    ep += hsz
            self.entries[fn] = (lh_off, csz, usz, method)

    def read(self, name: str) -> bytes:
        lh_off, csz, _usz, method = self.entries[name]
        # Fetch local header + payload in one single Range GET (+512B margin for local extra fields)
        buf = self._range_get(lh_off, lh_off + 30 + len(name) + 512 + csz)
        if buf[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Bad local header for {name}")
        lfn, lex = struct.unpack("<HH", buf[26:30])
        payload = buf[30 + lfn + lex : 30 + lfn + lex + csz]
        if method == 0:
            return payload
        if method == 8:
            return zlib.decompress(payload, -15)
        raise RuntimeError(f"Unsupported compression method {method} for {name}")


def parse_pose_file(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse CODa poses/dense_global/{seq}.txt -> (ts, R_w_os1 (N,3,3), t_w_os1 (N,3))."""
    rows = []
    for ln in text.splitlines():
        parts = ln.strip().split()
        if len(parts) >= 8:
            rows.append([float(x) for x in parts[:8]])
    arr = np.array(rows, dtype=np.float64)
    ts = arr[:, 0]
    t_w = arr[:, 1:4]
    # CODa quaternion order: qw, qx, qy, qz -> scipy expects [qx, qy, qz, qw]
    q_xyzw = np.column_stack([arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 4]])
    R_w = Rotation.from_quat(q_xyzw).as_matrix()
    return ts, R_w, t_w


def parse_cam0_calib(intr_yaml: str, ext_yaml: str) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Return (K_rect [fx,fy,cx,cy], (H,W), T_os1_cam0rect 4x4)."""
    ic = yaml.safe_load(intr_yaml)
    ec = yaml.safe_load(ext_yaml)
    W, H = int(ic.get("image_width", 1224)), int(ic.get("image_height", 1024))
    if "projection_matrix" in ic:
        P = np.array(ic["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        K_rect = np.array([P[0, 0], P[1, 1], P[0, 2], P[1, 2]], dtype=np.float64)
    else:
        Km = np.array(ic["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        K_rect = np.array([Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]], dtype=np.float64)
    if "rectification_matrix" in ic:
        R_rect = np.array(ic["rectification_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    else:
        R_rect = np.eye(3, dtype=np.float64)

    T_cam0_os1 = np.array(ec["extrinsic_matrix"]["data"], dtype=np.float64).reshape(4, 4)
    T_cam0rect_os1 = np.eye(4, dtype=np.float64)
    T_cam0rect_os1[:3, :3] = R_rect @ T_cam0_os1[:3, :3]
    T_cam0rect_os1[:3, 3] = R_rect @ T_cam0_os1[:3, 3]
    T_os1_cam0rect = np.linalg.inv(T_cam0rect_os1)
    return K_rect, (H, W), T_os1_cam0rect


def filter_dynamic_points_in_bboxes(pts_os1: np.ndarray, bbox_json: dict | None) -> np.ndarray:
    """Return boolean mask of points inside dynamic 3D bounding boxes (in os1 frame)."""
    dyn_mask = np.zeros(len(pts_os1), dtype=bool)
    if not bbox_json or "3dbbox" not in bbox_json:
        return dyn_mask
    for b in bbox_json["3dbbox"]:
        if b.get("classId") not in DYNAMIC_CLASSES:
            continue
        c = np.array([b["cX"], b["cY"], b["cZ"]], dtype=np.float64)
        l, w, h = float(b["l"]) + 0.20, float(b["w"]) + 0.20, float(b["h"]) + 0.20
        d = pts_os1 - c
        r2 = (d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
        rmax = 0.5 * np.sqrt(l * l + w * w + h * h)
        cand = np.nonzero(r2 <= rmax * rmax)[0]
        if len(cand) == 0:
            continue
        if "r" in b:
            R_b = Rotation.from_euler("xyz", [b["r"], b["p"], b["y"]], degrees=False).as_matrix()
        else:
            R_b = Rotation.from_quat([b["qx"], b["qy"], b["qz"], b["qw"]]).as_matrix()
        loc = d[cand] @ R_b  # R_b^T @ d
        inside = (np.abs(loc[:, 0]) <= 0.5 * l) & (np.abs(loc[:, 1]) <= 0.5 * w) & (np.abs(loc[:, 2]) <= 0.5 * h)
        dyn_mask[cand[inside]] = True
    return dyn_mask


def fit_local_ground_plane_os1(pts_os1: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Robust RANSAC ground plane (n · p + d = 0, n_z > 0) in os1 frame on nearby ground points."""
    r_xy = np.hypot(pts_os1[:, 0], pts_os1[:, 1])
    cand = pts_os1[(r_xy >= 1.0) & (r_xy <= 10.0) & (pts_os1[:, 2] <= -0.30) & (pts_os1[:, 2] >= -1.40)]
    if len(cand) < 100:
        return np.array([0.0, 0.0, 1.0]), 0.75, 0.05
    if len(cand) > 2500:
        cand = cand[rng.choice(len(cand), 2500, replace=False)]
    best_n, best_d, best_cnt = np.array([0.0, 0.0, 1.0]), float(-np.median(cand[:, 2])), -1
    for _ in range(35):
        idx = rng.choice(len(cand), 3, replace=False)
        p0, p1, p2 = cand[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(n))
        if nn < 1e-4:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < 0.93:
            continue
        d = -float(np.dot(n, p0))
        cnt = int(np.sum(np.abs(cand @ n + d) <= 0.06))
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, n, d
    inl = cand[np.abs(cand @ best_n + best_d) <= 0.07]
    if len(inl) >= 50:
        mu = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - mu, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        if n[2] >= 0.92:
            best_n = n
            best_d = -float(np.dot(best_n, mu))
    med_res = float(np.median(np.abs(inl @ best_n + best_d))) if len(inl) else 0.05
    return best_n, best_d, med_res


def select_walking_clips(
    t_w: np.ndarray,
    R_w: np.ndarray,
    avail_frames: set[int],
    bbox_frames: set[int],
    n_clips: int = 4,
    clip_frames_5hz: int = 120,
    stride: int = 2,
) -> list[list[int]]:
    """Select n_clips diverse non-overlapping 24s (120-frame @ 5Hz) active-walking segments."""
    span = clip_frames_5hz * stride  # 240 raw 10Hz frames
    n_tot = len(t_w)
    candidates = []
    for st in range(10, n_tot - span - 10, 40):
        frs = list(range(st, st + span, stride))
        if not all(f in avail_frames for f in frs):
            continue
        pts = t_w[frs]
        diffs = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        path_m = float(diffs.sum())
        move_frac = float((diffs >= 0.06).mean())  # >= 0.30 m/s at 5 Hz
        # Prefer clips with sustained movement (>= 11 m over 24s, >= 60% moving) + slight bonus for bbox overlap
        if path_m < 11.0 or move_frac < 0.55:
            continue
        n_bb = sum(1 for f in range(st, st + span) if f in bbox_frames)
        # Also reward turning / curved segments (more real obstacle encounters!)
        disp = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
        curv = path_m / max(disp, 1.0)
        score = path_m + 3.0 * min(curv, 2.5) + 0.5 * n_bb
        candidates.append((score, st, frs, path_m, n_bb))

    candidates.sort(key=lambda x: -x[0])
    chosen = []
    used_starts = []
    # Enforce spacing >= span * 1.5 so clips are spatially diverse across the campus route!
    for sc, st, frs, path_m, n_bb in candidates:
        if all(abs(st - u) >= int(span * 1.5) for u in used_starts):
            chosen.append((st, frs, path_m, n_bb))
            used_starts.append(st)
            if len(chosen) >= n_clips:
                break
    return chosen


def pack_one_clip(
    rz: RemoteZip64,
    seq: int,
    clip_idx: int,
    frames: list[int],
    R_w_os1_all: np.ndarray,
    t_w_os1_all: np.ndarray,
    K_rect: np.ndarray,
    orig_hw: tuple[int, int],
    T_os1_cam0rect: np.ndarray,
    out_dir: Path,
) -> dict | None:
    t0 = time.time()
    n = len(frames)

    def _fetch_frame(f: int):
        img_name = f"2d_rect/cam0/{seq}/2d_rect_cam0_{seq}_{f}.png"
        pc_name = f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_{f}.bin"
        bb_name = f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{f}.json"
        img_b = rz.read(img_name)
        pc_b = rz.read(pc_name)
        bb_j = json.loads(rz.read(bb_name).decode("utf-8")) if bb_name in rz.entries else None
        return f, img_b, pc_b, bb_j

    raw_frames = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(_fetch_frame, f) for f in frames]
        for fut in as_completed(futs):
            f, img_b, pc_b, bb_j = fut.result()
            raw_frames[f] = (img_b, pc_b, bb_j)
    t_dl = time.time() - t0

    rng = np.random.default_rng(1000 * seq + clip_idx)
    Rs = np.empty((n, 3, 3), dtype=np.float64)
    ps = np.empty((n, 3), dtype=np.float64)
    jpegs = []
    P_stat_w_all = []
    P_dyn_w_all = []
    P_ground_w_all = []
    ground_res_list = []
    cam_heights = []

    for i, f in enumerate(frames):
        img_b, pc_b, bb_j = raw_frames[f]
        bgr = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        bgr_rs = cv2.resize(bgr, (RGB_W, RGB_H), interpolation=cv2.INTER_AREA)
        ok_enc, enc = cv2.imencode(".jpg", bgr_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            return None
        jpegs.append(enc.tobytes())

        # Exact camera pose in world frame: T_w_cam0rect = T_w_os1 @ T_os1_cam0rect
        T_w_os1 = np.eye(4, dtype=np.float64)
        T_w_os1[:3, :3] = R_w_os1_all[f]
        T_w_os1[:3, 3] = t_w_os1_all[f]
        T_w_cam = T_w_os1 @ T_os1_cam0rect
        Rs[i] = T_w_cam[:3, :3]
        ps[i] = T_w_cam[:3, 3]

        # Parse OS1-128 motion-compensated LiDAR (131072, 4) float32
        pc = np.frombuffer(pc_b, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        r_xy = np.hypot(pc[:, 0], pc[:, 1])
        # Exclude robot body (r < 0.70m) and far points (r > 9.0m)
        ok_r = np.isfinite(pc).all(axis=1) & (r_xy >= 0.70) & (r_xy <= 9.0)
        pc = pc[ok_r]

        g_n, g_d, g_res = fit_local_ground_plane_os1(pc, rng)
        ground_res_list.append(g_res)
        # Camera height above local ground plane
        p_cam_os1 = T_os1_cam0rect[:3, 3]
        cam_heights.append(float(np.dot(g_n, p_cam_os1) + g_d))

        h_above = pc @ g_n + g_d
        is_gnd = (h_above >= -0.10) & (h_above <= 0.12)
        is_obs = (h_above >= HGT[0]) & (h_above <= HGT[1])

        dyn_bb = filter_dynamic_points_in_bboxes(pc, bb_j)
        gnd_os1 = pc[is_gnd]
        obs_stat_os1 = pc[is_obs & (~dyn_bb)]
        obs_dyn_os1 = pc[is_obs & dyn_bb]

        R_wo, t_wo = R_w_os1_all[f], t_w_os1_all[f]
        # Pre-voxelize to 5cm (VOX) immediately per frame -> cuts 131k raw LiDAR pts down to ~2k unique voxels (50x faster!)
        def _to_world_vox(pts_loc: np.ndarray, vsz: float) -> np.ndarray:
            if len(pts_loc) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            pw = (R_wo @ pts_loc.T).T + t_wo
            vi = np.floor(pw / vsz).astype(np.int64)
            k64 = ((vi[:, 0] + 100000) << 40) | ((vi[:, 1] + 100000) << 20) | (vi[:, 2] + 100000)
            _, u_idx = np.unique(k64, return_index=True)
            return pw[u_idx].astype(np.float32)

        P_ground_w_all.append(_to_world_vox(gnd_os1, 0.15))
        P_stat_w_all.append(_to_world_vox(obs_stat_os1, VOX))
        P_dyn_w_all.append(_to_world_vox(obs_dyn_os1, VOX))

    # Causal walking heading (identical protocol to SANPO/Aria/ARKitScenes)
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
    causal_fwd = np.zeros((n, 3), dtype=np.float64)
    valid_heading = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        cam_f = Rs[i, :2, 2]
        cam_fn = float(np.linalg.norm(cam_f))
        cam_f2 = (cam_f / cam_fn) if cam_fn > 1e-4 else np.array([1.0, 0.0])
        v_xy = None
        for back in range(2, min(6, i + 1)):
            dv = (ps[i] - ps[i - back])[:2]
            if np.linalg.norm(dv) >= 0.18:
                v_xy = dv
                break
        if v_xy is not None:
            f2 = v_xy / np.linalg.norm(v_xy)
            causal_fwd[i] = [f2[0], f2[1], 0.0]
            if float(np.dot(f2, cam_f2)) >= 0.25:
                valid_heading[i] = 1
        else:
            causal_fwd[i] = [cam_f2[0], cam_f2[1], 0.0]

    # Build drift-free +/-0.8s local-window static persistence GT + dynamic separation (100% vectorized NumPy!)
    eval_rows = np.full((n, 4), np.nan, dtype=np.float64)
    gt_bev_static = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    gt_bev_dynamic = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    VOX_PERS = 0.20  # 20cm voxel for multi-scan static persistence

    # Precompute packed int64 20cm persistence keys per frame (zero Python loops!)
    pers_k64_all = []
    for k in range(n):
        Pk = P_stat_w_all[k]
        if len(Pk) > 0:
            vk = np.floor(Pk / VOX_PERS).astype(np.int64)
            pers_k64_all.append(((vk[:, 0] + 100000) << 40) | ((vk[:, 1] + 100000) << 20) | (vk[:, 2] + 100000))
        else:
            pers_k64_all.append(np.zeros(0, dtype=np.int64))

    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        g_list = [P_ground_w_all[k] for k in range(lo, hi) if len(P_ground_w_all[k]) > 0]
        if not g_list:
            continue
        Gw = np.concatenate(g_list, axis=0)
        if len(Gw) < 40:
            continue
        g_ij = np.floor(Gw[:, :2] / CELL).astype(np.int64)
        g_k64 = ((g_ij[:, 0] + 100000) << 24) | (g_ij[:, 1] + 100000)
        ug_k64, inv = np.unique(g_k64, return_inverse=True)
        order = np.argsort(inv, kind="mergesort")
        z_s = Gw[order, 2]
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        gz = np.array([np.median(g) for g in np.split(z_s, cuts)], dtype=np.float64)
        keys = np.column_stack([(ug_k64 >> 24) - 100000, (ug_k64 & ((1 << 24) - 1)) - 100000])
        gtree = cKDTree((keys + 0.5) * CELL)
        cell_k64_set = set(ug_k64.tolist())
        z_g_med = float(np.median(Gw[:, 2]))

        def height_loc(Q: np.ndarray) -> np.ndarray:
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.5)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            miss = ~okg
            if np.any(miss):
                h[miss] = Q[miss, 2] - z_g_med
            return h

        # Vectorized multi-scan static persistence across [lo..hi): span >= 2 frames (0.4s)
        win_k_arrs, win_f_arrs = [], []
        for k in range(lo, hi):
            kk = pers_k64_all[k]
            if len(kk) > 0:
                uk = np.unique(kk)
                win_k_arrs.append(uk)
                win_f_arrs.append(np.full(len(uk), k, dtype=np.int32))
        if win_k_arrs:
            all_k = np.concatenate(win_k_arrs)
            all_f = np.concatenate(win_f_arrs)
            u_pers, inv_p = np.unique(all_k, return_inverse=True)
            f_min = np.full(len(u_pers), 10**9, dtype=np.int32)
            f_max = np.full(len(u_pers), -1, dtype=np.int32)
            np.minimum.at(f_min, inv_p, all_f)
            np.maximum.at(f_max, inv_p, all_f)
            pers_valid = u_pers[(f_max - f_min) >= 2]
        else:
            pers_valid = np.zeros(0, dtype=np.int64)

        stat_pts_list, trans_dyn_list = [], []
        for k in range(lo, hi):
            Pk = P_stat_w_all[k]
            if len(Pk) > 0:
                is_pers = np.isin(pers_k64_all[k], pers_valid) if len(pers_valid) > 0 else np.zeros(len(Pk), dtype=bool)
                if np.any(is_pers):
                    stat_pts_list.append(Pk[is_pers])
                if k == i and np.any(~is_pers):
                    trans_dyn_list.append(Pk[~is_pers])
            if k == i and len(P_dyn_w_all[k]) > 0:
                trans_dyn_list.append(P_dyn_w_all[k])

        fwd = causal_fwd[i]
        right = np.array([fwd[1], -fwd[0], 0.0])

        if stat_pts_list:
            Sw = np.concatenate(stat_pts_list, axis=0)
            vi = np.floor(Sw / VOX).astype(np.int64)
            sk64 = ((vi[:, 0] + 100000) << 40) | ((vi[:, 1] + 100000) << 20) | (vi[:, 2] + 100000)
            _, u_s = np.unique(sk64, return_index=True)
            vc = (vi[u_s] + 0.5) * VOX
            vc_h = height_loc(vc)
            ok_h = (vc_h >= HGT[0]) & (vc_h <= HGT[1])
            vc, vc_h = vc[ok_h], vc_h[ok_h]
            if len(vc) > 0:
                rel_v = vc - ps[i]
                gt_bev_static[i] = voxelize_bev(np.column_stack([rel_v @ right, rel_v @ fwd, vc_h]), min_count=2)
                c_ij = np.floor(vc[:, :2] / CELL).astype(np.int64)
                cell_k64_set.update((((c_ij[:, 0] + 100000) << 24) | (c_ij[:, 1] + 100000)).tolist())
        else:
            vc = np.zeros((0, 3), dtype=np.float64)

        dyn_flag = 0.0
        if trans_dyn_list:
            Dw = np.concatenate(trans_dyn_list, axis=0)
            dvi = np.floor(Dw / VOX).astype(np.int64)
            dk64 = ((dvi[:, 0] + 100000) << 40) | ((dvi[:, 1] + 100000) << 20) | (dvi[:, 2] + 100000)
            _, u_d = np.unique(dk64, return_index=True)
            dvc = (dvi[u_d] + 0.5) * VOX
            dvc_h = height_loc(dvc)
            ok_dh = (dvc_h >= HGT[0]) & (dvc_h <= HGT[1])
            dvc, dvc_h = dvc[ok_dh], dvc_h[ok_dh]
            if len(dvc) > 0:
                rel_d = dvc - ps[i]
                xd, zd = rel_d @ right, rel_d @ fwd
                gt_bev_dynamic[i] = voxelize_bev(np.column_stack([xd, zd, dvc_h]), min_count=2)
                inc_d = (np.abs(xd) <= X_HALF) & (zd >= FWD0) & (zd <= H_EVAL)
                if int(inc_d.sum()) >= MIN_VOX_V2:
                    dyn_flag = 1.0

        if not valid_heading[i]:
            continue

        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
        if np.linalg.norm(vt) < 0.12:
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
        if len(samp):
            s_ij = np.floor(samp / CELL).astype(np.int64)
            s_k64 = ((s_ij[:, 0] + 100000) << 24) | (s_ij[:, 1] + 100000)
            cover = float(np.mean([int(k) in cell_k64_set for k in s_k64]))
        else:
            cover = 1.0
        eval_rows[i] = [D, cover, dyn_flag, seg_len[i]]

    n_ok = int(np.isfinite(eval_rows[:, 1]).sum())
    if n_ok < 20:
        return None

    blob = bytearray()
    offs = np.zeros(n + 1, dtype=np.int64)
    for idx_j, jb in enumerate(jpegs):
        blob.extend(jb)
        offs[idx_j + 1] = len(blob)

    clip_name = f"coda_seq{seq:02d}_{SEQ_META[seq]}_c{clip_idx:02d}"
    out_p = out_dir / f"{clip_name}.npz"
    np.savez(
        out_p,
        rgb_blob=np.frombuffer(bytes(blob), dtype=np.uint8),
        rgb_offsets=offs,
        K_orig=K_rect.astype(np.float32),
        orig_hw=np.array(orig_hw, dtype=np.int32),
        poses_R=Rs.astype(np.float32),
        poses_t=ps.astype(np.float32),
        causal_fwd=causal_fwd.astype(np.float32),
        valid_heading=valid_heading,
        gt_bev_static=gt_bev_static,
        gt_bev_dynamic=gt_bev_dynamic,
        eval_rows=eval_rows.astype(np.float64),
        camera_mount=np.array("coda_husky_flir"),
        has_dyn_gt=np.array(1, dtype=np.uint8),
    )

    n_haz = int(np.sum(np.isfinite(eval_rows[:, 0]) & (eval_rows[:, 0] <= H_EVAL)))
    n_dyn = int(np.sum(np.isinf(eval_rows[:, 0]) & (eval_rows[:, 2] > 0.5)))
    path_m = float(seg_len.sum())
    return {
        "clip": clip_name, "seq": seq, "n_frames": n, "n_ok": n_ok,
        "n_haz": n_haz, "n_dyn_only": n_dyn, "path_m": round(path_m, 1),
        "cam_h_m": round(float(np.median(cam_heights)), 3),
        "gnd_res_cm": round(100.0 * float(np.median(ground_res_list)), 2),
        "sec": round(time.time() - t0, 1),
    }


def archive_to_drive(local_dir: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    npzs = sorted(local_dir.glob("*.npz"))
    tar_local = Path("/content/coda_zeroshot_v0_shard_000.tar")
    with tarfile.open(tar_local, "w") as tf:
        for p in npzs:
            tf.add(p, arcname=p.name)
    h = hashlib.sha256()
    with open(tar_local, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    dst_tar = drive_dir / "shard_000.tar"
    if dst_tar.exists():
        dst_tar.unlink()
    shutil.copy2(tar_local, dst_tar)
    man = {
        "dataset": "coda_zeroshot_v0",
        "source": "UT Austin CODa (CC BY-NC-SA 4.0) via TACC ZIP64 Range",
        "n_clips": len(npzs),
        "shards": [{"file": "shard_000.tar", "bytes": tar_local.stat().st_size, "sha256": h.hexdigest(), "count": len(npzs)}],
    }
    (drive_dir / "manifest_sha256.json").write_text(json.dumps(man, indent=2))
    print(f"[coda-pack] Archived {len(npzs)} clips ({tar_local.stat().st_size/1e6:.1f} MB, sha256={h.hexdigest()[:12]}) to {drive_dir}", flush=True)


def main():
    t_all = time.time()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    if len(list(LOCAL_OUT.glob("*.npz"))) >= 20:
        print(f"[coda-pack] Already have {len(list(LOCAL_OUT.glob('*.npz')))} CODa clips locally, skipping download!", flush=True)
        return

    # Check if already cached on Drive!
    man_p = DRIVE_OUT / "manifest_sha256.json"
    if man_p.exists() and (DRIVE_OUT / "shard_000.tar").exists():
        print("[coda-pack] Restoring cached coda_zeroshot_v0 from Drive...", flush=True)
        with tarfile.open(DRIVE_OUT / "shard_000.tar", "r") as tf:
            tf.extractall(LOCAL_OUT)
        print(f"[coda-pack] Restored {len(list(LOCAL_OUT.glob('*.npz')))} clips in {time.time()-t_all:.1f}s", flush=True)
        return

    stats = []
    for seq, tag in SEQ_META.items():
        ts = time.time()
        url = f"{TACC_BASE}/{seq}.zip"
        print(f"[coda-pack] Indexing seq {seq} ({tag}) from TACC...", flush=True)
        rz = RemoteZip64(url)
        pose_key = f"poses/dense_global/{seq}.txt" if f"poses/dense_global/{seq}.txt" in rz.entries else f"poses/dense/{seq}.txt"
        _ts, R_w, t_w = parse_pose_file(rz.read(pose_key).decode("utf-8"))
        K_rect, orig_hw, T_os1_cam0rect = parse_cam0_calib(
            rz.read(f"calibrations/{seq}/calib_cam0_intrinsics.yaml").decode("utf-8"),
            rz.read(f"calibrations/{seq}/calib_os1_to_cam0.yaml").decode("utf-8"),
        )
        # Find frames that have BOTH cam0 RGB and os1 motion-compensated LiDAR
        avail_cam, avail_pc, bbox_frs = set(), set(), set()
        pref_c = f"2d_rect/cam0/{seq}/2d_rect_cam0_{seq}_"
        pref_p = f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_"
        pref_b = f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_"
        for name in rz.entries:
            if name.startswith(pref_c) and name.endswith(".png"):
                avail_cam.add(int(name[len(pref_c):-4]))
            elif name.startswith(pref_p) and name.endswith(".bin"):
                avail_pc.add(int(name[len(pref_p):-4]))
            elif name.startswith(pref_b) and name.endswith(".json"):
                bbox_frs.add(int(name[len(pref_b):-5]))
        avail = avail_cam & avail_pc
        clips = select_walking_clips(t_w, R_w, avail, bbox_frs, n_clips=4, clip_frames_5hz=120, stride=2)
        print(f"  seq {seq}: poses={len(t_w)}, avail={len(avail)}, bbox={len(bbox_frs)}, selected={len(clips)} clips ({time.time()-ts:.1f}s)", flush=True)
        for ci, (st, frs, pm, n_bb) in enumerate(clips):
            info = pack_one_clip(rz, seq, ci, frs, R_w, t_w, K_rect, orig_hw, T_os1_cam0rect, LOCAL_OUT)
            if info:
                stats.append(info)
                print(
                    f"    packed {info['clip']}: ok={info['n_ok']}/{info['n_frames']} haz={info['n_haz']} "
                    f"dyn={info['n_dyn_only']} path={info['path_m']}m camH={info['cam_h_m']}m gRes={info['gnd_res_cm']}cm ({info['sec']}s)",
                    flush=True,
                )

    tot_fr = sum(s["n_frames"] for s in stats)
    tot_haz = sum(s["n_haz"] for s in stats)
    tot_dyn = sum(s["n_dyn_only"] for s in stats)
    tot_m = sum(s["path_m"] for s in stats)
    print(
        f"[coda-pack] DONE {len(stats)} clips, {tot_fr} frames, {tot_m/1000:.2f} km walked, "
        f"haz_frames={tot_haz}, dyn_only_frames={tot_dyn} in {time.time()-t_all:.1f}s!",
        flush=True,
    )
    archive_to_drive(LOCAL_OUT, DRIVE_OUT)
    (DRIVE_OUT / "pack_summary.json").write_text(json.dumps({"stats": stats, "total_km": round(tot_m / 1000.0, 3)}, indent=2))


if __name__ == "__main__":
    main()
