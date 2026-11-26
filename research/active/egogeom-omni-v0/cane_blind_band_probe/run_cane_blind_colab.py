"""Colab L4 Cane-Blind Mid/Upper-Body Height-Band Probe on Frozen EgoGeom-Omni v4.1.

Strictly zero-training EXPLORE audit across 5 real/external domains:
  1. Meta Aria Gen 2 (19 clips; 6 voxel height bands x 2 GT stages: centered_full & strict_past)
  2. ARKitScenes val30 Day (30 clips; matched gt_bev_static height bands)
  3. ARKitScenes val30 Dark 0.18x (30 clips; matched gt_bev_static height bands)
  4. UT Austin CODa Campus Rover (24 clips, ~1.22m mount; matched gt_bev_static height bands)
  5. SANPO-Real Zeroshot 40 (38 clips, split by all / camera_head / camera_chest)

Evaluates 8 strictly causal Stage-2 readouts on identical frozen v4.1 Pure-RGB predictions:
  - BEV v2.3 family: bev_full_ground, bev_midup_ground, bev_head_anchored_85,
    bev_head_anchored_85_pg, bev_head_anchored_70_pg
  - M6 4-head family: m6_full_ground, m6_midup_ground, m6_head_anchored_85_pg
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import cv2
import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))


def ensure_deps() -> None:
    import timm

    if not timm.is_model("convnext_small.dinov3_lvd1689m"):
        print("[setup] Upgrading timm for DINOv3 ConvNeXt-Small...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "timm"], check=True)
    for mod, pkg in [("numba", "numba"), ("sklearn", "scikit-learn")]:
        try:
            __import__(mod)
        except ImportError:
            print(f"[setup] Installing {pkg}...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


ensure_deps()

from numba import njit  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import (  # noqa: E402
    EgoGeomOmniV41Net,
    EgoGeomV41Config,
    batch_host_focal_from_K,
)
from egogeom_temporal_bev import (  # noqa: E402
    EgoGeomTemporalBEV,
    TemporalBEVConfig,
    _horizontal_fwd_right,
)
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3, TemporalBEVV3Config  # noqa: E402
from eval_closed_loop import make_episodes, score_episode  # noqa: E402

WORK = Path("/content/work")
INPUTS = Path("/content/inputs")
DATA_DIR = WORK / "data"
CKPT_DIR = WORK / "ckpt"
TAR_DIR = WORK / "tars"
OUT_DIR = WORK / "out"

H_IN, W_IN = 288, 384
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BW_SPEEDS_PROXY = {f"p{i}": np.linspace(0.78 + 0.04 * i, 1.15 + 0.04 * i, 10).tolist() for i in range(6)}

Q_GRID = (0.5, 0.65, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
FA_BUDGET = 5.0
LATE, AVOID, UNAVOID, FALSE, FFRAME, KM, CLEAR_WARN, ALERT, VALID, ON_CLEAR, CLEAR = range(11)

BEV_READOUT_ARMS = (
    "bev_full_ground",
    "bev_midup_ground",
    "bev_head_anchored_85",
    "bev_head_anchored_85_pg",
    "bev_head_anchored_70_pg",
)
M6_READOUT_ARMS = (
    "m6_full_ground",
    "m6_midup_ground",
    "m6_head_anchored_85_pg",
)
ALL_ARMS = BEV_READOUT_ARMS + M6_READOUT_ARMS


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def run_rclone_copy(remote: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", remote, str(local_dir), "--transfers", "16"], check=True)


def extract_tar(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / f".done_{tar_path.name}"
    if marker.exists():
        return
    print(f"[stage] Extracting {tar_path.name} -> {dst_dir}", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst_dir)
    marker.write_text("ok\n", encoding="utf-8")


def stage_assets() -> Path:
    import shutil
    t0 = time.time()
    for d in (DATA_DIR, CKPT_DIR, TAR_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if shutil.which("rclone") is None:
        print("[stage] Installing rclone...", flush=True)
        rc = subprocess.run("apt-get update -qq && apt-get install -y -qq rclone", shell=True)
        if rc.returncode != 0 or shutil.which("rclone") is None:
            subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=True)
        print("[stage] rclone ready:", shutil.which("rclone"), flush=True)

    ckpt_path = CKPT_DIR / "egogeom_omni_v41_pure_rgb_best.pt"
    if not ckpt_path.exists():
        print("[stage] Pulling v4.1 pure-RGB checkpoint from Drive...", flush=True)
        run_rclone_copy("gdrive:BlindAssist/runs/egogeom_omni_v41/egogeom_omni_v41_pure_rgb_best.pt", CKPT_DIR)

    extract_tar(INPUTS / "aria_g2_20clips.tar", DATA_DIR / "aria_gen2")

    arkit_tar = TAR_DIR / "arkit/arkit_indoor_val30_v0_shard00.tar"
    if not (DATA_DIR / "arkit_val30" / f".done_{arkit_tar.name}").exists():
        run_rclone_copy(
            "gdrive:BlindAssist/datasets/arkit_indoor_val30_v0/arkit_indoor_val30_v0_shard00.tar",
            TAR_DIR / "arkit",
        )
        extract_tar(arkit_tar, DATA_DIR / "arkit_val30")

    coda_tar = TAR_DIR / "coda/shard_000.tar"
    if not (DATA_DIR / "coda_campus" / f".done_{coda_tar.name}").exists():
        run_rclone_copy("gdrive:BlindAssist/datasets/coda_zeroshot_v0/shard_000.tar", TAR_DIR / "coda")
        extract_tar(coda_tar, DATA_DIR / "coda_campus")

    for sh in ("shard_000.tar", "shard_001.tar"):
        sanpo_tar = TAR_DIR / "sanpo" / sh
        if not (DATA_DIR / "sanpo_real_zs40" / f".done_{sh}").exists():
            run_rclone_copy(f"gdrive:BlindAssist/datasets/sanpo_real_zeroshot_40_v0/{sh}", TAR_DIR / "sanpo")
            extract_tar(sanpo_tar, DATA_DIR / "sanpo_real_zs40")

    for name, d in [
        ("aria_gen2", DATA_DIR / "aria_gen2"),
        ("arkit_val30", DATA_DIR / "arkit_val30"),
        ("coda_campus", DATA_DIR / "coda_campus"),
        ("sanpo_real_zs40", DATA_DIR / "sanpo_real_zs40"),
    ]:
        print(f"[stage] {name}: {len(list(d.rglob('*.npz')))} npz files", flush=True)

    print(f"[stage] All assets ready in {time.time() - t0:.1f}s", flush=True)
    return ckpt_path


# ---------------------------------------------------------------------------
# Decoding & Heading Parity
# ---------------------------------------------------------------------------

def decode_isotropic_frame(jpg_bytes: bytes, orig_hw: np.ndarray, K_orig: np.ndarray, dark_scale: float = 1.0):
    bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    target_ar = W_IN / float(H_IN)
    orig_ar = ow / float(oh)

    if orig_ar >= target_ar:
        ih = H_IN
        iw = max(W_IN, int(round(H_IN * orig_ar)))
    else:
        iw = W_IN
        ih = max(H_IN, int(round(W_IN / orig_ar)))
    rgb_iso = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA if iw < rgb.shape[1] else cv2.INTER_LINEAR)
    sx, sy = iw / float(ow), ih / float(oh)
    fx, fy, cx, cy = float(K_orig[0]) * sx, float(K_orig[1]) * sy, float(K_orig[2]) * sx, float(K_orig[3]) * sy

    y0 = max(0, (ih - H_IN) // 2)
    x0 = max(0, (iw - W_IN) // 2)
    rgb_crop = rgb_iso[y0 : y0 + H_IN, x0 : x0 + W_IN]
    if rgb_crop.shape[:2] != (H_IN, W_IN):
        rgb_crop = cv2.resize(rgb_crop, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
    K_out = np.array([[fx, 0.0, cx - x0], [0.0, fy, cy - y0], [0.0, 0.0, 1.0]], dtype=np.float32)

    rgb_f = rgb_crop.astype(np.float32) / 255.0
    if dark_scale < 0.99:
        rgb_f = np.clip(rgb_f * dark_scale, 0.0, 1.0)
    return rgb_f, K_out


def strict_headings(suite: str, packed: np.ndarray, poses_r: np.ndarray, poses_t: np.ndarray) -> np.ndarray:
    fwd = packed.copy()
    if suite == "aria_gen2":
        for i in range(min(2, len(fwd))):
            xy = poses_t[i, :2] - poses_t[0, :2]
            if np.linalg.norm(xy) < 0.15:
                xy = poses_r[i, :2, 2]
            norm = float(np.linalg.norm(xy))
            fwd[i, :2] = xy / norm if norm > 1e-6 else [1.0, 0.0]
            fwd[i, 2] = 0.0
    return fwd


# ---------------------------------------------------------------------------
# Causal Readout Engines
# ---------------------------------------------------------------------------

class HeadAnchoredBEV:
    """Zero-ground-dependency EgoGeomTemporalBEV in gravity-leveled camera-Z coordinates."""

    def __init__(
        self,
        z_rel_lo: float = -0.85,
        z_rel_hi: float = 0.25,
        pitch_gate_deg: float | None = None,
        cfg: TemporalBEVConfig | None = None,
    ):
        self.z_rel_lo = float(z_rel_lo)
        self.z_rel_hi = float(z_rel_hi)
        self.pitch_gate_deg = pitch_gate_deg
        self.cfg = cfg or TemporalBEVConfig(use_scale_lock=False)
        self.reset()

    def reset(self) -> None:
        self.buf: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._uv_hw: tuple[int, int] | None = None
        self._u_flat: np.ndarray | None = None
        self._v_flat: np.ndarray | None = None

    def _get_uv(self, H4: int, W4: int) -> tuple[np.ndarray, np.ndarray]:
        if self._uv_hw != (H4, W4):
            s = float(self.cfg.stride)
            v, u = np.mgrid[0:H4, 0:W4].astype(np.float32)
            self._u_flat = (u * s + 0.5 * s).reshape(-1)
            self._v_flat = (v * s + 0.5 * s).reshape(-1)
            self._uv_hw = (H4, W4)
        return self._u_flat, self._v_flat

    def step(
        self,
        depth_s4: np.ndarray,
        log_sigma_s4: np.ndarray | None,
        K: np.ndarray,
        R_wc: np.ndarray,
        t_w: np.ndarray,
        causal_fwd: np.ndarray,
    ) -> tuple[float, float, float]:
        c = self.cfg
        H4, W4 = depth_s4.shape
        u_f, v_f = self._get_uv(H4, W4)

        pitch_deg = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1.0, 1.0))))
        gated_out = (self.pitch_gate_deg is not None) and (pitch_deg > self.pitch_gate_deg)

        z = depth_s4.astype(np.float32).reshape(-1)
        ok = np.isfinite(z) & (z > 0.25) & (z < 9.5)
        fw, rt = _horizontal_fwd_right(R_wc, causal_fwd)

        occ_thr = c.occ_thr_area_m2 if c.use_area_norm else c.occ_thr_pix_eq
        n_bins = int(round((c.fwd_max_m - c.fwd_min_m) / c.z_bin_m)) + 1
        bin_cz = c.fwd_min_m + (np.arange(n_bins, dtype=np.float32) + 0.5) * c.z_bin_m
        z_mid = c.fwd_min_m + (np.arange(n_bins - 1, dtype=np.float32) + 1.0) * c.z_bin_m
        is_blind_bin = bin_cz < c.blind_zone_m

        risk_1f = 0.0
        if (not gated_out) and int(ok.sum()) >= 50:
            fx = max(float(K[0, 0]), 1.0)
            fy = max(float(K[1, 1]), 1.0)
            cx, cy = float(K[0, 2]), float(K[1, 2])
            z_ok = z[ok]
            P_cam = np.stack([(u_f[ok] - cx) * z_ok / fx, (v_f[ok] - cy) * z_ok / fy, z_ok], axis=-1)
            P_rel = (R_wc @ P_cam.T).T
            x_lat = P_rel @ rt
            z_fwd = P_rel @ fw
            z_rel = P_rel[:, 2]

            obs_m = (z_rel >= self.z_rel_lo) & (z_rel <= self.z_rel_hi) & (z_fwd >= c.fwd_min_m) & (z_fwd <= 5.0)
            if obs_m.any():
                if c.use_area_norm:
                    dA = np.clip(((c.stride * z_ok[obs_m]) ** 2) / (fx * fy), 2.0e-4, 0.012)
                else:
                    dA = np.full(int(obs_m.sum()), 0.004, dtype=np.float32)
                if c.use_uncertainty and log_sigma_s4 is not None:
                    ls_ok = log_sigma_s4.astype(np.float32).reshape(-1)[ok][obs_m]
                    conf = np.clip(np.exp(-np.clip(ls_ok, -3.0, 2.0)), 0.25, 2.5)
                else:
                    conf = np.ones_like(dA)
                wt = (dA * conf).astype(np.float32)
                pw = (P_rel[obs_m] + t_w[None, :]).astype(np.float32)
                self.buf.append((pw, wt, t_w.astype(np.float32), fw.astype(np.float32)))

                cm = (np.abs(x_lat[obs_m]) <= c.x_half_m) & (z_fwd[obs_m] <= c.fwd_max_m)
                if cm.any():
                    b1 = np.zeros(n_bins, dtype=np.float32)
                    bi1 = np.clip(((z_fwd[obs_m][cm] - c.fwd_min_m) / c.z_bin_m).astype(np.int32), 0, n_bins - 1)
                    np.add.at(b1, bi1, wt[cm])
                    s1 = b1[:-1] + b1[1:]
                    p1 = 1.0 - np.exp(-s1 / occ_thr)
                    risk_1f = float(np.max(p1 / (z_mid + 0.15)) + 0.06 * np.sum(np.log1p(s1 / occ_thr) / ((z_mid + 0.4) ** 2)))
            else:
                self.buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), t_w.astype(np.float32), fw.astype(np.float32)))
        else:
            self.buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), t_w.astype(np.float32), fw.astype(np.float32)))

        if len(self.buf) > c.window_frames:
            self.buf.pop(0)

        if gated_out:
            return float(np.inf), 0.0, 0.0

        bin_pts_w = t_w[None, :] + bin_cz[:, None] * fw[None, :]
        occ_num = np.zeros(n_bins, dtype=np.float32)
        vis_den = np.zeros(n_bins, dtype=np.float32)
        frm_cnt = np.zeros(n_bins, dtype=np.int32)
        k_len = len(self.buf)

        for age_idx, (pw, wt, t_old, fw_old) in enumerate(self.buf):
            age = (k_len - 1) - age_idx
            w_vis = float(c.decay_vis ** age)
            w_bld = float(c.decay_blind ** age)
            w_bin = np.where(is_blind_bin, w_bld, w_vis).astype(np.float32)

            z_at_old = (bin_pts_w - t_old[None, :]) @ fw_old
            in_vis = (z_at_old >= c.blind_zone_m) & (z_at_old <= 4.8)
            vis_den += w_bin * in_vis.astype(np.float32)

            if len(pw) == 0:
                continue
            rel = pw - t_w[None, :]
            xl = rel @ rt
            zf = rel @ fw
            m = (np.abs(xl) <= c.x_half_m) & (zf >= c.fwd_min_m) & (zf <= c.fwd_max_m)
            if not m.any():
                continue
            b_idx = np.clip(((zf[m] - c.fwd_min_m) / c.z_bin_m).astype(np.int32), 0, n_bins - 1)
            tmp = np.zeros(n_bins, dtype=np.float32)
            np.add.at(tmp, b_idx, wt[m])
            occ_num += tmp * w_bin
            frm_cnt += (tmp >= 0.25 * occ_thr).astype(np.int32)

        occ = occ_num / np.maximum(vis_den, 0.75)
        s2 = occ[:-1] + occ[1:]
        fc2 = np.maximum(frm_cnt[:-1], frm_cnt[1:])
        confirmed = (s2 >= occ_thr) & ((fc2 >= c.min_confirm_frames) | (s2 >= 2.6 * occ_thr))
        hits = np.flatnonzero(confirmed)
        pred_d = float(c.fwd_min_m + hits[0] * c.z_bin_m) if len(hits) else float(np.inf)

        conf_mult = np.where(fc2 >= c.min_confirm_frames, 1.0, 0.55)
        s2_eff = s2 * conf_mult
        p_occ = 1.0 - np.exp(-s2_eff / occ_thr)
        risk_bev = float(np.max(p_occ / (z_mid + 0.15)) + 0.06 * np.sum(np.log1p(s2_eff / occ_thr) / ((z_mid + 0.4) ** 2)))
        risk_hyb = c.hybrid_bev_weight * risk_bev + (1.0 - c.hybrid_bev_weight) * risk_1f
        return pred_d, risk_bev, float(risk_hyb)


class M6HeadAnchoredRisk:
    """Zero-ground-dependency M6 4-head geometric readout in camera-relative gravity Z [-0.85, +0.25] + pitch gate."""

    def __init__(
        self,
        z_rel_lo: float = -0.85,
        z_rel_hi: float = 0.25,
        pitch_gate_deg: float = 10.0,
        scale_ref: float = 1.50,
    ):
        self.z_rel_lo = float(z_rel_lo)
        self.z_rel_hi = float(z_rel_hi)
        self.pitch_gate_deg = float(pitch_gate_deg)
        self.scale_ref = float(scale_ref)
        self.reset()

    def reset(self) -> None:
        self._grid: tuple[np.ndarray, np.ndarray] | None = None
        self._prev: tuple[np.ndarray, np.ndarray] | None = None

    def step(
        self,
        depth_s4: np.ndarray,
        dh_s4: np.ndarray,
        nrm_s4: np.ndarray,
        ls_s4: np.ndarray,
        K: np.ndarray,
        R_wc: np.ndarray,
        t_w: np.ndarray,
        causal_fwd: np.ndarray,
    ) -> float:
        pitch_deg = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1.0, 1.0))))
        if pitch_deg > self.pitch_gate_deg:
            self._prev = None
            return 0.0

        sub = np.s_[::2, ::2]
        z = depth_s4[sub].astype(np.float32, copy=False).ravel()
        if self._grid is None or self._grid[0].shape != z.shape:
            v, u = np.indices(depth_s4[sub].shape, dtype=np.float32)
            self._grid = ((8.0 * u + 0.5).ravel(), (8.0 * v + 0.5).ravel())
        u, v = self._grid

        fx = max(float(K[0, 0]), 1.0)
        fy = max(float(K[1, 1]), 1.0)
        cx, cy = float(K[0, 2]), float(K[1, 2])
        xyz = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1) @ R_wc.T
        valid = np.isfinite(xyz).all(axis=1) & (z > 0.0)
        if not valid.any():
            self._prev = None
            return 0.0

        z_rel = xyz[:, 2]
        dh = dh_s4[sub].astype(np.float32, copy=False).ravel()
        n_down = nrm_s4[1][sub].astype(np.float32, copy=False).ravel()
        ls = ls_s4[sub].astype(np.float32, copy=False).ravel()
        ls_med = float(np.median(ls[valid]))

        conf = np.exp(-np.clip(ls - ls_med, -2.0, 2.0))
        band_ramp = np.clip((z_rel - self.z_rel_lo) / 0.30, 0.0, 1.0)
        dh_ramp = np.clip(dh / 0.25, 0.0, 1.0)
        vertical = np.clip(1.0 - n_down * n_down, 0.0, 1.0)
        weight = band_ramp * dh_ramp * vertical * conf

        keep = valid & (z_rel >= self.z_rel_lo) & (z_rel <= self.z_rel_hi) & (weight > 0.0)
        world = xyz[keep, :2] + t_w[:2]
        w_keep = weight[keep]

        fw = np.asarray(causal_fwd[:2], dtype=np.float64)
        nf = float(np.linalg.norm(fw))
        if nf < 1e-7:
            self._prev = None
            return 0.0
        fw = fw / nf
        rt = np.array([fw[1], -fw[0]], dtype=np.float64)

        n_b = 14
        fwd_lo_h, fwd_hi_h, lat_half_h = 0.20, 3.00, 0.22
        bin_w = (fwd_hi_h - fwd_lo_h) / float(n_b)

        def _bins(pts: np.ndarray, ev: np.ndarray) -> np.ndarray:
            if len(pts) == 0:
                return np.zeros(n_b - 1, dtype=np.float64)
            offset = (pts - t_w[:2]) / self.scale_ref
            forward = offset @ fw
            lateral = offset @ rt
            mask = (np.abs(lateral) <= lat_half_h) & (forward >= fwd_lo_h) & (forward < fwd_hi_h)
            if not mask.any():
                return np.zeros(n_b - 1, dtype=np.float64)
            idx = ((forward[mask] - fwd_lo_h) / bin_w).astype(np.int32).clip(0, n_b - 1)
            cnt = np.bincount(idx, weights=ev[mask], minlength=n_b)
            return cnt[:-1] + cnt[1:]

        now = _bins(world, w_keep)
        past = np.zeros_like(now) if self._prev is None else _bins(*self._prev)
        self._prev = (world, w_keep)
        support = 0.35 * now + 0.65 * np.minimum(now, past)
        proximity = 1.0 / (1.0 + (fwd_lo_h + bin_w + bin_w * np.arange(len(support), dtype=np.float64)) ** 2)
        return float(np.max(-np.expm1(-support / 10.0) * proximity))


def confirm_score(score: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(score, dtype=np.float64), (2, 0), constant_values=0.0)
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)


def run_all_readouts_on_clip(
    dep_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    ls_s4: np.ndarray,
    Ks: np.ndarray,
    poses_R: np.ndarray,
    poses_t: np.ndarray,
    strict_fwd: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    T = len(dep_s4)
    bev_full = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.15, hgt_hi_m=1.90, use_scale_lock=True))
    bev_midup = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.65, hgt_hi_m=1.90, use_scale_lock=True))
    ha85 = HeadAnchoredBEV(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=None)
    ha85_pg = HeadAnchoredBEV(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=10.0)
    ha70_pg = HeadAnchoredBEV(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=10.0)

    m6_full = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.06, geom_rel_h_hi=1.30))
    m6_midup = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.40, geom_rel_h_hi=1.30))
    m6_ha85_pg = M6HeadAnchoredRisk(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=10.0)

    out: dict[str, dict[str, np.ndarray]] = {
        arm: {
            "dist": np.full(T, np.inf, dtype=np.float32),
            "raw": np.zeros(T, dtype=np.float64),
        }
        for arm in ALL_ARMS
    }

    for i in range(T):
        args_bev = (dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        d, _, rh = bev_full.step(*args_bev)
        out["bev_full_ground"]["dist"][i], out["bev_full_ground"]["raw"][i] = d, rh

        d, _, rh = bev_midup.step(*args_bev)
        out["bev_midup_ground"]["dist"][i], out["bev_midup_ground"]["raw"][i] = d, rh

        d, _, rh = ha85.step(dep_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        out["bev_head_anchored_85"]["dist"][i], out["bev_head_anchored_85"]["raw"][i] = d, rh

        d, _, rh = ha85_pg.step(dep_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        out["bev_head_anchored_85_pg"]["dist"][i], out["bev_head_anchored_85_pg"]["raw"][i] = d, rh

        d, _, rh = ha70_pg.step(dep_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        out["bev_head_anchored_70_pg"]["dist"][i], out["bev_head_anchored_70_pg"]["raw"][i] = d, rh

        d_f, _, _, rg_f = m6_full.step(*args_bev)
        out["m6_full_ground"]["dist"][i], out["m6_full_ground"]["raw"][i] = d_f, rg_f

        d_m, _, _, rg_m = m6_midup.step(*args_bev)
        out["m6_midup_ground"]["dist"][i], out["m6_midup_ground"]["raw"][i] = d_m, rg_m

        rg_ha = m6_ha85_pg.step(*args_bev)
        out["m6_head_anchored_85_pg"]["dist"][i] = out["bev_head_anchored_85_pg"]["dist"][i]
        out["m6_head_anchored_85_pg"]["raw"][i] = rg_ha

    for arm in ALL_ARMS:
        out[arm]["confirmed"] = confirm_score(out[arm]["raw"])
    return out


# ---------------------------------------------------------------------------
# Matched Height-Band GT Builders
# ---------------------------------------------------------------------------

def build_bev_band_eval_rows(orig_rows: np.ndarray, gt_bev_static: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    """Derive matched corridor distance on central 3 columns (6:9) & z bins 0..18 ([0.30, 4.10) m)."""
    rows = orig_rows.astype(np.float64, copy=True)
    # gt_bev_static shape: (T, 3, 24, 15)
    occ = gt_bev_static[:, list(channels), :19, 6:9].any(axis=(1, 3))  # (T, 19)
    has = occ.any(axis=1)
    first_bin = np.argmax(occ, axis=1)
    d = np.where(has, 0.30 + 0.20 * first_bin, np.inf)
    rows[:, 0] = d
    return rows


def build_low_only_bev_mask(gt_bev_static: np.ndarray) -> np.ndarray:
    c_low = gt_bev_static[:, 0, :19, 6:9].any(axis=(1, 2))
    c_midup = gt_bev_static[:, 1:3, :19, 6:9].any(axis=(1, 2, 3))
    return c_low & (~c_midup)


# ---------------------------------------------------------------------------
# Residual Clear-Road False-Alarm Anatomy
# ---------------------------------------------------------------------------

def analyze_clip_fa_anatomy(
    dep_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    Ks: np.ndarray,
    poses_R: np.ndarray,
    strict_fwd: np.ndarray,
    pred_d_ha85: np.ndarray,
    pred_d_ha85_pg: np.ndarray,
    clear_mask: np.ndarray,
) -> dict[str, dict[str, int | list[float]]]:
    T, H4, W4 = dep_s4.shape
    v_g, u_g = np.mgrid[0:H4, 0:W4].astype(np.float32)
    u_f = (u_g * 4.0 + 2.0).reshape(-1)
    v_f = (v_g * 4.0 + 2.0).reshape(-1)

    res = {}
    for arm_tag, pd in (("ha85", pred_d_ha85), ("ha85_pg", pred_d_ha85_pg)):
        trig = clear_mask & np.isfinite(pd) & (pd <= 4.0)
        stats = {
            "n_clear": int(clear_mask.sum()),
            "n_trig": int(trig.sum()),
            "pitch_up_gt10": 0,
            "pitch_level": 0,
            "pitch_down_lt_n10": 0,
            "h_overhead": 0,
            "h_eye": 0,
            "h_chest": 0,
            "lat_edge_graze": 0,
            "lat_center": 0,
            "surf_non_vertical": 0,
            "surf_vertical": 0,
            "rng_near_lt15": 0,
            "rng_mid_15_28": 0,
            "rng_far_28_40": 0,
            "dh_med_lt010": 0,
            "dh_vals": [],
        }
        for i in np.flatnonzero(trig):
            R_wc = poses_R[i]
            pitch_deg = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1.0, 1.0))))
            if pitch_deg > 10.0:
                stats["pitch_up_gt10"] += 1
            elif pitch_deg < -10.0:
                stats["pitch_down_lt_n10"] += 1
            else:
                stats["pitch_level"] += 1

            if pd[i] < 1.5:
                stats["rng_near_lt15"] += 1
            elif pd[i] < 2.8:
                stats["rng_mid_15_28"] += 1
            else:
                stats["rng_far_28_40"] += 1

            z = dep_s4[i].reshape(-1)
            ok = np.isfinite(z) & (z > 0.25) & (z < 9.5)
            if not ok.any():
                continue
            K = Ks[i]
            fx, fy, cx, cy = max(float(K[0, 0]), 1.0), max(float(K[1, 1]), 1.0), float(K[0, 2]), float(K[1, 2])
            z_ok = z[ok]
            P_cam = np.stack([(u_f[ok] - cx) * z_ok / fx, (v_f[ok] - cy) * z_ok / fy, z_ok], axis=-1)
            P_rel = (R_wc @ P_cam.T).T
            fw, rt = _horizontal_fwd_right(R_wc, strict_fwd[i])
            xl = P_rel @ rt
            zf = P_rel @ fw
            zr = P_rel[:, 2]
            dh = dh_s4[i].reshape(-1)[ok]
            nd = nrm_s4[i, 1].reshape(-1)[ok]

            m = (zr >= -0.85) & (zr <= 0.25) & (np.abs(xl) <= 0.30) & (zf >= max(0.30, pd[i] - 0.2)) & (zf <= min(4.05, pd[i] + 0.45))
            if not m.any():
                m = (zr >= -0.85) & (zr <= 0.25) & (np.abs(xl) <= 0.30) & (zf >= 0.30) & (zf <= 4.0)
            if not m.any():
                continue

            zr_m = float(np.median(zr[m]))
            if zr_m > 0.05:
                stats["h_overhead"] += 1
            elif zr_m >= -0.35:
                stats["h_eye"] += 1
            else:
                stats["h_chest"] += 1

            if float(np.median(np.abs(xl[m]))) >= 0.20:
                stats["lat_edge_graze"] += 1
            else:
                stats["lat_center"] += 1

            if float(np.median(np.abs(nd[m]))) >= 0.75:
                stats["surf_non_vertical"] += 1
            else:
                stats["surf_vertical"] += 1

            dh_m = float(np.median(dh[m]))
            stats["dh_vals"].append(dh_m)
            if dh_m < 0.10:
                stats["dh_med_lt010"] += 1
        res[arm_tag] = stats
    return res


def merge_anatomy(records: list[dict]) -> dict:
    out = {}
    for arm_tag in ("ha85", "ha85_pg"):
        merged = {
            "n_clear": 0, "n_trig": 0,
            "pitch_up_gt10": 0, "pitch_level": 0, "pitch_down_lt_n10": 0,
            "h_overhead": 0, "h_eye": 0, "h_chest": 0,
            "lat_edge_graze": 0, "lat_center": 0,
            "surf_non_vertical": 0, "surf_vertical": 0,
            "rng_near_lt15": 0, "rng_mid_15_28": 0, "rng_far_28_40": 0,
            "dh_med_lt010": 0,
        }
        dhs = []
        for r in records:
            s = r[arm_tag]
            for k in merged:
                merged[k] += int(s[k])
            dhs.extend(s["dh_vals"])
        merged["trig_pct"] = round(100.0 * merged["n_trig"] / merged["n_clear"], 2) if merged["n_clear"] else 0.0
        merged["dh_median_m"] = round(float(np.median(dhs)), 4) if dhs else None
        out[arm_tag] = merged
    return out


# ---------------------------------------------------------------------------
# Closed-Loop Scorer, Exact Frontier, & Source-Held-Out Cross-Fit
# ---------------------------------------------------------------------------

@njit
def repeated_clear_warnings(alerts: np.ndarray, clear: np.ndarray, cumulative_m: np.ndarray, refractory_m: float) -> int:
    inside = False
    charged = False
    last = 0.0
    warnings = 0
    for i in range(len(alerts)):
        if not clear[i]:
            inside = False
            continue
        if not inside:
            inside = True
            charged = False
        if alerts[i] and (not charged or cumulative_m[i] - last > refractory_m):
            warnings += 1
            last = cumulative_m[i]
            charged = True
    return warnings


def source_id(domain: str, name: str) -> tuple[str, str]:
    if domain.startswith("arkit"):
        return "arkit", name
    if domain.startswith("sanpo"):
        # e.g. sanpo_real_xxx_camera_head -> group by underlying scene
        m = re.fullmatch(r"(.+)_(camera_(?:head|chest)|head|chest)$", name)
        return "sanpo", (m.group(1) if m else name)
    device = "aria" if domain.startswith("aria") else "coda"
    group = re.fullmatch(r"(.+)_c\d+", name)
    if group is None:
        return device, name
    return device, group.group(1)


def assign_source_folds(clips: list[dict]) -> None:
    devices: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        dev, src = source_id(c["domain"], c["name"])
        c["device"] = dev
        c["source"] = src
        devices[dev].add(src)
    folds = {}
    for dev, names in devices.items():
        ordered = sorted(names, key=lambda s: hashlib.sha256(("M3-alert-cost-v2/" + s).encode()).digest())
        for rank, name in enumerate(ordered):
            folds[(dev, name)] = rank % 2
    for c in clips:
        c["fold"] = folds[(c["device"], c["source"])]


class ClipEval:
    def __init__(self, domain: str, name: str, device: str, source: str, fold: int, rows: np.ndarray, arms: dict):
        self.domain = domain
        self.name = name
        self.device = device
        self.source = source
        self.fold = fold
        self.rows = rows
        self.arms = arms
        self.episodes = make_episodes(name, rows, BW_SPEEDS_PROXY, "blind")
        valid = self.episodes[0]["valid"]
        self.valid = valid
        self.clear = valid & (rows[:, 0] > 4.0) & (rows[:, 2] <= 0.5)
        ds = np.where(valid, rows[:, 3], 0.0)
        self.position = np.cumsum(ds, dtype=np.float64)
        self.cache: dict[tuple[str, bytes], np.ndarray] = {}

    def count(self, alerts: np.ndarray) -> np.ndarray:
        key = ("", np.packbits(alerts).tobytes())
        if key not in self.cache:
            rows = [score_episode(ep, alerts) for ep in self.episodes]
            frozen = np.sum(rows, axis=0)
            repeated = sum(
                repeated_clear_warnings(alerts, self.clear, self.position, 2.0 * ep["v"])
                for ep in self.episodes
            )
            counts = np.r_[
                frozen,
                repeated,
                (alerts & self.valid).sum(),
                self.valid.sum(),
                (alerts & self.clear).sum(),
                self.clear.sum(),
            ]
            self.cache[key] = counts
        return self.cache[key]


def summarize_counts(x: np.ndarray) -> dict:
    avoid = int(x[AVOID])
    late = int(x[LATE])
    km = float(x[KM])
    clear_n = int(x[CLEAR])
    valid_n = int(x[VALID])
    return {
        "timely": avoid - late,
        "late": late,
        "avoidable": avoid,
        "unavoidable": int(x[UNAVOID]),
        "late_pct": round(float(100.0 * late / avoid), 2) if avoid else None,
        "fa_episodes": int(x[FALSE]),
        "walker_km": round(km, 4),
        "fa_per_km": round(float(x[FALSE] / km), 3) if km > 1e-6 else None,
        "repeated_clear_warnings": int(x[CLEAR_WARN]),
        "clear_warnings_per_km": round(float(x[CLEAR_WARN] / km), 3) if km > 1e-6 else None,
        "clear_alert_frames": int(x[ON_CLEAR]),
        "known_clear_frames": clear_n,
        "clear_frame_duty": round(float(x[ON_CLEAR] / clear_n), 4) if clear_n else None,
        "alert_frames": int(x[ALERT]),
        "valid_frames": valid_n,
        "frame_duty": round(float(x[ALERT] / valid_n), 4) if valid_n else None,
    }


def frame_level_metrics(clip_objs: list[ClipEval], arm: str, hazard_thr_m: float) -> dict:
    yt4, yt_h, ys_raw, ys_conf = [], [], [], []
    err_fin, err_fix6 = [], []
    pos4_pred_fin = 0
    pos4_tot = 0
    for c in clip_objs:
        D_gt = c.rows[:, 0]
        val = c.valid
        if not val.any():
            continue
        D_pr = c.arms[arm]["dist"]
        sc_r = c.arms[arm]["raw"]
        sc_c = c.arms[arm]["confirmed"]
        yt4.append((D_gt[val] <= 4.0).astype(np.int32))
        yt_h.append((D_gt[val] <= hazard_thr_m).astype(np.int32))
        ys_raw.append(sc_r[val])
        ys_conf.append(sc_c[val])

        pos4 = val & np.isfinite(D_gt) & (D_gt >= 0.30) & (D_gt <= 4.0)
        if pos4.any():
            pos4_tot += int(pos4.sum())
            fin = pos4 & np.isfinite(D_pr) & (D_pr <= 6.0)
            pos4_pred_fin += int(fin.sum())
            if fin.any():
                err_fin.append(np.abs(D_pr[fin] - D_gt[fin]))
            d_fill = np.where(fin, D_pr, 6.0)
            err_fix6.append(np.abs(d_fill[pos4] - D_gt[pos4]))

    yt_eval = np.concatenate(yt_h)
    yt_4m = np.concatenate(yt4)
    ys_r = np.concatenate(ys_raw)
    ys_c = np.concatenate(ys_conf)

    out = {
        "n_valid": int(len(yt_eval)),
        "n_pos_eval_thr": int(yt_eval.sum()),
        "n_pos_4m": int(yt_4m.sum()),
        "hazard_thr_m": hazard_thr_m,
        "pred_finite_rate_on_pos": round(float(pos4_pred_fin / pos4_tot), 4) if pos4_tot else None,
        "mae_finite_m": round(float(np.concatenate(err_fin).mean()), 3) if err_fin else None,
        "mae_fixed6m_m": round(float(np.concatenate(err_fix6).mean()), 3) if err_fix6 else None,
    }
    for tag, scores in (("raw", ys_r), ("confirmed", ys_c)):
        if 0 < yt_eval.sum() < len(yt_eval):
            neg, pos = scores[yt_eval == 0], scores[yt_eval == 1]
            t5 = float(np.quantile(neg, 0.95))
            t10 = float(np.quantile(neg, 0.90))
            out[f"auroc_{tag}"] = round(float(roc_auc_score(yt_eval, scores)), 4)
            out[f"ap_{tag}"] = round(float(average_precision_score(yt_eval, scores)), 4)
            out[f"rec_fpr5_{tag}"] = round(float((pos >= t5).mean()), 4)
            out[f"rec_fpr10_{tag}"] = round(float((pos >= t10).mean()), 4)
        else:
            out[f"auroc_{tag}"] = None
            out[f"ap_{tag}"] = None
    return out


def exact_frontier(clip_objs: list[ClipEval], arm: str, field: str = "confirmed") -> dict:
    events: dict[float, np.ndarray] = {}
    base = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in clip_objs), np.zeros(11, dtype=np.float64))
    for c in clip_objs:
        vals = c.arms[arm][field]
        prev = c.count(np.zeros(len(vals), dtype=bool))
        for cut in np.unique(vals)[::-1]:
            nxt = c.count(vals >= cut)
            diff = nxt - prev
            if np.any(diff != 0):
                events[float(cut)] = events.get(float(cut), np.zeros(11, dtype=np.float64)) + diff
            prev = nxt

    rows = []
    cur = base.copy()

    def _pt(thr: float | None, arr: np.ndarray) -> dict:
        s = summarize_counts(arr)
        s["threshold"] = thr
        return s

    rows.append(_pt(None, cur))
    for cut in sorted(events, reverse=True):
        cur = cur + events[cut]
        rows.append(_pt(cut, cur))

    km = float(base[KM])
    avoid = int(base[AVOID])
    budget = 5.0 * km
    ok_clear = [r for r in rows if r["repeated_clear_warnings"] <= budget + 1e-9]
    ok_fa = [r for r in rows if r["fa_episodes"] <= budget + 1e-9]
    best_clear = max(ok_clear, key=lambda r: (r["timely"], -r["repeated_clear_warnings"]))
    best_fa = max(ok_fa, key=lambda r: (r["timely"], -r["fa_episodes"]))
    return {
        "avoidable": avoid,
        "walker_km": round(km, 4),
        "known_clear_frames": int(base[CLEAR]),
        "best_at_5_clear_km": best_clear,
        "best_at_5_orig_fa_km": best_fa,
    }


def fit_threshold(
    clip_objs: list[ClipEval],
    arm: str,
    field: str = "confirmed",
    robust: bool = True,
    device_only: bool = False,
    q_grid: tuple[float, ...] | np.ndarray = Q_GRID,
):
    thresholds = [-np.inf, np.inf]
    if not device_only and clip_objs:
        train_scores = np.concatenate([c.arms[arm][field] for c in clip_objs])
        thresholds += list(np.unique(np.quantile(train_scores, q_grid, method="higher")))
    candidates = []
    for thr in thresholds:
        counts = sum((c.count(c.arms[arm][field] >= thr) for c in clip_objs), np.zeros(11, dtype=np.float64))
        if counts[FALSE] > FA_BUDGET * counts[KM]:
            continue
        if robust and counts[CLEAR_WARN] > FA_BUDGET * counts[KM]:
            continue
        candidates.append(((counts[LATE], counts[CLEAR_WARN], counts[ON_CLEAR], counts[ALERT], -thr), thr, counts))
    _, threshold, best = min(candidates, key=lambda row: row[0])
    return threshold, summarize_counts(best)


FINE_Q_GRID = np.concatenate([np.linspace(0.50, 0.98, 25), np.linspace(0.982, 0.9995, 18)])


def run_2fold_crossfit(clip_objs: list[ClipEval], arm: str, field: str = "confirmed") -> dict:
    shared_rows, dev_rows, shared_fine_rows, dev_fine_rows = [], [], [], []
    for f in (0, 1):
        train = [c for c in clip_objs if c.fold != f]
        test = [c for c in clip_objs if c.fold == f]
        if not train or not test:
            continue
        assert {c.source for c in train}.isdisjoint({c.source for c in test})
        t_shared, _ = fit_threshold(train, arm, field=field, robust=True, q_grid=Q_GRID)
        t_shared_f, _ = fit_threshold(train, arm, field=field, robust=True, q_grid=FINE_Q_GRID)
        devs = sorted({c.device for c in clip_objs})
        t_dev, t_dev_f = {}, {}
        for d in devs:
            tr_d = [c for c in train if c.device == d]
            t_dev[d] = fit_threshold(tr_d, arm, field=field, robust=True, q_grid=Q_GRID)[0] if tr_d else np.inf
            t_dev_f[d] = fit_threshold(tr_d, arm, field=field, robust=True, q_grid=FINE_Q_GRID)[0] if tr_d else np.inf
        for c in test:
            vals = c.arms[arm][field]
            shared_rows.append((c.domain, c.count(vals >= t_shared)))
            dev_rows.append((c.domain, c.count(vals >= t_dev[c.device])))
            shared_fine_rows.append((c.domain, c.count(vals >= t_shared_f)))
            dev_fine_rows.append((c.domain, c.count(vals >= t_dev_f[c.device])))

    def _agg(items):
        pooled = sum((x for _, x in items), np.zeros(11, dtype=np.float64))
        doms = sorted({d for d, _ in items})
        return {
            "pooled": summarize_counts(pooled),
            "domains": {d: summarize_counts(sum((x for dd, x in items if dd == d), np.zeros(11, dtype=np.float64))) for d in doms},
        }

    return {
        "shared_crossfit": _agg(shared_rows),
        "device_crossfit": _agg(dev_rows),
        "shared_crossfit_fine": _agg(shared_fine_rows),
        "device_crossfit_fine": _agg(dev_fine_rows),
    }


def fixed_dist_diag(clip_objs: list[ClipEval], arm: str) -> dict:
    c2 = sum((c.count(c.arms[arm]["dist"] <= 2.0) for c in clip_objs), np.zeros(11, dtype=np.float64))
    c4 = sum((c.count(c.arms[arm]["dist"] <= 4.0) for c in clip_objs), np.zeros(11, dtype=np.float64))
    return {"dist_le_2m": summarize_counts(c2), "dist_le_4m": summarize_counts(c4)}


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_all = time.time()
    ckpt_path = stage_assets()
    aria_gt_bundle = np.load(INPUTS / "aria_height_band_gt.npz", allow_pickle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] Loading {ckpt_path.name} on {torch.cuda.get_device_name(0)}", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**state["cfg"]), pretrained_backbone=False)
    model.tof_branch = None
    model.load_state_dict({k: v for k, v in state["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    del state
    model.to(device).eval()

    suites_cfg = [
        ("aria_gen2", DATA_DIR / "aria_gen2", 1.0, 4.0),
        ("arkit_val30_day", DATA_DIR / "arkit_val30", 1.0, 2.0),
        ("arkit_val30_dark018", DATA_DIR / "arkit_val30", 0.18, 2.0),
        ("coda_campus", DATA_DIR / "coda_campus", 1.0, 4.0),
        ("sanpo_real_zs40", DATA_DIR / "sanpo_real_zs40", 1.0, 4.0),
    ]

    # Store compact per-clip records across bands
    # For Aria: bands = aria_centered_full_*, aria_strict_past_*
    # For ARKit/CODa/SANPO: bands = orig_full, bev_full, bev_midup, bev_head
    suite_clips: dict[str, list[dict]] = {}
    suite_meta: dict[str, dict] = {}
    anatomy_by_suite_band: dict[str, dict] = {}

    for sname, sdir, dark_scale, h_thr in suites_cfg:
        ts = time.time()
        npzs = sorted(sdir.rglob("*.npz"))
        print(f"\n[infer] {sname}: {len(npzs)} clips (dark_scale={dark_scale})", flush=True)
        clips_list = []
        cam_heights = defaultdict(list)
        low_only_cnt = 0
        valid_tot = 0
        sanpo_cam_counts = defaultdict(int)
        anatomy_recs = defaultdict(list)

        for ci, p in enumerate(npzs, 1):
            with np.load(p, allow_pickle=False) as d:
                offs, blob = d["rgb_offsets"], d["rgb_blob"]
                orig_hw, K_orig = d["orig_hw"], d["K_orig"]
                poses_R, poses_t = d["poses_R"].astype(np.float32), d["poses_t"].astype(np.float32)
                causal_fwd = d["causal_fwd"].astype(np.float32)
                eval_rows = d["eval_rows"].astype(np.float64)
                gt_bev_static = d["gt_bev_static"] if "gt_bev_static" in d else None
                cam_name = str(d["cam"]) if "cam" in d else ("rover_flir" if sname == "coda_campus" else "unknown")

            if sname == "sanpo_real_zs40":
                if cam_name == "unknown":
                    if "camera_head" in p.stem or p.stem.endswith("_head"):
                        cam_name = "camera_head"
                    elif "camera_chest" in p.stem or p.stem.endswith("_chest"):
                        cam_name = "camera_chest"
                sanpo_cam_counts[cam_name] += 1

            T = len(eval_rows)
            true_fwd = strict_headings(sname, causal_fwd, poses_R, poses_t)
            rgbs, Ks, gs = [], [], []
            for i in range(T):
                rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=dark_scale)
                rgbs.append(((rgb_f - IM_MEAN) / IM_STD).transpose(2, 0, 1))
                Ks.append(K)
                g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                gs.append(g_cam)
            Ks_arr = np.stack(Ks).astype(np.float32)

            dep_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            dh_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            nrm_s4 = np.empty((T, 3, H_IN // 4, W_IN // 4), dtype=np.float32)
            ls_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode():
                for i0 in range(0, T, 32):
                    i1 = min(T, i0 + 32)
                    x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                    Kt = torch.from_numpy(Ks_arr[i0:i1]).to(device)
                    gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device)
                    rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
                    hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        out = model(x, rays, None, None, host_focal=hf, return_features=False)
                    dep_s4[i0:i1] = out["depth_m"][:, 0, ::4, ::4].float().cpu().numpy()
                    dh_s4[i0:i1] = out["delta_h_m"][:, 0, ::4, ::4].float().cpu().numpy()
                    nrm_s4[i0:i1] = out["normal_grav"][:, :, ::4, ::4].float().cpu().numpy()
                    ls_s4[i0:i1] = out["log_sigma"][:, 0, ::4, ::4].float().cpu().numpy()

            # Sample camera height estimate from floor normals
            v_sub, u_sub = np.indices((H_IN // 4, W_IN // 4), dtype=np.float32)
            u_sub = 4.0 * u_sub + 2.0
            v_sub = 4.0 * v_sub + 2.0
            for idx_h in range(0, T, 10):
                z_i = dep_s4[idx_h]
                K_i = Ks_arr[idx_h]
                xyz = np.stack(((u_sub - K_i[0, 2]) * z_i / K_i[0, 0], (v_sub - K_i[1, 2]) * z_i / K_i[1, 1], z_i), axis=-1) @ poses_R[idx_h].T
                fl = (xyz[..., 2] < 0) & (np.abs(nrm_s4[idx_h, 1]) > 0.65) & (z_i > 0.3)
                if fl.sum() >= 20:
                    cam_heights[cam_name].append(float(-np.percentile(xyz[..., 2][fl], 15)))

            arms = run_all_readouts_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, true_fwd)

            gt_bands: dict[str, np.ndarray] = {}
            if sname == "aria_gen2":
                for st in ("centered_full", "strict_past"):
                    for b in ("full_ground", "low_only_ground", "midup_ground", "head_ground", "head_anchored_85", "head_anchored_70"):
                        gt_bands[f"{st}__{b}"] = aria_gt_bundle[f"{p.stem}__{st}__{b}"].astype(np.float64)
                val_cf = (np.isfinite(eval_rows[:, 0]) | np.isinf(eval_rows[:, 0])) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
                valid_tot += int(val_cf.sum())
                cf_full = gt_bands["centered_full__full_ground"][:, 0]
                cf_midup = gt_bands["centered_full__midup_ground"][:, 0]
                low_only_cnt += int((val_cf & (cf_full <= 4.0) & (cf_midup > 4.0)).sum())
                for b_ana in ("centered_full__full_ground", "centered_full__midup_ground", "centered_full__head_anchored_85", "strict_past__midup_ground", "strict_past__head_anchored_85"):
                    r_b = gt_bands[b_ana]
                    v_b = (np.isfinite(r_b[:, 0]) | np.isinf(r_b[:, 0])) & ~np.isnan(r_b[:, 1]) & (r_b[:, 1] >= 0.6)
                    clr_b = v_b & (r_b[:, 0] > 4.0) & (r_b[:, 2] <= 0.5)
                    anatomy_recs[b_ana].append(
                        analyze_clip_fa_anatomy(
                            dep_s4, dh_s4, nrm_s4, Ks_arr, poses_R, true_fwd,
                            arms["bev_head_anchored_85"]["dist"],
                            arms["bev_head_anchored_85_pg"]["dist"],
                            clr_b,
                        )
                    )
            else:
                gt_bands["orig_full"] = eval_rows
                if gt_bev_static is not None:
                    gt_bands["bev_full"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
                    gt_bands["bev_low_only"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0,))
                    gt_bands["bev_midup"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))
                    gt_bands["bev_head"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (2,))
                    val_b = (np.isfinite(eval_rows[:, 0]) | np.isinf(eval_rows[:, 0])) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
                    valid_tot += int(val_b.sum())
                    low_only_cnt += int((val_b & build_low_only_bev_mask(gt_bev_static)).sum())
                    for b_ana in ("bev_full", "bev_midup"):
                        r_b = gt_bands[b_ana]
                        clr_b = val_b & (r_b[:, 0] > 4.0) & (r_b[:, 2] <= 0.5)
                        anatomy_recs[b_ana].append(
                            analyze_clip_fa_anatomy(
                                dep_s4, dh_s4, nrm_s4, Ks_arr, poses_R, true_fwd,
                                arms["bev_head_anchored_85"]["dist"],
                                arms["bev_head_anchored_85_pg"]["dist"],
                                clr_b,
                            )
                        )

            clips_list.append({
                "domain": sname,
                "name": p.stem,
                "cam": cam_name,
                "arms": arms,
                "gt_bands": gt_bands,
            })
            if ci == 1 or ci % 10 == 0 or ci == len(npzs):
                print(f"  [{sname}] {ci}/{len(npzs)} clips", flush=True)

        assign_source_folds(clips_list)
        suite_clips[sname] = clips_list
        suite_meta[sname] = {
            "n_clips": len(clips_list),
            "hazard_thr_m": h_thr,
            "valid_frames": valid_tot,
            "low_only_frames": low_only_cnt,
            "cam_counts": dict(sanpo_cam_counts) if sanpo_cam_counts else None,
            "est_cam_height_median_m": {k: round(float(np.median(v)), 3) for k, v in cam_heights.items() if v},
            "elapsed_s": round(time.time() - ts, 2),
        }
        anatomy_by_suite_band[sname] = {b: merge_anatomy(recs) for b, recs in anatomy_recs.items()}
        print(f"  [{sname}] Done in {suite_meta[sname]['elapsed_s']}s | cam_h={suite_meta[sname]['est_cam_height_median_m']}", flush=True)

    # -----------------------------------------------------------------------
    # Build Report Sections
    # -----------------------------------------------------------------------
    print("\n[eval] Computing denominator breakdown, exact frontiers, and 2-fold cross-fits...", flush=True)

    def make_eval_objs(suite_list: list[str], band_map: dict[str, str], cam_filter: str | None = None) -> list[ClipEval]:
        objs = []
        for s in suite_list:
            bkey = band_map[s]
            for c in suite_clips[s]:
                if cam_filter is not None and c["cam"] != cam_filter:
                    continue
                if bkey not in c["gt_bands"]:
                    continue
                objs.append(ClipEval(c["domain"], c["name"], c["device"], c["source"], c["fold"], c["gt_bands"][bkey], c["arms"]))
        return objs

    # 1) Denominator Breakdown & Baseline Concordance
    denominators = {}
    aria_bands_all = [f"{st}__{b}" for st in ("centered_full", "strict_past") for b in ("full_ground", "low_only_ground", "midup_ground", "head_ground", "head_anchored_85", "head_anchored_70")]
    denominators["aria_gen2"] = {}
    for b in aria_bands_all:
        objs = make_eval_objs(["aria_gen2"], {"aria_gen2": b})
        zero = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in objs), np.zeros(11, dtype=np.float64))
        fm = frame_level_metrics(objs, "bev_full_ground", 4.0)
        denominators["aria_gen2"][b] = {
            "n_valid": fm["n_valid"],
            "n_pos_4m": fm["n_pos_4m"],
            "n_clear": int(zero[CLEAR]),
            "avoidable_6w": int(zero[AVOID]),
            "unavoidable_6w": int(zero[UNAVOID]),
            "walker_km": round(float(zero[KM]), 4),
        }
    for st in ("centered_full", "strict_past"):
        excl = 0
        for c in suite_clips["aria_gen2"]:
            rf = c["gt_bands"][f"{st}__full_ground"]
            rm = c["gt_bands"][f"{st}__midup_ground"]
            val = (np.isfinite(rf[:, 0]) | np.isinf(rf[:, 0])) & ~np.isnan(rf[:, 1]) & (rf[:, 1] >= 0.6)
            excl += int((val & (rf[:, 0] <= 4.0) & (rm[:, 0] > 4.0)).sum())
        denominators["aria_gen2"][f"{st}__low_only_exclusive_frames"] = excl

    for sname, h_thr in [("arkit_val30_day", 2.0), ("coda_campus", 4.0), ("sanpo_real_zs40", 4.0)]:
        denominators[sname] = {}
        for b in ("orig_full", "bev_full", "bev_low_only", "bev_midup", "bev_head"):
            objs = make_eval_objs([sname], {sname: b})
            zero = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in objs), np.zeros(11, dtype=np.float64))
            fm = frame_level_metrics(objs, "bev_full_ground", h_thr)
            denominators[sname][b] = {
                "n_valid": fm["n_valid"],
                "n_pos_eval_thr": fm["n_pos_eval_thr"],
                "n_pos_4m": fm["n_pos_4m"],
                "n_clear": int(zero[CLEAR]),
                "avoidable_6w": int(zero[AVOID]),
                "unavoidable_6w": int(zero[UNAVOID]),
                "walker_km": round(float(zero[KM]), 4),
            }
        # Concordance orig_full vs bev_full
        orig_pos = sum(int(((c.rows[:, 0] <= 4.0) & c.valid).sum()) for c in make_eval_objs([sname], {sname: "orig_full"}))
        bev_pos = sum(int(((c.rows[:, 0] <= 4.0) & c.valid).sum()) for c in make_eval_objs([sname], {sname: "bev_full"}))
        both_pos = 0
        for co, cb in zip(make_eval_objs([sname], {sname: "orig_full"}), make_eval_objs([sname], {sname: "bev_full"})):
            both_pos += int((co.valid & (co.rows[:, 0] <= 4.0) & (cb.rows[:, 0] <= 4.0)).sum())
        denominators[sname]["orig_vs_bev_concordance"] = {
            "orig_pos_4m": orig_pos,
            "bev_pos_4m": bev_pos,
            "both_pos_4m": both_pos,
            "low_only_exclusive_frames": suite_meta[sname]["low_only_frames"],
        }

    # 2) Single-Domain Evaluations across GT bands & Readout Arms
    per_domain_results = {}

    def eval_cohort(objs: list[ClipEval], h_thr: float, arms_to_run: tuple[str, ...]) -> dict:
        res = {}
        for arm in arms_to_run:
            res[arm] = {
                "frame": frame_level_metrics(objs, arm, h_thr),
                "frontier_confirmed": exact_frontier(objs, arm, "confirmed"),
                "frontier_raw": exact_frontier(objs, arm, "raw"),
                "fixed_dist": fixed_dist_diag(objs, arm),
            }
        return res

    # Aria (centered_full & strict_past across full_ground, midup_ground, head_anchored_85)
    per_domain_results["aria_gen2"] = {}
    for b in ("centered_full__full_ground", "centered_full__midup_ground", "centered_full__head_anchored_85",
              "strict_past__full_ground", "strict_past__midup_ground", "strict_past__head_anchored_85"):
        objs = make_eval_objs(["aria_gen2"], {"aria_gen2": b})
        per_domain_results["aria_gen2"][b] = eval_cohort(objs, 4.0, ALL_ARMS)

    # ARKit day & dark, CODa
    for sname, h_thr in [("arkit_val30_day", 2.0), ("arkit_val30_dark018", 2.0), ("coda_campus", 4.0)]:
        per_domain_results[sname] = {}
        for b in ("orig_full", "bev_full", "bev_midup"):
            objs = make_eval_objs([sname], {sname: b})
            per_domain_results[sname][b] = eval_cohort(objs, h_thr, ALL_ARMS)

    # SANPO-Real ZS40 (all, camera_head, camera_chest) + 2-fold scene-disjoint crossfit
    per_domain_results["sanpo_real_zs40"] = {}
    for cam_sub in ("all", "camera_head", "camera_chest"):
        cf_arg = None if cam_sub == "all" else cam_sub
        per_domain_results["sanpo_real_zs40"][cam_sub] = {}
        for b in ("orig_full", "bev_full", "bev_midup"):
            objs = make_eval_objs(["sanpo_real_zs40"], {"sanpo_real_zs40": b}, cam_filter=cf_arg)
            if not objs:
                continue
            cohort_res = eval_cohort(objs, 4.0, ALL_ARMS)
            if cam_sub == "all":
                for arm in ALL_ARMS:
                    cohort_res[arm]["crossfit_2fold"] = run_2fold_crossfit(objs, arm, "confirmed")
            per_domain_results["sanpo_real_zs40"][cam_sub][b] = cohort_res

    # 3) 4-Domain 38-Source Pooled Evaluation & 2-Fold Source-Held-Out Cross-Fit (M3/M6 parity)
    pooled_4dom_configs = {
        "full_ground_centered": {
            "aria_gen2": "centered_full__full_ground",
            "arkit_val30_day": "orig_full",
            "arkit_val30_dark018": "orig_full",
            "coda_campus": "orig_full",
        },
        "midup_ground_matched": {
            "aria_gen2": "centered_full__midup_ground",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
        "full_matched_bev": {
            "aria_gen2": "centered_full__full_ground",
            "arkit_val30_day": "bev_full",
            "arkit_val30_dark018": "bev_full",
            "coda_campus": "bev_full",
        },
        "midup_strict_past_aria": {
            "aria_gen2": "strict_past__midup_ground",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
        "head_anchored85_aria_midup_others": {
            "aria_gen2": "centered_full__head_anchored_85",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
    }
    four_domain_results = {}
    doms4 = ["aria_gen2", "arkit_val30_day", "arkit_val30_dark018", "coda_campus"]
    for cfg_name, bmap in pooled_4dom_configs.items():
        objs4 = make_eval_objs(doms4, bmap)
        four_domain_results[cfg_name] = {}
        for arm in ALL_ARMS:
            four_domain_results[cfg_name][arm] = {
                "frontier_confirmed": exact_frontier(objs4, arm, "confirmed"),
                "crossfit_2fold": run_2fold_crossfit(objs4, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag(objs4, arm),
            }

    # Sanity check against archived v4.1 Aria AUROC (~0.7060)
    aria_base = per_domain_results["aria_gen2"]["centered_full__full_ground"]["bev_full_ground"]
    aria_auroc = aria_base["frame"]["auroc_raw"]
    aria_n_val = aria_base["frame"]["n_valid"]
    aria_n_pos = aria_base["frame"]["n_pos_4m"]
    print(
        f"\n[sanity] Aria baseline centered_full/full_ground/bev_full_ground: "
        f"n_valid={aria_n_val}, n_pos={aria_n_pos}, AUROC_raw={aria_auroc}, "
        f"frontier_5clear={aria_base['frontier_confirmed']['best_at_5_clear_km']['timely']}/{aria_base['frontier_confirmed']['avoidable']}",
        flush=True,
    )
    if aria_auroc is None or abs(aria_auroc - 0.7060) > 0.015:
        raise RuntimeError(f"Aria baseline AUROC {aria_auroc} diverged >0.015 from archived 0.7060!")

    report = {
        "meta": {
            "checkpoint": ckpt_path.name,
            "gpu": torch.cuda.get_device_name(0),
            "total_elapsed_s": round(time.time() - t_all, 2),
            "suites": suite_meta,
        },
        "denominators": denominators,
        "per_domain": per_domain_results,
        "four_domain_38source": four_domain_results,
        "residual_clear_fa_anatomy": anatomy_by_suite_band,
    }

    out_json = OUT_DIR / "cane_blind_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] Saved {out_json} ({out_json.stat().st_size / 1e3:.1f} KB) in {report['meta']['total_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
