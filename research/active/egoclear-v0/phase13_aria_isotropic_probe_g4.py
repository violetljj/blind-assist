"""Phase 13 Step 1 (5-Min Surgical X-Ray Probe on Colab G4):
Tests the 1.762x Aspect-Ratio Nuclear Hypothesis + Failure Autopsy on Aria, CODa, ARKit, and SANPO-Real 40!

Zero retraining (~1 CU):
  1. Tests Squashed (baseline) vs Isotropic Letterbox (fx==fy, 100% FOV + valid_img_mask) vs
     Isotropic Center-Crop & Gravity-Horizon-Crop (fx==fy, zero black bars) on frozen v6.6, v8.5, v9, and Tri-ENS!
  2. Audits MoGe-2-ViT-S vs v9 vs MoGe-2-ViT-L floor n_up, scale, shift, and 3D vertical-surface SNR on Aria & CODa!
  3. Exact Failure Autopsy on Aria Gen 2 (193 events) & SANPO-Real 40 (111 events): pitch, head-yaw, blind-spot vis,
     and Tri-Champion union ceiling!
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, NY, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v7_model import FEAT_H, FEAT_W
from egoclear_v8_model import EgoClearV8Net
from egoclear_v9_model import EgoClearV9Net
from eval_closed_loop import H, MIN_EVENT_FRAMES_V2, eval_score_curve, make_episodes, summarize_at_fa_caps, weak_mask
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_train_v75_g4 import IDX_4M_EXACT
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_OUT = DRIVE_ROOT / "runs/egoclear_v0_phase13_probe"
LOCAL_OUT = Path("/content/egoclear_p13_probe")
CODA_DIR = Path("/content/coda_zeroshot_v0")

THRS = np.array(
    [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40,
     0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98],
    dtype=np.float64,
)


def summarize_scores(ev_rows, scores, bw_speeds) -> dict:
    s = summarize_at_fa_caps(eval_score_curve(ev_rows, scores, bw_speeds, THRS, higher_is_alert=True))
    def pick(k):
        m = s.get(k)
        return round(m["late_rate"] * 100.0, 1) if m else None
    return {
        "@80": pick("late_at_fa_fr<=80.0"),
        "@120": pick("late_at_fa_fr<=120.3"),
        "@200": pick("late_at_fa_fr<=200.0"),
        "@350": pick("late_at_fa_fr<=350.0"),
        "@500": pick("late_at_fa_fr<=500.0"),
    }


def logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def prepare_frontend_tensors(d: dict, device: torch.device, mode: str = "squashed") -> dict[str, torch.Tensor]:
    """Prepares exact pinhole-preserving inputs in 4 frontend modes:
      - squashed:      raw 518x294 (baseline, fx != fy for 1:1 / 5:4 / 4:3 cameras)
      - iso_letterbox: exact fx==fy==min(fx,fy), centered inside 518x294 with black letterbox bars (100% FOV!)
      - iso_crop:      exact fx==fy==max(fx,fy), vertically center-cropped (zero black bars!)
      - iso_grav_crop: exact fx==fy==max(fx,fy), vertically shifted upward by gravity pitch (zero black bars, horizon-centered!)
    """
    n = len(d["eval_rows"])
    sl = np.arange(n)
    rgb = decode_rgb_batch(d, sl)  # (T, 3, H, W) in [0, 1]
    K_o = d["K_orig"].astype(np.float32)
    Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
    K_base = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
    R_seq = d["poses_R"][sl].astype(np.float32)
    t_seq = d["poses_t"][sl].astype(np.float32)
    fwd_seq = d["causal_fwd"][sl].astype(np.float32)
    seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

    fx, fy, cx, cy = [float(x) for x in K_base]
    ratio = fx / max(fy, 1e-6)  # > 1.0 means horizontally stretched (e.g. Aria=1.762, CODa=1.474, ARKit=1.321)
    K_seq = np.repeat(K_base[None], n, axis=0)
    rgb_out = rgb

    if mode == "iso_letterbox" and abs(ratio - 1.0) > 0.03:
        if ratio > 1.0:
            # Compress width from RGB_W -> nw so fx_new == fy
            nw = max(32, int(round(RGB_W / ratio)))
            pad_l = (RGB_W - nw) // 2
            rgb_out = np.zeros_like(rgb)
            for ti in range(n):
                hwc = rgb[ti].transpose(1, 2, 0)
                rs = cv2.resize(hwc, (nw, RGB_H), interpolation=cv2.INTER_AREA)
                rgb_out[ti, :, :, pad_l : pad_l + nw] = rs.transpose(2, 0, 1)
            sx = nw / float(RGB_W)
            K_iso = np.array([fx * sx, fy, cx * sx + pad_l, cy], dtype=np.float32)
            K_seq = np.repeat(K_iso[None], n, axis=0)
        else:
            nh = max(32, int(round(RGB_H * ratio)))
            pad_t = (RGB_H - nh) // 2
            rgb_out = np.zeros_like(rgb)
            for ti in range(n):
                hwc = rgb[ti].transpose(1, 2, 0)
                rs = cv2.resize(hwc, (RGB_W, nh), interpolation=cv2.INTER_AREA)
                rgb_out[ti, :, pad_t : pad_t + nh, :] = rs.transpose(2, 0, 1)
            sy = nh / float(RGB_H)
            K_iso = np.array([fx, fy * sy, cx, cy * sy + pad_t], dtype=np.float32)
            K_seq = np.repeat(K_iso[None], n, axis=0)

    elif mode in ("iso_crop", "iso_grav_crop") and ratio > 1.03:
        # Un-squash height from RGB_H -> nh_full so fy_new == fx, then crop RGB_H rows!
        nh_full = int(round(RGB_H * ratio))
        sy = nh_full / float(RGB_H)
        cy_full = cy * sy
        rgb_out = np.zeros_like(rgb)
        K_seq = np.empty((n, 4), dtype=np.float32)
        for ti in range(n):
            hwc = rgb[ti].transpose(1, 2, 0)
            rs = cv2.resize(hwc, (RGB_W, nh_full), interpolation=cv2.INTER_LINEAR)
            if mode == "iso_crop":
                y0 = (nh_full - RGB_H) // 2
            else:
                # Causal IMU gravity-horizon crop: center window slightly below horizon (-8 deg) where corridor lives!
                pitch_rad = math.asin(float(np.clip(R_seq[ti, 2, 2], -1.0, 1.0)))
                # Target optical ray pitch = -8 deg => shift in y relative to principal point:
                # positive y in image points DOWN; if camera looks down (pitch < 0), horizon is ABOVE cy (smaller y)
                target_y = cy_full + (fy * sy) * math.tan(pitch_rad - math.radians(-8.0))
                y0 = int(round(target_y - 0.5 * RGB_H))
                y0 = max(0, min(nh_full - RGB_H, y0))
            rgb_out[ti] = rs[y0 : y0 + RGB_H, :, :].transpose(2, 0, 1)
            K_seq[ti] = np.array([fx, fy * sy, cx, cy_full - y0], dtype=np.float32)

    rgb_norm = (rgb_out - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
    return {
        "rgb": torch.from_numpy(rgb_norm)[None].to(device, non_blocking=True),
        "K_scaled": torch.from_numpy(K_seq)[None].to(device, non_blocking=True),
        "R_c2w": torch.from_numpy(R_seq)[None].to(device, non_blocking=True),
        "t_w": torch.from_numpy(t_seq)[None].to(device, non_blocking=True),
        "fwd_w": torch.from_numpy(fwd_seq)[None].to(device, non_blocking=True),
        "seg_len": torch.from_numpy(seg_seq)[None].to(device, non_blocking=True),
    }


def autopsy_missed_events(
    dmap: dict[str, dict],
    ov: dict[str, np.ndarray] | None,
    scores: dict[str, np.ndarray],
    bw_speeds: list[float],
    thr: float,
    vis_by_sc: dict[str, np.ndarray],
) -> dict:
    recs = []
    hits = 0
    for sc, d in dmap.items():
        if sc not in scores:
            continue
        er = ov[sc] if ov is not None else d["eval_rows"]
        if len(er) < 5:
            continue
        p4 = scores[sc]
        R = d["poses_R"].astype(np.float64)
        fwd = d["causal_fwd"].astype(np.float64)
        vis = vis_by_sc.get(sc, np.ones(len(er), dtype=np.float32))
        for ep in make_episodes(sc, er, bw_speeds, kind="blind"):
            r = ep["rows"]
            D, valid = r[:, 0], ep["valid"]
            haz = valid & (D <= H)
            al = (p4 >= thr) & valid
            i, n = 0, len(D)
            while i < n:
                if haz[i]:
                    j = i
                    while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                        j += 1
                    seg = slice(i, j + 1)
                    if haz[seg].sum() >= MIN_EVENT_FRAMES_V2:
                        ok = D[seg] >= ep["dreq"]
                        if ok.any():
                            if (al[seg] & ok).any():
                                hits += 1
                            else:
                                ok_idx = np.nonzero(ok)[0] + i
                                peak_fi = int(ok_idx[np.argmax(p4[ok_idx])])
                                cf = R[peak_fi, :, 2]
                                pitch_deg = float(math.degrees(math.asin(np.clip(cf[2], -1.0, 1.0))))
                                cf2 = cf[:2] / max(float(np.linalg.norm(cf[:2])), 1e-6)
                                yaw_deg = float(math.degrees(math.acos(np.clip(float(np.dot(cf2, fwd[peak_fi, :2])), -1.0, 1.0))))
                                recs.append({
                                    "pitch_deg": pitch_deg,
                                    "yaw_deg": yaw_deg,
                                    "max_vis": float(vis[ok_idx].max()),
                                    "mean_vis": float(vis[ok_idx].mean()),
                                    "n_ok": int(len(ok_idx)),
                                    "max_p": float(p4[ok_idx].max()),
                                    "d_min": float(D[ok_idx].min()),
                                })
                    i = j + 1
                else:
                    i += 1
    nm = len(recs)
    return {
        "avoidable": hits + nm,
        "missed": nm,
        "late_pct": round(100.0 * nm / max(hits + nm, 1), 1),
        "thr": thr,
        "frac_pitch_lt_neg25": round(float(np.mean([r["pitch_deg"] < -25.0 for r in recs])), 3) if nm else 0.0,
        "frac_yaw_ge_25": round(float(np.mean([r["yaw_deg"] >= 25.0 for r in recs])), 3) if nm else 0.0,
        "frac_blindspot_vis_lt_0.25": round(float(np.mean([r["max_vis"] < 0.25 for r in recs])), 3) if nm else 0.0,
        "frac_tight_le2fr": round(float(np.mean([r["n_ok"] <= 2 for r in recs])), 3) if nm else 0.0,
        "median_pitch_deg": round(float(np.median([r["pitch_deg"] for r in recs])), 1) if nm else None,
        "median_yaw_deg": round(float(np.median([r["yaw_deg"] for r in recs])), 1) if nm else None,
        "median_max_vis": round(float(np.median([r["max_vis"] for r in recs])), 3) if nm else None,
        "median_max_p": round(float(np.median([r["max_p"] for r in recs])), 3) if nm else None,
    }


def compute_corridor_vis_frac(d: dict, K_seq: np.ndarray) -> np.ndarray:
    """Fraction of straight-ahead corridor points (x in [-0.3,0.3], z in [0.5,4.0], y in [-1.2,0.3]) inside camera FOV."""
    n = len(d["eval_rows"])
    xs = np.array([-0.25, 0.0, 0.25], dtype=np.float32)
    zs = np.array([1.0, 2.0, 3.0, 3.8], dtype=np.float32)
    ys = np.array([-1.1, -0.5, 0.1], dtype=np.float32)
    gy, gz, gx = np.meshgrid(ys, zs, xs, indexing="ij")
    gx, gz, gy = gx.reshape(-1), gz.reshape(-1), gy.reshape(-1)
    R = d["poses_R"].astype(np.float32)
    t = d["poses_t"].astype(np.float32)
    f = d["causal_fwd"].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        right = np.array([f[i, 1], -f[i, 0], 0.0], dtype=np.float32)
        rel = gx[:, None] * right[None] + gz[:, None] * f[i][None] + gy[:, None] * np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        pc = rel @ R[i]
        zc = pc[:, 2]
        ok = zc > 0.2
        u = K_seq[i, 0] * (pc[:, 0] / np.maximum(zc, 0.2)) + K_seq[i, 2]
        v = K_seq[i, 1] * (pc[:, 1] / np.maximum(zc, 0.2)) + K_seq[i, 3]
        vis = ok & (u >= 0) & (u < RGB_W) & (v >= 0) & (v < RGB_H)
        out[i] = float(vis.mean())
    return out


def main():
    t0 = time.time()
    device = torch.device("cuda")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)

    paths = stage_all_p8_datasets_parallel()
    CODA_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(CODA_DIR.glob("*.npz"))) < 20:
        for tf_p in sorted((DRIVE_ROOT / "datasets/coda_zeroshot_v0").glob("*.tar")):
            tmp_t = CODA_DIR / tf_p.name
            shutil.copy2(tf_p, tmp_t)
            with tarfile.open(tmp_t, "r") as tf:
                tf.extractall(CODA_DIR, filter="data")
            tmp_t.unlink(missing_ok=True)

    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))
    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    # Load ALL 3 Pareto Champions: v6.6 (DAV2-S), v8.5 (MoGe-2-S), v9 (Dual-Geom MoGe-2-S)
    v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    v85 = EgoClearV8Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), freeze_moge=True, use_geom_residual=True, use_ground_relative=False).to(device).eval()
    v85.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase11_v85/egoclear_v85_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    v9 = EgoClearV9Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), freeze_moge=True).to(device).eval()
    v9.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase12_v9/egoclear_v9_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    # Check 1: Zero-Retrain Aspect-Ratio Nuclear Fix on Aria (1.762x), CODa (1.474x), ARKit (1.321x)!
    print("\n=== CHECK 1: Zero-Retrain Isotropic Frontend (Squashed vs Letterbox vs Crop vs GravCrop) ===", flush=True)
    frontend_results = {}
    aria_scores_cache = {}
    aria_vis_cache = {}

    for dname, dmap, ov, modes in [
        ("aria_g2_zeroshot", aria_dicts, None, ("squashed", "iso_letterbox", "iso_crop", "iso_grav_crop")),
        ("coda_campus_zeroshot", eval_coda, None, ("squashed", "iso_letterbox", "iso_crop")),
        ("arkit_indoor_val30", eval_arkit, None, ("squashed", "iso_letterbox")),
        ("real_40_local", eval_real, real_local_rows, ("squashed",)),
    ]:
        frontend_results[dname] = {}
        for mode in modes:
            ev_r, s66, s85, s9, s_tri, s_66_9 = {}, {}, {}, {}, {}, {}
            vis_m = {}
            for sc, d in dmap.items():
                er = ov[sc] if ov is not None else d["eval_rows"]
                if len(er) < 5:
                    continue
                inp = prepare_frontend_tensors(d, device, mode=mode)
                vis_m[sc] = compute_corridor_vis_frac(d, inp["K_scaled"][0].cpu().numpy())
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p66 = v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    p85 = v85.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    p9 = v9.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                ev_r[sc] = er
                s66[sc] = p66
                s85[sc] = p85
                s9[sc] = p9
                s_66_9[sc] = sigmoid(0.5 * logit(p66) + 0.5 * logit(p9))
                s_tri[sc] = sigmoid((logit(p66) + (logit(p85) + 1.0) + logit(p9)) / 3.0)

            res_m = {
                "v66": summarize_scores(ev_r, s66, bw_speeds),
                "v85": summarize_scores(ev_r, s85, bw_speeds),
                "v9": summarize_scores(ev_r, s9, bw_speeds),
                "ens_v66_v9": summarize_scores(ev_r, s_66_9, bw_speeds),
                "ens_tri": summarize_scores(ev_r, s_tri, bw_speeds),
            }
            frontend_results[dname][mode] = res_m
            print(f"  [{dname:20s} | {mode:13s}]", flush=True)
            for mk, mv in res_m.items():
                print(f"      {mk:11s}: {mv}", flush=True)

            if dname == "aria_g2_zeroshot" and mode in ("squashed", "iso_letterbox", "iso_grav_crop"):
                aria_scores_cache[mode] = {"v66": s66, "v85": s85, "v9": s9, "tri": s_tri}
                aria_vis_cache[mode] = vis_m
            if dname == "real_40_local":
                real_scores_cache = {"v66": s66, "v85": s85, "v9": s9, "tri": s_tri}
                real_vis_cache = vis_m

    # Check 2: Exact Failure Autopsy on Aria (193 events) and SANPO-Real 40 (111 events)
    print("\n=== CHECK 2: Exact Failure Autopsy (Aria Gen 2 & SANPO-Real 40) ===", flush=True)
    autopsies = {}
    for mode in ("squashed", "iso_letterbox", "iso_grav_crop"):
        for mk in ("v66", "v9", "tri"):
            c = summarize_at_fa_caps(eval_score_curve({k: d["eval_rows"] for k, d in aria_dicts.items()}, aria_scores_cache[mode][mk], bw_speeds, THRS, higher_is_alert=True))
            thr = float(c["late_at_fa_fr<=120.3"]["thr"]) if c.get("late_at_fa_fr<=120.3") else 0.32
            a = autopsy_missed_events(aria_dicts, None, aria_scores_cache[mode][mk], bw_speeds, thr, aria_vis_cache[mode])
            autopsies[f"aria_{mode}_{mk}"] = a
            print(f"  [Autopsy aria {mode:13s} {mk:4s} @120] {a}", flush=True)

    for mk in ("v66", "v85", "v9", "tri"):
        c = summarize_at_fa_caps(eval_score_curve(real_local_rows, real_scores_cache[mk], bw_speeds, THRS, higher_is_alert=True))
        thr = float(c["late_at_fa_fr<=120.3"]["thr"]) if c.get("late_at_fa_fr<=120.3") else 0.50
        a = autopsy_missed_events(eval_real, real_local_rows, real_scores_cache[mk], bw_speeds, thr, real_vis_cache)
        autopsies[f"real40_{mk}"] = a
        print(f"  [Autopsy real40 {mk:4s} @120] {a}", flush=True)

    # Check 3: MoGe-2 & v9 Geometric Quality on Aria (Floor n_up & Depth Shift under Squashed vs Isotropic!)
    print("\n=== CHECK 3: MoGe-2 / v9 Geometry Audit on Aria (Squashed vs Isotropic Letterbox vs GravCrop) ===", flush=True)
    geom_audit = {}
    for mode in ("squashed", "iso_letterbox", "iso_crop", "iso_grav_crop"):
        nup_low, z_meds, vert_haz, vert_clr = [], [], [], []
        for sc, d in list(aria_dicts.items())[:10]:
            inp = prepare_frontend_tensors(d, device, mode=mode)
            er = d["eval_rows"]
            sub = np.arange(0, len(er), 10)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, g = v9.encode_images(inp["rgb"][0, sub], inp["K_scaled"][0, sub], inp["R_c2w"][0, sub])
            g_np = g.float().cpu().numpy()  # (M, 10, 84, 148)
            # Lower-center strip (likely floor): rows 56..80, cols 52..96
            nup_low.append(float(np.median(g_np[:, 5, 56:80, 52:96])))
            z_meds.append(float(np.median(g_np[:, 0, 20:70, 44:104])))
            # Center strip verticality & proximity (< 4m) on hazard vs clear frames
            for k_i, fi in enumerate(sub):
                if d["valid_heading"][fi] < 1:
                    continue
                is_h = np.isfinite(er[fi, 0]) and er[fi, 0] <= H
                near_v = float(((g_np[k_i, 0, 20:68, 44:104] <= 4.0) * g_np[k_i, 6, 20:68, 44:104]).mean())
                (vert_haz if is_h else vert_clr).append(near_v)
        geom_audit[mode] = {
            "floor_n_up_med": round(float(np.median(nup_low)), 4),
            "center_z_med_m": round(float(np.median(z_meds)), 3),
            "near4m_vert_haz": round(float(np.mean(vert_haz)), 4) if vert_haz else None,
            "near4m_vert_clr": round(float(np.mean(vert_clr)), 4) if vert_clr else None,
            "vert_snr_haz_over_clr": round(float(np.mean(vert_haz) / max(np.mean(vert_clr), 1e-6)), 3) if (vert_haz and vert_clr) else None,
        }
        print(f"  [Geom Aria {mode:13s}] {geom_audit[mode]}", flush=True)

    out = {
        "wall_sec": round(time.time() - t0, 1),
        "frontend_results": frontend_results,
        "autopsies": autopsies,
        "geom_audit_aria": geom_audit,
    }
    out_p = LOCAL_OUT / "phase13_probe.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    shutil.copy2(out_p, DRIVE_OUT / out_p.name)
    print(f"\n[Phase 13 Probe DONE] in {out['wall_sec']}s -> {DRIVE_OUT / out_p.name}", flush=True)


if __name__ == "__main__":
    main()
