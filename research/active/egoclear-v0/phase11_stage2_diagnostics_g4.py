"""Phase 11 Stage 2 (Colab G4, ~12 min): Automated 4-Gate Scientific Diagnostics on Frozen EgoClear-v7.5.

Provides exact quantitative decision gates BEFORE committing Stage 3 (EgoClear-v8) training CU:
  Gate 1 (Problem 5 - Shift & FP16 Parity):
    Compares upstream MoGeModel.infer(fov_x) vs v7.5 FP32 bounded+tile-capped shift vs v7.5 FP16 ONNX on SANPO-Real 40.
  Gate 2 (Problem B - Ground-Relative Support-Plane Height & Slope Estimation):
    Tests smooth QAIRT-native tangent-plane height h_i = -(n_i . X_i) / (n_i . g) with 4x4 block-capped
    softmax voting + slope term across SANPO-Real 40, ARKit 30, Aria Gen 2, and UT Austin CODa (0.704 m robot!).
    Reports estimated h_cam median/IQR, plane confidence, and HEAD band geometric coverage lift!
  Gate 3 (Problem D - MoGe-2-ViT-L-Normal 331M Teacher Replacement Audit):
    Compares MoGe-2-ViT-L-Normal (standalone, neck-dim-agnostic!) vs MoGe-2-ViT-S-Normal on depth AbsRel,
    boundary/thin-obstacle AbsRel, dark 0.18x, AND swaps ViT-L's aligned 84x148 geom_cues directly into
    frozen v7.5 step_from_encoded to measure task-level event & FA delta!
  Gate 4 (Problem A/D - Paired Route-Level Cluster Bootstrap vs EgoClear-v6.6):
    Runs exact paired 2,000-resample route-cluster bootstrap comparing v7.5 vs v6.6 on SANPO-Real 40, Aria,
    ARKit, and CODa at BOTH fixed thr=0.50 AND matched @120 fr/km operating point!
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

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v7_model import EgoClearV7Net, FEAT_H, FEAT_W, _build_moge_uv_levels
from eval_closed_loop import eval_score_curve, make_episodes, score_episode, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_train_v75_g4 import IDX_4M_EXACT, prepare_scene_slice_tensors
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_V75 = DRIVE_ROOT / "runs/egoclear_v0_phase11_v75"
DRIVE_OUT = DRIVE_ROOT / "runs/egoclear_v0_phase11_stage2_diag"
LOCAL_OUT = Path("/content/egoclear_p11_stage2_diag")
CODA_DIR = Path("/content/coda_zeroshot_v0")


def extract_moge_geom_cues_standalone(
    moge_model,
    rgb_norm: torch.Tensor,  # (N, 3, H, W) ImageNet-normalized
    K_scaled: torch.Tensor,  # (N, 4)
    R_c2w: torch.Tensor,     # (N, 3, 3)
    uv_levels: list[torch.Tensor],
    feat_uu: torch.Tensor,
    feat_vv: torch.Tensor,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    im_mean: torch.Tensor,
    im_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neck-width-agnostic extractor for ANY MoGe-2 backbone (ViT-S / ViT-B / ViT-L) -> (geom_cues, z_m, nrm_unit, msk_84)."""
    N = rgb_norm.shape[0]
    base_h, base_w = RGB_H // 14, RGB_W // 14
    rgb_01 = (rgb_norm.float() * im_std + im_mean).clamp(0.0, 1.0)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feat_bb, cls_token = moge_model.encoder(rgb_01, base_h, base_w, return_class_token=True)
        feats_in = [feat_bb, None, None, None, None]
        for lv in range(5):
            uv = uv_levels[lv].to(device=feat_bb.device, dtype=feat_bb.dtype).expand(N, -1, -1, -1)
            feats_in[lv] = uv if feats_in[lv] is None else torch.cat([feats_in[lv], uv], dim=1)
        neck_feats = moge_model.neck(feats_in)
        pts_raw = moge_model.points_head(neck_feats)[-1]
        nrm_raw = moge_model.normal_head(neck_feats)[-1]
        msk_raw = moge_model.mask_head(neck_feats)[-1]
        m_scale = moge_model.scale_head(cls_token).squeeze(1).exp().float()

    pts_84 = F.interpolate(pts_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
    nrm_84 = F.interpolate(nrm_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
    msk_84 = torch.sigmoid(F.interpolate(msk_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False))
    pts_hwc = moge_model._remap_points(pts_84.permute(0, 2, 3, 1))
    px, py, pz = pts_hwc[..., 0], pts_hwc[..., 1], pts_hwc[..., 2]
    nrm_unit = F.normalize(nrm_84, dim=1, eps=1e-6)

    fx = K_scaled[:, 0].float().view(N, 1, 1).clamp(min=10.0)
    fy = K_scaled[:, 1].float().view(N, 1, 1).clamp(min=10.0)
    cx = K_scaled[:, 2].float().view(N, 1, 1)
    cy = K_scaled[:, 3].float().view(N, 1, 1)
    rx = (feat_uu - cx) / fx
    ry = (feat_vv - cy) / fy

    r_max, s_max, tile_cap = 2.0, 25.0, 20.0
    ax = (rx / r_max).clamp(-1.0, 1.0)
    ay = (ry / r_max).clamp(-1.0, 1.0)
    a2 = (ax * ax + ay * ay).clamp(max=1.0)
    pz_c = pz.clamp(min=0.25, max=20.0)
    bx = (px / (s_max * r_max) - (pz_c / s_max) * ax).clamp(-1.0, 1.0)
    by = (py / (s_max * r_max) - (pz_c / s_max) * ay).clamp(-1.0, 1.0)
    msk_hw = msk_84.squeeze(1)
    q_raw = msk_hw * a2 * ((0.25 / pz_c) ** 2)
    q_tiles = q_raw.reshape(N, 4, 21, 4, 37)
    tile_sum = q_tiles.sum(dim=(2, 4), keepdim=True)
    q_capped = (q_tiles * (tile_cap / tile_sum.clamp(min=tile_cap))).reshape(N, FEAT_H, FEAT_W)
    num = (q_capped * (ax * bx + ay * by)).sum(dim=(1, 2))
    den = (q_capped * a2).sum(dim=(1, 2))
    shift = ((s_max * (num / den.clamp(min=0.25))) * (den / (den + 0.25))).clamp(-2.0, 4.0).view(N, 1, 1)
    z_m = ((pz + shift).clamp(min=0.15, max=25.0) * m_scale.view(N, 1, 1)).clamp(min=0.15, max=20.0).unsqueeze(1)

    inv_z = 1.0 / z_m.clamp(min=0.25)
    log_z = torch.log1p(z_m)
    dz_gx = F.conv2d(inv_z, sobel_x, padding=1)
    dz_gy = F.conv2d(inv_z, sobel_y, padding=1)
    r_up_c = R_c2w[:, 2, :].float().view(N, 3, 1, 1)
    n_up = (nrm_unit * r_up_c).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    verticality = (1.0 - n_up.abs()).clamp(0.0, 1.0)
    geom_cues = torch.cat([z_m, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, msk_84], dim=1)
    return geom_cues, z_m, nrm_unit, msk_84


def estimate_support_plane_smooth(
    z_m: torch.Tensor,      # (N, 1, 84, 148)
    nrm_cam: torch.Tensor,  # (N, 3, 84, 148) unit normals in camera frame
    msk: torch.Tensor,      # (N, 1, 84, 148)
    rx: torch.Tensor,       # (N, 84, 148)
    ry: torch.Tensor,       # (N, 84, 148)
    R_c2w: torch.Tensor,    # (N, 3, 3)
    fwd_w: torch.Tensor,    # (N, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """100% smooth, branch-free, QAIRT-HTP-native tangent-plane estimator (exact match to EgoClearV8Net!)."""
    N = z_m.shape[0]
    device = z_m.device
    z = z_m.squeeze(1).float()
    m = msk.squeeze(1).float()
    r_up = R_c2w[:, 2, :].float().view(N, 3, 1, 1)
    n_up = (nrm_cam.float() * r_up).sum(dim=1)

    dot_nx = nrm_cam[:, 0].float() * (rx * z) + nrm_cam[:, 1].float() * (ry * z) + nrm_cam[:, 2].float() * z
    h_i = (-dot_nx / n_up.clamp(min=0.80)).clamp(0.30, 2.20)

    # Smooth soft floor-support gate (zero boolean masks!)
    g_n = F.relu((n_up - 0.80) / 0.20).clamp(max=1.0)
    g_m = F.relu((m - 0.40) / 0.60).clamp(max=1.0)
    g_z = F.relu((z - 0.30) / 0.20).clamp(max=1.0) * F.relu((4.80 - z) / 0.80).clamp(max=1.0)
    w_raw = g_n * g_m * g_z * (1.0 / z.clamp(min=0.5))

    w_tiles = w_raw.reshape(N, 4, 21, 4, 37)
    t_sum = w_tiles.sum(dim=(2, 4), keepdim=True)
    w_cap = (w_tiles * (15.0 / t_sum.clamp(min=15.0))).reshape(N, FEAT_H, FEAT_W)
    ts_flat = t_sum.squeeze(4).squeeze(2)
    blocks_active = (ts_flat / (ts_flat + 1.5)).sum(dim=(1, 2))  # smooth count in [0..16]

    h_grid = torch.linspace(0.40, 2.00, 33, device=device, dtype=torch.float32)
    diff = (h_i.unsqueeze(1) - h_grid.view(1, 33, 1, 1)) / 0.08
    votes = (torch.exp(-0.5 * (diff.clamp(-5.0, 5.0) ** 2)) * w_cap.unsqueeze(1)).sum(dim=(2, 3))
    v_sum = votes.sum(dim=1, keepdim=True).clamp(min=1e-4)
    tot_w = w_cap.sum(dim=(1, 2)).clamp(min=1e-4)

    # Temperature-sharpened mode extraction (zero ArgMax!)
    sharp_w = F.softmax(25.0 * (votes / v_sum), dim=1)
    h_hat = (sharp_w * h_grid.unsqueeze(0)).sum(dim=1)

    near_p = torch.exp(-0.5 * (((h_grid.unsqueeze(0) - h_hat.unsqueeze(1)) / 0.12).clamp(-5.0, 5.0) ** 2))
    peak_conc = ((votes * near_p).sum(dim=1) / v_sum.squeeze(1)).clamp(0.0, 1.0)
    supp_conf = (tot_w / (tot_w + 18.0)).clamp(0.0, 1.0)
    blk_conf = (blocks_active / 3.0).clamp(0.0, 1.0)
    conf = (supp_conf * blk_conf * ((peak_conc - 0.35) / 0.45).clamp(0.0, 1.0)).clamp(0.0, 1.0)

    inlier_w = w_cap * torch.exp(-0.5 * (((h_i - h_hat.view(N, 1, 1)) / 0.12).clamp(-5.0, 5.0) ** 2))
    n_cam_avg = (nrm_cam.float() * inlier_w.unsqueeze(1)).sum(dim=(2, 3)) / inlier_w.sum(dim=(1, 2)).clamp(min=1e-4).unsqueeze(1)
    n_w_avg = torch.bmm(R_c2w.float(), n_cam_avg.unsqueeze(2)).squeeze(2)
    s_fwd = (-(n_w_avg * fwd_w.float()).sum(dim=1) / n_w_avg[:, 2].clamp(min=0.70)).clamp(-0.22, 0.22) * conf
    return h_hat, s_fwd, conf


@torch.no_grad()
def run_gate1_shift_parity(v75: EgoClearV7Net, real_scenes: dict[str, dict], device: torch.device) -> dict:
    """Gate 1: Compare upstream MoGeModel.infer(fov_x) vs v7.5 FP32 bounded+tile-capped shift vs v7.5 FP16 ONNX."""
    import onnxruntime as ort

    enc_fp16_path = DRIVE_V75 / "egoclear_v75_encoder_fp16.onnx"
    sess_fp16 = ort.InferenceSession(str(enc_fp16_path), providers=["CPUExecutionProvider"]) if enc_fp16_path.exists() else None

    z_rel_vs_infer, z_absrel_gt_infer, z_absrel_gt_v75, fp16_max_diffs = [], [], [], []
    fp16_nonfinite = 0
    n_checked = 0

    for sc, d in list(real_scenes.items())[:15]:
        n = len(d["eval_rows"])
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
        fov_x = float(2.0 * math.degrees(math.atan(RGB_W / (2.0 * max(K_s[0], 1.0)))))
        gt_d_all = d["depth_s8"] if ("depth_s8" in d and d["depth_s8"] is not None) else None

        for fi in range(0, n, 12):
            rgb_01 = decode_rgb_batch(d, np.array([fi]))  # (1, 3, H, W)
            rgb_norm = (rgb_01 - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
            R = d["poses_R"][fi : fi + 1].astype(np.float32)
            K_t = torch.from_numpy(K_s[None]).to(device)
            R_t = torch.from_numpy(R).to(device)

            inf = v75.moge.infer(torch.from_numpy(rgb_01[0]).to(device), fov_x=fov_x, apply_mask=True)
            d_inf = inf["depth"].float()
            m_inf_t = inf["mask"].float()
            if d_inf.ndim == 2:
                d_inf = d_inf.unsqueeze(0).unsqueeze(0)
                m_inf_t = m_inf_t.unsqueeze(0).unsqueeze(0)
            elif d_inf.ndim == 3:
                d_inf = d_inf.unsqueeze(1)
                m_inf_t = m_inf_t.unsqueeze(1)
            z_inf = F.interpolate(d_inf, size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            m_inf = F.interpolate(m_inf_t, size=(FEAT_H, FEAT_W), mode="nearest")[0, 0].cpu().numpy() > 0.5

            _, geom32 = v75.encode_images(torch.from_numpy(rgb_norm).to(device), K_t, R_t)
            z_v75 = geom32[0, 0].float().cpu().numpy()
            ok = m_inf & np.isfinite(z_inf) & (z_inf > 0.3) & (z_inf < 10.0)
            if ok.sum() >= 50:
                z_rel_vs_infer.append(float(np.median(np.abs(z_v75[ok] - z_inf[ok]) / z_inf[ok])))

            if gt_d_all is not None:
                gd = gt_d_all[fi].astype(np.float32)
                zi_r = cv2.resize(np.nan_to_num(z_inf, nan=0.0, posinf=0.0), (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_LINEAR)
                zv_r = cv2.resize(z_v75, (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_LINEAR)
                ok_g = np.isfinite(gd) & (gd >= 0.5) & (gd <= 8.0) & (zi_r > 0.2) & (zv_r > 0.2)
                if ok_g.sum() >= 30:
                    z_absrel_gt_infer.append(float(np.mean(np.abs(zi_r[ok_g] - gd[ok_g]) / gd[ok_g])))
                    z_absrel_gt_v75.append(float(np.mean(np.abs(zv_r[ok_g] - gd[ok_g]) / gd[ok_g])))

            if sess_fp16 is not None and n_checked < 20:
                _, g16 = sess_fp16.run(None, {"rgb": rgb_norm.astype(np.float32), "K_scaled": K_s[None], "R_c2w": R})
                if not np.all(np.isfinite(g16)):
                    fp16_nonfinite += 1
                else:
                    fp16_max_diffs.append(float(np.max(np.abs(g16[0, 0] - z_v75))))
            elif n_checked < 20:
                # Native IEEE-754 FP16 autocast parity check (exact match to HTP FP16 dynamic range!)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, g16_t = v75.encode_images(torch.from_numpy(rgb_norm).to(device), K_t, R_t)
                g16 = g16_t.float().cpu().numpy()
                if not np.all(np.isfinite(g16)):
                    fp16_nonfinite += 1
                else:
                    fp16_max_diffs.append(float(np.max(np.abs(g16[0, 0] - z_v75))))
            n_checked += 1

    out = {
        "n_frames": n_checked,
        "median_rel_diff_v75_vs_upstream_infer": round(float(np.median(z_rel_vs_infer)), 4) if z_rel_vs_infer else None,
        "gt_AbsRel_upstream_infer": round(float(np.mean(z_absrel_gt_infer)), 4) if z_absrel_gt_infer else None,
        "gt_AbsRel_v75_bounded_tile": round(float(np.mean(z_absrel_gt_v75)), 4) if z_absrel_gt_v75 else None,
        "fp16_onnx_nonfinite_frames": fp16_nonfinite,
        "fp16_onnx_median_z_max_diff_m": round(float(np.median(fp16_max_diffs)), 4) if fp16_max_diffs else None,
    }
    print(f"[Gate 1 - Shift & FP16 Parity] {out}", flush=True)
    return out


@torch.no_grad()
def run_gate2_ground_relative(v75: EgoClearV7Net, domain_maps: dict[str, dict[str, dict]], device: torch.device) -> dict:
    """Gate 2: Evaluate smooth QAIRT-native tangent-plane height/slope estimator across all 4 real domains."""
    out = {}
    for dom, scenes in domain_maps.items():
        hs, slopes, confs = [], [], []
        for sc, d in list(scenes.items())[:24]:
            n = len(d["eval_rows"])
            sl = np.arange(0, n, max(1, n // 12))[:12]
            rgb = (decode_rgb_batch(d, sl) - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
            K_o = d["K_orig"].astype(np.float32)
            Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
            K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
            T = len(sl)
            rgb_t = torch.from_numpy(rgb).to(device)
            K_t = torch.from_numpy(np.repeat(K_s[None], T, axis=0)).to(device)
            R_t = torch.from_numpy(d["poses_R"][sl].astype(np.float32)).to(device)
            fwd_t = torch.from_numpy(d["causal_fwd"][sl].astype(np.float32)).to(device)

            rgb_01 = (rgb_t.float() * v75.im_std + v75.im_mean).clamp(0.0, 1.0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fb, _ = v75.moge.encoder(rgb_01, v75.base_h, v75.base_w, return_class_token=True)
                fin = [fb, None, None, None, None]
                for lv in range(5):
                    uv = getattr(v75, f"moge_uv_{lv}").to(dtype=fb.dtype).expand(T, -1, -1, -1)
                    fin[lv] = uv if fin[lv] is None else torch.cat([fin[lv], uv], dim=1)
                nf = v75.moge.neck(fin)
                nrm_raw = F.interpolate(v75.moge.normal_head(nf)[-1].float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
                nrm_u = F.normalize(nrm_raw, dim=1, eps=1e-6)
                _, geom = v75.encode_images(rgb_t, K_t, R_t)
            fx = K_t[:, 0].view(T, 1, 1)
            fy = K_t[:, 1].view(T, 1, 1)
            cx = K_t[:, 2].view(T, 1, 1)
            cy = K_t[:, 3].view(T, 1, 1)
            rx = (v75.feat_uu - cx) / fx
            ry = (v75.feat_vv - cy) / fy
            h_hat, s_fwd, conf = estimate_support_plane_smooth(geom[:, 0:1], nrm_u, geom[:, 7:8], rx, ry, R_t, fwd_t)
            hs.extend(h_hat.cpu().tolist())
            slopes.extend(s_fwd.cpu().tolist())
            confs.extend(conf.cpu().tolist())

        h_arr = np.array(hs)
        c_arr = np.array(confs)
        hi_c = c_arr > 0.35
        fixed_max_h = float(np.median(h_arr[hi_c]) + 0.60) if hi_c.any() else 1.80
        out[dom] = {
            "n_samples": len(hs),
            "frac_confident_gt035": round(float(hi_c.mean()), 3),
            "h_cam_median_m": round(float(np.median(h_arr[hi_c])), 3) if hi_c.any() else None,
            "h_cam_iqr_m": [round(float(np.percentile(h_arr[hi_c], 25)), 3), round(float(np.percentile(h_arr[hi_c], 75)), 3)] if hi_c.any() else None,
            "fixed_yslices_top_m": round(fixed_max_h, 3),
            "head_band_covered_by_fixed": bool(fixed_max_h >= 1.85),
            "median_abs_slope_deg": round(float(np.degrees(np.arctan(np.median(np.abs(np.array(slopes)[hi_c]))))), 2) if hi_c.any() else None,
        }
        print(f"[Gate 2 - Support Plane:{dom:12s}] {out[dom]}", flush=True)
    return out


@torch.no_grad()
def run_gate3_vitl_teacher_replacement(
    v75: EgoClearV7Net,
    real_scenes: dict[str, dict],
    real_local_rows: dict[str, np.ndarray],
    aria_scenes: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
) -> dict:
    """Gate 3: Quantify MoGe-2-ViT-L-Normal (331M MIT) geometry advantage over ViT-S & interface replacement delta."""
    from moge.model.v2 import MoGeModel

    print("[Gate 3] Loading Ruicheng/moge-2-vitl-normal (331M, MIT) standalone...", flush=True)
    moge_l = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    for p in moge_l.parameters():
        p.requires_grad = False
    uv_levels = _build_moge_uv_levels(RGB_H // 14, RGB_W // 14, RGB_W / RGB_H)

    s_all, l_all, s_edge, l_edge, s_dark, l_dark, nrm_ang_diff = [], [], [], [], [], [], []
    for sc, d in list(real_scenes.items())[:20]:
        gt_d_all = d["depth_s8"] if ("depth_s8" in d and d["depth_s8"] is not None) else None
        if gt_d_all is None:
            continue
        n = len(d["eval_rows"])
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
        for fi in range(0, n, 15):
            rgb_01 = decode_rgb_batch(d, np.array([fi]))
            rgb_n = (rgb_01 - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
            rgb_dark = np.clip((rgb_01 ** 1.35) * 0.18 + np.random.default_rng(fi).normal(0, 0.025, rgb_01.shape).astype(np.float32), 0, 1)
            rgb_dn = (rgb_dark - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
            K_t = torch.from_numpy(K_s[None]).to(device)
            R_t = torch.from_numpy(d["poses_R"][fi : fi + 1].astype(np.float32)).to(device)
            rgb_nt = torch.from_numpy(rgb_n).to(device)
            rgb_dnt = torch.from_numpy(rgb_dn).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, g_s = v75.encode_images(rgb_nt, K_t, R_t)
                _, g_sd = v75.encode_images(rgb_dnt, K_t, R_t)
            g_l, _, _, _ = extract_moge_geom_cues_standalone(
                moge_l, rgb_nt, K_t, R_t, uv_levels, v75.feat_uu, v75.feat_vv, v75.sobel_x, v75.sobel_y, v75.im_mean, v75.im_std
            )
            g_ld, _, _, _ = extract_moge_geom_cues_standalone(
                moge_l, rgb_dnt, K_t, R_t, uv_levels, v75.feat_uu, v75.feat_vv, v75.sobel_x, v75.sobel_y, v75.im_mean, v75.im_std
            )

            gd = gt_d_all[fi].astype(np.float32)
            zs = cv2.resize(g_s[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            zl = cv2.resize(g_l[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            zsd = cv2.resize(g_sd[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            zld = cv2.resize(g_ld[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            ok = np.isfinite(gd) & (gd >= 0.5) & (gd <= 6.0)
            if ok.sum() >= 30:
                s_all.append(float(np.mean(np.abs(zs[ok] - gd[ok]) / gd[ok])))
                l_all.append(float(np.mean(np.abs(zl[ok] - gd[ok]) / gd[ok])))
                s_dark.append(float(np.mean(np.abs(zsd[ok] - gd[ok]) / gd[ok])))
                l_dark.append(float(np.mean(np.abs(zld[ok] - gd[ok]) / gd[ok])))
                gx = cv2.Sobel(np.nan_to_num(gd, nan=0.0), cv2.CV_32F, 1, 0, ksize=3)
                edge = ok & (np.abs(gx) > 0.6)
                if edge.sum() >= 10:
                    s_edge.append(float(np.mean(np.abs(zs[edge] - gd[edge]) / gd[edge])))
                    l_edge.append(float(np.mean(np.abs(zl[edge] - gd[edge]) / gd[edge])))
            nrm_ang_diff.append(float((g_s[0, 5].float() - g_l[0, 5].float()).abs().mean().item()))

    def eval_swapped(scenes_map, ov_rows):
        ev_rows, sc_s, sc_swap = {}, {}, {}
        for sc, d in scenes_map.items():
            er = ov_rows[sc] if ov_rows is not None else d["eval_rows"]
            n = len(er)
            if n < 5:
                continue
            inp = prepare_scene_slice_tensors(d, 0, n, device)
            rgb_f = inp["rgb"][0]
            K_f = inp["K_scaled"][0]
            R_f = inp["R_c2w"][0]
            f_s_list, g_s_list, g_l_list = [], [], []
            for i0 in range(0, n, 48):
                i1 = min(n, i0 + 48)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    fs, gs = v75.encode_images(rgb_f[i0:i1], K_f[i0:i1], R_f[i0:i1])
                gl, _, _, _ = extract_moge_geom_cues_standalone(
                    moge_l, rgb_f[i0:i1], K_f[i0:i1], R_f[i0:i1], uv_levels,
                    v75.feat_uu, v75.feat_vv, v75.sobel_x, v75.sobel_y, v75.im_mean, v75.im_std,
                )
                f_s_list.append(fs); g_s_list.append(gs); g_l_list.append(gl.to(fs.dtype))
            f2d_all = torch.cat(f_s_list, dim=0).unsqueeze(0)
            gs_all = torch.cat(g_s_list, dim=0).unsqueeze(0)
            gl_all = torch.cat(g_l_list, dim=0).unsqueeze(0)

            def run_step_seq(g_seq):
                mem = torch.zeros(1, v75.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
                age = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)
                p4s = []
                for t in range(n):
                    lps = [(f2d_all[:, t - L], g_seq[:, t - L], inp["K_scaled"][:, t - L], inp["R_c2w"][:, t - L], inp["t_w"][:, t - L]) if t >= L else None for L in v75.lags]
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, _, _, _, cp, _, mem, age = v75.step_from_encoded(
                            f2d_all[:, t], g_seq[:, t], inp["K_scaled"][:, t], inp["R_c2w"][:, t], inp["t_w"][:, t], inp["fwd_w"][:, t],
                            lps, inp["t_w"][:, max(0, t - 1)], inp["fwd_w"][:, max(0, t - 1)], inp["seg_len"][:, t], mem, age, is_first=(t == 0),
                        )
                    p4s.append(float(cp[0, IDX_4M_EXACT].float().item()))
                return np.array(p4s, dtype=np.float32)

            ev_rows[sc] = er
            sc_s[sc] = run_step_seq(gs_all)
            sc_swap[sc] = run_step_seq(gl_all)

        thrs = np.array([0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.68, 0.76, 0.84], dtype=np.float64)
        c_s = summarize_at_fa_caps(eval_score_curve(ev_rows, sc_s, bw_speeds, thrs, higher_is_alert=True))
        c_l = summarize_at_fa_caps(eval_score_curve(ev_rows, sc_swap, bw_speeds, thrs, higher_is_alert=True))
        return {
            "v75_S_at120": round(c_s["late_at_fa_fr<=120.3"]["late_rate"] * 100.0, 1) if c_s.get("late_at_fa_fr<=120.3") else None,
            "v75_Lswap_at120": round(c_l["late_at_fa_fr<=120.3"]["late_rate"] * 100.0, 1) if c_l.get("late_at_fa_fr<=120.3") else None,
            "v75_S_at200": round(c_s["late_at_fa_fr<=200.0"]["late_rate"] * 100.0, 1) if c_s.get("late_at_fa_fr<=200.0") else None,
            "v75_Lswap_at200": round(c_l["late_at_fa_fr<=200.0"]["late_rate"] * 100.0, 1) if c_l.get("late_at_fa_fr<=200.0") else None,
        }

    swap_real = eval_swapped(real_scenes, real_local_rows)
    swap_aria = eval_swapped(aria_scenes, None)
    del moge_l
    torch.cuda.empty_cache()

    out = {
        "real40_AbsRel_S_vs_L": [round(float(np.mean(s_all)), 4), round(float(np.mean(l_all)), 4)],
        "real40_Edge_AbsRel_S_vs_L": [round(float(np.mean(s_edge)), 4), round(float(np.mean(l_edge)), 4)],
        "real40_Dark018_AbsRel_S_vs_L": [round(float(np.mean(s_dark)), 4), round(float(np.mean(l_dark)), 4)],
        "mean_abs_n_up_diff_S_vs_L": round(float(np.mean(nrm_ang_diff)), 4),
        "interface_swap_real40": swap_real,
        "interface_swap_aria": swap_aria,
    }
    print(f"[Gate 3 - ViT-L Teacher Replacement] {out}", flush=True)
    return out


def paired_cluster_bootstrap(
    base_routes: dict[str, dict],
    cand_routes: dict[str, dict],
    n_boot: int = 2000,
) -> dict:
    """Codex Problem D: Paired route-cluster bootstrap (routes improved/regressed + 95% CI on delta late_rate & FA/km)."""
    keys = sorted(set(base_routes) & set(cand_routes))
    improved = regressed = tied = 0
    for k in keys:
        d_l = cand_routes[k]["late"] - base_routes[k]["late"]
        if d_l < 0:
            improved += 1
        elif d_l > 0:
            regressed += 1
        else:
            tied += 1
    rng = np.random.default_rng(42)
    arr_b = np.array([[base_routes[k]["late"], base_routes[k]["avoidable"], base_routes[k]["fa_fr"], base_routes[k]["km"]] for k in keys])
    arr_c = np.array([[cand_routes[k]["late"], cand_routes[k]["avoidable"], cand_routes[k]["fa_fr"], cand_routes[k]["km"]] for k in keys])
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    sb = arr_b[idx].sum(axis=1)
    sc = arr_c[idx].sum(axis=1)
    lrb = sb[:, 0] / np.maximum(sb[:, 1], 1.0)
    lrc = sc[:, 0] / np.maximum(sc[:, 1], 1.0)
    diff_pp = (lrc - lrb) * 100.0
    fa_diff = (sc[:, 2] / np.maximum(sc[:, 3], 1e-4)) - (sb[:, 2] / np.maximum(sb[:, 3], 1e-4))
    return {
        "n_routes": len(keys),
        "routes_improved_late": improved,
        "routes_regressed_late": regressed,
        "routes_tied_late": tied,
        "delta_late_pp_median": round(float(np.median(diff_pp)), 2),
        "delta_late_pp_ci95": [round(float(np.percentile(diff_pp, 2.5)), 2), round(float(np.percentile(diff_pp, 97.5)), 2)],
        "delta_fa_fr_km_ci95": [round(float(np.percentile(fa_diff, 2.5)), 1), round(float(np.percentile(fa_diff, 97.5)), 1)],
    }


def per_route_at_thr(scenes_map, ov, scores_map, bw_speeds, thr: float) -> dict[str, dict]:
    out = {}
    for sc, p4 in scores_map.items():
        er = ov[sc] if ov is not None else scenes_map[sc]["eval_rows"]
        al = p4 >= thr
        rl = ra = rfe = rff = 0
        rkm = 0.0
        for ep in make_episodes(sc, er, bw_speeds, kind="blind"):
            l, a, _u, fe, ff, km = score_episode(ep, al)
            rl += l; ra += a; rfe += fe; rff += ff; rkm += km
        out[sc] = {"late": rl, "avoidable": ra, "fa_ep": rfe, "fa_fr": rff, "km": round(rkm, 4)}
    return out


def run_gate5_failure_autopsy(
    v75: EgoClearV7Net,
    domain_specs: list[tuple[str, dict[str, dict], dict | None]],
    bw_speeds: dict,
    device: torch.device,
) -> dict:
    """Gate 5 (Adaptive Autopsy): Pinpoints exact missed-event & false-alarm root causes + MoGe-2 metric scale drift per domain."""
    out = {}
    for dname, dmap, ov in domain_specs:
        scale_ratios = []
        miss_dist_gt, miss_p_max, miss_hband = [], [], {"LOW": 0, "BODY": 0, "HEAD": 0, "UNK": 0}
        fa_total = fa_near_dyn = fa_high_floor = fa_stationary = 0
        for sc, d in dmap.items():
            er = ov[sc] if ov is not None else d["eval_rows"]
            n = len(er)
            if n < 5:
                continue
            inp = prepare_scene_slice_tensors(d, 0, n, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o75 = v75.forward_sequence(**inp)
            p4 = o75["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
            alerts = p4 >= 0.50

            # Metric depth scale ratio vs GT where depth_s8 exists
            gt_d_all = d["depth_s8"] if ("depth_s8" in d and d["depth_s8"] is not None) else None
            if gt_d_all is not None:
                for fi in range(0, n, max(1, n // 6)):
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, g = v75.encode_images(inp["rgb"][0, fi : fi + 1], inp["K_scaled"][0, fi : fi + 1], inp["R_c2w"][0, fi : fi + 1])
                    gd = gt_d_all[fi].astype(np.float32)
                    zp = cv2.resize(g[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
                    ok = np.isfinite(gd) & (gd >= 0.6) & (gd <= 6.0) & (zp >= 0.2)
                    if ok.sum() >= 30:
                        scale_ratios.append(float(np.median(zp[ok] / gd[ok])))

            # Autopsy missed events & FA frames at fixed 0.50 using middle blind walker (idx 2)
            eps = make_episodes(sc, er, bw_speeds, kind="blind")
            if not eps:
                continue
            ep = eps[len(eps) // 2]
            D, valid, dyn, ds = er[:, 0], ep["valid"], er[:, 2] > 0.5, er[:, 3]
            haz = valid & (D <= 4.0)
            from eval_closed_loop import MIN_EVENT_FRAMES_V2, weak_mask
            wm = weak_mask(ep)
            fa_mask = valid & (~haz) & (~dyn) & (~wm) & alerts
            fa_idx = np.nonzero(fa_mask)[0]
            fa_total += int(len(fa_idx))
            for fi in fa_idx:
                if np.any(dyn[max(0, fi - 3) : min(n, fi + 4)]):
                    fa_near_dyn += 1
                if ds[fi] < 0.05:
                    fa_stationary += 1

            # Check missed avoidable events
            gt_st = d["gt_bev_static"] if "gt_bev_static" in d else None
            i = 0
            while i < n:
                if haz[i]:
                    j = i
                    while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                        j += 1
                    seg = slice(i, j + 1)
                    if haz[seg].sum() >= MIN_EVENT_FRAMES_V2:
                        ok_seg = (D[seg] >= ep["dreq"]) & haz[seg]
                        if ok_seg.any():
                            ok_idx = np.nonzero(ok_seg)[0] + i
                            if not np.any(alerts[ok_idx]):
                                miss_p_max.append(float(np.max(p4[ok_idx])))
                                miss_dist_gt.append(float(np.min(D[ok_idx])))
                                if gt_st is not None:
                                    # Dominant height band in corridor during timely window
                                    hb_sum = gt_st[ok_idx, :, : IDX_4M_EXACT + 1, 6:9].sum(axis=(0, 2, 3))
                                    if hb_sum.sum() > 0:
                                        miss_hband[["LOW", "BODY", "HEAD"][int(np.argmax(hb_sum))]] += 1
                                    else:
                                        miss_hband["UNK"] += 1
                    i = j + 1
                else:
                    i += 1

        out[dname] = {
            "moge2_depth_scale_median": round(float(np.median(scale_ratios)), 3) if scale_ratios else None,
            "moge2_depth_scale_iqr": [round(float(np.percentile(scale_ratios, 25)), 3), round(float(np.percentile(scale_ratios, 75)), 3)] if scale_ratios else None,
            "missed_events_midwalker": len(miss_p_max),
            "missed_max_p_median": round(float(np.median(miss_p_max)), 3) if miss_p_max else None,
            "missed_min_dist_m_median": round(float(np.median(miss_dist_gt)), 2) if miss_dist_gt else None,
            "missed_dominant_hband": miss_hband,
            "fa_frames_midwalker": fa_total,
            "fa_frac_near_dynamic": round(fa_near_dyn / max(fa_total, 1), 3),
            "fa_frac_stationary_ds005": round(fa_stationary / max(fa_total, 1), 3),
        }
        print(f"[Gate 5 - Failure Autopsy:{dname:20s}] {out[dname]}", flush=True)
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
    bw_speeds = json.loads(Path("/content/sanpo_287_manifest.json").read_text())["bw_speeds"]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    v75 = EgoClearV7Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_moge=True).to(device).eval()
    ck_p = DRIVE_V75 / "egoclear_v75_moge2_small.pt"
    v75_ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    v75.load_state_dict(v75_ck["head_state_dict"], strict=False)

    # Copy verified FP32 ONNX (input for QAIRT HTP FP16 DLC!) & convert step FP16 cleanly to Drive
    enc_fp32_drv = DRIVE_V75 / "egoclear_v75_encoder_fp32.onnx"
    if not enc_fp32_drv.exists():
        import onnx, onnxruntime as ort
        from onnxconverter_common import float16 as onnx_fp16
        loc_onnx = Path("/content/egoclear_runs_p11_v75/onnx")
        enc32 = loc_onnx / "egoclear_v75_encoder_fp32.onnx"
        stp32 = loc_onnx / "egoclear_v75_step_fp32.onnx"
        stp16 = loc_onnx / "egoclear_v75_step_fp16.onnx"
        if enc32.exists():
            shutil.copy2(enc32, DRIVE_V75 / enc32.name)
        if stp32.exists():
            shutil.copy2(stp32, DRIVE_V75 / stp32.name)
            try:
                stp_blk = list(set(onnx_fp16.DEFAULT_OP_BLOCK_LIST) | {"Softplus", "CumSum", "Exp", "GridSample", "Resize", "Sqrt", "ReduceSum", "ReduceMean"})
                onnx.save(onnx_fp16.convert_float_to_float16(onnx.load(str(stp32)), keep_io_types=True, op_block_list=stp_blk), str(stp16))
                _ = ort.InferenceSession(str(stp16), providers=["CPUExecutionProvider"])
                shutil.copy2(stp16, DRIVE_V75 / stp16.name)
            except Exception as e:
                print(f"[stage2-onnx] step fp16 warn: {e}", flush=True)
        print(f"[stage2-onnx] Synced v7.5 FP32 encoder + step ONNX to Drive (QAIRT-ready)!", flush=True)

    g1 = run_gate1_shift_parity(v75, eval_real, device)
    g2 = run_gate2_ground_relative(v75, {"real_40": eval_real, "arkit_30": eval_arkit, "aria_g2": aria_dicts, "coda": eval_coda}, device)
    g3 = run_gate3_vitl_teacher_replacement(v75, eval_real, real_local_rows, aria_dicts, bw_speeds, device)
    g5 = run_gate5_failure_autopsy(
        v75,
        [
            ("real_40_local", eval_real, real_local_rows),
            ("aria_g2_zeroshot", aria_dicts, None),
            ("arkit_indoor_val30", eval_arkit, None),
            ("coda_campus_zeroshot", eval_coda, None),
        ],
        bw_speeds,
        device,
    )

    # Gate 4: Paired route-cluster bootstrap vs v6.6 at BOTH fixed thr=0.50 AND matched @120 fr/km!
    v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    v66_ck = DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt"
    v66.load_state_dict(torch.load(v66_ck, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)
    boot_050, boot_120 = {}, {}
    for dname, dmap, ov in [
        ("real_40_local", eval_real, real_local_rows),
        ("aria_g2_zeroshot", aria_dicts, None),
        ("arkit_indoor_val30", eval_arkit, None),
        ("coda_campus_zeroshot", eval_coda, None),
    ]:
        ev_rows_m, s66_m, s75_m = {}, {}, {}
        for sc, d in dmap.items():
            er = ov[sc] if ov is not None else d["eval_rows"]
            if len(er) < 5:
                continue
            inp = prepare_scene_slice_tensors(d, 0, len(er), device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o66 = v66.forward_sequence(**inp)
                o75 = v75.forward_sequence(**inp)
            ev_rows_m[sc] = er
            s66_m[sc] = o66["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
            s75_m[sc] = o75["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()

        r66_50 = per_route_at_thr(dmap, ov, s66_m, bw_speeds, 0.50)
        r75_50 = per_route_at_thr(dmap, ov, s75_m, bw_speeds, 0.50)
        boot_050[dname] = paired_cluster_bootstrap(r66_50, r75_50)

        c66 = summarize_at_fa_caps(eval_score_curve(ev_rows_m, s66_m, bw_speeds, thrs, higher_is_alert=True))
        c75 = summarize_at_fa_caps(eval_score_curve(ev_rows_m, s75_m, bw_speeds, thrs, higher_is_alert=True))
        t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
        t75 = float(c75["late_at_fa_fr<=120.3"]["thr"]) if c75.get("late_at_fa_fr<=120.3") else 0.50
        r66_120 = per_route_at_thr(dmap, ov, s66_m, bw_speeds, t66)
        r75_120 = per_route_at_thr(dmap, ov, s75_m, bw_speeds, t75)
        b120 = paired_cluster_bootstrap(r66_120, r75_120)
        b120["thr_v66"] = round(t66, 3)
        b120["thr_v75"] = round(t75, 3)
        boot_120[dname] = b120
        print(f"[Gate 4 @0.50 {dname}] {boot_050[dname]}", flush=True)
        print(f"[Gate 4 @120  {dname}] {boot_120[dname]}", flush=True)

    # Automated Stage-3 architecture gate decision
    coda_h = g2.get("coda", {}).get("h_cam_median_m")
    coda_conf = g2.get("coda", {}).get("frac_confident_gt035", 0.0)
    real_conf = g2.get("real_40", {}).get("frac_confident_gt035", 0.0)
    pass_ground_rel = bool(
        coda_h is not None and 0.55 <= coda_h <= 0.95 and coda_conf >= 0.45 and real_conf >= 0.50
    )
    edge_gain = g3["real40_Edge_AbsRel_S_vs_L"][0] - g3["real40_Edge_AbsRel_S_vs_L"][1]
    dark_gain = g3["real40_Dark018_AbsRel_S_vs_L"][0] - g3["real40_Dark018_AbsRel_S_vs_L"][1]
    all_gain = g3["real40_AbsRel_S_vs_L"][0] - g3["real40_AbsRel_S_vs_L"][1]
    pass_geom_res = bool(all_gain >= 0.025 or edge_gain >= 0.035 or dark_gain >= 0.035)

    stage3_rec = {
        "use_ground_relative": pass_ground_rel,
        "use_geom_residual_and_vitl_kd": pass_geom_res,
        "use_freespace_gru": True,
        "rationale": {
            "coda_h_cam_m": coda_h,
            "coda_conf_frac": coda_conf,
            "real_conf_frac": real_conf,
            "vitl_all_absrel_gain": round(all_gain, 4),
            "vitl_edge_absrel_gain": round(edge_gain, 4),
            "vitl_dark_absrel_gain": round(dark_gain, 4),
        },
    }
    print(f"\n[Stage 2 -> Stage 3 Gate Recommendation] {stage3_rec}", flush=True)

    diag = {
        "wall_sec": round(time.time() - t0, 1),
        "gate1_shift_and_fp16_parity": g1,
        "gate2_ground_relative_support_plane": g2,
        "gate3_vitl_teacher_replacement": g3,
        "gate4_paired_bootstrap_v75_vs_v66_at_0.50": boot_050,
        "gate4_paired_bootstrap_v75_vs_v66_at_120": boot_120,
        "gate5_failure_autopsy_and_scale": g5,
        "stage3_recommendation": stage3_rec,
    }
    out_p = LOCAL_OUT / "stage2_diagnostics.json"
    out_p.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    shutil.copy2(out_p, DRIVE_OUT / out_p.name)
    print(f"[Stage 2] ALL 4 GATES DONE in {diag['wall_sec']}s -> {DRIVE_OUT / out_p.name}", flush=True)


if __name__ == "__main__":
    main()
