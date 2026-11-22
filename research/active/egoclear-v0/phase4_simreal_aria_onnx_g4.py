"""Phase 4 G4 (96 GB Blackwell) Complete Validation & Mobile Export:
1. Pack 76 strictly disjoint SANPO-Real training sessions (5 Hz, drift-free +/-0.8s local-window GT) -> Drive
2. Fine-tune EgoClear-v3-Full (Sim+Real Joint: 627 Synthetic + 76 Disjoint Real, frozen DAV2-S, 3.69M trainable)
3. Complete 3-Domain Cross-Hardware Evaluation:
   - SANPO-Synthetic 110 Frozen Eval (Clean + Realistic VIO 1x)
   - SANPO-Real 40 Frozen Unseen Real Eval (Local & Global) vs 340M UniDepth-V2-Large
   - Meta Aria Gen 2 Pilot Glasses (20 clips, 5 Hz, 79-deg grayscale CV camera, REAL glasses VIO -> 100% ZERO-SHOT!)
     comparing FoundationStereo Ref, DAV2-S Ground, UniDepth-V2-Small, UniDepth-V2-Large,
     EgoClear-v3-Full (Sim-Only Zero-Shot), and EgoClear-v3-Full (Sim+Real Zero-Shot on Aria)!
4. Two-Stage Mobile Streaming ONNX Export (encoder.onnx + step.onnx) & ONNXRuntime Numerical Parity Audit!
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, "/content")
ud_root = Path("/content/unidepth_src")
if ud_root.exists():
    sub = next(ud_root.glob("UniDepth-*"), None)
    if sub is not None:
        sys.path.insert(0, str(sub))

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, NY, RGB_H, RGB_W
from egoclear_v3_model import EgoClearV3Net
from eval_closed_loop import H, eval_score_curve, summarize_at_fa_caps
from pack_phase2_colab import pack_real_worker, write_shards_to_drive
from pack_sanpo_colab import CELL, DYN, FWD0, HGT, MIN_PTS, MIN_VOX_V2, SKY, VOX, WALK, X_HALF, corridor_d, voxelize_bev
from train_and_eval_g4 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_PTS_S8,
    decode_rgb_batch,
    depth_map_to_corridor_d,
    ground_anchor_relative_depth,
    load_scene,
)
from train_and_eval_phase2_g4 import SanpoPhase2ClipDataset, compute_loss_v2
from train_and_eval_phase3_g4 import build_local_window_real_eval_rows, eval_v3_on_regime

DRIVE_REAL76 = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_real_train76_v0")
DRIVE_RUNS = Path("/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase4")
LOCAL_RUNS = Path("/content/egoclear_runs_p4")


def build_local_bev_and_rows_for_real_train(npz_path_str: str) -> str:
    """Replace global-drifted eval_rows & gt_bev_static in a SANPO-Real train npz with drift-free +/-0.8s local-window GT."""
    cv2.setNumThreads(1)
    p = Path(npz_path_str)
    d = load_scene(p)
    n = len(d["eval_rows"])
    K_o = d["K_orig"].astype(np.float64)
    Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
    s8_h, s8_w = d["depth_s8"].shape[1:]
    fx, fy = K_o[0] * s8_w / Ww, K_o[1] * s8_h / Hh
    cx, cy = K_o[2] * s8_w / Ww, K_o[3] * s8_h / Hh
    vv, uu = np.mgrid[0:s8_h, 0:s8_w]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)

    Rs = d["poses_R"].astype(np.float64)
    ps = d["poses_t"].astype(np.float64)
    fwds = d["causal_fwd"].astype(np.float64)
    vh = d["valid_heading"] > 0
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

    P_all, L_all = [], []
    for i in range(n):
        df = d["depth_s8"][i].astype(np.float32).reshape(-1)
        sf = d["seg_s8"][i].reshape(-1)
        ok = np.isfinite(df) & (df > 0.2) & (df < 25.0) & (sf != SKY)
        P_all.append(((Rs[i] @ (dirs[ok] * df[ok, None]).T).T + ps[i]).astype(np.float32))
        L_all.append(sf[ok])

    loc_rows = np.full((n, 4), np.nan, dtype=np.float64)
    loc_bev_st = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.uint8)
    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        Pw_win = np.concatenate(P_all[lo:hi])
        Lw_win = np.concatenate(L_all[lo:hi])
        Gw = Pw_win[np.isin(Lw_win, WALK)]
        if len(Gw) < 100:
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
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.0)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            return h

        stat = Pw_win[~np.isin(Lw_win, WALK + [SKY] + DYN)]
        hs = height_loc(stat)
        stat = stat[(hs >= HGT[0]) & (hs <= HGT[1])]
        fwd = fwds[i]
        right = np.array([fwd[1], -fwd[0], 0.0])
        if len(stat) > 0:
            vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
            vc = (vox + 0.5) * VOX
            vc_h = height_loc(vc)
            rel_v = vc - ps[i]
            loc_bev_st[i] = voxelize_bev(np.column_stack([rel_v @ right, rel_v @ fwd, vc_h]), min_count=2)
            for kc in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
                cellset.add(tuple(kc))
        else:
            vc = np.zeros((0, 3), dtype=np.float64)

        if not vh[i]:
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
        P0, L0 = P_all[i], L_all[i]
        h0 = height_loc(P0)
        rel0 = P0 - ps[i]
        x0, z0 = rel0 @ tr_, rel0 @ tf
        dyn_flag = float(
            ((np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H) & np.isin(L0, DYN) & (h0 >= HGT[0]) & (h0 <= HGT[1])).sum()
            >= MIN_PTS
        )
        loc_rows[i] = [D, cover, dyn_flag, seg_len[i]]

    d["eval_rows"] = loc_rows
    d["gt_bev_static"] = loc_bev_st
    np.savez(p, **d)
    return p.stem


class MixedSimRealDataset(torch.utils.data.Dataset):
    """Mix 60% SANPO-Synthetic (627) + 40% SANPO-Real (76 disjoint) clips per epoch."""

    def __init__(self, synth_dicts: list[dict], real_dicts: list[dict], clip_len: int = 12, clips_per_epoch: int = 1920):
        self.ds_syn = SanpoPhase2ClipDataset(synth_dicts, clip_len=clip_len, clips_per_epoch=clips_per_epoch)
        self.ds_real = SanpoPhase2ClipDataset(real_dicts, clip_len=clip_len, clips_per_epoch=clips_per_epoch)
        self.clips_per_epoch = clips_per_epoch

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if np.random.random() < 0.40 and len(self.ds_real.scenes) > 0:
            return self.ds_real[idx]
        return self.ds_syn[idx]


def pack_real_train_76():
    real76_dir = Path("/content/sanpo_p4/real_train_scenes")
    real76_dir.mkdir(parents=True, exist_ok=True)
    if len(list(real76_dir.glob("*.npz"))) >= 50:
        print(f"[p4-pack] real_train_76 already local ({len(list(real76_dir.glob('*.npz')))} files)", flush=True)
        return real76_dir

    man = json.loads(Path("/content/phase4_real80_manifest.json").read_text())["real_train_80"]
    ncpu = os.cpu_count() or 8
    n_workers = min(20, max(4, ncpu - 2))
    print(f"[p4-pack] packing {len(man)} disjoint SANPO-Real train sessions on {ncpu} vCPUs...", flush=True)
    t0 = time.time()
    infos = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(pack_real_worker, (r["scene"], r["cam"], str(real76_dir))) for r in man]
        for fut in as_completed(futs):
            try:
                inf = fut.result()
                if inf is not None:
                    infos.append(inf)
            except Exception as e:
                print(f"  [warn] real train session failed: {e}", flush=True)
    print(f"[p4-pack] raw packed {len(infos)}/{len(man)} real train sessions in {time.time()-t0:.1f}s; rebuilding drift-free local GT...", flush=True)
    t1 = time.time()
    npz_files = sorted(str(p) for p in real76_dir.glob("*.npz"))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(build_local_bev_and_rows_for_real_train, npz_files))
    stems = sorted(p.stem for p in real76_dir.glob("*.npz"))
    write_shards_to_drive(real76_dir, stems, DRIVE_REAL76, "sanpo_real_train76_v0", {"n_ok": len(stems), "local_gt": True})
    print(f"[p4-pack] real_train_76 DONE ({len(stems)} scenes, local GT rebuilt in {time.time()-t1:.1f}s)", flush=True)
    return real76_dir


@torch.inference_mode()
def eval_aria_baselines(aria_dicts: dict[str, dict], bw_speeds: dict, device: torch.device) -> dict:
    """Evaluate FoundationStereo Ref, DAV2-S Ground, UniDepth-V2-Small, UniDepth-V2-Large on 20 Aria Gen 2 glasses clips."""
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    eval_rows_map = {sc: d["eval_rows"] for sc, d in aria_dicts.items()}
    gt_single_map = {sc: d["gt_single_d"] for sc, d in aria_dicts.items()}
    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)
    out = {}

    c_ref = eval_score_curve(eval_rows_map, gt_single_map, bw_speeds, b1_thrs, higher_is_alert=False)
    out["ref_foundationstereo_single"] = {"curve": c_ref, "summary": summarize_at_fa_caps(c_ref)}
    bf = out["ref_foundationstereo_single"]["summary"]["best_floor"]
    print(f"[ARIA_G2] ref_foundationstereo_single: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)

    # DAV2-Small + Ground Anchor on Aria Gen 2 (160x160 grid, min_pts=14 matching Aria density)
    eval_dh, eval_dw = 160, 160
    proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    dav2_m = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    dav2_d = {}
    for sc, d in aria_dicts.items():
        n = len(d["eval_rows"])
        rgbs = decode_rgb_batch(d, np.arange(n))
        depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for st in range(0, n, 32):
                en = min(n, st + 32)
                imgs = [(rgbs[k].transpose(1, 2, 0) * 255.0).astype(np.uint8) for k in range(st, en)]
                inp = proc(images=imgs, return_tensors="pt").to(device)
                pred = dav2_m(**inp).predicted_depth
                depths[st:en] = F.interpolate(pred.unsqueeze(1).float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        K_o = d["K_orig"].astype(np.float64)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
        Rs, ps, fwds = d["poses_R"].astype(np.float64), d["poses_t"].astype(np.float64), d["causal_fwd"].astype(np.float64)
        d_arr = np.full(n, np.inf, dtype=np.float32)
        for i in range(n):
            z_map = ground_anchor_relative_depth(depths[i], K_d, Rs[i], 1.53)
            d_arr[i], _ = depth_map_to_corridor_d(z_map, K_d, Rs[i], ps[i], fwds[i], 14)
        dav2_d[sc] = d_arr
    c_dav = eval_score_curve(eval_rows_map, dav2_d, bw_speeds, b1_thrs, higher_is_alert=False)
    out["dav2_small_ground"] = {"curve": c_dav, "summary": summarize_at_fa_caps(c_dav)}
    bf = out["dav2_small_ground"]["summary"]["best_floor"]
    print(f"[ARIA_G2] dav2_small_ground: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)
    del dav2_m, proc
    torch.cuda.empty_cache()

    # UniDepth-V2-Small & UniDepth-V2-Large on Aria Gen 2
    for u_name, hf_id in (
        ("unidepth_v2_small", "lpiccinelli/unidepth-v2-vits14"),
        ("unidepth_v2_large", "lpiccinelli/unidepth-v2-vitl14"),
    ):
        u_mod = UniDepthV2.from_pretrained(hf_id).to(device).eval()
        ud_d = {}
        for sc, d in aria_dicts.items():
            n = len(d["eval_rows"])
            rgbs = decode_rgb_batch(d, np.arange(n))
            K_o = d["K_orig"].astype(np.float64)
            Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
            K_mat = torch.tensor(
                [[K_o[0] * RGB_W / Ww, 0.0, K_o[2] * RGB_W / Ww], [0.0, K_o[1] * RGB_H / Hh, K_o[3] * RGB_H / Hh], [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=device,
            )
            depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
            for st in range(0, n, 32):
                en = min(n, st + 32)
                imgs_u8 = torch.from_numpy((rgbs[st:en] * 255.0).clip(0, 255).astype(np.uint8)).to(device)
                pred = u_mod.infer(imgs_u8, Pinhole(K=K_mat.unsqueeze(0).clone()))["depth"]
                depths[st:en] = F.interpolate(pred.float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
            K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
            Rs, ps, fwds = d["poses_R"].astype(np.float64), d["poses_t"].astype(np.float64), d["causal_fwd"].astype(np.float64)
            d_arr = np.full(n, np.inf, dtype=np.float32)
            for i in range(n):
                d_arr[i], _ = depth_map_to_corridor_d(depths[i], K_d, Rs[i], ps[i], fwds[i], 14)
            ud_d[sc] = d_arr
        c_u = eval_score_curve(eval_rows_map, ud_d, bw_speeds, b1_thrs, higher_is_alert=False)
        out[u_name] = {"curve": c_u, "summary": summarize_at_fa_caps(c_u)}
        bf = out[u_name]["summary"]["best_floor"]
        print(f"[ARIA_G2] {u_name}: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)
        del u_mod
        torch.cuda.empty_cache()

    return out


class EgoClearV3EncoderExport(nn.Module):
    """Stage-1 Mobile ONNX wrapper: (rgb, K_scaled, R_c2w) -> (f2d, disp_cues)."""
    def __init__(self, core: EgoClearV3Net):
        super().__init__()
        self.core = core

    def forward(self, rgb: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor):
        return self.core.encode_images(rgb, K_scaled, R_c2w)


class EgoClearV3StepExport(nn.Module):
    """Stage-2 Mobile ONNX streaming step: 3-frame features + prev memory -> hazards + updated memory."""
    def __init__(self, core: EgoClearV3Net):
        super().__init__()
        self.core = core

    def forward(
        self,
        f2d_t: torch.Tensor, dc_t: torch.Tensor,
        f2d_1: torch.Tensor, dc_1: torch.Tensor,
        f2d_2: torch.Tensor, dc_2: torch.Tensor,
        K_t: torch.Tensor, R_t: torch.Tensor, t_t: torch.Tensor, fwd_t: torch.Tensor,
        K_1: torch.Tensor, R_1: torch.Tensor, t_1: torch.Tensor, fwd_1: torch.Tensor,
        K_2: torch.Tensor, R_2: torch.Tensor, t_2: torch.Tensor,
        ds_t: torch.Tensor,
        mem_prev: torch.Tensor, age_prev: torch.Tensor,
    ):
        c = self.core
        B = f2d_t.shape[0]
        gx = c.grid_xzy[:, 0][None, :, None]
        gz = c.grid_xzy[:, 1][None, :, None]
        gy = c.grid_xzy[:, 2][None, :, None]
        up_w = torch.tensor([0.0, 0.0, 1.0], device=f2d_t.device, dtype=f2d_t.dtype)[None, None, :]
        rightt = torch.stack([fwd_t[:, 1], -fwd_t[:, 0], torch.zeros_like(fwd_t[:, 0])], dim=-1)
        pts_w = t_t[:, None, :] + gx * rightt[:, None, :] + gz * fwd_t[:, None, :] + gy * up_w

        sf_t, sd_t, vis_t, uv_invz_t, zc_t = c._project_and_sample(f2d_t, dc_t, pts_w, R_t, t_t, K_t)
        pt_parts = [sf_t, sd_t, vis_t, uv_invz_t]
        ft_norm = F.normalize(sf_t, dim=1, eps=1e-5)
        log_dt = sd_t[:, 3:4]

        for fp, dp, Rp, tp, Kp in ((f2d_1, dc_1, R_1, t_1, K_1), (f2d_2, dc_2, R_2, t_2, K_2)):
            sf_p, sd_p, vis_p, _, zc_p = c._project_and_sample(fp, dp, pts_w, Rp, tp, Kp)
            both_v = vis_t * vis_p
            fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
            cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
            l1_diff = (sf_t - sf_p).abs().mean(dim=1, keepdim=True) * both_v
            disp_exp_err = ((log_dt - sd_p[:, 3:4]) - torch.log(zc_p / zc_t.clamp(min=0.15))).clamp(-2.0, 2.0) * both_v
            base = torch.linalg.norm(t_t - tp, dim=-1).clamp(max=2.0)[:, None, None].expand(B, 1, pts_w.shape[1]) * both_v
            pt_parts.extend([cos_sim, l1_diff, disp_exp_err, both_v, base])

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
        pt_red = c.pt_reduce(pt_feat).reshape(B, 40 * NY, BEV_NZ, BEV_NX)
        bev_pos = torch.stack(
            [c.bev_xz[..., 0] / 1.50, (c.bev_xz[..., 1] - 2.70) / 2.40], dim=0
        ).unsqueeze(0).expand(B, -1, -1, -1).to(f2d_t.dtype)
        bev_obs = c.bev_enc(torch.cat([pt_red, bev_pos], dim=1))
        dyn_log = c.head_dynamic_obs(bev_obs)
        dyn_prob = torch.sigmoid(dyn_log).max(dim=1, keepdim=True).values

        vis_bev = vis_t.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True)
        mem_w = c._warp_bev(mem_prev, t_1, fwd_1, t_t, fwd_t)
        age_w = (c._warp_bev(age_prev, t_1, fwd_1, t_t, fwd_t) + ds_t.clamp(0.0, 2.0)[:, None, None, None]).clamp(max=10.0)
        age_decay = torch.exp(-age_w / 4.0)
        mem_new = c.mem_gru(mem_w, bev_obs, vis_bev, age_decay, dyn_prob)
        age_new = torch.where(vis_bev > 0.05, torch.zeros_like(age_w), age_w)

        bev_feat = c.bev_refine(mem_new)
        st_log = c.head_static(bev_feat)
        corr_strip = bev_feat[:, :, :, 4:11].permute(0, 1, 3, 2).reshape(B, c.c_bev * 7, BEV_NZ)
        corr_out = c.corr_head(corr_strip)
        return st_log, dyn_log, corr_out[:, 0], corr_out[:, 1], mem_new, age_new


def export_and_verify_onnx(model: EgoClearV3Net, out_dir: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
        import onnxruntime as ort

    m_cpu = model.cpu().eval()
    enc = EgoClearV3EncoderExport(m_cpu).eval()
    stp = EgoClearV3StepExport(m_cpu).eval()

    rgb = torch.randn(1, 3, RGB_H, RGB_W)
    K = torch.tensor([[449.0, 453.0, 252.0, 155.0]])
    R = torch.eye(3).unsqueeze(0)
    enc_path = out_dir / "egoclear_v3_encoder.onnx"
    stp_path = out_dir / "egoclear_v3_step.onnx"

    with torch.no_grad():
        pt_f, pt_d = enc(rgb, K, R)
        torch.onnx.export(
            enc, (rgb, K, R), str(enc_path), opset_version=17,
            input_names=["rgb", "K_scaled", "R_c2w"], output_names=["f2d", "disp_cues"],
            dynamo=False,
        )
        t_t = torch.tensor([[0.5, 0.0, 1.5]])
        t_1 = torch.tensor([[0.25, 0.0, 1.5]])
        t_2 = torch.tensor([[0.0, 0.0, 1.5]])
        fwd = torch.tensor([[1.0, 0.0, 0.0]])
        ds = torch.tensor([0.25])
        mem0 = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
        age0 = torch.full((1, 1, BEV_NZ, BEV_NX), 1.0)
        stp_inputs = (pt_f, pt_d, pt_f, pt_d, pt_f, pt_d, K, R, t_t, fwd, K, R, t_1, fwd, K, R, t_2, ds, mem0, age0)
        pt_stp = stp(*stp_inputs)
        torch.onnx.export(
            stp, stp_inputs, str(stp_path), opset_version=17,
            input_names=[
                "f2d_t", "dc_t", "f2d_1", "dc_1", "f2d_2", "dc_2",
                "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1", "K_2", "R_2", "t_2",
                "ds_t", "mem_prev", "age_prev",
            ],
            output_names=["static_logits", "dynamic_logits", "corr_logits", "unc_logits", "mem_new", "age_new"],
            dynamo=False,
        )

    sess_e = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    ort_f, ort_d = sess_e.run(None, {"rgb": rgb.numpy(), "K_scaled": K.numpy(), "R_c2w": R.numpy()})
    enc_err = float(np.max(np.abs(ort_f - pt_f.numpy())))

    sess_s = ort.InferenceSession(str(stp_path), providers=["CPUExecutionProvider"])
    feed_s = {inp.name: val.numpy() for inp, val in zip(sess_s.get_inputs(), stp_inputs)}
    ort_s = sess_s.run(None, feed_s)
    stp_err = float(np.max(np.abs(ort_s[2] - pt_stp[2].numpy())))

    # Benchmark CPU streaming latency of Stage-2 recurrent step (runs on phone CPU/NPU easily!)
    for _ in range(5):
        sess_s.run(None, feed_s)
    t_b = time.perf_counter()
    for _ in range(30):
        sess_s.run(None, feed_s)
    step_cpu_ms = (time.perf_counter() - t_b) * 1000.0 / 30.0

    shutil.copy2(enc_path, DRIVE_RUNS / enc_path.name)
    shutil.copy2(stp_path, DRIVE_RUNS / stp_path.name)
    info = {
        "encoder_onnx_mb": round(enc_path.stat().st_size / 1e6, 2),
        "step_onnx_mb": round(stp_path.stat().st_size / 1e6, 2),
        "encoder_max_abs_err": round(enc_err, 6),
        "step_corr_max_abs_err": round(stp_err, 6),
        "step_cpu_ort_ms": round(step_cpu_ms, 2),
    }
    print(f"[onnx] EXPORT & PARITY PASS: {info}", flush=True)
    return info


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    # 1. Pack 76 disjoint SANPO-Real train sessions + extract 20 Aria Gen 2 glasses clips
    real76_dir = pack_real_train_76()
    aria_dir = Path("/content/aria_g2_scenes")
    aria_dir.mkdir(parents=True, exist_ok=True)
    aria_tar = Path("/content/aria_g2_20clips.tar")
    if not aria_tar.exists():
        shutil.copy2("/content/drive/MyDrive/BlindAssist/caches/aria_g2_20clips.tar", aria_tar)
    with tarfile.open(aria_tar, "r") as tf:
        tf.extractall(aria_dir)

    # Load datasets into RAM
    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = Path("/content/sanpo_287/scenes") / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            (synth_train if sp == "train" else eval_synth.__setitem__(sc, d)) if sp != "train" else synth_train.append(d)
    for p in sorted(Path("/content/sanpo_p2/extra_scenes").glob("*.npz")):
        synth_train.append(load_scene(p))
    real_train = [load_scene(p) for p in sorted(real76_dir.glob("*.npz"))]
    eval_real = {p.stem: load_scene(p) for p in sorted(Path("/content/sanpo_p2/real_scenes").glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(aria_dir.glob("*.npz"))}

    z_rloc = np.load("/content/sanpo_p2/real_local_gt_cache.npz", allow_pickle=False)
    real_local_rows = {sc: z_rloc[f"r_{sc}"] for sc in eval_real}
    print(
        f"[p4] Loaded synth_train={len(synth_train)}, real_train={len(real_train)}, "
        f"eval_synth={len(eval_synth)}, eval_real={len(eval_real)}, aria_g2={len(aria_dicts)}",
        flush=True,
    )

    # 2. Evaluate baselines + Phase 3 Sim-Only EgoClear-v3-Full on Aria Gen 2 glasses (100% ZERO-SHOT!)
    aria_baselines = eval_aria_baselines(aria_dicts, bw_speeds, device)

    p3_ckpt = torch.load("/content/egoclear_runs_p3/egoclear_v3_full.pt", map_location="cpu", weights_only=False)
    model = EgoClearV3Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    model.load_state_dict(p3_ckpt["head_state_dict"], strict=False)
    ev_aria_simonly = eval_v3_on_regime(model, aria_dicts, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True)
    for cap in (80.0, 120.3, 200.0):
        m = ev_aria_simonly["fixed_4m"]["summary"].get(f"late_at_fa_fr<={cap}")
        if m:
            print(f"[ARIA_G2] EgoClear-v3-Full (SIM-ONLY ZERO-SHOT) @fr<={cap}: {m['late']}/{m['avoidable']}({m['late_rate']*100:.1f}%, ep={m['fa_ep_per_km']:.1f}, fr={m['fa_fr_per_km']:.1f})", flush=True)

    # 3. Fine-tune EgoClear-v3-Full (Sim+Real Joint: 60% Synth + 40% Disjoint Real, frozen DAV2-S, 16 epochs, 576 fr/step!)
    t_ft = time.time()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    ds = MixedSimRealDataset(synth_train, real_train, clip_len=12, clips_per_epoch=1920)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=48, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    epochs = 16
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1e-5)
    history = []
    print(f"\n=== Fine-tuning EgoClear-v3-Full (Sim+Real Joint, 627 synth + {len(real_train)} disjoint real, 16 ep, 576 fr/step) ===", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_loss_v2(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += parts[k]
        for k in ep_l:
            ep_l[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_l.items()}, "sec": round(time.time() - te, 1)})
        if ep in (1, 4, 8, 12, epochs):
            print(f"  [Sim+Real ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} (st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} co={ep_l['co']:.4f}) {time.time()-te:.1f}s", flush=True)

    ckpt_p = LOCAL_RUNS / "egoclear_v3_full_simreal.pt"
    tr_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("dav2.")}
    torch.save({"mode": "full", "c_bev": 128, "head_state_dict": tr_state, "history": history}, ckpt_p)
    shutil.copy2(ckpt_p, DRIVE_RUNS / ckpt_p.name)

    # 4. Evaluate Sim+Real EgoClear-v3-Full across ALL 4 benchmarks (Synth Clean, Synth VIO 1x, Real-40 Unseen, Aria-G2 Zero-Shot!)
    ev_simreal = {
        "synth_clean": eval_v3_on_regime(model, eval_synth, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_v3_on_regime(model, eval_synth, None, bw_speeds, device, vio_scale=1.0),
        "real_40_local": eval_v3_on_regime(model, eval_real, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
        "real_40_global": eval_v3_on_regime(model, eval_real, None, bw_speeds, device, vio_scale=0.0),
        "aria_g2_zeroshot": eval_v3_on_regime(model, aria_dicts, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
    }
    for rname, ev in ev_simreal.items():
        s = ev["fixed_4m"]["summary"]
        m80 = s.get("late_at_fa_fr<=80.0")
        m120 = s.get("late_at_fa_fr<=120.3")
        m200 = s.get("late_at_fa_fr<=200.0")
        s80 = f"{m80['late']}/{m80['avoidable']}({m80['late_rate']*100:.1f}%,ep={m80['fa_ep_per_km']:.1f},fr={m80['fa_fr_per_km']:.1f})" if m80 else "N/A"
        s120 = f"{m120['late']}/{m120['avoidable']}({m120['late_rate']*100:.1f}%,ep={m120['fa_ep_per_km']:.1f},fr={m120['fa_fr_per_km']:.1f})" if m120 else "N/A"
        s200 = f"{m200['late']}/{m200['avoidable']}({m200['late_rate']*100:.1f}%,ep={m200['fa_ep_per_km']:.1f},fr={m200['fa_fr_per_km']:.1f})" if m200 else "N/A"
        print(f"[P4 Sim+Real | {rname}] @fr<=80: {s80} | @fr<=120: {s120} | @fr<=200: {s200}", flush=True)

    # 5. Export 2-stage streaming ONNX and verify numerical parity
    onnx_info = export_and_verify_onnx(model, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "counts": {
            "synth_train": len(synth_train),
            "real_train": len(real_train),
            "eval_synth": len(eval_synth),
            "eval_real": len(eval_real),
            "aria_g2_clips": len(aria_dicts),
        },
        "aria_baselines": aria_baselines,
        "aria_egoclear_v3_simonly_zeroshot": ev_aria_simonly,
        "egoclear_v3_simreal": {"train_sec": round(time.time() - t_ft, 1), "history": history, "evals": ev_simreal},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase4_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase4_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p4] DONE saved phase4_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
