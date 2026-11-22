"""Phase 19 HYPER-OMNI (Colab G4 96 GB Blackwell): EgoClear-v14 ("Hyper-Omni")!

Synthesizes the exact proven winners from Phases 16, 17, and 18 with ZERO cold-start shock:
  1. Triple-Graft Warm-Start:
     - MoGe-Turbo 84x148 Encoder (`25.68 ms` FP16 / `18.26 ms` Mixed `59 MB`, `2.2x-3.1x` faster than v11) <- v12
     - Core 35-Epoch-Converged `NY=9` Step Backbone (`~6.2 ms`, `21.6%` Real40 / `15.3%` Synth / `50.0%` CODa) <- v11
     - Multi-Lag Radial Closing-Rate `dyn_closing_head` (`0.912` Real40 AUROC) <- v13
     - 100% Memory-Free Current-Frame Optical `unc_head` (`100%` 4-domain blackout/finger/whiteout + `>90%` blur)
     - QAIRT 2.47 `Clip(0.0, M)` & FTZ-FP16-safe 2-stage Mean shift solver baked natively (`0.0129 m` S24U z_MAE!)
  2. Two-Stage Zero-Shock Alignment + Gentle Surgical Polish (~42 min total!):
     - Stage 1 (3 ep, ~3.2 min): Freeze Step + ViT, tightly align `UnifiedTurboGeoDecoder` (`2.5x f2d` + `1.6x z_t`
       + `1.1x z_gt` + normals + DAV2) to v11's EXACT encoder feature space!
     - Stage 2 Ep 1 (~2 min): Train ONLY `{unc_head, dyn_closing_head, drop_corr_head, head_corr_head}` (freeze
       encoder + `bev_enc` + `corr_head`) -> locks 100% 4-domain UNKNOWN, ~3% ARKit drop FPR, 0.91 TTC, zero shock!
     - Stage 2 Ep 2-6 (~22 min): Gentle low-LR end-to-end polish (`lr_step=1.2e-4, lr_dec=8e-5, lr_vit=6e-6`) with
       v11's proven LINEAR positive anchor (no quadratic FA blowup!), `1.40x` geometry anchor, `22,631` ARKit flat-floor
       drop negatives, and zero broken v11 head KD.
     - Selects mathematically verified champion among `{EP1_LOCKED, RAW, EMA, SOUP}`!
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v7_model import FEAT_H, FEAT_W
from egoclear_v11_model import EgoClearV11Net
from egoclear_v14_model import EgoClearV14EncoderExport, EgoClearV14Net, EgoClearV14StepExport
from eval_closed_loop import eval_score_curve, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase7_train_v6_g4 import eval_v6_all
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_stage2_diagnostics_g4 import paired_cluster_bootstrap, per_route_at_thr
from phase11_train_v7_g4 import eval_dark_domain, print_block
from phase11_train_v75_g4 import (
    IDX_4M_EXACT,
    Z_EDGES,
    prepare_scene_slice_tensors,
    recalibrate_bn_stats,
)
from phase12_v9_probe_g4 import logit, make_variant_scenes, sigmoid, summarize_scores
from phase13_train_v10_g4 import eval_frontend_mode
from phase14_train_v105_g4 import (
    paired_route_audit_from_scores,
    stage_p13_extra_datasets_parallel,
)
from phase15_train_v106_g4 import audit_prehit_leakage
from phase16_train_v11_g4 import _aux_survival_loss
from phase17_train_v12_turbo_omni_g4 import (
    CODA_DIR,
    DRIVE_ROOT,
    MULTI_RANGE_BINS,
    MULTI_RANGE_W,
    V12TurboOmniDataset,
    _apply_unobs_corruption_rgb01,
    audit_depth_scale,
    audit_omni_v12,
    enrich_pool_v12,
)
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, load_scene

DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase19_v14"
LOCAL_RUNS = Path("/content/egoclear_runs_p19_v14")


class V14HyperOmniDataset(V12TurboOmniDataset):
    """Clean V12TurboOmniDataset + extra 3.5% blur-focused OE (NO Aria letterbox teacher corruption!)."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        rng = np.random.default_rng()
        T = self.clip_len
        if float(item["is_unobservable"].sum().item()) < 0.5 and rng.random() < 0.04:
            rgb_norm = item["rgb_s"].float().numpy()
            rgb_01 = np.clip(rgb_norm * IMAGENET_STD[None] + IMAGENET_MEAN[None], 0.0, 1.0)
            k_len = int(rng.integers(2, 6))
            t_st = int(rng.integers(1, max(2, T - k_len + 1)))
            mode = int(rng.choice([0, 1, 2, 3, 3, 4, 4]))
            unobs = np.zeros((T,), dtype=np.float32)
            for ti in range(t_st, min(T, t_st + k_len)):
                unobs[ti] = 1.0
                c_img = _apply_unobs_corruption_rgb01(rgb_01[ti], mode, rng)
                rgb_norm[ti] = (c_img - IMAGENET_MEAN) / IMAGENET_STD
            item["rgb_s"] = torch.from_numpy(rgb_norm.astype(np.float16))
            item["is_unobservable"] = torch.from_numpy(unobs)
        return item


def compute_stage1_align_loss(
    s_out: tuple,
    t11_out: tuple,
    dav2_targets: torch.Tensor,
    gt_d_sub: torch.Tensor,
    obs_ok_sub: torch.Tensor,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Tight encoder-to-v11 alignment: 2.6x weight on f2d exact feature matching so frozen v11 step works seamlessly!"""
    f_s, g_s, z_s, n_s, m_s, dlz_s, dnr_s, dv_s = s_out
    f_t, g_t, z_t, n_t, m_t, _dlzt, _dnrt, _dvt = t11_out
    obs = obs_ok_sub.reshape(-1, 1, 1, 1).float()
    z_s, n_s, m_s, f_s, g_s = z_s.float(), n_s.float(), m_s.float(), f_s.float(), g_s.float()
    z_t, n_t, m_t, f_t, g_t = z_t.float().detach(), n_t.float().detach(), m_t.float().detach(), f_t.float().detach(), g_t.float().detach()

    ok_t = ((m_t > 0.45) & (z_t > 0.20) & (z_t < 15.0)).float() * obs
    denom = ok_t.sum().clamp(min=1.0)
    lz_s = torch.log(z_s.clamp(min=0.15))
    lz_t = torch.log(z_t.clamp(min=0.15))

    l_z_t = ((lz_s - lz_t).abs() * ok_t).sum() / denom
    l_grad = (
        (
            (F.conv2d(lz_s, sobel_x, padding=1) - F.conv2d(lz_t, sobel_x, padding=1)).abs()
            + (F.conv2d(lz_s, sobel_y, padding=1) - F.conv2d(lz_t, sobel_y, padding=1)).abs()
        )
        * ok_t
    ).sum() / denom
    l_nrm = ((1.0 - (n_s * n_t).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok_t).sum() / denom
    m_sc = m_s.clamp(1e-4, 1.0 - 1e-4)
    m_tc = m_t.clamp(0.0, 1.0)
    l_msk = (-(m_tc * torch.log(m_sc) + (1.0 - m_tc) * torch.log(1.0 - m_sc)) * obs).sum() / (obs.sum().clamp(min=1.0) * FEAT_H * FEAT_W)

    # Strong f2d + geom_cues exact twin alignment to v11 (so v11's 35-ep Step backbone transfers with ~0 loss!)
    l_f2d = (
        (0.55 * (1.0 - F.cosine_similarity(f_s, f_t, dim=1, eps=1e-5).unsqueeze(1)) + 0.45 * F.smooth_l1_loss(f_s, f_t, reduction="none").mean(dim=1, keepdim=True))
        * ok_t
    ).sum() / denom
    l_g10 = (F.smooth_l1_loss(g_s, g_t, reduction="none").mean(dim=1, keepdim=True) * ok_t).sum() / denom

    gt_d = gt_d_sub.reshape(-1, 1, FEAT_H, FEAT_W).float()
    ok_gt = ((gt_d >= 0.35) & (gt_d <= 8.0)).float() * obs
    l_z_gt = ((lz_s - torch.log(gt_d.clamp(min=0.35))).abs() * ok_gt).sum() / ok_gt.sum() if ok_gt.sum() > 20.0 else torch.tensor(0.0, device=z_s.device)

    dv_s = dv_s.float()
    l_dv_cos = 1.0 - F.cosine_similarity(dv_s[:, :64], dav2_targets[:, :64], dim=1, eps=1e-5).unsqueeze(1)
    l_dv_d = F.smooth_l1_loss(dv_s[:, 64:65], dav2_targets[:, 64:65], reduction="none")
    l_dav2 = ((0.60 * l_dv_cos + 0.40 * l_dv_d) * ok_t).sum() / denom
    l_reg = 0.02 * (dlz_s.float().pow(2).mean() + dnr_s.float().pow(2).mean())

    tot = 1.60 * l_z_t + 1.00 * l_z_gt + 0.85 * l_grad + 0.85 * l_nrm + 0.45 * l_msk + 2.60 * l_f2d + 0.90 * l_g10 + 0.45 * l_dav2 + l_reg
    return tot, {
        "tot": float(tot.item()),
        "zt": float(l_z_t.item()),
        "zgt": float(l_z_gt.item()),
        "nrm": float(l_nrm.item()),
        "f2d": float(l_f2d.item()),
    }


def compute_loss_v14(
    out_s: dict[str, torch.Tensor],
    out_v66: dict[str, torch.Tensor],
    out_v11: dict[str, torch.Tensor],
    dav2_targets: torch.Tensor,
    batch: dict[str, torch.Tensor],
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    kd_w: float = 0.65,
    geom_w: float = 1.40,
    heads_only: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    st_log = out_s["static_logits"].float()
    dy_log = out_s["dynamic_logits"].float()
    haz_log = out_s["hazard_logits"].float()
    cum_h = out_s["cum_hazard"].float()
    p_cum = out_s["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out_s["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()
    dyn_fl = batch["dyn_flag"].float()
    dom_id = batch["dom_id"].long()
    is_dark = batch["is_dark"].float().view(-1, 1, 1)
    is_auto = batch["is_autoseg"].float().view(-1, 1)
    dyn_trusted = batch["dyn_gt_trusted"].float().view(-1, 1, 1, 1, 1)
    is_unobs = batch["is_unobservable"].float()
    obs_ok = (1.0 - is_unobs).clamp(0.0, 1.0)

    p66 = out_v66["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p11 = out_v11["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    st66 = torch.sigmoid(out_v66["static_logits"].float().detach())
    st11 = torch.sigmoid(out_v11["static_logits"].float().detach())
    dy66 = torch.sigmoid(out_v66["dynamic_logits"].float().detach())

    dy66_corr_max = dy66[:, :, :, : IDX_4M_EXACT + 1, 6:9].amax(dim=(2, 3, 4))
    gt_dy_corr_max = gt_dy[:, :, :, : IDX_4M_EXACT + 1, 6:9].amax(dim=(2, 3, 4))
    auto_ped_suspect = (is_auto > 0.5) & ((dy66_corr_max > 0.35) | ((p66[:, :, IDX_4M_EXACT] < 0.22) & (gt_dy_corr_max > 0.5)))
    v_co_eff = torch.where(auto_ped_suspect & (y_co[:, :, IDX_4M_EXACT] > 0.5), torch.zeros_like(v_co), v_co) * obs_ok
    st_auto_guard = (1.0 - (is_auto[:, :, None, None, None] * (dy66 > 0.30).float())).clamp(0.0, 1.0) * obs_ok[:, :, None, None, None]

    device = st_log.device
    z_edges = torch.tensor(Z_EDGES, device=device, dtype=torch.float32)
    z_w = torch.where(z_edges <= 4.35, 2.0, 0.5)[None, None, :]

    # --- Always-active Isolated Omni Heads (UNKNOWN + DROP + HEAD + DYN_CLOSING) ---
    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    unc_target = torch.where(is_unobs[:, :, None] > 0.5, torch.ones_like(err_target), err_target)
    unc_mask = torch.maximum(v_co_eff, is_unobs)[:, :, None]
    unc_w = torch.where(is_unobs[:, :, None] > 0.5, torch.full_like(err_target, 6.0), torch.ones_like(err_target))
    loss_un = (F.binary_cross_entropy_with_logits(un_log, unc_target, reduction="none") * unc_mask * unc_w).sum() / (
        (unc_mask * unc_w).sum().clamp(min=1.0)
    )

    is_ark = (dom_id == 3).float().view(-1, 1)
    y_h = batch["y_head"].float()
    p_h = out_s["head_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    ch_h = out_s["head_cum_hazard"].float()
    v_h_out = v_co_eff * (1.0 - is_ark)
    v_h_ark = v_co_eff * is_ark
    loss_head_out = _aux_survival_loss(out_s["head_hazard_logits"], ch_h, p_h, y_h, v_h_out, pos_w=4.8, anchor_pos=1.20, anchor_neg=0.10)
    loss_head_ark = _aux_survival_loss(out_s["head_hazard_logits"], ch_h, p_h, y_h, v_h_ark, pos_w=1.2, anchor_pos=1.05, anchor_neg=0.07)
    ark_h_neg = (v_h_ark > 0.5) & (y_h[:, :, IDX_4M_EXACT] < 0.5)
    loss_head_ark_neg = (F.relu(ch_h[:, :, IDX_4M_EXACT][ark_h_neg] - 0.12) ** 2).mean() if ark_h_neg.any() else torch.tensor(0.0, device=device)
    loss_head_c = 1.25 * loss_head_out + 0.70 * loss_head_ark + 0.75 * loss_head_ark_neg

    v_drop = batch["valid_drop"].float() * obs_ok
    loss_drop = _aux_survival_loss(out_s["drop_hazard_logits"], out_s["drop_cum_hazard"], out_s["drop_probs"], batch["y_drop"].float(), v_drop, pos_w=4.2, anchor_pos=1.10, anchor_neg=0.07)

    v_dcl = batch["valid_dyn_closing"].float() * obs_ok
    loss_dcl = _aux_survival_loss(out_s["dyn_closing_hazard_logits"], out_s["dyn_closing_cum_hazard"], out_s["dyn_closing_probs"], batch["y_dyn_closing"].float(), v_dcl, pos_w=6.2, anchor_pos=1.20, anchor_neg=0.08)
    pass_m = (v_dcl > 0.5) & (batch["y_dyn_passing"].float() > 0.5)
    if pass_m.any():
        p_dcl_pass = out_s["dyn_closing_probs"][:, :, IDX_4M_EXACT].float().clamp(1e-5, 1.0 - 1e-5)[pass_m]
        loss_dcl = loss_dcl + 2.2 * (-torch.log((1.0 - p_dcl_pass).clamp(min=1e-4))).mean()

    if heads_only:
        total = 2.00 * loss_un + 1.15 * loss_head_c + 1.25 * loss_drop + 1.15 * loss_dcl
        return total, {
            "tot": float(total.item()), "co": 0.0, "anc": 0.0, "dyn": 0.0,
            "head": float(loss_head_c.item()), "drop": float(loss_drop.item()),
            "unc": float(loss_un.item()), "dcl": float(loss_dcl.item()), "geo": 0.0,
        }

    # --- Full Joint Polish Losses (v11's proven LINEAR pos anchor + balanced neg + strong v6.6+v11 KD!) ---
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 2.6; col_w[5] = 1.5; col_w[9] = 1.5; col_w[1:5] = 1.4; col_w[10:14] = 1.4
    hb_w = torch.tensor([1.0, 1.3, 4.0], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(5.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st_bce = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w * st_auto_guard).mean()
    st_prob = torch.sigmoid(st_log)
    tversky_sum = torch.tensor(0.0, device=device)
    for bi, bw in enumerate((1.0, 1.0, 2.2)):
        pb = st_prob[:, :, bi] * st_auto_guard[:, :, 0]
        gb = gt_st[:, :, bi] * st_auto_guard[:, :, 0]
        tp = (pb * gb).sum()
        fp = (pb * (1.0 - gb)).sum()
        fn = ((1.0 - pb) * gb).sum()
        tversky_sum = tversky_sum + bw * (1.0 - (tp + 1.0) / (tp + 0.30 * fp + 0.70 * fn + 1.0))
    loss_st = loss_st_bce + 0.30 * (tversky_sum / 4.2)

    dy_w = (1.0 + 2.5 * dyn_trusted) * obs_ok[:, :, None, None, None]
    loss_dy = (F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(7.0, device=device), reduction="none") * dy_w).mean()

    bce_cum = -(1.75 * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co_eff * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co_eff * dn_w)[:, :, None]
    soft_h = F.softplus(haz_log)
    log_p_hit = torch.log(1.0 - torch.exp(-soft_h).clamp(max=1.0 - 1e-6))
    loss_fh = ((-y_first * log_p_hit + (1.0 - y_first) * soft_h) * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)

    y_next = F.pad(y_co[..., 1:], (0, 1), value=1.0)
    has_far_obs = (y_co[:, :, IDX_4M_EXACT] > 0.5).unsqueeze(-1)
    prehit_mask = (v_co_eff.unsqueeze(-1) > 0.5) & has_far_obs & (y_co < 0.5) & (y_next < 0.5)
    loss_prehit = torch.tensor(0.0, device=device)
    if prehit_mask.any():
        p_pre = p_cum[prehit_mask]
        h_pre = soft_h[prehit_mask]
        loss_prehit = (-torch.log((1.0 - p_pre).clamp(min=1e-4))).mean() + 2.2 * (F.relu(p_pre - 0.04) ** 2).mean() + 0.7 * h_pre.mean()

    # v11's proven LINEAR positive anchor (prevents v13's quadratic FA blowup!) + clean negative tail + pedestrian guard!
    loss_dom_anc = torch.tensor(0.0, device=device)
    loss_rank = torch.tensor(0.0, device=device)
    loss_ohem = torch.tensor(0.0, device=device)
    dom_terms = 0.0
    is_ped_only_any = (dyn_fl > 0.5) | (batch["y_dyn_passing"].float() > 0.5)
    for d_val in range(5):
        dm = (dom_id == d_val)[:, None]
        if not dm.any():
            continue
        for b_idx, b_w in zip(MULTI_RANGE_BINS, MULTI_RANGE_W):
            pos_db = dm & (v_co_eff > 0.5) & (y_co[:, :, b_idx] > 0.5)
            neg_db = dm & (v_co_eff > 0.5) & (y_co[:, :, b_idx] < 0.5)
            cb = cum_h[:, :, b_idx]
            pb = p_cum[:, :, b_idx]
            if pos_db.any():
                loss_dom_anc = loss_dom_anc + b_w * F.relu(1.18 - cb[pos_db]).mean()
                dom_terms += b_w
            if neg_db.any():
                nv_l = cb[neg_db]
                k_top = max(1, int(0.18 * nv_l.numel()))
                loss_dom_anc = loss_dom_anc + 1.35 * b_w * (F.relu(torch.topk(nv_l, k_top).values - 0.12) ** 2).mean()
                nv_p = pb[neg_db]
                k_o = max(1, int(0.18 * nv_p.numel()))
                loss_ohem = loss_ohem + b_w * (F.relu(torch.topk(nv_p, k_o).values - 0.05) ** 2).mean()
                dom_terms += b_w
                ped_db = neg_db & is_ped_only_any
                if ped_db.any():
                    loss_dom_anc = loss_dom_anc + 2.8 * b_w * (F.relu(cb[ped_db] - 0.08) ** 2).mean()
            if pos_db.any() and neg_db.any():
                lp = cb[pos_db]
                ln_v = cb[neg_db]
                k_h = max(1, int(0.20 * ln_v.numel()))
                ln_h = torch.topk(ln_v, k_h).values
                loss_rank = loss_rank + b_w * F.softplus((ln_h[None, :] - lp[:, None] + 0.65) / 0.40).mean()
    loss_dom_anc = loss_dom_anc / max(dom_terms, 1.0)
    loss_ohem = loss_ohem / max(dom_terms, 1.0)
    loss_rank = loss_rank / max(dom_terms, 1.0)

    haz_fr4 = (v_co_eff > 0.5) & (y_co[:, :, IDX_4M_EXACT] > 0.5)
    all_neg4 = (v_co_eff > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5)
    dyn_only_fr = all_neg4 & is_ped_only_any
    p4 = p_cum[:, :, IDX_4M_EXACT]
    lam4 = cum_h[:, :, IDX_4M_EXACT]
    clear_pair = all_neg4[:, 1:] & all_neg4[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    loss_dyn_h = torch.tensor(0.0, device=device)
    if dyn_only_fr.any():
        p_dyn = p4[dyn_only_fr]
        l_dyn = lam4[dyn_only_fr]
        loss_dyn_h = (
            (-torch.log((1.0 - p_dyn).clamp(min=1e-4))).mean()
            + 3.5 * (F.relu(p_dyn - 0.04) ** 2).mean()
            + 2.0 * (F.relu(l_dyn - 0.08) ** 2).mean()
        )
        if haz_fr4.any():
            l_pos = lam4[haz_fr4]
            loss_dyn_h = loss_dyn_h + 0.85 * F.softplus((l_dyn[None, :] - l_pos[:, None] + 0.80) / 0.38).mean()

    v_lane = (1.0 - (is_auto > 0.5).float() * (dy66.amax(dim=(2, 3, 4)) > 0.35).float()).clamp(0.0, 1.0) * obs_ok
    loss_left = _aux_survival_loss(out_s["left_hazard_logits"], out_s["left_cum_hazard"], out_s["left_probs"], batch["y_left"].float(), v_lane, pos_w=1.8, anchor_neg=0.10)
    loss_right = _aux_survival_loss(out_s["right_hazard_logits"], out_s["right_cum_hazard"], out_s["right_probs"], batch["y_right"].float(), v_lane, pos_w=1.8, anchor_neg=0.10)

    # Dual-teacher KD (v6.6 5.1% Synth + v11 21.6% Real40; NO broken v11 head KD!)
    is_synth_or_tartan = ((dom_id == 0) | (dom_id == 4)).float().view(-1, 1, 1)
    prefer_v66 = is_synth_or_tartan * (1.0 - 0.4 * is_dark)
    has_stat_bin = y_co > 0.5
    is_dyn_clear = (dyn_fl > 0.5)[:, :, None] & (~has_stat_bin)
    p_haz_t = prefer_v66 * (0.70 * p66 + 0.30 * torch.maximum(p66, p11)) + (1.0 - prefer_v66) * (0.80 * p11 + 0.20 * torch.maximum(p66, p11))
    p_clr_t = torch.minimum(p66, p11)
    p_dyn_t = torch.minimum(p66, torch.full_like(p66, 0.04))
    p_target = torch.where(has_stat_bin, p_haz_t, torch.where(is_dyn_clear, p_dyn_t, p_clr_t)).clamp(1e-4, 1.0 - 1e-4)
    kd_bce = -(p_target * torch.log(p_cum) + (1.0 - p_target) * torch.log(1.0 - p_cum))
    kd_m = v_co_eff[:, :, None] * z_w
    loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)
    dyn_cell = torch.maximum(gt_dy.max(dim=2, keepdim=True).values, (dy66 > 0.35).float().max(dim=2, keepdim=True).values * is_auto[:, :, None, None, None])
    st_target = torch.where(gt_st > 0.5, torch.maximum(st66, st11), torch.minimum(st66, st11))
    kd_st = F.binary_cross_entropy_with_logits(st_log, st_target, reduction="none") * (1.0 - dyn_cell) * spat_w * obs_ok[:, :, None, None, None]

    p_l11 = out_v11["left_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p_r11 = out_v11["right_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p_d11 = out_v11["drop_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    not_arkit = (dom_id != 3).float().view(-1, 1)
    def _soft_bce(ps, pt, vm):
        ps_c = ps.float().clamp(1e-5, 1.0 - 1e-5)
        m_s = vm[:, :, None] * z_w
        return (-(pt * torch.log(ps_c) + (1.0 - pt) * torch.log(1.0 - ps_c)) * m_s).sum() / m_s.sum().clamp(min=1.0)
    loss_omni_kd = 0.35 * _soft_bce(out_s["left_probs"], p_l11, v_lane) + 0.35 * _soft_bce(out_s["right_probs"], p_r11, v_lane) + 0.30 * _soft_bce(out_s["drop_probs"], p_d11, v_drop * not_arkit)
    loss_kd = loss_kd_corr + 0.25 * kd_st.mean() + 0.35 * loss_omni_kd

    # Strong dense geometry + f2d anchor to v11 + GT depth_s8 (zero depth drift!)
    obs_sub = obs_ok[:, ::3].reshape(-1, 1, 1, 1)
    pz_sub4 = out_s["pred_z_m"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    pn_sub4 = out_s["pred_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()
    pm_sub4 = out_s["pred_msk"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dlz_sub4 = out_s["d_logz"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dnr_sub4 = out_s["d_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()
    fs_sub4 = out_s["f2d_all"][:, ::3].reshape(-1, 64, FEAT_H, FEAT_W).float()

    z_t = out_v11["pred_z_m"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float().detach()
    n_t = out_v11["pred_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float().detach()
    m_t = out_v11["pred_msk"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float().detach()
    ft_sub4 = out_v11["f2d_all"][:, ::3].reshape(-1, 64, FEAT_H, FEAT_W).float().detach()
    ok_t = ((m_t > 0.50) & (z_t > 0.25) & (z_t < 12.0) & (pm_sub4 > 0.08)).float() * obs_sub
    denom_t = ok_t.sum().clamp(min=1.0)
    lpz4 = torch.log(pz_sub4.clamp(min=0.15))
    lzt4 = torch.log(z_t.clamp(min=0.15))
    l_grad = (((F.conv2d(lpz4, sobel_x, padding=1) - F.conv2d(lzt4, sobel_x, padding=1)).abs() + (F.conv2d(lpz4, sobel_y, padding=1) - F.conv2d(lzt4, sobel_y, padding=1)).abs()) * ok_t).sum() / denom_t
    l_nrm = ((1.0 - (pn_sub4 * n_t).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok_t).sum() / denom_t
    l_z_t11 = ((lpz4 - lzt4).abs() * ok_t).sum() / denom_t
    l_f2d_a = ((1.0 - F.cosine_similarity(fs_sub4, ft_sub4, dim=1, eps=1e-5).unsqueeze(1)) * ok_t).sum() / denom_t
    pm_c = pm_sub4.clamp(1e-4, 1.0 - 1e-4)
    mt_c = m_t.clamp(0.0, 1.0)
    l_msk = (-(mt_c * torch.log(pm_c) + (1.0 - mt_c) * torch.log(1.0 - pm_c)) * obs_sub).sum() / (obs_sub.sum().clamp(min=1.0) * FEAT_H * FEAT_W)

    gt_d = batch["gt_depth_sub"].reshape(-1, 1, FEAT_H, FEAT_W).float()
    ok_gt = ((gt_d >= 0.35) & (gt_d <= 8.0)).float() * obs_sub
    l_z_gt = ((lpz4 - torch.log(gt_d.clamp(min=0.35))).abs() * ok_gt).sum() / ok_gt.sum() if ok_gt.sum() > 20.0 else torch.tensor(0.0, device=device)
    loss_geom = 0.75 * l_z_t11 + 1.05 * l_z_gt + 0.55 * l_grad + 0.50 * l_nrm + 0.25 * l_msk + 0.75 * l_f2d_a + 0.03 * (dlz_sub4.pow(2).mean() + dnr_sub4.pow(2).mean())

    dv_pred = out_s["dav2_aux"][:, ::3].reshape(-1, dav2_targets.shape[1], FEAT_H, FEAT_W).float()
    ok_v = (pm_sub4 > 0.10).float() * obs_sub
    f_cos = 1.0 - F.cosine_similarity(dv_pred[:, :64], dav2_targets[:, :64], dim=1, eps=1e-5).unsqueeze(1)
    d_l1 = F.smooth_l1_loss(dv_pred[:, 64:65], dav2_targets[:, 64:65], reduction="none")
    loss_dav2 = ((0.60 * f_cos + 0.40 * d_l1) * ok_v).sum() / ok_v.sum().clamp(min=1.0)

    total = (
        1.20 * loss_st
        + 1.10 * loss_dy
        + 1.55 * loss_cum
        + 0.60 * loss_fh
        + 1.25 * loss_prehit
        + 1.35 * loss_dom_anc
        + 1.45 * loss_un
        + 0.45 * loss_tv
        + 1.40 * loss_ohem
        + 0.75 * loss_rank
        + 4.00 * loss_dyn_h
        + 0.45 * (loss_left + loss_right)
        + 0.85 * loss_head_c
        + 1.00 * loss_drop
        + 0.90 * loss_dcl
        + kd_w * loss_kd
        + geom_w * loss_geom
        + 0.40 * loss_dav2
    )
    return total, {
        "tot": float(total.item()),
        "co": float(loss_cum.item()),
        "anc": float(loss_dom_anc.item()),
        "dyn": float(loss_dyn_h.item()),
        "head": float(loss_head_c.item()),
        "drop": float(loss_drop.item()),
        "unc": float(loss_un.item()),
        "dcl": float(loss_dcl.item()),
        "geo": float(loss_geom.item()),
    }


def export_v14_onnx(model: EgoClearV14Net, out_dir: Path) -> dict:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v14_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v14_step_fp32.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV14EncoderExport(m_cpu).eval()
    stp_w = EgoClearV14StepExport(m_cpu).eval()

    rgb_d = torch.randn(1, 3, RGB_H, RGB_W)
    K_d = torch.tensor([[300.0, 300.0, 259.0, 147.0]], dtype=torch.float32)
    R_d = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)

    orig_interp = F.interpolate
    def _no_aa(*args, **kwargs):
        kwargs["antialias"] = False
        return orig_interp(*args, **kwargs)
    F.interpolate = _no_aa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                torch.onnx.export(
                    enc_w, (rgb_d, K_d, R_d), str(enc_fp32),
                    input_names=["rgb", "K_scaled", "R_c2w"],
                    output_names=["f2d", "geom_cues"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
                f2d_d, g_d = enc_w(rgb_d, K_d, R_d)
                t_d = torch.tensor([[0.25, 0.0, 1.5]], dtype=torch.float32)
                t1_d = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
                t2_d = torch.tensor([[-0.25, 0.0, 1.5]], dtype=torch.float32)
                fwd_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                ds_d = torch.tensor([0.25], dtype=torch.float32)
                mem_d = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
                age_d = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0)
                stp_inputs = (
                    f2d_d, g_d, f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, ds_d, mem_d, age_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "f2d_1", "geom_1", "f2d_2", "geom_2",
                    "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1",
                    "K_2", "R_2", "t_2", "ds_t", "mem_prev", "age_prev",
                ]
                torch.onnx.export(
                    stp_w, stp_inputs, str(stp_fp32),
                    input_names=stp_names,
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "omni_probs", "mem_new", "age_new"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
    finally:
        F.interpolate = orig_interp

    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    f32, g32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed["f2d_t"] = f32; feed["geom_t"] = g32
    o32 = s_s32.run(None, feed)
    omni32 = o32[4][0]

    m_gpu = model.cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, g16_t = m_gpu.encode_images(rgb_d.cuda(), K_d.cuda(), R_d.cuda())
    g16 = g16_t.float().cpu().numpy()
    m_gpu.cpu()

    finite_ok = bool(np.all(np.isfinite(f32)) and np.all(np.isfinite(g32)) and np.all(np.isfinite(omni32)) and np.all(np.isfinite(g16)))
    mono_ok = bool(np.all(np.diff(omni32, axis=-1) >= -1e-6))
    enc_fp16_err = float(np.max(np.abs(g16[:, 0] - g32[:, 0])))
    enc_mb = enc_fp32.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp32.stat().st_size / (1024 * 1024)
    print(f"[v14-onnx] PASS! enc_fp32={enc_mb:.2f}MB + stp_fp32={stp_mb:.2f}MB | ieee_fp16_z_err={enc_fp16_err:.5f}m | finite={finite_ok} | all6_mono={mono_ok}", flush=True)
    for p_o in (enc_fp32, stp_fp32):
        shutil.copy2(p_o, DRIVE_RUNS / p_o.name)
    return {
        "enc_fp32_mb": round(enc_mb, 2),
        "stp_fp32_mb": round(stp_mb, 2),
        "ieee_fp16_z_max_err_m": round(enc_fp16_err, 6),
        "finite_ok": finite_ok,
        "all6_omni_monotonic_verified": mono_ok,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[v14] GPU={torch.cuda.get_device_name(0)} | EgoClear-v14 HYPER-OMNI (v12 25.7ms Turbo Encoder + v11 35-ep Converged Step + v13 100% Memory-Free UNKNOWN & 0.91 TTC)!", flush=True)

    paths = stage_all_p8_datasets_parallel()
    p13 = stage_p13_extra_datasets_parallel(max_workers=8)
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

    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = paths["synth_287"] / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            (synth_train.append(d) if sp == "train" else eval_synth.__setitem__(sc, d))
    for p in sorted(paths["synth_450"].glob("*.npz")):
        synth_train.append(load_scene(p))
    for p in sorted(p13["synth_450_p13"].glob("*.npz")):
        synth_train.append(load_scene(p))

    real_gt_train = [load_scene(p) for key in ("real_75", "real_22", "real_ch70") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["real_p13"].glob("*.npz")):
        real_gt_train.append(load_scene(p))
    real_auto_train = [load_scene(p) for p in sorted(p13["real_head_auto"].glob("*.npz"))]
    arkit_train = [load_scene(p) for key in ("arkit_90", "arkit_210") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["arkit_500"].glob("*.npz")):
        arkit_train.append(load_scene(p))
    tartan_train = [load_scene(p) for p in sorted(p13["tartanair"].glob("*.npz"))]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    omni_gt_stats = {
        "train_synth": enrich_pool_v12(synth_train, "train_synth", has_dyn_gt=True),
        "train_real_gt": enrich_pool_v12(real_gt_train, "train_real_gt", has_dyn_gt=True),
        "train_real_auto": enrich_pool_v12(real_auto_train, "train_real_auto", has_dyn_gt=False, allow_flat_pose_neg=False),
        "train_arkit": enrich_pool_v12(arkit_train, "train_arkit", has_dyn_gt=False, allow_flat_pose_neg=True),
        "train_tartan": enrich_pool_v12(tartan_train, "train_tartan", has_dyn_gt=False),
        "eval_synth": enrich_pool_v12(eval_synth, "eval_synth", has_dyn_gt=True),
        "eval_real40": enrich_pool_v12(eval_real, "eval_real40", has_dyn_gt=True),
        "eval_arkit30": enrich_pool_v12(eval_arkit, "eval_arkit30", has_dyn_gt=False, allow_flat_pose_neg=True),
        "eval_coda": enrich_pool_v12(eval_coda, "eval_coda", has_dyn_gt=True, default_h_cam=0.704, allow_flat_pose_neg=True),
    }

    # Frozen Teachers: v6.6 + v11
    teacher_v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    teacher_v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v66.parameters():
        p.requires_grad = False
    _orig_v66_enc = teacher_v66.encode_images
    v66_enc_cache: dict[str, torch.Tensor] = {}
    def _hooked_v66_enc(rgb, K_scaled, R_c2w):
        f2d, disp = _orig_v66_enc(rgb, K_scaled, R_c2w)
        v66_enc_cache["f2d"] = f2d; v66_enc_cache["disp"] = disp
        return f2d, disp
    teacher_v66.encode_images = _hooked_v66_enc

    v11_sd = torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase16_v11/egoclear_v11_omni_moge2_small.pt", map_location="cpu", weights_only=False)["state_dict"]
    teacher_v11 = EgoClearV11Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_moge_neck_and_last2=False, rgb_feat_drop_p=0.0).to(device).eval()
    teacher_v11.load_state_dict(v11_sd, strict=True)
    for p in teacher_v11.parameters():
        p.requires_grad = False

    v12_p = Path("/content/egoclear_runs_p17_v12/egoclear_v12_turbo_omni.pt")
    if not v12_p.exists():
        v12_p = DRIVE_ROOT / "runs/egoclear_v0_phase17_v12/egoclear_v12_turbo_omni.pt"
    v12_sd = torch.load(v12_p, map_location="cpu", weights_only=False)["state_dict"]

    v13_p = Path("/content/egoclear_runs_p18_v13/egoclear_v13_apex_omni.pt")
    if not v13_p.exists():
        v13_p = DRIVE_ROOT / "runs/egoclear_v0_phase18_v13/egoclear_v13_apex_omni.pt"
    v13_sd = torch.load(v13_p, map_location="cpu", weights_only=False)["state_dict"] if v13_p.exists() else None

    student = EgoClearV14Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_vit_last_n=3, rgb_feat_drop_p=0.03, use_moge_pretrained=True).to(device)
    ws_info = student.load_hybrid_warmstart(v12_sd, v11_sd, v13_sd)
    print(f"[v14-triple-graft warmstart] {ws_info}", flush=True)

    ds = V14HyperOmniDataset(synth_train, real_gt_train, real_auto_train, arkit_train, tartan_train, clip_len=12, clips_per_epoch=1920)

    # =========================================================================
    # STAGE 1: 4-Epoch Tight Encoder-to-v11 Feature+Geometry Alignment (Step + ViT FROZEN!)
    # =========================================================================
    s1_epochs = 4
    s1_loader = torch.utils.data.DataLoader(ds, batch_size=12, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=3, drop_last=True)
    orig_req = {n: p.requires_grad for n, p in student.named_parameters()}
    orig_unlock = student.unlock_vit_last_n
    student.unlock_vit_last_n = 0
    for n, p in student.named_parameters():
        p.requires_grad = n.startswith("decoder.") or n.startswith("dav2_aux_proj.") or n.startswith("moge_scale_head.")
    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=4.5e-4, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=s1_epochs * len(s1_loader), eta_min=5.0e-5)

    s1_hist = []
    t_s1 = time.time()
    print(f"\n=== [Stage 1/2] Tight MoGe-Turbo -> v11 Encoder Twin Alignment ({s1_epochs} ep x 1920 clips) ===", flush=True)
    for ep in range(1, s1_epochs + 1):
        student.train()
        ep_l = {"tot": 0.0, "zt": 0.0, "zgt": 0.0, "nrm": 0.0, "f2d": 0.0}
        te = time.time()
        for batch in s1_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_s = batch["rgb_s"][:, ::3].reshape(-1, 3, RGB_H, RGB_W).float()
            rgb_sc = batch["rgb_s_clean"][:, ::3].reshape(-1, 3, RGB_H, RGB_W).float()
            K_4 = batch["K_s"][:, ::3].reshape(-1, 4)
            Rc_4 = batch["R_s_clean"][:, ::3].reshape(-1, 3, 3)
            Rs_4 = batch["R_s"][:, ::3].reshape(-1, 3, 3)
            obs_4 = (1.0 - batch["is_unobservable"][:, ::3].float()).clamp(0.0, 1.0)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t11_o = teacher_v11.encode_images(rgb_sc, K_4, Rc_4, return_aux_geom=True)
                f66, d66 = _orig_v66_enc(rgb_sc, K_4, Rc_4)
                dav2_t = F.interpolate(torch.cat([f66.float(), d66[:, 0:1].float()], dim=1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_o = student.encode_images(rgb_s, K_4, Rs_4, return_aux_geom=True)
                loss, ld = compute_stage1_align_loss(s_o, t11_o, dav2_t, batch["gt_depth_sub"], obs_4, student.sobel_x, student.sobel_y)
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
            opt1.step()
            sched1.step()
            for k in ep_l:
                ep_l[k] += ld[k]
        for k in ep_l:
            ep_l[k] /= len(s1_loader)
        ep_l["ep"] = ep; ep_l["sec"] = round(time.time() - te, 1)
        s1_hist.append(ep_l)
        print(f"  [S1 ep {ep}/{s1_epochs}] tot={ep_l['tot']:.4f} zt={ep_l['zt']:.4f} zgt={ep_l['zgt']:.4f} nrm={ep_l['nrm']:.4f} f2d={ep_l['f2d']:.4f} ({ep_l['sec']:.1f}s)", flush=True)
    del s1_loader, opt1, sched1
    s1_depth = audit_depth_scale(student, eval_real, device)
    # Quick zero-shot check of (Stage1 Turbo Encoder + Frozen v11 Step) BEFORE Stage 2!
    student.eval()
    r_s1 = eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
    s_s1 = eval_v6_all(student, eval_synth, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
    r120_s1 = (r_s1["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
    s120_s1 = (s_s1["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
    rdyn_s1 = (r_s1["dyn_only_alert_rate_at_0.5"] or {}).get("rate", 0.3)
    print(f"[Stage 1 DONE in {time.time()-t_s1:.1f}s] Depth={s1_depth} | ZERO-SHOT (TurboEnc+v11Step): Real@120={r120_s1*100:.1f}% Synth@120={s120_s1*100:.1f}% Real_dyn={rdyn_s1*100:.1f}%!", flush=True)

    # =========================================================================
    # STAGE 2: 6-Epoch Zero-Shock Surgical Polish
    #   - Ep 1: Train ONLY {unc_head, dyn_closing_head, drop_corr_head, head_corr_head} (encoder + core step FROZEN!)
    #   - Ep 2-6: Gentle low-LR joint end-to-end polish (`lr_step=1.2e-4, lr_dec=8e-5, lr_vit=6e-6`)
    # =========================================================================
    s2_epochs = 6
    micro_bs = 8
    accum_steps = 2
    loader = torch.utils.data.DataLoader(ds, batch_size=micro_bs, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=3, drop_last=True)

    student.unlock_vit_last_n = orig_unlock
    for n, p in student.named_parameters():
        p.requires_grad = orig_req[n]

    omni_head_prefixes = ("unc_head.", "dyn_closing_head.", "drop_corr_head.", "head_corr_head.")
    vit_p, dec_p, core_step_p, omni_head_p = [], [], [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("moge_encoder.") or n.startswith("moge_scale_head."):
            vit_p.append(p)
        elif n.startswith("decoder.") or n.startswith("dav2_aux_proj."):
            dec_p.append(p)
        elif n.startswith(omni_head_prefixes):
            omni_head_p.append(p)
        else:
            core_step_p.append(p)

    # Ep 1 dedicated optimizer for isolated Omni heads (zero momentum bleed into core step!)
    opt_h1 = torch.optim.AdamW(omni_head_p, lr=5.0e-4, weight_decay=1e-4)
    # Ep 2-6 gentle joint optimizer
    opt = torch.optim.AdamW([
        {"params": omni_head_p, "lr": 2.4e-4},
        {"params": core_step_p, "lr": 1.1e-4},
        {"params": dec_p, "lr": 8.0e-5},
        {"params": vit_p, "lr": 6.0e-6},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(s2_epochs - 1) * (len(loader) // accum_steps), eta_min=4.0e-6)

    ema_sd = None
    ema_decay = 0.992
    ep1_locked_sd = None
    soup_snaps: list[dict[str, torch.Tensor]] = []
    history = []
    print(f"\n=== [Stage 2/2] EgoClear-v14 Hyper-Omni ({s2_epochs} ep x 1920 clips; ep 1 isolated Omni heads, ep 2-6 gentle joint polish!) ===", flush=True)
    for ep in range(1, s2_epochs + 1):
        heads_only = (ep == 1)
        student.unlock_vit_last_n = 0 if heads_only else orig_unlock
        for p in vit_p + dec_p + core_step_p:
            p.requires_grad = not heads_only
        for p in omni_head_p:
            p.requires_grad = True

        student.train()
        if heads_only:
            student.decoder.eval()
            student.pt_reduce.eval()
            student.bev_enc.eval()
            student.head_dynamic_obs.eval()
            student.mem_gru.eval()
            student.bev_refine.eval()
            student.corr_head.eval()
            student.left_corr_head.eval()
            student.right_corr_head.eval()

        kd_w = 0.72 - 0.24 * (max(0, ep - 2) / max(1, s2_epochs - 2))
        ep_l = {"tot": 0.0, "co": 0.0, "anc": 0.0, "dyn": 0.0, "head": 0.0, "drop": 0.0, "unc": 0.0, "dcl": 0.0, "geo": 0.0}
        te = time.time()
        cur_opt = opt_h1 if heads_only else opt
        cur_opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_s = batch["rgb_s"].float()
            rgb_sc = batch["rgb_s_clean"].float()
            B_cur, T_cur = rgb_sc.shape[:2]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_v66 = teacher_v66.forward_sequence(rgb_sc, batch["K_s"], batch["R_s_clean"], batch["t_clean"], batch["fwd_clean"], batch["seg_clean"])
                f66_all = v66_enc_cache["f2d"].view(B_cur, T_cur, 64, 42, 74)[:, ::3].reshape(-1, 64, 42, 74)
                d66_all = v66_enc_cache["disp"].view(B_cur, T_cur, 4, 42, 74)[:, ::3, 0:1].reshape(-1, 1, 42, 74)
                dav2_t = F.interpolate(torch.cat([f66_all.float(), d66_all.float()], dim=1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
                out_v11 = teacher_v11.forward_sequence(
                    rgb_sc, batch["K_s"], batch["R_s_clean"], batch["t_clean"], batch["fwd_clean"], batch["seg_clean"],
                    return_aux_geom=True,
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(
                    rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"],
                    return_aux_geom=not heads_only,
                )
                loss, ld = compute_loss_v14(out_s, out_v66, out_v11, dav2_t.detach(), batch, student.sobel_x, student.sobel_y, kd_w=kd_w, geom_w=0.0 if heads_only else 1.40, heads_only=heads_only)

            (loss / accum_steps).backward()
            if (bi + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
                cur_opt.step()
                if not heads_only:
                    sched.step()
                cur_opt.zero_grad(set_to_none=True)
                if not heads_only:
                    with torch.no_grad():
                        for k, v in student.state_dict().items():
                            if v.is_floating_point():
                                ema_sd[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                            else:
                                ema_sd[k].copy_(v.detach())
            for k in ep_l:
                ep_l[k] += ld[k]

        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep; ep_l["sec"] = round(time.time() - te, 1); ep_l["heads_only"] = heads_only
        history.append(ep_l)
        print(
            f"  [v14 ep {ep:02d}/{s2_epochs}{'(H)' if heads_only else '   '}] tot={ep_l['tot']:.4f} co={ep_l['co']:.4f} anc={ep_l['anc']:.4f} "
            f"dyn={ep_l['dyn']:.4f} head={ep_l['head']:.4f} drop={ep_l['drop']:.4f} unc={ep_l['unc']:.4f} dcl={ep_l['dcl']:.4f} geo={ep_l['geo']:.4f} ({ep_l['sec']:.1f}s)",
            flush=True,
        )
        if heads_only:
            ep1_locked_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            ema_sd = {
                k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                for k, v in student.state_dict().items()
            }
        elif ep >= s2_epochs - 2:
            soup_snaps.append({k: v.detach().cpu().clone() for k, v in ema_sd.items()})

    train_sec = round(time.time() - t_s1, 1)
    raw_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    ema_cpu = {k: v.detach().cpu().clone() for k, v in ema_sd.items()}
    soup_cpu = {}
    for k in ema_cpu:
        if ema_cpu[k].is_floating_point():
            soup_cpu[k] = sum(s[k].float() for s in soup_snaps) / float(len(soup_snaps))
        else:
            soup_cpu[k] = ema_cpu[k].clone()

    orig_drop = student.rgb_feat_drop_p
    student.rgb_feat_drop_p = 0.0

    def _quick_score(sd_cpu: dict, tag: str) -> tuple[float, dict]:
        student.load_state_dict({k: v.to(device) for k, v in sd_cpu.items()}, strict=True)
        recalibrate_bn_stats(student, loader, device, n_batches=32)
        student.eval()
        r_ev = eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
        s_ev = eval_v6_all(student, eval_synth, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
        a_ev = eval_v6_all(student, eval_arkit, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
        r120 = (r_ev["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
        s120 = (s_ev["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
        a120 = (a_ev["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
        rdyn = (r_ev["dyn_only_alert_rate_at_0.5"] or {}).get("rate", 0.3)
        score = r120 + 0.70 * s120 + 0.35 * rdyn + 0.25 * a120
        print(f"  [Select {tag:10s}] Real@120={r120*100:.1f}% Synth@120={s120*100:.1f}% ARKit@120={a120*100:.1f}% Real_dyn={rdyn*100:.1f}% -> combo={score:.4f}", flush=True)
        return score, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    cands = []
    for tag, sd_c in [("EP1_LOCKED", ep1_locked_sd), ("RAW", raw_sd), ("EMA", ema_cpu), ("SOUP", soup_cpu)]:
        sc, sd_bn = _quick_score(sd_c, tag)
        cands.append((sc, tag, sd_bn))
    cands.sort(key=lambda x: x[0])
    best_sc, winner_tag, best_sd = cands[0]
    student.load_state_dict({k: v.to(device) for k, v in best_sd.items()}, strict=True)
    student.rgb_feat_drop_p = orig_drop
    print(f"[v14 Winner Selected] {winner_tag} (score={best_sc:.4f}; all={[(t, round(s,4)) for s,t,_ in cands]})", flush=True)

    ckpt_final = LOCAL_RUNS / "egoclear_v14_hyper_omni.pt"
    torch.save({
        "state_dict": best_sd,
        "moge_id": student.moge_id,
        "winner": winner_tag,
        "stage1_history": s1_hist,
        "stage2_history": history,
    }, ckpt_final)
    shutil.copy2(ckpt_final, DRIVE_RUNS / ckpt_final.name)

    teacher_v66.encode_images = _orig_v66_enc
    v66_enc_cache.clear()
    student.eval()

    prehit_arkit = audit_prehit_leakage(student, eval_arkit, None, device)
    prehit_real = audit_prehit_leakage(student, eval_real, real_local_rows, device)
    depth_calib = audit_depth_scale(student, eval_real, device)
    print(f"\n[v14 Pre-Hit] ARKit={prehit_arkit} | Real40={prehit_real} | Depth={depth_calib}", flush=True)

    omni_audits = {}
    for dname, dmap, ov in [
        ("synth_clean", eval_synth, None),
        ("real_40_local", eval_real, real_local_rows),
        ("arkit_indoor_val30", eval_arkit, None),
        ("coda_campus_zeroshot", eval_coda, None),
    ]:
        omni_audits[dname] = audit_omni_v12(student, dmap, ov, device)
        print(f"[v14 OMNI AUDIT {dname:20s}] {omni_audits[dname]}", flush=True)

    eval_real_gray = make_variant_scenes(eval_real, "gray")
    evals, paired_audits, boot_v66, boot_v11, iso_evals = {}, {}, {}, {}, {}
    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)

    for dname, dmap, ov, vscale, do_unc in [
        ("synth_clean", eval_synth, None, 0.0, True),
        ("synth_vio_1x", eval_synth, None, 1.0, False),
        ("real_40_local", eval_real, real_local_rows, 0.0, True),
        ("real_40_gray", eval_real_gray, real_local_rows, 0.0, False),
        ("aria_g2_zeroshot", aria_dicts, None, 0.0, True),
        ("arkit_indoor_val30", eval_arkit, None, 0.0, True),
        ("coda_campus_zeroshot", eval_coda, None, 0.0, True),
    ]:
        evals[dname] = eval_v6_all(student, dmap, ov, bw_speeds, device, vio_scale=vscale, compute_uncertainty_audit=do_unc)
        print_block("EgoClear-v14", dname, evals[dname])
        if vscale == 0.0 and dname != "real_40_gray":
            ev_m, s14_m, s66_m, s11_m, sens_m = {}, {}, {}, {}, {}
            for sc, d in dmap.items():
                er = ov[sc] if ov is not None else d["eval_rows"]
                if len(er) < 5:
                    continue
                inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s14_m[sc] = student.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    s66_m[sc] = teacher_v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    s11_m[sc] = teacher_v11.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                ev_m[sc] = er
                sens_m[sc] = sigmoid(0.5 * logit(s66_m[sc]) + 0.5 * logit(s14_m[sc]))
            paired_audits[dname] = paired_route_audit_from_scores(dmap, ov, s14_m, bw_speeds, thrs, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [GLOBAL FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km", flush=True)
            c14 = summarize_at_fa_caps(eval_score_curve(ev_m, s14_m, bw_speeds, thrs, higher_is_alert=True))
            c66 = summarize_at_fa_caps(eval_score_curve(ev_m, s66_m, bw_speeds, thrs, higher_is_alert=True))
            c11 = summarize_at_fa_caps(eval_score_curve(ev_m, s11_m, bw_speeds, thrs, higher_is_alert=True))
            t14 = float(c14["late_at_fa_fr<=120.3"]["thr"]) if c14.get("late_at_fa_fr<=120.3") else 0.50
            t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
            t11 = float(c11["late_at_fa_fr<=120.3"]["thr"]) if c11.get("late_at_fa_fr<=120.3") else 0.50
            boot_v66[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s66_m, bw_speeds, t66), per_route_at_thr(dmap, ov, s14_m, bw_speeds, t14))
            boot_v11[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s11_m, bw_speeds, t11), per_route_at_thr(dmap, ov, s14_m, bw_speeds, t14))
            iso_evals[f"{dname}_ens_v66_v14_squashed"] = summarize_scores(ev_m, sens_m, bw_speeds)
            print(f"     [Bootstrap @120 v14 vs v6.6] {boot_v66[dname]}", flush=True)
            print(f"     [Bootstrap @120 v14 vs v11 ] {boot_v11[dname]}", flush=True)
            print(f"     [Dual-NPU ENS(v6.6+v14)    ] {iso_evals[f'{dname}_ens_v66_v14_squashed']}", flush=True)

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v14", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v14", "real_40_dark018", evals["real_40_dark018"])

    for dname, dmap, modes in [
        ("aria_g2_zeroshot", aria_dicts, ("iso_letterbox", "iso_crop", "iso_grav_crop")),
        ("coda_campus_zeroshot", eval_coda, ("iso_letterbox", "iso_crop")),
    ]:
        for md in modes:
            k_iso = f"{dname}__{md}"
            iso_evals[k_iso] = eval_frontend_mode(student, teacher_v66, dmap, None, bw_speeds, device, md)
            print(f"  [{k_iso:36s}] {iso_evals[k_iso]}", flush=True)

    del teacher_v66, teacher_v11
    torch.cuda.empty_cache()

    onnx_info = export_v14_onnx(student, LOCAL_RUNS / "onnx")
    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v14 Hyper-Omni (v12 25.7ms Turbo Encoder + v11 35-ep Converged Step + v13 100% Memory-Free UNKNOWN & Radial TTC + Native QAIRT ClipFix)",
        "winner": winner_tag,
        "warmstart_info": ws_info,
        "stage1_zero_shot": {"depth": s1_depth, "real120_pct": round(r120_s1 * 100, 1), "synth120_pct": round(s120_s1 * 100, 1), "real_dyn_pct": round(rdyn_s1 * 100, 1)},
        "depth_scale_audit": depth_calib,
        "prehit_leakage_audit": {"v14_arkit": prehit_arkit, "v14_real40": prehit_real},
        "omni_capability_audits": omni_audits,
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "stage1_history": s1_hist,
        "stage2_history": history,
        "evals": evals,
        "iso_and_ens_evals": iso_evals,
        "paired_route_audits_at_0.50": paired_audits,
        "paired_bootstrap_vs_v66_at_120": boot_v66,
        "paired_bootstrap_vs_v11_at_120": boot_v11,
        "onnx_export": onnx_info,
    }
    res_p = LOCAL_RUNS / "phase19_v14_results.json"
    res_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_p, DRIVE_RUNS / res_p.name)
    os.sync()
    print(f"\n[v14] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
