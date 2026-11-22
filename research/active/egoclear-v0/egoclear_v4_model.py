"""EgoClear-v4: Unified Indoor/Outdoor Frozen-DAV2-S + Monotonic First-Hit Survival Hazard + Temporal Flicker Suppression.

Upgrades over EgoClear-v3:
  1. Monotonic First-Hit Survival Hazard Head:
     - Predicts per-range-bin conditional first-hit hazard logits h_k (k = 0..23).
     - Computes exact discrete survival cumulative probability:
         log S_k = sum_{j<=k} (-softplus(h_j))
         P(D <= z_k) = 1 - exp(log S_k)
       Guaranteed strictly monotonic: 0 <= P(<=z_0) <= P(<=z_1) <= ... <= P(<=z_23) <= 1,
       eliminating non-physical far-to-near spikes and improving walker-aware d_req calibration.
     - Uses 100% ONNX-17 / mobile NPU native ops (Softplus, CumSum, Exp, Sub).
  2. Proper negative hazard bias initialization (-3.5) so initial cumulative P(<=4m) ~= 0.38
     instead of saturating near 1.0 at epoch 0.
  3. Temporal Clear-Frame Flicker Suppression Loss (compute_loss_v4):
     - Penalizes frame-to-frame jitter |p_t - p_{t-1}| and isolated spikes ReLU(p_t - 0.12)^2
       on clear corridor sequences, plus mild 2nd-order jerk regularization on valid triples
       (suppressing 1-frame glitches without slowing true obstacle approach ramps).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, NY, RGB_H, RGB_W
from egoclear_v3_model import EgoClearV3Net

H_CORR = 4.0


def survival_cum_probs(hazard_logits: torch.Tensor) -> torch.Tensor:
    """Convert per-bin conditional first-hit hazard logits (..., K) -> strictly monotonic P(D <= z_k)."""
    log_surv = torch.cumsum(-F.softplus(hazard_logits.float()), dim=-1)
    return 1.0 - torch.exp(log_surv.clamp(min=-20.0))


class EgoClearV4Net(EgoClearV3Net):
    """EgoClear-v4 with Monotonic First-Hit Survival Hazard Head (fully ONNX-17 compatible)."""

    def __init__(
        self,
        mode: str = "full",
        dav2_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        c_feat: int = 64,
        c_bev: int = 128,
        freeze_dav2: bool = True,
    ):
        super().__init__(mode=mode, dav2_id=dav2_id, c_feat=c_feat, c_bev=c_bev, freeze_dav2=freeze_dav2)
        # Initialize conditional first-hit hazard bias to -3.5 so P(<=4m) starts near 0.38 (healthy gradients)
        last_conv = self.corr_head[-1]
        assert isinstance(last_conv, nn.Conv1d) and last_conv.out_channels == 2
        with torch.no_grad():
            last_conv.bias[0].fill_(-3.5)
            last_conv.bias[1].fill_(-1.5)

    def forward_sequence(
        self,
        rgb: torch.Tensor,
        K_scaled: torch.Tensor,
        R_c2w: torch.Tensor,
        t_w: torch.Tensor,
        fwd_w: torch.Tensor,
        seg_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = super().forward_sequence(rgb, K_scaled, R_c2w, t_w, fwd_w, seg_len)
        hazard_logits = out["corr_logits"]  # (B, T, 24)
        corr_probs = survival_cum_probs(hazard_logits)  # strictly monotonic in [0, 1], fp32
        out["hazard_logits"] = hazard_logits
        out["corr_probs"] = corr_probs
        return out


def compute_loss_v4(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Unified multi-domain loss: BEV static + dynamic + Monotonic Survival Hazard + Temporal Flicker Suppression."""
    st_log = out["static_logits"].float()
    dy_log = out["dynamic_logits"].float()
    haz_log = out["hazard_logits"].float()
    p_cum = out["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()          # (B, T, 24), monotonic binary step: 1 iff D_gt <= z_k
    v_co = batch["valid_corr"].float()      # (B, T)
    dn_w = batch["dyn_neg_w"].float()       # (B, T), 2.5 on dyn_only clear frames

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.2, 2.5], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    # 1. Static BEV focal BCE (3 height bands)
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()

    # 2. Dynamic BEV BCE (keeps moving pedestrians out of static memory; 0 on static indoor furniture)
    bce_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    loss_dy = bce_dy.mean()

    # 3. Monotonic Cumulative Corridor Survival Focal BCE
    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=device, dtype=torch.float32)
    idx_4m = int((z_edges <= H_CORR + 1e-4).sum().item()) - 1
    z_w = torch.where(z_edges <= H_CORR + 0.2, 2.0, 0.5)[None, None, :]

    # Weighted focal BCE directly on monotonic cumulative probability p_cum
    pos_w = 2.6
    bce_cum = -(pos_w * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    # 4. Exact Discrete First-Hit Survival NLL (supervises the exact bin where the first obstacle begins!)
    # y_first[..., k] = 1 iff k is the first bin with y_co == 1; mask_fh = 1 for all bins <= first_hit (or all bins if clear)
    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    bce_fh = F.binary_cross_entropy_with_logits(haz_log, y_first, pos_weight=torch.tensor(3.0, device=device), reduction="none")
    loss_fh = (bce_fh * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)
    loss_co = loss_cum + 0.35 * loss_fh

    # 5. Clear-Frame Temporal Flicker Suppression + 2nd-Order Jerk Regularization (cuts FA_ep/km fragmentation!)
    p4 = p_cum[:, :, idx_4m]   # (B, T)
    y4 = y_co[:, :, idx_4m]    # (B, T)
    clear = v_co * (1.0 - y4)  # valid clear-corridor frames
    clear_pair = clear[:, 1:] * clear[:, :-1]
    loss_clear_tv = ((p4[:, 1:] - p4[:, :-1]).abs() * clear_pair).sum() / clear_pair.sum().clamp(min=1.0)
    loss_clear_spike = (F.relu(p4 - 0.12).pow(2) * clear * dn_w).sum() / (clear * dn_w).sum().clamp(min=1.0)

    # Mild 2nd-order jerk penalty on all valid triples (kills 1-frame spikes while preserving true approach ramps!)
    if p4.shape[1] >= 3:
        v3 = v_co[:, 2:] * v_co[:, 1:-1] * v_co[:, :-2]
        jerk = ((p4[:, 2:] - p4[:, 1:-1]) - (p4[:, 1:-1] - p4[:, :-2])).abs()
        loss_jerk = (jerk * v3).sum() / v3.sum().clamp(min=1.0)
    else:
        loss_jerk = torch.zeros((), device=device)
    loss_flk = 0.60 * loss_clear_tv + 0.80 * loss_clear_spike + 0.20 * loss_jerk

    # 6. Epistemic Uncertainty Calibration
    with torch.no_grad():
        err_tgt = ((p_cum - y_co).abs() > 0.35).float()
    bce_un = F.binary_cross_entropy_with_logits(un_log, err_tgt, reduction="none")
    loss_un = (bce_un * m_co).sum() / m_co.sum().clamp(min=1.0)

    total = loss_st + 0.45 * loss_dy + 1.8 * loss_co + 1.0 * loss_flk + 0.2 * loss_un
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "dy": float(loss_dy.item()),
        "co": float(loss_co.item()),
        "flk": float(loss_flk.item()),
    }


def compute_loss_v4_calibrated(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Calibrated Phase 5b loss: preserves monotonic survival & flicker suppression without over-sharpening ranking."""
    st_log = out["static_logits"].float()
    dy_log = out["dynamic_logits"].float()
    haz_log = out["hazard_logits"].float()
    p_cum = out["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.2, 2.5], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()

    bce_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    loss_dy = bce_dy.mean()

    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=device, dtype=torch.float32)
    idx_4m = int((z_edges <= H_CORR + 1e-4).sum().item()) - 1
    z_w = torch.where(z_edges <= H_CORR + 0.2, 2.0, 0.5)[None, None, :]

    # Gentle label smoothing (0.015) + balanced pos_w=2.1 keeps probabilities well-calibrated across 0.02..0.995
    y_s = y_co * 0.97 + 0.015
    pos_w = 2.1
    bce_cum = -(pos_w * y_s * torch.log(p_cum) + (1.0 - y_s) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.2) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    bce_fh = F.binary_cross_entropy_with_logits(haz_log, y_first, pos_weight=torch.tensor(2.5, device=device), reduction="none")
    loss_fh = (bce_fh * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)
    loss_co = loss_cum + 0.30 * loss_fh

    # Calibrated flicker regularization (suppresses isolated clear spikes without crushing real mid-conf ranking)
    p4 = p_cum[:, :, idx_4m]
    y4 = y_co[:, :, idx_4m]
    clear = v_co * (1.0 - y4)
    clear_pair = clear[:, 1:] * clear[:, :-1]
    loss_clear_tv = ((p4[:, 1:] - p4[:, :-1]).abs() * clear_pair).sum() / clear_pair.sum().clamp(min=1.0)
    loss_clear_spike = (F.relu(p4 - 0.18).pow(2) * clear * dn_w).sum() / (clear * dn_w).sum().clamp(min=1.0)
    if p4.shape[1] >= 3:
        v3 = v_co[:, 2:] * v_co[:, 1:-1] * v_co[:, :-2]
        jerk = ((p4[:, 2:] - p4[:, 1:-1]) - (p4[:, 1:-1] - p4[:, :-2])).abs()
        loss_jerk = (jerk * v3).sum() / v3.sum().clamp(min=1.0)
    else:
        loss_jerk = torch.zeros((), device=device)
    loss_flk = 0.50 * loss_clear_tv + 0.40 * loss_clear_spike + 0.15 * loss_jerk

    with torch.no_grad():
        err_tgt = ((p_cum - y_co).abs() > 0.35).float()
    bce_un = F.binary_cross_entropy_with_logits(un_log, err_tgt, reduction="none")
    loss_un = (bce_un * m_co).sum() / m_co.sum().clamp(min=1.0)

    total = loss_st + 0.45 * loss_dy + 1.8 * loss_co + 0.35 * loss_flk + 0.2 * loss_un
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "dy": float(loss_dy.item()),
        "co": float(loss_co.item()),
        "flk": float(loss_flk.item()),
    }


class EgoClearV4StepExport(nn.Module):
    """Stage-2 Mobile ONNX streaming step for EgoClear-v4 (outputs strictly monotonic corr_probs directly!)."""

    def __init__(self, core: EgoClearV4Net):
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
        corr_probs = survival_cum_probs(corr_out[:, 0])
        return st_log, dyn_log, corr_probs, corr_out[:, 1], mem_new, age_new
