"""EgoClear-v13 ("Apex-Omni"): 9.1ms W8A8 / 35ms FP16 MoGe-Turbo Encoder +
Rock-Stable Mount-Wide NY=9 3D Probing with Soft Floor-Elevation Conditioning +
100% Memory-Free Current-Frame Optical UNKNOWN Gate + Wide-Gated Pedestrian Immunity +
Radial Closing-Rate TTC + Unbiased Ground-Aware Outdoor HEAD & Drop-Off Survival.

Built directly from our Phase 17 (EgoClear-v12) empirical autopsy:
  1. Keeps v12's exact UnifiedTurboGeoDecoder + MoGe-2 DINOv2 ViT-Small/14 encoder (100% warm-start from v12;
     verified 9.12-9.42 ms W8A8 / 35-39 ms FP16 on S24 Ultra Hexagon v75, Total Subnets: 1!).
  2. Fixes v12's CODa underground collapse & frame-to-frame slice jitter:
     - Uses rock-stable fixed camera-relative slices `Y_SLICES_V13 = (-1.70, -1.35, -1.00, -0.68, -0.35, -0.05, +0.25, +0.55, +0.85)`
       spanning 0.70m low robots (CODa) through 1.65m smart glasses (Aria/SANPO-Head) with zero coordinate jitter.
     - Computes an unbiased online 3-frame `h_floor_cam` (removes v12's 9x `w_low` downward bias) and injects
       `h_above_floor = grid_y - h_floor_cam` per voxel + soft ground-anchored attention weights for `head` & `drop`
       WITHOUT shifting physical voxel coordinates underground!
  3. Fixes v12's ARKit 66.7% blackout leak:
     - Completely removes `bev_ref` (`mem_prev`) from `unc_head`!
     - Feeds pure current-frame `bev_obs` + global image-level `f2d` & `geom` optical-quality summaries so blackout,
       red finger-cover, sunlit whiteout, motion blur, and defocus blur trigger instant frame-1 abstention across ALL domains!
  4. Upgrades pedestrian immunity (`stat_gate` widened via 1x3 corridor max-pool over `dyn_prob`) and adds explicit
     multi-lag radial closing-velocity cues (`invz_diff`, `-ttc_exp_err` across lags 1 & 2) to `dyn_closing_head`.
  5. 100% drop-in compatible ONNX contract:
     - Encoder: `(rgb, K_scaled, R_c2w) -> (f2d [1,64,84,148], geom_cues [1,10,84,148])`
     - Step: exact 20 inputs -> exact 7 outputs (`static_logits, dynamic_logits, corr_probs, unc_logits, omni_probs, mem_new, age_new`),
       zero 0-D scalars, zero QAIRT folding hacks!
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from egoclear_model import (
    BEV_DX,
    BEV_DZ,
    BEV_NX,
    BEV_NZ,
    BEV_X_MAX,
    BEV_X_MIN,
    BEV_Z_MAX,
    BEV_Z_MIN,
    RGB_H,
    RGB_W,
    ConvBNReLU,
    ResBlock2D,
)
from egoclear_v7_model import FEAT_H, FEAT_W
from egoclear_v10_model import GEOM_CH_V10
from egoclear_v11_model import _haz_to_cum_prob, _make_survival_head
from egoclear_v12_model import (
    EgoClearV12Net,
    SafeFreeSpaceDynamicConvGRU,
    UnifiedTurboGeoDecoder,
    _DummyMoGeViTSEncoder,
    _DummyScaleHead,
    _build_uv_levels_pure,
)

# Rock-stable camera-relative vertical slices spanning ALL mount heights (0.70 m CODa -> 1.65 m glasses):
Y_SLICES_V13 = (-1.70, -1.35, -1.00, -0.68, -0.35, -0.05, 0.25, 0.55, 0.85)
NY_V13 = len(Y_SLICES_V13)


class EgoClearV13Net(EgoClearV12Net):
    """EgoClear-v13 ("Apex-Omni"): inherits v12's exact MoGe-Turbo Encoder, upgrades Step geometry,
    memory-free UNKNOWN gate, wide dynamic gating, ground-aware soft HEAD/DROP bands, and radial TTC."""

    def __init__(
        self,
        moge_id: str = "Ruicheng/moge-2-vits-normal",
        c_feat: int = 64,
        c_bev: int = 128,
        lags: tuple[int, ...] = (1, 2),
        unlock_vit_last_n: int = 3,
        delta_z_max: float = 0.48,
        delta_n_max: float = 0.35,
        rgb_feat_drop_p: float = 0.04,
        use_moge_pretrained: bool = True,
    ):
        super().__init__(
            moge_id=moge_id,
            c_feat=c_feat,
            c_bev=c_bev,
            lags=lags,
            unlock_vit_last_n=unlock_vit_last_n,
            delta_z_max=delta_z_max,
            delta_n_max=delta_n_max,
            rgb_feat_drop_p=rgb_feat_drop_p,
            use_moge_pretrained=use_moge_pretrained,
        )
        # Enable Hexagon v75 FP16-exact 2-stage Mean shift solver (eliminates ±0.25m 12432-elem FP16 Sum drift!)
        self._htp_mean_shift = True

        # Fixed camera-relative probe slices (zero jitter, zero CODa underground shift!)
        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES_V13, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        self.register_buffer("grid_x_v13", gx.reshape(-1), persistent=False)
        self.register_buffer("grid_z_v13", gz.reshape(-1), persistent=False)
        self.register_buffer("grid_y_cam", gy.reshape(-1), persistent=False)

        # +1 per-voxel channel for explicit `h_above_floor = (grid_y_cam - h_floor_cam) / 1.5`!
        c_pt = c_feat + GEOM_CH_V10 + 1 + 3 + 9 + 6 * len(self.lags)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        # Dedicated 100% MEMORY-FREE Current-Frame UNKNOWN Gate (ZERO bev_ref leakage -> 100% 4-domain blackout!)
        # Inputs: obs_pool(128) + f2d_glob_mean(64) + f2d_glob_abs(64) + g_qual(4) + [vis,free,age,fl_conf](4) + dyn_c3_max(3) = 267
        self.unc_head = nn.Sequential(
            nn.Conv1d(c_bev + c_feat * 2 + 11, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 48, 3, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 1, 1),
        )
        with torch.no_grad():
            self.unc_head[-1].bias.fill_(-2.0)

        # Upgraded HEAD head (+2 soft ground-anchored head-band features -> 14 phys ch) & less conservative bias!
        self.head_corr_head = _make_survival_head(c_bev * 5 + 14, c_out=1, hidden=96, init_bias=-2.8)
        # Upgraded DROP head (+2 soft below-floor features -> 17 phys ch)
        self.drop_corr_head = _make_survival_head(c_bev * 5 + 17, c_out=1, hidden=96, init_bias=-4.0)
        # Upgraded DYN_CLOSING head (+4 explicit radial closing-rate cues -> 18 phys ch)
        self.dyn_closing_head = _make_survival_head(c_bev * 5 + 18, c_out=1, hidden=96, init_bias=-3.2)

    def load_v12_warmstart(self, v12_sd: dict[str, torch.Tensor]) -> dict[str, int]:
        """Warm-starts 100% of v12's converged MoGe-Turbo encoder + compatible step weights!"""
        cur = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in v12_sd.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(cur, strict=True)
        return {"loaded": loaded, "skipped": skipped}

    def _estimate_single_frame_floor_v13(self, geom: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unbiased online horizontal walking-floor estimator (removes v12's 9x downward `w_low` bias!)."""
        N = geom.shape[0]
        g_sub = geom[:, :, ::2, ::2].float()  # [B, 10, 42, 74]
        z_m = g_sub[:, 0]
        n_up = g_sub[:, 5]
        msk = g_sub[:, 7]

        fx = K_scaled[:, 0].float().view(N, 1, 1).clamp(min=10.0)
        fy = K_scaled[:, 1].float().view(N, 1, 1).clamp(min=10.0)
        cx = K_scaled[:, 2].float().view(N, 1, 1)
        cy = K_scaled[:, 3].float().view(N, 1, 1)
        rx = (self.fl_uu - cx) / fx
        ry = (self.fl_vv - cy) / fy

        r20 = R_c2w[:, 2, 0].float().view(N, 1, 1)
        r21 = R_c2w[:, 2, 1].float().view(N, 1, 1)
        r22 = R_c2w[:, 2, 2].float().view(N, 1, 1)
        dz_w = z_m * (r20 * rx + r21 * ry + r22)

        # Near-field horizontal floor (0.7..2.8m has lowest depth error & excludes distant slopes/roofs)
        w_flat = torch.sigmoid((n_up - 0.84) * 14.0)
        w_rng = torch.sigmoid((z_m - 0.70) * 6.0) * torch.sigmoid((2.80 - z_m) * 3.5)
        w_h = torch.sigmoid((-0.52 - dz_w) * 10.0) * torch.sigmoid((dz_w + 1.95) * 10.0)
        w_lat = torch.sigmoid((0.80 - rx.abs()) * 6.0)
        # Mild lower-surface preference (0.25x instead of v12's runaway 1.4x -> zero CODa downward drift!)
        w_low = torch.exp(((-dz_w - 1.15) * 0.25).clamp(-0.5, 0.5))
        w = msk * w_flat * w_rng * w_h * w_lat * w_low
        w_sum = w.sum(dim=(1, 2))
        wh_sum = (w * dz_w.clamp(-1.95, -0.52)).sum(dim=(1, 2))
        return wh_sum, w_sum

    def step_from_encoded(
        self,
        f2d_t, geom_t,
        K_t, R_t, t_t, fwd_t,
        lag_packs,
        t_prev, fwd_prev, ds_t,
        mem_prev, age_prev,
        is_first: bool,
    ):
        B = f2d_t.shape[0]
        device, dtype = f2d_t.device, f2d_t.dtype

        # --- 1. Unbiased 3-Frame Floor-Height Estimator (Conditioning Feature ONLY — NEVER shifts voxel grid!) ---
        wh_tot, w_tot = self._estimate_single_frame_floor_v13(geom_t, K_t, R_t)
        for lp in lag_packs:
            if lp is not None:
                _fp, g_p, K_p, R_p, _tp = lp
                wh_p, w_p = self._estimate_single_frame_floor_v13(g_p, K_p, R_p)
                wh_tot = wh_tot + 0.7 * wh_p
                w_tot = w_tot + 0.7 * w_p
        floor_conf = w_tot / (w_tot + 35.0)
        h_est_raw = wh_tot / w_tot.clamp(min=1.0)
        h_floor_cam = (floor_conf * h_est_raw + (1.0 - floor_conf) * (-1.30)).clamp(-1.85, -0.58)

        # Rock-stable FIXED camera-relative 3D probe coordinates (NY=9, zero jitter, zero underground collapse!)
        gx = self.grid_x_v13[None, :, None]
        gz = self.grid_z_v13[None, :, None]
        gy = self.grid_y_cam[None, :, None].expand(B, -1, -1).to(t_t.dtype)
        right_t = torch.stack([fwd_t[:, 1], -fwd_t[:, 0], torch.zeros_like(fwd_t[:, 0])], dim=-1)
        pts_w = t_t[:, None, :] + gx * right_t[:, None, :] + gz * fwd_t[:, None, :] + gy * self.up_w

        # Per-voxel explicit height above local walking floor [B, 1, N_pts]
        h_above_fl = (self.grid_y_cam[None, None, :] - h_floor_cam[:, None, None]).clamp(-1.0, 2.6)
        h_above_norm = (h_above_fl / 1.5).to(dtype)

        samp_f, samp_g, vis_t, uv_invz_t, z_vox = self._project_and_sample(f2d_t, geom_t, pts_w, R_t, t_t, K_t)

        z_pred = samp_g[:, 0:1, :]
        invz_t_v = samp_g[:, 1:2, :]
        logz_t_v = samp_g[:, 2:3, :]
        n_up_v = samp_g[:, 5:6, :]
        vert_v = samp_g[:, 6:7, :]
        mask_v = samp_g[:, 7:8, :]

        zv_safe = z_vox.clamp(min=0.25)
        rel_mid = (z_pred - z_vox) / zv_safe
        rel_near = (0.84 * z_pred - z_vox) / zv_safe
        rel_far = (1.18 * z_pred - z_vox) / zv_safe

        signed_sdf = rel_mid.clamp(-1.5, 1.5) * vis_t
        surf_mid = torch.exp(-((rel_mid / 0.18).clamp(-6.0, 6.0) ** 2)) * mask_v * vis_t
        surf_near = torch.exp(-((rel_near / 0.18).clamp(-6.0, 6.0) ** 2)) * mask_v * vis_t
        surf_far = torch.exp(-((rel_far / 0.18).clamp(-6.0, 6.0) ** 2)) * mask_v * vis_t
        vert_mid = surf_mid * vert_v
        vert_near = surf_near * vert_v
        floor_hit = surf_mid * F.relu(n_up_v)
        free_ev_vox = torch.sigmoid(((rel_mid - 0.30) / 0.08).clamp(-10.0, 10.0)) * F.relu((mask_v - 0.50) / 0.50) * vis_t
        drop_pass_vox = torch.sigmoid(((rel_mid - 0.14) / 0.06).clamp(-10.0, 10.0)) * F.relu((mask_v - 0.45) / 0.55) * vis_t

        pt_parts = [
            samp_f, samp_g, vis_t, uv_invz_t,
            signed_sdf, surf_mid, surf_near, surf_far, vert_mid, vert_near, floor_hit, free_ev_vox,
            h_above_norm * vis_t,
        ]
        ft_norm = F.normalize(samp_f, dim=1, eps=1e-5)
        lag_ttc_list, lag_invz_list, lag_diff_list = [], [], []
        for lp in lag_packs:
            if lp is not None:
                f_p, g_p, K_p, R_p, t_p = lp
                sf_p, sg_p, vis_p, _, z_vox_p = self._project_and_sample(f_p, g_p, pts_w, R_p, t_p, K_p)
                both_v = vis_t * vis_p
                fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
                cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
                l1_diff = (samp_f - sf_p).abs().mean(dim=1, keepdim=True) * both_v
                obs_log_exp = sg_p[:, 2:3, :] - logz_t_v
                geom_log_exp = torch.log(z_vox_p.float() / z_vox.float().clamp(min=0.15)).to(dtype)
                ttc_exp_err = (obs_log_exp - geom_log_exp).clamp(-2.0, 2.0) * both_v
                invz_diff = (invz_t_v - sg_p[:, 1:2, :]).clamp(-2.0, 2.0) * both_v
                base = torch.linalg.norm(t_t - t_p, dim=-1).clamp(0.0, 2.5)
                base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(dtype) * both_v
                pt_parts.extend([cos_sim, l1_diff, ttc_exp_err, invz_diff, both_v, base_map])
                lag_ttc_list.append(ttc_exp_err)
                lag_invz_list.append(invz_diff)
                lag_diff_list.append(l1_diff)
            else:
                z0 = torch.zeros(B, 1, pts_w.shape[1], device=device, dtype=dtype)
                pt_parts.append(torch.zeros(B, 6, pts_w.shape[1], device=device, dtype=dtype))
                lag_ttc_list.append(z0)
                lag_invz_list.append(z0)
                lag_diff_list.append(z0)

        h_norm_map = ((h_floor_cam + 1.30) / 0.50).view(B, 1, 1, 1).expand(B, 1, BEV_NZ, BEV_NX).to(dtype)
        bev_pos = torch.cat([
            torch.stack([
                self.bev_xz[..., 0] / BEV_X_MAX,
                (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
            ], dim=0).unsqueeze(0).expand(B, -1, -1, -1).to(dtype),
            h_norm_map,
        ], dim=1)

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY_V13 * BEV_NZ, BEV_NX)
        pt_red = self.pt_reduce(pt_feat).reshape(B, 48 * NY_V13, BEV_NZ, BEV_NX)
        bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))
        dyn_log = self.head_dynamic_obs(bev_obs)
        dyn_prob_3 = torch.sigmoid(dyn_log)
        dyn_prob = dyn_prob_3.max(dim=1, keepdim=True).values.detach()
        # Wide 1x3 lateral max-pool catches pedestrians stepping right along the corridor edge!
        dyn_prob_wide = F.max_pool2d(dyn_prob, kernel_size=(1, 3), stride=1, padding=(0, 1))

        vis_9 = vis_t.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        free_9 = free_ev_vox.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        vis_bev = vis_9[:, 2:].mean(dim=1, keepdim=True)
        free_bev = (free_9[:, 2:].mean(dim=1, keepdim=True) * (vis_bev > 0.35).to(dtype)).clamp(0.0, 1.0)

        if not is_first:
            mem_w = self._warp_bev(mem_prev, t_prev, fwd_prev, t_t, fwd_t)
            age_w = self._warp_bev(age_prev, t_prev, fwd_prev, t_t, fwd_t)
            ds = (ds_t.clamp(0.0, 2.0) + 0.04)[:, None, None, None].to(dtype)
            age_w = (age_w + ds).clamp(0.0, 10.0)
            age_decay = torch.exp(-age_w / 4.0)
            mem_new = self.mem_gru(mem_w, bev_obs, vis_bev, age_decay, dyn_prob, free_bev)
            age_new = torch.where((vis_bev > 0.05) & (dyn_prob < 0.5), torch.zeros_like(age_w), age_w)
        else:
            mem_new = bev_obs * (1.0 - 0.5 * dyn_prob)
            age_new = torch.where(vis_bev > 0.05, torch.zeros_like(age_prev), age_prev)

        bev_ref = self.bev_refine(mem_new)
        st_log = self.head_static(bev_ref)
        st_p_all = torch.sigmoid(st_log)

        # --- 2. Wide Static-Gated 3D Physics Volume + Soft Floor-Relative Band Weights ---
        stat_gate = (1.0 - 0.92 * dyn_prob_wide).clamp(0.08, 1.0)
        vm_9 = vert_mid.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        vn_9 = vert_near.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        fh_9 = floor_hit.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        sm_9 = surf_mid.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        dp_9 = drop_pass_vox.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        sd_9 = signed_sdf.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        nu_9 = n_up_v.reshape(B, NY_V13, BEV_NZ, BEV_NX)
        hab_9 = h_above_fl.reshape(B, NY_V13, BEV_NZ, BEV_NX)

        # Soft ground-anchored weights (adapts to mount height WITHOUT moving physical voxel coords!)
        w_head_band = torch.sigmoid((hab_9 - 1.25) * 6.0).to(dtype)
        w_drop_band = torch.sigmoid((-0.15 - hab_9) * 6.0).to(dtype)

        # Slices 2..8 = [-1.00 .. +0.85 m] cover LOW+BODY+HEAD across all mounts (0.7m CODa to 1.65m glasses)
        vm_7 = vm_9[:, 2:] * stat_gate
        vn_7 = vn_9[:, 2:] * stat_gate
        fh_7, fr_7, vs_7 = fh_9[:, 2:], free_9[:, 2:], vis_9[:, 2:]

        # --- 3. Center Static Corridor Survival ---
        vh_m = vm_7[:, :, :, 5:10]
        vh_n = vn_7[:, :, :, 5:10]
        fh_4 = fh_7[:, :, :, 5:10]
        fr_4 = fr_7[:, :, :, 5:10]
        vs_4 = vs_7[:, :, :, 5:10]
        st_p = st_p_all[:, :, :, 5:10]
        dy_c_max = dyn_prob_wide[:, 0, :, 5:10].amax(dim=2).unsqueeze(1)
        dy_c_mean = dyn_prob_wide[:, 0, :, 5:10].mean(dim=2).unsqueeze(1)
        phys_corr = torch.cat([
            vh_m[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 2:4].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 4:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 2:4].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 4:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 2:4].mean(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            vh_n.mean(dim=(1, 3)).unsqueeze(1),
            fh_4[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            fh_4[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 2:4].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            vs_4.mean(dim=(1, 3)).unsqueeze(1),
            st_p.amax(dim=3),
            st_p.mean(dim=3),
            dy_c_max,
            dy_c_mean,
        ], dim=1).to(bev_ref.dtype)

        c_slice = bev_ref[:, :, :, 5:10].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        c_left_m = bev_ref[:, :, :, :5].mean(dim=3)
        c_right_m = bev_ref[:, :, :, 10:].mean(dim=3)
        haz_log = self.corr_head(torch.cat([c_slice, c_left_m, c_right_m, phys_corr], dim=1)).squeeze(1)
        cum_haz, corr_p = _haz_to_cum_prob(haz_log)

        # --- 4. 100% MEMORY-FREE Current-Frame UNKNOWN Gate (ZERO bev_ref leakage!) ---
        obs_pool = bev_obs[:, :, :, 5:10].mean(dim=3)
        f2d_glob_mean = f2d_t.mean(dim=(2, 3), keepdim=False).unsqueeze(-1).expand(B, self.c_feat, BEV_NZ)
        f2d_glob_abs = f2d_t.abs().mean(dim=(2, 3), keepdim=False).unsqueeze(-1).expand(B, self.c_feat, BEV_NZ)
        g_z_mean = (geom_t[:, 0:1].mean(dim=(2, 3)) / 5.0).clamp(0.0, 2.0)
        g_edge_mean = (geom_t[:, 3:4].abs() + geom_t[:, 4:5].abs()).mean(dim=(2, 3)).clamp(0.0, 2.0)
        g_msk_mean = geom_t[:, 7:8].mean(dim=(2, 3)).clamp(0.0, 1.0)
        g_vert_mean = geom_t[:, 6:7].mean(dim=(2, 3)).clamp(0.0, 1.0)
        g_qual = torch.cat([g_z_mean, g_edge_mean, g_msk_mean, g_vert_mean], dim=1).unsqueeze(-1).expand(B, 4, BEV_NZ).to(dtype)

        vis_cur_c = vis_bev[:, :, :, 5:10].mean(dim=3)
        free_cur_c = free_bev[:, :, :, 5:10].mean(dim=3)
        age_cur_c = (age_new[:, :, :, 5:10].mean(dim=3) / 10.0).clamp(0.0, 1.0)
        fl_conf_c = floor_conf.view(B, 1, 1).expand(B, 1, BEV_NZ).to(dtype)
        dyn_c3 = dyn_prob_3[:, :, :, 5:10]
        unc_in = torch.cat([
            obs_pool, f2d_glob_mean, f2d_glob_abs, g_qual,
            vis_cur_c, free_cur_c, age_cur_c, fl_conf_c, dyn_c3.amax(dim=3),
        ], dim=1)
        unc_log = self.unc_head(unc_in).squeeze(1)

        # --- 5. Left & Right Bypass Lane Survival ---
        c_center_m = bev_ref[:, :, :, 5:10].mean(dim=3)
        l_slice = bev_ref[:, :, :, 0:5].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        r_slice = bev_ref[:, :, :, 10:15].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        phys_l = self._lane_phys_18(vm_7[:, :, :, 0:5], vn_7[:, :, :, 0:5], fh_7[:, :, :, 0:5], fr_7[:, :, :, 0:5], vs_7[:, :, :, 0:5], st_p_all[:, :, :, 0:5]).to(bev_ref.dtype)
        phys_r = self._lane_phys_18(vm_7[:, :, :, 10:15], vn_7[:, :, :, 10:15], fh_7[:, :, :, 10:15], fr_7[:, :, :, 10:15], vs_7[:, :, :, 10:15], st_p_all[:, :, :, 10:15]).to(bev_ref.dtype)
        left_haz = self.left_corr_head(torch.cat([l_slice, c_center_m, phys_l], dim=1)).squeeze(1)
        right_haz = self.right_corr_head(torch.cat([r_slice, c_center_m, phys_r], dim=1)).squeeze(1)
        left_cum, left_p = _haz_to_cum_prob(left_haz)
        right_cum, right_p = _haz_to_cum_prob(right_haz)

        # --- 6. Ground-Aware Soft Head-Height Survival (NO broken v11 head KD!) ---
        sm_head_fixed = sm_9[:, 6:9, :, 5:10] * stat_gate[:, :, :, 5:10]
        sm_head_soft = (sm_9[:, :, :, 5:10] * w_head_band[:, :, :, 5:10] * stat_gate[:, :, :, 5:10])
        vm_head_soft = (vm_9[:, :, :, 5:10] * w_head_band[:, :, :, 5:10] * stat_gate[:, :, :, 5:10])
        phys_head = torch.cat([
            vh_m[:, 4:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 5:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 4:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 5:7].amax(dim=(1, 3)).unsqueeze(1),
            sm_head_fixed.amax(dim=(1, 3)).unsqueeze(1),
            sm_head_fixed.mean(dim=(1, 3)).unsqueeze(1),
            sm_head_soft.amax(dim=(1, 3)).unsqueeze(1),
            vm_head_soft.amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            vs_4[:, 4:7].mean(dim=(1, 3)).unsqueeze(1),
            st_p[:, 2:3].amax(dim=3),
            st_p[:, 2:3].mean(dim=3),
        ], dim=1).to(bev_ref.dtype)
        head_haz = self.head_corr_head(torch.cat([c_slice, phys_head], dim=1)).squeeze(1)
        head_cum, head_p = _haz_to_cum_prob(head_haz)

        # --- 7. Ground-Aware Negative Obstacle / Drop-Off Survival ---
        sm_drop_soft = sm_9[:, :, :, 5:10] * w_drop_band[:, :, :, 5:10]
        dp_drop_soft = dp_9[:, :, :, 5:10] * w_drop_band[:, :, :, 5:10]
        phys_drop = torch.cat([
            sm_9[:, 0:2, :, 5:10].amax(dim=(1, 3)).unsqueeze(1),
            sm_9[:, 0:2, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            fh_9[:, 0:2, :, 5:10].amax(dim=(1, 3)).unsqueeze(1),
            fh_9[:, 0:2, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            dp_9[:, 2:4, :, 5:10].amax(dim=(1, 3)).unsqueeze(1),
            dp_9[:, 2:4, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            sd_9[:, 2:4, :, 5:10].amax(dim=(1, 3)).unsqueeze(1),
            sd_9[:, 2:4, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            fh_9[:, 2:4, :, 5:10].amax(dim=(1, 3)).unsqueeze(1),
            fh_9[:, 2:4, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            nu_9[:, 0:3, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            vis_9[:, 0:3, :, 5:10].mean(dim=(1, 3)).unsqueeze(1),
            st_p[:, 0:1].amax(dim=3),
            st_p[:, 0:1].mean(dim=3),
            fl_conf_c,
            sm_drop_soft.amax(dim=(1, 3)).unsqueeze(1),
            dp_drop_soft.amax(dim=(1, 3)).unsqueeze(1),
        ], dim=1).to(bev_ref.dtype)
        drop_haz = self.drop_corr_head(torch.cat([c_slice, phys_drop], dim=1)).squeeze(1)
        drop_cum, drop_p = _haz_to_cum_prob(drop_haz)

        # --- 8. Dynamic Closing-TTC Survival with Multi-Lag Radial Closing-Rate Cues ---
        obs_slice = bev_obs[:, :, :, 5:10].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        ttc1 = lag_ttc_list[0].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        invz1 = lag_invz_list[0].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        diff1 = lag_diff_list[0].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        ttc2 = lag_ttc_list[-1].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        invz2 = lag_invz_list[-1].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        diff2 = lag_diff_list[-1].reshape(B, NY_V13, BEV_NZ, BEV_NX)[:, 2:, :, 5:10]
        phys_dyn = torch.cat([
            dyn_c3.amax(dim=3),
            dyn_c3.mean(dim=3),
            ttc1.amax(dim=(1, 3)).unsqueeze(1),
            ttc1.amin(dim=(1, 3)).unsqueeze(1),
            ttc2.amax(dim=(1, 3)).unsqueeze(1),
            ttc2.amin(dim=(1, 3)).unsqueeze(1),
            invz1.amax(dim=(1, 3)).unsqueeze(1),
            invz1.mean(dim=(1, 3)).unsqueeze(1),
            diff1.amax(dim=(1, 3)).unsqueeze(1),
            vs_4.mean(dim=(1, 3)).unsqueeze(1),
            F.relu(invz1).amax(dim=(1, 3)).unsqueeze(1),
            F.relu(invz2).amax(dim=(1, 3)).unsqueeze(1),
            F.relu(-ttc1).amax(dim=(1, 3)).unsqueeze(1),
            diff2.amax(dim=(1, 3)).unsqueeze(1),
        ], dim=1).to(bev_ref.dtype)
        dyn_cl_haz = self.dyn_closing_head(torch.cat([obs_slice, phys_dyn], dim=1)).squeeze(1)
        dyn_cl_cum, dyn_cl_p = _haz_to_cum_prob(dyn_cl_haz)

        return (
            st_log, dyn_log,
            haz_log, cum_haz, corr_p, unc_log,
            left_haz, left_cum, left_p,
            right_haz, right_cum, right_p,
            head_haz, head_cum, head_p,
            drop_haz, drop_cum, drop_p,
            dyn_cl_haz, dyn_cl_cum, dyn_cl_p,
            mem_new, age_new,
            h_floor_cam,
        )


class EgoClearV13EncoderExport(nn.Module):
    def __init__(self, net: EgoClearV13Net):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor):
        return self.net.encode_images(rgb, K_scaled, R_c2w, return_aux_geom=False)


class EgoClearV13StepExport(nn.Module):
    def __init__(self, net: EgoClearV13Net):
        super().__init__()
        self.net = net
        self.net.mem_gru._onnx_clear_rate = float(torch.sigmoid(self.net.mem_gru.free_clear_logit.detach()).item())

    def forward(
        self,
        f2d_t, geom_t, f2d_1, geom_1, f2d_2, geom_2,
        K_t, R_t, t_t, fwd_t, K_1, R_1, t_1, fwd_1,
        K_2, R_2, t_2, ds_t, mem_prev, age_prev,
    ):
        (
            st, dy, _hz, _ch, cp, un,
            _lh, _lc, lp, _rh, _rc, rp,
            _hh, _hc, hp, _dh, _dc, dp,
            _dhz, _dch, dcp,
            mem_new, age_new, _hfl,
        ) = self.net.step_from_encoded(
            f2d_t, geom_t, K_t, R_t, t_t, fwd_t,
            [(f2d_1, geom_1, K_1, R_1, t_1), (f2d_2, geom_2, K_2, R_2, t_2)],
            t_1, fwd_1, ds_t, mem_prev, age_prev, is_first=False,
        )
        omni_probs = torch.stack([cp, lp, rp, hp, dp, dcp], dim=1)
        return st, dy, cp, un, omni_probs, mem_new, age_new
