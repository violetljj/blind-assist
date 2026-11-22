"""EgoClear-v8: Full Physics-Grounded Ego-Corridor Architecture.

Unifies all 5 mathematical & geometric breakthroughs co-designed in Phase 11:
  1. Frozen MoGe-2-ViT-S-Normal (35M MIT) + FP16-Bounded 4x4 Tile-Capped Closed-Form Shift (Problem 5).
  2. Optional Zero-Initialized Bounded Geometric Residual Head distilled from 331M MoGe-2-ViT-L-Normal (Problem D):
       log z' = log z_S + delta_z * tanh(r_z),  n' = normalize(n_S + delta_n * tanh(r_n))
     Initializes to exact MoGe-2-ViT-S output at step 0, cleanly distilling MoGe-2-ViT-L-Normal sharpness
     on thin obstacles, depth boundaries, and dark scenes without losing frozen ViT-S generalization.
  3. Optional Smooth QAIRT-HTP-Native Support-Plane Ground-Relative Slope-Aware Sampling (Problem B):
     Estimates tangent-plane camera height h_cam = -(n . X) / n_up and forward slope s_fwd with 4x4 block-capped
     temperature-sharpened softmax voting (ZERO ArgMax, ZERO boolean masks in encoder!) + unimodality confidence
     c_t in [0, 1] + rate-limited temporal filter |h_t - h_prev| <= c_t * v_h * dt, smoothly blending between
     ground-relative [0.18..1.88m] + slope*z and camera-relative fallback [-1.50..+0.60m].
  4. Explicit Free-Space Ray-Pass-Through Clearing vs Out-of-FOV Persistence in ConvGRU (Problem 4):
     Feeds free_bev (confident ray passing *through* voxel to farther surface) explicitly into FreeSpaceDynamicConvGRU
     with a learnable bounded clearing gate so cleared voxels reset cleanly while brief blind spots persist smoothly.
  5. Complete-Event SmoothMax + Per-Domain Proxy-Lagrangian in FP32 Cumulative Hazard space Lambda_t (Problem A).
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
    NY,
    RGB_H,
    RGB_W,
    Y_SLICES,
    ConvBNReLU,
    ResBlock2D,
)
from egoclear_v7_model import FEAT_H, FEAT_W, GEOM_CH, _build_moge_uv_levels

# Ground-relative vertical slices above estimated local walkable floor (meters): covers exact LOW/BODY/HEAD [0.15..1.90m]
GROUND_Y_SLICES = (0.18, 0.44, 0.72, 1.00, 1.30, 1.60, 1.88)


class FreeSpaceDynamicConvGRU(nn.Module):
    """Visibility-, dynamic-, AND free-space-ray-pass-through-gated ConvGRU on the 24x15 BEV grid (Codex Problem 4)."""

    def __init__(self, c_mem: int, c_obs: int):
        super().__init__()
        # warped_mem (c_mem) + obs (c_obs) + diff (c_mem) + vis_frac (1) + age_decay (1) + dyn_prob (1) + free_ev (1)
        c_in = c_mem + c_obs + c_mem + 4
        self.conv_zr = nn.Conv2d(c_in, 2 * c_mem, 3, padding=1)
        self.conv_h = nn.Conv2d(c_mem + c_obs + 2, c_mem, 3, padding=1)
        # Learnable bounded clearing strength initialized gently (~0.50) so training calibrates exact pass-through clearing
        self.free_clear_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        mem_w: torch.Tensor,
        obs: torch.Tensor,
        vis_frac: torch.Tensor,
        age_decay: torch.Tensor,
        dyn_prob: torch.Tensor,
        free_ev: torch.Tensor,
    ) -> torch.Tensor:
        clear_strength = 0.85 * torch.sigmoid(self.free_clear_logit).to(mem_w.dtype)
        mem_clean = mem_w * (1.0 - clear_strength * free_ev)
        diff = (obs - mem_clean).abs()
        x = torch.cat([mem_clean, obs, diff, vis_frac, age_decay, dyn_prob, free_ev], dim=1)
        zr = self.conv_zr(x)
        z_g, r_g = torch.chunk(zr, 2, dim=1)
        z = torch.sigmoid(z_g) * (vis_frac * (1.0 - 0.85 * dyn_prob) + 0.35 * free_ev).clamp(0.0, 1.0)
        r = torch.sigmoid(r_g) * (1.0 - 0.65 * free_ev)
        h_tilde = torch.tanh(self.conv_h(torch.cat([r * mem_clean, obs, dyn_prob, free_ev], dim=1)))
        return (1.0 - z) * (mem_clean * age_decay) + z * h_tilde


class EgoClearV8Net(nn.Module):
    """EgoClear-v8 unifying MoGe-2-ViT-S + Optional ViT-L Bounded Residual + Ground-Relative Slope Sampling + Free-Space GRU."""

    def __init__(
        self,
        moge_id: str = "Ruicheng/moge-2-vits-normal",
        c_feat: int = 64,
        c_bev: int = 128,
        lags: tuple[int, ...] = (1, 2),
        freeze_moge: bool = True,
        use_geom_residual: bool = True,
        use_ground_relative: bool = False,
        delta_z_max: float = 0.35,
        delta_n_max: float = 0.35,
        rgb_feat_drop_p: float = 0.35,
    ):
        super().__init__()
        from moge.model.v2 import MoGeModel

        self.moge_id = moge_id
        self.c_feat = c_feat
        self.c_bev = c_bev
        self.lags = tuple(lags)
        self.freeze_moge = freeze_moge
        self.use_geom_residual = use_geom_residual
        self.use_ground_relative = use_ground_relative
        self.delta_z_max = delta_z_max
        self.delta_n_max = delta_n_max
        self.rgb_feat_drop_p = rgb_feat_drop_p

        self.moge = MoGeModel.from_pretrained(moge_id)
        if hasattr(self.moge, "enable_onnx_compatible_mode"):
            self.moge.enable_onnx_compatible_mode()
        elif hasattr(self.moge, "encoder") and hasattr(self.moge.encoder, "backbone"):
            setattr(self.moge.encoder.backbone, "onnx_compatible_mode", True)
        if freeze_moge:
            for p in self.moge.parameters():
                p.requires_grad = False
            self.moge.eval()

        self.base_h = RGB_H // 14
        self.base_w = RGB_W // 14
        for lv, uv in enumerate(_build_moge_uv_levels(self.base_h, self.base_w, RGB_W / RGB_H)):
            self.register_buffer(f"moge_uv_{lv}", uv, persistent=False)

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("up_w", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).view(1, 1, 3), persistent=False)

        u = (torch.arange(FEAT_W, dtype=torch.float32) + 0.5) * (RGB_W / FEAT_W)
        v = (torch.arange(FEAT_H, dtype=torch.float32) + 0.5) * (RGB_H / FEAT_H)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("feat_uu", uu.unsqueeze(0), persistent=False)
        self.register_buffer("feat_vv", vv.unsqueeze(0), persistent=False)

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", kx, persistent=False)
        self.register_buffer("sobel_y", ky, persistent=False)

        self.neck1_proj = ConvBNReLU(256, 96, 1, 0)

        # Zero-initialized bounded geometric residual head (Codex Problem D): exact ViT-S identity at epoch 0!
        if use_geom_residual:
            self.geom_res_head = nn.Sequential(
                ConvBNReLU(96 + 128 + 5, 96, 3, 1),
                ResBlock2D(96),
                nn.Conv2d(96, 4, 1),  # [r_logz, r_nx, r_ny, r_nz]
            )
            nn.init.zeros_(self.geom_res_head[-1].weight)
            nn.init.zeros_(self.geom_res_head[-1].bias)
        else:
            self.geom_res_head = None

        self.fuse_2d = nn.Sequential(
            ConvBNReLU(96 + 128 + GEOM_CH + 3, 128, 3, 1),
            nn.Dropout2d(0.08),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys_cam = torch.tensor(Y_SLICES, dtype=torch.float32)
        ys_gnd = torch.tensor(GROUND_Y_SLICES, dtype=torch.float32)
        gy_c, gz, gx = torch.meshgrid(ys_cam, zs, xs, indexing="ij")
        gy_g, _, _ = torch.meshgrid(ys_gnd, zs, xs, indexing="ij")
        self.register_buffer("grid_x", gx.reshape(-1), persistent=False)
        self.register_buffer("grid_z", gz.reshape(-1), persistent=False)
        self.register_buffer("grid_y_cam", gy_c.reshape(-1), persistent=False)
        self.register_buffer("grid_y_gnd", gy_g.reshape(-1), persistent=False)
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)
        self.register_buffer("h_vote_grid", torch.linspace(0.40, 2.00, 33, dtype=torch.float32), persistent=False)

        # c_feat (64) + geom (8) + vis (1) + uv_invz (3) + phys_hits (5: sdf, surf_hit, vert_hit, floor_hit, free_ev) + 5*len(lags)
        c_pt = c_feat + GEOM_CH + 1 + 3 + 5 + 5 * len(self.lags)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        # +2 bev_pos + 2 support-plane state (h_cam_norm, plane_conf)
        self.bev_enc = nn.Sequential(
            ConvBNReLU(48 * NY + 4, c_bev, 3, 1),
            ResBlock2D(c_bev),
        )
        self.head_dynamic_obs = nn.Sequential(
            ConvBNReLU(c_bev, 64, 3, 1),
            nn.Conv2d(64, 3, 1),
        )
        self.mem_gru = FreeSpaceDynamicConvGRU(c_bev, c_bev)
        self.bev_refine = nn.Sequential(
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
        )
        self.head_static = nn.Conv2d(c_bev, 3, 1)
        # +18 explicit 3D physical corridor channels (LOW/BODY/HEAD vert_hit max/mean, floor, free-air, vis, static-max/mean)!
        self.corr_head = nn.Sequential(
            nn.Conv1d(c_bev * 7 + 18, 160, 3, padding=1, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.Conv1d(160, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 2, 1),
        )
        with torch.no_grad():
            self.corr_head[-1].bias[0].fill_(-3.5)
            self.corr_head[-1].bias[1].fill_(-1.5)

    def warm_start_from_v75(self, v75_head_sd: dict[str, torch.Tensor]) -> int:
        """Transfers all compatible weights from trained EgoClear-v7.5, zero-padding the 4 new physics channels cleanly."""
        own = self.state_dict()
        loaded = 0
        with torch.no_grad():
            for k, v in v75_head_sd.items():
                if k not in own or k.startswith("moge."):
                    continue
                tgt = own[k]
                if tgt.shape == v.shape:
                    tgt.copy_(v.to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif k == "pt_reduce.0.weight" and v.shape[1] == 95 and tgt.shape[1] == 96:
                    tgt.zero_()
                    tgt[:, :80].copy_(v[:, :80].to(tgt.device, dtype=tgt.dtype))
                    tgt[:, 81:].copy_(v[:, 80:].to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif k == "bev_enc.0.0.weight" and v.shape[1] == 338 and tgt.shape[1] == 340:
                    tgt.zero_()
                    tgt[:, :338].copy_(v.to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif k == "mem_gru.conv_zr.weight" and v.shape[1] == 387 and tgt.shape[1] == 388:
                    tgt.zero_()
                    tgt[:, :387].copy_(v.to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif k == "mem_gru.conv_h.weight" and v.shape[1] == 257 and tgt.shape[1] == 258:
                    tgt.zero_()
                    tgt[:, :257].copy_(v.to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
                elif k == "corr_head.0.weight" and v.shape[1] == self.c_bev * 7 and tgt.shape[1] == self.c_bev * 7 + 18:
                    tgt.zero_()
                    tgt[:, : self.c_bev * 7].copy_(v.to(tgt.device, dtype=tgt.dtype))
                    loaded += 1
        return loaded

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_moge:
            self.moge.eval()
        return self

    def encode_images(
        self,
        rgb: torch.Tensor,
        K_scaled: torch.Tensor,
        R_c2w: torch.Tensor,
        return_aux_geom: bool = False,
        rgb_keep_mask: torch.Tensor | None = None,
    ):
        N = rgb.shape[0]
        rgb_01 = (rgb.float() * self.im_std + self.im_mean).clamp(0.0, 1.0)

        ctx = torch.no_grad() if self.freeze_moge else torch.enable_grad()
        with ctx:
            feat_bb, cls_token = self.moge.encoder(rgb_01, self.base_h, self.base_w, return_class_token=True)
            feats_in = [feat_bb, None, None, None, None]
            for lv in range(5):
                uv = getattr(self, f"moge_uv_{lv}").to(dtype=feat_bb.dtype).expand(N, -1, -1, -1)
                feats_in[lv] = uv if feats_in[lv] is None else torch.cat([feats_in[lv], uv], dim=1)
            neck_feats = self.moge.neck(feats_in)
            pts_raw = self.moge.points_head(neck_feats)[-1]
            nrm_raw = self.moge.normal_head(neck_feats)[-1]
            msk_raw = self.moge.mask_head(neck_feats)[-1]
            m_scale = self.moge.scale_head(cls_token).squeeze(1).exp().float()

            pts_84 = F.interpolate(pts_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            nrm_84 = F.interpolate(nrm_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            msk_84 = torch.sigmoid(F.interpolate(msk_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False))
            pts_hwc = self.moge._remap_points(pts_84.permute(0, 2, 3, 1))
            px, py, pz = pts_hwc[..., 0], pts_hwc[..., 1], pts_hwc[..., 2]
            nrm_unit = F.normalize(nrm_84, dim=1, eps=1e-6)
            nf1 = neck_feats[1].detach()
            nf2 = neck_feats[2].detach()

        fx = K_scaled[:, 0].float().view(N, 1, 1).clamp(min=10.0)
        fy = K_scaled[:, 1].float().view(N, 1, 1).clamp(min=10.0)
        cx = K_scaled[:, 2].float().view(N, 1, 1)
        cy = K_scaled[:, 3].float().view(N, 1, 1)
        rx = (self.feat_uu - cx) / fx
        ry = (self.feat_vv - cy) / fy

        # FP16-bounded + 4x4 tile-capped closed-form shift (Problem 5)
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
        z_base = ((pz + shift).clamp(min=0.15, max=25.0) * m_scale.view(N, 1, 1)).clamp(min=0.15, max=20.0).unsqueeze(1)

        nf1_up = F.interpolate(self.neck1_proj(nf1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)

        # Optional zero-initialized bounded geometric residual head (Codex Problem D!)
        if self.geom_res_head is not None:
            res_in = torch.cat([nf1_up, nf2.to(nf1_up.dtype), torch.log(z_base).to(nf1_up.dtype), nrm_unit.to(nf1_up.dtype), msk_84.to(nf1_up.dtype)], dim=1)
            res_out = self.geom_res_head(res_in).float()
            d_logz = self.delta_z_max * torch.tanh(res_out[:, 0:1])
            d_nrm = self.delta_n_max * torch.tanh(res_out[:, 1:4])
            z_m = (z_base * torch.exp(d_logz)).clamp(min=0.15, max=20.0)
            nrm_ref = F.normalize(nrm_unit + d_nrm, dim=1, eps=1e-6)
        else:
            d_logz = torch.zeros_like(z_base)
            d_nrm = torch.zeros_like(nrm_unit)
            z_m = z_base
            nrm_ref = nrm_unit

        inv_z = 1.0 / z_m.clamp(min=0.25)
        log_z = torch.log1p(z_m)
        dz_gx = F.conv2d(inv_z, self.sobel_x, padding=1)
        dz_gy = F.conv2d(inv_z, self.sobel_y, padding=1)

        r_up_c = R_c2w[:, 2, :].float().view(N, 3, 1, 1)
        n_up = (nrm_ref * r_up_c).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        verticality = (1.0 - n_up.abs()).clamp(0.0, 1.0)
        geom_cues = torch.cat([z_m, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, msk_84], dim=1)

        rays = torch.stack([rx, ry, torch.ones_like(rx)], dim=-1)
        rays_u = F.normalize(rays, dim=-1)
        grav_dot = (rays_u * (-R_c2w[:, 2, :].float())[:, None, None, :]).sum(dim=-1)
        ray_geo = torch.stack([rx, ry, grav_dot], dim=1)
        if rgb_keep_mask is not None:
            km = rgb_keep_mask.to(nf1_up.dtype).view(N, 1, 1, 1)
            nf1_in = nf1_up * km
            nf2_in = nf2.to(nf1_up.dtype) * km
        else:
            nf1_in = nf1_up
            nf2_in = nf2.to(nf1_up.dtype)
        f2d = self.fuse_2d(torch.cat([nf1_in, nf2_in, geom_cues.to(nf1_up.dtype), ray_geo.to(nf1_up.dtype)], dim=1))

        # 100% smooth, branch-free, QAIRT-HTP-native support-plane estimator (ZERO ArgMax, ZERO boolean ops!)
        z_hw = z_m.squeeze(1)
        n_up_hw = n_up.squeeze(1)
        dot_nx = nrm_ref[:, 0] * (rx * z_hw) + nrm_ref[:, 1] * (ry * z_hw) + nrm_ref[:, 2] * z_hw
        h_i = (-dot_nx / n_up_hw.clamp(min=0.80)).clamp(0.30, 2.20)
        g_n = F.relu((n_up_hw - 0.80) / 0.20).clamp(max=1.0)
        g_m = F.relu((msk_hw - 0.40) / 0.60).clamp(max=1.0)
        g_z = F.relu((z_hw - 0.30) / 0.20).clamp(max=1.0) * F.relu((4.80 - z_hw) / 0.80).clamp(max=1.0)
        w_f = g_n * g_m * g_z * (1.0 / z_hw.clamp(min=0.5))

        wt = w_f.reshape(N, 4, 21, 4, 37)
        ts = wt.sum(dim=(2, 4), keepdim=True)
        wc = (wt * (15.0 / ts.clamp(min=15.0))).reshape(N, FEAT_H, FEAT_W)
        ts_flat = ts.squeeze(4).squeeze(2)
        blk_act = (ts_flat / (ts_flat + 1.5)).sum(dim=(1, 2))

        diff_h = (h_i.unsqueeze(1) - self.h_vote_grid.view(1, 33, 1, 1)) / 0.08
        votes = (torch.exp(-0.5 * (diff_h.clamp(-5.0, 5.0) ** 2)) * wc.unsqueeze(1)).sum(dim=(2, 3))
        v_sum = votes.sum(dim=1, keepdim=True).clamp(min=1e-4)
        tot_w = wc.sum(dim=(1, 2)).clamp(min=1e-4)

        sharp_w = F.softmax(25.0 * (votes / v_sum), dim=1)
        h_obs = (sharp_w * self.h_vote_grid.unsqueeze(0)).sum(dim=1)
        near_p = torch.exp(-0.5 * (((self.h_vote_grid.unsqueeze(0) - h_obs.unsqueeze(1)) / 0.12).clamp(-5.0, 5.0) ** 2))
        peak_conc = ((votes * near_p).sum(dim=1) / v_sum.squeeze(1)).clamp(0.0, 1.0)
        conf_obs = ((tot_w / (tot_w + 18.0)) * (blk_act / 3.0).clamp(0.0, 1.0) * ((peak_conc - 0.35) / 0.45).clamp(0.0, 1.0)).clamp(0.0, 1.0)

        inlier_w = wc * torch.exp(-0.5 * (((h_i - h_obs.view(N, 1, 1)) / 0.12).clamp(-5.0, 5.0) ** 2))
        n_avg = (nrm_ref * inlier_w.unsqueeze(1)).sum(dim=(2, 3)) / inlier_w.sum(dim=(1, 2)).clamp(min=1e-4).unsqueeze(1)
        nw_avg = torch.bmm(R_c2w.float(), n_avg.unsqueeze(2)).squeeze(2)
        plane_obs = torch.stack([h_obs, conf_obs, nw_avg[:, 0] / nw_avg[:, 2].clamp(min=0.70), nw_avg[:, 1] / nw_avg[:, 2].clamp(min=0.70)], dim=1)

        if return_aux_geom:
            return f2d, geom_cues.to(f2d.dtype), plane_obs.to(f2d.dtype), z_m, nrm_ref, msk_84, d_logz, d_nrm
        return f2d, geom_cues.to(f2d.dtype), plane_obs.to(f2d.dtype)

    def _project_and_sample(self, f2d, geom, pts_w, R_c2w, t_w, K_scaled):
        diff = pts_w - t_w[:, None, :]
        p_c = torch.bmm(diff, R_c2w.float())
        zc = p_c[..., 2]
        in_front = zc > 0.15
        zc_safe = zc.clamp(min=0.15)
        u = K_scaled[:, 0:1].float() * (p_c[..., 0] / zc_safe) + K_scaled[:, 2:3].float()
        v = K_scaled[:, 1:2].float() * (p_c[..., 1] / zc_safe) + K_scaled[:, 3:4].float()
        u_norm = 2.0 * (u / (RGB_W - 1.0)) - 1.0
        v_norm = 2.0 * (v / (RGB_H - 1.0)) - 1.0
        vis = in_front & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1).to(f2d.dtype)
        fg = torch.cat([f2d, geom.to(f2d.dtype)], dim=1)
        samp_fg = F.grid_sample(fg, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(2)
        vis_f = vis.unsqueeze(1).to(f2d.dtype)
        samp_fg = samp_fg * vis_f
        uv_invz = torch.stack([u_norm.clamp(-1.5, 1.5), v_norm.clamp(-1.5, 1.5), 1.0 / zc_safe], dim=1).to(f2d.dtype)
        return samp_fg[:, : self.c_feat, :], samp_fg[:, self.c_feat :, :], vis_f, uv_invz, zc_safe.unsqueeze(1).to(f2d.dtype)

    def _warp_bev(self, mem_prev, t_prev, fwd_prev, t_cur, fwd_cur):
        right_cur = torch.stack([fwd_cur[:, 1], -fwd_cur[:, 0]], dim=-1)
        right_prev = torch.stack([fwd_prev[:, 1], -fwd_prev[:, 0]], dim=-1)
        bx, bz = self.bev_xz[..., 0][None], self.bev_xz[..., 1][None]
        qw = t_cur[:, None, None, :2] + bx[..., None] * right_cur[:, None, None, :] + bz[..., None] * fwd_cur[:, None, None, :2]
        rel_prev = qw - t_prev[:, None, None, :2]
        x_prev = (rel_prev * right_prev[:, None, None, :]).sum(dim=-1)
        z_prev = (rel_prev * fwd_prev[:, None, None, :2]).sum(dim=-1)
        x_lo, x_hi = BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX
        z_lo, z_hi = BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ
        gx = 2.0 * (x_prev - x_lo) / (x_hi - x_lo) - 1.0
        gz = 2.0 * (z_prev - z_lo) / (z_hi - z_lo) - 1.0
        return F.grid_sample(mem_prev, torch.stack([gx, gz], dim=-1).to(mem_prev.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)

    def step_from_encoded(
        self,
        f2d_t, geom_t, plane_obs_t,
        K_t, R_t, t_t, fwd_t,
        lag_packs,
        t_prev, fwd_prev, ds_t,
        mem_prev, age_prev, h_state_prev,  # h_state_prev: (B, 2) [h_cam_est, h_conf]
        is_first: bool,
    ):
        B = f2d_t.shape[0]
        device, dtype = f2d_t.device, f2d_t.dtype

        # Rate-limited temporal camera-height filter (Codex Problem B: |h_t - h_prev| <= c_t * v_h * dt!)
        h_obs = plane_obs_t[:, 0].float()
        c_obs = plane_obs_t[:, 1].float()
        s_fwd_obs = (-(plane_obs_t[:, 2].float() * fwd_t[:, 0].float() + plane_obs_t[:, 3].float() * fwd_t[:, 1].float())).clamp(-0.22, 0.22) * c_obs

        if is_first:
            h_cur = torch.where(c_obs > 0.25, h_obs, torch.full_like(h_obs, 1.35))
            c_cur = c_obs
        else:
            h_prev = h_state_prev[:, 0].float()
            c_prev = h_state_prev[:, 1].float() * 0.94
            max_step = c_obs * 0.08
            h_cur = (h_prev + (h_obs - h_prev).clamp(-max_step, max_step)).clamp(0.45, 2.00)
            c_cur = torch.maximum(c_prev, c_obs).clamp(0.0, 1.0)
        h_state_new = torch.stack([h_cur, c_cur], dim=1).to(dtype)

        gx = self.grid_x[None, :, None]
        gz = self.grid_z[None, :, None]
        if self.use_ground_relative:
            blend = c_cur.view(B, 1, 1)
            y_gnd_rel = self.grid_y_gnd[None, :, None] - h_cur.view(B, 1, 1) + s_fwd_obs.view(B, 1, 1) * gz
            gy = blend * y_gnd_rel + (1.0 - blend) * self.grid_y_cam[None, :, None]
        else:
            gy = self.grid_y_cam[None, :, None]

        right_t = torch.stack([fwd_t[:, 1], -fwd_t[:, 0], torch.zeros_like(fwd_t[:, 0])], dim=-1)
        pts_w = t_t[:, None, :] + gx * right_t[:, None, :] + gz * fwd_t[:, None, :] + gy * self.up_w

        samp_f, samp_g, vis_t, uv_invz_t, z_vox = self._project_and_sample(f2d_t, geom_t, pts_w, R_t, t_t, K_t)

        z_pred = samp_g[:, 0:1, :]
        n_up_v = samp_g[:, 5:6, :]
        vert_v = samp_g[:, 6:7, :]
        mask_v = samp_g[:, 7:8, :]
        rel_diff = (z_pred - z_vox) / z_vox.clamp(min=0.25)
        signed_sdf = rel_diff.clamp(-1.5, 1.5) * vis_t
        surf_hit = torch.exp(-((rel_diff / 0.18).clamp(-6.0, 6.0) ** 2)) * mask_v * vis_t
        vert_hit = surf_hit * vert_v
        floor_hit = surf_hit * F.relu(n_up_v)
        # Explicit Free-Space Ray-Pass-Through evidence (Codex Problem 4: reliable surface strictly behind voxel!)
        free_ev_vox = torch.sigmoid(((rel_diff - 0.28) / 0.08).clamp(-10.0, 10.0)) * F.relu((mask_v - 0.50) / 0.50) * vis_t

        pt_parts = [samp_f, samp_g, vis_t, uv_invz_t, signed_sdf, surf_hit, vert_hit, floor_hit, free_ev_vox]
        ft_norm = F.normalize(samp_f, dim=1, eps=1e-5)
        invz_t_v = samp_g[:, 1:2, :]
        for lp in lag_packs:
            if lp is not None:
                f_p, g_p, K_p, R_p, t_p = lp
                sf_p, sg_p, vis_p, _, _ = self._project_and_sample(f_p, g_p, pts_w, R_p, t_p, K_p)
                both_v = vis_t * vis_p
                fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
                cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
                l1_diff = (samp_f - sf_p).abs().mean(dim=1, keepdim=True) * both_v
                invz_diff = (invz_t_v - sg_p[:, 1:2, :]).clamp(-2.0, 2.0) * both_v
                base = torch.linalg.norm(t_t - t_p, dim=-1).clamp(max=2.5)
                base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(dtype) * both_v
                pt_parts.extend([cos_sim, l1_diff, invz_diff, both_v, base_map])
            else:
                pt_parts.append(torch.zeros(B, 5, pts_w.shape[1], device=device, dtype=dtype))

        bev_pos = torch.stack([
            self.bev_xz[..., 0] / BEV_X_MAX,
            (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
        ], dim=0).unsqueeze(0).expand(B, -1, -1, -1).to(dtype)
        plane_map = torch.stack([(h_cur - 1.20) / 0.60, c_cur], dim=1).view(B, 2, 1, 1).expand(B, 2, BEV_NZ, BEV_NX).to(dtype)

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
        pt_red = self.pt_reduce(pt_feat).reshape(B, 48 * NY, BEV_NZ, BEV_NX)
        bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos, plane_map], dim=1))
        dyn_log = self.head_dynamic_obs(bev_obs)
        dyn_prob = torch.sigmoid(dyn_log).max(dim=1, keepdim=True).values.detach()

        vis_bev = vis_t.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True)
        free_bev = (free_ev_vox.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True) * (vis_bev > 0.35).to(dtype)).clamp(0.0, 1.0)

        if not is_first:
            mem_w = self._warp_bev(mem_prev, t_prev, fwd_prev, t_t, fwd_t)
            age_w = self._warp_bev(age_prev, t_prev, fwd_prev, t_t, fwd_t)
            ds = (ds_t.clamp(0.0, 2.0) + 0.04)[:, None, None, None].to(dtype)
            age_w = (age_w + ds).clamp(max=10.0)
            age_decay = torch.exp(-age_w / 4.0)
            mem_new = self.mem_gru(mem_w, bev_obs, vis_bev, age_decay, dyn_prob, free_bev)
            age_new = torch.where((vis_bev > 0.05) & (dyn_prob < 0.5), torch.zeros_like(age_w), age_w)
        else:
            mem_new = bev_obs * (1.0 - 0.5 * dyn_prob)
            age_new = torch.where(vis_bev > 0.05, torch.zeros_like(age_prev), age_prev)

        bev_ref = self.bev_refine(mem_new)
        st_log = self.head_static(bev_ref)

        # Direct 18-channel 3D Physical Corridor Shortcut (preserves LOW/BODY/HEAD thin poles, curbs & overhead clearance!)
        vh_4d = vert_hit.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        fh_4d = floor_hit.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        fr_4d = free_ev_vox.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        vs_4d = vis_t.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        st_p = torch.sigmoid(st_log[:, :, :, 5:10])
        phys_corr = torch.cat([
            vh_4d[:, 0:2].amax(dim=(1, 3), keepdim=False).unsqueeze(1),  # max_vert LOW
            vh_4d[:, 2:5].amax(dim=(1, 3), keepdim=False).unsqueeze(1),  # max_vert BODY
            vh_4d[:, 5:7].amax(dim=(1, 3), keepdim=False).unsqueeze(1),  # max_vert HEAD
            vh_4d[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),                 # mean_vert LOW
            vh_4d[:, 2:5].mean(dim=(1, 3)).unsqueeze(1),                 # mean_vert BODY
            vh_4d[:, 5:7].mean(dim=(1, 3)).unsqueeze(1),                 # mean_vert HEAD
            fh_4d[:, 0:2].amax(dim=(1, 3), keepdim=False).unsqueeze(1),  # max_floor LOW
            fh_4d[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),                 # mean_floor LOW
            fr_4d[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),                 # mean_free LOW
            fr_4d[:, 2:5].mean(dim=(1, 3)).unsqueeze(1),                 # mean_free BODY
            fr_4d[:, 5:7].mean(dim=(1, 3)).unsqueeze(1),                 # mean_free HEAD
            vs_4d.mean(dim=(1, 3)).unsqueeze(1),                         # mean_vis corr
            st_p.amax(dim=3),                                            # BEV static prob max (3 bands)
            st_p.mean(dim=3),                                            # BEV static prob mean (3 bands)
        ], dim=1).to(bev_ref.dtype)  # (B, 18, BEV_NZ)

        c_slice = bev_ref[:, :, :, 5:10].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        c_left = bev_ref[:, :, :, :5].mean(dim=3)
        c_right = bev_ref[:, :, :, 10:].mean(dim=3)
        corr_out = self.corr_head(torch.cat([c_slice, c_left, c_right, phys_corr], dim=1))
        haz_log = corr_out[:, 0, :]
        unc_log = corr_out[:, 1, :]
        cum_haz = torch.cumsum(F.softplus(haz_log.float()), dim=-1)
        corr_p = 1.0 - torch.exp((-cum_haz).clamp(min=-20.0))
        return st_log, dyn_log, haz_log, cum_haz, corr_p, unc_log, mem_new, age_new, h_state_new

    def forward_sequence(
        self,
        rgb, K_scaled, R_c2w, t_w, fwd_w, seg_len,
        return_aux_geom: bool = False,
        mask_lags: tuple[int, ...] = (),
    ):
        B, T = rgb.shape[:2]
        device = rgb.device
        bt = B * T
        rgb_f, K_f, R_f = rgb.reshape(bt, 3, RGB_H, RGB_W), K_scaled.reshape(bt, 4), R_c2w.reshape(bt, 3, 3)

        # Clip-level RGB-neck modality dropout during training: forces pure 3D metric+normal generalization to Aria/CODa!
        keep_flat = None
        if self.training and self.rgb_feat_drop_p > 0.0:
            keep_clip = (torch.rand(B, 1, device=device) >= self.rgb_feat_drop_p).float()
            keep_flat = keep_clip.expand(B, T).reshape(bt)

        chunk = 96
        f_l, g_l, p_l, zm_l, nr_l, mk_l, dlz_l, dnr_l = [], [], [], [], [], [], [], []
        for i in range(0, bt, chunk):
            km_i = keep_flat[i : i + chunk] if keep_flat is not None else None
            res = self.encode_images(
                rgb_f[i : i + chunk], K_f[i : i + chunk], R_f[i : i + chunk],
                return_aux_geom=return_aux_geom, rgb_keep_mask=km_i,
            )
            if return_aux_geom:
                fi, gi, pi, zmi, nri, mki, dlzi, dnri = res
                zm_l.append(zmi); nr_l.append(nri); mk_l.append(mki); dlz_l.append(dlzi); dnr_l.append(dnri)
            else:
                fi, gi, pi = res
            f_l.append(fi); g_l.append(gi); p_l.append(pi)
        f2d_all = torch.cat(f_l, dim=0).reshape(B, T, self.c_feat, FEAT_H, FEAT_W)
        geom_all = torch.cat(g_l, dim=0).reshape(B, T, GEOM_CH, FEAT_H, FEAT_W)
        plane_all = torch.cat(p_l, dim=0).reshape(B, T, 4)

        mem = torch.zeros(B, self.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
        age = torch.full((B, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)
        h_state = torch.tensor([[1.35, 0.0]], device=device, dtype=f2d_all.dtype).expand(B, 2)

        st_s, dy_s, hz_s, ch_s, cp_s, un_s = [], [], [], [], [], []
        for t in range(T):
            lps = [
                (f2d_all[:, t - L], geom_all[:, t - L], K_scaled[:, t - L], R_c2w[:, t - L], t_w[:, t - L])
                if (t >= L and L not in mask_lags)
                else None
                for L in self.lags
            ]
            st, dy, hz, ch, cp, un, mem, age, h_state = self.step_from_encoded(
                f2d_all[:, t], geom_all[:, t], plane_all[:, t],
                K_scaled[:, t], R_c2w[:, t], t_w[:, t], fwd_w[:, t],
                lps, t_w[:, max(0, t - 1)], fwd_w[:, max(0, t - 1)], seg_len[:, t],
                mem, age, h_state, is_first=(t == 0),
            )
            st_s.append(st); dy_s.append(dy); hz_s.append(hz); ch_s.append(ch); cp_s.append(cp); un_s.append(un)

        out = {
            "static_logits": torch.stack(st_s, dim=1),
            "dynamic_logits": torch.stack(dy_s, dim=1),
            "hazard_logits": torch.stack(hz_s, dim=1),
            "cum_hazard": torch.stack(ch_s, dim=1),
            "corr_probs": torch.stack(cp_s, dim=1),
            "unc_logits": torch.stack(un_s, dim=1),
        }
        if return_aux_geom:
            out["pred_z_m"] = torch.cat(zm_l, dim=0).reshape(B, T, 1, FEAT_H, FEAT_W)
            out["pred_nrm"] = torch.cat(nr_l, dim=0).reshape(B, T, 3, FEAT_H, FEAT_W)
            out["pred_msk"] = torch.cat(mk_l, dim=0).reshape(B, T, 1, FEAT_H, FEAT_W)
            out["d_logz"] = torch.cat(dlz_l, dim=0).reshape(B, T, 1, FEAT_H, FEAT_W)
            out["d_nrm"] = torch.cat(dnr_l, dim=0).reshape(B, T, 3, FEAT_H, FEAT_W)
        return out


class EgoClearV8EncoderExport(nn.Module):
    def __init__(self, net: EgoClearV8Net):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor):
        return self.net.encode_images(rgb, K_scaled, R_c2w, return_aux_geom=False)


class EgoClearV8StepExport(nn.Module):
    """Export wrapper for leaner lags=(1, 2) step model (5 fewer inputs, ~20% faster Android CPU step!)."""
    def __init__(self, net: EgoClearV8Net):
        super().__init__()
        self.net = net

    def forward(
        self,
        f2d_t, geom_t, plane_obs_t,
        f2d_1, geom_1, f2d_2, geom_2,
        K_t, R_t, t_t, fwd_t, K_1, R_1, t_1, fwd_1,
        K_2, R_2, t_2,
        ds_t, mem_prev, age_prev, h_state_prev,
    ):
        st, dy, _hz, _ch, cp, un, mem_new, age_new, h_new = self.net.step_from_encoded(
            f2d_t, geom_t, plane_obs_t,
            K_t, R_t, t_t, fwd_t,
            [(f2d_1, geom_1, K_1, R_1, t_1), (f2d_2, geom_2, K_2, R_2, t_2)],
            t_1, fwd_1, ds_t, mem_prev, age_prev, h_state_prev, is_first=False,
        )
        return st, dy, cp, un, mem_new, age_new, h_new
