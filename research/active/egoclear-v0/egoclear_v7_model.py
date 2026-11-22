"""EgoClear-v7: Metric-3D + Gravity-Aligned Surface-Normal Corridor Perception (MoGe-2-ViT-S-Normal, 35M MIT).

Generational upgrades over EgoClear-v6.6 (DAV2-Small relative disparity):
  1. Frozen MoGe-2-ViT-S-Normal (35.1M, MIT, DINOv2 ViT-S/14, ONNX-17 & NPU compatible):
     - Eliminates affine-invariant (d - q10)/(q90 - q10) distortion that caused 10x Indoor/Outdoor/Zero-Shot shift.
     - Closed-form focal-consistent shift recovery from known K_scaled -> true metric depth z_metric (meters).
     - Dense 3D surface normals rotated by hardware gravity R_c2w -> n_up (+1 walkable ground, 0 vertical blocker).
     - Pixel-wise geometric validity confidence mask for low-light / blank-wall / sky robustness.
  2. High-Resolution 84x148 Geometry-Aware Feature Pyramid (4x pixel density vs v6.6's 42x74):
     - Taps MoGe-2's native neck level-2 (128ch @ 84x148) + level-1 (256ch @ 42x74) for sharp poles/curbs/HEAD edges.
  3. Explicit 3D Voxel Metric-Occupancy & Normal Consistency (per 2,520 corridor hypotheses):
     - signed_sdf = clamp((z_pred - z_vox) / z_vox, -1.5, 1.5)
     - surf_hit = exp(-((z_pred - z_vox) / (0.18 * z_vox))^2) * mask
     - vert_hit = surf_hit * (1 - |n_up|)   [vertical obstacle hit probability]
     - floor_hit = surf_hit * relu(n_up)    [walkable floor hit probability]
  4. 3-Lag (1, 2, 4) Multi-Scale Parallax Stereo + Dynamic-Suppressed ConvGRU + Monotonic First-Hit Survival.
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
from egoclear_v3_model import DynamicSuppressedConvGRU
from egoclear_v4_model import H_CORR, survival_cum_probs

FEAT_H, FEAT_W = 84, 148
GEOM_CH = 8  # [z_metric, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, mask]


def _build_moge_uv_levels(base_h: int, base_w: int, aspect_ratio: float) -> list[torch.Tensor]:
    """Precompute static normalized view-plane UVs for MoGe-2 neck levels 0..4 (100% ONNX-17 static buffers)."""
    from moge.utils.geometry_torch import normalized_view_plane_uv

    uvs = []
    for level in range(5):
        uv = normalized_view_plane_uv(
            width=base_w * (2 ** level),
            height=base_h * (2 ** level),
            aspect_ratio=aspect_ratio,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )  # (H, W, 2)
        uvs.append(uv.permute(2, 0, 1).unsqueeze(0).contiguous())  # (1, 2, H, W)
    return uvs


class EgoClearV7Net(nn.Module):
    """EgoClear-v7 with Frozen MoGe-2-ViT-S-Normal + 84x148 Metric/Normal Voxel Consistency + 3-Lag MVS."""

    def __init__(
        self,
        moge_id: str = "Ruicheng/moge-2-vits-normal",
        c_feat: int = 64,
        c_bev: int = 128,
        lags: tuple[int, ...] = (1, 2, 4),
        freeze_moge: bool = True,
    ):
        super().__init__()
        from moge.model.v2 import MoGeModel

        self.moge_id = moge_id
        self.c_feat = c_feat
        self.c_bev = c_bev
        self.lags = tuple(lags)
        self.freeze_moge = freeze_moge

        self.moge = MoGeModel.from_pretrained(moge_id)
        if hasattr(self.moge, "enable_onnx_compatible_mode"):
            self.moge.enable_onnx_compatible_mode()
        elif hasattr(self.moge, "encoder") and hasattr(self.moge.encoder, "backbone"):
            setattr(self.moge.encoder.backbone, "onnx_compatible_mode", True)

        if freeze_moge:
            for p in self.moge.parameters():
                p.requires_grad = False
            self.moge.eval()

        # Precompute static UV buffers for 294x518 -> 21x37 tokens (zero runtime dynamic grid creation)
        self.base_h = RGB_H // 14  # 21
        self.base_w = RGB_W // 14  # 37
        for lv, uv in enumerate(_build_moge_uv_levels(self.base_h, self.base_w, RGB_W / RGB_H)):
            self.register_buffer(f"moge_uv_{lv}", uv, persistent=False)

        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        # Static pixel grid at FEAT_H x FEAT_W (84 x 148) for closed-form shift & ray geometry
        u = (torch.arange(FEAT_W, dtype=torch.float32) + 0.5) * (RGB_W / FEAT_W)
        v = (torch.arange(FEAT_H, dtype=torch.float32) + 0.5) * (RGB_H / FEAT_H)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("feat_uu", uu.unsqueeze(0), persistent=False)  # (1, 84, 148)
        self.register_buffer("feat_vv", vv.unsqueeze(0), persistent=False)  # (1, 84, 148)

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", kx, persistent=False)
        self.register_buffer("sobel_y", ky, persistent=False)

        # Multi-scale 2D feature fusion at 84x148:
        # neck[1] (256ch @ 42x74 -> 96ch @ 84x148) + neck[2] (128ch @ 84x148) + geom_cues (8ch) + rays (3ch)
        self.neck1_proj = ConvBNReLU(256, 96, 1, 0)
        self.fuse_2d = nn.Sequential(
            ConvBNReLU(96 + 128 + GEOM_CH + 3, 128, 3, 1),
            nn.Dropout2d(0.08),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        # Canonical 3D corridor voxel grid (NY=7 x NZ=24 x NX=15 = 2520 hypotheses)
        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        self.register_buffer("grid_xzy", torch.stack([gx, gz, gy], dim=-1).reshape(-1, 3), persistent=False)
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)

        # Per-voxel features:
        #   f2d (64) + geom (8) + vis (1) + uv_invz (3) + physical_voxel_hits (4: sdf, surf_hit, vert_hit, floor_hit)
        #   + 5 * len(lags) multi-scale parallax stereo (cos_sim, l1_diff, invz_diff, both_vis, baseline)
        c_pt = c_feat + GEOM_CH + 1 + 3 + 4 + 5 * len(self.lags)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.bev_enc = nn.Sequential(
            ConvBNReLU(48 * NY + 2, c_bev, 3, 1),
            ResBlock2D(c_bev),
        )
        self.head_dynamic_obs = nn.Sequential(
            ConvBNReLU(c_bev, 64, 3, 1),
            nn.Conv2d(64, 3, 1),
        )
        self.mem_gru = DynamicSuppressedConvGRU(c_bev, c_bev)
        self.bev_refine = nn.Sequential(
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
        )
        self.head_static = nn.Conv2d(c_bev, 3, 1)

        # Corridor hazard & uncertainty head: reads central 5 cols + left/right context (7 * c_bev)
        self.corr_head = nn.Sequential(
            nn.Conv1d(c_bev * 7, 160, 3, padding=1, bias=False),
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_moge:
            self.moge.eval()
        return self

    def encode_images(
        self,
        rgb: torch.Tensor,      # (N, 3, 294, 518) ImageNet-normalized
        K_scaled: torch.Tensor, # (N, 4) [fx, fy, cx, cy]
        R_c2w: torch.Tensor,    # (N, 3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run frozen MoGe-2-ViT-S-Normal + closed-form metric shift + gravity normal fusion -> (f2d, geom_cues) @ 84x148."""
        N = rgb.shape[0]
        rgb_01 = (rgb.float() * self.im_std + self.im_mean).clamp(0.0, 1.0)

        ctx = torch.no_grad() if self.freeze_moge else torch.enable_grad()
        with ctx:
            feat_bb, cls_token = self.moge.encoder(rgb_01, self.base_h, self.base_w, return_class_token=True)
            feats_in = [feat_bb, None, None, None, None]
            for lv in range(5):
                uv = getattr(self, f"moge_uv_{lv}").to(dtype=feat_bb.dtype).expand(N, -1, -1, -1)
                feats_in[lv] = uv if feats_in[lv] is None else torch.cat([feats_in[lv], uv], dim=1)
            neck_feats = self.moge.neck(feats_in)  # [21x37(384), 42x74(256), 84x148(128), 168x296(64), 336x592(32)]

            pts_raw = self.moge.points_head(neck_feats)[-1]   # (N, 3, H_out, W_out)
            nrm_raw = self.moge.normal_head(neck_feats)[-1]   # (N, 3, H_out, W_out)
            msk_raw = self.moge.mask_head(neck_feats)[-1]     # (N, 1, H_out, W_out)
            m_scale = self.moge.scale_head(cls_token).squeeze(1).exp().float()  # (N,)

            # Downsample predictions directly to high-res 84x148 (4x denser than v6.6's 42x74!)
            pts_84 = F.interpolate(pts_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            nrm_84 = F.interpolate(nrm_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            msk_84 = torch.sigmoid(
                F.interpolate(msk_raw.float(), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
            )  # (N, 1, 84, 148)

            # Exact MoGe-2 point remapping: (x, y) = xy * exp(z_raw), z = exp(z_raw)
            pts_hwc = pts_84.permute(0, 2, 3, 1)
            pts_hwc = self.moge._remap_points(pts_hwc)  # (N, 84, 148, 3)
            px = pts_hwc[..., 0]
            py = pts_hwc[..., 1]
            pz = pts_hwc[..., 2]

            # Unit camera-frame surface normal (N, 3, 84, 148)
            nrm_unit = F.normalize(nrm_84, dim=1, eps=1e-6)

            nf1 = neck_feats[1].detach()  # (N, 256, 42, 74)
            nf2 = neck_feats[2].detach()  # (N, 128, 84, 148)

        # Camera unit rays at 84x148 using exact known intrinsics K_scaled
        fx = K_scaled[:, 0].float().view(N, 1, 1).clamp(min=10.0)
        fy = K_scaled[:, 1].float().view(N, 1, 1).clamp(min=10.0)
        cx = K_scaled[:, 2].float().view(N, 1, 1)
        cy = K_scaled[:, 3].float().view(N, 1, 1)
        rx = (self.feat_uu - cx) / fx  # (N, 84, 148)
        ry = (self.feat_vv - cy) / fy  # (N, 84, 148)

        # Closed-form FP16-bounded + 4x4 spatial-tile-capped shift recovery (Codex Problem 5 fix):
        # Prevents FP16 sum overflow (max sum <= 16*20 = 320 << 65,504) and outer-border ||r||^4 dominance!
        r_max = 2.0
        s_max = 25.0
        tile_cap = 20.0
        ax = (rx / r_max).clamp(-1.0, 1.0)
        ay = (ry / r_max).clamp(-1.0, 1.0)
        a2 = (ax * ax + ay * ay).clamp(max=1.0)
        pz_c = pz.clamp(min=0.25, max=20.0)
        bx = (px / (s_max * r_max) - (pz_c / s_max) * ax).clamp(-1.0, 1.0)
        by = (py / (s_max * r_max) - (pz_c / s_max) * ay).clamp(-1.0, 1.0)
        msk_hw = msk_84.squeeze(1)
        q_raw = msk_hw * a2 * ((0.25 / pz_c) ** 2)  # strictly in [0, 1]

        # Exact integer 4x4 tile partitioning of 84x148 -> (4 x 21) x (4 x 37)
        q_tiles = q_raw.reshape(N, 4, 21, 4, 37)
        tile_sum = q_tiles.sum(dim=(2, 4), keepdim=True)
        q_capped = (q_tiles * (tile_cap / tile_sum.clamp(min=tile_cap))).reshape(N, FEAT_H, FEAT_W)

        num = (q_capped * (ax * bx + ay * by)).sum(dim=(1, 2))
        den = (q_capped * a2).sum(dim=(1, 2))
        # Smooth low-support fallback to shift=0 when mask/rays vanish (e.g. dark blank wall)
        shift_raw = s_max * (num / den.clamp(min=0.25))
        support_gate = den / (den + 0.25)
        shift = (shift_raw * support_gate).clamp(-2.0, 4.0).view(N, 1, 1)

        # True metric depth in meters!
        z_metric = ((pz + shift).clamp(min=0.15, max=25.0) * m_scale.view(N, 1, 1)).clamp(min=0.15, max=20.0)
        z_m = z_metric.unsqueeze(1)  # (N, 1, 84, 148)
        inv_z = 1.0 / z_m.clamp(min=0.25)
        log_z = torch.log1p(z_m)
        dz_gx = F.conv2d(inv_z, self.sobel_x, padding=1)
        dz_gy = F.conv2d(inv_z, self.sobel_y, padding=1)

        # Gravity-aligned surface normal: world UP component n_up = R_c2w[2, :] . n_cam
        r_up_c = R_c2w[:, 2, :].float().view(N, 3, 1, 1)
        n_up = (nrm_unit * r_up_c).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        verticality = (1.0 - n_up.abs()).clamp(0.0, 1.0)

        geom_cues = torch.cat([z_m, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, msk_84], dim=1)  # (N, 8, 84, 148)

        # Per-pixel gravity ray alignment
        rays = torch.stack([rx, ry, torch.ones_like(rx)], dim=-1)
        rays_u = F.normalize(rays, dim=-1)
        grav_c = -R_c2w[:, 2, :].float()
        grav_dot = (rays_u * grav_c[:, None, None, :]).sum(dim=-1)
        ray_geo = torch.stack([rx, ry, grav_dot], dim=1)  # (N, 3, 84, 148)

        nf1_up = F.interpolate(self.neck1_proj(nf1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
        f2d = self.fuse_2d(
            torch.cat([nf1_up, nf2.to(nf1_up.dtype), geom_cues.to(nf1_up.dtype), ray_geo.to(nf1_up.dtype)], dim=1)
        )
        return f2d, geom_cues.to(f2d.dtype)

    def _project_and_sample(
        self,
        f2d: torch.Tensor,       # (B, c_feat, 84, 148)
        geom: torch.Tensor,      # (B, 8, 84, 148)
        pts_w: torch.Tensor,     # (B, M, 3)
        R_c2w: torch.Tensor,     # (B, 3, 3)
        t_w: torch.Tensor,       # (B, 3)
        K_scaled: torch.Tensor,  # (B, 4)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        samp_f = samp_fg[:, : self.c_feat, :]
        samp_g = samp_fg[:, self.c_feat :, :]
        uv_invz = torch.stack([u_norm.clamp(-1.5, 1.5), v_norm.clamp(-1.5, 1.5), 1.0 / zc_safe], dim=1).to(f2d.dtype)
        return samp_f, samp_g, vis_f, uv_invz, zc_safe.unsqueeze(1).to(f2d.dtype)

    def _warp_bev(
        self,
        mem_prev: torch.Tensor,
        t_prev: torch.Tensor,
        fwd_prev: torch.Tensor,
        t_cur: torch.Tensor,
        fwd_cur: torch.Tensor,
    ) -> torch.Tensor:
        right_cur = torch.stack([fwd_cur[:, 1], -fwd_cur[:, 0]], dim=-1)
        right_prev = torch.stack([fwd_prev[:, 1], -fwd_prev[:, 0]], dim=-1)
        bx = self.bev_xz[..., 0][None]
        bz = self.bev_xz[..., 1][None]
        qw = (
            t_cur[:, None, None, :2]
            + bx[..., None] * right_cur[:, None, None, :]
            + bz[..., None] * fwd_cur[:, None, None, :2]
        )
        rel_prev = qw - t_prev[:, None, None, :2]
        x_prev = (rel_prev * right_prev[:, None, None, :]).sum(dim=-1)
        z_prev = (rel_prev * fwd_prev[:, None, None, :2]).sum(dim=-1)
        x_lo, x_hi = BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX
        z_lo, z_hi = BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ
        gx = 2.0 * (x_prev - x_lo) / (x_hi - x_lo) - 1.0
        gz = 2.0 * (z_prev - z_lo) / (z_hi - z_lo) - 1.0
        grid = torch.stack([gx, gz], dim=-1).to(mem_prev.dtype)
        return F.grid_sample(mem_prev, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def step_from_encoded(
        self,
        f2d_t: torch.Tensor,
        geom_t: torch.Tensor,
        K_t: torch.Tensor,
        R_t: torch.Tensor,
        t_t: torch.Tensor,
        fwd_t: torch.Tensor,
        lag_packs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None],
        t_prev: torch.Tensor,
        fwd_prev: torch.Tensor,
        ds_t: torch.Tensor,
        mem_prev: torch.Tensor,
        age_prev: torch.Tensor,
        is_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single streaming step (shared by sequence training AND ONNX step export)."""
        B = f2d_t.shape[0]
        device = f2d_t.device
        dtype = f2d_t.dtype

        bev_pos = torch.stack(
            [
                self.bev_xz[..., 0] / BEV_X_MAX,
                (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
            ],
            dim=0,
        ).unsqueeze(0).expand(B, -1, -1, -1).to(dtype)

        gx = self.grid_xzy[:, 0][None, :, None]
        gz = self.grid_xzy[:, 1][None, :, None]
        gy = self.grid_xzy[:, 2][None, :, None]
        up_w = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)[None, None, :]

        right_t = torch.stack([fwd_t[:, 1], -fwd_t[:, 0], torch.zeros_like(fwd_t[:, 0])], dim=-1)
        pts_w = t_t[:, None, :] + gx * right_t[:, None, :] + gz * fwd_t[:, None, :] + gy * up_w

        samp_f, samp_g, vis_t, uv_invz_t, z_vox = self._project_and_sample(f2d_t, geom_t, pts_w, R_t, t_t, K_t)

        # Explicit 3D physical voxel consistency from MoGe-2 metric depth + gravity normal + mask!
        z_pred = samp_g[:, 0:1, :]       # metric depth at pixel (u, v)
        n_up_v = samp_g[:, 5:6, :]       # gravity-aligned normal (+1 = flat floor, 0 = vertical obstacle)
        vert_v = samp_g[:, 6:7, :]       # 1 - |n_up|
        mask_v = samp_g[:, 7:8, :]       # validity mask
        rel_diff = (z_pred - z_vox) / z_vox.clamp(min=0.25)
        signed_sdf = rel_diff.clamp(-1.5, 1.5) * vis_t
        surf_hit = torch.exp(-((rel_diff / 0.18).clamp(-6.0, 6.0) ** 2)) * mask_v * vis_t
        vert_hit = surf_hit * vert_v
        floor_hit = surf_hit * F.relu(n_up_v)

        pt_parts = [samp_f, samp_g, vis_t, uv_invz_t, signed_sdf, surf_hit, vert_hit, floor_hit]

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

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
        pt_red = self.pt_reduce(pt_feat).reshape(B, 48 * NY, BEV_NZ, BEV_NX)
        bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))
        dyn_log = self.head_dynamic_obs(bev_obs)
        dyn_prob = torch.sigmoid(dyn_log).max(dim=1, keepdim=True).values.detach()

        vis_bev = vis_t.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True)
        if not is_first:
            mem_w = self._warp_bev(mem_prev, t_prev, fwd_prev, t_t, fwd_t)
            age_w = self._warp_bev(age_prev, t_prev, fwd_prev, t_t, fwd_t)
            ds = ds_t.clamp(0.0, 2.0)[:, None, None, None].to(dtype)
            age_w = (age_w + ds).clamp(max=10.0)
            age_decay = torch.exp(-age_w / 4.0)
            mem_new = self.mem_gru(mem_w, bev_obs, vis_bev, age_decay, dyn_prob)
            age_new = torch.where((vis_bev > 0.05) & (dyn_prob < 0.5), torch.zeros_like(age_w), age_w)
        else:
            mem_new = bev_obs * (1.0 - 0.5 * dyn_prob)
            age_new = torch.where(vis_bev > 0.05, torch.zeros_like(age_prev), age_prev)

        bev_ref = self.bev_refine(mem_new)
        st_log = self.head_static(bev_ref)

        c_slice = bev_ref[:, :, :, 5:10].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        c_left = bev_ref[:, :, :, :5].mean(dim=3)
        c_right = bev_ref[:, :, :, 10:].mean(dim=3)
        corr_out = self.corr_head(torch.cat([c_slice, c_left, c_right], dim=1))
        haz_log = corr_out[:, 0, :]
        unc_log = corr_out[:, 1, :]
        cum_haz = torch.cumsum(F.softplus(haz_log.float()), dim=-1)  # FP32 cumulative hazard Lambda_t >= 0
        corr_p = 1.0 - torch.exp((-cum_haz).clamp(min=-20.0))
        return st_log, dyn_log, haz_log, cum_haz, corr_p, unc_log, mem_new, age_new

    def forward_sequence(
        self,
        rgb: torch.Tensor,
        K_scaled: torch.Tensor,
        R_c2w: torch.Tensor,
        t_w: torch.Tensor,
        fwd_w: torch.Tensor,
        seg_len: torch.Tensor,
        mask_lags: tuple[int, ...] = (),
    ) -> dict[str, torch.Tensor]:
        B, T = rgb.shape[:2]
        device = rgb.device
        bt = B * T
        rgb_flat = rgb.reshape(bt, 3, RGB_H, RGB_W)
        K_flat = K_scaled.reshape(bt, 4)
        R_flat = R_c2w.reshape(bt, 3, 3)

        # Chunk encoder forward pass for long eval videos (84x148 tensors are 4x larger than 42x74)
        chunk = 96
        if bt <= chunk:
            f2d_flat, geom_flat = self.encode_images(rgb_flat, K_flat, R_flat)
        else:
            f_list, g_list = [], []
            for i in range(0, bt, chunk):
                fi, gi = self.encode_images(rgb_flat[i : i + chunk], K_flat[i : i + chunk], R_flat[i : i + chunk])
                f_list.append(fi)
                g_list.append(gi)
            f2d_flat = torch.cat(f_list, dim=0)
            geom_flat = torch.cat(g_list, dim=0)

        f2d_all = f2d_flat.reshape(B, T, self.c_feat, FEAT_H, FEAT_W)
        geom_all = geom_flat.reshape(B, T, GEOM_CH, FEAT_H, FEAT_W)

        mem = torch.zeros(B, self.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
        age = torch.full((B, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)

        st_seq, dy_seq, hz_seq, ch_seq, cp_seq, un_seq = [], [], [], [], [], []
        for t in range(T):
            lag_packs = []
            for lag in self.lags:
                if t >= lag and lag not in mask_lags:
                    tp = t - lag
                    lag_packs.append((f2d_all[:, tp], geom_all[:, tp], K_scaled[:, tp], R_c2w[:, tp], t_w[:, tp]))
                else:
                    lag_packs.append(None)
            st, dy, hz, ch, cp, un, mem, age = self.step_from_encoded(
                f2d_t=f2d_all[:, t],
                geom_t=geom_all[:, t],
                K_t=K_scaled[:, t],
                R_t=R_c2w[:, t],
                t_t=t_w[:, t],
                fwd_t=fwd_w[:, t],
                lag_packs=lag_packs,
                t_prev=t_w[:, max(0, t - 1)],
                fwd_prev=fwd_w[:, max(0, t - 1)],
                ds_t=seg_len[:, t],
                mem_prev=mem,
                age_prev=age,
                is_first=(t == 0),
            )
            st_seq.append(st)
            dy_seq.append(dy)
            hz_seq.append(hz)
            ch_seq.append(ch)
            cp_seq.append(cp)
            un_seq.append(un)

        return {
            "static_logits": torch.stack(st_seq, dim=1),
            "dynamic_logits": torch.stack(dy_seq, dim=1),
            "hazard_logits": torch.stack(hz_seq, dim=1),
            "cum_hazard": torch.stack(ch_seq, dim=1),
            "corr_probs": torch.stack(cp_seq, dim=1),
            "unc_logits": torch.stack(un_seq, dim=1),
        }


class EgoClearV7EncoderExport(nn.Module):
    """ONNX-17 static encoder wrapper: (rgb, K_scaled, R_c2w) -> (f2d [1,64,84,148], geom_cues [1,8,84,148])."""

    def __init__(self, net: EgoClearV7Net):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor):
        return self.net.encode_images(rgb, K_scaled, R_c2w)


class EgoClearV7StepExport(nn.Module):
    """ONNX-17 static 3-lag streaming step wrapper."""

    def __init__(self, net: EgoClearV7Net):
        super().__init__()
        self.net = net

    def forward(
        self,
        f2d_t: torch.Tensor,
        geom_t: torch.Tensor,
        f2d_1: torch.Tensor,
        geom_1: torch.Tensor,
        f2d_2: torch.Tensor,
        geom_2: torch.Tensor,
        f2d_4: torch.Tensor,
        geom_4: torch.Tensor,
        K_t: torch.Tensor,
        R_t: torch.Tensor,
        t_t: torch.Tensor,
        fwd_t: torch.Tensor,
        K_1: torch.Tensor,
        R_1: torch.Tensor,
        t_1: torch.Tensor,
        fwd_1: torch.Tensor,
        K_2: torch.Tensor,
        R_2: torch.Tensor,
        t_2: torch.Tensor,
        K_4: torch.Tensor,
        R_4: torch.Tensor,
        t_4: torch.Tensor,
        ds_t: torch.Tensor,
        mem_prev: torch.Tensor,
        age_prev: torch.Tensor,
    ):
        st, dy, _hz, _ch, cp, un, mem_new, age_new = self.net.step_from_encoded(
            f2d_t=f2d_t,
            geom_t=geom_t,
            K_t=K_t,
            R_t=R_t,
            t_t=t_t,
            fwd_t=fwd_t,
            lag_packs=[
                (f2d_1, geom_1, K_1, R_1, t_1),
                (f2d_2, geom_2, K_2, R_2, t_2),
                (f2d_4, geom_4, K_4, R_4, t_4),
            ],
            t_prev=t_1,
            fwd_prev=fwd_1,
            ds_t=ds_t,
            mem_prev=mem_prev,
            age_prev=age_prev,
            is_first=False,
        )
        return st, dy, cp, un, mem_new, age_new
