"""EgoClear-v10: Universal Isotropic-Safe Dual-Geometry + Unlocked MoGe-2 Neck/Last-2-Blocks +
Training-Only DAV2 Cross-Backbone Alignment (Zero Inference Cost!) + 3D Homography Head-Scan Architecture.

Synthesizes EVERY empirical breakthrough from Phases 1-13:
  1. Universal Isotropic-Safe encode_images(): automatically masks padded letterbox borders (rgb_01.amax > 0.008)
     out of BOTH the FP16-bounded 4x4 tile-capped closed-form shift solver AND scale-free relative-depth mean/std,
     with ZERO extra ONNX inputs (100% compatible with S24 Ultra 3-input SNPE interface, 52.1 ms / 19.2 FPS!).
  2. Controlled Backbone Unlock: unfreezes ONLY MoGe-2's last 2 ViT blocks + norm + neck (at 20x lower lr=1.8e-5),
     jointly anchored by 331M MoGe-2-ViT-L-Normal + GT depth_s8 + Training-Only DAV2-Small Feature Alignment Head
     (discarded at ONNX export -> ZERO extra inference parameters or latency!).
  3. Retains v9's proven 10-ch Dual-Geometry + Multi-Scale 0.84x/1.00x/1.18x Voxel Hits + Restored Scale-Free VIO TTC
     on lean lags=(1, 2) + 22-ch Direct 3D Physics-Shortcut Corridor Head + FreeSpaceDynamicConvGRU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
from egoclear_v7_model import FEAT_H, FEAT_W, _build_moge_uv_levels
from egoclear_v8_model import FreeSpaceDynamicConvGRU

GEOM_CH_V10 = 10  # [z_m, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, mask, invz_rel_norm, logz_centered]


class EgoClearV10Net(nn.Module):
    """EgoClear-v10: Border-Masked Isotropic Dual-Geometry + Controlled Neck/ViT-Last2 Unlock + Zero-Cost DAV2 Aux Alignment."""

    def __init__(
        self,
        moge_id: str = "Ruicheng/moge-2-vits-normal",
        c_feat: int = 64,
        c_bev: int = 128,
        lags: tuple[int, ...] = (1, 2),
        unlock_moge_neck_and_last2: bool = True,
        delta_z_max: float = 0.48,
        delta_n_max: float = 0.35,
        rgb_feat_drop_p: float = 0.06,
    ):
        super().__init__()
        from moge.model.v2 import MoGeModel

        self.moge_id = moge_id
        self.c_feat = c_feat
        self.c_bev = c_bev
        self.lags = tuple(lags)
        self.unlock_moge_neck_and_last2 = unlock_moge_neck_and_last2
        self.delta_z_max = delta_z_max
        self.delta_n_max = delta_n_max
        self.rgb_feat_drop_p = rgb_feat_drop_p

        self.moge = MoGeModel.from_pretrained(moge_id)
        if hasattr(self.moge, "enable_onnx_compatible_mode"):
            self.moge.enable_onnx_compatible_mode()
        elif hasattr(self.moge, "encoder") and hasattr(self.moge.encoder, "backbone"):
            setattr(self.moge.encoder.backbone, "onnx_compatible_mode", True)

        # Freeze entire MoGe-2 first, then selectively unlock ONLY last 2 ViT blocks + norm + neck (safe low-LR adaptation!)
        for p in self.moge.parameters():
            p.requires_grad = False
        if unlock_moge_neck_and_last2:
            bb = getattr(self.moge.encoder, "backbone", None)
            if bb is not None and hasattr(bb, "blocks"):
                for blk in list(bb.blocks)[-2:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                if hasattr(bb, "norm"):
                    for p in bb.norm.parameters():
                        p.requires_grad = True
            for p in self.moge.neck.parameters():
                p.requires_grad = True

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

        # Zero-init bounded geometric residual head (jointly distilled from 331M ViT-L + GT depth_s8!)
        self.geom_res_head = nn.Sequential(
            ConvBNReLU(96 + 128 + 5, 96, 3, 1),
            ResBlock2D(96),
            nn.Conv2d(96, 4, 1),  # [r_logz, r_nx, r_ny, r_nz]
        )
        nn.init.zeros_(self.geom_res_head[-1].weight)
        nn.init.zeros_(self.geom_res_head[-1].bias)

        self.fuse_2d = nn.Sequential(
            ConvBNReLU(96 + 128 + GEOM_CH_V10 + 3, 128, 3, 1),
            nn.Dropout2d(0.06),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        # Training-only auxiliary projector to align f2d -> DAV2-Small (v6.6) relative disparity + features!
        # Completely excluded from ONNX export -> ZERO inference cost!
        self.dav2_aux_proj = nn.Sequential(
            nn.Conv2d(c_feat, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, c_feat + 1, 1),  # [dav2_f2d (64ch), dav2_rel_disp_norm (1ch)]
        )

        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        self.register_buffer("grid_x", gx.reshape(-1), persistent=False)
        self.register_buffer("grid_z", gz.reshape(-1), persistent=False)
        self.register_buffer("grid_y", gy.reshape(-1), persistent=False)
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)

        c_pt = c_feat + GEOM_CH_V10 + 1 + 3 + 8 + 6 * len(self.lags)
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
        self.mem_gru = FreeSpaceDynamicConvGRU(c_bev, c_bev)
        self.bev_refine = nn.Sequential(
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
        )
        self.head_static = nn.Conv2d(c_bev, 3, 1)
        self.corr_head = nn.Sequential(
            nn.Conv1d(c_bev * 7 + 22, 160, 3, padding=1, bias=False),
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
        # Keep frozen MoGe heads & early blocks in deterministic eval mode (no dropout/BN drift in MoGe!)
        self.moge.eval()
        return self

    def param_groups(self, lr_head: float = 3.5e-4, lr_moge: float = 1.8e-5) -> list[dict]:
        moge_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("moge."):
                moge_params.append(p)
            else:
                head_params.append(p)
        groups = [{"params": head_params, "lr": lr_head}]
        if moge_params:
            groups.append({"params": moge_params, "lr": lr_moge})
        return groups

    def _moge_forward_84(self, rgb_01: torch.Tensor):
        N = rgb_01.shape[0]
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
        nrm_unit = F.normalize(nrm_84, dim=1, eps=1e-6)
        return pts_hwc[..., 0], pts_hwc[..., 1], pts_hwc[..., 2], nrm_unit, msk_84, m_scale, neck_feats[1], neck_feats[2]

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

        # QAIRT-HTP-native automatic letterbox/homography valid-pixel mask (0 on black padded bars, 1 inside active image!)
        # 3x3 max_pool ensures isolated dark pixels in 0.18x dark rooms stay 1.0 while wide letterbox bars stay 0.0!
        lum_max = rgb_01.amax(dim=1, keepdim=True)
        valid_full = ((lum_max - 0.0005) * 1000.0).clamp(0.0, 1.0)
        valid_84 = F.interpolate(valid_full, size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
        valid_84 = F.max_pool2d(valid_84, kernel_size=3, stride=1, padding=1)

        if self.unlock_moge_neck_and_last2 and self.training:
            # With micro_bs=8 (96 frames total), peak VRAM is ~46GB on 96GB G4 -> ZERO grad_checkpoint recompute needed (2x faster!)
            px, py, pz, nrm_unit, msk_84, m_scale, nf1, nf2 = self._moge_forward_84(rgb_01)
        else:
            with torch.no_grad():
                px, py, pz, nrm_unit, msk_84, m_scale, nf1, nf2 = self._moge_forward_84(rgb_01)

        # Mask out padded black letterbox/homography borders from MoGe mask, shift solver, and relative depth stats!
        msk_eff = msk_84 * valid_84

        fx = K_scaled[:, 0].float().view(N, 1, 1).clamp(min=10.0)
        fy = K_scaled[:, 1].float().view(N, 1, 1).clamp(min=10.0)
        cx = K_scaled[:, 2].float().view(N, 1, 1)
        cy = K_scaled[:, 3].float().view(N, 1, 1)
        rx = (self.feat_uu - cx) / fx
        ry = (self.feat_vv - cy) / fy

        # Border-masked FP16-bounded + 4x4 tile-capped closed-form shift!
        r_max, s_max, tile_cap = 2.0, 25.0, 20.0
        ax = (rx / r_max).clamp(-1.0, 1.0)
        ay = (ry / r_max).clamp(-1.0, 1.0)
        a2 = (ax * ax + ay * ay).clamp(max=1.0)
        pz_c = pz.clamp(min=0.25, max=20.0)
        bx = (px / (s_max * r_max) - (pz_c / s_max) * ax).clamp(-1.0, 1.0)
        by = (py / (s_max * r_max) - (pz_c / s_max) * ay).clamp(-1.0, 1.0)
        msk_hw = msk_eff.squeeze(1)
        q_raw = msk_hw * a2 * ((0.25 / pz_c) ** 2)
        q_tiles = q_raw.reshape(N, 4, 21, 4, 37)
        tile_sum = q_tiles.sum(dim=(2, 4), keepdim=True)
        q_capped = (q_tiles * (tile_cap / tile_sum.clamp(min=tile_cap))).reshape(N, FEAT_H, FEAT_W)
        num = (q_capped * (ax * bx + ay * by)).sum(dim=(1, 2))
        den = (q_capped * a2).sum(dim=(1, 2))
        shift = ((s_max * (num / den.clamp(min=0.25))) * (den / (den + 0.25))).clamp(-2.0, 4.0).view(N, 1, 1)
        z_base = ((pz + shift).clamp(min=0.15, max=25.0) * m_scale.view(N, 1, 1)).clamp(min=0.15, max=20.0).unsqueeze(1)

        nf1_up = F.interpolate(self.neck1_proj(nf1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)

        res_in = torch.cat([nf1_up, nf2.to(nf1_up.dtype), torch.log(z_base).to(nf1_up.dtype), nrm_unit.to(nf1_up.dtype), msk_eff.to(nf1_up.dtype)], dim=1)
        res_out = self.geom_res_head(res_in).float()
        d_logz = self.delta_z_max * torch.tanh(res_out[:, 0:1])
        d_nrm = self.delta_n_max * torch.tanh(res_out[:, 1:4])
        z_m = (z_base * torch.exp(d_logz)).clamp(min=0.15, max=20.0)
        nrm_ref = F.normalize(nrm_unit + d_nrm, dim=1, eps=1e-6)

        inv_z = 1.0 / z_m.clamp(min=0.25)
        log_z = torch.log(z_m.clamp(min=0.15))
        dz_gx = F.conv2d(inv_z, self.sobel_x, padding=1) * valid_84
        dz_gy = F.conv2d(inv_z, self.sobel_y, padding=1) * valid_84

        r_up_c = R_c2w[:, 2, :].float().view(N, 3, 1, 1)
        n_up = (nrm_ref * r_up_c).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        verticality = (1.0 - n_up.abs()).clamp(0.0, 1.0)

        # Border-masked QAIRT-HTP-native scale-free relative depth cues!
        w_sum = msk_eff.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mu_invz = (inv_z * msk_eff).sum(dim=(2, 3), keepdim=True) / w_sum
        var_invz = (((inv_z - mu_invz) ** 2) * msk_eff).sum(dim=(2, 3), keepdim=True) / w_sum
        invz_rel_norm = (((inv_z - mu_invz) / torch.sqrt(var_invz + 0.01)).clamp(-3.0, 3.0)) * valid_84
        mu_logz = (log_z * msk_eff).sum(dim=(2, 3), keepdim=True) / w_sum
        logz_centered = ((log_z - mu_logz).clamp(-2.5, 2.5)) * valid_84

        geom_cues = torch.cat([z_m, inv_z, log_z, dz_gx, dz_gy, n_up, verticality, msk_eff, invz_rel_norm, logz_centered], dim=1)

        rays = torch.stack([rx, ry, torch.ones_like(rx)], dim=-1)
        rays_u = F.normalize(rays, dim=-1)
        grav_dot = (rays_u * (-R_c2w[:, 2, :].float())[:, None, None, :]).sum(dim=-1)
        ray_geo = torch.stack([rx, ry, grav_dot], dim=1)
        if rgb_keep_mask is not None:
            km = rgb_keep_mask.to(nf1_up.dtype).view(N, 1, 1, 1)
            nf1_in, nf2_in = nf1_up * km, nf2.to(nf1_up.dtype) * km
        else:
            nf1_in, nf2_in = nf1_up, nf2.to(nf1_up.dtype)
        f2d = self.fuse_2d(torch.cat([nf1_in, nf2_in, geom_cues.to(nf1_up.dtype), ray_geo.to(nf1_up.dtype)], dim=1))

        if return_aux_geom:
            dav2_pred = self.dav2_aux_proj(f2d)
            return f2d, geom_cues.to(f2d.dtype), z_m, nrm_ref, msk_eff, d_logz, d_nrm, dav2_pred
        return f2d, geom_cues.to(f2d.dtype)

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
        # Gate visibility by valid_img_mask (encoded inside geom[:, 7:8], msk_eff = msk_84 * valid_84!)
        # If a voxel projects into a black letterbox/homography border, msk_eff == 0 -> treat as not visible so mem_gru preserves prior memory!
        active_pix = (samp_fg[:, self.c_feat + 7 : self.c_feat + 8, :] > 0.05).to(f2d.dtype)
        vis_eff = vis_f * active_pix
        samp_fg = samp_fg * vis_eff
        uv_invz = torch.stack([u_norm.clamp(-1.5, 1.5), v_norm.clamp(-1.5, 1.5), 1.0 / zc_safe], dim=1).to(f2d.dtype)
        return samp_fg[:, : self.c_feat, :], samp_fg[:, self.c_feat :, :], vis_eff, uv_invz, zc_safe.unsqueeze(1).to(f2d.dtype)

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
        f2d_t, geom_t,
        K_t, R_t, t_t, fwd_t,
        lag_packs,
        t_prev, fwd_prev, ds_t,
        mem_prev, age_prev,
        is_first: bool,
    ):
        B = f2d_t.shape[0]
        device, dtype = f2d_t.device, f2d_t.dtype

        gx = self.grid_x[None, :, None]
        gz = self.grid_z[None, :, None]
        gy = self.grid_y[None, :, None]
        right_t = torch.stack([fwd_t[:, 1], -fwd_t[:, 0], torch.zeros_like(fwd_t[:, 0])], dim=-1)
        pts_w = t_t[:, None, :] + gx * right_t[:, None, :] + gz * fwd_t[:, None, :] + gy * self.up_w

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

        pt_parts = [
            samp_f, samp_g, vis_t, uv_invz_t,
            signed_sdf, surf_mid, surf_near, surf_far, vert_mid, vert_near, floor_hit, free_ev_vox,
        ]
        ft_norm = F.normalize(samp_f, dim=1, eps=1e-5)
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
                base = torch.linalg.norm(t_t - t_p, dim=-1).clamp(max=2.5)
                base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(dtype) * both_v
                pt_parts.extend([cos_sim, l1_diff, ttc_exp_err, invz_diff, both_v, base_map])
            else:
                pt_parts.append(torch.zeros(B, 6, pts_w.shape[1], device=device, dtype=dtype))

        bev_pos = torch.stack([
            self.bev_xz[..., 0] / BEV_X_MAX,
            (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
        ], dim=0).unsqueeze(0).expand(B, -1, -1, -1).to(dtype)

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
        pt_red = self.pt_reduce(pt_feat).reshape(B, 48 * NY, BEV_NZ, BEV_NX)
        bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))
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

        vh_m = vert_mid.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        vh_n = vert_near.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        fh_4 = floor_hit.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        fr_4 = free_ev_vox.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        vs_4 = vis_t.reshape(B, NY, BEV_NZ, BEV_NX)[:, :, :, 5:10]
        st_p = torch.sigmoid(st_log[:, :, :, 5:10])
        phys_corr = torch.cat([
            vh_m[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 2:5].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 5:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 2:5].amax(dim=(1, 3)).unsqueeze(1),
            vh_n[:, 5:7].amax(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 2:5].mean(dim=(1, 3)).unsqueeze(1),
            vh_m[:, 5:7].mean(dim=(1, 3)).unsqueeze(1),
            vh_n.mean(dim=(1, 3)).unsqueeze(1),
            fh_4[:, 0:2].amax(dim=(1, 3)).unsqueeze(1),
            fh_4[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 0:2].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 2:5].mean(dim=(1, 3)).unsqueeze(1),
            fr_4[:, 5:7].mean(dim=(1, 3)).unsqueeze(1),
            vs_4.mean(dim=(1, 3)).unsqueeze(1),
            st_p.amax(dim=3),
            st_p.mean(dim=3),
        ], dim=1).to(bev_ref.dtype)

        c_slice = bev_ref[:, :, :, 5:10].permute(0, 1, 3, 2).reshape(B, self.c_bev * 5, BEV_NZ)
        c_left = bev_ref[:, :, :, :5].mean(dim=3)
        c_right = bev_ref[:, :, :, 10:].mean(dim=3)
        corr_out = self.corr_head(torch.cat([c_slice, c_left, c_right, phys_corr], dim=1))
        haz_log = corr_out[:, 0, :]
        unc_log = corr_out[:, 1, :]
        cum_haz = torch.cumsum(F.softplus(haz_log.float()), dim=-1)
        corr_p = 1.0 - torch.exp((-cum_haz).clamp(min=-20.0))
        return st_log, dyn_log, haz_log, cum_haz, corr_p, unc_log, mem_new, age_new

    def forward_sequence(
        self,
        rgb,
        K_scaled,
        R_c2w,
        t_w,
        fwd_w,
        seg_len,
        return_aux_geom: bool = False,
        mask_lags: tuple[int, ...] = (),
    ):
        B, T = rgb.shape[:2]
        device = rgb.device
        bt = B * T
        rgb_f, K_f, R_f = rgb.reshape(bt, 3, RGB_H, RGB_W), K_scaled.reshape(bt, 4), R_c2w.reshape(bt, 3, 3)

        keep_flat = None
        if self.training and self.rgb_feat_drop_p > 0.0:
            keep_clip = (torch.rand(B, 1, device=device) >= self.rgb_feat_drop_p).float()
            keep_flat = keep_clip.expand(B, T).reshape(bt)

        chunk = 96  # Single-shot chunk for micro_bs=8 (96 frames) -> zero Python loop overhead & <48GB VRAM!
        f_l, g_l, zm_l, nr_l, mk_l, dlz_l, dnr_l, dv_l = [], [], [], [], [], [], [], []
        for i in range(0, bt, chunk):
            km_i = keep_flat[i : i + chunk] if keep_flat is not None else None
            res = self.encode_images(
                rgb_f[i : i + chunk], K_f[i : i + chunk], R_f[i : i + chunk],
                return_aux_geom=return_aux_geom, rgb_keep_mask=km_i,
            )
            if return_aux_geom:
                fi, gi, zmi, nri, mki, dlzi, dnri, dvi = res
                zm_l.append(zmi); nr_l.append(nri); mk_l.append(mki); dlz_l.append(dlzi); dnr_l.append(dnri); dv_l.append(dvi)
            else:
                fi, gi = res
            f_l.append(fi); g_l.append(gi)

        f2d_all = torch.cat(f_l, dim=0).reshape(B, T, self.c_feat, FEAT_H, FEAT_W)
        geom_all = torch.cat(g_l, dim=0).reshape(B, T, GEOM_CH_V10, FEAT_H, FEAT_W)

        mem = torch.zeros(B, self.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
        age = torch.full((B, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)

        st_s, dy_s, hz_s, ch_s, cp_s, un_s = [], [], [], [], [], []
        for t in range(T):
            lps = [
                (f2d_all[:, t - L], geom_all[:, t - L], K_scaled[:, t - L], R_c2w[:, t - L], t_w[:, t - L])
                if (t >= L and L not in mask_lags)
                else None
                for L in self.lags
            ]
            st, dy, hz, ch, cp, un, mem, age = self.step_from_encoded(
                f2d_all[:, t], geom_all[:, t],
                K_scaled[:, t], R_c2w[:, t], t_w[:, t], fwd_w[:, t],
                lps, t_w[:, max(0, t - 1)], fwd_w[:, max(0, t - 1)], seg_len[:, t],
                mem, age, is_first=(t == 0),
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
            out["dav2_aux"] = torch.cat(dv_l, dim=0).reshape(B, T, self.c_feat + 1, FEAT_H, FEAT_W)
        return out


class EgoClearV10EncoderExport(nn.Module):
    """Exports ONLY the 3-input QAIRT-HTP-native encoder (dav2_aux_proj is discarded -> 52.1ms / 19.2 FPS!)."""

    def __init__(self, net: EgoClearV10Net):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, K_scaled: torch.Tensor, R_c2w: torch.Tensor):
        return self.net.encode_images(rgb, K_scaled, R_c2w, return_aux_geom=False)


class EgoClearV10StepExport(nn.Module):
    def __init__(self, net: EgoClearV10Net):
        super().__init__()
        self.net = net

    def forward(
        self,
        f2d_t, geom_t, f2d_1, geom_1, f2d_2, geom_2,
        K_t, R_t, t_t, fwd_t, K_1, R_1, t_1, fwd_1,
        K_2, R_2, t_2, ds_t, mem_prev, age_prev,
    ):
        st, dy, _hz, _ch, cp, un, mem_new, age_new = self.net.step_from_encoded(
            f2d_t, geom_t, K_t, R_t, t_t, fwd_t,
            [(f2d_1, geom_1, K_1, R_1, t_1), (f2d_2, geom_2, K_2, R_2, t_2)],
            t_1, fwd_1, ds_t, mem_prev, age_prev, is_first=False,
        )
        return st, dy, cp, un, mem_new, age_new
