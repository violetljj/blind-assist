"""EgoClear-v6: Frozen DAV2-Base (97.5M, 768-dim) + Multi-Scale Long-Baseline 3-Lag MVS (lag=1,2,4)
+ Dynamic-Suppressed ConvGRU + Monotonic First-Hit Survival Hazard Head.

Key upgrades over EgoClear-v4/v5:
  1. Frozen Depth-Anything-V2-Base (ViT-B/14, 97.5M, 768-dim, Apache-2.0, 62M real images):
     - Much sharper relative disparity & semantic geometry on thin poles, fences, curbs, low-light outdoor scenes.
     - 100% frozen backbone prevents synthetic texture overfitting; trainable head remains compact (~4.4M params).
  2. Multi-Scale 3-Lag Stereo Parallax & Disparity-Expansion TTC (lags = 1, 2, 4 -> 0.2s, 0.4s, 0.8s):
     - Doubles physical stereo baseline to ~0.7-1.0m at normal walking speed, sharpening metric triangulation at 3.0-5.1m.
  3. Monotonic First-Hit Survival Hazard Head + has_dyn_gt loss masking (cleanly trains on real outdoor camera_head
     sessions without semantic dynamic masks, without poisoning the dynamic-pedestrian suppression head!).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

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


class EgoClearV6Net(nn.Module):
    """EgoClear-v6 with frozen DAV2-Base (or DAV2-Small) + 3-lag (1,2,4) multi-scale stereo + monotonic survival."""

    def __init__(
        self,
        dav2_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
        c_feat: int = 64,
        c_bev: int = 128,
        lags: tuple[int, ...] = (1, 2, 4),
        freeze_dav2: bool = True,
    ):
        super().__init__()
        self.dav2_id = dav2_id
        self.c_feat = c_feat
        self.c_bev = c_bev
        self.lags = tuple(lags)
        self.freeze_dav2 = freeze_dav2

        self.dav2 = AutoModelForDepthEstimation.from_pretrained(dav2_id)
        if freeze_dav2:
            for p in self.dav2.parameters():
                p.requires_grad = False
            self.dav2.eval()

        # Backbone hidden dim: 768 for DAV2-Base (384 for DAV2-Small); tap layers (6, 9, 12)
        d_vit = self.dav2.config.backbone_config.hidden_size
        self.vis_neck = nn.Sequential(
            ConvBNReLU(3 * d_vit, 192, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 21x37 -> 42x74
        )
        self.fuse_2d = nn.Sequential(
            ConvBNReLU(192 + 4 + 3, 128, 3, 1),
            nn.Dropout2d(0.10),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", kx, persistent=False)
        self.register_buffer("sobel_y", ky, persistent=False)

        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        self.register_buffer("grid_xzy", torch.stack([gx, gz, gy], dim=-1).reshape(-1, 3), persistent=False)
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)

        # c_feat + 8 (disp4 + vis + uv_invz3) + 5 * len(lags) multi-scale stereo channels
        c_pt = c_feat + 8 + 5 * len(self.lags)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 44, 1, bias=False),
            nn.BatchNorm2d(44),
            nn.GELU(),
        )
        self.bev_enc = nn.Sequential(
            ConvBNReLU(44 * NY + 2, c_bev, 3, 1),
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
        if self.freeze_dav2:
            self.dav2.eval()
        return self

    def encode_images(
        self,
        rgb: torch.Tensor,
        K_scaled: torch.Tensor,
        R_c2w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if self.freeze_dav2 else torch.enable_grad()
        with ctx:
            bb_out = self.dav2.backbone(pixel_values=rgb, output_hidden_states=True)
            dpt_feats = self.dav2.neck(bb_out.feature_maps, patch_height=RGB_H // 14, patch_width=RGB_W // 14)
            disp_raw = self.dav2.head(dpt_feats, patch_height=RGB_H // 14, patch_width=RGB_W // 14)

            hs = bb_out.hidden_states
            h_p, w_p = RGB_H // 14, RGB_W // 14
            toks = []
            for idx in (6, 9, 12):
                tk = hs[idx][:, 1:, :]
                toks.append(tk.transpose(1, 2).reshape(-1, tk.shape[-1], h_p, w_p))
            vit_cat = torch.cat(toks, dim=1)

            d_42 = F.interpolate(disp_raw.unsqueeze(1).float(), size=(42, 74), mode="bilinear", align_corners=False)
            d_sorted, _ = torch.sort(d_42.flatten(1), dim=1)
            n_px = d_sorted.shape[1]
            q10 = d_sorted[:, int(0.10 * (n_px - 1))].view(-1, 1, 1, 1)
            q90 = d_sorted[:, int(0.90 * (n_px - 1))].view(-1, 1, 1, 1)
            d_norm = ((d_42 - q10) / (q90 - q10).clamp(min=1e-3)).clamp(-0.5, 2.5)
            d_gx = F.conv2d(d_norm, self.sobel_x, padding=1)
            d_gy = F.conv2d(d_norm, self.sobel_y, padding=1)
            d_log = torch.log1p(d_42.clamp(min=0.0))
            disp_cues = torch.cat([d_norm, d_gx, d_gy, d_log], dim=1)

        f2d = self.vis_neck(vit_cat.to(rgb.dtype))
        disp_cues = disp_cues.to(f2d.dtype)
        N, _, fh, fw = f2d.shape

        device = f2d.device
        u = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * (RGB_W / fw)
        v = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * (RGB_H / fh)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        fx, fy = K_scaled[:, 0, None, None], K_scaled[:, 1, None, None]
        cx, cy = K_scaled[:, 2, None, None], K_scaled[:, 3, None, None]
        rx = (uu[None] - cx) / fx
        ry = (vv[None] - cy) / fy
        rays_unit = F.normalize(torch.stack([rx, ry, torch.ones_like(rx)], dim=-1), dim=-1)
        grav_c = -R_c2w[:, 2, :]
        grav_dot = (rays_unit * grav_c[:, None, None, :]).sum(dim=-1)
        geom = torch.stack([rx, ry, grav_dot], dim=1).to(f2d.dtype)

        feat_2d = self.fuse_2d(torch.cat([f2d, disp_cues, geom], dim=1))
        return feat_2d, disp_cues

    def _project_and_sample(
        self,
        f2d: torch.Tensor,
        disp_cues: torch.Tensor,
        pts_w: torch.Tensor,
        R_c2w: torch.Tensor,
        t_w: torch.Tensor,
        K_scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        diff = pts_w - t_w[:, None, :]
        p_c = torch.bmm(diff, R_c2w)
        zc = p_c[..., 2]
        in_front = zc > 0.15
        zc_safe = zc.clamp(min=0.15)
        u = K_scaled[:, 0:1] * (p_c[..., 0] / zc_safe) + K_scaled[:, 2:3]
        v = K_scaled[:, 1:2] * (p_c[..., 1] / zc_safe) + K_scaled[:, 3:4]
        u_norm = 2.0 * (u / (RGB_W - 1.0)) - 1.0
        v_norm = 2.0 * (v / (RGB_H - 1.0)) - 1.0
        vis = in_front & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1).to(f2d.dtype)
        both = torch.cat([f2d, disp_cues], dim=1)
        samp_all = F.grid_sample(both, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(2)
        vis_f = vis.unsqueeze(1).to(f2d.dtype)
        samp_all = samp_all * vis_f
        samp_f = samp_all[:, : self.c_feat]
        samp_d = samp_all[:, self.c_feat :]
        uv_invz = torch.stack([u_norm.clamp(-1.5, 1.5), v_norm.clamp(-1.5, 1.5), 1.0 / zc_safe], dim=1).to(f2d.dtype)
        return samp_f, samp_d, vis_f, uv_invz, zc_safe.unsqueeze(1).to(f2d.dtype)

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

    def forward_sequence(
        self,
        rgb: torch.Tensor,
        K_scaled: torch.Tensor,
        R_c2w: torch.Tensor,
        t_w: torch.Tensor,
        fwd_w: torch.Tensor,
        seg_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T = rgb.shape[:2]
        device = rgb.device
        bt = B * T
        rgb_flat = rgb.reshape(bt, 3, RGB_H, RGB_W)
        K_flat = K_scaled.reshape(bt, 4)
        R_flat = R_c2w.reshape(bt, 3, 3)

        # Chunk DAV2-Base encoding to keep peak VRAM smooth on 96 GB G4
        chunk_sz = 192
        if bt <= chunk_sz:
            f2d_flat, dc_flat = self.encode_images(rgb_flat, K_flat, R_flat)
        else:
            outs = [self.encode_images(rgb_flat[i : i + chunk_sz], K_flat[i : i + chunk_sz], R_flat[i : i + chunk_sz]) for i in range(0, bt, chunk_sz)]
            f2d_flat = torch.cat([o[0] for o in outs], dim=0)
            dc_flat = torch.cat([o[1] for o in outs], dim=0)

        f2d_all = f2d_flat.reshape(B, T, self.c_feat, 42, 74)
        dc_all = dc_flat.reshape(B, T, 4, 42, 74)

        bev_pos = torch.stack(
            [
                self.bev_xz[..., 0] / BEV_X_MAX,
                (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
            ],
            dim=0,
        ).unsqueeze(0).expand(B, -1, -1, -1).to(f2d_all.dtype)

        mem = torch.zeros(B, self.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
        age = torch.full((B, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)

        static_logits_seq, dynamic_logits_seq, hazard_logits_seq, unc_logits_seq = [], [], [], []
        gx = self.grid_xzy[:, 0][None, :, None]
        gz = self.grid_xzy[:, 1][None, :, None]
        gy = self.grid_xzy[:, 2][None, :, None]
        up_w = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)[None, None, :]

        for t in range(T):
            ft, dct = f2d_all[:, t], dc_all[:, t]
            Rt, tt, Kt, fwdt = R_c2w[:, t], t_w[:, t], K_scaled[:, t], fwd_w[:, t]
            rightt = torch.stack([fwdt[:, 1], -fwdt[:, 0], torch.zeros_like(fwdt[:, 0])], dim=-1)
            pts_w = tt[:, None, :] + gx * rightt[:, None, :] + gz * fwdt[:, None, :] + gy * up_w

            sf_t, sd_t, vis_t, uv_invz_t, zc_t = self._project_and_sample(ft, dct, pts_w, Rt, tt, Kt)
            pt_parts = [sf_t, sd_t, vis_t, uv_invz_t]
            ft_norm = F.normalize(sf_t, dim=1, eps=1e-5)
            log_dt = sd_t[:, 3:4]

            # Multi-scale 3-lag stereo matching (lag = 1, 2, 4 -> 0.2s, 0.4s, 0.8s!)
            for lag in self.lags:
                if t >= lag:
                    tp = t - lag
                    sf_p, sd_p, vis_p, _, zc_p = self._project_and_sample(
                        f2d_all[:, tp], dc_all[:, tp], pts_w, R_c2w[:, tp], t_w[:, tp], K_scaled[:, tp]
                    )
                    both_v = vis_t * vis_p
                    fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
                    cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
                    l1_diff = (sf_t - sf_p).abs().mean(dim=1, keepdim=True) * both_v
                    obs_log_exp = log_dt - sd_p[:, 3:4]
                    geom_log_exp = torch.log(zc_p / zc_t.clamp(min=0.15))
                    disp_exp_err = (obs_log_exp - geom_log_exp).clamp(-2.0, 2.0) * both_v
                    base = torch.linalg.norm(tt - t_w[:, tp], dim=-1).clamp(max=2.5)
                    base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(ft.dtype) * both_v
                    pt_parts.extend([cos_sim, l1_diff, disp_exp_err, both_v, base_map])
                else:
                    pt_parts.append(torch.zeros(B, 5, pts_w.shape[1], device=device, dtype=ft.dtype))

            pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
            pt_red = self.pt_reduce(pt_feat).reshape(B, 44 * NY, BEV_NZ, BEV_NX)
            bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))

            dyn_log = self.head_dynamic_obs(bev_obs)
            dyn_prob_any = torch.sigmoid(dyn_log).max(dim=1, keepdim=True).values.detach()

            vis_bev = vis_t.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True)
            if t > 0:
                mem_w = self._warp_bev(mem, t_w[:, t - 1], fwd_w[:, t - 1], tt, fwdt)
                age_w = self._warp_bev(age, t_w[:, t - 1], fwd_w[:, t - 1], tt, fwdt)
                ds = seg_len[:, t].clamp(0.0, 2.0)[:, None, None, None].to(ft.dtype)
                age_w = (age_w + ds).clamp(max=10.0)
                age_decay = torch.exp(-age_w / 4.0)
                mem = self.mem_gru(mem_w, bev_obs, vis_bev, age_decay, dyn_prob_any)
                age = torch.where(vis_bev > 0.05, torch.zeros_like(age_w), age_w)
            else:
                mem = bev_obs * (1.0 - 0.85 * dyn_prob_any)
                age = torch.where(vis_bev > 0.05, torch.zeros_like(age), torch.full_like(age, 10.0))

            bev_feat = self.bev_refine(mem)
            static_logits_seq.append(self.head_static(bev_feat))
            dynamic_logits_seq.append(dyn_log)
            corr_strip = bev_feat[:, :, :, 4:11].permute(0, 1, 3, 2).reshape(B, self.c_bev * 7, BEV_NZ)
            corr_out = self.corr_head(corr_strip)
            hazard_logits_seq.append(corr_out[:, 0])
            unc_logits_seq.append(corr_out[:, 1])

        haz_log = torch.stack(hazard_logits_seq, dim=1)
        return {
            "static_logits": torch.stack(static_logits_seq, dim=1),
            "dynamic_logits": torch.stack(dynamic_logits_seq, dim=1),
            "hazard_logits": haz_log,
            "corr_probs": survival_cum_probs(haz_log),
            "unc_logits": torch.stack(unc_logits_seq, dim=1),
        }


class EgoClearV6StepExport(nn.Module):
    """Stage-2 Mobile ONNX streaming step for EgoClear-v6 (3-lag: t-1, t-2, t-4 -> monotonic corr_probs)."""

    def __init__(self, core: EgoClearV6Net):
        super().__init__()
        self.core = core

    def forward(
        self,
        f2d_t: torch.Tensor, dc_t: torch.Tensor,
        f2d_1: torch.Tensor, dc_1: torch.Tensor,
        f2d_2: torch.Tensor, dc_2: torch.Tensor,
        f2d_4: torch.Tensor, dc_4: torch.Tensor,
        K_t: torch.Tensor, R_t: torch.Tensor, t_t: torch.Tensor, fwd_t: torch.Tensor,
        K_1: torch.Tensor, R_1: torch.Tensor, t_1: torch.Tensor, fwd_1: torch.Tensor,
        K_2: torch.Tensor, R_2: torch.Tensor, t_2: torch.Tensor,
        K_4: torch.Tensor, R_4: torch.Tensor, t_4: torch.Tensor,
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

        for fp, dp, Rp, tp, Kp in ((f2d_1, dc_1, R_1, t_1, K_1), (f2d_2, dc_2, R_2, t_2, K_2), (f2d_4, dc_4, R_4, t_4, K_4)):
            sf_p, sd_p, vis_p, _, zc_p = c._project_and_sample(fp, dp, pts_w, Rp, tp, Kp)
            both_v = vis_t * vis_p
            fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
            cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
            l1_diff = (sf_t - sf_p).abs().mean(dim=1, keepdim=True) * both_v
            disp_exp_err = ((log_dt - sd_p[:, 3:4]) - torch.log(zc_p / zc_t.clamp(min=0.15))).clamp(-2.0, 2.0) * both_v
            base = torch.linalg.norm(t_t - tp, dim=-1).clamp(max=2.5)[:, None, None].expand(B, 1, pts_w.shape[1]) * both_v
            pt_parts.extend([cos_sim, l1_diff, disp_exp_err, both_v, base])

        pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
        pt_red = c.pt_reduce(pt_feat).reshape(B, 44 * NY, BEV_NZ, BEV_NX)
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
        return st_log, dyn_log, survival_cum_probs(corr_out[:, 0]), corr_out[:, 1], mem_new, age_new


def compute_loss_v6(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """V6 loss with has_dyn_gt mask (supports seg-free real camera_head sessions) + Sub-MinVox soft mask + 5m boost."""
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
    has_dyn = batch["has_dyn_gt"].float()  # (B,) 1.0 if dynamic labels valid, 0.0 if seg-free outdoor head cam

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
    dy_mask = has_dyn[:, None, None, None, None].expand_as(bce_dy)
    loss_dy = (bce_dy * dy_mask).sum() / dy_mask.sum().clamp(min=1.0)

    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=device, dtype=torch.float32)
    idx_4m = int((z_edges <= H_CORR + 1e-4).sum().item()) - 1
    z_w = torch.where(z_edges <= H_CORR + 0.2, 2.0, 1.2)[None, None, :]

    y_s = y_co * 0.97 + 0.015
    pos_w = 2.2
    bce_cum = -(pos_w * y_s * torch.log(p_cum) + (1.0 - y_s) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.2) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    bce_fh = F.binary_cross_entropy_with_logits(haz_log, y_first, pos_weight=torch.tensor(2.6, device=device), reduction="none")
    loss_fh = (bce_fh * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)
    loss_co = loss_cum + 0.30 * loss_fh

    p4 = p_cum[:, :, idx_4m]
    y4 = y_co[:, :, idx_4m]
    clear = v_co * (1.0 - y4) * (dn_w >= 0.5).float()
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
        "tot": float(total.item()), "st": float(loss_st.item()),
        "dy": float(loss_dy.item()), "co": float(loss_co.item()), "flk": float(loss_flk.item()),
    }
