"""EgoClear-v3: Frozen-Geometry DAV2-S Foundation + Multi-View Parallax & Disparity-Expansion TTC + Dynamic-Suppressed Latent Memory.

Designed specifically for ZERO-SHOT SIM-TO-REAL transfer (synthetic -> real outdoor ZED / Aria glasses / phone):
  1. Frozen Depth-Anything-V2-Small (pretrained on 62M real images, Apache-2.0, mobile-NPU proven):
     - Outputs domain-invariant relative disparity + Sobel disparity edges (thin-pole sharp!)
     - Extracts multi-scale frozen visual features projected via lightweight domain-robust adapter
     - Freezing the ViT-S/14 + DPT weights prevents synthetic texture overfitting!
  2. Corridor 3D Lifting + Joint Multi-View Parallax & Disparity-Expansion Verification:
     - Projects 2,520 gravity- & heading-aligned 3D corridor hypotheses (7 vertical slices x 24 ranges x 15 lateral)
       into frame t, t-1, and t-2 using causal relative ego-motion.
     - Computes BOTH visual feature matching (cos_sim, l1_diff) AND geometric disparity-expansion consistency:
       comparing observed disparity ratio disp_t / disp_{t-k} against expected static inverse-depth expansion
       z_{t-k} / z_t (scale-free TTC / epipolar verification!).
  3. Dynamic-Suppressed Visibility-Gated Latent BEV Memory:
     - First predicts current-frame dynamic occupancy p_dyn(t) from the current observation + parallax cues.
     - Feeds (1 - p_dyn(t)) + visibility mask + feature contradiction (obs - warped_mem) into ConvGRU,
       preventing moving pedestrians from entering static memory while preserving out-of-FOV static hazards.
  4. Decision-Aligned Corridor Hazard + Height-Band (LOW/BODY/HEAD) + Epistemic Uncertainty Heads.
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


class DynamicSuppressedConvGRU(nn.Module):
    """Visibility-, contradiction-, and dynamic-suppression-aware ConvGRU on 24x15 BEV grid."""

    def __init__(self, c_mem: int, c_obs: int):
        super().__init__()
        # Inputs: mem_warped (c_mem), obs_clean (c_obs), diff (c_mem), vis_frac (1), age_decay (1), dyn_prob (1)
        c_in = c_mem + c_obs + c_mem + 3
        self.conv_zr = nn.Conv2d(c_in, 2 * c_mem, 3, padding=1)
        self.conv_h = nn.Conv2d(c_mem + c_obs + 3, c_mem, 3, padding=1)

    def forward(
        self,
        mem_warped: torch.Tensor,
        obs: torch.Tensor,
        vis_frac: torch.Tensor,
        age_decay: torch.Tensor,
        dyn_prob: torch.Tensor,
    ) -> torch.Tensor:
        # Attenuate observation write-in where current frame detects a moving pedestrian
        obs_clean = obs * (1.0 - 0.85 * dyn_prob)
        diff = obs_clean - mem_warped
        ctx = torch.cat([vis_frac, age_decay, dyn_prob], dim=1)
        zr = torch.sigmoid(self.conv_zr(torch.cat([mem_warped, obs_clean, diff, ctx], dim=1)))
        z_gate, r_gate = zr.chunk(2, dim=1)
        # Further suppress update gate when dynamic pedestrian occludes a static background cell
        z_eff = z_gate * (1.0 - 0.7 * dyn_prob)
        cand = torch.tanh(self.conv_h(torch.cat([r_gate * mem_warped, obs_clean, ctx], dim=1)))
        in_fov = (vis_frac > 0.01).float()
        mem_vis = (1.0 - z_eff) * mem_warped + z_eff * cand
        mem_unvis = mem_warped * age_decay
        return in_fov * mem_vis + (1.0 - in_fov) * mem_unvis


class EgoClearV3Net(nn.Module):
    def __init__(
        self,
        mode: str = "full",  # "single", "mvs", "full"
        dav2_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        c_feat: int = 64,
        c_bev: int = 128,
        freeze_dav2: bool = True,
    ):
        super().__init__()
        assert mode in ("single", "mvs", "full")
        self.mode = mode
        self.c_feat = c_feat
        self.c_bev = c_bev
        self.freeze_dav2 = freeze_dav2

        # Pretrained Depth-Anything-V2-Small (Apache-2.0, 62M real images)
        self.dav2 = AutoModelForDepthEstimation.from_pretrained(dav2_id)
        if freeze_dav2:
            for p in self.dav2.parameters():
                p.requires_grad = False
            self.dav2.eval()

        # Backbone hidden dim = 384 (ViT-S/14); we tap layers (6, 9, 12) + DPT relative disparity cues (4 ch)
        d_vit = self.dav2.config.backbone_config.hidden_size
        self.vis_neck = nn.Sequential(
            ConvBNReLU(3 * d_vit, 160, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 21x37 -> 42x74
        )
        # Fuse frozen multi-scale ViT features (160) + DAV2 disparity & Sobel edges (4) + ray/gravity (3) -> c_feat
        self.fuse_2d = nn.Sequential(
            ConvBNReLU(160 + 4 + 3, 128, 3, 1),
            nn.Dropout2d(0.10),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        # Sobel kernels for thin-pole & obstacle-edge disparity gradient extraction
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", kx, persistent=False)
        self.register_buffer("sobel_y", ky, persistent=False)

        # Canonical 3D walker-frame query grid (2520, 3) = [x_right, z_fwd, y_rel]
        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        self.register_buffer("grid_xzy", torch.stack([gx, gz, gy], dim=-1).reshape(-1, 3), persistent=False)
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)

        # Per-3D-point channels:
        #   single: c_feat + 4 (disp_norm, gx, gy, log_disp) + 1 (vis_t) + 3 (u_norm, v_norm, 1/z_hyp) = c_feat + 8
        #   mvs/full: + 10 temporal parallax + disparity-expansion channels (2 lags x 5: cos_sim, l1, disp_exp_err, vis, base)
        c_pt = c_feat + 8 + (10 if mode in ("mvs", "full") else 0)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 40, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
        )
        self.bev_enc = nn.Sequential(
            ConvBNReLU(40 * NY + 2, c_bev, 3, 1),
            ResBlock2D(c_bev),
        )

        # Instantaneous dynamic-agent detector on current BEV observation (runs BEFORE memory write!)
        self.head_dynamic_obs = nn.Sequential(
            ConvBNReLU(c_bev, 64, 3, 1),
            nn.Conv2d(64, 3, 1),
        )

        if mode == "full":
            self.mem_gru = DynamicSuppressedConvGRU(c_bev, c_bev)

        self.bev_refine = nn.Sequential(
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
        )
        self.head_static = nn.Conv2d(c_bev, 3, 1)

        # Central 7 columns (corridor + 2 side-wall context cols each side: x in [-0.70, +0.70] m)
        self.corr_head = nn.Sequential(
            nn.Conv1d(c_bev * 7, 160, 3, padding=1, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.Conv1d(160, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 2, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dav2:
            self.dav2.eval()
        return self

    def encode_images(
        self,
        rgb: torch.Tensor,      # (N, 3, 294, 518) ImageNet-normalized
        K_scaled: torch.Tensor, # (N, 4)
        R_c2w: torch.Tensor,    # (N, 3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (f2d: (N, c_feat, 42, 74), disp_cues: (N, 4, 42, 74))."""
        ctx = torch.no_grad() if self.freeze_dav2 else torch.enable_grad()
        with ctx:
            # Backbone hidden states + full DPT relative disparity map
            bb_out = self.dav2.backbone(pixel_values=rgb, output_hidden_states=True)
            # Run DPT neck + head directly from backbone features
            dpt_feats = self.dav2.neck(bb_out.feature_maps, patch_height=RGB_H // 14, patch_width=RGB_W // 14)
            disp_raw = self.dav2.head(dpt_feats, patch_height=RGB_H // 14, patch_width=RGB_W // 14)  # (N, H', W')

            # Extract ViT-S/14 spatial tokens from hidden_states (layers 6, 9, 12)
            hs = bb_out.hidden_states
            h_p, w_p = RGB_H // 14, RGB_W // 14
            toks = []
            for idx in (6, 9, 12):
                tk = hs[idx][:, 1:, :]
                toks.append(tk.transpose(1, 2).reshape(-1, tk.shape[-1], h_p, w_p))
            vit_cat = torch.cat(toks, dim=1)

            # Downsample relative disparity cleanly to (42, 74) and normalize per-frame (ONNX/HTP-native sort quantiles!)
            d_42 = F.interpolate(disp_raw.unsqueeze(1).float(), size=(42, 74), mode="bilinear", align_corners=False)
            d_sorted, _ = torch.sort(d_42.flatten(1), dim=1)
            n_px = d_sorted.shape[1]
            q10 = d_sorted[:, int(0.10 * (n_px - 1))].view(-1, 1, 1, 1)
            q90 = d_sorted[:, int(0.90 * (n_px - 1))].view(-1, 1, 1, 1)
            d_norm = ((d_42 - q10) / (q90 - q10).clamp(min=1e-3)).clamp(-0.5, 2.5)
            d_gx = F.conv2d(d_norm, self.sobel_x, padding=1)
            d_gy = F.conv2d(d_norm, self.sobel_y, padding=1)
            d_log = torch.log1p(d_42.clamp(min=0.0))
            disp_cues = torch.cat([d_norm, d_gx, d_gy, d_log], dim=1)  # (N, 4, 42, 74)

        f2d = self.vis_neck(vit_cat.to(rgb.dtype))  # (N, 160, 42, 74)
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

        rgb_flat = rgb.reshape(B * T, 3, RGB_H, RGB_W)
        K_flat = K_scaled.reshape(B * T, 4)
        R_flat = R_c2w.reshape(B * T, 3, 3)
        bt = B * T
        if bt <= 512:
            f2d_flat, dc_flat = self.encode_images(rgb_flat, K_flat, R_flat)
        else:
            outs = [self.encode_images(rgb_flat[i : i + 256], K_flat[i : i + 256], R_flat[i : i + 256]) for i in range(0, bt, 256)]
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

        static_logits_seq, dynamic_logits_seq, corr_logits_seq, unc_logits_seq = [], [], [], []
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
            pt_parts = [sf_t, sd_t, vis_t, uv_invz_t]  # c_feat + 8

            if self.mode in ("mvs", "full"):
                ft_norm = F.normalize(sf_t, dim=1, eps=1e-5)
                # log_disp is channel 3 of sd_t -> expm1 recovers positive disparity
                log_dt = sd_t[:, 3:4]
                for lag in (1, 2):
                    if t >= lag:
                        tp = t - lag
                        sf_p, sd_p, vis_p, _, zc_p = self._project_and_sample(
                            f2d_all[:, tp], dc_all[:, tp], pts_w, R_c2w[:, tp], t_w[:, tp], K_scaled[:, tp]
                        )
                        both_v = vis_t * vis_p
                        fp_norm = F.normalize(sf_p, dim=1, eps=1e-5)
                        cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
                        l1_diff = (sf_t - sf_p).abs().mean(dim=1, keepdim=True) * both_v
                        # Scale-free disparity-expansion / TTC consistency:
                        # For a static point at exact hypothesized depth, log(disp_t/disp_p) ~= log(zc_p/zc_t)
                        obs_log_exp = log_dt - sd_p[:, 3:4]
                        geom_log_exp = torch.log(zc_p / zc_t.clamp(min=0.15))
                        disp_exp_err = (obs_log_exp - geom_log_exp).clamp(-2.0, 2.0) * both_v
                        base = torch.linalg.norm(tt - t_w[:, tp], dim=-1).clamp(max=2.0)
                        base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(ft.dtype) * both_v
                        pt_parts.extend([cos_sim, l1_diff, disp_exp_err, both_v, base_map])
                    else:
                        pt_parts.append(torch.zeros(B, 5, pts_w.shape[1], device=device, dtype=ft.dtype))

            pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
            pt_red = self.pt_reduce(pt_feat).reshape(B, 40 * NY, BEV_NZ, BEV_NX)
            bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))

            # Predict current-frame dynamic occupancy BEFORE writing into static memory
            dyn_log = self.head_dynamic_obs(bev_obs)
            dyn_prob_any = torch.sigmoid(dyn_log).max(dim=1, keepdim=True).values.detach()

            if self.mode == "full":
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
            else:
                bev_feat = self.bev_refine(bev_obs)

            static_logits_seq.append(self.head_static(bev_feat))
            dynamic_logits_seq.append(dyn_log)
            corr_strip = bev_feat[:, :, :, 4:11].permute(0, 1, 3, 2).reshape(B, self.c_bev * 7, BEV_NZ)
            corr_out = self.corr_head(corr_strip)
            corr_logits_seq.append(corr_out[:, 0])
            unc_logits_seq.append(corr_out[:, 1])

        return {
            "static_logits": torch.stack(static_logits_seq, dim=1),
            "dynamic_logits": torch.stack(dynamic_logits_seq, dim=1),
            "corr_logits": torch.stack(corr_logits_seq, dim=1),
            "unc_logits": torch.stack(unc_logits_seq, dim=1),
        }
