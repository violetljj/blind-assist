"""EgoClear-Net: Pure-Vision Corridor-Aligned Multi-View Parallax + Dynamic-Aware Latent Memory.

Three progressive architectural variants sharing the exact same mobile-friendly ViT-S/14 backbone:
  1. `single`: Single-frame camera- & gravity-aware 3D corridor BEV lifting + decision-aligned heads
  2. `mvs`:    + Built-in 2-frame/3-frame static motion-parallax stereo cost volume in corridor BEV
  3. `full`:   + Exact metric-warped, visibility- & contradiction-gated recurrent latent BEV memory

Zero ToF, zero ground-plane fitting, zero exact eye-height assumption.
Inputs per frame: RGB (294x518), camera intrinsics K, gravity (from R_cam2world), causal relative
pose + causal walking heading (fwd, right).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Egocentric walker-frame BEV grid matching pack_sanpo_colab.py
BEV_X_MIN, BEV_X_MAX, BEV_DX, BEV_NX = -1.50, 1.50, 0.20, 15
BEV_Z_MIN, BEV_Z_MAX, BEV_DZ, BEV_NZ = 0.30, 5.10, 0.20, 24
# 7 vertical slices relative to camera optical center (meters, gravity-up), covering ground-to-head
# across head-mounted (~1.55 m) and chest-mounted (~1.20 m) cameras with zero floor-fitting needed
Y_SLICES = (-1.50, -1.15, -0.80, -0.45, -0.10, 0.25, 0.60)
NY = len(Y_SLICES)
RGB_H, RGB_W = 294, 518


class ConvBNReLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock2D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(x + h)


class ConvGRU2D(nn.Module):
    """Visibility- & contradiction-aware ConvGRU on the 24x15 egocentric BEV grid."""

    def __init__(self, c_mem: int, c_obs: int):
        super().__init__()
        # Input channels: warped_mem (c_mem), obs (c_obs), diff (c_mem), vis_frac (1), age_decay (1)
        c_in = c_mem + c_obs + c_mem + 2
        self.conv_zr = nn.Conv2d(c_in, 2 * c_mem, 3, padding=1)
        self.conv_h = nn.Conv2d(c_mem + c_obs + 2, c_mem, 3, padding=1)

    def forward(
        self,
        mem_warped: torch.Tensor,
        obs: torch.Tensor,
        vis_frac: torch.Tensor,
        age_decay: torch.Tensor,
    ) -> torch.Tensor:
        diff = obs - mem_warped
        zr = torch.sigmoid(self.conv_zr(torch.cat([mem_warped, obs, diff, vis_frac, age_decay], dim=1)))
        z_gate, r_gate = zr.chunk(2, dim=1)
        cand = torch.tanh(self.conv_h(torch.cat([r_gate * mem_warped, obs, vis_frac, age_decay], dim=1)))
        # Where cell is completely outside current FOV (vis_frac == 0), gently decay warped memory
        # rather than overwriting with zero observation
        in_fov = (vis_frac > 0.01).float()
        mem_vis = (1.0 - z_gate) * mem_warped + z_gate * cand
        mem_unvis = mem_warped * age_decay
        return in_fov * mem_vis + (1.0 - in_fov) * mem_unvis


class EgoClearNet(nn.Module):
    def __init__(
        self,
        mode: str = "full",  # "single", "mvs", "full"
        backbone_name: str = "facebook/dinov2-small",
        c_feat: int = 64,
        c_bev: int = 96,
        unfreeze_last_k: int = 4,
    ):
        super().__init__()
        assert mode in ("single", "mvs", "full")
        self.mode = mode
        self.c_feat = c_feat
        self.c_bev = c_bev

        # Pretrained mobile-exportable ViT-S/14 (22M params, Apache-2.0)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        d_vit = self.backbone.config.hidden_size  # 384 for dinov2-small

        # Freeze early blocks, fine-tune last K blocks for corridor/thin-pole adaptation
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_k > 0:
            layers = self.backbone.encoder.layer
            for blk in layers[-unfreeze_last_k:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        # Multi-layer neck (layers 5, 8, 11 -> 3*384) + ray/gravity conditioning (3 ch) -> c_feat at 42x74
        self.neck_proj = nn.Sequential(
            ConvBNReLU(3 * d_vit, 192, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.neck_fuse = nn.Sequential(
            ConvBNReLU(192 + 3, 128, 3, 1),
            ConvBNReLU(128, c_feat, 3, 1),
        )

        # Build canonical 3D walker-frame query grid (NY, NZ, NX, 3) -> (x_right, z_fwd, y_up_rel)
        xs = torch.linspace(BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX, BEV_NX)
        zs = torch.linspace(BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ, BEV_NZ)
        ys = torch.tensor(Y_SLICES, dtype=torch.float32)
        gy, gz, gx = torch.meshgrid(ys, zs, xs, indexing="ij")
        # Store as (NY*NZ*NX, 3) = (2520, 3) with columns [x_right, z_fwd, y_rel]
        self.register_buffer("grid_xzy", torch.stack([gx, gz, gy], dim=-1).reshape(-1, 3), persistent=False)
        # 2D BEV cell centers (NZ, NX, 2) for metric ego-motion warping
        bz, bx = torch.meshgrid(zs, xs, indexing="ij")
        self.register_buffer("bev_xz", torch.stack([bx, bz], dim=-1), persistent=False)

        # Per-3D-point feature channels before collapsing Y slices:
        #   single: c_feat + 1 (vis_t) + 2 (norm_u, norm_v) + 1 (inv_depth_hyp) = c_feat + 4
        #   mvs/full: + 8 MVS temporal parallax channels (sim_1, l1_1, vis_1, base_1, sim_2, l1_2, vis_2, base_2)
        c_pt = c_feat + 4 + (8 if mode in ("mvs", "full") else 0)
        self.pt_reduce = nn.Sequential(
            nn.Conv2d(c_pt, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Collapse NY=7 vertical slices into BEV tensor (32 * 7 = 224 -> c_bev)
        self.bev_enc = nn.Sequential(
            ConvBNReLU(32 * NY + 2, c_bev, 3, 1),  # +2 for (x, z) positional coords
            ResBlock2D(c_bev),
        )

        if mode == "full":
            self.mem_gru = ConvGRU2D(c_bev, c_bev)

        self.bev_refine = nn.Sequential(
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
            ResBlock2D(c_bev),
        )

        # Multi-task decision-aligned heads
        # 1. Static obstacle BEV occupancy (3 height bands: LOW, BODY, HEAD)
        self.head_static = nn.Conv2d(c_bev, 3, 1)
        # 2. Current-frame dynamic agent BEV occupancy (3 height bands)
        self.head_dynamic = nn.Conv2d(c_bev, 3, 1)
        # 3. Direct cumulative corridor hazard probability P(static obstacle in [-0.3,+0.3] at dist <= z_k)
        #    Pools central 7 columns (corridor + 2 side-wall context cols each side: x in [-0.7, +0.7])
        self.corr_head = nn.Sequential(
            nn.Conv1d(c_bev * 7, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 2, 1),  # [0]: per-bin hazard logit, [1]: epistemic uncertainty logit
        )

    def encode_images(
        self,
        rgb: torch.Tensor,      # (N, 3, 294, 518) ImageNet-normalized
        K_scaled: torch.Tensor, # (N, 4) [fx, fy, cx, cy] at 518x294
        R_c2w: torch.Tensor,    # (N, 3, 3)
    ) -> torch.Tensor:
        out = self.backbone(pixel_values=rgb, interpolate_pos_encoding=True, output_hidden_states=True)
        # DINOv2 hidden_states: [embed, layer1, ..., layer12]; remove CLS token (index 0)
        hs = out.hidden_states
        h_p, w_p = RGB_H // 14, RGB_W // 14  # 21, 37
        feats = []
        for idx in (6, 9, 12):
            tok = hs[idx][:, 1:, :]  # (N, 777, 384)
            feats.append(tok.transpose(1, 2).reshape(-1, tok.shape[-1], h_p, w_p))
        f2d = self.neck_proj(torch.cat(feats, dim=1))  # (N, 192, 42, 74)
        N, _, fh, fw = f2d.shape

        # Compute per-pixel camera unit rays + gravity alignment at (42, 74)
        device, dtype = f2d.device, f2d.dtype
        u = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * (RGB_W / fw)
        v = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * (RGB_H / fh)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # (fh, fw)
        fx = K_scaled[:, 0, None, None]
        fy = K_scaled[:, 1, None, None]
        cx = K_scaled[:, 2, None, None]
        cy = K_scaled[:, 3, None, None]
        rx = (uu[None] - cx) / fx
        ry = (vv[None] - cy) / fy
        rz = torch.ones_like(rx)
        rays = torch.stack([rx, ry, rz], dim=-1)  # (N, fh, fw, 3)
        rays_unit = F.normalize(rays, dim=-1)
        # Gravity in camera frame: R_c2w^T @ [0, 0, -1] = -R_c2w[:, 2, :]
        grav_c = -R_c2w[:, 2, :]  # (N, 3)
        grav_dot = (rays_unit * grav_c[:, None, None, :]).sum(dim=-1)  # (N, fh, fw)
        geom = torch.stack([rx, ry, grav_dot], dim=1).to(dtype)
        return self.neck_fuse(torch.cat([f2d, geom], dim=1))  # (N, c_feat, 42, 74)

    def _project_and_sample(
        self,
        f2d: torch.Tensor,      # (B, c_feat, fh, fw)
        pts_w: torch.Tensor,    # (B, M, 3) world coordinates relative to target camera origin? No, absolute world
        R_c2w: torch.Tensor,    # (B, 3, 3)
        t_w: torch.Tensor,      # (B, 3)
        K_scaled: torch.Tensor, # (B, 4)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project world points pts_w (B, M, 3) into camera (R_c2w, t_w, K_scaled) and sample f2d."""
        # p_c = R^T (p_w - t)
        diff = pts_w - t_w[:, None, :]  # (B, M, 3)
        p_c = torch.bmm(diff, R_c2w)    # since (R^T v)^T = v^T R -> (B, M, 3)
        zc = p_c[..., 2]
        in_front = zc > 0.15
        zc_safe = zc.clamp(min=0.15)
        u = K_scaled[:, 0:1] * (p_c[..., 0] / zc_safe) + K_scaled[:, 2:3]
        v = K_scaled[:, 1:2] * (p_c[..., 1] / zc_safe) + K_scaled[:, 3:4]
        u_norm = 2.0 * (u / (RGB_W - 1.0)) - 1.0
        v_norm = 2.0 * (v / (RGB_H - 1.0)) - 1.0
        vis = in_front & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1).to(f2d.dtype)  # (B, 1, M, 2)
        samp = F.grid_sample(f2d, grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(2)  # (B, C, M)
        vis_f = vis.unsqueeze(1).to(f2d.dtype)  # (B, 1, M)
        samp = samp * vis_f
        uv_invz = torch.stack([u_norm.clamp(-1.5, 1.5), v_norm.clamp(-1.5, 1.5), 1.0 / zc_safe], dim=1).to(f2d.dtype)
        return samp, vis_f, uv_invz

    def _warp_bev(
        self,
        mem_prev: torch.Tensor,   # (B, C, NZ, NX)
        t_prev: torch.Tensor,     # (B, 3)
        fwd_prev: torch.Tensor,   # (B, 3)
        t_cur: torch.Tensor,      # (B, 3)
        fwd_cur: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:
        """Exact metric 2D affine warp of BEV tensor from walker frame (t-1) to walker frame t."""
        B = mem_prev.shape[0]
        right_cur = torch.stack([fwd_cur[:, 1], -fwd_cur[:, 0]], dim=-1)    # (B, 2)
        right_prev = torch.stack([fwd_prev[:, 1], -fwd_prev[:, 0]], dim=-1) # (B, 2)
        bx = self.bev_xz[..., 0][None]  # (1, NZ, NX)
        bz = self.bev_xz[..., 1][None]  # (1, NZ, NX)
        # World XY of each current BEV cell
        qw = (
            t_cur[:, None, None, :2]
            + bx[..., None] * right_cur[:, None, None, :]
            + bz[..., None] * fwd_cur[:, None, None, :2]
        )
        rel_prev = qw - t_prev[:, None, None, :2]
        x_prev = (rel_prev * right_prev[:, None, None, :]).sum(dim=-1)
        z_prev = (rel_prev * fwd_prev[:, None, None, :2]).sum(dim=-1)

        # Normalize to [-1, 1] matching cell centers of mem_prev (width=NX along x, height=NZ along z)
        x_lo, x_hi = BEV_X_MIN + 0.5 * BEV_DX, BEV_X_MAX - 0.5 * BEV_DX
        z_lo, z_hi = BEV_Z_MIN + 0.5 * BEV_DZ, BEV_Z_MAX - 0.5 * BEV_DZ
        gx = 2.0 * (x_prev - x_lo) / (x_hi - x_lo) - 1.0
        gz = 2.0 * (z_prev - z_lo) / (z_hi - z_lo) - 1.0
        grid = torch.stack([gx, gz], dim=-1).to(mem_prev.dtype)  # (B, NZ, NX, 2)
        return F.grid_sample(mem_prev, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def forward_sequence(
        self,
        rgb: torch.Tensor,        # (B, T, 3, 294, 518)
        K_scaled: torch.Tensor,   # (B, T, 4)
        R_c2w: torch.Tensor,      # (B, T, 3, 3)
        t_w: torch.Tensor,        # (B, T, 3)
        fwd_w: torch.Tensor,      # (B, T, 3) causal horizontal unit forward
        seg_len: torch.Tensor,    # (B, T) traveled distance since previous frame
    ) -> dict[str, torch.Tensor]:
        B, T = rgb.shape[:2]
        device = rgb.device

        # 1. Batch-encode all B*T RGB frames through ViT-S/14 + camera-aware neck (chunked for long eval videos)
        rgb_flat = rgb.reshape(B * T, 3, RGB_H, RGB_W)
        K_flat = K_scaled.reshape(B * T, 4)
        R_flat = R_c2w.reshape(B * T, 3, 3)
        bt = B * T
        if bt <= 512:
            f2d_flat = self.encode_images(rgb_flat, K_flat, R_flat)
        else:
            f2d_flat = torch.cat(
                [self.encode_images(rgb_flat[i : i + 256], K_flat[i : i + 256], R_flat[i : i + 256]) for i in range(0, bt, 256)],
                dim=0,
            )
        f2d_all = f2d_flat.reshape(B, T, self.c_feat, 42, 74)

        # Normalized BEV positional coordinates (2, NZ, NX)
        bev_pos = torch.stack(
            [
                self.bev_xz[..., 0] / BEV_X_MAX,
                (self.bev_xz[..., 1] - 0.5 * (BEV_Z_MIN + BEV_Z_MAX)) / (0.5 * (BEV_Z_MAX - BEV_Z_MIN)),
            ],
            dim=0,
        ).unsqueeze(0).expand(B, -1, -1, -1).to(f2d_all.dtype)  # (B, 2, NZ, NX)

        mem = torch.zeros(B, self.c_bev, BEV_NZ, BEV_NX, device=device, dtype=f2d_all.dtype)
        age = torch.full((B, 1, BEV_NZ, BEV_NX), 10.0, device=device, dtype=f2d_all.dtype)

        static_logits_seq = []
        dynamic_logits_seq = []
        corr_logits_seq = []
        unc_logits_seq = []

        # Canonical grid coordinates (M, 3) = [x_right, z_fwd, y_rel]
        gx = self.grid_xzy[:, 0][None, :, None]  # (1, M, 1)
        gz = self.grid_xzy[:, 1][None, :, None]
        gy = self.grid_xzy[:, 2][None, :, None]
        up_w = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)[None, None, :]

        for t in range(T):
            ft = f2d_all[:, t]
            Rt = R_c2w[:, t]
            tt = t_w[:, t]
            Kt = K_scaled[:, t]
            fwdt = fwd_w[:, t]
            rightt = torch.stack([fwdt[:, 1], -fwdt[:, 0], torch.zeros_like(fwdt[:, 0])], dim=-1)

            # World coordinates of all 2520 corridor hypotheses for frame t
            pts_w = (
                tt[:, None, :]
                + gx * rightt[:, None, :]
                + gz * fwdt[:, None, :]
                + gy * up_w
            )  # (B, M, 3)

            samp_t, vis_t, uv_invz_t = self._project_and_sample(ft, pts_w, Rt, tt, Kt)
            pt_parts = [samp_t, vis_t, uv_invz_t]  # (B, C+4, M)

            if self.mode in ("mvs", "full"):
                ft_norm = F.normalize(samp_t, dim=1, eps=1e-5)
                for lag in (1, 2):
                    if t >= lag:
                        tp = t - lag
                        samp_p, vis_p, _ = self._project_and_sample(
                            f2d_all[:, tp], pts_w, R_c2w[:, tp], t_w[:, tp], K_scaled[:, tp]
                        )
                        both_v = vis_t * vis_p
                        fp_norm = F.normalize(samp_p, dim=1, eps=1e-5)
                        cos_sim = (ft_norm * fp_norm).sum(dim=1, keepdim=True) * both_v
                        l1_diff = (samp_t - samp_p).abs().mean(dim=1, keepdim=True) * both_v
                        base = torch.linalg.norm(tt - t_w[:, tp], dim=-1).clamp(max=2.0)
                        base_map = base[:, None, None].expand(B, 1, pts_w.shape[1]).to(ft.dtype) * both_v
                        pt_parts.extend([cos_sim, l1_diff, both_v, base_map])
                    else:
                        z0 = torch.zeros(B, 4, pts_w.shape[1], device=device, dtype=ft.dtype)
                        pt_parts.append(z0)

            pt_feat = torch.cat(pt_parts, dim=1).reshape(B, -1, NY * BEV_NZ, BEV_NX)
            pt_red = self.pt_reduce(pt_feat).reshape(B, 32 * NY, BEV_NZ, BEV_NX)
            bev_obs = self.bev_enc(torch.cat([pt_red, bev_pos], dim=1))  # (B, c_bev, NZ, NX)

            if self.mode == "full":
                vis_bev = vis_t.reshape(B, NY, BEV_NZ, BEV_NX).mean(dim=1, keepdim=True)  # (B, 1, NZ, NX)
                if t > 0:
                    mem_w = self._warp_bev(mem, t_w[:, t - 1], fwd_w[:, t - 1], tt, fwdt)
                    age_w = self._warp_bev(age, t_w[:, t - 1], fwd_w[:, t - 1], tt, fwdt)
                    # Unmapped border pixels warp to 0 -> set high age where mem_w norm is ~0
                    ds = seg_len[:, t].clamp(0.0, 2.0)[:, None, None, None].to(ft.dtype)
                    age_w = (age_w + ds).clamp(max=10.0)
                    age_decay = torch.exp(-age_w / 4.0)
                    mem = self.mem_gru(mem_w, bev_obs, vis_bev, age_decay)
                    # Reset age to 0 wherever cell is currently visible
                    age = torch.where(vis_bev > 0.05, torch.zeros_like(age_w), age_w)
                else:
                    mem = bev_obs
                    age = torch.where(vis_bev > 0.05, torch.zeros_like(age), torch.full_like(age, 10.0))
                bev_feat = self.bev_refine(mem)
            else:
                bev_feat = self.bev_refine(bev_obs)

            static_logits_seq.append(self.head_static(bev_feat))
            dynamic_logits_seq.append(self.head_dynamic(bev_feat))

            # Central 7 columns (x indices 4..10 -> [-0.70, +0.70] m, corridor + side-wall context)
            corr_strip = bev_feat[:, :, :, 4:11].permute(0, 1, 3, 2).reshape(B, self.c_bev * 7, BEV_NZ)
            corr_out = self.corr_head(corr_strip)  # (B, 2, NZ)
            corr_logits_seq.append(corr_out[:, 0])
            unc_logits_seq.append(corr_out[:, 1])

        return {
            "static_logits": torch.stack(static_logits_seq, dim=1),   # (B, T, 3, NZ, NX)
            "dynamic_logits": torch.stack(dynamic_logits_seq, dim=1), # (B, T, 3, NZ, NX)
            "corr_logits": torch.stack(corr_logits_seq, dim=1),       # (B, T, NZ)
            "unc_logits": torch.stack(unc_logits_seq, dim=1),         # (B, T, NZ)
        }
