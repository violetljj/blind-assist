"""EgoGeom-Omni v4.1 Trainer: M5 Silicon-Exact + M1 Real-H3 Signed Correlated CNH + Indoor/Wide-FOV Scale Polish.

Runs natively with `rclone` (zero Google Drive OAuth clicks, 3-5x faster staging, works on ANY colabm account!)
or falls back to `/content/drive` if mounted.

Total budget on L4 (`violjjet` Pro+): ~2.2-2.5 hr (~4.5-5.0 CU):
  - Fast `rclone` NVMe staging (~12-15 min instead of 56 min!)
  - Phase 1 (4 Ep Warm-Start from v4 Pure-RGB, ~60 min at 54 img/s):
      * Tightens indoor/wide-FOV `med_scale` (`w_fscale=0.95` + upweighted real indoor TUM+Bonn & GlassRecon)
        while preserving `SANPO-Real ZS40 >= 80% delta1` & `p95 >= 15 m` sky horizon!
      * Evaluates Ep 0 (original v4 baseline) first so `v41_pure_rgb_best.pt` is GUARANTEED >= v4!
  - Phase 2 (5 Ep Frozen-RGB Universal ToF/CNH, ~45 min at 92 img/s):
      * Locks 100% of the 50.75M Pure-RGB trunk (`0.000000e+00` silicon zero-interference!).
      * Trains `Physical1DCNHBranchV41` on 40% EXACT Real-H3 Signed Firmware CNH (`300.28 mm/bin`,
        4.804 m span, signed reference-residual with +0.25 cross-bin correlation & 13-48% negative entries,
        50-93% distance_mm dropout, -30 deg tilt, glass foreground-rise + background-dip) +
        30% Standard Non-Negative CNH (0.25 m/bin, 74% bin0) + 30% 8x8 Sparse Depth!
      * Exports Dual M5-Fixed Static ONNX-17 (`host_focal` + `stable_softplus_htp`) to Drive via `rclone`!
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from egogeom_dataset import (
    EgoClearNPZIndex,
    GlassReconIndex,
    HypersimTrajIndex,
    M3EDCompactIndex,
    MultiDomainEgoGeomDataset,
    TUMBonnIndex,
    stage_glassrecon_local,
    stage_hypersim_fast,
    stage_m3ed_fast,
    stage_tar_dir,
    stage_tum_bonn_local,
)
from egogeom_losses_and_physics import (
    compute_gravity_aligned_ray_map,
    compute_local_ground_height_residual,
    rotation_cam_to_gravity_level,
    simulate_tof_and_cnh_prompt,
)
from egogeom_omni_v41_model import (
    EgoGeomOmniV41ExportWrapper,
    EgoGeomOmniV41Net,
    EgoGeomOmniV41PureRGBExportWrapper,
    EgoGeomV41Config,
    batch_host_focal_from_K,
)
from egogeom_v4_loss import EgoGeomV4Loss

LOCAL_RCLONE_SRC = Path("/content/rclone_datasets")
LOCAL_STAGE = Path("/content/egogeom_stage_data_v41")
LOCAL_RUNS = Path("/content/egogeom_runs_v41")
RCLONE_REMOTE_DATA = "gdrive:BlindAssist/datasets"
RCLONE_REMOTE_V4 = "gdrive:BlindAssist/runs/egogeom_omni_v4"
RCLONE_REMOTE_OUT = "gdrive:BlindAssist/runs/egogeom_omni_v41"

# Exact Cholesky factor of 16x16 cross-bin positive correlation matrix (rho=0.25, calibrated to real H3 Rs=0.247!)
_CORR16 = 0.75 * np.eye(16, dtype=np.float32) + 0.25 * np.ones((16, 16), dtype=np.float32)
_CHOL16_NP = np.linalg.cholesky(_CORR16).astype(np.float32)
_CHOL16_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_chol16(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _CHOL16_CACHE:
        _CHOL16_CACHE[key] = torch.from_numpy(_CHOL16_NP).to(device=device, dtype=dtype)
    return _CHOL16_CACHE[key]


@torch.no_grad()
def simulate_signed_h3_cnh_prompt(
    depth_m: torch.Tensor,
    valid_mask: torch.Tensor,
    K: torch.Tensor,
    glass_mask: torch.Tensor | None = None,
    extreme_dropout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Real-H3 Signed Firmware CNH Simulator calibrated to 1,209 real ST VL53L8CH frames (M1 SIGNED_H3.md).

    Matches real hardware properties:
      - 16 bins x 300.2784 mm/bin (4.804 m span, +/-2% calibration jitter)
      - Wide camera-ToF mounting tilt (yaw +/-20 deg, pitch -35..+15 deg covering -30 deg tilted H3!)
      - Cover-glass bin0 pedestal spanning 15% (close cabinet) to 82% (open corridor)
      - Signed background/reference-residual subtraction with cross-bin correlated Gaussian noise (rho=0.25)
        yielding realistic 13-48% negative tail bins!
      - Glass foreground-rise + background-attenuation paired signature!
      - Real H3 distance_mm dropout (50-92% missing zones in bright/open corridors)!
    """
    depth_m = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 20.0)
    B, _, H, W = depth_m.shape
    H4, W4 = H // 4, W // 4
    device, dtype = depth_m.device, depth_m.dtype
    prompt = torch.zeros((B, 20, H4, W4), device=device, dtype=dtype)
    has_tof = torch.ones((B, 1, 1, 1), device=device, dtype=dtype)

    GH, GW = 48, 48
    fov_scale = 0.48 + 0.34 * torch.rand((B, 2), device=device, dtype=dtype)
    shift_x = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 0.56
    shift_y = (torch.rand(B, device=device, dtype=dtype) - 0.30) * 0.62

    theta = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = fov_scale[:, 0]
    theta[:, 1, 1] = fov_scale[:, 1]
    theta[:, 0, 2] = shift_x
    theta[:, 1, 2] = shift_y
    grid = F.affine_grid(theta, size=(B, 1, GH, GW), align_corners=False)

    z_grid = F.grid_sample(depth_m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    v_grid = F.grid_sample(valid_mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    g_grid = (
        F.grid_sample(glass_mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        if glass_mask is not None
        else torch.zeros_like(z_grid)
    )

    z_z = z_grid.view(B, 8, 6, 8, 6).permute(0, 1, 3, 2, 4).reshape(B, 64, 36)
    v_z = (v_grid.view(B, 8, 6, 8, 6).permute(0, 1, 3, 2, 4).reshape(B, 64, 36) > 0.5) & (z_z > 0.12) & (z_z < 4.75)
    g_z = (g_grid.view(B, 8, 6, 8, 6).permute(0, 1, 3, 2, 4).reshape(B, 64, 36) > 0.5).any(dim=-1)

    n_bins = 16
    # Exact H3 firmware coarse bin width: 8 * 37.5348 mm = 300.2784 mm (+/-2% jitter)
    bin_m = (0.3002784 * (0.98 + 0.04 * torch.rand((B, 1, 1), device=device, dtype=dtype)))
    # Soft bilinear/IRF spread across adjacent coarse bins (matches real H3 IRF width ~0.10m)
    bin_f = (z_z / bin_m).clamp(0.0, n_bins - 1.001)
    b0 = bin_f.long()
    b1 = (b0 + 1).clamp_max(n_bins - 1)
    frac = bin_f - b0.to(dtype)
    w_base = torch.where(v_z, 1.0 / torch.clamp(z_z, min=0.25).pow(2), torch.zeros_like(z_z))

    hist = torch.zeros((B, 64, n_bins), device=device, dtype=dtype)
    hist.scatter_add_(2, b0, w_base * (1.0 - frac))
    hist.scatter_add_(2, b1, w_base * frac)

    # Real glass dual-surface physics: attenuated foreground glass echo + stronger transmitted background peak
    dom_bin = hist.argmax(dim=-1)
    bg_bin = torch.clamp(dom_bin + torch.randint(1, 5, (B, 64), device=device), max=n_bins - 1)
    fg_peak = hist.max(dim=-1).values
    bg_boost = torch.where(g_z, (0.40 + 0.50 * torch.rand((B, 64), device=device, dtype=dtype)) * fg_peak, torch.zeros_like(fg_peak))
    hist.scatter_add_(2, bg_bin.unsqueeze(-1), bg_boost.unsqueeze(-1))

    # Normalize positive echo energy
    zone_has_sig = (hist.sum(dim=-1, keepdim=True) > 1e-6)
    hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)

    # Blend cover-glass bin0/bin1 pedestal (15% close target -> 82% open bright corridor!)
    xtalk_frac = torch.where(
        zone_has_sig,
        0.15 + 0.67 * torch.rand((B, 64, 1), device=device, dtype=dtype),
        0.85 + 0.12 * torch.rand((B, 64, 1), device=device, dtype=dtype),
    )
    ped = torch.zeros_like(hist)
    ped[:, :, 0:1] = 0.93
    ped[:, :, 1:2] = 0.07
    hist = (1.0 - xtalk_frac) * hist + xtalk_frac * ped

    # Inject signed correlated zero-mean residual noise (rho=0.25 Cholesky -> 13-48% negative bins just like real H3!)
    chol = _get_chol16(device, dtype)
    noise_std = (0.015 + 0.055 * torch.rand((B, 64, 1), device=device, dtype=dtype))
    eps = torch.randn((B, 64, 16), device=device, dtype=dtype) @ chol.T
    # Small signed DC residual drift (emulates imperfect firmware ambient/reference subtraction)
    dc_shift = (torch.rand((B, 64, 1), device=device, dtype=dtype) - 0.55) * 0.025
    hist_signed = hist + noise_std * eps + dc_shift
    # Scale-invariant L1-abs normalization (matches real H3 arbitrary signal units AND Std-CNH dynamic range!)
    hist_signed = hist_signed / (hist_signed.abs().sum(dim=-1, keepdim=True) + 1e-6)

    # Real H3 distance_mm dropout (50-92% missing zones in bright/open corridors!)
    z_clean = torch.where(v_z, z_z, torch.full_like(z_z, 99.0))
    z_min_zone = z_clean.min(dim=-1).values
    valid_zone = z_min_zone < 4.75
    if extreme_dropout:
        keep_prob = 0.08 + 0.45 * torch.rand((B, 1), device=device, dtype=dtype)  # 8%..53% valid!
    else:
        keep_prob = torch.full((B, 1), 0.50, device=device, dtype=dtype)
    drop_dist = torch.rand((B, 64), device=device, dtype=dtype) > keep_prob
    valid_dist = valid_zone & (~drop_dist)

    # Splat back to camera stride-4 canvas using EXACT unified 20-channel contract:
    #   ch 0: log1p(z_est) * d_m, ch 1: d_m, ch 2..17: signed log1p(10*hist_signed), ch 18: c_m, ch 19: has_tof!
    z_est = (z_min_zone + torch.randn_like(z_min_zone) * (0.015 + 0.02 * z_min_zone)).clamp(0.12, 4.75)
    zv_f = valid_dist.to(dtype)
    cnh_signed_log = torch.sign(hist_signed) * torch.log1p(torch.abs(hist_signed) * 10.0)

    grid_8 = torch.zeros((B, 20, 8, 8), device=device, dtype=dtype)
    grid_8[:, 0] = (torch.log1p(z_est) * zv_f).view(B, 8, 8)
    grid_8[:, 1] = zv_f.view(B, 8, 8)
    grid_8[:, 2:18] = cnh_signed_log.permute(0, 2, 1).view(B, 16, 8, 8)
    grid_8[:, 18] = 1.0  # CNH waveform always present even when distance_mm drops out!
    grid_8[:, 19] = 1.0

    # Project 8x8 ToF FOV back onto stride-4 camera grid via inverse affine transform
    inv_theta = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    inv_theta[:, 0, 0] = 1.0 / fov_scale[:, 0].clamp_min(0.2)
    inv_theta[:, 1, 1] = 1.0 / fov_scale[:, 1].clamp_min(0.2)
    inv_theta[:, 0, 2] = -shift_x / fov_scale[:, 0].clamp_min(0.2)
    inv_theta[:, 1, 2] = -shift_y / fov_scale[:, 1].clamp_min(0.2)
    inv_grid = F.affine_grid(inv_theta, size=(B, 20, H4, W4), align_corners=False)
    prompt = F.grid_sample(grid_8, inv_grid, mode="nearest", padding_mode="zeros", align_corners=False)
    prompt[:, 19:20] = 1.0
    return prompt, has_tof


@torch.no_grad()
def simulate_mixed_v41_prompt(
    depth_m: torch.Tensor,
    valid_mask: torch.Tensor,
    K: torch.Tensor,
    glass_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mix inside every batch: 40% Real-H3 Signed CNH + 30% Std CNH + 30% 8x8 Depth."""
    B = depth_m.shape[0]
    p_h3, ht_h3 = simulate_signed_h3_cnh_prompt(depth_m, valid_mask, K, glass_mask=glass_mask, extreme_dropout=True)
    p_std, ht_std = simulate_tof_and_cnh_prompt(depth_m, valid_mask, K, glass_mask=glass_mask, mode="cnh", realistic_hardware=True)
    p_dep, ht_dep = simulate_tof_and_cnh_prompt(depth_m, valid_mask, K, glass_mask=glass_mask, mode="depth", realistic_hardware=False)

    r = torch.rand((B, 1, 1, 1), device=depth_m.device)
    prompt = torch.where(r < 0.40, p_h3, torch.where(r < 0.70, p_std, p_dep))
    has_tof = torch.ones((B, 1, 1, 1), device=depth_m.device, dtype=depth_m.dtype)
    return prompt, has_tof


def _rclone_pull_one(sd: str, src_root: Path, max_tars: int | None = None) -> Path:
    dst = src_root / sd
    marker = dst / ".rclone_done"
    if marker.exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    if max_tars is not None:
        out = subprocess.check_output(["rclone", "lsf", f"{RCLONE_REMOTE_DATA}/{sd}"], text=True)
        tars = sorted([x.strip() for x in out.splitlines() if x.strip().endswith(".tar")])[:max_tars]
        inc_file = dst / "_include.txt"
        inc_file.write_text("\n".join(tars) + "\n", encoding="utf-8")
        print(f"  [rclone] pulling {sd} ({len(tars)} shards)...", flush=True)
        subprocess.run(
            ["rclone", "copy", f"{RCLONE_REMOTE_DATA}/{sd}", str(dst), "--files-from", str(inc_file), "--transfers", str(min(16, len(tars)))],
            check=True,
        )
    else:
        cmd = ["rclone", "copy", f"{RCLONE_REMOTE_DATA}/{sd}", str(dst), "--transfers", "16", "--checkers", "16"]
        if sd == "bonn_dynamic_v1":
            cmd += ["--exclude", "*dataset.zip*", "--exclude", "*groundtruth*"]
        print(f"  [rclone] pulling {sd}...", flush=True)
        subprocess.run(cmd, check=True)
    marker.write_text("ok", encoding="utf-8")
    return dst


def rclone_push_file(local_path: Path, remote_dir: str = RCLONE_REMOTE_OUT) -> None:
    subprocess.run(["rclone", "copy", str(local_path), remote_dir, "--transfers", "4"], check=True)


def build_v41_datasets(data_root: Path, out_hw: tuple[int, int], epoch_samples: int):
    """Disk-safe pull -> unpack -> purge-raw NVMe staging (keeps peak usage <120 GB on 236 GB L4 VM)."""
    import shutil

    t0 = time.time()
    n_cpu = max(8, os.cpu_count() or 8)
    print(f"=== Disk-Safe Pull/Unpack NVMe datasets ({n_cpu} vCPUs) ===", flush=True)

    def _stage_tar_clean(raw_sd: str, stage_name: str, max_shards: int | None = None, workers: int = 8) -> list[Path]:
        ldir = LOCAL_STAGE / stage_name
        done = ldir / ".stage_done"
        if done.exists():
            return sorted(ldir.glob("*.npz"))
        rdir = _rclone_pull_one(raw_sd, data_root, max_tars=max_shards)
        res = stage_tar_dir(rdir, ldir, max_shards=max_shards, workers=workers)
        done.write_text("ok", encoding="utf-8")
        shutil.rmtree(rdir, ignore_errors=True)
        return res

    sanpo_real_npzs = (
        _stage_tar_clean("sanpo_real_train76_v0", "sanpo_real_train76", workers=8)
        + _stage_tar_clean("sanpo_real_chest240_v0", "sanpo_real_chest240", workers=8)
        + _stage_tar_clean("sanpo_real_head100_autoseg_v0", "sanpo_real_head100", workers=8)
    )
    sanpo_synth_npzs = (
        _stage_tar_clean("sanpo_synth_287_v0", "sanpo_synth_287", workers=8)
        + _stage_tar_clean("sanpo_synth_extra450_v0", "sanpo_synth_extra450", max_shards=14, workers=8)
    )
    tartan_npzs = _stage_tar_clean("tartanair_widefov_p13_v0", "tartanair", workers=8)

    m3_dir = LOCAL_STAGE / "m3ed_spot_fast"
    if (m3_dir / ".stage_done").exists():
        m3ed_npzs = sorted(m3_dir.glob("*.npz"))
    else:
        r_m3 = _rclone_pull_one("m3ed_spot_train_v1", data_root)
        m3ed_npzs = stage_m3ed_fast(r_m3, m3_dir, target_hw=out_hw, workers=8)
        (m3_dir / ".stage_done").write_text("ok", encoding="utf-8")
        shutil.rmtree(r_m3, ignore_errors=True)

    g_dir = LOCAL_STAGE / "glassrecon"
    if (g_dir / ".stage_done").exists():
        glass_dir = g_dir
    else:
        r_g = _rclone_pull_one("glassrecon_v1", data_root)
        glass_dir = stage_glassrecon_local(r_g, g_dir)
        (g_dir / ".stage_done").write_text("ok", encoding="utf-8")
        shutil.rmtree(r_g, ignore_errors=True)

    tb_dir = LOCAL_STAGE / "tum_bonn"
    if (tb_dir / ".stage_done").exists():
        tb_dirs = sorted({p.parent for p in tb_dir.rglob("rgb.txt")})
    else:
        r_t = _rclone_pull_one("tum_rgbd_v1", data_root)
        r_b = _rclone_pull_one("bonn_dynamic_v1", data_root)
        tb_dirs = stage_tum_bonn_local(r_t, r_b, tb_dir, max_archives=28, workers=8)
        (tb_dir / ".stage_done").write_text("ok", encoding="utf-8")
        shutil.rmtree(r_t, ignore_errors=True)
        shutil.rmtree(r_b, ignore_errors=True)

    sanpo_zs40_npzs = _stage_tar_clean("sanpo_real_zeroshot_40_v0", "sanpo_real_zs40", workers=4)
    print(f"  [stage] 6 datasets ready ({time.time()-t0:.0f}s), staging Hypersim...", flush=True)

    hyp_dir = LOCAL_STAGE / "hypersim_fast_v2"
    if (hyp_dir / ".stage_done").exists():
        hyp_npzs = sorted(hyp_dir.glob("*.npz"))
    else:
        r_h = _rclone_pull_one("hypersim_full_v1", data_root)
        hyp_npzs = stage_hypersim_fast(r_h, hyp_dir, target_hw=out_hw, workers=min(12, n_cpu))
        (hyp_dir / ".stage_done").write_text("ok", encoding="utf-8")
        shutil.rmtree(r_h, ignore_errors=True)
    print(f"=== All 7 NVMe datasets ready in {time.time()-t0:.1f}s (Hypersim={len(hyp_npzs)}) ===", flush=True)

    hyp_train = HypersimTrajIndex(hyp_npzs, split_filter="train")
    hyp_val = HypersimTrajIndex(hyp_npzs, split_filter="val")
    sr_train = EgoClearNPZIndex(sanpo_real_npzs, "sanpo_real", stride=1, allow_teacher_fill=False)
    ss_train = EgoClearNPZIndex(sanpo_synth_npzs, "sanpo_synth", stride=2, allow_teacher_fill=False)
    ta_train = EgoClearNPZIndex(tartan_npzs, "tartanair", stride=1, allow_teacher_fill=False)
    m3_train = M3EDCompactIndex(m3ed_npzs, stride=1)
    gls_train = GlassReconIndex(glass_dir, split_filter="train")
    gls_val = GlassReconIndex(glass_dir, split_filter="val")
    tb_train = TUMBonnIndex(tb_dirs, stride=2)
    sr_zs40 = EgoClearNPZIndex(sanpo_zs40_npzs, "sanpo_real_zs40", stride=4, allow_teacher_fill=False)

    # Indoor-scale-reinforced balanced mix (50% Indoor / 50% Outdoor, stronger real handheld TUM+Bonn & GlassRecon!)
    train_ds = MultiDomainEgoGeomDataset(
        sources=[
            (hyp_train, 0.22),
            (tb_train, 0.16),
            (gls_train, 0.12),
            (sr_train, 0.24),
            (ss_train, 0.12),
            (ta_train, 0.07),
            (m3_train, 0.07),
        ],
        out_hw=out_hw,
        epoch_samples=epoch_samples,
        is_train=True,
    )
    eval_sets = {
        "sanpo_real_zs40": MultiDomainEgoGeomDataset([(sr_zs40, 1.0)], out_hw=out_hw, is_train=False),
        "hypersim_val": MultiDomainEgoGeomDataset([(hyp_val, 1.0)], out_hw=out_hw, is_train=False),
        "glassrecon_val": MultiDomainEgoGeomDataset([(gls_val, 1.0)], out_hw=out_hw, is_train=False),
    }
    return train_ds, eval_sets


class SkyGuardMoGeS:
    """Fast MoGe-2-ViT-S upper-frame sky-mask guard on SANPO-Real (~35 ms/batch)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.moge_s = None
        self.is_logits_s = False
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        try:
            from moge.model.v2 import MoGeModel
            ms = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval().to(device)
            for p in ms.parameters():
                p.requires_grad = False
            self.moge_s = ms
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pr = ms.forward(torch.rand(1, 3, 168, 224, device=device), num_tokens=224)["mask"].float()
            self.is_logits_s = bool(pr.min().item() < -0.05 or pr.max().item() > 1.05)
            print(f"[SkyGuard v4.1] MoGe-2-ViT-S ready (is_logits={self.is_logits_s})", flush=True)
        except Exception as e:
            print(f"[SkyGuard v4.1 WARN] MoGe unavailable ({e})", flush=True)

    @torch.no_grad()
    def get_sky_mask(self, rgb_norm: torch.Tensor, valid_mask: torch.Tensor, sources: list[str]) -> torch.Tensor | None:
        if self.moge_s is None:
            return None
        sr_idx = [i for i, s in enumerate(sources) if s == "sanpo_real"]
        if not sr_idx:
            return None
        B, _, H, W = rgb_norm.shape
        idx_t = torch.tensor(sr_idx, device=self.device, dtype=torch.long)
        rgb01 = (rgb_norm.index_select(0, idx_t).float() * self.im_std + self.im_mean).clamp(0.0, 1.0)
        rgb01_s = F.interpolate(rgb01, size=(168, 224), mode="bilinear", align_corners=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            mk = self.moge_s.forward(rgb01_s, num_tokens=224)["mask"].float().unsqueeze(1)
        prob = torch.sigmoid(mk) if self.is_logits_s else mk
        prob_hw = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        upper = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.bool)
        upper[:, :, : int(H * 0.65), :] = True
        sky_mask = torch.zeros((B, 1, H, W), device=self.device, dtype=torch.float32)
        sky_sr = ((prob_hw < 0.25) & (valid_mask.index_select(0, idx_t) < 0.5) & upper).float()
        sky_mask.index_copy_(0, idx_t, sky_sr)
        return sky_mask


@torch.no_grad()
def evaluate_v41_domain(
    model: EgoGeomOmniV41Net,
    loader: DataLoader,
    device: torch.device,
    tof_mode: str = "none",
    dark_scale: float = 1.0,
    realistic_hardware: bool = False,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    absrel_sum, sq_sum, d1_sum, dh_mae_sum, edge_err_sum, glass_absrel_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    far_absrel_sum, n_far = 0.0, 0.0
    n_pix, n_dh, n_edge, n_glass = 0.0, 0.0, 0.0, 0.0
    fr_ratios, ssi_errs, p95_list = [], [], []
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        rgb = batch["rgb"].to(device, non_blocking=True)
        if dark_scale < 0.99:
            rgb01 = (rgb * im_std + im_mean).clamp(0.0, 1.0) * dark_scale
            rgb = (rgb01 - im_mean) / im_std
        K = batch["K"].to(device, non_blocking=True)
        g_cam = batch["g_cam"].to(device, non_blocking=True)
        gt_d = batch["depth_m"].to(device, non_blocking=True)
        vm = batch["valid_mask"].to(device, non_blocking=True)
        gls = batch["glass_mask"].to(device, non_blocking=True)
        H, W = rgb.shape[-2:]

        rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
        hf = batch_host_focal_from_K(K, W, H, model.cfg.canonical_focal_px)
        gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if tof_mode == "none":
                out = model(rgb, rays, None, None, host_focal=hf)
            elif tof_mode == "signed_h3":
                tof_p, has_tof = simulate_signed_h3_cnh_prompt(gt_d, vm, K, glass_mask=gls, extreme_dropout=True)
                out = model(rgb, rays, tof_p, has_tof, host_focal=hf)
            else:
                tof_p, has_tof = simulate_tof_and_cnh_prompt(
                    gt_d, vm, K, glass_mask=gls, mode=tof_mode, realistic_hardware=realistic_hardware
                )
                out = model(rgb, rays, tof_p, has_tof, host_focal=hf)

        pred_d = out["depth_m"].float()
        pred_dh = out["delta_h_m"].float()
        for b in range(pred_d.shape[0]):
            p95_list.append(float(torch.quantile(pred_d[b, 0].reshape(-1), 0.95).item()))

        m = (vm > 0.5) & (gt_d >= 0.30) & (gt_d <= 4.0)
        if m.any():
            pd, gd = pred_d[m], gt_d[m]
            absrel_sum += ((pd - gd).abs() / gd).sum().item()
            sq_sum += ((pd - gd) ** 2).sum().item()
            d1_sum += (torch.maximum(pd / gd, gd / pd) < 1.25).float().sum().item()
            n_pix += float(m.sum().item())
            for b in range(pred_d.shape[0]):
                mb = m[b, 0]
                if mb.sum() > 64:
                    pb, gb = pred_d[b, 0][mb], gt_d[b, 0][mb]
                    r = float(torch.median(pb / gb).item())
                    fr_ratios.append(r)
                    ssi_errs.append(float(((pb / max(r, 1e-4) - gb).abs() / gb).mean().item()))

        m_far = (vm > 0.5) & (gt_d > 4.0) & (gt_d <= 12.0)
        if m_far.any():
            pf, gf = pred_d[m_far], gt_d[m_far]
            far_absrel_sum += ((pf - gf).abs() / gf).sum().item()
            n_far += float(m_far.sum().item())

        m_dh = (dh_m > 0.5) & (gt_dh.abs() > 0.04) & (gt_dh.abs() < 0.35) & m
        if m_dh.any():
            dh_mae_sum += (pred_dh[m_dh] - gt_dh[m_dh]).abs().sum().item()
            n_dh += float(m_dh.sum().item())

        gt_log = torch.log(gt_d.clamp(0.15, 12.0))
        dx = (gt_log[..., :, 1:] - gt_log[..., :, :-1]).abs()
        edge_m = F.pad((dx > 0.15), (0, 1)) & m
        if edge_m.any():
            edge_err_sum += ((pred_d[edge_m] - gt_d[edge_m]).abs() / gt_d[edge_m]).sum().item()
            n_edge += float(edge_m.sum().item())

        m_gls = (gls > 0.5) & m
        if m_gls.any():
            glass_absrel_sum += ((pred_d[m_gls] - gt_d[m_gls]).abs() / gt_d[m_gls]).sum().item()
            n_glass += float(m_gls.sum().item())

    return {
        "near_absrel": round(absrel_sum / max(n_pix, 1.0), 4),
        "near_rmse_m": round(math.sqrt(sq_sum / max(n_pix, 1.0)), 4),
        "near_delta1": round(d1_sum / max(n_pix, 1.0), 4),
        "far_absrel": round(far_absrel_sum / max(n_far, 1.0), 4) if n_far > 0 else -1.0,
        "p95_m": round(float(np.median(p95_list)), 2) if p95_list else -1.0,
        "med_scale": round(float(np.median(fr_ratios)), 3) if fr_ratios else -1.0,
        "ssi_absrel": round(float(np.mean(ssi_errs)), 4) if ssi_errs else -1.0,
        "shallow_dh_mae_cm": round(100.0 * dh_mae_sum / max(n_dh, 1.0), 2),
        "thin_edge_absrel": round(edge_err_sum / max(n_edge, 1.0), 4),
        "glass_absrel": round(glass_absrel_sum / max(n_glass, 1.0), 4) if n_glass > 0 else -1.0,
    }


def _export_v41_onnx(ema_model: EgoGeomOmniV41Net, H: int, W: int, pure_name: str, omni_name: str | None = None, push_remote: bool = True) -> None:
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    cpu_copy = copy.deepcopy(ema_model).cpu().eval()
    d_rgb = torch.randn(1, 3, H, W)
    d_rays = torch.randn(1, 3, H, W)
    d_hf = torch.tensor([[[[1.50]]]], dtype=torch.float32)

    p_rgb = LOCAL_RUNS / pure_name
    torch.onnx.export(
        EgoGeomOmniV41PureRGBExportWrapper(copy.deepcopy(cpu_copy)).eval(),
        (d_rgb, d_rays, d_hf),
        str(p_rgb),
        opset_version=17,
        input_names=["rgb", "grav_rays", "host_focal"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
        do_constant_folding=True,
        dynamo=False,
    )
    if push_remote:
        rclone_push_file(p_rgb)

    if omni_name is not None:
        d_tof = torch.zeros(1, 20, H // 4, W // 4)
        d_ht = torch.ones(1, 1, 1, 1)
        p_om = LOCAL_RUNS / omni_name
        torch.onnx.export(
            EgoGeomOmniV41ExportWrapper(cpu_copy).eval(),
            (d_rgb, d_rays, d_tof, d_ht, d_hf),
            str(p_om),
            opset_version=17,
            input_names=["rgb", "grav_rays", "tof_grid_s4", "has_tof", "host_focal"],
            output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
            do_constant_folding=True,
            dynamo=False,
        )
        if push_remote:
            rclone_push_file(p_om)


def _eval_p1(model: EgoGeomOmniV41Net, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_v41_domain(model, loader, device, tof_mode="none")
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__no_tof"] = evaluate_v41_domain(model, loader, device, tof_mode="none", dark_scale=0.18)
    s = res["sanpo_real_zs40__no_tof"]
    h = res["hypersim_val__no_tof"]
    g = res["glassrecon_val__no_tof"]
    sf = s["far_absrel"] if s["far_absrel"] > 0 else 0.30
    scale_pen = 0.08 * (abs(s["med_scale"] - 1.0) + abs(h["med_scale"] - 1.0) + abs(g["med_scale"] - 1.0))
    comp = (
        0.24 * s["near_absrel"]
        + 0.14 * sf
        + 0.18 * (1.0 - s["near_delta1"])
        + 0.18 * g["near_absrel"]
        + 0.26 * (1.0 - h["near_delta1"])
        + scale_pen
    )
    return res, round(comp, 4)


def _eval_p2(model: EgoGeomOmniV41Net, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_v41_domain(model, loader, device, tof_mode="none")
        res[f"{name}__with_signed_h3"] = evaluate_v41_domain(model, loader, device, tof_mode="signed_h3")
        res[f"{name}__with_cnh"] = evaluate_v41_domain(model, loader, device, tof_mode="cnh", realistic_hardware=False)
        res[f"{name}__with_cnh_real_tilt"] = evaluate_v41_domain(model, loader, device, tof_mode="cnh", realistic_hardware=True)
        res[f"{name}__with_depth"] = evaluate_v41_domain(model, loader, device, tof_mode="depth", realistic_hardware=False)
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__with_signed_h3"] = evaluate_v41_domain(model, loader, device, tof_mode="signed_h3", dark_scale=0.18)
            res[f"{name}_dark018__with_cnh"] = evaluate_v41_domain(model, loader, device, tof_mode="cnh", dark_scale=0.18)
    comp = (
        0.25 * res["sanpo_real_zs40__with_signed_h3"]["near_absrel"]
        + 0.20 * res["sanpo_real_zs40__with_cnh"]["near_absrel"]
        + 0.20 * res["hypersim_val__with_signed_h3"]["near_absrel"]
        + 0.15 * res["glassrecon_val__with_signed_h3"]["near_absrel"]
        + 0.10 * res["glassrecon_val__with_cnh"]["near_absrel"]
        + 0.10 * res["sanpo_real_zs40__with_depth"]["near_absrel"]
    )
    return res, round(comp, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1-epochs", type=int, default=4)
    p.add_argument("--p2-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epoch-samples", type=int, default=48000)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--smoke-synthetic", action="store_true")
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EgoGeomV41Config.small_dinov3()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if not args.smoke_synthetic and vram_gb >= 85.0:
            args.batch_size = 64
        print(
            f"=== EgoGeom-Omni v4.1 Silicon-Exact + Signed-H3 ({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB) | "
            f"bs={args.batch_size} | P1={args.p1_epochs}ep | P2={args.p2_epochs}ep ===",
            flush=True,
        )

    if args.smoke_synthetic:
        m = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device)
        crit = EgoGeomV4Loss(w_fscale=0.95)
        B, H, W = 2, args.height, args.width
        rgb = torch.randn(B, 3, H, W, device=device)
        K = torch.tensor([[[240.0, 0.0, W / 2], [0.0, 240.0, H / 2], [0.0, 0.0, 1.0]]] * B, device=device)
        g = torch.tensor([[0.0, 0.98, 0.20]] * B, device=device)
        gt_d = torch.rand(B, 1, H, W, device=device) * 3.5 + 0.4
        gt_d[:, :, : H // 3, :] = 60.0
        vm = torch.ones(B, 1, H, W, device=device)
        sky = torch.zeros(B, 1, H, W, device=device)
        sky[:, :, : H // 4, :] = 1.0
        rays = compute_gravity_aligned_ray_map(K, g, H, W)
        hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
        tof, ht = simulate_signed_h3_cnh_prompt(gt_d, vm, K, extreme_dropout=True)
        neg_frac = float((tof[:, 2:18] < 0).float().mean().item()) * 100.0
        o1 = m(rgb, rays, tof, ht, host_focal=hf)
        loss, logs = crit(o1, gt_d, vm, sky_mask=sky)
        loss.backward()
        assert torch.isfinite(loss) and logs["l_sky"] > 0.0 and neg_frac > 5.0
        m.eval()
        with torch.no_grad():
            o_rgb = m(rgb, rays, None, None, host_focal=hf)
            o_zero = m(rgb, rays, torch.zeros_like(tof), torch.zeros_like(ht), host_focal=hf)
            zi = float((o_rgb["depth_m"] - o_zero["depth_m"]).abs().max().item())
        assert zi == 0.0
        n_all = sum(p.numel() for p in m.parameters()) / 1e6
        n_tof = sum(p.numel() for p in m.tof_branch.parameters()) / 1e6
        print(
            f"SMOKE v4.1 PASSED! params={n_all:.2f}M (ToF={n_tof:.2f}M) loss={loss.item():.4f} "
            f"l_sky={logs['l_sky']:.4f} signed_h3_neg={neg_frac:.1f}% zero_interference={zi:.2e}",
            flush=True,
        )
        return

    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    # 1. Pull v4 pure_rgb warm-start checkpoint + datasets via fast 16-thread rclone!
    v4_warm_local = LOCAL_RUNS / "egogeom_omni_v4_pure_rgb_best.pt"
    if not v4_warm_local.exists():
        print("=== Pulling v4 Pure-RGB warm-start checkpoint via rclone ===", flush=True)
        subprocess.run(["rclone", "copy", f"{RCLONE_REMOTE_V4}/egogeom_omni_v4_pure_rgb_best.pt", str(LOCAL_RUNS)], check=True)

    train_ds, eval_sets = build_v41_datasets(LOCAL_RCLONE_SRC, (args.height, args.width), args.epoch_samples)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=3 if args.num_workers > 0 else None,
    )
    eval_loaders = {
        k: DataLoader(v, batch_size=max(16, args.batch_size // 2), shuffle=False, num_workers=4, pin_memory=True)
        for k, v in eval_sets.items()
    }

    hist_path = LOCAL_RUNS / "v41_history.json"
    history: list[dict] = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    p1_done_flag = LOCAL_RUNS / "v41_p1_done.flag"
    p1_best_path = LOCAL_RUNS / "egogeom_omni_v41_pure_rgb_best.pt"
    p2_best_path = LOCAL_RUNS / "egogeom_omni_v41_unified_best.pt"

    # =========================================================================
    # PHASE 1: Surgical Indoor / Wide-FOV Scale Polish (Warm-Start from v4!)
    # =========================================================================
    if not p1_done_flag.exists():
        print(f"\n================ v4.1 PHASE 1: Warm-Start Scale & Indoor Polish ({args.p1_epochs} Ep) ================", flush=True)
        model = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device).to(memory_format=torch.channels_last)
        ck_v4 = torch.load(v4_warm_local, map_location=device, weights_only=False)
        model.load_v4_pure_rgb_trunk(ck_v4["ema"])
        ema_model = copy.deepcopy(model).eval()
        for pe in ema_model.parameters():
            pe.requires_grad = False
        if model.tof_branch is not None:
            for pt in model.tof_branch.parameters():
                pt.requires_grad = False
            model.tof_branch.eval()

        # Baseline Ep 0 eval (exact v4 trunk with M5 stable Softplus + analytical host_focal!)
        ep0_eval, best_p1_score = _eval_p1(ema_model, eval_loaders, device)
        print(f"=== [v4.1 Phase 1 Baseline Ep 0 (v4+M5)] composite={best_p1_score:.4f} ===", flush=True)
        for k, v in ep0_eval.items():
            print(f"  {k}: {v}", flush=True)
        history.append({"phase": "v41_P1_ep0_baseline", "epoch": 0, "composite": best_p1_score, "eval": ep0_eval})
        torch.save({"epoch": -1, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": best_p1_score, "eval": ep0_eval}, p1_best_path)
        _export_v41_onnx(ema_model, args.height, args.width, "egogeom_omni_v41_best_pure_rgb_fp32.onnx", push_remote=True)

        enc_params, dec_params = [], []
        for n, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if n.startswith("enc."):
                enc_params.append(param)
            else:
                dec_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": 1.5e-5, "weight_decay": 1e-3},
                {"params": dec_params, "lr": 1.5e-4, "weight_decay": 1e-4},
            ],
            fused=True,
        )
        total_steps = args.p1_epochs * len(train_loader)
        warmup_steps = 150

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])
        sky_guard = SkyGuardMoGeS(device)
        criterion = EgoGeomV4Loss(w_fscale=0.95, w_td_nrm=0.0)
        train_p1 = torch.compile(model, mode="default")
        all_rgb_params = enc_params + dec_params

        for epoch in range(args.p1_epochs):
            t_ep = time.time()
            train_p1.train()
            if model.tof_branch is not None:
                model.tof_branch.eval()
            running: dict[str, float] = {}
            n_steps = 0

            for step, batch in enumerate(train_loader, 1):
                rgb = batch["rgb"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
                K = batch["K"].to(device, non_blocking=True)
                g_cam = batch["g_cam"].to(device, non_blocking=True)
                gt_d = batch["depth_m"].to(device, non_blocking=True)
                vm = batch["valid_mask"].to(device, non_blocking=True)
                gls = batch["glass_mask"].to(device, non_blocking=True)
                nrm_cam = batch["normal_cam"].to(device, non_blocking=True)
                nrm_m = batch["normal_mask"].to(device, non_blocking=True)
                sources = list(batch["source"])
                H, W = rgb.shape[-2:]

                rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
                hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
                gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
                dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
                R_lvl = rotation_cam_to_gravity_level(g_cam)
                gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)
                sky_m = sky_guard.get_sky_mask(rgb, vm, sources)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_rgb = train_p1(rgb, rays, None, None, host_focal=hf)
                    loss, logs = criterion(out_rgb, gt_d, vm, gt_dh, dh_m, gt_nrm_lvl, nrm_m, gls, sky_mask=sky_m)

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_rgb_params, 2.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.data.mul_(0.996).add_(pm.data, alpha=0.004)
                    for be, bm in zip(ema_model.buffers(), model.buffers()):
                        be.data.copy_(bm.data)

                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                if step in (1, 25, 100) or step % 250 == 0 or step == len(train_loader):
                    avg = {k: round(v / n_steps, 4) for k, v in running.items()}
                    fps = (step * args.batch_size) / max(time.time() - t_ep, 0.1)
                    lrs = scheduler.get_last_lr()
                    print(f"[v4.1-P1 Ep {epoch+1}/{args.p1_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr=({lrs[0]:.1e}/{lrs[1]:.1e}) {avg}", flush=True)

            eval_res, comp_score = _eval_p1(ema_model, eval_loaders, device)
            print(f"=== [v4.1 Phase 1 Polish] Epoch {epoch+1}/{args.p1_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | composite={comp_score:.4f}) ===", flush=True)
            for k, v in eval_res.items():
                print(f"  {k}: {v}", flush=True)

            history.append({"phase": "v41_P1_pure_rgb", "epoch": epoch + 1, "composite": comp_score, "eval": eval_res})
            hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            rclone_push_file(hist_path)

            if comp_score < best_p1_score:
                best_p1_score = comp_score
                torch.save({"epoch": epoch, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": best_p1_score, "eval": eval_res}, p1_best_path)
                rclone_push_file(p1_best_path)
                _export_v41_onnx(ema_model, args.height, args.width, "egogeom_omni_v41_best_pure_rgb_fp32.onnx", push_remote=True)
                print(f"  [NEW BEST v4.1 P1] composite={best_p1_score:.4f} -> pushed pure_rgb_best.pt + ONNX-17!", flush=True)

        del model, ema_model, train_p1, optimizer, scheduler, sky_guard
        gc.collect()
        torch.cuda.empty_cache()
        p1_done_flag.write_text("done", encoding="utf-8")
        print("=== v4.1 PHASE 1 COMPLETE! ===", flush=True)

    # =========================================================================
    # PHASE 2: Frozen-RGB Universal Signed-H3 + Standard-CNH + Depth (5 Ep)
    # =========================================================================
    print(f"\n================ v4.1 PHASE 2: Frozen-RGB Real-H3 Signed CNH + Std-CNH + Depth ({args.p2_epochs} Ep) ================", flush=True)
    model = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device).to(memory_format=torch.channels_last)
    ck_p1 = torch.load(p1_best_path, map_location=device, weights_only=False)
    model.load_v4_pure_rgb_trunk(ck_p1["ema"])

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("tof_branch.")
    tof_params = [param for param in model.parameters() if param.requires_grad]
    n_tof = sum(param.numel() for param in tof_params)
    n_rgb = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    print(f"[v4.1 P2] Frozen Pure-RGB ({n_rgb/1e6:.2f}M locked 0.00e+00!) | Signed-H3 Physical1DCNHBranchV41: {n_tof/1e6:.2f}M", flush=True)

    ema_model = copy.deepcopy(model).eval()
    for pe in ema_model.parameters():
        pe.requires_grad = False

    optimizer = torch.optim.AdamW(tof_params, lr=6.0e-4, weight_decay=1e-4, fused=True)
    total_steps = args.p2_epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)

    def p2_lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p2_lr_lambda)
    criterion_p2 = EgoGeomV4Loss(w_fscale=0.80, w_td_nrm=0.0)
    train_p2 = torch.compile(model, mode="default")
    best_p2_score = 999.0

    for epoch in range(args.p2_epochs):
        t_ep = time.time()
        model.eval()
        assert model.tof_branch is not None
        model.tof_branch.train()
        running: dict[str, float] = {}
        n_steps = 0

        for step, batch in enumerate(train_loader, 1):
            rgb = batch["rgb"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            K = batch["K"].to(device, non_blocking=True)
            g_cam = batch["g_cam"].to(device, non_blocking=True)
            gt_d = batch["depth_m"].to(device, non_blocking=True)
            vm = batch["valid_mask"].to(device, non_blocking=True)
            gls = batch["glass_mask"].to(device, non_blocking=True)
            nrm_cam = batch["normal_cam"].to(device, non_blocking=True)
            nrm_m = batch["normal_mask"].to(device, non_blocking=True)
            H, W = rgb.shape[-2:]

            rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
            hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
            gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
            dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
            R_lvl = rotation_cam_to_gravity_level(g_cam)
            gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)

            tof_p, has_tof = simulate_mixed_v41_prompt(gt_d, vm, K, glass_mask=gls)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_tof = train_p2(rgb, rays, tof_p, has_tof, host_focal=hf)
                loss, logs = criterion_p2(out_tof, gt_d, vm, gt_dh, dh_m, gt_nrm_lvl, nrm_m, gls)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(tof_params, 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    if pm.requires_grad:
                        pe.data.mul_(0.992).add_(pm.data, alpha=0.008)

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            n_steps += 1
            if step in (1, 25, 100) or step % 250 == 0 or step == len(train_loader):
                avg = {k: round(v / n_steps, 4) for k, v in running.items()}
                fps = (step * args.batch_size) / max(time.time() - t_ep, 0.1)
                print(f"[v4.1-P2 Ep {epoch+1}/{args.p2_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr={scheduler.get_last_lr()[0]:.2e} {avg}", flush=True)

        eval_res, omni_comp = _eval_p2(ema_model, eval_loaders, device)
        print(f"=== [v4.1 Phase 2 Unified Omni] Epoch {epoch+1}/{args.p2_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | omni_composite={omni_comp:.4f}) ===", flush=True)
        for k, v in eval_res.items():
            print(f"  {k}: {v}", flush=True)

        history.append({"phase": "v41_P2_unified_omni", "epoch": epoch + 1, "composite": omni_comp, "eval": eval_res})
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        rclone_push_file(hist_path)

        if omni_comp < best_p2_score:
            best_p2_score = omni_comp
            torch.save({"epoch": epoch, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": best_p2_score, "eval": eval_res}, p2_best_path)
            rclone_push_file(p2_best_path)
            _export_v41_onnx(
                ema_model, args.height, args.width,
                "egogeom_omni_v41_best_pure_rgb_fp32.onnx",
                "egogeom_omni_v41_best_omni_fp32.onnx",
                push_remote=True,
            )
            print(f"  [NEW BEST v4.1 P2] omni_composite={best_p2_score:.4f} -> pushed unified_best.pt + Dual M5-Fixed ONNX-17!", flush=True)

    print("\n=== EgoGeom-Omni v4.1 SILICON-EXACT REAL-H3 TRAINING COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
