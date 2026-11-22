"""EgoGeom-Omni v4 Pinnacle Trainer (1.689B-Image Meta DINOv3 ConvNeXt-Small + Canonical Disparity + Physical 1D-CNH).

Two-Phase Decoupled Mastery on L4 (~3 hrs ≈ ~6 CU) or G4 (~1 hr):
  - Phase 1 (8 Ep Pure-RGB Mastery, 48,000 samples/ep, bs=32):
      * 1.689B-Image Meta DINOv3 `convnext_small.dinov3_lvd1689m` (50.7M, verified 17.72 ms HTP / 56.4 FPS on S24 Ultra!)
      * First 300 steps: freeze DINOv3 backbone while decoder warms up (`lr=6e-4`), then unfreeze DINOv3 at 10x lower LR (`6e-5`)!
      * Canonical-Focal Disparity (`disp = 1/Z >= 0`, `sky = 0.000`): ZERO near-vs-far trade-off!
      * Clean MoGe-2-ViT-L strictly-masked normal distillation (Ep 1-4, `mask > 0.5`, NEVER poisons sky!) +
        MoGe-2-ViT-S upper-frame sky-mask-only guard on real outdoor `SANPO-Real`!
      * Balanced 50/50 Outdoor/Indoor gold-GT pool across all 7 datasets (zero test/val leakage!).
  - Phase 2 (5 Ep Frozen-RGB Physical 1D-CNH + 8x8 Depth Attachment):
      * Locks 100% of the 50.7M DINOv3 Pure-RGB trunk (`0.000000e+00` live-silicon zero-interference!).
      * Trains ONLY the `~1.2M` GroupNorm(8) `Physical1DCNHBranchV4` (bin0-free tail + 1D range-axis differential +
        dual multiplicative & additive disparity residuals) on mixed clean + real H3 hardware physics!
      * Exports Dual Static ONNX-17 (`egogeom_omni_v4_best_pure_rgb_fp32.onnx` & `egogeom_omni_v4_best_omni_fp32.onnx`) to Drive!
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
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
from egogeom_omni_v4_model import (
    EgoGeomOmniV4ExportWrapper,
    EgoGeomOmniV4Net,
    EgoGeomOmniV4PureRGBExportWrapper,
    EgoGeomV4Config,
)
from egogeom_v4_loss import EgoGeomV4Loss

DRIVE_DATA = Path("/content/drive/MyDrive/BlindAssist/datasets")
DRIVE_RUNS = Path("/content/drive/MyDrive/BlindAssist/runs/egogeom_omni_v4")
LOCAL_STAGE = Path("/content/egogeom_stage_data_v2")
LOCAL_RUNS = Path("/content/egogeom_runs_v4")


class CleanMoGeTeacherV4:
    """Clean GPU-only MoGe-2 helper: strictly masked normals (Ep 1-4) + upper-frame sky mask on SANPO-Real (all P1).

    NEVER feeds unshifted `pts_z` into depth loss!
    """

    def __init__(self, device: torch.device, load_normal_teacher: bool = True):
        self.device = device
        self.moge_l = None
        self.moge_s = None
        self.is_logits_l = False
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
            print(f"[Teacher v4] MoGe-2-ViT-S sky guard ready (is_logits={self.is_logits_s})", flush=True)
            if load_normal_teacher:
                ml = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval().to(device)
                for p in ml.parameters():
                    p.requires_grad = False
                self.moge_l = ml
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prl = ml.forward(torch.rand(1, 3, 168, 224, device=device), num_tokens=224)["mask"].float()
                self.is_logits_l = bool(prl.min().item() < -0.05 or prl.max().item() > 1.05)
                print(f"[Teacher v4] MoGe-2-ViT-L masked-normal teacher ready (is_logits={self.is_logits_l})", flush=True)
        except Exception as e:
            print(f"[Teacher v4 WARN] MoGe unavailable ({e}); training with pure GT + native >=12m sky anchors.", flush=True)

    @torch.no_grad()
    def extract(
        self,
        rgb_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        sources: list[str],
        step: int,
        want_normals: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Returns (sky_mask, teacher_normal_cam, teacher_normal_mask)."""
        B, _, H, W = rgb_norm.shape
        sky_mask = None
        t_nrm = None
        t_nrm_m = None

        # 1) Every 2nd step during Ep 1-4: MoGe-2-ViT-L clean masked normals + SANPO-Real sky mask in ONE pass!
        if want_normals and self.moge_l is not None and (step % 2 == 1):
            rgb01 = (rgb_norm.float() * self.im_std + self.im_mean).clamp(0.0, 1.0)
            rgb01_s = F.interpolate(rgb01, size=(168, 224), mode="bilinear", align_corners=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                oc = self.moge_l.forward(rgb01_s, num_tokens=224)
            mk = oc["mask"].float().unsqueeze(1)
            prob = torch.sigmoid(mk) if self.is_logits_l else mk
            prob_hw = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
            if "normal" in oc and oc["normal"] is not None:
                nrm = oc["normal"].float().permute(0, 3, 1, 2)
                t_nrm = F.normalize(F.interpolate(nrm, size=(H, W), mode="bilinear", align_corners=False), dim=1, eps=1e-6)
                t_nrm_m = (prob_hw > 0.60).float()
            sr_idx = [i for i, s in enumerate(sources) if s == "sanpo_real"]
            if sr_idx:
                sky_mask = torch.zeros((B, 1, H, W), device=self.device, dtype=torch.float32)
                v_cut = int(H * 0.65)
                upper = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.bool)
                upper[:, :, :v_cut, :] = True
                idx_t = torch.tensor(sr_idx, device=self.device, dtype=torch.long)
                sky_sr = ((prob_hw.index_select(0, idx_t) < 0.25) & (valid_mask.index_select(0, idx_t) < 0.5) & upper).float()
                sky_mask.index_copy_(0, idx_t, sky_sr)
            return sky_mask, t_nrm, t_nrm_m

        # 2) Otherwise: fast MoGe-2-ViT-S sky-only guard ONLY on the ~8 sanpo_real frames (~35 ms/batch!)
        if self.moge_s is not None:
            sr_idx = [i for i, s in enumerate(sources) if s == "sanpo_real"]
            if sr_idx:
                idx_t = torch.tensor(sr_idx, device=self.device, dtype=torch.long)
                rgb01 = (rgb_norm.index_select(0, idx_t).float() * self.im_std + self.im_mean).clamp(0.0, 1.0)
                rgb01_s = F.interpolate(rgb01, size=(168, 224), mode="bilinear", align_corners=False)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    mk = self.moge_s.forward(rgb01_s, num_tokens=224)["mask"].float().unsqueeze(1)
                prob = torch.sigmoid(mk) if self.is_logits_s else mk
                prob_hw = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
                v_cut = int(H * 0.65)
                upper = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.bool)
                upper[:, :, :v_cut, :] = True
                sky_mask = torch.zeros((B, 1, H, W), device=self.device, dtype=torch.float32)
                sky_sr = ((prob_hw < 0.25) & (valid_mask.index_select(0, idx_t) < 0.5) & upper).float()
                sky_mask.index_copy_(0, idx_t, sky_sr)
        return sky_mask, None, None

    def unload_large(self) -> None:
        if self.moge_l is not None:
            self.moge_l = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Teacher v4] Unloaded MoGe-2-ViT-L for Pure-GT razor refinement!", flush=True)

    def unload_all(self) -> None:
        self.moge_l = None
        self.moge_s = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_v4_domain(
    model: EgoGeomOmniV4Net,
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
    fr_ratios: list[float] = []
    ssi_errs: list[float] = []
    p95_list: list[float] = []
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
        gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if tof_mode == "none":
                out = model(rgb, rays, None, None, return_features=False)
            else:
                tof_p, has_tof = simulate_tof_and_cnh_prompt(
                    gt_d, vm, K, glass_mask=gls, mode=tof_mode, realistic_hardware=realistic_hardware
                )
                out = model(rgb, rays, tof_p, has_tof, return_features=False)

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


def build_v4_datasets(out_hw: tuple[int, int], epoch_samples: int):
    t0 = time.time()
    n_cpu = max(8, os.cpu_count() or 8)
    print(f"=== Staging/Loading NVMe datasets ({n_cpu} vCPUs) ===", flush=True)
    sanpo_real_npzs = (
        stage_tar_dir(DRIVE_DATA / "sanpo_real_train76_v0", LOCAL_STAGE / "sanpo_real_train76", workers=8)
        + stage_tar_dir(DRIVE_DATA / "sanpo_real_chest240_v0", LOCAL_STAGE / "sanpo_real_chest240", workers=8)
        + stage_tar_dir(DRIVE_DATA / "sanpo_real_head100_autoseg_v0", LOCAL_STAGE / "sanpo_real_head100", workers=8)
    )
    print(f"  [stage] SANPO-Real: {len(sanpo_real_npzs)} npz ({time.time()-t0:.0f}s)", flush=True)
    sanpo_synth_npzs = (
        stage_tar_dir(DRIVE_DATA / "sanpo_synth_287_v0", LOCAL_STAGE / "sanpo_synth_287", workers=8)
        + stage_tar_dir(DRIVE_DATA / "sanpo_synth_extra450_v0", LOCAL_STAGE / "sanpo_synth_extra450", max_shards=14, workers=8)
    )
    print(f"  [stage] SANPO-Synth: {len(sanpo_synth_npzs)} npz ({time.time()-t0:.0f}s)", flush=True)
    tartan_npzs = stage_tar_dir(DRIVE_DATA / "tartanair_widefov_p13_v0", LOCAL_STAGE / "tartanair", workers=8)
    m3ed_npzs = stage_m3ed_fast(DRIVE_DATA / "m3ed_spot_train_v1", LOCAL_STAGE / "m3ed_spot_fast", target_hw=out_hw, workers=8)
    glass_dir = stage_glassrecon_local(DRIVE_DATA / "glassrecon_v1", LOCAL_STAGE / "glassrecon")
    tb_dirs = stage_tum_bonn_local(DRIVE_DATA / "tum_rgbd_v1", DRIVE_DATA / "bonn_dynamic_v1", LOCAL_STAGE / "tum_bonn", max_archives=28, workers=8)
    sanpo_zs40_npzs = stage_tar_dir(DRIVE_DATA / "sanpo_real_zeroshot_40_v0", LOCAL_STAGE / "sanpo_real_zs40", workers=4)
    print(f"  [stage] Tartan+M3ED+Glass+TUM+ZS40 done ({time.time()-t0:.0f}s), staging Hypersim...", flush=True)
    hyp_npzs = stage_hypersim_fast(DRIVE_DATA / "hypersim_full_v1", LOCAL_STAGE / "hypersim_fast_v2", target_hw=out_hw, workers=min(24, n_cpu))
    print(f"=== Dataset cache ready in {time.time()-t0:.1f}s (Hypersim={len(hyp_npzs)}) ===", flush=True)

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

    # 52% Outdoor / 48% Indoor balanced mix for ALL-RANGE indoor+outdoor mastery!
    train_ds = MultiDomainEgoGeomDataset(
        sources=[
            (hyp_train, 0.28),
            (sr_train, 0.24),
            (ss_train, 0.12),
            (tb_train, 0.11),
            (gls_train, 0.09),
            (ta_train, 0.08),
            (m3_train, 0.08),
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


def _export_v4_onnx(ema_model: EgoGeomOmniV4Net, H: int, W: int, pure_name: str, omni_name: str | None = None) -> None:
    d_rgb = torch.randn(1, 3, H, W)
    d_ray = torch.randn(1, 3, H, W)
    exp_rgb = EgoGeomOmniV4PureRGBExportWrapper(copy.deepcopy(ema_model).cpu().eval())
    p_loc = LOCAL_RUNS / pure_name
    torch.onnx.export(
        exp_rgb, (d_rgb, d_ray), str(p_loc), opset_version=17,
        input_names=["rgb", "grav_rays"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
        dynamo=False,
    )
    shutil.copyfile(p_loc, DRIVE_RUNS / pure_name)
    if omni_name is not None:
        d_tof = torch.zeros(1, 20, H // 4, W // 4)
        d_has = torch.zeros(1, 1, 1, 1)
        exp_om = EgoGeomOmniV4ExportWrapper(copy.deepcopy(ema_model).cpu().eval())
        p_om = LOCAL_RUNS / omni_name
        torch.onnx.export(
            exp_om, (d_rgb, d_ray, d_tof, d_has), str(p_om), opset_version=17,
            input_names=["rgb", "grav_rays", "tof_grid_s4", "has_tof"],
            output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
            dynamo=False,
        )
        shutil.copyfile(p_om, DRIVE_RUNS / omni_name)


def _eval_p1(model: EgoGeomOmniV4Net, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_v4_domain(model, loader, device, tof_mode="none")
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__no_tof"] = evaluate_v4_domain(model, loader, device, tof_mode="none", dark_scale=0.18)
    s = res["sanpo_real_zs40__no_tof"]
    h = res["hypersim_val__no_tof"]
    g = res["glassrecon_val__no_tof"]
    sf = s["far_absrel"] if s["far_absrel"] > 0 else 0.30
    comp = (
        0.28 * s["near_absrel"]
        + 0.17 * sf
        + 0.20 * (1.0 - s["near_delta1"])
        + 0.15 * g["near_absrel"]
        + 0.20 * (1.0 - h["near_delta1"])
    )
    return res, round(comp, 4)


def _eval_p2(model: EgoGeomOmniV4Net, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_v4_domain(model, loader, device, tof_mode="none")
        res[f"{name}__with_cnh"] = evaluate_v4_domain(model, loader, device, tof_mode="cnh", realistic_hardware=False)
        res[f"{name}__with_cnh_real_tilt"] = evaluate_v4_domain(model, loader, device, tof_mode="cnh", realistic_hardware=True)
        res[f"{name}__with_depth"] = evaluate_v4_domain(model, loader, device, tof_mode="depth", realistic_hardware=False)
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__no_tof"] = evaluate_v4_domain(model, loader, device, tof_mode="none", dark_scale=0.18)
            res[f"{name}_dark018__with_cnh"] = evaluate_v4_domain(model, loader, device, tof_mode="cnh", dark_scale=0.18)
    comp = (
        0.35 * res["sanpo_real_zs40__with_cnh"]["near_absrel"]
        + 0.25 * res["hypersim_val__with_cnh"]["near_absrel"]
        + 0.15 * res["glassrecon_val__with_cnh"]["near_absrel"]
        + 0.15 * res["sanpo_real_zs40__with_depth"]["near_absrel"]
        + 0.10 * res["sanpo_real_zs40__with_cnh_real_tilt"]["near_absrel"]
    )
    return res, round(comp, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="convnext_small.dinov3_lvd1689m")
    p.add_argument("--p1-epochs", type=int, default=8)
    p.add_argument("--p1-normal-teacher-epochs", type=int, default=4)
    p.add_argument("--p2-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epoch-samples", type=int, default=48000)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--smoke-synthetic", action="store_true")
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EgoGeomV4Config(backbone=args.backbone)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if not args.smoke_synthetic and vram_gb >= 85.0:
            args.batch_size = 64
        print(
            f"=== EgoGeom-Omni v4 DINOv3 Pinnacle ({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB) | "
            f"backbone={cfg.backbone} | bs={args.batch_size} | P1={args.p1_epochs}ep | P2={args.p2_epochs}ep ===",
            flush=True,
        )

    if args.smoke_synthetic:
        m = EgoGeomOmniV4Net(cfg, pretrained_backbone=False).to(device)
        crit = EgoGeomV4Loss()
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
        tof, ht = simulate_tof_and_cnh_prompt(gt_d, vm, K, mode="cnh", realistic_hardware=True)
        o1 = m(rgb, rays, tof, ht)
        loss, logs = crit(o1, gt_d, vm, sky_mask=sky)
        loss.backward()
        assert torch.isfinite(loss) and logs["l_sky"] > 0.0
        m.eval()
        with torch.no_grad():
            o_rgb = m(rgb, rays, None, None)
            o_zero = m(rgb, rays, torch.zeros_like(tof), torch.zeros_like(ht))
            zi = float((o_rgb["depth_m"] - o_zero["depth_m"]).abs().max().item())
        assert zi == 0.0, f"Expected 0.0 zero-interference, got {zi}"
        n_all = sum(p.numel() for p in m.parameters()) / 1e6
        n_tof = sum(p.numel() for p in m.tof_branch.parameters()) / 1e6
        print(f"SMOKE v4 PASSED! params={n_all:.2f}M (ToF={n_tof:.2f}M) loss={loss.item():.4f} l_sky={logs['l_sky']:.4f} zero_interference={zi:.2e}", flush=True)
        return

    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    train_ds, eval_sets = build_v4_datasets((args.height, args.width), args.epoch_samples)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=3 if args.num_workers > 0 else None,
    )
    eval_loaders = {
        k: DataLoader(v, batch_size=max(16, args.batch_size // 2), shuffle=False, num_workers=4, pin_memory=True)
        for k, v in eval_sets.items()
    }

    hist_path = DRIVE_RUNS / "v4_history.json"
    history: list[dict] = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    p1_done_flag = DRIVE_RUNS / "v4_p1_done.flag"
    p1_best_path = DRIVE_RUNS / "egogeom_omni_v4_pure_rgb_best.pt"
    p1_latest_path = DRIVE_RUNS / "egogeom_omni_v4_p1_latest.pt"
    p2_best_path = DRIVE_RUNS / "egogeom_omni_v4_unified_best.pt"
    p2_latest_path = DRIVE_RUNS / "egogeom_omni_v4_p2_latest.pt"

    # =========================================================================
    # PHASE 1: 1.689B-Image DINOv3 Pure-RGB Mastery (8 Ep)
    # =========================================================================
    if not p1_done_flag.exists():
        print(f"\n================ v4 PHASE 1: 1.689B DINOv3 Pure-RGB Mastery ({args.p1_epochs} Ep) ================", flush=True)
        start_ep = 0
        best_p1_score = 999.0
        resuming = p1_latest_path.exists()
        model = EgoGeomOmniV4Net(cfg, pretrained_backbone=(not resuming)).to(device).to(memory_format=torch.channels_last)

        if resuming:
            ck = torch.load(p1_latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema_model = copy.deepcopy(model).eval()
            ema_model.load_state_dict(ck["ema"])
            start_ep = int(ck["epoch"]) + 1
            best_p1_score = float(ck.get("best_score", 999.0))
            print(f"[v4 P1 Resume] epoch {start_ep}/{args.p1_epochs} best={best_p1_score:.4f}", flush=True)
        else:
            ema_model = copy.deepcopy(model).eval()
            print(f"[v4 P1 Init] Loaded 1.689B-Image Meta DINOv3 {cfg.backbone}!", flush=True)

        if model.tof_branch is not None:
            for pt in model.tof_branch.parameters():
                pt.requires_grad = False
            model.tof_branch.eval()

        enc_params, dec_params = [], []
        for n, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if n.startswith("enc."):
                enc_params.append(param)
            else:
                dec_params.append(param)

        # Differential LR: protect 1.689B DINOv3 prior with 10x lower LR (6e-5) + 300-step freeze warmup!
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": 6.0e-5, "weight_decay": 1e-3},
                {"params": dec_params, "lr": 6.0e-4, "weight_decay": 1e-4},
            ],
            fused=True,
        )
        total_steps = args.p1_epochs * len(train_loader)
        freeze_steps = 300
        warmup_steps = 300

        def enc_lr_lambda(step: int) -> float:
            if step < freeze_steps:
                return 0.0  # Keep 1.689B DINOv3 100% frozen while decoder aligns cleanly!
            s = step - freeze_steps
            if s < warmup_steps:
                return float(s + 1) / float(warmup_steps)
            prog = (s - warmup_steps) / float(max(1, total_steps - freeze_steps - warmup_steps))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        def dec_lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [enc_lr_lambda, dec_lr_lambda])
        if resuming:
            optimizer.load_state_dict(ck["opt"])
            scheduler.load_state_dict(ck["sched"])

        for pe in ema_model.parameters():
            pe.requires_grad = False

        teacher = CleanMoGeTeacherV4(device, load_normal_teacher=(start_ep < args.p1_normal_teacher_epochs))
        criterion = EgoGeomV4Loss()
        train_p1 = torch.compile(model, mode="default")
        all_rgb_params = enc_params + dec_params
        global_step = start_ep * len(train_loader)

        for epoch in range(start_ep, args.p1_epochs):
            want_nrm = epoch < args.p1_normal_teacher_epochs
            if not want_nrm:
                teacher.unload_large()

            t_ep = time.time()
            train_p1.train()
            if model.tof_branch is not None:
                model.tof_branch.eval()
            running: dict[str, float] = {}
            n_steps = 0

            for step, batch in enumerate(train_loader, 1):
                global_step += 1
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
                gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
                dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
                R_lvl = rotation_cam_to_gravity_level(g_cam)
                gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)

                sky_m, t_nrm_cam, t_nrm_m = teacher.extract(rgb, vm, sources, global_step, want_normals=want_nrm)
                t_nrm_lvl = (
                    F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, t_nrm_cam), dim=1, eps=1e-6)
                    if t_nrm_cam is not None
                    else None
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_rgb = train_p1(rgb, rays, None, None, return_features=False)
                    loss, logs = criterion(
                        out_rgb, gt_d, vm, gt_dh, dh_m, gt_nrm_lvl, nrm_m, gls,
                        sky_mask=sky_m, teacher_normal_grav=t_nrm_lvl, teacher_normal_mask=t_nrm_m,
                    )

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
                    print(f"[v4-P1 Ep {epoch+1}/{args.p1_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr=({lrs[0]:.1e}/{lrs[1]:.1e}) {avg}", flush=True)

            eval_res, comp_score = _eval_p1(ema_model, eval_loaders, device)
            print(f"=== [v4 Phase 1 DINOv3 Pure-RGB] Epoch {epoch+1}/{args.p1_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | composite={comp_score:.4f}) ===", flush=True)
            for k, v in eval_res.items():
                print(f"  {k}: {v}", flush=True)

            history.append({"phase": "v4_P1_pure_rgb", "epoch": epoch + 1, "composite": comp_score, "eval": eval_res})
            ck_loc = LOCAL_RUNS / "egogeom_omni_v4_p1_latest.pt"
            torch.save(
                {
                    "epoch": epoch, "cfg": cfg.__dict__, "model": model.state_dict(),
                    "ema": ema_model.state_dict(), "opt": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_score": min(best_p1_score, comp_score),
                    "eval": eval_res,
                },
                ck_loc,
            )
            shutil.copyfile(ck_loc, p1_latest_path)
            if comp_score < best_p1_score:
                best_p1_score = comp_score
                shutil.copyfile(ck_loc, p1_best_path)
                _export_v4_onnx(ema_model, args.height, args.width, "egogeom_omni_v4_best_pure_rgb_fp32.onnx")
                print(f"  [NEW BEST v4 P1] composite={best_p1_score:.4f} -> saved pure_rgb_best.pt + Pure-RGB ONNX-17!", flush=True)
            hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        teacher.unload_all()
        del model, ema_model, train_p1, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        p1_done_flag.write_text("done", encoding="utf-8")
        print("=== v4 PHASE 1 COMPLETE! 1.689B DINOv3 Pure-RGB Master locked & saved! ===", flush=True)

    # =========================================================================
    # PHASE 2: Frozen-RGB Physical 1D-CNH + 8x8 Depth Attachment (5 Ep)
    # =========================================================================
    print(f"\n================ v4 PHASE 2: Frozen-RGB Physical 1D-CNH + Depth ({args.p2_epochs} Ep) ================", flush=True)
    model = EgoGeomOmniV4Net(cfg, pretrained_backbone=False).to(device).to(memory_format=torch.channels_last)
    ck_p1 = torch.load(p1_best_path if p1_best_path.exists() else p1_latest_path, map_location=device, weights_only=False)
    model.load_state_dict(ck_p1["ema"], strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("tof_branch.")
    tof_params = [param for param in model.parameters() if param.requires_grad]
    n_tof = sum(param.numel() for param in tof_params)
    n_rgb = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    print(f"[v4 P2] Frozen 1.689B DINOv3 Pure-RGB ({n_rgb/1e6:.2f}M locked 0.00e+00!) | Physical 1D-CNH: {n_tof/1e6:.2f}M", flush=True)

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
    start_ep2 = 0
    best_p2_score = 999.0

    if p2_latest_path.exists():
        ck2 = torch.load(p2_latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck2["model"])
        ema_model.load_state_dict(ck2["ema"])
        optimizer.load_state_dict(ck2["opt"])
        scheduler.load_state_dict(ck2["sched"])
        start_ep2 = int(ck2["epoch"]) + 1
        best_p2_score = float(ck2.get("best_score", 999.0))
        print(f"[v4 P2 Resume] epoch {start_ep2}/{args.p2_epochs} best={best_p2_score:.4f}", flush=True)

    criterion_p2 = EgoGeomV4Loss(w_td_nrm=0.0)
    train_p2 = torch.compile(model, mode="default")

    for epoch in range(start_ep2, args.p2_epochs):
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
            gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
            dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
            R_lvl = rotation_cam_to_gravity_level(g_cam)
            gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)

            use_real_hw = (step % 2 == 0)
            tof_p, has_tof = simulate_tof_and_cnh_prompt(
                gt_d, vm, K, glass_mask=gls, mode="mixed_tof", realistic_hardware=use_real_hw
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_tof = train_p2(rgb, rays, tof_p, has_tof, return_features=False)
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
                print(f"[v4-P2 Ep {epoch+1}/{args.p2_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr={scheduler.get_last_lr()[0]:.2e} {avg}", flush=True)

        eval_res, omni_comp = _eval_p2(ema_model, eval_loaders, device)
        print(f"=== [v4 Phase 2 Unified Omni] Epoch {epoch+1}/{args.p2_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | omni_composite={omni_comp:.4f}) ===", flush=True)
        for k, v in eval_res.items():
            print(f"  {k}: {v}", flush=True)

        history.append({"phase": "v4_P2_unified_omni", "epoch": epoch + 1, "composite": omni_comp, "eval": eval_res})
        ck2_loc = LOCAL_RUNS / "egogeom_omni_v4_p2_latest.pt"
        torch.save(
            {
                "epoch": epoch, "cfg": cfg.__dict__, "model": model.state_dict(),
                "ema": ema_model.state_dict(), "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(), "best_score": min(best_p2_score, omni_comp),
                "eval": eval_res,
            },
            ck2_loc,
        )
        shutil.copyfile(ck2_loc, p2_latest_path)
        if omni_comp < best_p2_score:
            best_p2_score = omni_comp
            shutil.copyfile(ck2_loc, p2_best_path)
            _export_v4_onnx(
                ema_model, args.height, args.width,
                "egogeom_omni_v4_best_pure_rgb_fp32.onnx",
                "egogeom_omni_v4_best_omni_fp32.onnx",
            )
            print(f"  [NEW BEST v4 P2] omni_composite={best_p2_score:.4f} -> saved unified_best.pt + Dual ONNX-17!", flush=True)
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n=== EgoGeom-Omni v4 DINOv3 PINNACLE TRAINING COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
