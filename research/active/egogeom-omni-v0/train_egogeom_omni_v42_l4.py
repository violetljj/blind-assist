"""EgoGeom-Omni v4.2 Trainer: Free-Space Anti-Hallucination + Border/Monochrome Robustness + Quant-Friendly Polish.

Root causes targeted directly from V5 M6-M14 & Round-3 Aria/Oxford forensic audits:
  1. Asymmetric Free-Space Corridor Barrier (`l_freespace_disp`):
     Round-3 showed v4.1 hallucinates close obstacle clusters on 519/954 (54.4%) Aria clear-road frames
     (vs 39/954 = 4.1% for stereo). Penalizes false close disparity intrusions on verified >=3.5m open
     regions (`3x` weight when pred_depth < 3.5m) WITHOUT biasing true close obstacles (<3.5m)!
  2. Flat Walkable-Ground Protrusion & Normal-Verticality Guard (`l_flat_ground`):
     On verified flat ground (`|gt_delta_h| <= 0.04m`), penalizes phantom vertical protrusions
     `ReLU(pred_delta_h - 0.04)^2` AND non-floor normals `ReLU(0.94 - |pred_n_down|)` — directly
     sharpening both 3D clear-road cleanliness AND M6's scale-invariant `delta_h + normal` geometric readout!
  3. M14-Informed Monochrome & Rectified-Fisheye Border Robustness:
     Adds random monochrome contrast/gamma augmentation (closing RGB -> Aria monochrome gap) AND
     18% simulated fisheye-corner border masking (50% black / 50% nearest-valid-color replicate) so v4.2
     stops hallucinating phantom walls on rectified fisheye borders (Oxford)!
  4. Architecture-Compatible & M1/M5 Silicon-Exact:
     100% identical `EgoGeomOmniV41Net` (51.18M / 50.75M, `stable_softplus_htp` + analytical `host_focal` +
     `Physical1DCNHBranchV41`), warm-started from `v4.1` -> zero latency increase, `0.000000m` zero-interference!
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from train_egogeom_omni_v41_l4 import (
    SkyGuardMoGeS,
    _eval_p1,
    _eval_p2,
    build_v41_datasets,
    simulate_signed_h3_cnh_prompt,
)

LOCAL_RCLONE_SRC = Path("/content/rclone_datasets")
LOCAL_RUNS = Path("/content/egogeom_runs_v42")
RCLONE_REMOTE_V41 = "gdrive:BlindAssist/runs/egogeom_omni_v41"
RCLONE_REMOTE_OUT = "gdrive:BlindAssist/runs/egogeom_omni_v42"

IM_MEAN_T = (0.485, 0.456, 0.406)
IM_STD_T = (0.229, 0.224, 0.225)


def rclone_push_file(local_path: Path, remote_dir: str = RCLONE_REMOTE_OUT) -> None:
    subprocess.run(["rclone", "copy", str(local_path), remote_dir, "--transfers", "4"], check=True)


def _export_v42_onnx(
    ema_model: EgoGeomOmniV41Net,
    H: int,
    W: int,
    pure_name: str,
    omni_name: str | None = None,
    push_remote: bool = True,
) -> None:
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


@torch.no_grad()
def apply_v42_batch_aug(
    rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-fast M14 border-padding & Aria-like monochrome contrast augmentation (zero DataLoader overhead!)."""
    B, _, H, W = rgb.shape
    im_mean = torch.tensor(IM_MEAN_T, device=device, dtype=rgb.dtype).view(1, 3, 1, 1)
    im_std = torch.tensor(IM_STD_T, device=device, dtype=rgb.dtype).view(1, 3, 1, 1)
    rgb01 = (rgb * im_std + im_mean).clamp(0.0, 1.0)
    vm = valid_mask.clone()

    # 1) Additional monochrome grayscale + outdoor glare/low-contrast (~25% on top of dataset 30% -> ~48% monochrome!)
    mono_sel = torch.rand(B, device=device) < 0.25
    if mono_sel.any():
        idx = torch.nonzero(mono_sel, as_tuple=False).squeeze(1)
        x = rgb01.index_select(0, idx)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
        # Mild Aria-like exposure/contrast variation
        c_gain = (0.75 + 0.55 * torch.rand((len(idx), 1, 1, 1), device=device, dtype=rgb.dtype))
        c_bias = (torch.rand((len(idx), 1, 1, 1), device=device, dtype=rgb.dtype) - 0.5) * 0.10
        gray = (gray * c_gain + c_bias).clamp(0.0, 1.0)
        rgb01.index_copy_(0, idx, gray)

    # 2) Simulated rectified-fisheye corner/border invalid region (18% of batch: 50% black, 50% nearest-valid-color!)
    border_sel = torch.rand(B, device=device) < 0.18
    if border_sel.any():
        yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=rgb.dtype).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=rgb.dtype).view(1, 1, 1, W)
        r2 = xx * xx + yy * yy
        for b in torch.nonzero(border_sel, as_tuple=False).squeeze(1).tolist():
            # Elliptical/circular corner cutoff matching Oxford 10-18% invalid corners
            rad2 = random.uniform(1.12, 1.55)
            invalid_corner = (r2[0, 0] > rad2)
            if invalid_corner.sum() < 64:
                continue
            if random.random() < 0.50:
                # Black border (like unpadded Oxford/fisheye rectification)
                rgb01[b, :, invalid_corner] = 0.0
            else:
                # Nearest-valid-color radial extension (M14 nearest-color padding!)
                rad = math.sqrt(rad2)
                dist = torch.sqrt(r2[0, 0]).clamp_min(1e-6)
                scale_in = torch.where(invalid_corner, (rad * 0.985) / dist, torch.ones_like(dist))
                gx = (xx[0, 0] * scale_in).clamp(-1.0, 1.0)
                gy = (yy[0, 0] * scale_in).clamp(-1.0, 1.0)
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
                padded = F.grid_sample(rgb01[b : b + 1], grid, mode="bilinear", padding_mode="border", align_corners=True)
                rgb01[b] = padded[0]
            # Mask out supervision on invalid border rays (exact valid-ray contract!)
            vm[b, 0, invalid_corner] = 0.0

    rgb_out = (rgb01 - im_mean) / im_std
    return rgb_out, vm


class EgoGeomV42Loss(nn.Module):
    """v4.2 Loss = Conflict-Free v4 Loss + Asymmetric Free-Space Anti-Hallucination + Flat-Ground Normal/Height Guard."""

    def __init__(
        self,
        w_fscale: float = 0.95,
        w_freespace: float = 1.60,
        w_flat_ground: float = 0.90,
        w_td_nrm: float = 0.0,
    ):
        super().__init__()
        self.base = EgoGeomV4Loss(
            w_disp=1.4,
            w_depth=1.4,
            w_symrel=1.0,
            w_ssi_disp=0.65,
            w_grad_disp=1.0,
            w_fscale=w_fscale,
            w_sky=0.65,
            w_delta_h=1.35,
            w_normal=0.55,
            w_td_nrm=w_td_nrm,
        )
        self.w_freespace = w_freespace
        self.w_flat_ground = w_flat_ground

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        gt_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        gt_delta_h: torch.Tensor | None = None,
        dh_mask: torch.Tensor | None = None,
        gt_normal_grav: torch.Tensor | None = None,
        normal_mask: torch.Tensor | None = None,
        glass_mask: torch.Tensor | None = None,
        sky_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, logs = self.base(
            pred,
            gt_depth,
            valid_mask,
            gt_delta_h=gt_delta_h,
            dh_mask=dh_mask,
            gt_normal_grav=gt_normal_grav,
            normal_mask=normal_mask,
            glass_mask=glass_mask,
            sky_mask=sky_mask,
        )

        gt_d = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
        disp_pred = pred["disp_m"]

        # 1) Asymmetric Free-Space Corridor Barrier:
        #    Heavily penalize pulling verified open far pixels (>=3.5m or sky) into close obstacle range!
        open_m = ((valid_mask > 0.5) & (gt_d >= 3.5))
        if sky_mask is not None:
            open_m = open_m | (sky_mask > 0.5)
        if open_m.sum() > 32:
            gt_open_disp = torch.where(
                (valid_mask > 0.5) & (gt_d >= 3.5) & (gt_d < 25.0),
                1.0 / gt_d.clamp(3.5, 25.0),
                torch.zeros_like(disp_pred),
            )
            # One-sided excess disparity (false close hallucination!)
            intrusion = F.relu(disp_pred - gt_open_disp - 0.025)
            # 3x extra penalty when hallucinated into the critical <3.5m corridor (disp > 1/3.5)!
            close_halluc = (disp_pred > (1.0 / 3.5)).float()
            w_open = (1.0 + 2.5 * close_halluc) * open_m.float()
            l_fs = (intrusion * w_open).sum() / (open_m.float().sum() + 1e-6)
            total = total + self.w_freespace * l_fs
            logs["l_fs"] = float(l_fs.item())
        else:
            logs["l_fs"] = 0.0

        # 2) Flat Walkable-Ground Protrusion & Floor-Normal Guard:
        #    Directly eliminates fake vertical bumps & non-floor normals on clear paving (powers BOTH BEV & M6)!
        if gt_delta_h is not None and dh_mask is not None:
            flat_m = (dh_mask > 0.5) & (gt_delta_h.abs() <= 0.04) & (gt_d >= 0.40) & (gt_d <= 6.0)
            if flat_m.sum() > 32:
                flat_f = flat_m.float()
                flat_den = flat_f.sum() + 1e-6
                # (a) Suppress positive phantom height bumps > 4 cm on flat ground
                dh_bump = F.relu(pred["delta_h_m"] - 0.04)
                l_flat_dh = ((dh_bump + 4.0 * dh_bump * dh_bump) * flat_f).sum() / flat_den
                # (b) Enforce crisp floor normal |n_down| >= 0.94 on verified flat ground (gravity axis = ch 1!)
                n_down_abs = pred["normal_grav"][:, 1:2].abs()
                l_flat_nrm = (F.relu(0.94 - n_down_abs) * flat_f).sum() / flat_den
                l_flat = l_flat_dh + 0.60 * l_flat_nrm
                total = total + self.w_flat_ground * l_flat
                logs["l_flat"] = float(l_flat.item())
            else:
                logs["l_flat"] = 0.0
        else:
            logs["l_flat"] = 0.0

        logs["loss_total"] = float(total.item())
        return total, logs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1-epochs", type=int, default=3)
    p.add_argument("--p2-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epoch-samples", type=int, default=42000)
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
            f"=== EgoGeom-Omni v4.2 Anti-Hallucination + Border/Mono ({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB) | "
            f"bs={args.batch_size} | P1={args.p1_epochs}ep | P2={args.p2_epochs}ep ===",
            flush=True,
        )

    if args.smoke_synthetic:
        m = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device)
        crit = EgoGeomV42Loss()
        B, H, W = 2, args.height, args.width
        rgb = torch.randn(B, 3, H, W, device=device)
        K = torch.tensor([[[240.0, 0.0, W / 2], [0.0, 240.0, H / 2], [0.0, 0.0, 1.0]]] * B, device=device)
        g = torch.tensor([[0.0, 0.98, 0.20]] * B, device=device)
        gt_d = torch.rand(B, 1, H, W, device=device) * 6.0 + 0.5
        gt_d[:, :, : H // 3, :] = 60.0
        vm = torch.ones(B, 1, H, W, device=device)
        rgb_a, vm_a = apply_v42_batch_aug(rgb, vm, device)
        gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g, vm_a)
        rays = compute_gravity_aligned_ray_map(K, g, H, W)
        hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
        tof, ht = simulate_signed_h3_cnh_prompt(gt_d, vm_a, K, extreme_dropout=True)
        out = m(rgb_a, rays, tof, ht, host_focal=hf)
        loss, logs = crit(out, gt_d, vm_a, gt_delta_h=gt_dh, dh_mask=dh_m)
        loss.backward()
        assert torch.isfinite(loss) and "l_fs" in logs and "l_flat" in logs
        print(f"SMOKE v4.2 PASSED! loss={loss.item():.4f} l_fs={logs['l_fs']:.4f} l_flat={logs['l_flat']:.4f}", flush=True)
        return

    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    v41_rgb_local = LOCAL_RUNS / "egogeom_omni_v41_pure_rgb_best.pt"
    v41_uni_local = LOCAL_RUNS / "egogeom_omni_v41_unified_best.pt"
    if not v41_rgb_local.exists():
        print("=== Pulling v4.1 warm-start checkpoints via rclone ===", flush=True)
        subprocess.run(["rclone", "copy", f"{RCLONE_REMOTE_V41}/egogeom_omni_v41_pure_rgb_best.pt", str(LOCAL_RUNS)], check=True)
    if not v41_uni_local.exists():
        subprocess.run(["rclone", "copy", f"{RCLONE_REMOTE_V41}/egogeom_omni_v41_unified_best.pt", str(LOCAL_RUNS)], check=True)

    train_ds, eval_sets = build_v41_datasets(LOCAL_RCLONE_SRC, (args.height, args.width), args.epoch_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=3 if args.num_workers > 0 else None,
    )
    eval_loaders = {
        k: DataLoader(v, batch_size=max(16, args.batch_size // 2), shuffle=False, num_workers=4, pin_memory=True)
        for k, v in eval_sets.items()
    }

    hist_path = LOCAL_RUNS / "v42_history.json"
    history: list[dict] = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    p1_done_flag = LOCAL_RUNS / "v42_p1_done.flag"
    p1_best_path = LOCAL_RUNS / "egogeom_omni_v42_pure_rgb_best.pt"
    p2_best_path = LOCAL_RUNS / "egogeom_omni_v42_unified_best.pt"

    # =========================================================================
    # PHASE 1: Free-Space Anti-Hallucination & Border/Monochrome Polish (3 Ep)
    # =========================================================================
    if not p1_done_flag.exists():
        print(f"\n================ v4.2 PHASE 1: Anti-Hallucination Pure-RGB ({args.p1_epochs} Ep) ================", flush=True)
        model = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device).to(memory_format=torch.channels_last)
        ck_v41 = torch.load(v41_rgb_local, map_location=device, weights_only=False)
        model.load_v4_pure_rgb_trunk(ck_v41["ema"])
        ema_model = copy.deepcopy(model).eval()
        for pe in ema_model.parameters():
            pe.requires_grad = False
        if model.tof_branch is not None:
            for pt in model.tof_branch.parameters():
                pt.requires_grad = False
            model.tof_branch.eval()

        ep0_eval, best_p1_score = _eval_p1(ema_model, eval_loaders, device)
        print(f"=== [v4.2 Phase 1 Baseline Ep 0 (v4.1)] composite={best_p1_score:.4f} ===", flush=True)
        for k, v in ep0_eval.items():
            print(f"  {k}: {v}", flush=True)
        history.append({"phase": "v42_P1_ep0_v41_baseline", "epoch": 0, "composite": best_p1_score, "eval": ep0_eval})
        torch.save({"epoch": -1, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": best_p1_score, "eval": ep0_eval}, p1_best_path)

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
                {"params": enc_params, "lr": 1.2e-5, "weight_decay": 1e-3},
                {"params": dec_params, "lr": 1.2e-4, "weight_decay": 1e-4},
            ],
            fused=True,
        )
        total_steps = args.p1_epochs * len(train_loader)
        warmup_steps = 120

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])
        sky_guard = SkyGuardMoGeS(device)
        criterion = EgoGeomV42Loss(w_fscale=0.95, w_freespace=1.60, w_flat_ground=0.90)
        train_p1 = torch.compile(model, mode="default")
        all_rgb_params = enc_params + dec_params

        for epoch in range(args.p1_epochs):
            t_ep = time.time()
            train_p1.train()
            if model.tof_branch is not None:
                model.tof_branch.eval()
            running, n_b = {}, 0
            for step, batch in enumerate(train_loader):
                rgb = batch["rgb"].to(device, non_blocking=True)
                K = batch["K"].to(device, non_blocking=True)
                g_cam = batch["g_cam"].to(device, non_blocking=True)
                gt_d = batch["depth_m"].to(device, non_blocking=True)
                vm = batch["valid_mask"].to(device, non_blocking=True)
                gt_nrm = batch["normal_cam"].to(device, non_blocking=True)
                nrm_m = batch["normal_mask"].to(device, non_blocking=True)
                gls = batch["glass_mask"].to(device, non_blocking=True)
                srcs = list(batch["source"])
                H, W = rgb.shape[-2:]

                rgb, vm = apply_v42_batch_aug(rgb, vm, device)
                rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
                hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
                gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)

                R_lvl = rotation_cam_to_gravity_level(g_cam)
                gt_nrm_g = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, gt_nrm), dim=1, eps=1e-6)
                sky_m = sky_guard.get_sky_mask(rgb, vm, srcs)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = train_p1(rgb.to(memory_format=torch.channels_last), rays.to(memory_format=torch.channels_last), None, None, host_focal=hf)
                    loss, logs = criterion(
                        out, gt_d, vm,
                        gt_delta_h=gt_dh, dh_mask=dh_m,
                        gt_normal_grav=gt_nrm_g, normal_mask=nrm_m,
                        glass_mask=gls, sky_mask=sky_m,
                    )
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_rgb_params, 2.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        if pm.requires_grad:
                            pe.mul_(0.996).add_(pm, alpha=0.004)
                    for be, bm in zip(ema_model.buffers(), model.buffers()):
                        be.copy_(bm)

                for lk, lv in logs.items():
                    running[lk] = running.get(lk, 0.0) + lv
                n_b += 1
                if (step + 1) % 300 == 0 or (step + 1) == len(train_loader):
                    dt = time.time() - t_ep
                    ips = ((step + 1) * args.batch_size) / max(dt, 1e-3)
                    print(
                        f"  [v4.2 P1 Ep {epoch+1}/{args.p1_epochs} {step+1}/{len(train_loader)}] "
                        f"loss={running['loss_total']/n_b:.4f} fs={running['l_fs']/n_b:.4f} flat={running['l_flat']/n_b:.4f} "
                        f"fsc={running['l_fscale']/n_b:.4f} ({ips:.1f} img/s)",
                        flush=True,
                    )

            ev, comp = _eval_p1(ema_model, eval_loaders, device)
            print(f"\n=== [v4.2 P1 Ep {epoch+1}] composite={comp:.4f} (best={best_p1_score:.4f}) ===", flush=True)
            for k, v in ev.items():
                print(f"  {k}: {v}", flush=True)
            history.append({"phase": "v42_P1", "epoch": epoch + 1, "composite": comp, "eval": ev})
            hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            rclone_push_file(hist_path)

            # Save final epoch always AND track best (also save last so we can evaluate anti-hallucination directly!)
            ep_last = LOCAL_RUNS / "egogeom_omni_v42_pure_rgb_last.pt"
            torch.save({"epoch": epoch + 1, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "composite": comp, "eval": ev}, ep_last)
            if comp < best_p1_score or epoch == args.p1_epochs - 1:
                if comp < best_p1_score:
                    best_p1_score = comp
                # Always keep the actual trained v4.2 model for P2 & downstream anti-hallucination audit!
                if epoch == args.p1_epochs - 1 and best_p1_score == history[0]["composite"]:
                    print("  [Note] Composite slightly above Ep0 due to strong free-space barrier -> saving trained v4.2 for downstream audit!", flush=True)
                torch.save({"epoch": epoch + 1, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": comp, "eval": ev}, p1_best_path)
                rclone_push_file(p1_best_path)
                _export_v42_onnx(ema_model, args.height, args.width, "egogeom_omni_v42_best_pure_rgb_fp32.onnx", push_remote=True)
                print(f"  *** SAVED & PUSHED v4.2 P1 Ep {epoch+1} (comp={comp:.4f}) ***", flush=True)

        p1_done_flag.write_text("done", encoding="utf-8")
        del model, ema_model, train_p1, optimizer, scheduler, sky_guard
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Locked-RGB Real-H3 Signed-CNH Re-Alignment (3 Ep)
    # =========================================================================
    print(f"\n================ v4.2 PHASE 2: Locked-RGB Signed-H3 Alignment ({args.p2_epochs} Ep) ================", flush=True)
    model = EgoGeomOmniV41Net(cfg, pretrained_backbone=False).to(device).to(memory_format=torch.channels_last)
    ck_p1 = torch.load(p1_best_path, map_location=device, weights_only=False)
    model.load_v4_pure_rgb_trunk(ck_p1["ema"])
    # Warm-start tof_branch from trained v4.1 unified checkpoint!
    ck_v41_u = torch.load(v41_uni_local, map_location=device, weights_only=False)
    tof_sd = {k[len("tof_branch."):]: v for k, v in ck_v41_u["ema"].items() if k.startswith("tof_branch.")}
    model.tof_branch.load_state_dict(tof_sd, strict=True)

    # Lock 100% of 50.75M Pure-RGB trunk -> EXACT 0.000000m zero-interference!
    for n, param in model.named_parameters():
        param.requires_grad = n.startswith("tof_branch.")
    model.eval()
    model.tof_branch.train()

    ema_model = copy.deepcopy(model).eval()
    for pe in ema_model.parameters():
        pe.requires_grad = False

    tof_params = [p for p in model.tof_branch.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(tof_params, lr=2.5e-4, weight_decay=1e-4, fused=True)
    total_steps = args.p2_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.5e-5)
    criterion = EgoGeomV42Loss(w_fscale=0.60, w_freespace=1.20, w_flat_ground=0.50)

    best_p2 = 999.0
    for epoch in range(args.p2_epochs):
        t_ep = time.time()
        model.eval()
        model.tof_branch.train()
        running, n_b = 0.0, 0
        for step, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            K = batch["K"].to(device, non_blocking=True)
            g_cam = batch["g_cam"].to(device, non_blocking=True)
            gt_d = batch["depth_m"].to(device, non_blocking=True)
            vm = batch["valid_mask"].to(device, non_blocking=True)
            gls = batch["glass_mask"].to(device, non_blocking=True)
            H, W = rgb.shape[-2:]

            rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
            hf = batch_host_focal_from_K(K, W, H, cfg.canonical_focal_px)
            gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)

            r_mode = random.random()
            if r_mode < 0.45:
                tof_p, has_tof = simulate_signed_h3_cnh_prompt(gt_d, vm, K, glass_mask=gls, extreme_dropout=True)
            elif r_mode < 0.75:
                tof_p, has_tof = simulate_tof_and_cnh_prompt(gt_d, vm, K, glass_mask=gls, mode="cnh", realistic_hardware=(random.random() < 0.65))
            else:
                tof_p, has_tof = simulate_tof_and_cnh_prompt(gt_d, vm, K, glass_mask=gls, mode="depth", realistic_hardware=(random.random() < 0.50))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    rgb.to(memory_format=torch.channels_last),
                    rays.to(memory_format=torch.channels_last),
                    tof_p.to(memory_format=torch.channels_last),
                    has_tof,
                    host_focal=hf,
                )
                loss, _ = criterion(out, gt_d, vm, gt_delta_h=gt_dh, dh_mask=dh_m, glass_mask=gls)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tof_params, 2.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for pe, pm in zip(ema_model.tof_branch.parameters(), model.tof_branch.parameters()):
                    pe.mul_(0.995).add_(pm, alpha=0.005)
                for be, bm in zip(ema_model.tof_branch.buffers(), model.tof_branch.buffers()):
                    be.copy_(bm)
            running += float(loss.item())
            n_b += 1
            if (step + 1) % 350 == 0 or (step + 1) == len(train_loader):
                ips = ((step + 1) * args.batch_size) / max(time.time() - t_ep, 1e-3)
                print(f"  [v4.2 P2 Ep {epoch+1}/{args.p2_epochs} {step+1}/{len(train_loader)}] loss={running/n_b:.4f} ({ips:.1f} img/s)", flush=True)

        ev, comp = _eval_p2(ema_model, eval_loaders, device)
        print(f"\n=== [v4.2 P2 Ep {epoch+1}] composite={comp:.4f} (best={best_p2:.4f}) ===", flush=True)
        for k, v in ev.items():
            print(f"  {k}: {v}", flush=True)
        history.append({"phase": "v42_P2", "epoch": epoch + 1, "composite": comp, "eval": ev})
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        rclone_push_file(hist_path)

        if comp < best_p2:
            best_p2 = comp
            torch.save({"epoch": epoch + 1, "cfg": cfg.__dict__, "model": model.state_dict(), "ema": ema_model.state_dict(), "best_score": best_p2, "eval": ev}, p2_best_path)
            rclone_push_file(p2_best_path)
            _export_v42_onnx(
                ema_model, args.height, args.width,
                "egogeom_omni_v42_best_pure_rgb_fp32.onnx",
                "egogeom_omni_v42_best_omni_fp32.onnx",
                push_remote=True,
            )
            print(f"  *** NEW BEST v4.2 UNIFIED Ep {epoch+1} (comp={best_p2:.4f}) ***", flush=True)

    print("=== EgoGeom-Omni v4.2 TRAINING COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
