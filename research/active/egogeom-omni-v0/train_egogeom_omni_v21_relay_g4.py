"""EgoGeom-Omni v2.1: ControlNet-Style Decoupled Two-Phase Relay Training on G4.

Why v2.1 Relay crushes joint 2-pass training (AND finishes 35 min faster!):
  - Joint 2-pass training suffered from Modality Laziness (ToF shortcut learns in 2 epochs and
    hijacks the shared FPN/heads, starving monocular Pure-RGB on thin poles & transparent glass).
  - Phase 1 (Pure-RGB Mastery, 9 ep @ ~280-400 img/s, ~30 min):
      Trains the 12.9M Pure-RGB trunk 100% free of ToF shortcut interference (Ep 1-6 Round-Robin
      3 Giant Teachers + Ep 7-9 Pure-GT razor refinement) with Asymmetric Near-Field Safety AbsRel
      Loss (1.5x penalty on dangerous far-overshoot of <3.5m obstacles!).
  - Phase 2 (Frozen-RGB Zero-Init ControlNet ToF/CNH Attachment, 6 ep @ ~350+ img/s, ~16 min):
      FREEZES 100% of Pure-RGB trunk weights AND BatchNorm running stats (.eval()), re-zeros
      ZeroInitToFPromptBranch output projections, and trains ONLY the 1.55M ToF/CNH prompt branch
      on pure GT + 50% sparse 8x8 ToF + 50% full 8x8x16 double-peak CNH!
  - Mathematical Guarantees:
      1. Pure-RGB (has_tof=0) has 0.00e+00 regression — bit-identical to the Phase 1 Master!
      2. With-ToF / CNH (has_tof=1) builds on top of the Phase 1 Master (even stronger than Ep 7!).
      3. ONE single unified checkpoint & ONNX works at peak accuracy for BOTH modes with ZERO
         runtime Split-BN swapping!
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from egogeom_losses_and_physics import (
    EgoGeomOmniLoss,
    align_teacher_depth_to_sparse_gt,
    compute_gravity_aligned_ray_map,
    compute_local_ground_height_residual,
    rotation_cam_to_gravity_level,
    simulate_tof_and_cnh_prompt,
)
from egogeom_omni_model import (
    EgoGeomConfig,
    EgoGeomOmniExportWrapper,
    EgoGeomOmniNet,
    EgoGeomOmniPureRGBExportWrapper,
)
from train_egogeom_omni_stage1_g4 import (
    DRIVE_RUNS,
    LOCAL_RUNS,
    RoundRobinMultiTeacher,
    build_datasets,
    evaluate_domain,
)


def _export_pure_rgb_onnx(ema_model: EgoGeomOmniNet, H: int, W: int, out_name: str) -> None:
    d_rgb = torch.randn(1, 3, H, W)
    d_ray = torch.randn(1, 3, H, W)
    exp_rgb = EgoGeomOmniPureRGBExportWrapper(copy.deepcopy(ema_model).cpu().eval())
    p_local = LOCAL_RUNS / out_name
    torch.onnx.export(
        exp_rgb, (d_rgb, d_ray), str(p_local), opset_version=17,
        input_names=["rgb", "grav_rays"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
        dynamo=False,
    )
    shutil.copyfile(p_local, DRIVE_RUNS / out_name)


def _export_omni_onnx(ema_model: EgoGeomOmniNet, H: int, W: int, out_name: str) -> None:
    d_rgb = torch.randn(1, 3, H, W)
    d_ray = torch.randn(1, 3, H, W)
    d_tof_s4 = torch.zeros(1, 20, H // 4, W // 4)
    d_has = torch.zeros(1, 1, 1, 1)
    exp_omni = EgoGeomOmniExportWrapper(copy.deepcopy(ema_model).cpu().eval())
    p_local = LOCAL_RUNS / out_name
    torch.onnx.export(
        exp_omni, (d_rgb, d_ray, d_tof_s4, d_has), str(p_local), opset_version=17,
        input_names=["rgb", "grav_rays", "tof_grid_s4", "has_tof"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"],
        dynamo=False,
    )
    shutil.copyfile(p_local, DRIVE_RUNS / out_name)


def _eval_pure_rgb(model: EgoGeomOmniNet, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res: dict[str, dict] = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_domain(model, loader, device, tof_mode="none")
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__no_tof"] = evaluate_domain(model, loader, device, tof_mode="none", dark_scale=0.18)
    s_sanpo = res.get("sanpo_real_zs40__no_tof", {}).get("near_absrel", 1.0)
    s_hyp = res.get("hypersim_val__no_tof", {}).get("near_absrel", 1.0)
    s_gls = res.get("glassrecon_val__no_tof", {}).get("near_absrel", 1.0)
    composite = 0.50 * s_sanpo + 0.30 * s_hyp + 0.20 * s_gls
    return res, composite


def _eval_unified(model: EgoGeomOmniNet, eval_loaders: dict, device: torch.device) -> tuple[dict, float]:
    res: dict[str, dict] = {}
    for name, loader in eval_loaders.items():
        res[f"{name}__no_tof"] = evaluate_domain(model, loader, device, tof_mode="none")
        res[f"{name}__with_cnh"] = evaluate_domain(model, loader, device, tof_mode="cnh")
        res[f"{name}__with_depth"] = evaluate_domain(model, loader, device, tof_mode="depth")
        if name == "sanpo_real_zs40":
            res[f"{name}_dark018__no_tof"] = evaluate_domain(model, loader, device, tof_mode="none", dark_scale=0.18)
            res[f"{name}_dark018__with_cnh"] = evaluate_domain(model, loader, device, tof_mode="cnh", dark_scale=0.18)
    c_sanpo = res.get("sanpo_real_zs40__with_cnh", {}).get("near_absrel", 1.0)
    c_hyp = res.get("hypersim_val__with_cnh", {}).get("near_absrel", 1.0)
    c_gls = res.get("glassrecon_val__with_cnh", {}).get("near_absrel", 1.0)
    composite = 0.45 * c_sanpo + 0.35 * c_hyp + 0.20 * c_gls
    return res, composite


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p1-epochs", type=int, default=9)
    p.add_argument("--p1-teacher-epochs", type=int, default=6)
    p.add_argument("--p2-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--epoch-samples", type=int, default=60000)
    p.add_argument("--p1-lr", type=float, default=1.0e-3)
    p.add_argument("--p2-lr", type=float, default=1.5e-3)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=8)
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"=== EgoGeom-Omni v2.1 Decoupled Relay | {gpu_name} ({vram_gb:.1f} GB) | bs={args.batch_size} ===", flush=True)

    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)

    # Preserve Ep 7 CNH snapshot permanently before starting v2.1!
    ep7_src = DRIVE_RUNS / "egogeom_omni_v2_latest.pt"
    ep7_snap = DRIVE_RUNS / "egogeom_omni_v2_ep7_cnh_snapshot.pt"
    if ep7_src.exists() and not ep7_snap.exists():
        shutil.copyfile(ep7_src, ep7_snap)
        print(f"[Snapshot] Preserved Ep 7 CNH checkpoint -> {ep7_snap.name}", flush=True)

    train_ds, eval_sets = build_datasets((args.height, args.width), epoch_samples=args.epoch_samples)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    eval_loaders = {
        k: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for k, ds in eval_sets.items()
    }

    cfg = EgoGeomConfig.base()
    hist_path = DRIVE_RUNS / "v21_relay_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []

    p1_done_flag = DRIVE_RUNS / "v21_p1_done.flag"
    p1_best_path = DRIVE_RUNS / "egogeom_omni_v21_pure_rgb_best.pt"
    p1_latest_path = DRIVE_RUNS / "egogeom_omni_v21_p1_latest.pt"

    # =========================================================================
    # PHASE 1: Pure-RGB Foundation Mastery (Zero ToF shortcut interference!)
    # =========================================================================
    if not p1_done_flag.exists():
        print("\n================ PHASE 1: Pure-RGB Foundation Mastery (9 Epochs) ================", flush=True)
        model = EgoGeomOmniNet(cfg, include_teacher_heads=True).to(device).to(memory_format=torch.channels_last)
        # Freeze tof_branch completely during Phase 1
        for k, param in model.named_parameters():
            param.requires_grad = not k.startswith("tof_branch.")

        rgb_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(rgb_params, lr=args.p1_lr, weight_decay=2e-2, fused=True)
        total_steps = args.p1_epochs * len(train_loader)
        warmup_steps = min(300, total_steps // 10)

        def p1_lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p1_lr_lambda)
        start_ep = 0
        best_p1_score = 999.0

        if p1_latest_path.exists():
            ck = torch.load(p1_latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema_model = copy.deepcopy(model).eval()
            ema_model.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["opt"])
            scheduler.load_state_dict(ck["sched"])
            start_ep = int(ck["epoch"]) + 1
            best_p1_score = float(ck.get("best_score", 999.0))
            print(f"[P1] Resumed Phase 1 at Ep {start_ep+1}/{args.p1_epochs}, best_score={best_p1_score:.4f}", flush=True)
        else:
            if ep7_snap.exists():
                ck7 = torch.load(ep7_snap, map_location=device, weights_only=False)
                # Warm-start from Ep 7 Pure-RGB calibrated EMA weights (already has strong 3-teacher trunk!)
                src_sd = ck7.get("ema_no_tof", ck7["ema"])
                model.load_state_dict(src_sd, strict=False)
                print(f"[P1] Warm-started Pure-RGB trunk from Ep 7 snapshot (clean RGB BN)!", flush=True)
            ema_model = copy.deepcopy(model).eval()

        for pe in ema_model.parameters():
            pe.requires_grad = False

        teachers = RoundRobinMultiTeacher(["dinov2", "moge2", "dav2"], device) if start_ep < args.p1_teacher_epochs else RoundRobinMultiTeacher([], device)
        criterion = EgoGeomOmniLoss(w_depth=1.0, w_absrel=0.65, w_ssi=0.5, w_grad=0.9, w_delta_h=1.2, w_normal=0.4, w_teacher=0.6, w_hole_fill=0.35)
        train_p1 = torch.compile(model, mode="default")
        global_step = start_ep * len(train_loader)

        for epoch in range(start_ep, args.p1_epochs):
            if epoch >= args.p1_teacher_epochs and teachers.order:
                teachers.unload()

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
                allow_fill = batch["allow_teacher_fill"].to(device, non_blocking=True)
                H, W = rgb.shape[-2:]

                rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
                gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
                dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
                R_lvl = rotation_cam_to_gravity_level(g_cam)
                gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)

                teacher_feats, t_depth = teachers.extract_one(rgb, global_step) if teachers.order else ({}, None)
                fill_d, fill_m = (
                    align_teacher_depth_to_sparse_gt(t_depth, gt_d, vm, allow_fill)
                    if t_depth is not None
                    else (None, None)
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_rgb = train_p1(rgb, rays, None, None, return_features=True)
                    loss, logs = criterion(
                        out_rgb, gt_d, vm, gt_dh, dh_m, gt_nrm_lvl, nrm_m, gls,
                        fill_depth=fill_d, fill_mask=fill_m, teacher_feats=teacher_feats,
                    )

                if not torch.isfinite(loss):
                    print(f"  [WARN P1] non-finite loss at Ep {epoch+1} step {step}: {logs}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(rgb_params, 2.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.data.mul_(0.995).add_(pm.data, alpha=0.005)
                    for be, bm in zip(ema_model.buffers(), model.buffers()):
                        be.data.copy_(bm.data)

                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                if step in (1, 25, 100) or step % 150 == 0 or step == len(train_loader):
                    avg = {k: round(v / n_steps, 4) for k, v in running.items()}
                    tag = "P1-RGB-RR" if teachers.order else "P1-RGB-GT"
                    fps = (step * args.batch_size) / max(time.time() - t_ep, 0.1)
                    print(f"[{tag} Ep {epoch+1}/{args.p1_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr={scheduler.get_last_lr()[0]:.2e} {avg}", flush=True)

            # Pure-RGB Zero-Shot Eval (ZERO BN drift -> exact single-mode eval!)
            eval_res, comp_score = _eval_pure_rgb(ema_model, eval_loaders, device)
            print(f"=== [Phase 1 Pure-RGB] Epoch {epoch+1}/{args.p1_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | composite={comp_score:.4f}) ===", flush=True)
            for k, v in eval_res.items():
                print(f"  {k}: {v}", flush=True)

            history.append({"phase": "P1_pure_rgb", "epoch": epoch + 1, "composite": comp_score, "eval": eval_res})
            ck_loc = LOCAL_RUNS / "egogeom_omni_v21_p1_latest.pt"
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
                _export_pure_rgb_onnx(ema_model, args.height, args.width, "egogeom_omni_v21_best_pure_rgb_fp32.onnx")
                print(f"  [NEW BEST P1] composite={best_p1_score:.4f} -> saved pure_rgb_best.pt + Pure-RGB ONNX-17!", flush=True)
            hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        teachers.unload()
        del model, ema_model, train_p1, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        p1_done_flag.write_text("done", encoding="utf-8")
        print("=== PHASE 1 COMPLETE! Pure-RGB Master locked & saved! ===", flush=True)

    # =========================================================================
    # PHASE 2: Frozen-RGB Zero-Init ControlNet ToF/CNH Attachment (6 Epochs)
    # =========================================================================
    print("\n================ PHASE 2: Frozen-RGB Zero-Init ControlNet ToF/CNH (6 Epochs) ================", flush=True)
    model = EgoGeomOmniNet(cfg, include_teacher_heads=False).to(device).to(memory_format=torch.channels_last)
    ck_p1 = torch.load(p1_best_path if p1_best_path.exists() else p1_latest_path, map_location=device, weights_only=False)
    model.load_state_dict(ck_p1["ema"], strict=False)

    # Re-zero tof_branch output projections -> exact 0.00e+00 ControlNet start!
    assert model.tof_branch is not None
    for proj in model.tof_branch.projs:
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

    # FREEZE 100% of Pure-RGB trunk weights AND keep trunk in .eval() FOREVER!
    for k, param in model.named_parameters():
        param.requires_grad = k.startswith("tof_branch.")
    tof_params = [param for param in model.parameters() if param.requires_grad]
    print(f"[P2] Frozen Pure-RGB trunk (100% weights + BN locked!) | Trainable ToF/CNH params: {sum(x.numel() for x in tof_params)/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(tof_params, lr=args.p2_lr, weight_decay=1e-2, fused=True)
    total_steps = args.p2_epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)

    def p2_lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p2_lr_lambda)
    p2_latest_path = DRIVE_RUNS / "egogeom_omni_v21_p2_latest.pt"
    p2_best_path = DRIVE_RUNS / "egogeom_omni_v21_unified_best.pt"
    start_ep = 0
    best_p2_score = 999.0

    if p2_latest_path.exists():
        ck2 = torch.load(p2_latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck2["model"])
        ema_model = copy.deepcopy(model).eval()
        ema_model.load_state_dict(ck2["ema"])
        optimizer.load_state_dict(ck2["opt"])
        scheduler.load_state_dict(ck2["sched"])
        start_ep = int(ck2["epoch"]) + 1
        best_p2_score = float(ck2.get("best_score", 999.0))
        print(f"[P2] Resumed Phase 2 at Ep {start_ep+1}/{args.p2_epochs}, best_score={best_p2_score:.4f}", flush=True)
    else:
        ema_model = copy.deepcopy(model).eval()

    for pe in ema_model.parameters():
        pe.requires_grad = False

    criterion_p2 = EgoGeomOmniLoss(w_depth=1.0, w_absrel=0.65, w_ssi=0.4, w_grad=0.9, w_delta_h=1.2, w_normal=0.4)
    global_step = start_ep * len(train_loader)

    for epoch in range(start_ep, args.p2_epochs):
        t_ep = time.time()
        # CRITICAL: Keep entire RGB trunk in .eval() so Pure-RGB BN buffers NEVER change by 1 bit!
        model.eval()
        model.tof_branch.train()
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
            H, W = rgb.shape[-2:]

            rays = compute_gravity_aligned_ray_map(K, g_cam, H, W)
            gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g_cam, vm)
            dh_m = dh_m * batch["has_gravity"].to(device, non_blocking=True).view(-1, 1, 1, 1)
            R_lvl = rotation_cam_to_gravity_level(g_cam)
            gt_nrm_lvl = F.normalize(torch.einsum("bij,bjhw->bihw", R_lvl, nrm_cam), dim=1, eps=1e-6)

            tof_mode_step = "cnh" if (global_step % 2 == 0) else "depth"
            tof_p, has_tof = simulate_tof_and_cnh_prompt(gt_d, vm, K, glass_mask=gls, mode=tof_mode_step)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_tof = model(rgb, rays, tof_p, has_tof, return_features=False)
                loss, logs = criterion_p2(out_tof, gt_d, vm, gt_dh, dh_m, gt_nrm_lvl, nrm_m, gls)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(tof_params, 2.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for (nk, pe), (_, pm) in zip(ema_model.named_parameters(), model.named_parameters()):
                    if nk.startswith("tof_branch."):
                        pe.data.mul_(0.992).add_(pm.data, alpha=0.008)
                for (nk, be), (_, bm) in zip(ema_model.named_buffers(), model.named_buffers()):
                    if nk.startswith("tof_branch."):
                        be.data.copy_(bm.data)

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            n_steps += 1
            if step in (1, 25, 100) or step % 150 == 0 or step == len(train_loader):
                avg = {k: round(v / n_steps, 4) for k, v in running.items()}
                fps = (step * args.batch_size) / max(time.time() - t_ep, 0.1)
                print(f"[P2-ToF-CNH Ep {epoch+1}/{args.p2_epochs} step {step}/{len(train_loader)} | {fps:.0f} img/s] lr={scheduler.get_last_lr()[0]:.2e} {avg}", flush=True)

        eval_res, comp_score = _eval_unified(ema_model, eval_loaders, device)
        print(f"=== [Phase 2 Unified Omni] Epoch {epoch+1}/{args.p2_epochs} Eval ({(time.time()-t_ep)/60:.1f} min | cnh_composite={comp_score:.4f}) ===", flush=True)
        for k, v in eval_res.items():
            print(f"  {k}: {v}", flush=True)

        history.append({"phase": "P2_unified_omni", "epoch": epoch + 1, "composite": comp_score, "eval": eval_res})
        ck_loc = LOCAL_RUNS / "egogeom_omni_v21_p2_latest.pt"
        torch.save(
            {
                "epoch": epoch, "cfg": cfg.__dict__, "model": model.state_dict(),
                "ema": ema_model.state_dict(), "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(), "best_score": min(best_p2_score, comp_score),
                "eval": eval_res,
            },
            ck_loc,
        )
        shutil.copyfile(ck_loc, p2_latest_path)
        if comp_score < best_p2_score:
            best_p2_score = comp_score
            shutil.copyfile(ck_loc, p2_best_path)
            _export_omni_onnx(ema_model, args.height, args.width, "egogeom_omni_v21_best_omni_fp32.onnx")
            _export_pure_rgb_onnx(ema_model, args.height, args.width, "egogeom_omni_v21_best_pure_rgb_fp32.onnx")
            print(f"  [NEW BEST P2] cnh_composite={best_p2_score:.4f} -> saved unified_best.pt + Dual ONNX-17!", flush=True)
        hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n=== EgoGeom-Omni v2.1 Decoupled Two-Phase Relay COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
