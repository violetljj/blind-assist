"""Comprehensive Unit Tests for EgoGeom-Omni v2 (Physics + Thin-Pole + Slope-Safe delta_h + Dual ONNX-17)."""
from __future__ import annotations

import copy
import io
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogeom_dataset import apply_glasses_augment, downsample_depth_preserve_thin
from egogeom_losses_and_physics import (
    EgoGeomOmniLoss,
    align_teacher_depth_to_sparse_gt,
    compute_gravity_aligned_ray_map,
    compute_local_ground_height_residual,
    simulate_tof_and_cnh_prompt,
)
from egogeom_omni_model import (
    EgoGeomConfig,
    EgoGeomOmniExportWrapper,
    EgoGeomOmniNet,
    EgoGeomOmniPureRGBExportWrapper,
)


def test_jump_aware_thin_pole_downsampling() -> None:
    """Verify 2-px thin vertical pole at 1.5 m against 5.0 m background survives 1024x768 -> 288x384 100%."""
    depth = np.full((768, 1024), 5.0, dtype=np.float32)
    depth[100:700, 511:513] = 1.5  # 2-pixel-wide thin steel pole!
    ds = downsample_depth_preserve_thin(depth, (288, 384), jump_m=0.30)
    pole_px = int((ds < 2.0).sum())
    assert pole_px >= 200, f"Expected thin pole to survive 100%, got only {pole_px} px!"
    print(f"  [PASS] Jump-Aware Thin-Pole Downsample: 2-px pole survived with {pole_px} px at 288x384!")


def test_slope_safe_local_ground_plane() -> None:
    """Verify 6-deg uphill ramp yields delta_h ~ 0.00 m while a 10 cm wheel stop ON the ramp reads ~+0.10 m!"""
    H, W = 288, 384
    K = torch.tensor([[[280.0, 0.0, W / 2.0], [0.0, 280.0, H / 2.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    g = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)  # level camera (+Y down)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32) + 0.5, torch.arange(W, dtype=torch.float32) + 0.5, indexing="ij")
    xn = (u - W / 2.0) / 280.0
    yn = (v - H / 2.0) / 280.0  # >0 below horizon

    # Plane y_up = -1.50 + tan(6 deg) * z  =>  Y_c = -y_up = 1.50 - 0.1051 * z = yn * z
    # => z = 1.50 / (yn + 0.1051)
    slope = math.tan(math.radians(6.0))
    denom = yn + slope
    z_ramp = torch.where(denom > 0.15, 1.50 / denom, torch.full_like(denom, 8.0)).clamp(0.4, 8.0)

    # Place a 10 cm wheel stop on the ramp at ~2.7 m ahead (inside level camera FOV)
    z_scene = z_ramp.clone()
    ws_mask = (z_ramp > 2.60) & (z_ramp < 2.90) & (xn.abs() < 0.25)
    assert ws_mask.sum().item() > 20
    # Raising surface by +0.10 m => Y_c = 1.40 - slope * z => z_ws = 1.40 / denom
    z_scene = torch.where(ws_mask, (1.40 / denom).clamp(0.4, 8.0), z_scene)

    gt_d = z_scene.unsqueeze(0).unsqueeze(0)
    vm = torch.ones_like(gt_d)
    dh, dh_m = compute_local_ground_height_residual(gt_d, K, g, vm)

    ramp_only = (z_ramp > 2.45) & (z_ramp < 4.2) & (~ws_mask)
    ramp_err_cm = float(dh[0, 0][ramp_only].abs().mean().item() * 100.0)
    ws_h_cm = float(dh[0, 0][ws_mask].median().item() * 100.0)
    assert ramp_err_cm < 1.5, f"Ramp should be ~0 cm, got {ramp_err_cm:.2f} cm"
    assert 8.0 < ws_h_cm < 12.0, f"10 cm wheel stop on 6-deg ramp should read ~10 cm, got {ws_h_cm:.2f} cm"
    print(f"  [PASS] Slope-Safe Ground Plane: 6-deg ramp residual={ramp_err_cm:.2f} cm | 10cm wheel stop={ws_h_cm:.2f} cm!")


def test_isotropic_3d_homography() -> None:
    """Verify 1:1 Aria-like input keeps fx == fy after isotropic crop + 3D head-scan homography."""
    rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dep = np.full((512, 512), 2.5, dtype=np.float32)
    val = np.ones((512, 512), dtype=np.float32)
    K = np.array([[300.0, 0.0, 256.0], [0.0, 300.0, 256.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _, _, _, K_out, g_out, _, _ = apply_glasses_augment(rgb, dep, val, K, g, None, None, (288, 384), is_train=True)
    ratio = abs(K_out[0, 0] / K_out[1, 1] - 1.0)
    assert ratio < 0.02, f"Expected isotropic fx == fy, got fx={K_out[0,0]:.2f}, fy={K_out[1,1]:.2f}"
    assert abs(float(np.linalg.norm(g_out)) - 1.0) < 1e-4
    print(f"  [PASS] Isotropic 3D Homography: fx/fy={K_out[0,0]/K_out[1,1]:.4f}, |g|={np.linalg.norm(g_out):.4f}!")


def test_model_and_dual_onnx() -> None:
    torch.manual_seed(42)
    net = EgoGeomOmniNet(EgoGeomConfig.base(), include_teacher_heads=True)
    n_tot = sum(p.numel() for p in net.parameters())
    n_inf = sum(
        p.numel() for k, p in net.named_parameters()
        if not k.startswith(("proj_dinov2", "proj_moge2", "proj_dav2"))
    )

    B, H, W = 2, 288, 384
    rgb = torch.randn(B, 3, H, W)
    K = torch.tensor([[[280.0, 0.0, 192.0], [0.0, 280.0, 144.0], [0.0, 0.0, 1.0]]] * B)
    g = torch.tensor([[0.0, 0.96, 0.28], [0.05, 0.99, 0.10]])
    rays = compute_gravity_aligned_ray_map(K, g, H, W)
    gt_d = torch.rand(B, 1, H, W) * 4.0 + 0.4
    vm = torch.ones(B, 1, H, W)
    gt_dh, dh_m = compute_local_ground_height_residual(gt_d, K, g, vm)
    tof_p, has_tof = simulate_tof_and_cnh_prompt(gt_d, vm, K, mode="cnh")
    assert tof_p.shape == (B, 20, H // 4, W // 4)

    # 1. Bit-exact zero interference check: None == masked-0 == Pure-RGB wrapper!
    net.eval()
    exp_rgb = EgoGeomOmniPureRGBExportWrapper(copy.deepcopy(net))
    with torch.no_grad():
        o_none = net(rgb, rays, None, None)
        o_m0 = net(rgb, rays, tof_p, torch.zeros_like(has_tof))
        d_pure, dh_pure, _, _ = exp_rgb(rgb, rays)
    diff_m0 = (o_none["depth_m"] - o_m0["depth_m"]).abs().max().item()
    diff_pure = (o_none["depth_m"] - d_pure).abs().max().item()
    assert diff_m0 == 0.0 and diff_pure == 0.0, f"Expected exact zero diff, got {diff_m0}, {diff_pure}"

    # 2. Full 2-pass backward + aligned teacher hole-filling
    net.train()
    o_tof = net(rgb, rays, tof_p, has_tof, return_features=True)
    o_rgb = net(rgb, rays, torch.zeros_like(tof_p), torch.zeros_like(has_tof), return_features=True)
    fill_d, fill_m = align_teacher_depth_to_sparse_gt(gt_d * 0.5, gt_d, vm * (torch.rand_like(vm) > 0.2).float(), torch.tensor([1.0, 0.0]))
    crit = EgoGeomOmniLoss()
    loss, logs = crit(o_rgb, gt_d, vm, gt_dh, dh_m, fill_depth=fill_d, fill_mask=fill_m, pred_tof_teacher=o_tof)
    loss.backward()
    assert torch.isfinite(loss)

    # 3. Dual Static ONNX-17 Export Check
    net.eval()
    b_omni, b_rgb = io.BytesIO(), io.BytesIO()
    torch.onnx.export(
        EgoGeomOmniExportWrapper(copy.deepcopy(net)), (rgb[:1], rays[:1], tof_p[:1], has_tof[:1]), b_omni,
        opset_version=17, input_names=["rgb", "grav_rays", "tof_grid_s4", "has_tof"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"], dynamo=False,
    )
    torch.onnx.export(
        EgoGeomOmniPureRGBExportWrapper(copy.deepcopy(net)), (rgb[:1], rays[:1]), b_rgb,
        opset_version=17, input_names=["rgb", "grav_rays"],
        output_names=["depth_m", "delta_h_m", "normal_grav", "log_sigma"], dynamo=False,
    )
    print(
        f"  [PASS] EgoGeom-Omni v2 Model & Dual ONNX-17:\n"
        f"         params_total={n_tot/1e6:.2f}M | params_infer={n_inf/1e6:.2f}M\n"
        f"         ONNX FP32: Omni={len(b_omni.getvalue())/1e6:.2f} MB | Pure-RGB={len(b_rgb.getvalue())/1e6:.2f} MB (FP16 ~{len(b_rgb.getvalue())/2e6:.1f} MB)\n"
        f"         Zero-Interference diff={diff_m0:.2e} | PureRGB-vs-Omni0 diff={diff_pure:.2e}\n"
        f"         loss_logs={logs}"
    )


if __name__ == "__main__":
    print("=== Running EgoGeom-Omni v2 Full Verification Suite ===")
    test_jump_aware_thin_pole_downsampling()
    test_slope_safe_local_ground_plane()
    test_isotropic_3d_homography()
    test_model_and_dual_onnx()
    print("=== ALL v2 TESTS PASSED! ===")
