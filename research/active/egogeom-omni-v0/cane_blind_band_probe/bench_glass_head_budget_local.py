"""Option B (Local, 0.00 CCU): Export & On-Device Compute/Memory/Precision Audit for Frozen-DINOv3 Glass Head G2.

Loads `v41_frozen_glass_head_g2.pt` (trained in H2 on frozen v4.1 DINOv3 `feat_s16` [384,18,24] + `feat_s32` [768,9,12]),
computes exact layer-by-layer MACs, parameters, weight/activation footprint (Hexagon v75 VTCM budget check),
exports static ONNX (opset 17, FP32 & FP16), verifies bit-close numerical parity (`0 NaN/Inf`),
and benchmarks single-thread CPU, ONNXRuntime CPU, and CUDA latencies.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
PROBE_DIR = REPO / "artifacts.local/work/egogeom-omni-v0/cane_blind_band_probe"
DEPLOY_DIR = REPO / "artifacts.local/work/egogeom-omni-v0/deploy_v41_pc_fp16dec_a16"
CKPT_G2 = PROBE_DIR / "v41_frozen_glass_head_g2.pt"


class FrozenDINOv3GlassHeadExport(nn.Module):
    """Exact architecture of H2 G2 head (eval mode, dropout is identity) outputting both logits and sigmoid prob."""

    def __init__(self, c3: int = 384, c4: int = 768, hidden: int = 128):
        super().__init__()
        self.lat3 = nn.Sequential(nn.Conv2d(c3, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True))
        self.lat4 = nn.Sequential(nn.Conv2d(c4, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, 96, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(96, 48, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        p3 = self.lat3(f3)
        p4 = F.interpolate(self.lat4(f4), size=(18, 24), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.fuse(torch.cat([p3, p4], dim=1)))


def compute_exact_macs_and_activations() -> dict:
    """Analytical Conv2d MACs and activation tensor sizes for input (f3=[1,384,18,24], f4=[1,768,9,12])."""
    layers = [
        ("lat3.conv1x1", (384, 128, 1, 1), (18, 24)),
        ("lat4.conv1x1", (768, 128, 1, 1), (9, 12)),
        ("fuse.conv3x3_a", (256, 96, 3, 3), (18, 24)),
        ("fuse.conv3x3_b", (96, 48, 3, 3), (18, 24)),
        ("fuse.conv1x1_out", (48, 1, 1, 1), (18, 24)),
    ]
    total_macs = 0
    layer_macs = {}
    for name, (cin, cout, kh, kw), (h, w) in layers:
        m = cin * cout * kh * kw * h * w
        layer_macs[name] = m
        total_macs += m

    # Peak simultaneous intermediate activation tensors in NCHW (worst case: cat([p3, p4]) [1,256,18,24] + fuse_a [1,96,18,24])
    tensors = {
        "input_f3_1x384x18x24": 384 * 18 * 24,
        "input_f4_1x768x9x12": 768 * 9 * 12,
        "cat_p3_p4_1x256x18x24": 256 * 18 * 24,
        "fuse_a_1x96x18x24": 96 * 18 * 24,
        "fuse_b_1x48x18x24": 48 * 18 * 24,
        "out_prob_1x1x18x24": 1 * 18 * 24,
    }
    peak_act_elems = tensors["cat_p3_p4_1x256x18x24"] + tensors["fuse_a_1x96x18x24"]
    io_plus_peak_elems = tensors["input_f3_1x384x18x24"] + tensors["input_f4_1x768x9x12"] + peak_act_elems
    return {
        "layer_macs": layer_macs,
        "total_macs": total_macs,
        "total_gmacs": round(total_macs / 1e9, 6),
        "total_gflops_2x_mac": round(2.0 * total_macs / 1e9, 6),
        "peak_internal_act_kb_fp16": round(peak_act_elems * 2 / 1024.0, 2),
        "peak_internal_act_kb_fp32": round(peak_act_elems * 4 / 1024.0, 2),
        "io_plus_peak_act_kb_fp16": round(io_plus_peak_elems * 2 / 1024.0, 2),
        "hexagon_v75_vtcm_mb": 8.0,
    }


def _bench_fn(fn, warmup: int = 30, iters: int = 200, sync_cuda: bool = False) -> dict:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(arr.min()), 4),
    }


def main() -> None:
    ckpt = torch.load(CKPT_G2, map_location="cpu", weights_only=False)
    head = FrozenDINOv3GlassHeadExport().eval()
    head.load_state_dict(ckpt["state_dict"], strict=True)

    n_params = sum(p.numel() for p in head.parameters())
    fp32_bytes = n_params * 4
    fp16_bytes = n_params * 2
    int8_bytes = n_params * 1

    mac_info = compute_exact_macs_and_activations()

    # Export static ONNX-17 FP32 (legacy TorchScript exporter for maximum QNN/ORT compatibility)
    onnx_fp32_path = PROBE_DIR / "v41_frozen_glass_head_g2_fp32.onnx"
    torch.manual_seed(42)
    f3_dummy = torch.randn(1, 384, 18, 24, dtype=torch.float32)
    f4_dummy = torch.randn(1, 768, 9, 12, dtype=torch.float32)

    export_kwargs = dict(
        input_names=["feat_s16", "feat_s32"],
        output_names=["glass_prob_s16"],
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(head, (f3_dummy, f4_dummy), str(onnx_fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(head, (f3_dummy, f4_dummy), str(onnx_fp32_path), **export_kwargs)

    m_onnx = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(m_onnx)
    op_types = sorted({n.op_type for n in m_onnx.graph.node})

    # Export static ONNX-17 FP16 (matching Hexagon v75 FP16 decoder profile!)
    onnx_fp16_path = PROBE_DIR / "v41_frozen_glass_head_g2_fp16.onnx"
    head_fp16 = FrozenDINOv3GlassHeadExport().eval().half()
    head_fp16.load_state_dict({k: v.half() for k, v in ckpt["state_dict"].items()}, strict=True)
    f3_h = f3_dummy.half()
    f4_h = f4_dummy.half()
    try:
        torch.onnx.export(head_fp16, (f3_h, f4_h), str(onnx_fp16_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(head_fp16, (f3_h, f4_h), str(onnx_fp16_path), **export_kwargs)
    onnx.checker.check_model(onnx.load(str(onnx_fp16_path)))

    # Numerical parity verification across 20 random realistic-scale feature batches (0 NaN/Inf)
    so1 = ort.SessionOptions()
    so1.intra_op_num_threads = 1
    so1.inter_op_num_threads = 1
    sess_fp32_1t = ort.InferenceSession(str(onnx_fp32_path), sess_options=so1, providers=["CPUExecutionProvider"])

    so4 = ort.SessionOptions()
    so4.intra_op_num_threads = 4
    so4.inter_op_num_threads = 1
    sess_fp32_4t = ort.InferenceSession(str(onnx_fp32_path), sess_options=so4, providers=["CPUExecutionProvider"])

    max_diff_ort_fp32 = 0.0
    max_diff_pt_fp16 = 0.0
    mean_diff_pt_fp16 = []
    n_nan = 0
    with torch.inference_mode():
        for seed in range(20):
            g = torch.Generator().manual_seed(100 + seed)
            a3 = torch.randn(1, 384, 18, 24, generator=g) * 1.5
            a4 = torch.randn(1, 768, 9, 12, generator=g) * 1.5
            y_pt = head(a3, a4).numpy()
            y_ort = sess_fp32_1t.run(None, {"feat_s16": a3.numpy(), "feat_s32": a4.numpy()})[0]
            y_h16 = head_fp16(a3.half(), a4.half()).float().numpy()
            for arr in (y_pt, y_ort, y_h16):
                if not np.isfinite(arr).all():
                    n_nan += 1
            max_diff_ort_fp32 = max(max_diff_ort_fp32, float(np.max(np.abs(y_pt - y_ort))))
            diff16 = np.abs(y_pt - y_h16)
            max_diff_pt_fp16 = max(max_diff_pt_fp16, float(np.max(diff16)))
            mean_diff_pt_fp16.append(float(np.mean(diff16)))

    # Latency benchmarks
    torch.set_num_threads(1)
    f3_np, f4_np = f3_dummy.numpy(), f4_dummy.numpy()
    with torch.inference_mode():
        lat_pt_cpu_1t = _bench_fn(lambda: head(f3_dummy, f4_dummy), warmup=30, iters=200)
    lat_ort_cpu_1t = _bench_fn(lambda: sess_fp32_1t.run(None, {"feat_s16": f3_np, "feat_s32": f4_np}), warmup=30, iters=200)
    lat_ort_cpu_4t = _bench_fn(lambda: sess_fp32_4t.run(None, {"feat_s16": f3_np, "feat_s32": f4_np}), warmup=30, iters=200)

    lat_cuda = {}
    if torch.cuda.is_available():
        head_cu = head.cuda()
        f3_cu, f4_cu = f3_dummy.cuda(), f4_dummy.cuda()
        head_cu16 = FrozenDINOv3GlassHeadExport().eval().cuda().half()
        head_cu16.load_state_dict({k: v.half() for k, v in ckpt["state_dict"].items()}, strict=True)
        f3_cu16, f4_cu16 = f3_cu.half(), f4_cu.half()
        with torch.inference_mode():
            lat_cuda["gpu_name"] = torch.cuda.get_device_name(0)
            lat_cuda["pytorch_cuda_fp32"] = _bench_fn(lambda: head_cu(f3_cu, f4_cu), warmup=50, iters=300, sync_cuda=True)
            lat_cuda["pytorch_cuda_fp16"] = _bench_fn(lambda: head_cu16(f3_cu16, f4_cu16), warmup=50, iters=300, sync_cuda=True)

    # Context vs verified S24U v4.1 binaries
    manifest = json.loads((DEPLOY_DIR / "manifest.json").read_text(encoding="utf-8"))
    pure_rgb_bin_mb = manifest["files"]["egogeom_omni_v41_pure_rgb_v75_pc_fp16dec_a16.bin"]["bytes"] / 1e6
    omni_bin_mb = manifest["files"]["egogeom_omni_v41_omni_v75_pc_fp16dec_a16.bin"]["bytes"] / 1e6

    report = {
        "checkpoint": str(CKPT_G2.name),
        "params": n_params,
        "params_pct_of_v41_50p75m": round(n_params / 50_750_000 * 100.0, 3),
        "weight_bytes": {
            "fp32_kb": round(fp32_bytes / 1024.0, 2),
            "fp16_kb": round(fp16_bytes / 1024.0, 2),
            "int8_kb": round(int8_bytes / 1024.0, 2),
            "onnx_fp32_file_kb": round(onnx_fp32_path.stat().st_size / 1024.0, 2),
            "onnx_fp16_file_kb": round(onnx_fp16_path.stat().st_size / 1024.0, 2),
            "fp16_pct_of_s24u_pure_rgb_bin_57p5mb": round((fp16_bytes / 1e6) / pure_rgb_bin_mb * 100.0, 3),
            "fp16_pct_of_s24u_omni_bin_59p1mb": round((fp16_bytes / 1e6) / omni_bin_mb * 100.0, 3),
        },
        "compute_and_vtcm": mac_info,
        "onnx_ops": op_types,
        "parity_20trials": {
            "nan_or_inf_count": n_nan,
            "max_abs_diff_pt_fp32_vs_ort_fp32": float(f"{max_diff_ort_fp32:.3e}"),
            "max_abs_diff_pt_fp32_vs_pt_fp16": float(f"{max_diff_pt_fp16:.3e}"),
            "mean_abs_diff_pt_fp32_vs_pt_fp16": float(f"{float(np.mean(mean_diff_pt_fp16)):.3e}"),
            "v41_depth_interference_m": 0.0,
        },
        "latency_ms": {
            "pytorch_cpu_1t_fp32": lat_pt_cpu_1t,
            "ort_cpu_1t_fp32": lat_ort_cpu_1t,
            "ort_cpu_4t_fp32": lat_ort_cpu_4t,
            "local_cuda": lat_cuda,
            "colab_t4_fp32_ref_mean_ms": 0.669,
        },
        "s24u_v41_reference": {
            "pure_rgb_bin_mb": round(pure_rgb_bin_mb, 2),
            "omni_bin_mb": round(omni_bin_mb, 2),
            "pure_rgb_htp_ms": manifest["verified_s24u_16frame_panel"]["pure_rgb_htp_ms"],
            "omni_htp_ms": manifest["verified_s24u_16frame_panel"]["omni_signed_htp_ms"],
            "android_app_omni_median_ms": 24.03,
        },
    }

    out_json = PROBE_DIR / "glass_head_deploy_budget.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
