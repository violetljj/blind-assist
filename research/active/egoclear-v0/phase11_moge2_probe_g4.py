"""Phase 11 Step 1 (Colab G4): Quantitative Zero-Shot Probe of MoGe-2-ViT-S-Normal (35M, MIT)
across all 4 real-world / zero-shot EgoClear domains (SANPO-Real 40, Aria Gen 2, ARKitScenes 30, UT Austin CODa).

Checks 4 make-or-break scientific hypotheses before launching EgoClear-v7 training:
  1. Architecture & ONNX-17 static-shape exportability for Snapdragon 8 Gen 3 Hexagon v75 NPU.
  2. Cross-domain metric scale stability (median Z_pred / Z_gt & AbsRel across Indoor / Outdoor / 0.7m Robot).
  3. Zero-shot 3D pointmap + gravity-aligned surface-normal (|n_up| < tau) corridor collision performance
     vs UniDepth-L (340M) and EgoClear-v6.6 (57 MB) on exact eval_closed_loop metrics.
  4. Native low-light (0.18x luma + sensor noise) stability to calibrate v7 Clean-to-Degraded KD.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from eval_closed_loop import H, eval_score_curve, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from train_and_eval_g4 import decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_OUT = DRIVE_ROOT / "runs/egoclear_v0_phase11_moge2_probe"
LOCAL_OUT = Path("/content/egoclear_p11_probe")


def ensure_moge_installed():
    try:
        import moge  # noqa: F401
        print("[probe] moge already installed", flush=True)
        return
    except ImportError:
        pass
    print("[probe] Installing MoGe (MIT, Microsoft Research)...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/microsoft/MoGe.git"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print("[probe] MoGe installed.", flush=True)


def extract_tar_dir(drive_dir: Path, local_dir: Path, min_npz: int) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(local_dir.glob("*.npz"))
    if len(existing) >= min_npz:
        return existing
    tars = sorted(drive_dir.glob("*.tar"))
    if tars:
        for tf_path in tars:
            tmp_tar = local_dir / tf_path.name
            shutil.copy2(tf_path, tmp_tar)
            with tarfile.open(tmp_tar, "r") as tf:
                tf.extractall(local_dir, filter="data")
            tmp_tar.unlink(missing_ok=True)
    else:
        for npz in sorted(drive_dir.glob("*.npz")):
            dst = local_dir / npz.name
            if not dst.exists():
                shutil.copy2(npz, dst)
    return sorted(local_dir.glob("*.npz"))


def stage_eval_4domains() -> dict[str, list[Path]]:
    t0 = time.time()
    specs = {
        "real_40": (DRIVE_ROOT / "datasets/sanpo_real_zeroshot_40_v0", Path("/content/sanpo_p2/real_scenes"), 38),
        "arkit_va30": (DRIVE_ROOT / "datasets/arkit_indoor_val30_v0", Path("/content/arkit_p5/val_scenes"), 28),
        "coda": (DRIVE_ROOT / "datasets/coda_zeroshot_v0", Path("/content/coda_zeroshot_v0"), 20),
    }
    out = {}
    def extract_aria() -> list[Path]:
        l_dir = Path("/content/aria_g2_scenes")
        l_dir.mkdir(parents=True, exist_ok=True)
        ex_f = sorted(l_dir.glob("*.npz"))
        if len(ex_f) >= 18:
            return ex_f
        src_tar = DRIVE_ROOT / "caches/aria_g2_20clips.tar"
        tmp_tar = l_dir / src_tar.name
        shutil.copy2(src_tar, tmp_tar)
        with tarfile.open(tmp_tar, "r") as tf:
            tf.extractall(l_dir, filter="data")
        tmp_tar.unlink(missing_ok=True)
        return sorted(l_dir.glob("*.npz"))

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(extract_tar_dir, d_dir, l_dir, mn) for k, (d_dir, l_dir, mn) in specs.items()}
        futs["aria_g2"] = ex.submit(extract_aria)
        for k, fut in futs.items():
            out[k] = fut.result()
    print(
        f"[probe] Staged 4 eval domains in {time.time() - t0:.1f}s: "
        + ", ".join(f"{k}={len(v)}" for k, v in out.items()),
        flush=True,
    )
    return out


def inspect_moge2_arch_and_onnx(model: torch.nn.Module, device: torch.device) -> dict:
    n_total = sum(p.numel() for p in model.parameters())
    n_bb = sum(p.numel() for p in model.backbone.parameters()) if hasattr(model, "backbone") else 0
    submodules = {name: type(mod).__name__ for name, mod in model.named_children()}
    print(f"[probe] MoGe-2 params: total={n_total/1e6:.2f}M, backbone={n_bb/1e6:.2f}M, children={submodules}", flush=True)

    # Measure forward latency at 294x518 and inspect exact output keys/shapes
    dummy = torch.rand(1, 3, RGB_H, RGB_W, device=device, dtype=torch.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward(dummy, num_tokens=777)
    out_shapes = {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    print(f"[probe] forward(1x3x{RGB_H}x{RGB_W}, num_tokens=777) keys/shapes: {out_shapes}", flush=True)

    # Test pure-tensor static ONNX-17 export of forward()
    class StaticMogeWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m.forward(x, num_tokens=777)
            return o["points"], o["normal"], o["mask"]

    onnx_ok = False
    onnx_err = None
    onnx_ops = {}
    onnx_mb = 0.0
    try:
        LOCAL_OUT.mkdir(parents=True, exist_ok=True)
        onnx_path = LOCAL_OUT / "moge2_vits_normal_test.onnx"
        if hasattr(model, "enable_onnx_compatible_mode"):
            model.enable_onnx_compatible_mode()
        elif hasattr(model, "encoder") and hasattr(model.encoder, "backbone"):
            setattr(model.encoder.backbone, "onnx_compatible_mode", True)
        wrap = StaticMogeWrapper(model).eval().float().cpu()
        dx = torch.rand(1, 3, RGB_H, RGB_W, dtype=torch.float32)
        orig_interp = F.interpolate
        def _interp_no_aa(*args, **kwargs):
            kwargs["antialias"] = False
            return orig_interp(*args, **kwargs)
        F.interpolate = _interp_no_aa
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    torch.onnx.export(
                        wrap,
                        (dx,),
                        str(onnx_path),
                        input_names=["rgb"],
                        output_names=["points", "normal", "mask"],
                        opset_version=17,
                        do_constant_folding=True,
                        dynamo=False,
                    )
        finally:
            F.interpolate = orig_interp
        onnx_mb = onnx_path.stat().st_size / (1024 * 1024)
        try:
            import onnx
            gm = onnx.load(str(onnx_path), load_external_data=False)
            for node in gm.graph.node:
                onnx_ops[node.op_type] = onnx_ops.get(node.op_type, 0) + 1
        except Exception:
            pass
        onnx_ok = True
        print(f"[probe] ONNX-17 export PASS! size={onnx_mb:.1f} MB, ops={len(onnx_ops)} unique", flush=True)
    except Exception as e:
        onnx_err = f"{type(e).__name__}: {e}"
        print(f"[probe] ONNX-17 export warning: {onnx_err}", flush=True)
    finally:
        model.to(device).eval()

    return {
        "total_params_m": round(n_total / 1e6, 2),
        "backbone_params_m": round(n_bb / 1e6, 2),
        "children": submodules,
        "forward_shapes": out_shapes,
        "onnx17_export_ok": onnx_ok,
        "onnx17_mb": round(onnx_mb, 2),
        "onnx17_err": onnx_err,
        "onnx17_top_ops": dict(sorted(onnx_ops.items(), key=lambda kv: -kv[1])[:20]),
    }


def score_corridor_from_points_and_normals(
    pts_cam: np.ndarray,      # (H, W, 3) metric camera-frame 3D points (x right, y down, z fwd)
    norm_cam: np.ndarray,     # (H, W, 3) unit surface normals in camera frame
    mask: np.ndarray,         # (H, W) bool/float valid mask
    R_c2w: np.ndarray,        # (3, 3)
    fwd_w: np.ndarray,        # (3,)
    use_normal_filter: bool,
    normal_up_max: float = 0.65,
    h_rel_range: tuple[float, float] = (-1.45, 0.55),
) -> tuple[float, float]:
    """Zero-shot metric 3D corridor first-hit distance D_hat & smooth probability score in [0, 1]."""
    valid = (mask > 0.5) & np.isfinite(pts_cam[..., 2]) & (pts_cam[..., 2] > 0.20) & (pts_cam[..., 2] < 12.0)
    if not np.any(valid):
        return float(np.inf), 0.0

    # Transform to walker corridor frame (rel_w = R_c2w @ pts_cam)
    p = pts_cam[valid]  # (N, 3)
    rel_w = p @ R_c2w.T  # (N, 3)
    right_w = np.array([fwd_w[1], -fwd_w[0], 0.0], dtype=np.float32)
    x_c = rel_w @ right_w
    z_c = rel_w @ fwd_w
    y_c = rel_w[:, 2]  # height relative to camera optical center (Z-up)

    # Estimate local ground level from lowest horizontal points in front strip (for non-normal baseline)
    in_corridor = (np.abs(x_c) <= 0.30) & (z_c >= 0.30) & (z_c <= 5.5)

    if use_normal_filter:
        # Gravity-aligned surface normal verticality: horizontal walkable ground has |n_up| ~ 1.0!
        # Vertical obstacles (walls, poles, bollards, curbs, hanging branches) have |n_up| < normal_up_max!
        n_c = norm_cam[valid]
        n_up = n_c @ R_c2w[2, :]  # world Z component of unit normal
        is_blocker = in_corridor & (np.abs(n_up) < normal_up_max) & (y_c >= h_rel_range[0]) & (y_c <= h_rel_range[1])
    else:
        # Fallback height-only filter without surface normals (uses camera-relative floor percentile)
        strip = (np.abs(x_c) <= 1.2) & (z_c >= 0.8) & (z_c <= 4.5)
        g_est = float(np.percentile(y_c[strip], 5)) if strip.sum() >= 50 else -1.45
        h_g = y_c - g_est
        is_blocker = in_corridor & (h_g >= 0.15) & (h_g <= 1.90)

    zb = z_c[is_blocker]
    if zb.size < 25:
        return float(np.inf), 0.0
    # Robust 5th percentile of blocking points inside ±0.30m corridor -> estimated distance D_hat
    d_hat = float(np.percentile(zb, 5))
    # Convert metric distance into smooth hazard probability matching [0.3, 4.0m] corridor
    # p = sigmoid((4.2 - d_hat) / 0.55), calibrated so d_hat=4.0m -> ~0.59, d_hat=2.5m -> ~0.95, d_hat=5.5m -> ~0.08
    score = float(1.0 / (1.0 + math.exp((d_hat - 4.15) / 0.55)))
    return d_hat, score


@torch.no_grad()
def eval_domain_zeroshot(
    model: torch.nn.Module,
    name: str,
    scene_paths: list[Path],
    bw_speeds: dict,
    device: torch.device,
    override_eval_rows: dict[str, np.ndarray] | None = None,
    darken_factor: float = 1.0,
) -> dict:
    t0 = time.time()
    scenes = {p.stem: load_scene(p) for p in scene_paths}
    eval_rows_map = {
        sc: (override_eval_rows[sc] if (override_eval_rows is not None and sc in override_eval_rows) else d["eval_rows"])
        for sc, d in scenes.items()
    }

    scores_height: dict[str, np.ndarray] = {}
    scores_normal: dict[str, np.ndarray] = {}
    scale_ratios = []
    abs_rels = []
    n_frames_tot = 0

    for sc, d in scenes.items():
        n = len(d["eval_rows"])
        n_frames_tot += n
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        fx_s = K_o[0] * RGB_W / Ww
        fov_x_deg = float(2.0 * math.degrees(math.atan(RGB_W / (2.0 * max(fx_s, 1.0)))))

        rgb_all = decode_rgb_batch(d, np.arange(n))  # (N, 3, H, W) float32 in [0, 1]
        if darken_factor < 0.99:
            rng = np.random.default_rng(42)
            noise = rng.normal(0.0, 0.025, size=rgb_all.shape).astype(np.float32)
            rgb_all = np.clip((rgb_all ** 1.35) * darken_factor + noise, 0.0, 1.0)

        sc_h = np.zeros(n, dtype=np.float32)
        sc_n = np.zeros(n, dtype=np.float32)

        has_gt_d = "depth_s8" in d and d["depth_s8"] is not None
        gt_d_all = d["depth_s8"] if has_gt_d else None

        bs = 16
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            imgs_t = torch.from_numpy(rgb_all[i0:i1]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                res = model.infer(imgs_t, fov_x=fov_x_deg, apply_mask=True)
            pts_np = res["points"].float().cpu().numpy()   # (B, H, W, 3)
            nrm_np = res["normal"].float().cpu().numpy()   # (B, H, W, 3)
            msk_np = res["mask"].float().cpu().numpy()     # (B, H, W)

            for bi in range(i1 - i0):
                fi = i0 + bi
                R = d["poses_R"][fi].astype(np.float32)
                fwd = d["causal_fwd"][fi].astype(np.float32)
                _, sh = score_corridor_from_points_and_normals(pts_np[bi], nrm_np[bi], msk_np[bi], R, fwd, use_normal_filter=False)
                _, sn = score_corridor_from_points_and_normals(pts_np[bi], nrm_np[bi], msk_np[bi], R, fwd, use_normal_filter=True)
                sc_h[fi] = sh
                sc_n[fi] = sn

                if has_gt_d and fi % 4 == 0:
                    gd = gt_d_all[fi].astype(np.float32)
                    pz = cv2.resize(pts_np[bi, :, :, 2], (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_LINEAR)
                    pm = cv2.resize(msk_np[bi], (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
                    ok = pm & np.isfinite(gd) & np.isfinite(pz) & (gd >= 0.5) & (gd <= 8.0) & (pz >= 0.2) & (pz <= 15.0)
                    if ok.sum() >= 30:
                        scale_ratios.append(float(np.median(pz[ok] / gd[ok])))
                        abs_rels.append(float(np.mean(np.abs(pz[ok] - gd[ok]) / gd[ok])))

        scores_height[sc] = sc_h
        scores_normal[sc] = sc_n

    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.95, 0.975, 0.99], dtype=np.float64)

    def summarize_curve(sc_map: dict[str, np.ndarray]):
        curve = eval_score_curve(eval_rows_map, sc_map, bw_speeds, thrs, higher_is_alert=True, hysteresis_mode="raw")
        raw = summarize_at_fa_caps(curve)
        out_s = {}
        for k in ["late_at_fa_fr<=80.0", "late_at_fa_fr<=120.3", "late_at_fa_fr<=200.0", "late_at_fa_fr<=350.0", "late_at_fa_fr<=500.0"]:
            r = raw.get(k)
            if r is not None:
                out_s[k.split("<=")[1]] = {
                    "late_pct": round(r["late_rate"] * 100.0, 1),
                    "thr": round(r["thr"], 3),
                    "ep_km": round(r["fa_ep_per_km"], 1),
                }
        return out_s

    sh_sum = summarize_curve(scores_height)
    sn_sum = summarize_curve(scores_normal)
    wall = time.time() - t0
    metric_scale = {
        "n_checked": len(scale_ratios),
        "median_Z_ratio": round(float(np.median(scale_ratios)), 4) if scale_ratios else None,
        "p25_p75_Z_ratio": [round(float(np.percentile(scale_ratios, 25)), 3), round(float(np.percentile(scale_ratios, 75)), 3)] if scale_ratios else None,
        "mean_AbsRel": round(float(np.mean(abs_rels)), 4) if abs_rels else None,
    }
    print(
        f"[probe:{name}{'_dark' if darken_factor < 0.99 else ''}] frames={n_frames_tot} wall={wall:.1f}s "
        f"scale={metric_scale['median_Z_ratio']} AbsRel={metric_scale['mean_AbsRel']}\n"
        f"   HeightOnly : {sh_sum}\n"
        f"   +GravNormal: {sn_sum}",
        flush=True,
    )
    return {
        "frames": n_frames_tot,
        "wall_sec": round(wall, 1),
        "metric_scale": metric_scale,
        "zeroshot_height_only": sh_sum,
        "zeroshot_gravity_normal": sn_sum,
    }


def main():
    t_start = time.time()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(f"[probe] GPU={torch.cuda.get_device_name(0)}", flush=True)

    ensure_moge_installed()
    from moge.model.v2 import MoGeModel

    print("[probe] Loading Ruicheng/moge-2-vits-normal (35M, MIT)...", flush=True)
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").to(device).eval()

    arch_info = inspect_moge2_arch_and_onnx(model, device)
    dom_paths = stage_eval_4domains()

    # Load exact BlindWays walker speeds manifest from local /content or Drive
    m287_candidates = [
        Path("/content/sanpo_287_manifest.json"),
        DRIVE_ROOT / "datasets/sanpo_synth_287_v0/sanpo_287_manifest.json",
        DRIVE_ROOT / "caches/sanpo_287_manifest.json",
    ]
    m287_path = next((p for p in m287_candidates if p.exists()), None)
    assert m287_path is not None, f"Missing sanpo_287_manifest.json in {m287_candidates}"
    bw_speeds = json.loads(m287_path.read_text())["bw_speeds"]

    real_local_rows = ensure_real_local_gt_cache(Path("/content/sanpo_p2/real_scenes"), Path("/content/sanpo_p2/real_local_gt_cache.npz"))

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "model_id": "Ruicheng/moge-2-vits-normal",
        "arch_and_onnx": arch_info,
        "domains": {},
    }
    for dom in ["real_40", "arkit_va30", "aria_g2", "coda"]:
        ov = real_local_rows if dom == "real_40" else None
        results["domains"][dom] = eval_domain_zeroshot(model, dom, dom_paths[dom], bw_speeds, device, override_eval_rows=ov, darken_factor=1.0)

    # Low-light stability probe on ARKitScenes 30 & SANPO-Real 40 (0.18x luma + sensor noise)
    results["domains"]["arkit_va30_dark018"] = eval_domain_zeroshot(model, "arkit_va30", dom_paths["arkit_va30"], bw_speeds, device, darken_factor=0.18)
    results["domains"]["real_40_dark018"] = eval_domain_zeroshot(model, "real_40", dom_paths["real_40"], bw_speeds, device, override_eval_rows=real_local_rows, darken_factor=0.18)

    results["total_wall_sec"] = round(time.time() - t_start, 1)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    out_local = LOCAL_OUT / "probe_results.json"
    out_drive = DRIVE_OUT / "probe_results.json"
    out_local.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(out_local, out_drive)
    print(f"[probe] DONE in {results['total_wall_sec']}s -> saved {out_drive}", flush=True)


if __name__ == "__main__":
    main()
