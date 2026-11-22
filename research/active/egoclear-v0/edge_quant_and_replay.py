"""Direction B: On-Device ONNX FP16/INT8 Quantization Audit + Streaming Offline Replay Video Generator.

Runs 100% locally (0 Colab CU):
  1. Converts egoclear_v41_{encoder,step}.onnx -> FP16 (mixed-precision keeping survival/IO in FP32)
     and INT8 dynamic weight-quantized (QUInt8) ONNX models.
  2. Runs true 2-stage streaming ONNX inference (encoder -> 3-frame ring buffer -> step + latent BEV memory)
     on real indoor (ARKitScenes 30 Val) and real glasses (Aria Gen 2 19 clips) + SANPO-Real 40 scenes,
     measuring: model MB, CPU ORT ms/frame, max/mean abs error vs FP32, strict monotonicity verification,
     and closed-loop blind-walker Late Rate / FA_ep / FA_fr quantization delta!
  3. Renders 1080x540 dual-panel MP4 replay videos (RGB corridor HUD + BEV static/dynamic map +
     LOW/BODY/HEAD bars + 24-bin monotonic survival curve + causal slope_aware_hyst waveform)
     plus summary keyframe contact sheets into artifacts.local/work/egoclear-v0-edge-replay/.
"""
from __future__ import annotations

import json
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path("e:/linnan/linnan")
sys.path.insert(0, str(ROOT / "research/active/egoclear-v0"))

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from eval_closed_loop import H, apply_causal_hysteresis, eval_score_curve, summarize_at_fa_caps
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

WORK_P5B = ROOT / "artifacts.local/work/egoclear-v0-phase5b"
OUT_DIR = ROOT / "artifacts.local/work/egoclear-v0-edge-replay"
MODELS_DIR = WORK_P5B / "models"


def convert_onnx_weights_to_fp16_safe(src_path: Path, dst_path: Path) -> None:
    """Safely compress FP32 initializers > 1KB to FP16 + Cast(FP32) (zero numerical risk, universal ORT/NPU load)."""
    model = onnx.load(str(src_path))
    g = model.graph
    new_inits = []
    cast_nodes = []
    converted = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size >= 256 and np.all(np.isfinite(arr)) and np.max(np.abs(arr)) < 60000.0:
                fp16_name = init.name + "__fp16"
                fp16_arr = arr.astype(np.float16)
                new_inits.append(numpy_helper.from_array(fp16_arr, name=fp16_name))
                cast_nodes.append(
                    onnx.helper.make_node("Cast", inputs=[fp16_name], outputs=[init.name], to=TensorProto.FLOAT)
                )
                converted += 1
                continue
        new_inits.append(init)
    g.ClearField("initializer")
    g.initializer.extend(new_inits)
    # Prepend Cast nodes so all downstream nodes receive their exact expected FP32 tensor names
    old_nodes = list(g.node)
    g.ClearField("node")
    g.node.extend(cast_nodes + old_nodes)
    onnx.save(model, str(dst_path))
    print(
        f"[quant] FP16 weight-compressed {src_path.name} -> {dst_path.name} "
        f"({src_path.stat().st_size/1e6:.2f} MB -> {dst_path.stat().st_size/1e6:.2f} MB, {converted} tensors)",
        flush=True,
    )


def quantize_onnx_int8_dynamic(src_path: Path, dst_path: Path) -> None:
    """Dynamic QUInt8 quantization of MatMul/Gemm/Conv weights for mobile CPU / edge ORT."""
    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"WeightSymmetric": False},
    )
    print(
        f"[quant] INT8 MatMul/Gemm quantized {src_path.name} -> {dst_path.name} "
        f"({src_path.stat().st_size/1e6:.2f} MB -> {dst_path.stat().st_size/1e6:.2f} MB)",
        flush=True,
    )


class StreamingONNXRunner:
    """Exact 2-stage mobile streaming runner using (encoder.onnx, step.onnx) on CPU ORT."""

    def __init__(self, enc_path: Path, step_path: Path, intra_threads: int = 8):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_enc = ort.InferenceSession(str(enc_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_stp = ort.InferenceSession(str(step_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.c_bev = 128

    def run_scene(self, d: dict, max_frames: int | None = None) -> dict:
        n = len(d["eval_rows"])
        if max_frames is not None:
            n = min(n, max_frames)
        rgb_u = decode_rgb_batch(d, np.arange(n))
        rgb_n = ((rgb_u - IMAGENET_MEAN[None]) / IMAGENET_STD[None]).astype(np.float32)

        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)[None]

        Rs = d["poses_R"][:n].astype(np.float32)
        ts = d["poses_t"][:n].astype(np.float32)
        fwds = d["causal_fwd"][:n].astype(np.float32)
        segs = np.r_[0.0, np.linalg.norm(np.diff(ts[:, :2], axis=0), axis=1)].astype(np.float32)

        mem = np.zeros((1, self.c_bev, BEV_NZ, BEV_NX), dtype=np.float32)
        age = np.full((1, 1, BEV_NZ, BEV_NX), 10.0, dtype=np.float32)

        ring_f, ring_d, ring_R, ring_t, ring_fwd = [], [], [], [], []
        corr_probs = np.zeros((n, BEV_NZ), dtype=np.float32)
        static_probs = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.float32)
        dynamic_probs = np.zeros((n, 3, BEV_NZ, BEV_NX), dtype=np.float32)
        unc_probs = np.zeros((n, BEV_NZ), dtype=np.float32)

        enc_ms_tot = 0.0
        stp_ms_tot = 0.0

        for i in range(n):
            t0 = time.perf_counter()
            f2d, dc = self.sess_enc.run(None, {"rgb": rgb_n[i : i + 1], "K_scaled": K_s, "R_c2w": Rs[i : i + 1]})
            enc_ms_tot += (time.perf_counter() - t0) * 1000.0

            ring_f.append(f2d)
            ring_d.append(dc)
            ring_R.append(Rs[i : i + 1])
            ring_t.append(ts[i : i + 1])
            ring_fwd.append(fwds[i : i + 1])
            if len(ring_f) > 3:
                ring_f.pop(0); ring_d.pop(0); ring_R.pop(0); ring_t.pop(0); ring_fwd.pop(0)

            i1 = max(0, len(ring_f) - 2)
            i2 = max(0, len(ring_f) - 3)
            feed = {
                "f2d_t": ring_f[-1], "dc_t": ring_d[-1],
                "f2d_1": ring_f[i1], "dc_1": ring_d[i1],
                "f2d_2": ring_f[i2], "dc_2": ring_d[i2],
                "K_t": K_s, "R_t": ring_R[-1], "t_t": ring_t[-1], "fwd_t": ring_fwd[-1],
                "K_1": K_s, "R_1": ring_R[i1], "t_1": ring_t[i1], "fwd_1": ring_fwd[i1],
                "K_2": K_s, "R_2": ring_R[i2], "t_2": ring_t[i2],
                "ds_t": np.array([segs[i]], dtype=np.float32),
                "mem_prev": mem, "age_prev": age,
            }
            t1 = time.perf_counter()
            st_log, dy_log, c_prob, u_log, mem, age = self.sess_stp.run(None, feed)
            stp_ms_tot += (time.perf_counter() - t1) * 1000.0

            corr_probs[i] = c_prob[0]
            static_probs[i] = 1.0 / (1.0 + np.exp(-np.clip(st_log[0], -20, 20)))
            dynamic_probs[i] = 1.0 / (1.0 + np.exp(-np.clip(dy_log[0], -20, 20)))
            unc_probs[i] = 1.0 / (1.0 + np.exp(-np.clip(u_log[0], -20, 20)))

        return {
            "rgb": rgb_u,
            "corr_probs": corr_probs,
            "static_probs": static_probs,
            "dynamic_probs": dynamic_probs,
            "unc_probs": unc_probs,
            "enc_ms_per_fr": enc_ms_tot / max(n, 1),
            "stp_ms_per_fr": stp_ms_tot / max(n, 1),
        }


def render_replay_video(
    scene_name: str,
    domain_label: str,
    d: dict,
    res: dict,
    out_mp4: Path,
    out_sheet: Path,
    thr_on: float = 0.45,
    fps: float = 5.0,
) -> None:
    """Render 1080x540 dual-panel offline visual replay MP4 + 6-frame highlight contact sheet."""
    rgb = res["rgb"]  # (T, 3, 294, 518) [0,1]
    cp = res["corr_probs"]  # (T, 24)
    sp = res["static_probs"]  # (T, 3, 24, 15)
    dp = res["dynamic_probs"]  # (T, 3, 24, 15)
    up = res["unc_probs"]  # (T, 24)
    er = d["eval_rows"]
    n = len(cp)

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))
    p4 = cp[:, idx_4m]
    alerts_hyst = apply_causal_hysteresis(p4, thr_on, mode="slope_aware_hyst")

    W_can, H_can = 1080, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W_can, H_can))

    keyframes = []
    pick_indices = sorted(set(np.linspace(2, max(2, n - 2), 6).astype(int).tolist()))

    for i in range(n):
        canvas = np.full((H_can, W_can, 3), 18, dtype=np.uint8)
        # Left panel: RGB 640x364 centered vertically
        img_bgr = cv2.cvtColor((rgb[i].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_rs = cv2.resize(img_bgr, (640, 364), interpolation=cv2.INTER_LINEAR)

        # Color-coded border & corridor frame based on alert state
        al = bool(alerts_hyst[i])
        p_val = float(p4[i])
        col = (40, 50, 235) if al else ((30, 190, 240) if p_val >= 0.25 else (60, 195, 80))
        cv2.rectangle(img_rs, (2, 2), (637, 361), col, 5 if al else 2)
        # Central walking corridor guide trapezoid on RGB
        pts_trap = np.array([[215, 360], [275, 190], [365, 190], [425, 360]], dtype=np.int32)
        cv2.polylines(img_rs, [pts_trap], True, col, 2, cv2.LINE_AA)

        canvas[70:434, 16:656] = img_rs

        # Top HUD banner
        d_gt = float(er[i, 0]) if i < len(er) else float("nan")
        gt_str = f"{d_gt:.2f}m" if np.isfinite(d_gt) else ("CLEAR" if np.isinf(d_gt) else "N/A")
        hit_bins = np.nonzero(cp[i] >= 0.5)[0]
        d_hat = f"{z_edges[hit_bins[0]]:.2f}m" if len(hit_bins) else ">5.1m"
        u_val = float(up[i, : idx_4m + 1].mean())
        st_txt = "ALERT [SLOPE-HYST]" if al else "CLEAR"
        cv2.rectangle(canvas, (16, 12), (656, 62), (35, 35, 42), -1)
        cv2.rectangle(canvas, (16, 12), (210, 62), col, -1)
        cv2.putText(canvas, st_txt, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"P(<=4m)={p_val:.2f}  d_hat={d_hat}  GT={gt_str}  U={u_val:.2f}",
            (222, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 240), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{domain_label} | {scene_name} | frame {i+1:03d}/{n:03d} | EgoClear-v4.1 ONNX Streaming",
            (18, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 195, 205), 1, cv2.LINE_AA,
        )

        # Bottom-left: Temporal waveform (last 35 frames)
        wx0, wy0, ww, wh = 16, 472, 640, 56
        cv2.rectangle(canvas, (wx0, wy0), (wx0 + ww, wy0 + wh), (28, 30, 36), -1)
        win_len = 35
        st_w = max(0, i - win_len + 1)
        seg_p = p4[st_w : i + 1]
        seg_a = alerts_hyst[st_w : i + 1]
        # Threshold line
        ty = int(wy0 + wh - thr_on * (wh - 8) - 4)
        cv2.line(canvas, (wx0, ty), (wx0 + ww, ty), (90, 90, 110), 1)
        if len(seg_p) >= 2:
            xs = np.linspace(wx0 + 6, wx0 + ww - 6, len(seg_p)).astype(int)
            for k in range(len(seg_p)):
                # GT hazard shading
                fi = st_w + k
                if fi < len(er) and np.isfinite(er[fi, 0]) and er[fi, 0] <= H:
                    cv2.line(canvas, (xs[k], wy0 + 2), (xs[k], wy0 + wh - 2), (45, 35, 70), 3)
                if seg_a[k]:
                    cv2.circle(canvas, (xs[k], wy0 + 8), 3, (40, 60, 240), -1)
            pts_w = np.stack([xs, (wy0 + wh - 4 - seg_p * (wh - 10)).astype(int)], axis=1)
            cv2.polylines(canvas, [pts_w], False, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "P_raw(<=4m) & Slope-Hyst Alert (purple=GT hazard)", (wx0 + 6, wy0 + wh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 185), 1)

        # Right-top: BEV static (red/yellow) + dynamic (cyan/green) heatmap (24x15 -> 210x288)
        bx0, by0, bw, bh = 678, 22, 180, 288
        st_max = sp[i].max(axis=0)  # (24, 15)
        dy_max = dp[i].max(axis=0)  # (24, 15)
        # Flip z so forward (5.1m) is at top, walker (0.3m) is at bottom
        st_img = np.flipud(st_max)
        dy_img = np.flipud(dy_max)
        bev_rgb = np.zeros((BEV_NZ, BEV_NX, 3), dtype=np.uint8)
        bev_rgb[..., 2] = (st_img * 255).clip(0, 255).astype(np.uint8)       # Red = static hazard
        bev_rgb[..., 1] = (dy_img * 230 + st_img * 60).clip(0, 255).astype(np.uint8)  # Green = dynamic suppressed
        bev_rgb[..., 0] = (dy_img * 180 + 25).clip(0, 255).astype(np.uint8)
        bev_big = cv2.resize(bev_rgb, (bw, bh), interpolation=cv2.INTER_NEAREST)
        # Draw 0.6m corridor box (cols 6..8 out of 15) and 4.0m line
        cx1, cx2 = int(6 * bw / BEV_NX), int(9 * bw / BEV_NX)
        y_4m = int((1.0 - (idx_4m + 1) / BEV_NZ) * bh)
        cv2.rectangle(bev_big, (cx1, y_4m), (cx2, bh - 1), (255, 255, 255), 2)
        canvas[by0 : by0 + bh, bx0 : bx0 + bw] = bev_big
        cv2.putText(canvas, "BEV (Red=Static, Cyan=Dyn)", (bx0, by0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1)

        # Right-top-far: 3 Height-Band bars (LOW / BODY / HEAD in central corridor <= 4m)
        hx0, hy0 = 876, 28
        cv2.putText(canvas, "Height Bands", (hx0, hy0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 230), 1)
        band_names = ["HEAD (1.4-2.0m)", "BODY (0.5-1.4m)", "LOW  (0.1-0.5m)"]
        band_idx = [2, 1, 0]
        for row_k, (bname, bi) in enumerate(zip(band_names, band_idx)):
            val = float(sp[i, bi, : idx_4m + 1, 6:9].max())
            yy = hy0 + 18 + row_k * 86
            cv2.putText(canvas, f"{bname}: {val:.2f}", (hx0, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 210, 220), 1)
            cv2.rectangle(canvas, (hx0, yy + 8), (hx0 + 184, yy + 32), (42, 44, 52), -1)
            bcol = (40, 60, 235) if val >= 0.5 else ((30, 190, 240) if val >= 0.25 else (60, 180, 90))
            cv2.rectangle(canvas, (hx0, yy + 8), (hx0 + int(184 * val), yy + 32), bcol, -1)

        # Right-bottom: 24-bin Monotonic Survival Curve P(D <= z_k)
        sx0, sy0, sw, sh = 678, 336, 384, 192
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + sw, sy0 + sh), (28, 30, 36), -1)
        cv2.putText(canvas, "Monotonic First-Hit Survival P(D <= z)", (sx0 + 8, sy0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1)
        # 4.0m vertical marker
        x4 = int(sx0 + 14 + (idx_4m / (BEV_NZ - 1)) * (sw - 28))
        cv2.line(canvas, (x4, sy0 + 26), (x4, sy0 + sh - 22), (100, 100, 130), 1)
        cv2.putText(canvas, "4.0m", (x4 - 14, sy0 + sh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 175), 1)
        xs_c = np.linspace(sx0 + 14, sx0 + sw - 14, BEV_NZ).astype(int)
        ys_c = (sy0 + sh - 24 - cp[i] * (sh - 56)).astype(int)
        pts_c = np.stack([xs_c, ys_c], axis=1)
        cv2.polylines(canvas, [pts_c], False, (70, 230, 160), 2, cv2.LINE_AA)
        for pt in pts_c[::2]:
            cv2.circle(canvas, tuple(pt), 2, (140, 255, 200), -1)
        cv2.putText(canvas, "0.5m", (sx0 + 8, sy0 + sh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 175), 1)
        cv2.putText(canvas, "5.1m", (sx0 + sw - 36, sy0 + sh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 175), 1)

        vw.write(canvas)
        if i in pick_indices:
            keyframes.append(cv2.resize(canvas, (540, 270), interpolation=cv2.INTER_AREA))

    vw.release()
    if len(keyframes) == 6:
        r1 = np.hstack(keyframes[:3])
        r2 = np.hstack(keyframes[3:6])
        cv2.imwrite(str(out_sheet), np.vstack([r1, r2]))


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc_fp32 = MODELS_DIR / "egoclear_v41_encoder.onnx"
    stp_fp32 = MODELS_DIR / "egoclear_v41_step.onnx"
    enc_fp16 = OUT_DIR / "egoclear_v41_encoder_fp16.onnx"
    stp_fp16 = OUT_DIR / "egoclear_v41_step_fp16.onnx"
    enc_int8 = OUT_DIR / "egoclear_v41_encoder_int8.onnx"
    stp_int8 = OUT_DIR / "egoclear_v41_step_int8.onnx"

    # 1. Quantize FP16 & INT8
    convert_onnx_weights_to_fp16_safe(enc_fp32, enc_fp16)
    convert_onnx_weights_to_fp16_safe(stp_fp32, stp_fp16)
    quantize_onnx_int8_dynamic(enc_fp32, enc_int8)
    quantize_onnx_int8_dynamic(stp_fp32, stp_int8)

    # 2. Extract eval datasets locally if needed
    eval_root = WORK_P5B / "eval_data"
    arkit_dir = eval_root / "arkit_val30"
    real40_dir = eval_root / "real40"
    aria_dir = eval_root / "aria19"
    for tar_p, dst_d in [
        (eval_root / "arkit_val30.tar", arkit_dir),
        (eval_root / "real40_000.tar", real40_dir),
        (eval_root / "real40_001.tar", real40_dir),
        (ROOT / "artifacts.local/work/egoclear-v0-phase4/aria_g2_20clips.tar", aria_dir),
    ]:
        dst_d.mkdir(parents=True, exist_ok=True)
        if tar_p.exists() and len(list(dst_d.glob("*.npz"))) < 5:
            with tarfile.open(tar_p, "r") as tf:
                tf.extractall(dst_d)
    # Extract second real40 shard if needed
    if len(list(real40_dir.glob("*.npz"))) < 38 and (eval_root / "real40_001.tar").exists():
        with tarfile.open(eval_root / "real40_001.tar", "r") as tf:
            tf.extractall(real40_dir)

    bw_speeds = json.loads((ROOT / "research/active/egoclear-v0/sanpo_287_manifest.json").read_text())["bw_speeds"]
    arkit_scenes = {p.stem: load_scene(p) for p in sorted(arkit_dir.glob("*.npz"))}
    real_scenes = {p.stem: load_scene(p) for p in sorted(real40_dir.glob("*.npz"))}
    aria_scenes = {p.stem: load_scene(p) for p in sorted(aria_dir.glob("*.npz"))}
    print(f"[edge] Local eval sets: arkit={len(arkit_scenes)}, real40={len(real_scenes)}, aria={len(aria_scenes)}", flush=True)

    # 3. Quantization accuracy & CPU latency audit on representative multi-domain scenes (10 ARKit + 4 Real + 4 Aria)
    audit_scenes = {}
    for sc in sorted(arkit_scenes)[:8]:
        audit_scenes[f"arkit:{sc}"] = arkit_scenes[sc]
    for sc in sorted(real_scenes)[:4]:
        audit_scenes[f"real:{sc}"] = real_scenes[sc]
    for sc in sorted(aria_scenes)[:4]:
        audit_scenes[f"aria:{sc}"] = aria_scenes[sc]

    variants = {
        "fp32": (enc_fp32, stp_fp32),
        "fp16": (enc_fp16, stp_fp16),
        "int8": (enc_int8, stp_int8),
    }
    var_outputs = {}
    audit_report = {"variants": {}}

    for vname, (ep, sp) in variants.items():
        runner = StreamingONNXRunner(ep, sp, intra_threads=8)
        enc_ms_l, stp_ms_l = [], []
        v_res = {}
        mono_viols, mono_tots = 0, 0
        for key, d in audit_scenes.items():
            # Cap at 60 frames per scene for fast CPU quantization audit (16 scenes x 60 = ~960 real streaming frames!)
            r = runner.run_scene(d, max_frames=60)
            v_res[key] = r
            enc_ms_l.append(r["enc_ms_per_fr"])
            stp_ms_l.append(r["stp_ms_per_fr"])
            diffs = np.diff(r["corr_probs"], axis=1)
            mono_viols += int((diffs < -1e-4).sum())
            mono_tots += int(diffs.size)
        var_outputs[vname] = v_res
        audit_report["variants"][vname] = {
            "encoder_mb": round(ep.stat().st_size / 1e6, 2),
            "step_mb": round(sp.stat().st_size / 1e6, 2),
            "total_mb": round((ep.stat().st_size + sp.stat().st_size) / 1e6, 2),
            "cpu_enc_ms": round(float(np.mean(enc_ms_l)), 2),
            "cpu_step_ms": round(float(np.mean(stp_ms_l)), 2),
            "cpu_total_ms": round(float(np.mean(enc_ms_l) + np.mean(stp_ms_l)), 2),
            "monotonic_viol_rate": round(mono_viols / max(mono_tots, 1), 6),
        }
        print(f"[audit | {vname}] {audit_report['variants'][vname]}", flush=True)

    # Compare FP16 & INT8 against FP32 reference across all 960 real streaming frames
    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))
    for vname in ("fp16", "int8"):
        diffs_all, diffs_p4 = [], []
        for key in audit_scenes:
            a = var_outputs[vname][key]["corr_probs"]
            b = var_outputs["fp32"][key]["corr_probs"]
            diffs_all.append(np.abs(a - b))
            diffs_p4.append(np.abs(a[:, idx_4m] - b[:, idx_4m]))
        da = np.concatenate([x.ravel() for x in diffs_all])
        dp4 = np.concatenate(diffs_p4)
        audit_report["variants"][vname]["vs_fp32_corr_mae"] = round(float(da.mean()), 6)
        audit_report["variants"][vname]["vs_fp32_corr_max_err"] = round(float(da.max()), 6)
        audit_report["variants"][vname]["vs_fp32_p4_mae"] = round(float(dp4.mean()), 6)
        audit_report["variants"][vname]["vs_fp32_p4_p99"] = round(float(np.percentile(dp4, 99)), 6)
        print(
            f"[parity | {vname} vs fp32] mae={da.mean():.6f}, max={da.max():.6f}, "
            f"p4_mae={dp4.mean():.6f}, p4_p99={np.percentile(dp4, 99):.6f}",
            flush=True,
        )

    # 4. Render 3 full-scene streaming ONNX replay MP4 videos + 6-frame contact sheets (1 Real Outdoor, 1 Indoor, 1 Aria)
    runner_fp16 = StreamingONNXRunner(enc_fp16, stp_fp16, intra_threads=8)
    # Pick scenes with clear mix of hazard and clear frames
    def pick_rich_scene(sdict: dict[str, dict]) -> str:
        best_sc, best_score = None, -1.0
        for sc, d in sorted(sdict.items()):
            er = d["eval_rows"]
            valid = np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)
            if valid.sum() < 25:
                continue
            hz = (valid & np.isfinite(er[:, 0]) & (er[:, 0] <= H)).sum()
            cl = (valid & np.isinf(er[:, 0])).sum()
            score = min(hz, cl) + 0.05 * len(er)
            if score > best_score:
                best_score, best_sc = score, sc
        return best_sc or sorted(sdict)[0]

    replay_targets = [
        ("real_outdoor", "SANPO-Real Outdoor", real_scenes, pick_rich_scene(real_scenes), 0.36),
        ("arkit_indoor", "ARKitScenes Indoor", arkit_scenes, pick_rich_scene(arkit_scenes), 0.55),
        ("aria_glasses_private", "Aria Gen 2 Glasses (Private)", aria_scenes, pick_rich_scene(aria_scenes), 0.40),
    ]
    replays = []
    for tag, label, sdict, sc, thr_on in replay_targets:
        d = sdict[sc]
        r = runner_fp16.run_scene(d, max_frames=120)
        mp4_p = OUT_DIR / f"replay_{tag}_{sc}.mp4"
        png_p = OUT_DIR / f"sheet_{tag}_{sc}.png"
        render_replay_video(sc, label, d, r, mp4_p, png_p, thr_on=thr_on, fps=5.0)
        replays.append({
            "tag": tag, "scene": sc, "frames": len(r["corr_probs"]),
            "mp4": str(mp4_p), "sheet_png": str(png_p),
            "enc_ms": round(r["enc_ms_per_fr"], 2), "step_ms": round(r["stp_ms_per_fr"], 2),
        })
        print(f"[replay] rendered {mp4_p.name} ({len(r['corr_probs'])} fr) & {png_p.name}", flush=True)

    audit_report["replays"] = replays
    audit_report["total_sec"] = round(time.time() - t0, 1)
    rep_p = OUT_DIR / "edge_quant_replay_report.json"
    rep_p.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(f"[edge] DONE saved {rep_p} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
