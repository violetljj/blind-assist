"""Probe the v4 focal factor on the archived NCHW phone input and controlled pinhole rays."""
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
WORK = REPO / "artifacts.local/work/egogeom-omni-v0"
SPANS = (1, 4, 16, 32, 64)


def focal_from_rays(rays, span, dtype):
    encoded = rays.astype(dtype)
    diff = encoded[:, :, span:] - encoded[:, :, :-span]
    norm = np.sqrt(np.sum(diff * diff, axis=0))
    chord = float(norm.max())
    # Exact for a pair bracketing the principal point symmetrically on its row.
    # Preserve the actual frozen one-pixel arithmetic when span=1.
    curvature = np.sqrt(1.0 - .25 * chord**2) if span > 1 else 1.0
    return float(np.clip(350. * chord / (span * curvature), .20, 4.0))


def probe(rays):
    original = focal_from_rays(rays, 1, np.float32)
    rows = []
    for span in SPANS:
        fp32 = focal_from_rays(rays, span, np.float32)
        fp16 = focal_from_rays(rays, span, np.float16)
        rows.append({"span_px": span, "fp32": fp32, "fp16": fp16,
                         "fp16_vs_original_pct": 100 * (fp16 / original - 1),
                         "intrinsic_fp32_change_pct": 100 * (fp32 / original - 1)})
    return rows


def pinhole(fx, fy, cx, cy, rot):
    v, u = np.mgrid[:288, :384] + .5
    r = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(v)))
    r /= np.linalg.norm(r, axis=0)
    return (rot @ r.reshape(3, -1)).reshape(r.shape).astype(np.float32)


def global_scale_control(io, input_rays, factors):
    """Best single multiplier of the archived device depth, evaluated on that same frame."""
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(WORK / "egogeom_omni_v4_best_pure_rgb_fp32.onnx"),
                                   opt, providers=["CPUExecutionProvider"])
    rgb = np.fromfile(io / "rgb.raw", dtype="<f4").reshape(1, 3, 288, 384)
    ref = runtime.run(["depth_m"], {"rgb": rgb, "grav_rays": input_rays[None]})[0].ravel().astype(float)
    gpu = np.fromfile(io / "out_pure_rgb/depth_m.raw", dtype="<f4").astype(float)
    # Minimize mean |c*u_i-1| = mean u_i*|c-1/u_i| by a weighted median.
    u = gpu / ref
    order = np.argsort(1 / u)
    j = np.searchsorted(np.cumsum(u[order]), .5 * u.sum())
    optimum = float(1 / u[order[j]])
    def cost(c):
        return float(100 * np.mean(np.abs(c * u - 1)))
    correction = factors[0]["fp16"] / factors[0]["fp32"]
    return {"n_pixels": len(ref), "archived_absrel_pct": cost(1.),
                "best_posthoc_multiplier": optimum, "best_posthoc_absrel_pct": cost(optimum),
                "hypothetical_focal_correction": correction, "corrected_absrel_pct": cost(correction),
                "precision_scope": "one unclipped global multiplier of archived pure-RGB metric depth"}


def main():
    source = WORK / "qnn_v75_v4_final/grav_rays.raw"
    rays = np.fromfile(source, dtype="<f4").reshape(3, 288, 384)
    assert np.max(np.abs(np.linalg.norm(rays, axis=0) - 1)) < 1e-5
    real = probe(rays)
    rng = np.random.default_rng(293)
    rows = []
    for fx in (150., 230., 350., 700.):
        for _ in range(8):
            rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rot[:, -1] *= np.linalg.det(rot)
            px = rng.uniform(130., 250.)
            py = rng.uniform(80., 200.)
            rows.append(probe(pinhole(fx, fx * 1.08, px, py, rot)))
    controlled = {}
    for i, span in enumerate(SPANS):
        errors = np.abs([row[i]["fp16_vs_original_pct"] for row in rows])
        controlled[span] = {"median_abs_pct": float(np.median(errors)), "max_abs_pct": float(np.max(errors))}
    result = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "raw_archived_observation": real, "controlled_32_camera_rotations": controlled,
                  "oracle_global_scale_control": global_scale_control(source.parent, rays, real),
                  "precision": "host IEEE binary16 rays, differencing, sum of squares and square root; final scalar FP64",
                  "target": "frozen FP32 span-one scalar; isolated factor, no phone intermediate or model intervention"}
    dest = WORK / "math_theory/m5_info_geom_quant_reparam/focal_probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
