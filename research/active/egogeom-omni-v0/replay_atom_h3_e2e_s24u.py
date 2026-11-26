"""End-to-end replay of REAL synchronized Atom S3R RGB + ST VL53L8CH H3 signed CNH captures
on live Samsung Galaxy S24 Ultra (v4.1 pc_fp16dec_a16, 59.1 MB) -> EgoGeomTemporalBEVV3,
plus packaging the clean deployable v4.1 pc_fp16dec_a16 bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROUTE = Path(__file__).resolve().parent
REPO = ROUTE.parents[2]
M13 = ROUTE / "math_theory/m13_fisher_so_c_w8a8_silicon"
sys.path[:0] = [
    str(ROUTE),
    str(M13),
    str(ROUTE.parent / "dtr-r0/nearfield"),
]

from egogeom_dataset import IMAGENET_MEAN, IMAGENET_STD
from egogeom_omni_v41_model import analytical_host_focal_from_K
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3
from eval_v42_real_h3_replay import pack_real_h3_to_20ch
from cnh_h3_recording_diagnostics import load_recording
from eval_empirical import IO, OUT as M13_OUT, dump, sha

ASSETS = REPO / "artifacts.local/work/egogeom-omni-v0"
CAPTURE = REPO / "artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e"
RUNTIME = "/data/local/tmp/ba_qairt_htp_r0"

H_IN, W_IN = 288, 384
# Atom S3R OV2640/GC0308 ~66 deg HFOV nominal 640x480 pinhole approximation (for rays & focal contract)
ATOM_FX_640 = 490.0
ATOM_FY_640 = 490.0


def build_atom_rays_and_focal(H: int = H_IN, W: int = W_IN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gravity-aligned rays (assuming level handheld mount g=[0,1,0]), K, and analytical host_focal."""
    sx = W / 640.0
    sy = H / 480.0
    fx, fy = ATOM_FX_640 * sx, ATOM_FY_640 * sy
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    u = np.arange(W, dtype=np.float32) + 0.5
    v = np.arange(H, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(u, v)
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    rays = np.stack([x / norm, y / norm, z / norm], axis=0).astype(np.float32)  # [3,H,W]
    hf = np.array([[[[analytical_host_focal_from_K(fx, fy, cx, cy, W, H, 350.0)]]]], dtype=np.float32)
    return rays, K, hf


def pair_segment_frames(seg_dir: Path, stride: int = 5) -> list[dict]:
    """Match each sampled real H3 CNH frame with the nearest strictly-causal Atom JPEG frame."""
    cam_rows = [json.loads(l) for l in (seg_dir / "camera/frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cam_ts = np.array([int(r["host_received_monotonic_ns"]) for r in cam_rows], dtype=np.int64)

    tof_rows = [json.loads(l) for l in (seg_dir / "tof/frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = load_recording(seg_dir / "tof/raw.bin")
    n_tof = min(len(tof_rows), len(rec["H"]))
    tof_20 = pack_real_h3_to_20ch(rec["H"][:n_tof], rec["D"][:n_tof], rec["S"][:n_tof], rec["N"][:n_tof])

    pairs = []
    for idx in range(0, n_tof, stride):
        t_tof = int(tof_rows[idx]["host_received_monotonic_ns"])
        # Nearest causal (or within 20 ms) Atom JPEG
        pos = int(np.searchsorted(cam_ts, t_tof, side="right")) - 1
        if pos < 0:
            pos = 0
        dt_ms = (t_tof - int(cam_ts[pos])) / 1e6
        img_path = seg_dir / "camera" / cam_rows[pos]["filename"]
        if not img_path.is_file():
            continue
        # Center 4x4 firmware valid median (mm) for ground-truth reference
        d_grid = rec["D"][idx].reshape(8, 8)
        s_grid = rec["S"][idx].reshape(8, 8)
        c_valid = d_grid[2:6, 2:6][(s_grid[2:6, 2:6] == 5) | (s_grid[2:6, 2:6] == 9)]
        fw_med_m = float(np.median(c_valid) / 1000.0) if len(c_valid) else float("nan")
        pairs.append({
            "tof_idx": idx,
            "img_path": img_path,
            "dt_ms": round(dt_ms, 2),
            "fw_center_med_m": round(fw_med_m, 4) if np.isfinite(fw_med_m) else None,
            "tof_20ch": tof_20[idx],  # [20,8,8]
        })
    return pairs


def preprocess_atom_rgb(img_path: Path) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter_ns()
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to read {img_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W_IN, H_IN), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
    ms = (time.perf_counter_ns() - t0) * 1e-6
    return x, ms


def get_adb():
    exe = shutil.which("adb.exe") or shutil.which("adb")
    devs = subprocess.check_output([exe, "devices"], text=True, timeout=30)
    online = [s.split()[0] for s in devs.splitlines() if s.endswith("\tdevice")]
    assert len(online) == 1
    serial = online[0]

    def adb(*argv):
        for attempt in range(4):
            p = subprocess.run([exe, "-s", serial, *map(str, argv)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            diag = p.stdout + p.stderr
            if p.returncode == 0 or not any(x in diag for x in ("cannot connect to daemon", "failed to start daemon")):
                break
            time.sleep(1.5 + attempt)
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        return p.stdout.strip()

    return adb, lambda cmd: adb("shell", cmd), serial


def package_deploy_bundle(e2e_summary: dict) -> Path:
    bundle = ASSETS / "deploy_v41_pc_fp16dec_a16"
    bundle.mkdir(parents=True, exist_ok=True)
    src_rgb = M13_OUT / "perchannel_probe/pc_fp16dec_a16_pure_rgb/graph.bin"
    src_omni = M13_OUT / "perchannel_probe/pc_fp16dec_a16_omni/graph.bin"
    dst_rgb = bundle / "egogeom_omni_v41_pure_rgb_v75_pc_fp16dec_a16.bin"
    dst_omni = bundle / "egogeom_omni_v41_omni_v75_pc_fp16dec_a16.bin"
    shutil.copy2(src_rgb, dst_rgb)
    shutil.copy2(src_omni, dst_omni)

    for name, src_cfg in [("htp_rgb.json", IO / "htp_rgb.json"), ("htp_omni.json", IO / "htp_omni.json")]:
        cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
        cfg["graphs"][0]["graph_names"] = ["base"]
        dump(bundle / name, cfg)

    manifest = {
        "bundle": "egogeom_omni_v41_pc_fp16dec_a16_hexagon_v75",
        "target_soc": "SM-S9280 / Snapdragon 8 Gen 3 / Hexagon v75 (QAIRT 2.47)",
        "quantization_profile": "per-channel/per-row W8A16 ConvNeXt encoder + FP16 FPN/ToF/heads (exact 0.000000 m zero-interference)",
        "stage2_engine": "research/active/egogeom-omni-v0/egogeom_temporal_bev_v3.py (EgoGeomTemporalBEVV3)",
        "files": {
            dst_rgb.name: {"bytes": dst_rgb.stat().st_size, "sha256": sha(dst_rgb)},
            dst_omni.name: {"bytes": dst_omni.stat().st_size, "sha256": sha(dst_omni)},
        },
        "verified_s24u_16frame_panel": {
            "pure_rgb_htp_ms": 14.28,
            "pure_rgb_err_pct": 1.229,
            "omni_signed_htp_ms": 17.04,
            "omni_signed_err_pct": 1.011,
            "no_tof_max_abs_m": 0.0,
        },
        "real_atom_h3_e2e_summary": e2e_summary,
    }
    dump(bundle / "manifest.json", manifest)
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6, help="Sample every Nth H3 frame per segment (~128 paired frames total)")
    args = ap.parse_args()

    segments = [
        ("01_background", "01-background-attempt1"),
        ("02_object", "02-object-attempt2"),
        ("03_movement", "03-movement-attempt3"),
        ("04_restored", "04-restored-attempt4"),
    ]
    rays, K, hf = build_atom_rays_and_focal()
    all_pairs = []
    pre_ms = []
    for seg_label, folder in segments:
        ps = pair_segment_frames(CAPTURE / folder, stride=args.stride)
        for p in ps:
            p["seg"] = seg_label
            x, ms = preprocess_atom_rgb(p["img_path"])
            p["rgb_nchw"] = x
            pre_ms.append(ms)
        all_pairs.extend(ps)
        print(f"Paired {seg_label}: {len(ps)} real Atom+H3 frames (median |dt|={np.median([abs(x['dt_ms']) for x in ps]):.1f} ms)", flush=True)

    # Upsample 8x8 ToF 20-ch to [20, 72, 96] using nearest/bilinear matching ONNX contract
    work = ASSETS / "atom_h3_e2e_s24u"
    work.mkdir(parents=True, exist_ok=True)
    rays_p = work / "rays.raw"
    hf_p = work / "hf.raw"
    ht1_p = work / "ht1.raw"
    ht0_p = work / "ht0.raw"
    ztof_p = work / "ztof.raw"
    rays.astype("<f4").tofile(rays_p)
    hf.astype("<f4").tofile(hf_p)
    np.ones((1, 1, 1, 1), dtype="<f4").tofile(ht1_p)
    np.zeros((1, 1, 1, 1), dtype="<f4").tofile(ht0_p)
    np.zeros((1, 20, H_IN // 4, W_IN // 4), dtype="<f4").tofile(ztof_p)

    adb, shell, serial = get_adb()
    assert shell("getprop ro.product.model") == "SM-S9280"
    temp = shell("mktemp -d /data/local/tmp/atom_e2e_XXXXXXXX")
    env_prefix = f"LD_LIBRARY_PATH={RUNTIME} ADSP_LIBRARY_PATH='{RUNTIME};/vendor/lib/rfsa/adsp;/dsp'"

    src_omni = M13_OUT / "perchannel_probe/pc_fp16dec_a16_omni/graph.bin"
    src_rgb = M13_OUT / "perchannel_probe/pc_fp16dec_a16_pure_rgb/graph.bin"
    assert src_omni.is_file() and src_rgb.is_file()

    try:
        for name, lp in [("rays.raw", rays_p), ("hf.raw", hf_p), ("ht1.raw", ht1_p), ("ht0.raw", ht0_p), ("ztof.raw", ztof_p)]:
            adb("push", lp, f"{temp}/{name}")
        adb("push", src_omni, f"{temp}/omni.bin")
        adb("push", src_rgb, f"{temp}/rgb.bin")

        for tag, cfg_src in [("omni", IO / "htp_omni.json"), ("rgb", IO / "htp_rgb.json")]:
            cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
            cfg["graphs"][0]["graph_names"] = ["base"]
            dump(work / f"htp_{tag}.json", cfg)
            adb("push", work / f"htp_{tag}.json", f"{temp}/htp_{tag}.json")
            dump(work / f"ext_{tag}.json", {"backend_extensions": {
                "shared_library_path": f"{RUNTIME}/libQnnHtpNetRunExtensions.so",
                "config_file_path": f"{temp}/htp_{tag}.json",
            }})
            adb("push", work / f"ext_{tag}.json", f"{temp}/ext_{tag}.json")

        # Stage all paired Atom RGB & H3 ToF raw tensors in one tar push for speed
        in_dir = work / "inputs"
        if in_dir.exists():
            shutil.rmtree(in_dir)
        in_dir.mkdir(parents=True)
        omni_lines, zero_lines, rgb_lines = [], [], []
        for i, p in enumerate(all_pairs):
            p["rgb_nchw"].astype("<f4").tofile(in_dir / f"rgb_{i:04d}.raw")
            # Resize 20x8x8 -> 20x72x96 with nearest neighbor (exact 9x12 block per VL53L8CH zone)
            t20 = np.repeat(np.repeat(p["tof_20ch"], 9, axis=1), 12, axis=2).astype("<f4")
            t20.tofile(in_dir / f"tof_{i:04d}.raw")
            omni_lines.append(
                f"rgb:={temp}/inputs/rgb_{i:04d}.raw grav_rays:={temp}/rays.raw "
                f"tof_grid_s4:={temp}/inputs/tof_{i:04d}.raw has_tof:={temp}/ht1.raw host_focal:={temp}/hf.raw"
            )
            zero_lines.append(
                f"rgb:={temp}/inputs/rgb_{i:04d}.raw grav_rays:={temp}/rays.raw "
                f"tof_grid_s4:={temp}/ztof.raw has_tof:={temp}/ht0.raw host_focal:={temp}/hf.raw"
            )
            rgb_lines.append(
                f"rgb:={temp}/inputs/rgb_{i:04d}.raw grav_rays:={temp}/rays.raw host_focal:={temp}/hf.raw"
            )
        (work / "omni_list.txt").write_text("\n".join(omni_lines) + "\n", encoding="utf-8")
        (work / "zero_list.txt").write_text("\n".join(zero_lines) + "\n", encoding="utf-8")
        (work / "rgb_list.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")

        adb("push", in_dir, f"{temp}/inputs")
        for fn in ("omni_list.txt", "zero_list.txt", "rgb_list.txt"):
            adb("push", work / fn, f"{temp}/{fn}")

        N = len(all_pairs)
        timings = {}
        for mode, bin_name, ext_name, list_name in [
            ("omni_h3", "omni.bin", "ext_omni.json", "omni_list.txt"),
            ("omni_zero", "omni.bin", "ext_omni.json", "zero_list.txt"),
            ("pure_rgb", "rgb.bin", "ext_rgb.json", "rgb_list.txt"),
        ]:
            dest = f"{temp}/out_{mode}"
            shell(
                f"{env_prefix} {RUNTIME}/qnn-net-run "
                f"--backend {RUNTIME}/libQnnHtp.so "
                f"--config_file {temp}/{ext_name} "
                f"--retrieve_context {temp}/{bin_name} "
                f"--input_list {temp}/{list_name} "
                f"--output_dir {dest} "
                f"--num_inferences {N} "
                f"--keep_num_outputs {N} "
                f"--perf_profile burst "
                f"--profiling_level basic >/dev/null 2>&1"
            )
            prof = shell(f"{env_prefix} {RUNTIME}/qnn-profile-viewer --input_log {dest}/qnn-profiling-data.log")
            tail = prof.split("Execute Stats (Average):")[-1]
            htp_m = re.search(r"Backend \(Accelerator \(execute\) time\):\s*([\d.]+) us", tail)
            nr_m = re.search(r"NetRun:\s*([\d.]+) us", tail)
            timings[mode] = {
                "frames": N,
                "htp_ms": round(float(htp_m[1]) / 1000.0, 3) if htp_m else None,
                "netrun_ms": round(float(nr_m[1]) / 1000.0, 3) if nr_m else None,
            }
            print(f"S24U live Atom+H3 [{mode}]: {timings[mode]}", flush=True)

        # Pull outputs in tar stream and extract *.raw safely via Python tarfile
        import tarfile
        for mode in ("omni_h3", "omni_zero", "pure_rgb"):
            shell(f"tar -cf {temp}/{mode}.tar -C {temp} out_{mode}")
            tar_p = work / f"{mode}.tar"
            adb("pull", f"{temp}/{mode}.tar", tar_p)
            out_dir = work / f"out_{mode}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            with tarfile.open(tar_p, "r") as tf:
                raw_members = [m for m in tf.getmembers() if m.name.endswith(".raw")]
                tf.extractall(path=work, members=raw_members)
            tar_p.unlink(missing_ok=True)
    finally:
        shell(f"rm -rf {shlex.quote(temp)}")

    # Verify exact chip zero-interference & run EgoGeomTemporalBEVV3 end-to-end
    max_zi = 0.0
    nan_count = 0
    bev_ms = []
    per_seg = {k: {"frames": 0, "dt_ms": [], "fw_m": [], "rgb_roi_m": [], "h3_roi_m": [], "bev_d_m": [], "m6_risk": []} for k, _ in segments}

    # World Z-up camera extrinsics: cam +X(right)->world -Y, cam +Y(down)->world -Z, cam +Z(fwd)->world +X
    R_id = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    t_zero = np.array([0.0, 0.0, 1.4], dtype=np.float32)
    fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cur_seg = None
    bev = None
    for i, p in enumerate(all_pairs):
        if p["seg"] != cur_seg:
            cur_seg = p["seg"]
            bev = EgoGeomTemporalBEVV3()
        d_h3 = np.fromfile(work / f"out_omni_h3/Result_{i}/depth_m.raw", "<f4").reshape(H_IN, W_IN)
        dh_h3 = np.fromfile(work / f"out_omni_h3/Result_{i}/delta_h_m.raw", "<f4").reshape(H_IN, W_IN)
        nrm_h3 = np.fromfile(work / f"out_omni_h3/Result_{i}/normal_grav.raw", "<f4").reshape(3, H_IN, W_IN)
        ls_h3 = np.fromfile(work / f"out_omni_h3/Result_{i}/log_sigma.raw", "<f4").reshape(H_IN, W_IN)

        d_z = np.fromfile(work / f"out_omni_zero/Result_{i}/depth_m.raw", "<f4").reshape(H_IN, W_IN)
        d_rgb = np.fromfile(work / f"out_pure_rgb/Result_{i}/depth_m.raw", "<f4").reshape(H_IN, W_IN)
        max_zi = max(max_zi, float(np.max(np.abs(d_z - d_rgb))))
        nan_count += int((~np.isfinite(d_h3)).sum())

        # Subsample to s4 (72x96) for EgoGeomTemporalBEVV3
        t0 = time.perf_counter_ns()
        d_bev, r_bev, r_hyb, r_m6 = bev.step(
            d_h3[::4, ::4], dh_h3[::4, ::4], nrm_h3[:, ::4, ::4], ls_h3[::4, ::4],
            K, R_id, t_zero, fwd,
        )
        bev_ms.append((time.perf_counter_ns() - t0) * 1e-6)

        rec = per_seg[p["seg"]]
        rec["frames"] += 1
        rec["dt_ms"].append(abs(p["dt_ms"]))
        if p["fw_center_med_m"] is not None:
            rec["fw_m"].append(p["fw_center_med_m"])
        rec["rgb_roi_m"].append(float(d_rgb[72:216, 96:288].mean()))
        rec["h3_roi_m"].append(float(d_h3[72:216, 96:288].mean()))
        if np.isfinite(d_bev):
            rec["bev_d_m"].append(float(d_bev))
        rec["m6_risk"].append(float(r_m6))

    seg_summary = {}
    for seg_label, rec in per_seg.items():
        seg_summary[seg_label] = {
            "frames": rec["frames"],
            "sync_abs_dt_p50_ms": round(float(np.median(rec["dt_ms"])), 2),
            "fw_center_med_m": round(float(np.mean(rec["fw_m"])), 4) if rec["fw_m"] else None,
            "s24u_pure_rgb_roi_mean_m": round(float(np.mean(rec["rgb_roi_m"])), 4),
            "s24u_pure_rgb_roi_std_m": round(float(np.std(rec["rgb_roi_m"])), 4),
            "s24u_omni_h3_roi_mean_m": round(float(np.mean(rec["h3_roi_m"])), 4),
            "s24u_omni_h3_roi_std_m": round(float(np.std(rec["h3_roi_m"])), 4),
            "bev_v3_dist_mean_m": round(float(np.mean(rec["bev_d_m"])), 4) if rec["bev_d_m"] else None,
            "bev_v3_m6_risk_p50": round(float(np.median(rec["m6_risk"])), 4),
        }

    e2e_summary = {
        "capture_root": str(CAPTURE),
        "total_paired_frames": len(all_pairs),
        "nan_count": nan_count,
        "chip_zero_interference_max_abs_m": round(max_zi, 6),
        "pipeline_latency_ms": {
            "host_jpeg_preprocess_p50": round(float(np.median(pre_ms)), 3),
            "s24u_htp_omni_h3_ms": timings["omni_h3"]["htp_ms"],
            "s24u_netrun_omni_h3_ms": timings["omni_h3"]["netrun_ms"],
            "s24u_htp_pure_rgb_ms": timings["pure_rgb"]["htp_ms"],
            "bev_v3_cpu_p50_ms": round(float(np.median(bev_ms)), 3),
            "bev_v3_cpu_p95_ms": round(float(np.percentile(bev_ms, 95)), 3),
            "e2e_total_omni_ms": round(float(np.median(pre_ms)) + timings["omni_h3"]["htp_ms"] + float(np.median(bev_ms)), 2),
        },
        "cabinet_response": {
            "background_fw_m": seg_summary["01_background"]["fw_center_med_m"],
            "background_pure_rgb_m": seg_summary["01_background"]["s24u_pure_rgb_roi_mean_m"],
            "background_omni_h3_m": seg_summary["01_background"]["s24u_omni_h3_roi_mean_m"],
            "obj_insertion_delta_pure_rgb_m": round(
                seg_summary["02_object"]["s24u_pure_rgb_roi_mean_m"] - seg_summary["01_background"]["s24u_pure_rgb_roi_mean_m"], 4
            ),
            "obj_insertion_delta_omni_h3_m": round(
                seg_summary["02_object"]["s24u_omni_h3_roi_mean_m"] - seg_summary["01_background"]["s24u_omni_h3_roi_mean_m"], 4
            ),
            "restored_vs_background_err_omni_h3_m": round(
                abs(seg_summary["04_restored"]["s24u_omni_h3_roi_mean_m"] - seg_summary["01_background"]["s24u_omni_h3_roi_mean_m"]), 4
            ),
        },
        "segments": seg_summary,
    }
    dump(work / "atom_h3_e2e_s24u_report.json", e2e_summary)
    bundle = package_deploy_bundle(e2e_summary)
    print("\n=== REAL ATOM RGB + H3 S24U E2E SUMMARY ===", flush=True)
    print(json.dumps(e2e_summary, indent=2), flush=True)
    print(f"Packaged deploy bundle -> {bundle}", flush=True)


if __name__ == "__main__":
    main()
