"""Focused Colab L4 Runner for SANPO-Real Zeroshot 40 (EgoGeom-Omni v4.1 Pure-RGB -> Compact 1D Bundle).

Stages ONLY `egogeom_omni_v41_pure_rgb_best.pt` and `sanpo_real_zeroshot_40_v0` (~82s on L4),
runs all 8 causal Stage-2 readout arms + height-band GT + residual FA anatomy,
saves `sanpo_real_zs40_1d_bundle.npz` + `sanpo_real_zs40_meta.json`, and backs up to Drive.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import (  # noqa: E402
    EgoGeomOmniV41Net,
    EgoGeomV41Config,
    batch_host_focal_from_K,
)
from run_cane_blind_colab import (  # noqa: E402
    ALL_ARMS,
    CKPT_DIR,
    DATA_DIR,
    H_IN,
    IM_MEAN,
    IM_STD,
    OUT_DIR,
    TAR_DIR,
    W_IN,
    analyze_clip_fa_anatomy,
    build_bev_band_eval_rows,
    build_low_only_bev_mask,
    decode_isotropic_frame,
    extract_tar,
    merge_anatomy,
    run_all_readouts_on_clip,
    run_rclone_copy,
)

DRIVE_BACKUP = "gdrive:BlindAssist/runs/egogeom_cane_blind_band_probe"


def stage_sanpo_only() -> Path:
    t0 = time.time()
    for d in (DATA_DIR, CKPT_DIR, TAR_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if shutil.which("rclone") is None:
        print("[stage] Installing rclone...", flush=True)
        rc = subprocess.run("apt-get update -qq && apt-get install -y -qq rclone", shell=True)
        if rc.returncode != 0 or shutil.which("rclone") is None:
            subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=True)
        print("[stage] rclone ready:", shutil.which("rclone"), flush=True)

    ckpt_path = CKPT_DIR / "egogeom_omni_v41_pure_rgb_best.pt"
    if not ckpt_path.exists():
        print("[stage] Pulling v4.1 pure-RGB checkpoint...", flush=True)
        run_rclone_copy("gdrive:BlindAssist/runs/egogeom_omni_v41/egogeom_omni_v41_pure_rgb_best.pt", CKPT_DIR)

    sanpo_dir = DATA_DIR / "sanpo_real_zs40"
    for sh in ("shard_000.tar", "shard_001.tar"):
        if not (sanpo_dir / f".done_{sh}").exists():
            print(f"[stage] Pulling SANPO {sh}...", flush=True)
            run_rclone_copy(f"gdrive:BlindAssist/datasets/sanpo_real_zeroshot_40_v0/{sh}", TAR_DIR / "sanpo")
            extract_tar(TAR_DIR / "sanpo" / sh, sanpo_dir)

    npzs = sorted(sanpo_dir.rglob("*.npz"))
    print(f"[stage] SANPO ready: {len(npzs)} npz files in {time.time() - t0:.1f}s", flush=True)
    return ckpt_path


def main() -> None:
    t0 = time.time()
    ckpt_path = stage_sanpo_only()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[model] Loading {ckpt_path.name} on {gpu_name}", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**state["cfg"]), pretrained_backbone=False)
    model.tof_branch = None
    model.load_state_dict({k: v for k, v in state["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    del state
    model.to(device).eval()

    npzs = sorted((DATA_DIR / "sanpo_real_zs40").rglob("*.npz"))
    if not npzs:
        raise RuntimeError("No SANPO .npz files found")

    bundle: dict[str, np.ndarray] = {}
    clips_meta: list[dict] = []
    cam_heights: dict[str, list[float]] = defaultdict(list)
    cam_counts: dict[str, int] = defaultdict(int)
    low_only_cnt = 0
    valid_tot = 0
    anatomy_recs: dict[str, list[dict]] = defaultdict(list)

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    v_sub, u_sub = np.indices((H_IN // 4, W_IN // 4), dtype=np.float32)
    u_sub = 4.0 * u_sub + 2.0
    v_sub = 4.0 * v_sub + 2.0

    for ci, p in enumerate(npzs, 1):
        with np.load(p, allow_pickle=False) as d:
            offs, blob = d["rgb_offsets"], d["rgb_blob"]
            orig_hw, K_orig = d["orig_hw"], d["K_orig"]
            poses_R, poses_t = d["poses_R"].astype(np.float32), d["poses_t"].astype(np.float32)
            causal_fwd = d["causal_fwd"].astype(np.float32)
            eval_rows = d["eval_rows"].astype(np.float64)
            gt_bev_static = d["gt_bev_static"] if "gt_bev_static" in d else None
            cam_name = str(d["cam"]) if "cam" in d else "unknown"

        if cam_name == "unknown":
            if "camera_head" in p.stem or p.stem.endswith("_head"):
                cam_name = "camera_head"
            elif "camera_chest" in p.stem or p.stem.endswith("_chest"):
                cam_name = "camera_chest"
        cam_counts[cam_name] += 1

        T = len(eval_rows)
        rgbs, Ks, gs = [], [], []
        for i in range(T):
            rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=1.0)
            rgbs.append(((rgb_f - IM_MEAN) / IM_STD).transpose(2, 0, 1))
            Ks.append(K)
            g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
            g_cam /= np.linalg.norm(g_cam) + 1e-6
            gs.append(g_cam)
        Ks_arr = np.stack(Ks).astype(np.float32)

        dep_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
        dh_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
        nrm_s4 = np.empty((T, 3, H_IN // 4, W_IN // 4), dtype=np.float32)
        ls_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)

        with torch.inference_mode():
            for i0 in range(0, T, 32):
                i1 = min(T, i0 + 32)
                x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                Kt = torch.from_numpy(Ks_arr[i0:i1]).to(device)
                gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device)
                rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
                hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(x, rays, None, None, host_focal=hf, return_features=False)
                dep_s4[i0:i1] = out["depth_m"][:, 0, ::4, ::4].float().cpu().numpy()
                dh_s4[i0:i1] = out["delta_h_m"][:, 0, ::4, ::4].float().cpu().numpy()
                nrm_s4[i0:i1] = out["normal_grav"][:, :, ::4, ::4].float().cpu().numpy()
                ls_s4[i0:i1] = out["log_sigma"][:, 0, ::4, ::4].float().cpu().numpy()

        for idx_h in range(0, T, 10):
            z_i = dep_s4[idx_h]
            K_i = Ks_arr[idx_h]
            xyz = np.stack(((u_sub - K_i[0, 2]) * z_i / K_i[0, 0], (v_sub - K_i[1, 2]) * z_i / K_i[1, 1], z_i), axis=-1) @ poses_R[idx_h].T
            fl = (xyz[..., 2] < 0) & (np.abs(nrm_s4[idx_h, 1]) > 0.65) & (z_i > 0.3)
            if fl.sum() >= 20:
                cam_heights[cam_name].append(float(-np.percentile(xyz[..., 2][fl], 15)))

        arms = run_all_readouts_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, causal_fwd)

        stem = p.stem
        for arm in ALL_ARMS:
            for field in ("dist", "raw", "confirmed"):
                bundle[f"{stem}__arm__{arm}__{field}"] = arms[arm][field]

        bundle[f"{stem}__gt__orig_full"] = eval_rows
        if gt_bev_static is not None:
            gt_bev_full = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
            gt_bev_low = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0,))
            gt_bev_midup = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))
            gt_bev_head = build_bev_band_eval_rows(eval_rows, gt_bev_static, (2,))
            bundle[f"{stem}__gt__bev_full"] = gt_bev_full
            bundle[f"{stem}__gt__bev_low_only"] = gt_bev_low
            bundle[f"{stem}__gt__bev_midup"] = gt_bev_midup
            bundle[f"{stem}__gt__bev_head"] = gt_bev_head
            low_excl_mask = build_low_only_bev_mask(gt_bev_static)
            bundle[f"{stem}__low_excl_mask"] = low_excl_mask.astype(np.uint8)

            val_b = (np.isfinite(eval_rows[:, 0]) | np.isinf(eval_rows[:, 0])) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
            valid_tot += int(val_b.sum())
            low_only_cnt += int((val_b & low_excl_mask).sum())
            for b_ana, r_b in (("bev_full", gt_bev_full), ("bev_midup", gt_bev_midup)):
                clr_b = val_b & (r_b[:, 0] > 4.0) & (r_b[:, 2] <= 0.5)
                anatomy_recs[b_ana].append(
                    analyze_clip_fa_anatomy(
                        dep_s4, dh_s4, nrm_s4, Ks_arr, poses_R, causal_fwd,
                        arms["bev_head_anchored_85"]["dist"],
                        arms["bev_head_anchored_85_pg"]["dist"],
                        clr_b,
                    )
                )

        clips_meta.append({"name": stem, "cam": cam_name, "frames": T, "has_gt_bev_static": gt_bev_static is not None})
        if ci == 1 or ci % 10 == 0 or ci == len(npzs):
            print(f"  [sanpo_real_zs40] {ci}/{len(npzs)} clips", flush=True)

    bundle_path = OUT_DIR / "sanpo_real_zs40_1d_bundle.npz"
    meta_path = OUT_DIR / "sanpo_real_zs40_meta.json"
    np.savez_compressed(bundle_path, **bundle)
    meta_payload = {
        "checkpoint": ckpt_path.name,
        "gpu": gpu_name,
        "elapsed_s": round(time.time() - t0, 2),
        "n_clips": len(clips_meta),
        "valid_frames": valid_tot,
        "low_only_frames": low_only_cnt,
        "cam_counts": dict(cam_counts),
        "est_cam_height_median_m": {k: round(float(np.median(v)), 3) for k, v in cam_heights.items() if v},
        "anatomy_by_band": {b: merge_anatomy(recs) for b, recs in anatomy_recs.items()},
        "clips": clips_meta,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"[save] {bundle_path} ({bundle_path.stat().st_size / 1e6:.2f} MB), elapsed={meta_payload['elapsed_s']}s", flush=True)
    subprocess.run(["rclone", "copy", str(bundle_path), DRIVE_BACKUP, "--transfers", "4"], check=True)
    subprocess.run(["rclone", "copy", str(meta_path), DRIVE_BACKUP, "--transfers", "4"], check=True)
    print("[done] SANPO bundle & meta backed up to Drive!", flush=True)


if __name__ == "__main__":
    main()
