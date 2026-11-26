"""Unified Colab L4 Runner for Cane-Blind Pillar 1 (Overhangs), Pillar 2 (GlassRecon), & Pillar 3 (Counterfactual).

Strictly zero-training EXPLORE diagnostic on frozen `egogeom_omni_v41_pure_rgb_best.pt`:
  - Pillar 1: "底透顶实" pure overhang mining & evaluation across `fresh_sanpo_real_unseen`,
              `fresh_sanpo_synth_unseen`, `sanpo_real_zs40`, and `aria_gen2`.
  - Pillar 2: Held-out `GlassRecon Val` (15% split) real glass door/wall pixel + corridor audit.
  - Pillar 3: Controlled "Ground-Contact Erasure" counterfactual causal ablation (N=60 frames:
              30 Aria + 30 SANPO-Real) proving whether upper-body depth depends on base contact.

PRIVACY / LICENSE BOUNDARY:
  - Aria Gen 2 (CC BY-NC) pixels/GT remain strictly local to the ephemeral VM & local report JSON;
  - All uploaded Drive contact sheets (`overhang_sanpo_contact_sheet.png`, `glassrecon_val_contact_sheet.png`,
    `counterfactual_base_erase_sheet.png`) contain ONLY SANPO or GlassRecon pixels (zero Aria frames!).
"""
from __future__ import annotations

import argparse
import contextlib
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CODE_DIR = Path(__file__).resolve().parent
PARENT_DIR = CODE_DIR.parent
for _p in (str(CODE_DIR), str(PARENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from egogeom_dataset import GlassReconIndex, IMAGENET_MEAN, IMAGENET_STD, apply_glasses_augment  # noqa: E402
from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K  # noqa: E402
from egogeom_temporal_bev import EgoGeomTemporalBEV, TemporalBEVConfig  # noqa: E402
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3, TemporalBEVV3Config  # noqa: E402
from eval_cane_neutral_and_ablation import HeadAnchoredGeomBEV, M6HeadAnchoredUpgraded  # noqa: E402
from run_cane_blind_colab import (  # noqa: E402
    M6HeadAnchoredRisk,
    build_bev_band_eval_rows,
    confirm_score,
    decode_isotropic_frame,
    strict_headings,
)

WORK = Path("/content/work")
INPUTS = Path("/content/inputs")
DATA_DIR = WORK / "data"
CKPT_DIR = WORK / "ckpt"
TAR_DIR = WORK / "tars"
OUT_DIR = WORK / "out"
GLASS_DIR = WORK / "glassrecon"
ARIA_DIR_OVERRIDE: Path | None = None
ARIA_GT_PATH: Path = INPUTS / "aria_height_band_gt.npz"
CKPT_PATH_OVERRIDE: Path | None = None
RCLONE_BIN: str = shutil.which("rclone") or r"E:\codex-tools\tools\rclone\rclone.exe"

H_IN, W_IN = 288, 384
DRIVE_BACKUP = "gdrive:BlindAssist/runs/egogeom_cane_blind_band_probe/pillar123_v0"
SAFE_UPLOAD_FILES = (
    "overhang_glass_cf_report.json",
    "overhang_sanpo_contact_sheet.png",
    "glassrecon_val_contact_sheet.png",
    "counterfactual_base_erase_sheet.png",
)

PILLAR1_ARMS = (
    "bev_full_ground",
    "bev_midup_ground",
    "bev_ha85_geom",
    "m6_full_ground",
    "m6_midup_ground",
    "m6_head_anchored_85_pg",
    "m6_ha70_nopg",
    "m6_ha_adapt",
)


def _resolve_rclone() -> str:
    win_rclone = Path(r"E:\codex-tools\tools\rclone\rclone.exe")
    if win_rclone.exists():
        return str(win_rclone)
    found = shutil.which("rclone")
    if found:
        return found
    raise FileNotFoundError("rclone binary not found")


def run_rclone_copy(remote: str, local_dir: Path, is_local: bool = False) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if is_local:
        cmd = [
            RCLONE_BIN, "copy", remote, str(local_dir),
            "--multi-thread-streams", "0", "--transfers", "1",
            "--timeout", "45s", "--contimeout", "15s",
            "--low-level-retries", "10", "--retries", "5",
            "--stats", "20s", "--stats-one-line", "-v",
        ]
    else:
        cmd = [RCLONE_BIN, "copy", remote, str(local_dir), "--transfers", "16"]
    subprocess.run(cmd, check=True)


def run_rclone_copyto(local_file: Path, remote_file: str) -> None:
    subprocess.run(
        [RCLONE_BIN, "copyto", str(local_file), remote_file, "--timeout", "45s", "--contimeout", "15s", "--retries", "5"],
        check=True,
    )


def extract_tar(tar_path: Path, dst_dir: Path, unlink_after: bool = False) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / f".done_{tar_path.name}"
    if marker.exists():
        if unlink_after and tar_path.exists():
            tar_path.unlink(missing_ok=True)
        return
    print(f"[stage] Extracting {tar_path.name} -> {dst_dir}", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst_dir)
    marker.write_text("ok\n", encoding="utf-8")
    if unlink_after and tar_path.exists():
        tar_path.unlink(missing_ok=True)


def ensure_deps(is_local: bool = False) -> None:
    import timm
    if is_local:
        assert timm.is_model("convnext_small.dinov3_lvd1689m"), "timm missing convnext_small.dinov3_lvd1689m"
        assert Path(RCLONE_BIN).exists() or shutil.which(RCLONE_BIN), f"rclone not found: {RCLONE_BIN}"
        return
    if not timm.is_model("convnext_small.dinov3_lvd1689m"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "timm"], check=True)
    if shutil.which("rclone") is None:
        rc = subprocess.run("apt-get update -qq && apt-get install -y -qq rclone", shell=True)
        if rc.returncode != 0 or shutil.which("rclone") is None:
            subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=True)


def stage_all_assets(is_local: bool = False, pillars: set[str] | None = None) -> Path:
    global RCLONE_BIN
    if pillars is None:
        pillars = {"1", "2", "3"}
    need_p13 = bool(pillars & {"1", "3"})
    need_p2 = "2" in pillars

    t0 = time.time()
    ensure_deps(is_local=is_local)
    RCLONE_BIN = _resolve_rclone()
    for d in (DATA_DIR, CKPT_DIR, TAR_DIR, OUT_DIR, GLASS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Clean up or reuse test_dl shard if present
    test_dl_dir = TAR_DIR / "test_dl"
    if test_dl_dir.exists():
        cached_sh01 = test_dl_dir / "fresh_sanpo_real_unseen_shard01.tar"
        target_sh01_dir = TAR_DIR / "fresh_sanpo_real_unseen"
        if cached_sh01.exists():
            target_sh01_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cached_sh01), str(target_sh01_dir / "fresh_sanpo_real_unseen_shard01.tar"))
        shutil.rmtree(test_dl_dir, ignore_errors=True)

    if CKPT_PATH_OVERRIDE is not None and CKPT_PATH_OVERRIDE.exists():
        ckpt_path = CKPT_PATH_OVERRIDE
        print(f"[stage] Using local checkpoint: {ckpt_path}", flush=True)
    else:
        ckpt_path = CKPT_DIR / "egogeom_omni_v41_pure_rgb_best.pt"
        if not ckpt_path.exists():
            print("[stage] Pulling v4.1 pure-RGB checkpoint...", flush=True)
            run_rclone_copy("gdrive:BlindAssist/runs/egogeom_omni_v41/egogeom_omni_v41_pure_rgb_best.pt", CKPT_DIR, is_local=is_local)

    if need_p13:
        # 1) Aria local dir or tar
        if ARIA_DIR_OVERRIDE is not None and ARIA_DIR_OVERRIDE.exists():
            print(f"[stage] Using local Aria dir: {ARIA_DIR_OVERRIDE}", flush=True)
        elif (INPUTS / "aria_g2_20clips.tar").exists():
            extract_tar(INPUTS / "aria_g2_20clips.tar", DATA_DIR / "aria_gen2", unlink_after=False)

        # 2) SANPO-Real ZS40
        for sh in ("shard_000.tar", "shard_001.tar"):
            if not (DATA_DIR / "sanpo_real_zs40" / f".done_{sh}").exists():
                tar_file = TAR_DIR / "sanpo_zs40" / sh
                if not tar_file.exists():
                    run_rclone_copy(f"gdrive:BlindAssist/datasets/sanpo_real_zeroshot_40_v0/{sh}", TAR_DIR / "sanpo_zs40", is_local=is_local)
                extract_tar(tar_file, DATA_DIR / "sanpo_real_zs40", unlink_after=True)

        # 3) Fresh SANPO-Real & SANPO-Synth unseen pools
        for sub, tars in [
            ("fresh_sanpo_real_unseen", ("fresh_sanpo_real_unseen_shard00.tar", "fresh_sanpo_real_unseen_shard01.tar")),
            ("fresh_sanpo_synth_unseen", ("fresh_sanpo_synth_unseen_shard00.tar", "fresh_sanpo_synth_unseen_shard01.tar")),
        ]:
            dst = DATA_DIR / sub
            for sh in tars:
                if not (dst / f".done_{sh}").exists():
                    tar_file = TAR_DIR / sub / sh
                    if not tar_file.exists():
                        run_rclone_copy(f"gdrive:BlindAssist/datasets/unseen_fresh_blind_probe_v0/{sub}/{sh}", TAR_DIR / sub, is_local=is_local)
                    extract_tar(tar_file, dst, unlink_after=True)

        aria_dir = ARIA_DIR_OVERRIDE if ARIA_DIR_OVERRIDE is not None else (DATA_DIR / "aria_gen2")
        for name, dpath in (
            ("aria_gen2", aria_dir),
            ("sanpo_real_zs40", DATA_DIR / "sanpo_real_zs40"),
            ("fresh_sanpo_real_unseen", DATA_DIR / "fresh_sanpo_real_unseen"),
            ("fresh_sanpo_synth_unseen", DATA_DIR / "fresh_sanpo_synth_unseen"),
        ):
            print(f"[stage] {name}: {len(list(dpath.rglob('*.npz')))} npz", flush=True)

    if need_p2:
        # 4) GlassRecon (9.56 GB)
        gls_marker = GLASS_DIR / ".done_glassrecon"
        if not gls_marker.exists():
            print("[stage] Pulling GlassRecon v1 (9.56 GB)...", flush=True)
            zpath = TAR_DIR / "glassrecon/glassrecon_release.zip"
            if not zpath.exists():
                run_rclone_copy("gdrive:BlindAssist/datasets/glassrecon_v1/glassrecon_release.zip", TAR_DIR / "glassrecon", is_local=is_local)
            print("[stage] Extracting glassrecon_release.zip...", flush=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(GLASS_DIR)
            zpath.unlink(missing_ok=True)
            gls_marker.write_text("ok\n", encoding="utf-8")

    print(f"[stage] Requested assets ready in {time.time() - t0:.1f}s (pillars={sorted(pillars)})", flush=True)
    return ckpt_path


# ---------------------------------------------------------------------------
# Helpers: Model Inference & Readouts
# ---------------------------------------------------------------------------

def load_v41_model(ckpt_path: Path, device: torch.device) -> EgoGeomOmniV41Net:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**state["cfg"]), pretrained_backbone=False)
    model.tof_branch = None
    model.load_state_dict({k: v for k, v in state["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    del state
    model.to(device).eval()
    return model


def infer_rgb_tensors(
    model: EgoGeomOmniV41Net,
    device: torch.device,
    rgbs_chw: list[np.ndarray],
    Ks_arr: np.ndarray,
    gs_arr: np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = len(rgbs_chw)
    H4, W4 = H_IN // 4, W_IN // 4
    dep_s4 = np.empty((T, H4, W4), dtype=np.float32)
    dh_s4 = np.empty((T, H4, W4), dtype=np.float32)
    nrm_s4 = np.empty((T, 3, H4, W4), dtype=np.float32)
    ls_s4 = np.empty((T, H4, W4), dtype=np.float32)

    use_cuda_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda_amp and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.inference_mode():
        for i0 in range(0, T, batch_size):
            i1 = min(T, i0 + batch_size)
            x = torch.from_numpy(np.stack(rgbs_chw[i0:i1])).to(device)
            Kt = torch.from_numpy(Ks_arr[i0:i1]).to(device)
            gt = torch.from_numpy(gs_arr[i0:i1]).to(device)
            rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
            hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_cuda_amp else contextlib.nullcontext()
            with amp_ctx:
                out = model(x, rays, None, None, host_focal=hf, return_features=False)
            dep_s4[i0:i1] = out["depth_m"][:, 0, ::4, ::4].float().cpu().numpy()
            dh_s4[i0:i1] = out["delta_h_m"][:, 0, ::4, ::4].float().cpu().numpy()
            nrm_s4[i0:i1] = out["normal_grav"][:, :, ::4, ::4].float().cpu().numpy()
            ls_s4[i0:i1] = out["log_sigma"][:, 0, ::4, ::4].float().cpu().numpy()
    return dep_s4, dh_s4, nrm_s4, ls_s4


def run_pillar1_readouts(
    dep_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    ls_s4: np.ndarray,
    Ks: np.ndarray,
    poses_R: np.ndarray,
    poses_t: np.ndarray,
    strict_fwd: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    T = len(dep_s4)
    bev_full = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.15, hgt_hi_m=1.90, use_scale_lock=True))
    bev_midup = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.65, hgt_hi_m=1.90, use_scale_lock=True))
    bev_ha85_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    bev_ha70_g = HeadAnchoredGeomBEV(z_rel_lo=-0.70, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    bev_adapt_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)

    m6_full = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.06, geom_rel_h_hi=1.30))
    m6_midup = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.40, geom_rel_h_hi=1.30))
    m6_ha85_pg = M6HeadAnchoredRisk(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=10.0)
    m6_ha70_nopg = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_ha_adapt = M6HeadAnchoredUpgraded(dh_cal=True, adaptive_band=True, vertical_power=1.0, dh_scale=0.25, band_ramp_m=0.30)

    out = {
        arm: {"dist": np.full(T, np.inf, dtype=np.float32), "raw": np.zeros(T, dtype=np.float64)}
        for arm in PILLAR1_ARMS
    }
    for i in range(T):
        args_bev = (dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        d, _, rh = bev_full.step(*args_bev)
        out["bev_full_ground"]["dist"][i], out["bev_full_ground"]["raw"][i] = d, rh

        d, _, rh = bev_midup.step(*args_bev)
        out["bev_midup_ground"]["dist"][i], out["bev_midup_ground"]["raw"][i] = d, rh

        d_85g, _, rh_85g = bev_ha85_g.step(*args_bev)
        out["bev_ha85_geom"]["dist"][i], out["bev_ha85_geom"]["raw"][i] = d_85g, rh_85g

        d_f, _, _, rg_f = m6_full.step(*args_bev)
        out["m6_full_ground"]["dist"][i], out["m6_full_ground"]["raw"][i] = d_f, rg_f

        d_m, _, _, rg_m = m6_midup.step(*args_bev)
        out["m6_midup_ground"]["dist"][i], out["m6_midup_ground"]["raw"][i] = d_m, rg_m

        out["m6_head_anchored_85_pg"]["dist"][i] = d_85g
        out["m6_head_anchored_85_pg"]["raw"][i] = m6_ha85_pg.step(*args_bev)

        d_70g, _, _ = bev_ha70_g.step(*args_bev)
        out["m6_ha70_nopg"]["dist"][i] = d_70g
        out["m6_ha70_nopg"]["raw"][i] = m6_ha70_nopg.step(*args_bev)

        rg_ad, zlo_ad, fdh_ad = m6_ha_adapt.step(*args_bev)
        d_ad, _, _ = bev_adapt_g.step(*args_bev, z_rel_lo_override=zlo_ad, floor_dh_bias=fdh_ad)
        out["m6_ha_adapt"]["dist"][i] = d_ad
        out["m6_ha_adapt"]["raw"][i] = rg_ad

    for arm in PILLAR1_ARMS:
        out[arm]["confirmed"] = confirm_score(out[arm]["raw"])
    return out


# ---------------------------------------------------------------------------
# Pillar 1: Overhang Mining & Evaluation
# ---------------------------------------------------------------------------

def _episode_hits(mask_pos: np.ndarray, alerts: np.ndarray) -> tuple[int, int]:
    """Count >=2-frame contiguous overhang episodes and episodes alerted on >=1 frame."""
    n = len(mask_pos)
    ep = hit = i = 0
    while i < n:
        if mask_pos[i]:
            j = i
            while j + 1 < n and mask_pos[j + 1]:
                j += 1
            if (j - i + 1) >= 2:
                ep += 1
                if alerts[i : j + 1].any():
                    hit += 1
            i = j + 1
        else:
            i += 1
    return ep, hit


def run_pillar1(model: EgoGeomOmniV41Net, device: torch.device, batch_size: int = 32) -> tuple[dict, dict]:
    print("\n================ [PILLAR 1] Pure-Overhang Mining & Eval ================", flush=True)
    aria_gt = np.load(ARIA_GT_PATH, allow_pickle=False)
    aria_dir = ARIA_DIR_OVERRIDE if ARIA_DIR_OVERRIDE is not None else (DATA_DIR / "aria_gen2")
    ds_cfgs = [
        ("fresh_sanpo_real_unseen", DATA_DIR / "fresh_sanpo_real_unseen"),
        ("fresh_sanpo_synth_unseen", DATA_DIR / "fresh_sanpo_synth_unseen"),
        ("sanpo_real_zs40", DATA_DIR / "sanpo_real_zs40"),
        ("aria_gen2", aria_dir),
    ]

    p1_report: dict = {}
    cached_clips: dict[str, list[dict]] = {}
    ovh_candidates: list[dict] = []

    for dname, ddir in ds_cfgs:
        t_ds = time.time()
        npzs = sorted(ddir.rglob("*.npz"))
        clips_data = []
        for ci, p in enumerate(npzs, 1):
            with np.load(p, allow_pickle=False) as d:
                offs = d["rgb_offsets"] if "rgb_offsets" in d else d["rgb_jpeg_offsets"]
                blob = d["rgb_blob"] if "rgb_blob" in d else d["rgb_jpeg_bytes"]
                orig_hw, K_orig = d["orig_hw"], d["K_orig"]
                poses_R, poses_t = d["poses_R"].astype(np.float32), d["poses_t"].astype(np.float32)
                causal_fwd = d["causal_fwd"].astype(np.float32)
                eval_rows = d["eval_rows"].astype(np.float64)
                gt_bev_static = d["gt_bev_static"] if "gt_bev_static" in d else None

            T = len(eval_rows)
            strict_fwd = strict_headings(dname, causal_fwd, poses_R, poses_t)
            rgbs_norm, rgbs_u8, Ks, gs = [], [], [], []
            pitches = np.empty(T, dtype=np.float32)
            for i in range(T):
                rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=1.0)
                rgbs_u8.append(np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8))
                rgbs_norm.append(((rgb_f - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
                Ks.append(K)
                g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                gs.append(g_cam)
                pitches[i] = float(np.degrees(np.arcsin(np.clip(poses_R[i, 2, 2], -1.0, 1.0))))
            Ks_arr = np.stack(Ks).astype(np.float32)
            gs_arr = np.stack(gs).astype(np.float32)

            dep_s4, dh_s4, nrm_s4, ls_s4 = infer_rgb_tensors(model, device, rgbs_norm, Ks_arr, gs_arr, batch_size=batch_size)
            arms = run_pillar1_readouts(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, strict_fwd)

            if dname == "aria_gen2":
                r_full = aria_gt[f"{p.stem}__centered_full__full_ground"].astype(np.float64)
                r_low = aria_gt[f"{p.stem}__centered_full__low_only_ground"].astype(np.float64)
                r_mid = aria_gt[f"{p.stem}__centered_full__midup_ground"].astype(np.float64)
                r_head = aria_gt[f"{p.stem}__centered_full__head_ground"].astype(np.float64)
            else:
                r_full = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
                r_low = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0,))
                r_mid = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))
                r_head = build_bev_band_eval_rows(eval_rows, gt_bev_static, (2,))

            valid = (np.isfinite(r_full[:, 0]) | np.isinf(r_full[:, 0])) & (~np.isnan(r_full[:, 1])) & (r_full[:, 1] >= 0.6)
            dl, dm, dh, df = r_low[:, 0], r_mid[:, 0], r_head[:, 0], r_full[:, 0]
            dyn = r_full[:, 2] > 0.5

            # GT-only disjoint/diagnostic masks
            fin_ld = np.isfinite(dl) & np.isfinite(dm)
            dl_dm_diff = np.where(fin_ld, np.abs(dl - dm), np.inf)
            dl_minus_dm = np.where(fin_ld, dl - dm, np.inf)
            m_grounded = valid & (dm <= 4.0) & (dl <= 4.0) & (dl_dm_diff < 0.8)
            m_ovh_midup = valid & (dm <= 4.0) & (dl > 4.0)
            m_ovh_head = valid & (dh <= 4.0) & (dl > 4.0)
            m_ovh_lvl = m_ovh_midup & (np.abs(pitches) <= 6.0) & (dm <= 3.5)
            m_ovh_gap08 = valid & (dm <= 4.0) & ((dl > 4.0) | (dl_minus_dm >= 0.8))
            m_clear = valid & (df > 4.0) & (~dyn)

            # Collect visual candidates strictly from SANPO (NO Aria pixels!)
            if dname != "aria_gen2":
                idx_cands = np.flatnonzero(m_ovh_lvl | (m_ovh_head & (np.abs(pitches) <= 8.0)))
                if len(idx_cands) == 0:
                    idx_cands = np.flatnonzero(m_ovh_midup & (np.abs(pitches) <= 10.0))
                for fi in idx_cands[::2]:  # sample nicely across sequences
                    ovh_candidates.append({
                        "dataset": dname,
                        "clip": p.stem,
                        "frame": int(fi),
                        "pitch": float(pitches[fi]),
                        "d_low": float(dl[fi]),
                        "d_midup": float(dm[fi]),
                        "d_head": float(dh[fi]),
                        "d_pr_ha85g": float(arms["bev_ha85_geom"]["dist"][fi]),
                        "d_pr_midup": float(arms["bev_midup_ground"]["dist"][fi]),
                        "rgb": rgbs_u8[fi],
                        "dep_s4": dep_s4[fi],
                        "prio": (0 if np.isfinite(dh[fi]) and dh[fi] <= 4.0 else 1, float(dm[fi])),
                    })

            clips_data.append({
                "stem": p.stem, "T": T, "rgbs_u8": rgbs_u8, "Ks": Ks_arr, "poses_R": poses_R,
                "dep_s4": dep_s4, "dh_s4": dh_s4, "nrm_s4": nrm_s4, "ls_s4": ls_s4, "strict_fwd": strict_fwd,
                "arms": arms, "dl": dl, "dm": dm, "dh": dh, "df": df, "valid": valid,
                "subsets": {
                    "grounded_midup": m_grounded,
                    "ovh_midup": m_ovh_midup,
                    "ovh_head": m_ovh_head,
                    "ovh_level_gaze": m_ovh_lvl,
                    "ovh_gap08": m_ovh_gap08,
                },
                "strict_clear": m_clear,
            })
            if ci == 1 or ci % 15 == 0 or ci == len(npzs):
                print(f"  [{dname}] {ci}/{len(npzs)} clips", flush=True)

        # Aggregate Pillar 1 metrics for this dataset
        sub_names = ("grounded_midup", "ovh_midup", "ovh_head", "ovh_level_gaze", "ovh_gap08")
        d_res: dict = {"n_clips": len(clips_data), "elapsed_s": round(time.time() - t_ds, 2), "subsets": {}}
        clr_cat = {
            arm: {fld: np.concatenate([c["arms"][arm][fld][c["strict_clear"]] for c in clips_data]) for fld in ("raw", "confirmed")}
            for arm in PILLAR1_ARMS
        }
        n_clear_tot = sum(int(c["strict_clear"].sum()) for c in clips_data)
        d_res["n_strict_clear"] = n_clear_tot

        for sname in sub_names:
            n_pos = sum(int(c["subsets"][sname].sum()) for c in clips_data)
            n_clips_pos = sum(1 for c in clips_data if c["subsets"][sname].any())
            s_entry = {"n_frames": n_pos, "n_clips_with_frames": n_clips_pos, "arms": {}}
            if n_pos == 0 or n_clear_tot == 0:
                d_res["subsets"][sname] = s_entry
                continue

            d_gt_all = np.concatenate([c["dm"][c["subsets"][sname]] for c in clips_data])
            for arm in PILLAR1_ARMS:
                d_pr_all = np.concatenate([c["arms"][arm]["dist"][c["subsets"][sname]] for c in clips_data])
                sc_r = np.concatenate([c["arms"][arm]["raw"][c["subsets"][sname]] for c in clips_data])
                sc_c = np.concatenate([c["arms"][arm]["confirmed"][c["subsets"][sname]] for c in clips_data])

                fin4 = np.isfinite(d_pr_all) & (d_pr_all <= 4.0)
                fin6 = np.isfinite(d_pr_all) & (d_pr_all <= 6.0)
                errs = d_pr_all[fin6] - d_gt_all[fin6]
                fill6 = np.where(fin6, d_pr_all, 6.0)

                t5_r = float(np.quantile(clr_cat[arm]["raw"], 0.95))
                t5_c = float(np.quantile(clr_cat[arm]["confirmed"], 0.95))
                y_bin = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_clear_tot, dtype=np.int32)])
                au_r = float(roc_auc_score(y_bin, np.concatenate([sc_r, clr_cat[arm]["raw"]])))
                au_c = float(roc_auc_score(y_bin, np.concatenate([sc_c, clr_cat[arm]["confirmed"]])))

                arm_rec = {
                    "pred_finite_rate_le4m": round(float(fin4.mean()), 4),
                    "pred_finite_rate_le6m": round(float(fin6.mean()), 4),
                    "mae_finite_m": round(float(np.abs(errs).mean()), 3) if len(errs) else None,
                    "mae_fixed6m_m": round(float(np.abs(fill6 - d_gt_all).mean()), 3),
                    "median_signed_err_m": round(float(np.median(errs)), 3) if len(errs) else None,
                    "recall_at_5pct_clr_raw": round(float((sc_r >= t5_r).mean()), 4),
                    "recall_at_5pct_clr_confirmed": round(float((sc_c >= t5_c).mean()), 4),
                    "t5_clr_raw": round(t5_r, 5),
                    "t5_clr_confirmed": round(t5_c, 5),
                    "zero_tie_at_t5_confirmed": bool(t5_c <= 1e-12),
                    "auroc_vs_clear_raw": round(au_r, 4),
                    "auroc_vs_clear_confirmed": round(au_c, 4),
                }
                if sname in ("ovh_midup", "ovh_level_gaze"):
                    ep_tot = hit_t5 = hit_d4 = 0
                    for c in clips_data:
                        mp = c["subsets"][sname]
                        e1, h1 = _episode_hits(mp, c["arms"][arm]["confirmed"] >= t5_c)
                        _, h2 = _episode_hits(mp, np.isfinite(c["arms"][arm]["dist"]) & (c["arms"][arm]["dist"] <= 4.0))
                        ep_tot += e1
                        hit_t5 += h1
                        hit_d4 += h2
                    arm_rec["overhang_episodes"] = ep_tot
                    arm_rec["episode_hit_rate_t5c"] = round(float(hit_t5 / ep_tot), 4) if ep_tot else None
                    arm_rec["episode_hit_rate_le4m"] = round(float(hit_d4 / ep_tot), 4) if ep_tot else None
                s_entry["arms"][arm] = arm_rec
            d_res["subsets"][sname] = s_entry

        p1_report[dname] = d_res
        cached_clips[dname] = clips_data

    # Build SANPO-only contact sheet (4 rows x 5 pairs = 20 frames, side-by-side RGB + depth = 4x10 subplots -> crisp 5 cols of (RGB|Dep))
    ovh_candidates.sort(key=lambda x: x["prio"])
    # Deduplicate by clip so diverse clips appear
    picked, seen_clip = [], defaultdict(int)
    for item in ovh_candidates:
        if seen_clip[item["clip"]] >= 1:
            continue
        picked.append(item)
        seen_clip[item["clip"]] += 1
        if len(picked) >= 20:
            break
    if len(picked) < 20:
        for item in ovh_candidates:
            if item not in picked:
                picked.append(item)
            if len(picked) >= 20:
                break

    sheet_path = OUT_DIR / "overhang_sanpo_contact_sheet.png"
    if picked:
        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols * 2, figsize=(24, 13))
        fig.suptitle("Mined Cane-Blind Overhangs (SANPO-Real & Synth ONLY — ZERO Aria Pixels)", fontsize=14)
        for idx_p in range(rows * cols):
            r_i, c_i = divmod(idx_p, cols)
            ax_rgb = axes[r_i, 2 * c_i]
            ax_dep = axes[r_i, 2 * c_i + 1]
            ax_rgb.axis("off")
            ax_dep.axis("off")
            if idx_p >= len(picked):
                continue
            it = picked[idx_p]
            ax_rgb.imshow(it["rgb"])
            dl_s = f"{it['d_low']:.1f}" if np.isfinite(it["d_low"]) else ">4"
            dm_s = f"{it['d_midup']:.2f}" if np.isfinite(it["d_midup"]) else ">4"
            dh_s = f"{it['d_head']:.2f}" if np.isfinite(it["d_head"]) else ">4"
            dp_s = f"{it['d_pr_ha85g']:.2f}" if np.isfinite(it["d_pr_ha85g"]) else "inf"
            ax_rgb.set_title(f"{it['dataset'][:11]} p={it['pitch']:.0f}\nL={dl_s} M={dm_s} H={dh_s}", fontsize=7.5)
            dep_vis = np.clip(it["dep_s4"], 0.3, 6.0)
            ax_dep.imshow(dep_vis, cmap="turbo_r", vmin=0.3, vmax=6.0)
            ax_dep.set_title(f"{it['clip'][:10]}#{it['frame']}\nha85g={dp_s}m", fontsize=7.5)
        plt.tight_layout()
        fig.savefig(sheet_path, dpi=140)
        plt.close(fig)
        print(f"[Pillar 1] Saved contact sheet {sheet_path} ({len(picked)} frames)", flush=True)

    return p1_report, cached_clips


# ---------------------------------------------------------------------------
# Pillar 2: Real Glass Door & Glass Wall Audit (GlassRecon Val)
# ---------------------------------------------------------------------------

def run_pillar2(model: EgoGeomOmniV41Net, device: torch.device, batch_size: int = 8) -> dict:
    print("\n================ [PILLAR 2] GlassRecon Val Audit ================", flush=True)
    gls_val = GlassReconIndex(GLASS_DIR, split_filter="val")
    if len(gls_val.items) == 0:
        subdirs = [p for p in GLASS_DIR.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            gls_val = GlassReconIndex(subdirs[0], split_filter="val")
    n_total = len(gls_val.items)
    print(f"[GlassRecon Val] Found {n_total} held-out validation frames (streaming chunk={batch_size})", flush=True)
    if n_total == 0:
        raise RuntimeError(f"GlassReconIndex returned 0 val items under {GLASS_DIR}!")

    all_z_gt_gls, all_z_pr_gls = [], []
    all_z_gt_opq, all_z_pr_opq = [], []
    all_nd_gls, all_nd_opq = [], []
    all_dh_gls, all_dh_opq = [], []
    all_ls_gls, all_ls_opq = [], []

    corr_records = []
    worst_heap: list[tuple[float, int, dict]] = []  # min-heap of (pen_rate, orig_idx, rec) -> keeps top-12 highest pen_rate
    best_heap: list[tuple[float, int, dict]] = []   # min-heap of (-pen_rate, orig_idx, rec) -> keeps top-12 lowest pen_rate
    n_with_glass_50 = 0

    H4, W4 = H_IN // 4, W_IN // 4
    vv_f, uu_f = np.mgrid[0:H_IN, 0:W_IN].astype(np.float32)
    vv_s, uu_s = np.mgrid[0:H4, 0:W4].astype(np.float32)
    uu_sf = (4.0 * uu_s + 0.5).ravel()
    vv_sf = (4.0 * vv_s + 0.5).ravel()
    use_cuda_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for i0 in range(0, n_total, batch_size):
        i1 = min(n_total, i0 + batch_size)
        chunk = []
        for idx in range(i0, i1):
            raw = gls_val.load(idx)
            rgb_n, dep_rs, val_rs, K_out, g_out, _, gls_rs = apply_glasses_augment(
                raw["rgb"], raw["depth"], raw["valid"], raw["K"], raw["g_cam"],
                raw["normal_cam"], raw["glass"], (H_IN, W_IN), is_train=False,
            )
            rgb_u8 = np.clip((rgb_n * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
            chunk.append((idx, rgb_n.transpose(2, 0, 1), rgb_u8, dep_rs, val_rs > 0.5, gls_rs > 0.5, K_out, g_out))

        with torch.inference_mode():
            x = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
            Kt = torch.from_numpy(np.stack([c[6] for c in chunk]).astype(np.float32)).to(device)
            gt = torch.from_numpy(np.stack([c[7] for c in chunk]).astype(np.float32)).to(device)
            rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
            hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_cuda_amp else contextlib.nullcontext()
            with amp_ctx:
                out = model(x, rays, None, None, host_focal=hf, return_features=False)
            dep_c = out["depth_m"][:, 0].float().cpu().numpy()
            dh_c = out["delta_h_m"][:, 0].float().cpu().numpy()
            nrm_c = out["normal_grav"].float().cpu().numpy()
            ls_c = out["log_sigma"][:, 0].float().cpu().numpy()

        for k, (orig_idx, _, rgb_u8, dep_gt, val_gt, gls_m, K, _) in enumerate(chunk):
            if int((val_gt & gls_m).sum()) >= 50:
                n_with_glass_50 += 1

            zp = dep_c[k]
            nd_abs = np.abs(nrm_c[k, 1])
            dh_i = dh_c[k]
            ls_i = ls_c[k]

            mg = val_gt & gls_m & np.isfinite(zp) & (zp > 0.05)
            mo = val_gt & (~gls_m) & np.isfinite(zp) & (zp > 0.05)
            if mg.any():
                n_gls_px = int(mg.sum())
                z_g_px = dep_gt[mg]
                z_p_px = zp[mg]
                all_z_gt_gls.append(z_g_px)
                all_z_pr_gls.append(z_p_px)
                all_nd_gls.append(nd_abs[mg])
                all_dh_gls.append(dh_i[mg])
                all_ls_gls.append(ls_i[mg])
                if n_gls_px >= 500:
                    pen_r = float((z_p_px > 1.35 * z_g_px).mean())
                    rec_vis = {
                        "idx": int(orig_idx),
                        "n_gls": n_gls_px,
                        "med_err": float(np.median(z_p_px - z_g_px)),
                        "pen_rate": pen_r,
                        "gt_med": float(np.median(z_g_px)),
                        "pr_med": float(np.median(z_p_px)),
                        "rgb": rgb_u8, "gls": gls_m, "dep_pr": zp,
                    }
                    if len(worst_heap) < 12:
                        heapq.heappush(worst_heap, (pen_r, int(orig_idx), rec_vis))
                    elif pen_r > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, (pen_r, int(orig_idx), rec_vis))
                    if len(best_heap) < 12:
                        heapq.heappush(best_heap, (-pen_r, int(orig_idx), rec_vis))
                    elif (-pen_r) > best_heap[0][0]:
                        heapq.heapreplace(best_heap, (-pen_r, int(orig_idx), rec_vis))

            if mo.any():
                sub_idx = np.flatnonzero(mo.ravel())[::4]
                all_z_gt_opq.append(dep_gt.ravel()[sub_idx])
                all_z_pr_opq.append(zp.ravel()[sub_idx])
                all_nd_opq.append(nd_abs.ravel()[sub_idx])
                all_dh_opq.append(dh_i.ravel()[sub_idx])
                all_ls_opq.append(ls_i.ravel()[sub_idx])

            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            if val_gt.any():
                z_g = dep_gt[val_gt]
                x_g = (uu_f[val_gt] - cx) * z_g / fx
                y_g = (vv_f[val_gt] - cy) * z_g / fy
                z_rel_g = -y_g
                in_corr = (np.abs(x_g) <= 0.30) & (z_rel_g >= -0.85) & (z_rel_g <= 0.25) & (z_g >= 0.30) & (z_g <= 4.0)
                gl_corr = in_corr & gls_m[val_gt]
                op_corr = in_corr & (~gls_m[val_gt])
                d_gls_corr = float(np.percentile(z_g[gl_corr], 10)) if int(gl_corr.sum()) >= 10 else float(np.inf)
                d_opq_corr = float(np.percentile(z_g[op_corr], 10)) if int(op_corr.sum()) >= 10 else float(np.inf)

                if d_gls_corr <= 2.5 and d_opq_corr > (d_gls_corr + 0.5):
                    zp_s = zp[::4, ::4].ravel()
                    dh_s = dh_i[::4, ::4].ravel()
                    nd_s = nd_abs[::4, ::4].ravel()
                    ok_p = np.isfinite(zp_s) & (zp_s >= 0.25) & (zp_s <= 9.5)
                    xp = (uu_sf[ok_p] - cx) * zp_s[ok_p] / fx
                    yp = (vv_sf[ok_p] - cy) * zp_s[ok_p] / fy
                    zp_v = zp_s[ok_p]
                    in_p = (np.abs(xp) <= 0.30) & (-yp >= -0.85) & (-yp <= 0.25) & (zp_v >= 0.30) & (zp_v <= 4.0)
                    d_pred_raw = float(np.percentile(zp_v[in_p], 10)) if int(in_p.sum()) >= 4 else float(np.inf)
                    in_pg = in_p & (dh_s[ok_p] >= 0.10) & (nd_s[ok_p] < 0.75)
                    d_pred_geom = float(np.percentile(zp_v[in_pg], 10)) if int(in_pg.sum()) >= 4 else float(np.inf)
                    corr_records.append({
                        "d_gt_glass": d_gls_corr,
                        "d_gt_opaque": d_opq_corr,
                        "d_pred_raw": d_pred_raw,
                        "d_pred_geom": d_pred_geom,
                    })

        if (i1 == n_total) or ((i1 // batch_size) % 12 == 0) or (i0 == 0):
            print(f"  [GlassRecon Val] {i1}/{n_total} frames processed", flush=True)

    def _pixel_stats(zg_l, zp_l, nd_l, dh_l, ls_l):
        zg = np.concatenate(zg_l)
        zp = np.concatenate(zp_l)
        nd = np.concatenate(nd_l)
        dh = np.concatenate(dh_l)
        ls = np.concatenate(ls_l)
        err = zp - zg
        near2 = zg <= 2.0
        out = {
            "n_pixels": int(len(zg)),
            "abs_rel": round(float(np.mean(np.abs(err) / zg)), 4),
            "delta1_125": round(float(np.mean(np.maximum(zp / zg, zg / zp) < 1.25)), 4),
            "mae_m": round(float(np.mean(np.abs(err))), 3),
            "median_signed_err_m": round(float(np.median(err)), 3),
            "penetration_rate_35pct": round(float(np.mean(zp > 1.35 * zg)), 4),
            "penetration_rate_08m": round(float(np.mean(err > 0.80)), 4),
            "dangerous_miss_gt2_pred3m": round(float(np.mean(zp[near2] > 3.0)), 4) if near2.any() else None,
            "n_pixels_le2m": int(near2.sum()),
            "aux_median_abs_ndown": round(float(np.median(nd)), 4),
            "aux_frac_vertical_lt075": round(float(np.mean(nd < 0.75)), 4),
            "aux_median_dh_m": round(float(np.median(dh)), 4),
            "aux_median_log_sigma": round(float(np.median(ls)), 4),
            "by_dist_bin": {},
        }
        for bname, lo, hi in [("[0.5,1.5)", 0.5, 1.5), ("[1.5,2.5)", 1.5, 2.5), ("[2.5,4.0)", 2.5, 4.0)]:
            bm = (zg >= lo) & (zg < hi)
            if bm.any():
                be = zp[bm] - zg[bm]
                out["by_dist_bin"][bname] = {
                    "n_pixels": int(bm.sum()),
                    "abs_rel": round(float(np.mean(np.abs(be) / zg[bm])), 4),
                    "delta1": round(float(np.mean(np.maximum(zp[bm] / zg[bm], zg[bm] / zp[bm]) < 1.25)), 4),
                    "median_signed_err_m": round(float(np.median(be)), 3),
                    "penetration_35pct": round(float(np.mean(zp[bm] > 1.35 * zg[bm])), 4),
                    "penetration_08m": round(float(np.mean(be > 0.80)), 4),
                }
        return out

    n_pure_gls = len(corr_records)
    corr_summary = {"n_pure_glass_corridors_le25": n_pure_gls}
    if n_pure_gls > 0:
        dgt = np.array([r["d_gt_glass"] for r in corr_records])
        for key in ("d_pred_raw", "d_pred_geom"):
            dpr = np.array([r[key] for r in corr_records])
            fin = np.isfinite(dpr) & (dpr <= 4.0)
            corr_summary[key] = {
                "hit_le_25m": round(float(np.mean(np.isfinite(dpr) & (dpr <= 2.5))), 4),
                "hit_le_30m": round(float(np.mean(np.isfinite(dpr) & (dpr <= 3.0))), 4),
                "hit_le_40m": round(float(fin.mean()), 4),
                "mae_finite_m": round(float(np.abs(dpr[fin] - dgt[fin]).mean()), 3) if fin.any() else None,
                "median_signed_err_m": round(float(np.median(dpr[fin] - dgt[fin])), 3) if fin.any() else None,
            }

    # Build GlassRecon Val visual sheet: 8 worst penetrations + 8 best detections (from >=500 glass px frames)
    worst8 = [x[2] for x in sorted(worst_heap, key=lambda t: t[0], reverse=True)][:8]
    used_ids = {x["idx"] for x in worst8}
    best8 = [x[2] for x in sorted(best_heap, key=lambda t: -t[0]) if x[2]["idx"] not in used_ids][:8]
    chosen = worst8 + best8
    if chosen:
        fig, axes = plt.subplots(4, 8, figsize=(24, 12))
        fig.suptitle("GlassRecon Held-Out Val: Top Rows = Worst Penetration, Bottom Rows = Best Detection", fontsize=13)
        for k_i in range(16):
            r_row = k_i // 4
            c_pair = k_i % 4
            ax_rgb = axes[r_row, 2 * c_pair]
            ax_dep = axes[r_row, 2 * c_pair + 1]
            ax_rgb.axis("off")
            ax_dep.axis("off")
            if k_i >= len(chosen):
                continue
            it = chosen[k_i]
            rgb_ov = it["rgb"].copy()
            cnts, _ = cv2.findContours(it["gls"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_ov, cnts, -1, (0, 255, 255), 2)
            ax_rgb.imshow(rgb_ov)
            tag = "WORST" if k_i < len(worst8) else "BEST"
            ax_rgb.set_title(f"{tag}#{it['idx']} pen={it['pen_rate']*100:.0f}%\nmed_err={it['med_err']:+.2f}m", fontsize=7.5)
            ax_dep.imshow(np.clip(it["dep_pr"], 0.3, 6.0), cmap="turbo_r", vmin=0.3, vmax=6.0)
            ax_dep.set_title(f"GT_g={it['gt_med']:.2f}m\nPr_g={it['pr_med']:.2f}m", fontsize=7.5)
        plt.tight_layout()
        gls_sheet = OUT_DIR / "glassrecon_val_contact_sheet.png"
        fig.savefig(gls_sheet, dpi=140)
        plt.close(fig)
        print(f"[Pillar 2] Saved GlassRecon contact sheet {gls_sheet}", flush=True)

    return {
        "n_val_frames_total": int(n_total),
        "n_val_frames_glass_ge50": int(n_with_glass_50),
        "glass_pixels": _pixel_stats(all_z_gt_gls, all_z_pr_gls, all_nd_gls, all_dh_gls, all_ls_gls),
        "opaque_pixels": _pixel_stats(all_z_gt_opq, all_z_pr_opq, all_nd_opq, all_dh_opq, all_ls_opq),
        "corridor_pure_glass": corr_summary,
    }


# ---------------------------------------------------------------------------
# Pillar 3: Controlled "Ground-Contact Erasure" Counterfactual
# ---------------------------------------------------------------------------

def _floor_copy_erase(rgb_u8: np.ndarray, m_base: np.ndarray, m_upper: np.ndarray, m_floor_ref: np.ndarray) -> np.ndarray:
    """Replace `m_base` pixels row-wise with real adjacent floor texture without altering `m_upper`."""
    H, W, _ = rgb_u8.shape
    out = rgb_u8.copy()
    usable = (~m_base) & (~m_upper)
    floor_pref = usable & m_floor_ref

    ys_base = np.flatnonzero(m_base.any(axis=1))
    xs_grid = np.arange(W, dtype=np.int32)
    for y in ys_base:
        cols_b = np.flatnonzero(m_base[y])
        # Search nearby rows [y-4..y+4] for floor_pref pixels first, else usable row pixels
        ref_xs = np.flatnonzero(floor_pref[y])
        ref_rgb_row = rgb_u8[y]
        if len(ref_xs) < 12:
            # Look in narrow vertical window for clean floor texture
            y0, y1 = max(0, y - 8), min(H, y + 9)
            cand_yx = np.argwhere(floor_pref[y0:y1])
            if len(cand_yx) >= 12:
                pick = np.linspace(0, len(cand_yx) - 1, len(cols_b), dtype=np.int32)
                ry = y0 + cand_yx[pick, 0]
                rx = cand_yx[pick, 1]
                out[y, cols_b] = rgb_u8[ry, rx]
                continue
            ref_xs = np.flatnonzero(usable[y])
        if len(ref_xs) == 0:
            continue
        # Map each base col to nearest usable floor col with gentle horizontal mirroring
        pos = np.searchsorted(ref_xs, cols_b)
        pos_l = np.clip(pos - 1, 0, len(ref_xs) - 1)
        pos_r = np.clip(pos, 0, len(ref_xs) - 1)
        dl = np.abs(cols_b - ref_xs[pos_l])
        dr = np.abs(ref_xs[pos_r] - cols_b)
        nearest_x = np.where(dl <= dr, ref_xs[pos_l], ref_xs[pos_r])
        out[y, cols_b] = ref_rgb_row[nearest_x]

    # Mild boundary feather strictly outside m_upper so edges don't look razor-artificial
    edge_band = (cv2.dilate(m_base.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (~m_upper)
    blurred = cv2.GaussianBlur(out, (3, 3), 0)
    out = np.where(edge_band[:, :, None], blurred, out)
    out = np.where(m_upper[:, :, None], rgb_u8, out)
    return out


def _bg_control_mask(H: int, W: int, target_area: int, m_upper: np.ndarray, m_base: np.ndarray) -> np.ndarray | None:
    forb = cv2.dilate((m_upper | m_base).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    free = ~forb
    # Prefer upper-half/side patches of matched area
    side = int(round(math.sqrt(target_area)))
    for h_box in (max(12, side), max(12, side // 2), max(12, target_area // 64)):
        w_box = max(12, int(round(target_area / max(1, h_box))))
        for y0 in range(0, max(1, int(0.6 * H) - h_box + 1), 8):
            for x0 in (0, max(0, W - w_box), W // 4):
                x1 = min(W, x0 + w_box)
                y1 = min(H, y0 + h_box)
                patch = np.zeros((H, W), dtype=bool)
                patch[y0:y1, x0:x1] = True
                cand = patch & free
                if int(cand.sum()) >= int(0.75 * target_area):
                    return cand
    # Fallback: take top target_area pixels from upper free area
    ys, xs = np.nonzero(free & (np.arange(H)[:, None] < int(0.65 * H)))
    if len(ys) < int(0.6 * target_area):
        ys, xs = np.nonzero(free)
    if len(ys) < 150:
        return None
    take = min(len(ys), target_area)
    cand = np.zeros((H, W), dtype=bool)
    cand[ys[:take], xs[:take]] = True
    return cand


def _m6_1f_ha70(dep_s4: np.ndarray, dh_s4: np.ndarray, nrm_s4: np.ndarray, ls_s4: np.ndarray, K: np.ndarray, R_wc: np.ndarray, t_w: np.ndarray, fwd: np.ndarray) -> float:
    m6 = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    return float(m6.step(dep_s4, dh_s4, nrm_s4, ls_s4, K, R_wc, t_w, fwd))


def run_pillar3(model: EgoGeomOmniV41Net, device: torch.device, cached_clips: dict[str, list[dict]], batch_size: int = 32) -> dict:
    print("\n================ [PILLAR 3] Ground-Contact Erasure Counterfactual ================", flush=True)
    H4, W4 = H_IN // 4, W_IN // 4
    vv_f, uu_f = np.mgrid[0:H_IN, 0:W_IN].astype(np.float32)

    def _select_from_pool(
        pool_list: list[dict],
        target_n: int,
        tag: str,
        pass_cfgs: tuple[tuple[int, int], ...] = ((2, 10), (4, 4)),
    ) -> list[dict]:
        selected = []
        seen_keys: set[tuple[str, int]] = set()
        for max_per_clip, min_gap in pass_cfgs:
            per_clip_counts: dict[str, int] = defaultdict(int)
            last_idx_map: dict[str, int] = {}
            for item in selected:
                per_clip_counts[item["clip"]] += 1
                last_idx_map[item["clip"]] = max(last_idx_map.get(item["clip"], -100), item["frame"])
            for c in pool_list:
                if per_clip_counts[c["stem"]] >= max_per_clip:
                    continue
                for i in range(0, c["T"], 1):
                    if (c["stem"], i) in seen_keys:
                        continue
                    last_i = last_idx_map.get(c["stem"], -100)
                    if (i - last_i) < min_gap:
                        continue
                    if not c["valid"][i]:
                        continue
                    dm, dl = float(c["dm"][i]), float(c["dl"][i])
                    if not (np.isfinite(dm) and np.isfinite(dl) and 0.6 <= dm <= 3.8 and dl <= 3.8 and abs(dl - dm) <= 1.0):
                        continue
                    K = c["Ks"][i]
                    R_wc = c["poses_R"][i]
                    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
                    dep_up = cv2.resize(c["dep_s4"][i], (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
                    nd_up = cv2.resize(np.abs(c["nrm_s4"][i, 1]), (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)

                    ok = np.isfinite(dep_up) & (dep_up > 0.25) & (dep_up < 9.5)
                    P_cam = np.stack([(uu_f - cx) * dep_up / fx, (vv_f - cy) * dep_up / fy, dep_up], axis=-1)
                    P_rel = P_cam @ R_wc.T
                    z_rel = P_rel[..., 2]
                    fw = c["strict_fwd"][i, :2]
                    nf = float(np.linalg.norm(fw))
                    if nf < 1e-5:
                        continue
                    fw3 = np.array([fw[0] / nf, fw[1] / nf, 0.0], dtype=np.float32)
                    rt3 = np.array([fw3[1], -fw3[0], 0.0], dtype=np.float32)
                    z_fwd = P_rel @ fw3
                    x_lat = P_rel @ rt3

                    strip = ok & (np.abs(x_lat) <= 1.2) & (z_fwd >= 0.8) & (z_fwd <= 4.5)
                    g0 = float(np.percentile(z_rel[strip], 8)) if int(strip.sum()) >= 300 else -1.60
                    if not (-2.6 <= g0 <= -0.8):
                        g0 = -1.60
                    h = z_rel - g0

                    raw_upper = ok & (np.abs(x_lat) <= 0.60) & (h >= 0.70) & (h <= 1.90) & (z_fwd <= 4.5) & (nd_up < 0.80)
                    if int(raw_upper.sum()) < 250:
                        continue
                    x_lo, x_hi = np.percentile(x_lat[raw_upper], [5, 95])
                    z_lo, z_hi = np.percentile(z_fwd[raw_upper], [5, 95])
                    m_base = (
                        ok
                        & (x_lat >= max(-0.85, float(x_lo) - 0.15))
                        & (x_lat <= min(0.85, float(x_hi) + 0.15))
                        & (h >= 0.02)
                        & (h <= 0.68)
                        & (z_fwd >= max(0.30, float(z_lo) - 0.60))
                        & (z_fwd <= min(4.80, float(z_hi) + 0.60))
                    )
                    forb_base = cv2.dilate(m_base.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                    m_upper = raw_upper & (~forb_base)

                    if int(m_base.sum()) < 300 or int(m_upper.sum()) < 250:
                        continue
                    if int(m_upper[::4, ::4].sum()) < 15:
                        continue
                    m_bg = _bg_control_mask(H_IN, W_IN, int(m_base.sum()), m_upper, m_base)
                    if m_bg is None:
                        continue
                    m_floor_ref = ok & (nd_up >= 0.75) & (h < 0.15)
                    selected.append({
                        "domain": tag,
                        "clip": c["stem"],
                        "frame": i,
                        "d_gt_midup": dm,
                        "rgb_orig": c["rgbs_u8"][i],
                        "K": K,
                        "R_wc": R_wc,
                        "fwd": c["strict_fwd"][i],
                        "m_base": m_base,
                        "m_upper": m_upper,
                        "m_upper_s4": m_upper[::4, ::4],
                        "m_bg": m_bg,
                        "m_floor_ref": m_floor_ref,
                    })
                    seen_keys.add((c["stem"], i))
                    per_clip_counts[c["stem"]] += 1
                    last_idx_map[c["stem"]] = i
                    if per_clip_counts[c["stem"]] >= max_per_clip or len(selected) >= target_n:
                        break
                if len(selected) >= target_n:
                    break
            if len(selected) >= target_n:
                break
        return selected

    sel_aria = _select_from_pool(cached_clips.get("aria_gen2", []), 30, "aria_gen2", pass_cfgs=((3, 6), (6, 3)))
    sanpo_pool = cached_clips.get("fresh_sanpo_real_unseen", []) + cached_clips.get("sanpo_real_zs40", [])
    sel_sanpo = _select_from_pool(sanpo_pool, 35, "sanpo_real", pass_cfgs=((2, 10), (4, 5)))
    sel_synth = _select_from_pool(cached_clips.get("fresh_sanpo_synth_unseen", []), 20, "sanpo_synth", pass_cfgs=((2, 10), (4, 5)))
    all_sel = sel_aria + sel_sanpo + sel_synth
    print(f"[Pillar 3] Selected N={len(all_sel)} frames (Aria={len(sel_aria)}, SANPO-Real={len(sel_sanpo)}, SANPO-Synth={len(sel_synth)})", flush=True)
    if not all_sel:
        raise RuntimeError("No counterfactual frames selected!")

    # Build 4 conditions & run inference
    cond_names = ("orig", "inpaint_base", "floor_copy_base", "control_bg_inpaint")
    cond_results = {cn: [] for cn in cond_names}
    sanpo_vis_rows = []

    for rec in all_sel:
        rgb0 = rec["rgb_orig"]
        mu = rec["m_upper"]
        mb = rec["m_base"]
        mbg = rec["m_bg"]

        rgb_inp = cv2.inpaint(rgb0, mb.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        rgb_inp = np.where(mu[:, :, None], rgb0, rgb_inp)

        rgb_flc = _floor_copy_erase(rgb0, mb, mu, rec["m_floor_ref"])

        rgb_ctrl = cv2.inpaint(rgb0, mbg.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        rgb_ctrl = np.where(mu[:, :, None], rgb0, rgb_ctrl)

        imgs = {"orig": rgb0, "inpaint_base": rgb_inp, "floor_copy_base": rgb_flc, "control_bg_inpaint": rgb_ctrl}
        for cn, im in imgs.items():
            assert np.array_equal(im[mu], rgb0[mu]), f"Pixel identity violated in M_upper for {cn}!"
        rec["imgs"] = imgs

    # Batch run v4.1 on all 4*N images
    for cn in cond_names:
        rgbs_n = [((r["imgs"][cn].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1) for r in all_sel]
        Ks_a = np.stack([r["K"] for r in all_sel]).astype(np.float32)
        gs_a = np.stack([
            (r["R_wc"].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)) /
            (np.linalg.norm(r["R_wc"].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)) + 1e-6)
            for r in all_sel
        ]).astype(np.float32)
        dep_s4, dh_s4, nrm_s4, ls_s4 = infer_rgb_tensors(model, device, rgbs_n, Ks_a, gs_a, batch_size=batch_size)
        cond_results[cn] = (dep_s4, dh_s4, nrm_s4, ls_s4)

    dep0, dh0, nrm0, ls0 = cond_results["orig"]

    def _summarize_cf(subset_idx: list[int]) -> dict:
        out = {"n_frames": len(subset_idx)}
        if not subset_idx:
            return out
        out["median_base_px"] = int(np.median([int(all_sel[i]["m_base"].sum()) for i in subset_idx]))
        out["median_upper_px"] = int(np.median([int(all_sel[i]["m_upper"].sum()) for i in subset_idx]))
        out["median_gt_midup_m"] = round(float(np.median([all_sel[i]["d_gt_midup"] for i in subset_idx])), 3)
        # Precompute orig upper median & m6_1f
        z_orig_med = []
        s_orig_1f = []
        t_zero = np.zeros(3, dtype=np.float32)
        for idx in subset_idx:
            r = all_sel[idx]
            mu4 = r["m_upper_s4"]
            z_orig_med.append(float(np.median(dep0[idx][mu4])))
            s_orig_1f.append(_m6_1f_ha70(dep0[idx], dh0[idx], nrm0[idx], ls0[idx], r["K"], r["R_wc"], t_zero, r["fwd"]))
        z_orig_med = np.array(z_orig_med)
        s_orig_1f = np.array(s_orig_1f)

        for cn in cond_names:
            dep_c, dh_c, nrm_c, ls_c = cond_results[cn]
            dz_meds, rel_pcts, sc_ratios, err_gt, z_meds, sc_nz = [], [], [], [], [], []
            for k_pos, idx in enumerate(subset_idx):
                r = all_sel[idx]
                mu4 = r["m_upper_s4"]
                zo = dep0[idx][mu4]
                zc = dep_c[idx][mu4]
                dz = zc - zo
                dz_meds.append(float(np.median(dz)))
                rel_pcts.append(float(np.median(dz / np.maximum(zo, 0.2))) * 100.0)
                zc_m = float(np.median(zc))
                z_meds.append(zc_m)
                err_gt.append(abs(zc_m - r["d_gt_midup"]))
                sc_c = _m6_1f_ha70(dep_c[idx], dh_c[idx], nrm_c[idx], ls_c[idx], r["K"], r["R_wc"], t_zero, r["fwd"])
                sc_nz.append(1.0 if sc_c > 1e-3 else 0.0)
                if s_orig_1f[k_pos] > 1e-4:
                    sc_ratios.append(float(sc_c / s_orig_1f[k_pos]))
            dz_a = np.array(dz_meds)
            rel_a = np.array(rel_pcts)
            out[cn] = {
                "median_dz_m": round(float(np.median(dz_a)), 4),
                "mean_dz_m": round(float(np.mean(dz_a)), 4),
                "median_rel_shift_pct": round(float(np.median(rel_a)), 2),
                "collapse_rate_05m": round(float(np.mean(dz_a >= 0.50)), 4),
                "collapse_rate_25pct": round(float(np.mean(rel_a >= 25.0)), 4),
                "abs_err_vs_gt_midup_m": round(float(np.mean(err_gt)), 3),
                "m6_ha70_1f_score_median_ratio": round(float(np.median(sc_ratios)), 4) if sc_ratios else 1.0,
                "m6_ha70_1f_nonzero_rate": round(float(np.mean(sc_nz)), 4),
            }
        return out

    idx_all = list(range(len(all_sel)))
    idx_aria = [i for i, r in enumerate(all_sel) if r["domain"] == "aria_gen2"]
    idx_sanpo = [i for i, r in enumerate(all_sel) if r["domain"] == "sanpo_real"]
    idx_synth = [i for i, r in enumerate(all_sel) if r["domain"] == "sanpo_synth"]
    idx_real = idx_aria + idx_sanpo

    # Generate visual contact sheet from SANPO-Real ONLY (top 6 by |dz| under floor_copy_base)
    dep_flc = cond_results["floor_copy_base"][0]
    sanpo_ranked = sorted(
        idx_sanpo,
        key=lambda i: abs(float(np.median(dep_flc[i][all_sel[i]["m_upper_s4"]] - dep0[i][all_sel[i]["m_upper_s4"]]))),
        reverse=True,
    )[:6]
    if sanpo_ranked:
        fig, axes = plt.subplots(len(sanpo_ranked), 4, figsize=(18, 3.2 * len(sanpo_ranked)))
        if len(sanpo_ranked) == 1:
            axes = axes[None, :]
        fig.suptitle("Counterfactual Ground-Contact Erasure (SANPO-Real ONLY — ZERO Aria Pixels)", fontsize=13)
        for row_k, idx in enumerate(sanpo_ranked):
            r = all_sel[idx]
            mu4 = r["m_upper_s4"]
            dz_val = float(np.median(dep_flc[idx][mu4] - dep0[idx][mu4]))
            rel_v = float(np.median((dep_flc[idx][mu4] - dep0[idx][mu4]) / np.maximum(dep0[idx][mu4], 0.2))) * 100.0
            axes[row_k, 0].imshow(r["imgs"]["orig"])
            axes[row_k, 0].set_title(f"{r['clip'][:12]}#{r['frame']} (GT={r['d_gt_midup']:.2f}m)", fontsize=8.5)
            axes[row_k, 1].imshow(r["imgs"]["floor_copy_base"])
            axes[row_k, 1].set_title(f"Floor-Copy Erased (px_b={int(r['m_base'].sum())})", fontsize=8.5)
            axes[row_k, 2].imshow(np.clip(dep0[idx], 0.3, 6.0), cmap="turbo_r", vmin=0.3, vmax=6.0)
            axes[row_k, 2].set_title(f"Orig Depth (med={np.median(dep0[idx][mu4]):.2f}m)", fontsize=8.5)
            axes[row_k, 3].imshow(np.clip(dep_flc[idx], 0.3, 6.0), cmap="turbo_r", vmin=0.3, vmax=6.0)
            axes[row_k, 3].set_title(f"Erased Depth (dz={dz_val:+.2f}m, {rel_v:+.0f}%)", fontsize=8.5)
            for c_j in range(4):
                axes[row_k, c_j].axis("off")
        plt.tight_layout()
        cf_sheet = OUT_DIR / "counterfactual_base_erase_sheet.png"
        fig.savefig(cf_sheet, dpi=140)
        plt.close(fig)
        print(f"[Pillar 3] Saved counterfactual sheet {cf_sheet}", flush=True)

    return {
        "pooled_real": _summarize_cf(idx_real),
        "pooled_all": _summarize_cf(idx_all),
        "aria_gen2": _summarize_cf(idx_aria),
        "sanpo_real": _summarize_cf(idx_sanpo),
        "sanpo_synth": _summarize_cf(idx_synth),
    }


def _scan_finite(obj, p="root"):
    if isinstance(obj, float):
        if not np.isfinite(obj):
            raise RuntimeError(f"Non-finite at {p}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _scan_finite(v, f"{p}.{k}")
    elif isinstance(obj, list):
        for idx_i, v in enumerate(obj):
            _scan_finite(v, f"{p}[{idx_i}]")


def main() -> None:
    global WORK, DATA_DIR, TAR_DIR, GLASS_DIR, OUT_DIR, ARIA_DIR_OVERRIDE, ARIA_GT_PATH, CKPT_PATH_OVERRIDE, RCLONE_BIN

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run on local workstation GPU instead of /content Colab paths")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size (default: 8 for --local, 32 otherwise)")
    parser.add_argument("--keep-stage", action="store_true", help="Keep downloaded GlassRecon & tar staging dirs after finish")
    parser.add_argument("--pillars", type=str, default="1,2,3", help="Comma-separated pillars to run, e.g. '1,3' or '2'")
    args = parser.parse_args()

    pillars = {x.strip() for x in args.pillars.split(",") if x.strip()}
    if not pillars or not pillars.issubset({"1", "2", "3"}):
        raise ValueError(f"Invalid --pillars: {args.pillars}")
    if "3" in pillars and "1" not in pillars:
        raise ValueError("Pillar 3 depends on Pillar 1 cached_clips; pass --pillars 1,3")

    batch_size = args.batch_size if args.batch_size is not None else (8 if args.local else 32)
    if args.local:
        RCLONE_BIN = _resolve_rclone()
        local_probe = Path(r"e:\linnan\linnan\artifacts.local\work\egogeom-omni-v0\cane_blind_band_probe")
        WORK = local_probe / "_stage"
        DATA_DIR = WORK / "data"
        TAR_DIR = WORK / "tars"
        GLASS_DIR = WORK / "glassrecon"
        OUT_DIR = local_probe
        ARIA_DIR_OVERRIDE = Path(r"e:\linnan\linnan\artifacts.local\work\egogeom-omni-v0\zeroshot_eval\aria_g2_20clips")
        ARIA_GT_PATH = local_probe / "aria_height_band_gt.npz"
        CKPT_PATH_OVERRIDE = Path(r"e:\linnan\linnan\artifacts.local\work\egogeom-omni-v0\egogeom_omni_v41_pure_rgb_best.pt")

    t0 = time.time()
    ckpt_path = stage_all_assets(is_local=args.local, pillars=pillars)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[model] Loading {ckpt_path.name} on {gpu_name} (batch_size={batch_size}, pillars={sorted(pillars)})", flush=True)
    model = load_v41_model(ckpt_path, device)

    p13_path = OUT_DIR / "pillar13_partial.json"
    p2_path = OUT_DIR / "pillar2_partial.json"

    if "1" in pillars:
        t_p1 = time.time()
        p1_report, cached_clips = run_pillar1(model, device, batch_size=batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        p3_report = run_pillar3(model, device, cached_clips, batch_size=batch_size) if "3" in pillars else None
        del cached_clips
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        p13_payload = {
            "meta_p13": {
                "checkpoint": ckpt_path.name,
                "gpu": gpu_name,
                "mode": "local" if args.local else "colab",
                "batch_size": int(batch_size),
                "elapsed_s": round(time.time() - t_p1, 2),
            },
            "pillar1_overhang_mining": p1_report,
            "pillar3_ground_contact_counterfactual": p3_report,
        }
        _scan_finite(p13_payload, "pillar13")
        p13_path.write_text(json.dumps(p13_payload, indent=2), encoding="utf-8")
        print(f"[done] Saved {p13_path} ({p13_path.stat().st_size / 1e3:.1f} KB); 0 NaN/Inf!", flush=True)

    if "2" in pillars:
        t_p2 = time.time()
        p2_report = run_pillar2(model, device, batch_size=batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        p2_payload = {
            "meta_p2": {
                "checkpoint": ckpt_path.name,
                "gpu": gpu_name,
                "mode": "local" if args.local else "colab",
                "batch_size": int(batch_size),
                "elapsed_s": round(time.time() - t_p2, 2),
            },
            "pillar2_glassrecon_val": p2_report,
        }
        _scan_finite(p2_payload, "pillar2")
        p2_path.write_text(json.dumps(p2_payload, indent=2), encoding="utf-8")
        print(f"[done] Saved {p2_path} ({p2_path.stat().st_size / 1e3:.1f} KB); 0 NaN/Inf!", flush=True)

    if p13_path.exists() and p2_path.exists():
        p13_d = json.loads(p13_path.read_text(encoding="utf-8"))
        p2_d = json.loads(p2_path.read_text(encoding="utf-8"))
        if p13_d.get("pillar3_ground_contact_counterfactual") is None:
            print("[merge] pillar13_partial.json missing Pillar 3; skipping final merge.", flush=True)
        else:
            p13_el = float(p13_d["meta_p13"]["elapsed_s"])
            p2_el = float(p2_d["meta_p2"]["elapsed_s"])
            report = {
                "meta": {
                    "checkpoint": ckpt_path.name,
                    "gpu": gpu_name,
                    "mode": "local" if args.local else "colab",
                    "batch_size": int(batch_size),
                    "p13_elapsed_s": p13_el,
                    "p2_elapsed_s": p2_el,
                    "total_elapsed_s": round(p13_el + p2_el, 2),
                },
                "pillar1_overhang_mining": p13_d["pillar1_overhang_mining"],
                "pillar2_glassrecon_val": p2_d["pillar2_glassrecon_val"],
                "pillar3_ground_contact_counterfactual": p13_d["pillar3_ground_contact_counterfactual"],
            }
            _scan_finite(report, "merged")
            out_json = OUT_DIR / "overhang_glass_cf_report.json"
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\n[done] Merged & saved {out_json} ({out_json.stat().st_size / 1e3:.1f} KB); 0 NaN/Inf!", flush=True)

            # Back up ONLY the 4 Drive-safe files (report JSON + 3 SANPO/GlassRecon PNGs; NEVER upload Aria files!)
            for fname in SAFE_UPLOAD_FILES:
                assert "aria" not in fname.lower(), f"Privacy guard tripped for {fname}"
                fpath = OUT_DIR / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Expected safe output missing: {fpath}")
                run_rclone_copyto(fpath, f"{DRIVE_BACKUP}/{fname}")
            print(f"[backup] Uploaded {len(SAFE_UPLOAD_FILES)} safe outputs to {DRIVE_BACKUP}", flush=True)

            if args.local and not args.keep_stage:
                for clean_dir in (GLASS_DIR, TAR_DIR):
                    if clean_dir.exists():
                        print(f"[cleanup] Removing staging dir {clean_dir}", flush=True)
                        shutil.rmtree(clean_dir, ignore_errors=True)
    else:
        print(f"[stage-run] Finished pillars={sorted(pillars)}; waiting for complementary partial before final merge.", flush=True)


if __name__ == "__main__":
    main()
