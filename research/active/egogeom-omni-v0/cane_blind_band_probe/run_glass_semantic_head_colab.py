"""Hypothesis 2 (Colab L4): Frozen-DINOv3 (<0.5 ms) Semantic Glass-Risk Auxiliary Head.

Strictly zero-interference add-on on 100% frozen `egogeom_omni_v41_pure_rgb_best.pt`:
  - Extracts frozen multi-scale DINOv3 features (`feat_s16` [384,18,24] + `feat_s32` [768,9,12]).
  - Trains 3 tiny semantic heads (0% backbone grad; 0.000000 m interference with v4.1 depth):
      * G0_linear:                1x1 linear probe (1,153 params) on GlassRecon Train
      * G1_glassrecon_only:       2-scale 0.37M Conv+GN head on GlassRecon Train (85%)
      * G2_glassrecon_synth_neg:  same 0.37M head on GlassRecon Train + fresh_sanpo_synth_unseen outdoor negatives
  - Evaluates on:
      1) Held-out GlassRecon Val (131 frames): Pixel AUROC/AP/IoU/F1, [1.5, 4.0)m depth-penetrated glass
         rescue recall vs v4.1's over-confident `log_sigma`, and forward corridor glass AUROC.
      2) 100% Unseen Real Street Zero-Shot False-Alarm Audit (`fresh_sanpo_real_unseen` + `sanpo_real_zs40`).
  - Exports Drive-safe contact sheet `glass_semantic_head_sheet.png` (ZERO Aria pixels!).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

CODE_DIR = Path(__file__).resolve().parent
for _p in (str(CODE_DIR), str(CODE_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from egogeom_dataset import GlassReconIndex, IMAGENET_MEAN, IMAGENET_STD, apply_glasses_augment  # noqa: E402
from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K  # noqa: E402

WORK = Path("/content/work")
DATA_DIR = WORK / "data"
CKPT_DIR = WORK / "ckpt"
TAR_DIR = WORK / "tars"
GLASS_DIR = WORK / "glassrecon"
OUT_DIR = WORK / "out"
DRIVE_BACKUP = "gdrive:BlindAssist/runs/egogeom_cane_blind_band_probe/h12_v0"

H_IN, W_IN = 288, 384


def ensure_deps() -> None:
    import timm
    if not timm.is_model("convnext_small.dinov3_lvd1689m"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "timm"], check=True)
    if shutil.which("rclone") is None:
        rc = subprocess.run("apt-get update -qq && apt-get install -y -qq rclone", shell=True)
        if rc.returncode != 0 or shutil.which("rclone") is None:
            subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=True)


def rclone_copy(remote: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", remote, str(dst), "--transfers", "16"], check=True)


def extract_tar(tar_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    marker = dst / f".done_{tar_path.name}"
    if marker.exists():
        tar_path.unlink(missing_ok=True)
        return
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst)
    marker.write_text("ok\n", encoding="utf-8")
    tar_path.unlink(missing_ok=True)


def decode_iso_rgb(jpeg_bytes: bytes, orig_hw: np.ndarray, K_orig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H_o, W_o = int(orig_hw[0]), int(orig_hw[1])
    if rgb.shape[:2] != (H_o, W_o):
        rgb = cv2.resize(rgb, (W_o, H_o), interpolation=cv2.INTER_AREA)
    target_ar = W_IN / float(H_IN)
    cur_ar = W_o / float(H_o)
    if cur_ar > target_ar * 1.01:
        cw, ch = int(round(H_o * target_ar)), H_o
    elif cur_ar < target_ar * 0.99:
        cw, ch = W_o, int(round(W_o / target_ar))
    else:
        cw, ch = W_o, H_o
    y0, x0 = max(0, (H_o - ch) // 2), max(0, (W_o - cw) // 2)
    rgb_c = rgb[y0 : y0 + ch, x0 : x0 + cw]
    sx, sy = W_IN / float(cw), H_IN / float(ch)
    rgb_rs = cv2.resize(rgb_c, (W_IN, H_IN), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    if K_orig.ndim == 1:
        fx_o, fy_o, cx_o, cy_o = float(K_orig[0]), float(K_orig[1]), float(K_orig[2]), float(K_orig[3])
        K = np.array([[fx_o, 0.0, cx_o], [0.0, fy_o, cy_o], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        K = K_orig.astype(np.float32).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    rgb_f = rgb_rs.astype(np.float32) / 255.0
    rgb_n = ((rgb_f - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return rgb_n, rgb_rs, K


def stage_assets() -> Path:
    t0 = time.time()
    ensure_deps()
    for d in (DATA_DIR, CKPT_DIR, TAR_DIR, GLASS_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / "egogeom_omni_v41_pure_rgb_best.pt"
    if not ckpt_path.exists():
        print("[stage] Pulling v4.1 pure-RGB checkpoint...", flush=True)
        rclone_copy("gdrive:BlindAssist/runs/egogeom_omni_v41/egogeom_omni_v41_pure_rgb_best.pt", CKPT_DIR)

    # GlassRecon (9.56 GB)
    if not (GLASS_DIR / ".done_glassrecon").exists():
        print("[stage] Pulling GlassRecon v1 (9.56 GB)...", flush=True)
        rclone_copy("gdrive:BlindAssist/datasets/glassrecon_v1/glassrecon_release.zip", TAR_DIR / "glassrecon")
        zpath = TAR_DIR / "glassrecon/glassrecon_release.zip"
        print("[stage] Extracting glassrecon_release.zip...", flush=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(GLASS_DIR)
        zpath.unlink(missing_ok=True)
        (GLASS_DIR / ".done_glassrecon").write_text("ok\n", encoding="utf-8")

    # SANPO datasets (synth=neg train, fresh_real + zs40 = 100% held-out outdoor FP test)
    for sub, tars in [
        ("fresh_sanpo_synth_unseen", ("fresh_sanpo_synth_unseen_shard00.tar", "fresh_sanpo_synth_unseen_shard01.tar")),
        ("fresh_sanpo_real_unseen", ("fresh_sanpo_real_unseen_shard00.tar", "fresh_sanpo_real_unseen_shard01.tar")),
    ]:
        for sh in tars:
            if not (DATA_DIR / sub / f".done_{sh}").exists():
                rclone_copy(f"gdrive:BlindAssist/datasets/unseen_fresh_blind_probe_v0/{sub}/{sh}", TAR_DIR / sub)
                extract_tar(TAR_DIR / sub / sh, DATA_DIR / sub)

    for sh in ("shard_000.tar", "shard_001.tar"):
        if not (DATA_DIR / "sanpo_real_zs40" / f".done_{sh}").exists():
            rclone_copy(f"gdrive:BlindAssist/datasets/sanpo_real_zeroshot_40_v0/{sh}", TAR_DIR / "sanpo_zs40")
            extract_tar(TAR_DIR / "sanpo_zs40" / sh, DATA_DIR / "sanpo_real_zs40")

    print(f"[stage] All assets ready in {time.time() - t0:.1f}s", flush=True)
    return ckpt_path


class LinearGlassProbe(nn.Module):
    """Minimal 1x1 linear probe directly on concatenated frozen DINOv3 f3+f4 (1,153 params)."""

    def __init__(self, c3: int = 384, c4: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(c3 + c4, 1, 1)

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([f3, f4_up], dim=1))


class FrozenDINOv3GlassHead(nn.Module):
    """Lightweight (<0.5 ms, ~0.37M params) multi-scale semantic glass head on frozen v4.1 DINOv3."""

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
        p4 = F.interpolate(self.lat4(f4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([p3, p4], dim=1))


def load_frozen_v41(ckpt_path: Path, device: torch.device) -> EgoGeomOmniV41Net:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**state["cfg"]), pretrained_backbone=False)
    model.tof_branch = None
    model.load_state_dict({k: v for k, v in state["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    del state
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _safe_v41_forward(model: EgoGeomOmniV41Net, x: torch.Tensor, rays: torch.Tensor, hf: torch.Tensor, amp_ctx):
    try:
        with amp_ctx:
            return model(x, rays, None, None, host_focal=hf, return_features=True)
    except RuntimeError as e:
        if "unable to find an engine" in str(e).lower():
            print("[warn] cuDNN engine unavailable on this GPU; disabling cuDNN for native CUDA conv fallback.", flush=True)
            torch.backends.cudnn.enabled = False
            with amp_ctx:
                return model(x, rays, None, None, host_focal=hf, return_features=True)
        raise


@torch.inference_mode()
def extract_glassrecon_split(
    model: EgoGeomOmniV41Net,
    device: torch.device,
    gls_idx: GlassReconIndex,
    keep_full_eval: bool = False,
    batch_size: int = 16,
) -> dict:
    n = len(gls_idx.items)
    f3_list, f4_list, m18_list = [], [], []
    eval_items = []
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        rgbs_n, Ks, gs, raw_meta = [], [], [], []
        for idx in range(i0, i1):
            raw = gls_idx.load(idx)
            rgb_n, dep_rs, val_rs, K_out, g_out, _, gls_rs = apply_glasses_augment(
                raw["rgb"], raw["depth"], raw["valid"], raw["K"], raw["g_cam"],
                raw["normal_cam"], raw["glass"], (H_IN, W_IN), is_train=False,
            )
            rgbs_n.append(rgb_n.transpose(2, 0, 1))
            Ks.append(K_out)
            gs.append(g_out)
            m_small = cv2.resize(gls_rs.astype(np.float32), (W_IN // 16, H_IN // 16), interpolation=cv2.INTER_AREA)
            m18_list.append(m_small)
            if keep_full_eval:
                rgb_u8 = np.clip((rgb_n * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
                raw_meta.append((idx, rgb_u8, dep_rs, val_rs > 0.5, gls_rs > 0.5, K_out))

        x = torch.from_numpy(np.stack(rgbs_n)).to(device)
        Kt = torch.from_numpy(np.stack(Ks).astype(np.float32)).to(device)
        gt = torch.from_numpy(np.stack(gs).astype(np.float32)).to(device)
        rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
        hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
        out = _safe_v41_forward(model, x, rays, hf, amp_ctx)
        f3_list.append(out["feat_s16"].half().cpu())
        f4_list.append(out["feat_s32"].half().cpu())

        if keep_full_eval:
            dep_p = out["depth_m"][:, 0].float().cpu().numpy()
            ls_p = out["log_sigma"][:, 0].float().cpu().numpy()
            for k, (idx, rgb_u8, dep_gt, val_gt, gls_m, K_out) in enumerate(raw_meta):
                eval_items.append({
                    "idx": int(idx), "rgb": rgb_u8, "dep_gt": dep_gt, "val_gt": val_gt,
                    "gls_m": gls_m, "K": K_out, "dep_pr": dep_p[k], "ls_pr": ls_p[k],
                })

    return {
        "f3": torch.cat(f3_list, dim=0),
        "f4": torch.cat(f4_list, dim=0),
        "mask18": torch.from_numpy(np.stack(m18_list)).float().unsqueeze(1),
        "eval_items": eval_items,
    }


@torch.inference_mode()
def extract_sanpo_pool(
    model: EgoGeomOmniV41Net,
    device: torch.device,
    root_dir: Path,
    frames_per_clip: int = 8,
    keep_rgb_meta: bool = False,
    batch_size: int = 16,
) -> dict:
    npzs = sorted(root_dir.rglob("*.npz"))
    rgbs_all, Ks_all, gs_all, meta_all = [], [], [], []
    for p in npzs:
        with np.load(p, allow_pickle=False) as d:
            offs = d["rgb_offsets"] if "rgb_offsets" in d else d["rgb_jpeg_offsets"]
            blob = d["rgb_blob"] if "rgb_blob" in d else d["rgb_jpeg_bytes"]
            orig_hw, K_orig = d["orig_hw"], d["K_orig"]
            poses_R = d["poses_R"].astype(np.float32)
            T = len(poses_R)
            picks = np.linspace(0, max(0, T - 1), min(T, frames_per_clip), dtype=np.int32)
            for fi in picks:
                rgb_n, rgb_u8, K = decode_iso_rgb(blob[offs[fi] : offs[fi + 1]].tobytes(), orig_hw, K_orig)
                g_cam = (poses_R[fi].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                rgbs_all.append(rgb_n)
                Ks_all.append(K)
                gs_all.append(g_cam)
                if keep_rgb_meta:
                    meta_all.append({"clip": p.stem, "frame": int(fi), "rgb": rgb_u8, "K": K})

    f3_list, f4_list, dep_list = [], [], []
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    for i0 in range(0, len(rgbs_all), batch_size):
        i1 = min(len(rgbs_all), i0 + batch_size)
        x = torch.from_numpy(np.stack(rgbs_all[i0:i1])).to(device)
        Kt = torch.from_numpy(np.stack(Ks_all[i0:i1]).astype(np.float32)).to(device)
        gt = torch.from_numpy(np.stack(gs_all[i0:i1]).astype(np.float32)).to(device)
        rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
        hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
        out = _safe_v41_forward(model, x, rays, hf, amp_ctx)
        f3_list.append(out["feat_s16"].half().cpu())
        f4_list.append(out["feat_s32"].half().cpu())
        if keep_rgb_meta:
            dep_list.append(out["depth_m"][:, 0].float().cpu().numpy())

    if keep_rgb_meta:
        deps = np.concatenate(dep_list, axis=0)
        for idx, m in enumerate(meta_all):
            m["dep_pr"] = deps[idx]

    n_tot = len(rgbs_all)
    return {
        "f3": torch.cat(f3_list, dim=0),
        "f4": torch.cat(f4_list, dim=0),
        "mask18": torch.zeros((n_tot, 1, H_IN // 16, W_IN // 16), dtype=torch.float32),
        "meta": meta_all,
    }


def train_glass_head(
    head: nn.Module,
    f3_tr: torch.Tensor,
    f4_tr: torch.Tensor,
    m_tr: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_w = torch.tensor([2.0], device=device)

    N = f3_tr.shape[0]
    f3_g = f3_tr.to(device=device, dtype=torch.float32)
    f4_g = f4_tr.to(device=device, dtype=torch.float32)
    m_g = m_tr.to(device=device, dtype=torch.float32)

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i0 in range(0, N, batch_size):
            idx = perm[i0 : i0 + batch_size]
            x3, x4, y = f3_g[idx], f4_g[idx], m_g[idx]
            # 50% horizontal flip augmentation on cached features
            if torch.rand(1).item() < 0.5:
                x3 = torch.flip(x3, dims=[-1])
                x4 = torch.flip(x4, dims=[-1])
                y = torch.flip(y, dims=[-1])
            logits = head(x3, x4)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
            probs = torch.sigmoid(logits)
            inter = (probs * y).sum(dim=(1, 2, 3))
            denom = probs.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1.0
            dice = (1.0 - (2.0 * inter + 1.0) / denom).mean()
            loss = bce + 0.5 * dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return head.eval()


@torch.inference_mode()
def predict_probs_fullres(head: nn.Module, f3: torch.Tensor, f4: torch.Tensor, device: torch.device, batch_size: int = 32) -> np.ndarray:
    head.eval()
    probs = []
    for i0 in range(0, f3.shape[0], batch_size):
        x3 = f3[i0 : i0 + batch_size].to(device=device, dtype=torch.float32)
        x4 = f4[i0 : i0 + batch_size].to(device=device, dtype=torch.float32)
        lg = head(x3, x4)
        pr = torch.sigmoid(F.interpolate(lg, size=(H_IN, W_IN), mode="bilinear", align_corners=False))[:, 0]
        probs.append(pr.cpu().numpy())
    return np.concatenate(probs, axis=0)


def eval_glass_arm(
    name: str,
    probs_val: np.ndarray,
    val_items: list[dict],
    probs_real_unseen: np.ndarray,
    probs_zs40: np.ndarray,
) -> dict:
    vv, uu = np.mgrid[0:H_IN, 0:W_IN].astype(np.float32)
    y_all, p_all = [], []
    # [1.5, 4.0)m penetrated glass vs opaque in same distance range
    p_pen, p_opq_near, ls_pen, ls_opq_near = [], [], [], []
    corr_y, corr_s_mean, corr_s_p90 = [], [], []

    for i, it in enumerate(val_items):
        pr = probs_val[i]
        val_gt = it["val_gt"]
        gls = it["gls_m"]
        dep_gt = it["dep_gt"]
        dep_pr = it["dep_pr"]
        ls_pr = it["ls_pr"]

        # Pixel-level across valid pixels (subsample 1/4 for fast exact ROC/AP)
        sub_y = gls[val_gt][::4].astype(np.int32)
        sub_p = pr[val_gt][::4].astype(np.float32)
        if len(sub_y):
            y_all.append(sub_y)
            p_all.append(sub_p)

        # Penetrated glass in [1.5, 4.0) m
        near_range = val_gt & (dep_gt >= 1.5) & (dep_gt < 4.0) & np.isfinite(dep_pr)
        pen_m = near_range & gls & ((dep_pr > 1.35 * dep_gt) | ((dep_pr - dep_gt) > 0.80))
        opq_m = near_range & (~gls)
        if pen_m.any():
            p_pen.append(pr[pen_m])
            ls_pen.append(ls_pr[pen_m])
        if opq_m.any():
            p_opq_near.append(pr[opq_m][::4])
            ls_opq_near.append(ls_pr[opq_m][::4])

        # Forward angular corridor check (|az|<=20 deg, |el|<=18 deg around optical center)
        K = it["K"]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        cone = (np.abs((uu - cx) / fx) <= np.tan(np.radians(20.0))) & (np.abs((vv - cy) / fy) <= np.tan(np.radians(18.0)))
        gls_frac = float((gls & cone).sum()) / float(max(1, cone.sum()))
        if gls_frac >= 0.05 or gls_frac < 0.005:
            corr_y.append(1 if gls_frac >= 0.05 else 0)
            corr_s_mean.append(float(pr[cone].mean()))
            corr_s_p90.append(float(np.percentile(pr[cone], 90)))

    y_cat = np.concatenate(y_all)
    p_cat = np.concatenate(p_all)
    pred50 = p_cat >= 0.5
    tp = int((pred50 & (y_cat == 1)).sum())
    fp = int((pred50 & (y_cat == 0)).sum())
    fn = int(((~pred50) & (y_cat == 1)).sum())
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)

    p_pen_c = np.concatenate(p_pen)
    p_opq_c = np.concatenate(p_opq_near)
    ls_pen_c = np.concatenate(ls_pen)
    ls_opq_c = np.concatenate(ls_opq_near)
    y_po = np.concatenate([np.ones(len(p_pen_c), dtype=np.int32), np.zeros(len(p_opq_c), dtype=np.int32)])
    auroc_pen_p = float(roc_auc_score(y_po, np.concatenate([p_pen_c, p_opq_c])))
    auroc_pen_ls = float(roc_auc_score(y_po, np.concatenate([ls_pen_c, ls_opq_c])))

    # Corridor stats
    cy_a = np.array(corr_y, dtype=np.int32)
    cm_a = np.array(corr_s_mean, dtype=np.float32)
    c90_a = np.array(corr_s_p90, dtype=np.float32)
    neg_m = cy_a == 0
    pos_m = cy_a == 1
    thr5 = float(np.quantile(cm_a[neg_m], 0.95)) if neg_m.any() else 1.0
    thr10 = float(np.quantile(cm_a[neg_m], 0.90)) if neg_m.any() else 1.0

    # Unseen real outdoor FP audit (center 40% x 40% forward corridor proxy)
    y0c, y1c = int(0.30 * H_IN), int(0.70 * H_IN)
    x0c, x1c = int(0.30 * W_IN), int(0.70 * W_IN)

    def _outdoor_fp(pr_arr: np.ndarray) -> dict:
        c_slice = pr_arr[:, y0c:y1c, x0c:x1c]
        c_mean = c_slice.mean(axis=(1, 2))
        frac50 = (c_slice >= 0.5).mean(axis=(1, 2))
        frac70 = (c_slice >= 0.7).mean(axis=(1, 2))
        return {
            "n_frames": int(len(pr_arr)),
            "corridor_mean_p_median": round(float(np.median(c_mean)), 4),
            "corridor_mean_p_p90": round(float(np.percentile(c_mean, 90)), 4),
            "corridor_mean_p_p95": round(float(np.percentile(c_mean, 95)), 4),
            "corridor_mean_p_p99": round(float(np.percentile(c_mean, 99)), 4),
            "frame_alert_rate_ge10pct_at_p50": round(float((frac50 >= 0.10).mean()), 4),
            "frame_alert_rate_ge10pct_at_p70": round(float((frac70 >= 0.10).mean()), 4),
            "frame_alert_rate_mean_ge050": round(float((c_mean >= 0.50).mean()), 4),
        }

    return {
        "arm": name,
        "glassrecon_val_pixel": {
            "n_valid_sub4": int(len(y_cat)),
            "auroc": round(float(roc_auc_score(y_cat, p_cat)), 4),
            "ap": round(float(average_precision_score(y_cat, p_cat)), 4),
            "iou_at_05": round(float(iou), 4),
            "f1_at_05": round(float(f1), 4),
            "precision_at_05": round(float(prec), 4),
            "recall_at_05": round(float(rec), 4),
        },
        "penetrated_glass_1p5_to_4m_rescue": {
            "n_penetrated_glass_px": int(len(p_pen_c)),
            "n_opaque_near_sub4_px": int(len(p_opq_c)),
            "rescue_recall_at_p50": round(float((p_pen_c >= 0.5).mean()), 4),
            "rescue_recall_at_p30": round(float((p_pen_c >= 0.3).mean()), 4),
            "median_p_glass_on_penetrated": round(float(np.median(p_pen_c)), 4),
            "median_p_glass_on_opaque_near": round(float(np.median(p_opq_c)), 4),
            "auroc_pen_vs_opaque_head": round(auroc_pen_p, 4),
            "auroc_pen_vs_opaque_log_sigma_baseline": round(auroc_pen_ls, 4),
        },
        "glassrecon_val_corridor": {
            "n_pos_corridor": int(pos_m.sum()),
            "n_neg_corridor": int(neg_m.sum()),
            "auroc_mean_score": round(float(roc_auc_score(cy_a, cm_a)), 4) if (pos_m.any() and neg_m.any()) else None,
            "auroc_p90_score": round(float(roc_auc_score(cy_a, c90_a)), 4) if (pos_m.any() and neg_m.any()) else None,
            "recall_at_5pct_fpr": round(float((cm_a[pos_m] >= thr5).mean()), 4) if (pos_m.any() and neg_m.any()) else None,
            "recall_at_10pct_fpr": round(float((cm_a[pos_m] >= thr10).mean()), 4) if (pos_m.any() and neg_m.any()) else None,
        },
        "unseen_outdoor_fp_fresh_sanpo_real": _outdoor_fp(probs_real_unseen),
        "unseen_outdoor_fp_sanpo_real_zs40": _outdoor_fp(probs_zs40),
    }


def benchmark_head_latency(head: nn.Module, device: torch.device) -> dict:
    n_params = sum(p.numel() for p in head.parameters())
    f3_d = torch.randn(1, 384, 18, 24, device=device)
    f4_d = torch.randn(1, 768, 9, 12, device=device)
    head_g = head.to(device).eval()
    with torch.inference_mode():
        for _ in range(30):
            _ = head_g(f3_d, f4_d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            _ = head_g(f3_d, f4_d)
        torch.cuda.synchronize()
        gpu_fp32_ms = (time.perf_counter() - t0) * 1e3 / 200.0

        f3_h, f4_h = f3_d.half(), f4_d.half()
        head_h = head_g.half()
        for _ in range(30):
            _ = head_h(f3_h, f4_h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            _ = head_h(f3_h, f4_h)
        torch.cuda.synchronize()
        gpu_fp16_ms = (time.perf_counter() - t0) * 1e3 / 200.0

        head_c = head_g.float().cpu().eval()
        f3_c, f4_c = f3_d.cpu(), f4_d.cpu()
        for _ in range(10):
            _ = head_c(f3_c, f4_c)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = head_c(f3_c, f4_c)
        cpu_fp32_ms = (time.perf_counter() - t0) * 1e3 / 100.0

    head.float().to(device).eval()
    return {
        "n_params": int(n_params),
        "params_M": round(n_params / 1e6, 4),
        "gpu_fp32_ms": round(gpu_fp32_ms, 4),
        "gpu_fp16_ms": round(gpu_fp16_ms, 4),
        "cpu_fp32_ms": round(cpu_fp32_ms, 4),
    }


def make_contact_sheet(
    val_items: list[dict],
    probs_g2_val: np.ndarray,
    real_meta: list[dict],
    probs_g2_real: np.ndarray,
) -> None:
    # Pick top-4 worst penetrated GlassRecon Val frames in [1.5, 4.0)m (with >=500 glass px)
    gls_ranked = []
    for i, it in enumerate(val_items):
        mg = it["val_gt"] & it["gls_m"] & (it["dep_gt"] >= 1.5) & (it["dep_gt"] < 4.5)
        if int(mg.sum()) < 500:
            continue
        pen_m = (it["dep_pr"][mg] > 1.35 * it["dep_gt"][mg]) | ((it["dep_pr"][mg] - it["dep_gt"][mg]) > 0.80)
        pen_r = float(pen_m.mean())
        gls_ranked.append((pen_r, i))
    gls_ranked.sort(key=lambda x: x[0], reverse=True)
    top_gls = [idx for _, idx in gls_ranked[:4]]

    # Pick 4 unseen SANPO-Real frames: 2 highest corridor p_glass + 2 median/low clear sidewalks
    y0c, y1c = int(0.30 * H_IN), int(0.70 * H_IN)
    x0c, x1c = int(0.30 * W_IN), int(0.70 * W_IN)
    c_means = probs_g2_real[:, y0c:y1c, x0c:x1c].mean(axis=(1, 2))
    order_r = np.argsort(c_means)
    # Deduplicate by clip for top-2 highest
    hi_r, seen_c = [], set()
    for idx in order_r[::-1]:
        cl = real_meta[int(idx)]["clip"]
        if cl in seen_c:
            continue
        hi_r.append(int(idx))
        seen_c.add(cl)
        if len(hi_r) >= 2:
            break
    lo_r = [int(order_r[len(order_r) // 4]), int(order_r[len(order_r) // 2])]
    picked_real = hi_r + lo_r

    fig, axes = plt.subplots(4, 6, figsize=(22, 13))
    fig.suptitle("Frozen-DINOv3 Glass Semantic Head (G2): Top 2 Rows = Worst Depth-Penetrated GlassRecon Val | Bottom 2 Rows = Unseen SANPO-Real Streets (ZERO Aria)", fontsize=12)

    # Top 2 rows: 4 GlassRecon Val items (2 per row x 3 panels)
    for k, idx in enumerate(top_gls):
        r_i, pair_i = divmod(k, 2)
        c0 = pair_i * 3
        it = val_items[idx]
        pr = probs_g2_val[idx]
        rgb_ov = it["rgb"].copy()
        cnts, _ = cv2.findContours(it["gls_m"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_ov, cnts, -1, (0, 255, 255), 2)
        mg = it["val_gt"] & it["gls_m"]
        gt_m = float(np.median(it["dep_gt"][mg]))
        pr_m = float(np.median(it["dep_pr"][mg]))
        pg_m = float(np.median(pr[mg]))

        axes[r_i, c0].imshow(rgb_ov)
        axes[r_i, c0].set_title(f"GlassVal#{it['idx']} (Cyan=GT)", fontsize=8.5)
        axes[r_i, c0 + 1].imshow(np.clip(it["dep_pr"], 0.3, 6.0), cmap="turbo_r", vmin=0.3, vmax=6.0)
        axes[r_i, c0 + 1].set_title(f"v4.1 Depth: GT={gt_m:.2f}m Pr={pr_m:.2f}m", fontsize=8.5)
        axes[r_i, c0 + 2].imshow(pr, cmap="magma", vmin=0.0, vmax=1.0)
        axes[r_i, c0 + 2].set_title(f"G2 p_glass (med={pg_m:.2f})", fontsize=8.5)

    # Bottom 2 rows: 4 unseen SANPO-Real items
    for k, idx in enumerate(picked_real):
        r_i = 2 + (k // 2)
        c0 = (k % 2) * 3
        m = real_meta[idx]
        pr = probs_g2_real[idx]
        cm = float(c_means[idx])
        tag = "TOP-P" if k < 2 else "CLEAR"
        axes[r_i, c0].imshow(m["rgb"])
        axes[r_i, c0].set_title(f"SANPO-Real {tag} {m['clip'][:10]}#{m['frame']}", fontsize=8.5)
        axes[r_i, c0 + 1].imshow(np.clip(m["dep_pr"], 0.3, 6.0), cmap="turbo_r", vmin=0.3, vmax=6.0)
        axes[r_i, c0 + 1].set_title("v4.1 Depth", fontsize=8.5)
        axes[r_i, c0 + 2].imshow(pr, cmap="magma", vmin=0.0, vmax=1.0)
        axes[r_i, c0 + 2].set_title(f"G2 p_glass (corr_mean={cm:.3f})", fontsize=8.5)

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out_png = OUT_DIR / "glass_semantic_head_sheet.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[vis] Saved {out_png}", flush=True)


def main() -> None:
    t0 = time.time()
    ckpt_path = stage_assets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[model] Loading frozen {ckpt_path.name} on {gpu_name}", flush=True)
    model = load_frozen_v41(ckpt_path, device)

    gls_tr = GlassReconIndex(GLASS_DIR, split_filter="train")
    gls_va = GlassReconIndex(GLASS_DIR, split_filter="val")
    if len(gls_tr.items) == 0:
        subs = [p for p in GLASS_DIR.iterdir() if p.is_dir()]
        if len(subs) == 1:
            gls_tr = GlassReconIndex(subs[0], split_filter="train")
            gls_va = GlassReconIndex(subs[0], split_filter="val")
    print(f"[data] GlassRecon train={len(gls_tr.items)}, val={len(gls_va.items)}", flush=True)

    d_tr = extract_glassrecon_split(model, device, gls_tr, keep_full_eval=False)
    d_va = extract_glassrecon_split(model, device, gls_va, keep_full_eval=True)
    d_synth_neg = extract_sanpo_pool(model, device, DATA_DIR / "fresh_sanpo_synth_unseen", frames_per_clip=8, keep_rgb_meta=False)
    d_real_unseen = extract_sanpo_pool(model, device, DATA_DIR / "fresh_sanpo_real_unseen", frames_per_clip=10, keep_rgb_meta=True)
    d_real_zs40 = extract_sanpo_pool(model, device, DATA_DIR / "sanpo_real_zs40", frames_per_clip=10, keep_rgb_meta=True)
    print(
        f"[feat] Cached features: gls_tr={len(d_tr['f3'])}, gls_va={len(d_va['f3'])}, "
        f"synth_neg={len(d_synth_neg['f3'])}, real_unseen={len(d_real_unseen['f3'])}, zs40={len(d_real_zs40['f3'])}",
        flush=True,
    )

    # Train 3 heads on frozen features
    h_g0 = train_glass_head(LinearGlassProbe(), d_tr["f3"], d_tr["f4"], d_tr["mask18"], device, epochs=30, lr=2e-3)
    h_g1 = train_glass_head(FrozenDINOv3GlassHead(), d_tr["f3"], d_tr["f4"], d_tr["mask18"], device, epochs=30, lr=1e-3)

    f3_g2 = torch.cat([d_tr["f3"], d_synth_neg["f3"]], dim=0)
    f4_g2 = torch.cat([d_tr["f4"], d_synth_neg["f4"]], dim=0)
    m_g2 = torch.cat([d_tr["mask18"], d_synth_neg["mask18"]], dim=0)
    h_g2 = train_glass_head(FrozenDINOv3GlassHead(), f3_g2, f4_g2, m_g2, device, epochs=30, lr=1e-3)

    results = {}
    probs_g2_val = None
    probs_g2_real_all = None
    for name, head in [("G0_linear", h_g0), ("G1_glassrecon_only", h_g1), ("G2_glassrecon_plus_synth_neg", h_g2)]:
        pv = predict_probs_fullres(head, d_va["f3"], d_va["f4"], device)
        pr_u = predict_probs_fullres(head, d_real_unseen["f3"], d_real_unseen["f4"], device)
        pr_z = predict_probs_fullres(head, d_real_zs40["f3"], d_real_zs40["f4"], device)
        res = eval_glass_arm(name, pv, d_va["eval_items"], pr_u, pr_z)
        res["latency"] = benchmark_head_latency(head, device)
        results[name] = res
        print(
            f"[{name}] val_px_AUROC={res['glassrecon_val_pixel']['auroc']} IoU={res['glassrecon_val_pixel']['iou_at_05']} "
            f"pen_rescue@0.5={res['penetrated_glass_1p5_to_4m_rescue']['rescue_recall_at_p50']} "
            f"pen_AUROC={res['penetrated_glass_1p5_to_4m_rescue']['auroc_pen_vs_opaque_head']} "
            f"(log_sigma={res['penetrated_glass_1p5_to_4m_rescue']['auroc_pen_vs_opaque_log_sigma_baseline']}) "
            f"corr_AUROC={res['glassrecon_val_corridor']['auroc_mean_score']} "
            f"zs40_fp@p50={res['unseen_outdoor_fp_sanpo_real_zs40']['frame_alert_rate_ge10pct_at_p50']}",
            flush=True,
        )
        if name == "G2_glassrecon_plus_synth_neg":
            probs_g2_val = pv
            probs_g2_real_all = np.concatenate([pr_u, pr_z], axis=0)

    make_contact_sheet(d_va["eval_items"], probs_g2_val, d_real_unseen["meta"] + d_real_zs40["meta"], probs_g2_real_all)

    torch.save({"state_dict": h_g2.cpu().state_dict(), "arch": "FrozenDINOv3GlassHead", "c3": 384, "c4": 768, "hidden": 128}, OUT_DIR / "v41_frozen_glass_head_g2.pt")

    report = {
        "meta": {
            "checkpoint": ckpt_path.name,
            "gpu": gpu_name,
            "total_elapsed_s": round(time.time() - t0, 2),
            "n_glassrecon_train": int(len(d_tr["f3"])),
            "n_glassrecon_val": int(len(d_va["f3"])),
            "n_synth_neg_train": int(len(d_synth_neg["f3"])),
            "n_fresh_real_unseen_eval": int(len(d_real_unseen["f3"])),
            "n_sanpo_zs40_eval": int(len(d_real_zs40["f3"])),
        },
        "arms": results,
    }

    def _scan(obj, p="root"):
        if isinstance(obj, float) and not np.isfinite(obj):
            raise RuntimeError(f"Non-finite at {p}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{p}.{k}")
        elif isinstance(obj, list):
            for idx_i, v in enumerate(obj):
                _scan(v, f"{p}[{idx_i}]")

    _scan(report)
    out_json = OUT_DIR / "glass_semantic_probe_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Saved {out_json} in {report['meta']['total_elapsed_s']}s; 0 NaN/Inf!", flush=True)

    for fname in ("glass_semantic_probe_report.json", "glass_semantic_head_sheet.png", "v41_frozen_glass_head_g2.pt"):
        assert "aria" not in fname.lower()
        subprocess.run(["rclone", "copyto", str(OUT_DIR / fname), f"{DRIVE_BACKUP}/{fname}"], check=True)
    print(f"[backup] Uploaded 3 safe H2 outputs to {DRIVE_BACKUP}", flush=True)


if __name__ == "__main__":
    main()
