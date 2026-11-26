"""Zero-Shot Indoor (ARKitScenes val30) & Outdoor Campus (CODa) False-Positive / Hard-Case Audit for G2 Glass Head.

Runs 100% locally on RTX 5060 (`0.00 CCU`) using frozen `egogeom_omni_v41_pure_rgb_best.pt` + `v41_frozen_glass_head_g2.pt`.
Evaluates:
  - ARKitScenes val30: 30 real indoor apartments (stride=3 -> ~1,200 frames + consecutive pairs for 2-frame lock)
  - CODa campus:       24 real outdoor campus walks (stride=3 -> ~960 frames + consecutive pairs)
Computes corridor glass trigger rates (`@p50`, `@p70`, single-frame vs 2-frame confirmed) and exports a high-res
visual contact sheet (`g2_indoor_arkit_coda_audit_sheet.png`) of the top-12 highest-responding ARKitScenes indoor
frames + top-6 CODa campus frames so genuine glass/mirrors vs non-glass glossy/TV false alarms can be visually audited.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
ZS_DIR = ART / "zeroshot_eval"

sys.path[:0] = [str(HERE), str(ROUTE)]

from egogeom_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K  # noqa: E402

H_IN, W_IN = 288, 384


class FrozenDINOv3GlassHead(nn.Module):
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


def corridor_mask_s16() -> np.ndarray:
    """Central forward corridor mask matching H2 evaluation (rows 4..14, cols 6..18 on 18x24)."""
    m = np.zeros((18, 24), dtype=bool)
    m[4:14, 6:18] = True
    return m


@torch.inference_mode()
def audit_suite(
    suite_name: str,
    folder: Path,
    v41: EgoGeomOmniV41Net,
    g2: FrozenDINOv3GlassHead,
    device: torch.device,
    stride: int = 3,
    top_k_clips: int = 12,
) -> tuple[dict, list[dict]]:
    paths = sorted(folder.rglob("*.npz"))
    cmask = corridor_mask_s16()
    records = []
    # Keep best frame per clip so contact sheet spans distinct apartments/scenes!
    clip_best: dict[str, dict] = {}
    consec_p50_hits = 0
    consec_p70_hits = 0
    total_pairs = 0

    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            blob = d["rgb_blob"]
            offs = d["rgb_offsets"]
            orig_hw = d["orig_hw"]
            K_orig = d["K_orig"]
            poses_R = d["poses_R"]
            T = len(offs) - 1
            # Evaluate contiguous pairs (t, t+1) every `stride` frames so we get exact 2-frame confirmation rates!
            frame_ids = []
            for t in range(0, T - 1, stride):
                frame_ids.append(t)
                frame_ids.append(t + 1)
            frame_ids = sorted(set(frame_ids))

            rgbs_n, rgbs_u8, Ks, gs = [], [], [], []
            for fi in frame_ids:
                jb = bytes(blob[offs[fi] : offs[fi + 1]])
                rn, ru, K = decode_iso_rgb(jb, orig_hw, K_orig)
                # Gravity down in camera frame = R_wc^T @ [0, 0, -1]
                g_cam = (poses_R[fi].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32))
                rgbs_n.append(rn)
                rgbs_u8.append(ru)
                Ks.append(K)
                gs.append(g_cam)

        # Batch inference on RTX 5060
        probs_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        bs = 16
        for i0 in range(0, len(frame_ids), bs):
            i1 = min(len(frame_ids), i0 + bs)
            x = torch.from_numpy(np.stack(rgbs_n[i0:i1])).to(device, dtype=torch.float32)
            Kt = torch.from_numpy(np.stack(Ks[i0:i1])).to(device, dtype=torch.float32)
            gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device, dtype=torch.float32)
            rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
            hf = batch_host_focal_from_K(Kt, W_IN, H_IN).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = v41(x, rays, None, None, host_focal=hf, return_features=True)
                gp = g2(out["feat_s16"].float(), out["feat_s32"].float())
            gp_np = gp[:, 0].float().cpu().numpy()
            dep_np = out["depth_m"][:, 0].float().cpu().numpy()
            for j, fi in enumerate(frame_ids[i0:i1]):
                probs_map[fi] = (gp_np[j], dep_np[j])

        # Gather single-frame metrics on stride frames and 2-frame confirmation metrics on (t, t+1)
        idx_lookup = {fi: idx for idx, fi in enumerate(frame_ids)}
        for t in range(0, T - 1, stride):
            p0, dep0 = probs_map[t]
            p1, _ = probs_map[t + 1]
            c0 = p0[cmask]
            c1 = p1[cmask]
            m0 = float(c0.mean())
            f50_0 = float((c0 >= 0.50).mean())
            f70_0 = float((c0 >= 0.70).mean())
            f50_1 = float((c1 >= 0.50).mean())
            f70_1 = float((c1 >= 0.70).mean())

            records.append({"clip": p.stem, "frame": t, "corr_mean": m0, "area_p50": f50_0, "area_p70": f70_0})
            total_pairs += 1
            if f50_0 >= 0.10 and f50_1 >= 0.10:
                consec_p50_hits += 1
            if f70_0 >= 0.10 and f70_1 >= 0.10:
                consec_p70_hits += 1

            prev_best = clip_best.get(p.stem)
            if prev_best is None or m0 > prev_best["corr_mean"]:
                clip_best[p.stem] = {
                    "suite": suite_name,
                    "clip": p.stem,
                    "frame": t,
                    "corr_mean": m0,
                    "area_p50": f50_0,
                    "area_p70": f70_0,
                    "rgb": rgbs_u8[idx_lookup[t]],
                    "prob": p0,
                    "depth": dep0,
                }

    means = np.array([r["corr_mean"] for r in records])
    a50 = np.array([r["area_p50"] for r in records])
    a70 = np.array([r["area_p70"] for r in records])
    n = len(records)
    clips_with_p70 = sum(1 for v in clip_best.values() if v["area_p70"] >= 0.10)
    clips_with_p50 = sum(1 for v in clip_best.values() if v["area_p50"] >= 0.10)

    top_items = sorted(clip_best.values(), key=lambda x: x["corr_mean"], reverse=True)[:top_k_clips]
    stats = {
        "n_clips": len(paths),
        "n_frames_audited": n,
        "corr_mean_median": round(float(np.median(means)), 5),
        "corr_mean_p90": round(float(np.percentile(means, 90)), 5),
        "corr_mean_p95": round(float(np.percentile(means, 95)), 5),
        "corr_mean_p99": round(float(np.percentile(means, 99)), 5),
        "single_frame_rate_p50_ge10pct": round(float((a50 >= 0.10).mean()), 4),
        "single_frame_rate_p70_ge10pct": round(float((a70 >= 0.10).mean()), 4),
        "two_frame_confirm_rate_p50_ge10pct": round(consec_p50_hits / max(1, total_pairs), 4),
        "two_frame_confirm_rate_p70_ge10pct": round(consec_p70_hits / max(1, total_pairs), 4),
        "clips_with_any_p50_ge10pct": f"{clips_with_p50}/{len(paths)}",
        "clips_with_any_p70_ge10pct": f"{clips_with_p70}/{len(paths)}",
        "top_clips": [
            {
                "clip": it["clip"],
                "frame": it["frame"],
                "corr_mean": round(it["corr_mean"], 4),
                "area_p50": round(it["area_p50"], 4),
                "area_p70": round(it["area_p70"], 4),
            }
            for it in top_items
        ],
    }
    return stats, top_items


def render_contact_sheet(arkit_top: list[dict], coda_top: list[dict], out_path: Path) -> None:
    items = arkit_top[:12] + coda_top[:6]
    nrows = 6
    ncols = 6  # 3 items per row x (RGB + G2 overlay)
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 18), dpi=130)
    for idx, it in enumerate(items):
        r = idx // 3
        c = (idx % 3) * 2
        rgb = it["rgb"]
        prob_up = cv2.resize(it["prob"], (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
        ax_rgb = axes[r, c]
        ax_ov = axes[r, c + 1]
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(
            f"{it['suite']} {it['clip'][:14]}#{it['frame']}\nmean={it['corr_mean']:.3f} a70={it['area_p70']*100:.0f}%",
            fontsize=8.5,
        )
        ax_rgb.axis("off")

        ax_ov.imshow(rgb)
        ax_ov.imshow(prob_up, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.58)
        # Draw corridor box (rows 4..14 -> y 64..224, cols 6..18 -> x 96..288)
        rect = plt.Rectangle((96, 64), 192, 160, fill=False, edgecolor="lime", linewidth=1.2, linestyle="--")
        ax_ov.add_patch(rect)
        ax_ov.set_title(f"G2 p_glass (max={float(it['prob'].max()):.2f})", fontsize=8.5)
        ax_ov.axis("off")

    fig.suptitle(
        "G2 Frozen-DINOv3 Glass Head Zero-Shot Audit: Top-12 ARKitScenes Indoor (Rows 1-4) + Top-6 CODa Campus (Rows 5-6)\n"
        "Green dashed box = forward corridor (18x24 rows 4..14, cols 6..18). Inspect for genuine glass/windows/mirrors vs non-glass FP!",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ART / "egogeom_omni_v41_pure_rgb_best.pt", map_location="cpu", weights_only=False)
    v41 = EgoGeomOmniV41Net(EgoGeomV41Config(**st["cfg"]), pretrained_backbone=False)
    v41.tof_branch = None
    v41.load_state_dict({k: v for k, v in st["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    v41.to(device).eval()

    g2_st = torch.load(PROBE_DIR / "v41_frozen_glass_head_g2.pt", map_location="cpu", weights_only=False)
    g2 = FrozenDINOv3GlassHead().to(device).eval()
    g2.load_state_dict(g2_st["state_dict"], strict=True)

    arkit_stats, arkit_top = audit_suite("ARKit", ZS_DIR / "arkit_val30_shard00", v41, g2, device, stride=3, top_k_clips=12)
    coda_stats, coda_top = audit_suite("CODa", ZS_DIR / "coda_zeroshot_shard000", v41, g2, device, stride=3, top_k_clips=6)

    sheet_path = PROBE_DIR / "g2_indoor_arkit_coda_audit_sheet.png"
    render_contact_sheet(arkit_top, coda_top, sheet_path)

    report = {
        "meta": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "elapsed_s": round(time.time() - t0, 2),
            "sheet_path": str(sheet_path),
        },
        "arkit_val30_indoor": arkit_stats,
        "coda_campus_outdoor": coda_stats,
    }
    out_json = PROBE_DIR / "g2_indoor_arkit_coda_audit_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] Saved {out_json} and {sheet_path} in {report['meta']['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
