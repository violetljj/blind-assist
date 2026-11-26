"""Follow-up Diagnostic (Local RTX 5060, 0.00 CCU):
Decompose G2's ARKitScenes indoor triggers into:
  (A) Fixed image-center box (no gravity/normal gate)          -> baseline from first audit
  (B) Gravity-aligned forward cone (|az|<=14°, el∈[-14°,+14°]) -> matches Kotlin GlassCorridorRiskEvaluator
  (C) Gravity cone + v4.1 verticality gate (|n_down| < 0.65)   -> rejects flat horizontal floors/ceilings!
Exports `g2_arkit_grav_vert_sheet.png` showing top-12 remaining ARKitScenes frames under (C) for visual inspection.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
ZS_DIR = ART / "zeroshot_eval"

sys.path[:0] = [str(HERE), str(ROUTE)]

from audit_g2_indoor_arkit_coda_local import FrozenDINOv3GlassHead, H_IN, W_IN, corridor_mask_s16, decode_iso_rgb  # noqa: E402
from egogeom_losses_and_physics import compute_gravity_aligned_ray_map  # noqa: E402
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K  # noqa: E402
from eval_scalefree_angular_ttc_local import _compute_angular_rays_s4  # noqa: E402


@torch.inference_mode()
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

    paths = sorted((ZS_DIR / "arkit_val30_shard00").rglob("*.npz"))
    cmask_img = corridor_mask_s16()
    az_lim = np.radians(14.0)
    el_lo, el_hi = np.radians(-14.0), np.radians(14.0)

    modes = ("A_img_box", "B_grav_cone", "C_grav_cone_vert")
    stats_acc = {m: {"a50": [], "a70": [], "mean": [], "c2_50": 0, "c2_70": 0, "pairs": 0} for m in modes}
    clip_best_c: dict[str, dict] = {}
    stride = 3

    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            blob, offs = d["rgb_blob"], d["rgb_offsets"]
            orig_hw, K_orig = d["orig_hw"], d["K_orig"]
            poses_R, fwd_arr = d["poses_R"], d["causal_fwd"]
            T = len(offs) - 1
            fids = sorted(set(x for t in range(0, T - 1, stride) for x in (t, t + 1)))
            rgbs_n, rgbs_u8, Ks, gs = [], [], [], []
            for fi in fids:
                rn, ru, K = decode_iso_rgb(bytes(blob[offs[fi] : offs[fi + 1]]), orig_hw, K_orig)
                g_cam = poses_R[fi].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
                rgbs_n.append(rn); rgbs_u8.append(ru); Ks.append(K); gs.append(g_cam)

        res_map = {}
        for i0 in range(0, len(fids), 16):
            i1 = min(len(fids), i0 + 16)
            x = torch.from_numpy(np.stack(rgbs_n[i0:i1])).to(device, dtype=torch.float32)
            Kt = torch.from_numpy(np.stack(Ks[i0:i1])).to(device, dtype=torch.float32)
            gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device, dtype=torch.float32)
            rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
            hf = batch_host_focal_from_K(Kt, W_IN, H_IN).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = v41(x, rays, None, None, host_focal=hf, return_features=True)
                gp = g2(out["feat_s16"].float(), out["feat_s32"].float())
                nd16 = F.interpolate(out["normal_grav"][:, 1:2].float().abs(), size=(18, 24), mode="area")
            gp_np = gp[:, 0].float().cpu().numpy()
            nd_np = nd16[:, 0].float().cpu().numpy()
            for j, fi in enumerate(fids[i0:i1]):
                res_map[fi] = (gp_np[j], nd_np[j])

        idx_lk = {fi: idx for idx, fi in enumerate(fids)}

        def eval_frame(fi: int) -> dict[str, tuple[float, float, float, np.ndarray]]:
            gp, nd = res_map[fi]
            K = Ks[idx_lk[fi]]
            ang = _compute_angular_rays_s4(18, 24, K, poses_R[fi], fwd_arr[fi], stride=16, step=1)
            out_m = {}
            # A: fixed image box
            cA = gp[cmask_img]
            out_m["A_img_box"] = (float(cA.mean()), float((cA >= 0.50).mean()), float((cA >= 0.70).mean()), cmask_img)
            if ang is None:
                empty = np.zeros((18, 24), dtype=bool)
                out_m["B_grav_cone"] = (0.0, 0.0, 0.0, empty)
                out_m["C_grav_cone_vert"] = (0.0, 0.0, 0.0, empty)
                return out_m
            az, el, fd = ang
            mB = (fd > 0.25) & (np.abs(az) <= az_lim) & (el >= el_lo) & (el <= el_hi)
            if mB.any():
                cB = gp[mB]
                out_m["B_grav_cone"] = (float(cB.mean()), float((cB >= 0.50).mean()), float((cB >= 0.70).mean()), mB)
                # C: inside gravity cone, zero out horizontal floor/ceiling cells (|n_down| >= 0.65)
                gp_vert = np.where(nd < 0.65, gp, 0.0)
                cC = gp_vert[mB]
                out_m["C_grav_cone_vert"] = (float(cC.mean()), float((cC >= 0.50).mean()), float((cC >= 0.70).mean()), mB)
            else:
                out_m["B_grav_cone"] = (0.0, 0.0, 0.0, mB)
                out_m["C_grav_cone_vert"] = (0.0, 0.0, 0.0, mB)
            return out_m

        for t in range(0, T - 1, stride):
            e0 = eval_frame(t)
            e1 = eval_frame(t + 1)
            for m in modes:
                m0, f50_0, f70_0, mask_b = e0[m]
                _, f50_1, f70_1, _ = e1[m]
                st_m = stats_acc[m]
                st_m["mean"].append(m0); st_m["a50"].append(f50_0); st_m["a70"].append(f70_0)
                st_m["pairs"] += 1
                if f50_0 >= 0.10 and f50_1 >= 0.10:
                    st_m["c2_50"] += 1
                if f70_0 >= 0.10 and f70_1 >= 0.10:
                    st_m["c2_70"] += 1
            mC, f50C, f70C, mB = e0["C_grav_cone_vert"]
            prev = clip_best_c.get(p.stem)
            if prev is None or mC > prev["corr_mean"]:
                gp0, nd0 = res_map[t]
                clip_best_c[p.stem] = {
                    "clip": p.stem, "frame": t, "corr_mean": mC, "area_p50": f50C, "area_p70": f70C,
                    "rgb": rgbs_u8[idx_lk[t]], "prob_vert": np.where(nd0 < 0.65, gp0, 0.0), "cone_mask": mB,
                }

    summary = {}
    for m in modes:
        s = stats_acc[m]
        a50, a70, mn = np.array(s["a50"]), np.array(s["a70"]), np.array(s["mean"])
        summary[m] = {
            "n_frames": len(mn),
            "corr_mean_median": round(float(np.median(mn)), 5),
            "corr_mean_p95": round(float(np.percentile(mn, 95)), 5),
            "single_p50_ge10pct": round(float((a50 >= 0.10).mean()), 4),
            "single_p70_ge10pct": round(float((a70 >= 0.10).mean()), 4),
            "two_frame_p50_ge10pct": round(s["c2_50"] / max(1, s["pairs"]), 4),
            "two_frame_p70_ge10pct": round(s["c2_70"] / max(1, s["pairs"]), 4),
        }

    top12 = sorted(clip_best_c.values(), key=lambda x: x["corr_mean"], reverse=True)[:12]
    fig, axes = plt.subplots(4, 6, figsize=(21, 12.5), dpi=130)
    for idx, it in enumerate(top12):
        r, c = idx // 3, (idx % 3) * 2
        rgb = it["rgb"]
        p_up = cv2.resize(it["prob_vert"].astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
        cm_up = cv2.resize(it["cone_mask"].astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_NEAREST)
        axes[r, c].imshow(rgb)
        axes[r, c].set_title(f"{it['clip']}#{it['frame']}\nC_mean={it['corr_mean']:.3f} a70={it['area_p70']*100:.0f}%", fontsize=8.5)
        axes[r, c].axis("off")
        axes[r, c + 1].imshow(rgb)
        axes[r, c + 1].imshow(p_up, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.58)
        axes[r, c + 1].contour(cm_up, levels=[0.5], colors=["lime"], linewidths=1.2)
        axes[r, c + 1].set_title("G2 x Verticality (|n_down|<0.65)", fontsize=8.5)
        axes[r, c + 1].axis("off")
    fig.suptitle("ARKitScenes Top-12 Remaining Triggers AFTER Gravity Cone (el in [-14,+14]) + Verticality Gate (|n_down|<0.65)", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    sheet = PROBE_DIR / "g2_arkit_grav_vert_sheet.png"
    fig.savefig(sheet, bbox_inches="tight")
    plt.close(fig)

    out_p = PROBE_DIR / "g2_arkit_grav_vert_report.json"
    out_p.write_text(json.dumps({"elapsed_s": round(time.time() - t0, 2), "summary": summary, "top12": [{k: v for k, v in it.items() if k not in ("rgb", "prob_vert", "cone_mask")} for it in top12]}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] Saved {out_p} and {sheet}", flush=True)


if __name__ == "__main__":
    main()
