"""Direction A Stage 1: Quantitative Failure-Mode Autopsy on SANPO-Real 40 & Aria Gen 2 (Local RTX 5060, 0 CU).

Diagnoses the exact physical mechanisms behind every missed/late event and top false-alarm frames
for EgoClear-v4.1-Calibrated on:
  - SANPO-Real 40 (REAL_40_LOCAL drift-free +/-0.8s GT, 111 avoidable events; split by camera_head vs camera_chest!)
  - Meta Aria Gen 2 Glasses (19 clips, 193 avoidable events, 100% cross-hardware zero-shot)

Outputs:
  - Quantitative JSON breakdown (artifacts.local/work/egoclear-v0-phase6-autopsy/autopsy_report.json):
      * Camera mount split: head vs chest Late Rate & FA
      * Missed-event taxonomy: tight margin (<=2 ok frames), heading/FOV offset (>20 deg), LOW-only obstacle,
        unseen (max_p < 0.25) vs threshold-squeezed (0.25 <= max_p < thr)
      * High-confidence FA frame taxonomy (p >= 0.50 on GT clear frames):
          - near_edge_corridor: obstacle just outside 0.30m corridor (0.30 < |x| <= 0.70m, z <= 4.0m)!
          - early_far_warning: real static obstacle at 4.0 < z <= 5.5m (just past 4.0m cutoff)!
          - dynamic_pedestrian: dyn_flag == 1
          - pure_phantom: truly clear even within |x|<=0.70m, z<=5.5m
  - Visual diagnostic contact sheets (Top-12 Missed Events + Top-12 High-Conf FA frames on SANPO-Real)
    for direct inspection!
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("e:/linnan/linnan")
sys.path.insert(0, str(ROOT / "research/active/egoclear-v0"))

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from eval_closed_loop import H, MIN_EVENT_FRAMES_V2, make_episodes, weak_mask
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase3_g4 import build_local_window_real_eval_rows

WORK_P5B = ROOT / "artifacts.local/work/egoclear-v0-phase5b"
OUT_DIR = ROOT / "artifacts.local/work/egoclear-v0-phase6-autopsy"


def _local_gt_worker(path_str: str):
    p = Path(path_str)
    d = load_scene(p)
    rows, sd = build_local_window_real_eval_rows(d)
    return p.stem, rows, sd


def get_real_local_rows(real40_dir: Path) -> dict[str, np.ndarray]:
    cache_p = WORK_P5B / "eval_data/real40_local_gt_cache.npz"
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=False)
        return {k[2:]: z[k] for k in z.files if k.startswith("r_")}
    print("[autopsy] building drift-free +/-0.8s local GT cache for 40 real scenes...", flush=True)
    files = sorted(str(p) for p in real40_dir.glob("*.npz"))
    save_d, out = {}, {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for sc, rows, sd in ex.map(_local_gt_worker, files):
            save_d[f"r_{sc}"] = rows
            save_d[f"d_{sc}"] = sd
            out[sc] = rows
    np.savez(cache_p, **save_d)
    return out


@torch.inference_mode()
def run_v41_gpu(model: EgoClearV4Net, scenes: dict[str, dict], device: torch.device) -> dict[str, dict]:
    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))
    preds = {}
    for sc, d in scenes.items():
        n = len(d["eval_rows"])
        if n < 5:
            continue
        rgb = (decode_rgb_batch(d, np.arange(n)) - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
        R_seq = d["poses_R"].astype(np.float32)
        t_seq = d["poses_t"].astype(np.float32)
        fwd_seq = d["causal_fwd"].astype(np.float32)
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        batch = {
            "rgb": torch.from_numpy(rgb)[None].to(device),
            "K_scaled": torch.from_numpy(np.repeat(K_s[None], n, axis=0))[None].to(device),
            "R_c2w": torch.from_numpy(R_seq)[None].to(device),
            "t_w": torch.from_numpy(t_seq)[None].to(device),
            "fwd_w": torch.from_numpy(fwd_seq)[None].to(device),
            "seg_len": torch.from_numpy(seg_seq)[None].to(device),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_sequence(**batch)
        p_corr = out["corr_probs"][0].float().cpu().numpy()
        preds[sc] = {
            "p4": p_corr[:, idx_4m],
            "p_corr": p_corr,
            "p_stat": torch.sigmoid(out["static_logits"][0].float()).cpu().numpy(),
        }
    return preds


def autopsy_domain(
    name: str,
    scenes: dict[str, dict],
    eval_rows_map: dict[str, np.ndarray],
    preds: dict[str, dict],
    bw_speeds: dict,
    thr: float,
    cam_map: dict[str, str] | None = None,
    save_sheets: bool = False,
) -> dict:
    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))

    # 1. Event-level missed/late taxonomy
    miss_records = []
    hit_count = 0
    miss_count = 0
    by_cam = {}

    for sc, er in eval_rows_map.items():
        if sc not in preds or len(er) < 5:
            continue
        d = scenes[sc]
        p4 = preds[sc]["p4"]
        Rs = d["poses_R"].astype(np.float64)
        fwds = d["causal_fwd"].astype(np.float64)
        gt_st = d["gt_bev_static"]  # (T, 3, 24, 15)
        cam_tag = cam_map.get(sc, "unknown") if cam_map else "all"
        by_cam.setdefault(cam_tag, {"late": 0, "avoid": 0, "fa_fr": 0, "km": 0.0})

        eps = make_episodes(sc, er, bw_speeds, kind="blind")
        # Track unique physical event segments (evaluate across all 6 walkers for exact parity)
        for ep_idx, ep in enumerate(eps):
            r = ep["rows"]
            D, valid = r[:, 0], ep["valid"]
            haz = valid & (D <= H)
            al = (p4 >= thr) & valid
            weak = weak_mask(ep)
            dyn = r[:, 2] > 0.5
            by_cam[cam_tag]["fa_fr"] += int((al & ~haz & ~dyn & ~weak).sum())
            by_cam[cam_tag]["km"] += float(r[valid, 3].sum() / 1000.0)

            i, n = 0, len(D)
            while i < n:
                if haz[i]:
                    j = i
                    while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                        j += 1
                    seg = slice(i, j + 1)
                    if haz[seg].sum() >= MIN_EVENT_FRAMES_V2:
                        ok = D[seg] >= ep["dreq"]
                        if ok.any():
                            by_cam[cam_tag]["avoid"] += 1
                            is_miss = not (al[seg] & ok).any()
                            if is_miss:
                                miss_count += 1
                                by_cam[cam_tag]["late"] += 1
                                ok_idx = np.nonzero(ok)[0] + i
                                max_p = float(p4[ok_idx].max())
                                peak_fi = int(ok_idx[np.argmax(p4[ok_idx])])
                                # Camera optical axis vs walking heading angle
                                cam_f = Rs[peak_fi, :2, 2]
                                cam_f = cam_f / max(np.linalg.norm(cam_f), 1e-6)
                                cos_h = float(np.clip(np.dot(cam_f, fwds[peak_fi, :2]), -1.0, 1.0))
                                ang_deg = float(np.degrees(np.arccos(cos_h)))
                                # GT height bands inside corridor during ok frames
                                hb_low = int(gt_st[ok_idx, 0, : idx_4m + 1, 6:9].sum())
                                hb_body = int(gt_st[ok_idx, 1, : idx_4m + 1, 6:9].sum())
                                hb_head = int(gt_st[ok_idx, 2, : idx_4m + 1, 6:9].sum())
                                miss_records.append({
                                    "scene": sc, "cam": cam_tag, "walker": ep_idx,
                                    "frame": peak_fi, "n_ok_frames": int(len(ok_idx)),
                                    "d_min_ok": round(float(D[ok_idx].min()), 2),
                                    "dreq": round(ep["dreq"], 2),
                                    "max_p": round(max_p, 4),
                                    "heading_ang_deg": round(ang_deg, 1),
                                    "low_only": bool(hb_low > 0 and hb_body == 0 and hb_head == 0),
                                    "head_only": bool(hb_head > 0 and hb_low == 0 and hb_body == 0),
                                    "near_thr": bool(max_p >= 0.25),
                                    "tight_window": bool(len(ok_idx) <= 2),
                                    "off_axis_20deg": bool(ang_deg >= 20.0),
                                })
                            else:
                                hit_count += 1
                    i = j + 1
                else:
                    i += 1

    # 2. High-confidence FA frame taxonomy (p4 >= thr on clear frames: valid & ~haz & ~dyn & ~weak)
    fa_records = []
    for sc, er in eval_rows_map.items():
        if sc not in preds or len(er) < 5:
            continue
        d = scenes[sc]
        p4 = preds[sc]["p4"]
        gt_st = d["gt_bev_static"]  # (T, 3, 24, 15)
        ep0 = make_episodes(sc, er, bw_speeds, kind="blind")[0]
        valid = ep0["valid"]
        haz = valid & (er[:, 0] <= H)
        dyn = er[:, 2] > 0.5
        weak = weak_mask(ep0)
        fa_idx = np.nonzero((p4 >= thr) & valid & ~haz & ~dyn & ~weak)[0]
        for fi in fa_idx:
            # Check BEV static GT just outside corridor (cols 4,5 & 9,10 -> 0.30..0.70m lateral, z<=4m)
            edge_vox = int(gt_st[fi, :, : idx_4m + 1, [4, 5, 9, 10]].sum())
            # Check BEV static GT just past 4.0m (z in 4.0..5.1m inside corridor cols 5..9)
            far_vox = int(gt_st[fi, :, idx_4m + 1 :, 5:10].sum())
            # Check inner corridor BEV voxels (e.g., sub-threshold < 8 voxels or single-frame noise)
            in_vox = int(gt_st[fi, :, : idx_4m + 1, 6:9].sum())
            if in_vox > 0:
                reason = "sub_minvox_obstacle"
            elif edge_vox >= 2:
                reason = "near_edge_lateral_30_70cm"
            elif far_vox >= 2:
                reason = "early_far_4_to_5m"
            else:
                reason = "pure_phantom_clear"
            fa_records.append({
                "scene": sc, "cam": cam_map.get(sc, "unknown") if cam_map else "all",
                "frame": int(fi), "p4": round(float(p4[fi]), 4),
                "reason": reason, "in_vox": in_vox, "edge_vox": edge_vox, "far_vox": far_vox,
            })

    # Summarize counts
    n_miss = len(miss_records)
    miss_summary = {
        "total_avoidable": hit_count + miss_count,
        "total_missed": miss_count,
        "late_rate": round(miss_count / max(hit_count + miss_count, 1), 4),
        "thr_used": thr,
        "frac_tight_window_le2fr": round(np.mean([m["tight_window"] for m in miss_records]), 4) if n_miss else 0.0,
        "frac_off_axis_ge20deg": round(np.mean([m["off_axis_20deg"] for m in miss_records]), 4) if n_miss else 0.0,
        "frac_near_thr_p_ge_0.25": round(np.mean([m["near_thr"] for m in miss_records]), 4) if n_miss else 0.0,
        "frac_unseen_p_lt_0.25": round(np.mean([not m["near_thr"] for m in miss_records]), 4) if n_miss else 0.0,
        "frac_low_only": round(np.mean([m["low_only"] for m in miss_records]), 4) if n_miss else 0.0,
        "median_max_p_on_missed": round(float(np.median([m["max_p"] for m in miss_records])), 4) if n_miss else 0.0,
        "median_heading_ang_deg": round(float(np.median([m["heading_ang_deg"] for m in miss_records])), 2) if n_miss else 0.0,
    }

    n_fa = len(fa_records)
    fa_reasons = {}
    for k in ("sub_minvox_obstacle", "near_edge_lateral_30_70cm", "early_far_4_to_5m", "pure_phantom_clear"):
        cnt = sum(1 for r in fa_records if r["reason"] == k)
        fa_reasons[k] = {"count": cnt, "frac": round(cnt / max(n_fa, 1), 4)}

    cam_summary = {}
    for k, v in by_cam.items():
        cam_summary[k] = {
            "late": v["late"], "avoid": v["avoid"],
            "late_rate": round(v["late"] / max(v["avoid"], 1), 4) if v["avoid"] else None,
            "fa_fr_per_km": round(v["fa_fr"] / max(v["km"], 1e-6), 2),
        }

    # Save visual contact sheets for SANPO-Real (Top-12 missed unique scenes + Top-12 FA frames)
    if save_sheets:
        # Unique scene-frame misses sorted by lowest max_p (hardest true misses)
        seen = set()
        uniq_m = []
        for m in sorted(miss_records, key=lambda x: x["max_p"]):
            key = (m["scene"], m["frame"] // 5)
            if key not in seen:
                seen.add(key)
                uniq_m.append(m)
            if len(uniq_m) >= 12:
                break
        if uniq_m:
            tiles = []
            for m in uniq_m:
                img = (decode_rgb_batch(scenes[m["scene"]], [m["frame"]])[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                bgr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (360, 204))
                cv2.rectangle(bgr, (0, 0), (360, 34), (20, 20, 28), -1)
                cv2.putText(
                    bgr,
                    f"MISS [{m['cam'][:4]}] p={m['max_p']:.2f} d={m['d_min_ok']}m ok={m['n_ok_frames']}fr ang={m['heading_ang_deg']:.0f}d",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 140, 255), 1, cv2.LINE_AA,
                )
                tiles.append(bgr)
            while len(tiles) < 12:
                tiles.append(np.zeros((204, 360, 3), dtype=np.uint8))
            grid = np.vstack([np.hstack(tiles[r * 4 : (r + 1) * 4]) for r in range(3)])
            cv2.imwrite(str(OUT_DIR / f"top12_missed_{name}.png"), grid)

        seen_f = set()
        uniq_f = []
        for f in sorted(fa_records, key=lambda x: -x["p4"]):
            key = (f["scene"], f["frame"] // 5)
            if key not in seen_f:
                seen_f.add(key)
                uniq_f.append(f)
            if len(uniq_f) >= 12:
                break
        if uniq_f:
            tiles = []
            for f in uniq_f:
                img = (decode_rgb_batch(scenes[f["scene"]], [f["frame"]])[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                bgr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (360, 204))
                cv2.rectangle(bgr, (0, 0), (360, 34), (20, 20, 28), -1)
                cv2.putText(
                    bgr,
                    f"FA p={f['p4']:.2f} {f['reason'][:18]} [{f['cam'][:4]}]",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 255), 1, cv2.LINE_AA,
                )
                tiles.append(bgr)
            while len(tiles) < 12:
                tiles.append(np.zeros((204, 360, 3), dtype=np.uint8))
            grid = np.vstack([np.hstack(tiles[r * 4 : (r + 1) * 4]) for r in range(3)])
            cv2.imwrite(str(OUT_DIR / f"top12_fa_{name}.png"), grid)

    return {
        "miss_summary": miss_summary,
        "by_camera_mount": cam_summary,
        "fa_total_frames": n_fa,
        "fa_reason_breakdown": fa_reasons,
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[autopsy] running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})", flush=True)

    bw_speeds = json.loads((ROOT / "research/active/egoclear-v0/sanpo_287_manifest.json").read_text())["bw_speeds"]
    p2_man = json.loads((ROOT / "research/active/egoclear-v0/phase2_manifest.json").read_text())
    cam_map_real = {f"{r['scene']}__{r['cam']}": r["cam"] for r in p2_man["sanpo_real_zeroshot_40"]}

    eval_root = WORK_P5B / "eval_data"
    real_scenes = {p.stem: load_scene(p) for p in sorted((eval_root / "real40").glob("*.npz"))}
    aria_scenes = {p.stem: load_scene(p) for p in sorted((eval_root / "aria19").glob("*.npz"))}
    real_local_rows = get_real_local_rows(eval_root / "real40")

    # Map scene stems to camera_head / camera_chest
    cam_map_resolved = {}
    for sc in real_scenes:
        if "chest" in sc:
            cam_map_resolved[sc] = "camera_chest"
        elif "head" in sc:
            cam_map_resolved[sc] = "camera_head"
        else:
            cam_map_resolved[sc] = cam_map_real.get(sc, "unknown")

    ckpt = torch.load(WORK_P5B / "models/egoclear_v41_full_calibrated.pt", map_location="cpu", weights_only=False)
    model = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    model.load_state_dict(ckpt["head_state_dict"], strict=False)

    print("[autopsy] running V4.1 GPU inference on SANPO-Real 40 & Aria Gen 2...", flush=True)
    preds_real = run_v41_gpu(model, real_scenes, device)
    preds_aria = run_v41_gpu(model, aria_scenes, device)

    # Pick exact thresholds corresponding to @fr<=120 and @fr<=200 on SANPO-Real (thr=0.52 and 0.36) and Aria (thr=0.56 and 0.32)
    rep = {
        "real40_local_at_thr0.52_lowFA": autopsy_domain(
            "real40_thr052", real_scenes, real_local_rows, preds_real, bw_speeds, thr=0.52, cam_map=cam_map_resolved, save_sheets=True
        ),
        "real40_local_at_thr0.32_midFA": autopsy_domain(
            "real40_thr032", real_scenes, real_local_rows, preds_real, bw_speeds, thr=0.32, cam_map=cam_map_resolved, save_sheets=False
        ),
        "aria_g2_at_thr0.48_lowFA": autopsy_domain(
            "aria_thr048", aria_scenes, {sc: d["eval_rows"] for sc, d in aria_scenes.items()}, preds_aria, bw_speeds, thr=0.48, save_sheets=False
        ),
        "aria_g2_at_thr0.24_midFA": autopsy_domain(
            "aria_thr024", aria_scenes, {sc: d["eval_rows"] for sc, d in aria_scenes.items()}, preds_aria, bw_speeds, thr=0.24, save_sheets=False
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rep_p = OUT_DIR / "autopsy_report.json"
    rep_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    print(f"[autopsy] DONE saved {rep_p} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
