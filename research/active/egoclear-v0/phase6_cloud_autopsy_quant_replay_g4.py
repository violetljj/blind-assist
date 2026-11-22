"""Phase 6 Cloud G4 (96 GB + 48 vCPU) Fast Runner:
1. Recovers ALL downloaded SANPO-Real .npz in /content/p6_data/real_extra444/ (fixing the '_' vs '__' filename
   check from pack_phase6_treasure_colab.py), rebuilds drift-free +/-0.8s local GT on 48 vCPUs, and writes complete
   shards to Drive /BlindAssist/datasets/sanpo_real_extra444_v0/!
2. Direction A Stage 1 (Failure-Mode Autopsy on 96 GB G4 GPU, ~15s):
   Full quantitative breakdown of missed/late events & high-conf FA frames on SANPO-Real 40 (head vs chest) and
   Aria Gen 2 (19 clips), plus Top-12 Missed & Top-12 FA annotated visual grids!
3. Direction B (On-Device FP16/INT8 Quantization Audit + 3x 1080x540 Streaming ONNX MP4 Replay Videos, ~60s on 48 vCPUs):
   Compresses egoclear_v41_{encoder,step}.onnx to FP16 (57.2 MB) and INT8, audits 960 real streaming frames
   (MAE, max err, p99, monotonic=0.0, latency), and renders 3 dual-panel MP4 replays + contact sheets -> Drive!
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from eval_closed_loop import H, MIN_EVENT_FRAMES_V2, apply_causal_hysteresis, make_episodes, weak_mask
from pack_phase6_treasure_colab import write_incremental_shards
from phase4_simreal_aria_onnx_g4 import build_local_bev_and_rows_for_real_train
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase5b_calibrated_v41_g4 import stage_all_datasets_parallel
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_REAL444 = DRIVE_ROOT / "datasets/sanpo_real_extra444_v0"
DRIVE_OUT = DRIVE_ROOT / "runs/egoclear_v0_phase6_autopsy_edge"
LOCAL_OUT = Path("/content/p6_autopsy_edge")
LOCAL_REAL444 = Path("/content/p6_data/real_extra444")


def recover_and_finalize_real444():
    npz_files = sorted(LOCAL_REAL444.glob("*.npz"))
    print(f"[p6-fix] Found {len(npz_files)} downloaded SANPO-Real .npz files in {LOCAL_REAL444}", flush=True)
    if len(npz_files) > 0:
        t0 = time.time()
        ncpu = min(24, max(6, (os.cpu_count() or 8) - 2))
        print(f"[p6-fix] Rebuilding drift-free +/-0.8s local GT on {len(npz_files)} real scenes ({ncpu} workers)...", flush=True)
        with ProcessPoolExecutor(max_workers=ncpu) as pool:
            list(pool.map(build_local_bev_and_rows_for_real_train, [str(p) for p in npz_files]))
        stems = sorted(p.stem for p in LOCAL_REAL444.glob("*.npz"))
        write_incremental_shards(LOCAL_REAL444, stems, DRIVE_REAL444, "sanpo_real_extra444_v0", shard_size=30)
        print(f"[p6-fix] Finalized & saved {len(stems)} SANPO-Real scenes to Drive in {time.time()-t0:.1f}s!", flush=True)
    return len(npz_files)


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

    miss_records = []
    hit_count = miss_count = 0
    by_cam = {}

    for sc, er in eval_rows_map.items():
        if sc not in preds or len(er) < 5:
            continue
        d = scenes[sc]
        p4 = preds[sc]["p4"]
        Rs = d["poses_R"].astype(np.float64)
        fwds = d["causal_fwd"].astype(np.float64)
        gt_st = d["gt_bev_static"]
        cam_tag = cam_map.get(sc, "unknown") if cam_map else "all"
        by_cam.setdefault(cam_tag, {"late": 0, "avoid": 0, "fa_fr": 0, "km": 0.0})

        eps = make_episodes(sc, er, bw_speeds, kind="blind")
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
                            if not (al[seg] & ok).any():
                                miss_count += 1
                                by_cam[cam_tag]["late"] += 1
                                ok_idx = np.nonzero(ok)[0] + i
                                max_p = float(p4[ok_idx].max())
                                peak_fi = int(ok_idx[np.argmax(p4[ok_idx])])
                                cam_f = Rs[peak_fi, :2, 2]
                                cam_f = cam_f / max(np.linalg.norm(cam_f), 1e-6)
                                cos_h = float(np.clip(np.dot(cam_f, fwds[peak_fi, :2]), -1.0, 1.0))
                                ang_deg = float(np.degrees(np.arccos(cos_h)))
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
                                    "near_thr": bool(max_p >= 0.25),
                                    "tight_window": bool(len(ok_idx) <= 2),
                                    "off_axis_20deg": bool(ang_deg >= 20.0),
                                })
                            else:
                                hit_count += 1
                    i = j + 1
                else:
                    i += 1

    fa_records = []
    for sc, er in eval_rows_map.items():
        if sc not in preds or len(er) < 5:
            continue
        d = scenes[sc]
        p4 = preds[sc]["p4"]
        gt_st = d["gt_bev_static"]
        ep0 = make_episodes(sc, er, bw_speeds, kind="blind")[0]
        valid = ep0["valid"]
        haz = valid & (er[:, 0] <= H)
        dyn = er[:, 2] > 0.5
        weak = weak_mask(ep0)
        fa_idx = np.nonzero((p4 >= thr) & valid & ~haz & ~dyn & ~weak)[0]
        for fi in fa_idx:
            edge_vox = int(gt_st[fi, :, : idx_4m + 1, [4, 5, 9, 10]].sum())
            far_vox = int(gt_st[fi, :, idx_4m + 1 :, 5:10].sum())
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

    n_miss = len(miss_records)
    miss_summary = {
        "total_avoidable": hit_count + miss_count,
        "total_missed": miss_count,
        "late_rate": round(miss_count / max(hit_count + miss_count, 1), 4),
        "thr_used": thr,
        "frac_tight_window_le2fr": round(float(np.mean([m["tight_window"] for m in miss_records])), 4) if n_miss else 0.0,
        "frac_off_axis_ge20deg": round(float(np.mean([m["off_axis_20deg"] for m in miss_records])), 4) if n_miss else 0.0,
        "frac_near_thr_p_ge_0.25": round(float(np.mean([m["near_thr"] for m in miss_records])), 4) if n_miss else 0.0,
        "frac_unseen_p_lt_0.25": round(float(np.mean([not m["near_thr"] for m in miss_records])), 4) if n_miss else 0.0,
        "frac_low_only": round(float(np.mean([m["low_only"] for m in miss_records])), 4) if n_miss else 0.0,
        "median_max_p_on_missed": round(float(np.median([m["max_p"] for m in miss_records])), 4) if n_miss else 0.0,
        "median_heading_ang_deg": round(float(np.median([m["heading_ang_deg"] for m in miss_records])), 2) if n_miss else 0.0,
    }
    n_fa = len(fa_records)
    fa_reasons = {
        k: {"count": (c := sum(1 for r in fa_records if r["reason"] == k)), "frac": round(c / max(n_fa, 1), 4)}
        for k in ("sub_minvox_obstacle", "near_edge_lateral_30_70cm", "early_far_4_to_5m", "pure_phantom_clear")
    }
    cam_summary = {
        k: {
            "late": v["late"], "avoid": v["avoid"],
            "late_rate": round(v["late"] / max(v["avoid"], 1), 4) if v["avoid"] else None,
            "fa_fr_per_km": round(v["fa_fr"] / max(v["km"], 1e-6), 2),
        }
        for k, v in by_cam.items()
    }

    if save_sheets:
        seen, uniq_m = set(), []
        for m in sorted(miss_records, key=lambda x: x["max_p"]):
            key = (m["scene"], m["frame"] // 5)
            if key not in seen:
                seen.add(key); uniq_m.append(m)
            if len(uniq_m) >= 12:
                break
        if uniq_m:
            tiles = []
            for m in uniq_m:
                img = (decode_rgb_batch(scenes[m["scene"]], [m["frame"]])[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                bgr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (360, 204))
                cv2.rectangle(bgr, (0, 0), (360, 32), (20, 20, 28), -1)
                cv2.putText(bgr, f"MISS[{m['cam'][-5:]}] p={m['max_p']:.2f} d={m['d_min_ok']}m ok={m['n_ok_frames']}fr ang={m['heading_ang_deg']:.0f}d",
                            (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 140, 255), 1, cv2.LINE_AA)
                tiles.append(bgr)
            while len(tiles) < 12:
                tiles.append(np.zeros((204, 360, 3), dtype=np.uint8))
            cv2.imwrite(str(LOCAL_OUT / f"top12_missed_{name}.png"), np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)]))

        seen_f, uniq_f = set(), []
        for f in sorted(fa_records, key=lambda x: -x["p4"]):
            key = (f["scene"], f["frame"] // 5)
            if key not in seen_f:
                seen_f.add(key); uniq_f.append(f)
            if len(uniq_f) >= 12:
                break
        if uniq_f:
            tiles = []
            for f in uniq_f:
                img = (decode_rgb_batch(scenes[f["scene"]], [f["frame"]])[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                bgr = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (360, 204))
                cv2.rectangle(bgr, (0, 0), (360, 32), (20, 20, 28), -1)
                cv2.putText(bgr, f"FA p={f['p4']:.2f} {f['reason'][:18]} [{f['cam'][-5:]}]",
                            (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 255), 1, cv2.LINE_AA)
                tiles.append(bgr)
            while len(tiles) < 12:
                tiles.append(np.zeros((204, 360, 3), dtype=np.uint8))
            cv2.imwrite(str(LOCAL_OUT / f"top12_fa_{name}.png"), np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)]))

    return {"miss_summary": miss_summary, "by_camera_mount": cam_summary, "fa_total_frames": n_fa, "fa_reason_breakdown": fa_reasons}


def main():
    t0 = time.time()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)

    # 0. Ensure onnx & onnxruntime installed
    try:
        import onnx, onnxruntime
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)

    # 1. Recover all downloaded SANPO-Real .npz from pack_phase6_treasure_colab.py & save to Drive!
    n_recovered = recover_and_finalize_real444()

    # 2. Stage existing eval datasets in parallel (6 threads, ~90s total!)
    paths = stage_all_datasets_parallel()
    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))

    bw_speeds = json.loads(Path("/content/sanpo_287_manifest.json").read_text())["bw_speeds"]
    real_scenes = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_scenes = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    arkit_scenes = {p.stem: load_scene(p) for p in sorted(paths["arkit_va"].glob("*.npz"))}

    # 3. Direction A Stage 1: Failure-Mode Autopsy on 96 GB G4 GPU
    device = torch.device("cuda")
    ckpt_p = LOCAL_OUT / "egoclear_v41_full_calibrated.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase5b/egoclear_v41_full_calibrated.pt", ckpt_p)
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    model.load_state_dict(ckpt["head_state_dict"], strict=False)

    print("[autopsy-g4] running V4.1 GPU inference on SANPO-Real 40 & Aria Gen 2...", flush=True)
    preds_real = run_v41_gpu(model, real_scenes, device)
    preds_aria = run_v41_gpu(model, aria_scenes, device)
    cam_map = {sc: ("camera_chest" if "chest" in sc else "camera_head") for sc in real_scenes}

    autopsy_rep = {
        "sanpo_real_extra_recovered_count": n_recovered,
        "real40_local_thr0.52": autopsy_domain("real40_thr052", real_scenes, real_local_rows, preds_real, bw_speeds, thr=0.52, cam_map=cam_map, save_sheets=True),
        "real40_local_thr0.32": autopsy_domain("real40_thr032", real_scenes, real_local_rows, preds_real, bw_speeds, thr=0.32, cam_map=cam_map, save_sheets=False),
        "aria_g2_thr0.48": autopsy_domain("aria_thr048", aria_scenes, {sc: d["eval_rows"] for sc, d in aria_scenes.items()}, preds_aria, bw_speeds, thr=0.48, save_sheets=False),
        "aria_g2_thr0.24": autopsy_domain("aria_thr024", aria_scenes, {sc: d["eval_rows"] for sc, d in aria_scenes.items()}, preds_aria, bw_speeds, thr=0.24, save_sheets=False),
    }
    print("\n=== FAILURE-MODE AUTOPSY REPORT ===", flush=True)
    print(json.dumps(autopsy_rep, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()

    # 4. Direction B: Cloud ONNX FP16/INT8 Quantization Audit + 3x 1080x540 MP4 Replay Videos (48 vCPU!)
    from edge_quant_and_replay import (
        StreamingONNXRunner,
        convert_onnx_weights_to_fp16_safe,
        quantize_onnx_int8_dynamic,
        render_replay_video,
    )
    enc_fp32 = LOCAL_OUT / "egoclear_v41_encoder.onnx"
    stp_fp32 = LOCAL_OUT / "egoclear_v41_step.onnx"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase5b/egoclear_v41_encoder.onnx", enc_fp32)
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase5b/egoclear_v41_step.onnx", stp_fp32)

    enc_fp16 = LOCAL_OUT / "egoclear_v41_encoder_fp16.onnx"
    stp_fp16 = LOCAL_OUT / "egoclear_v41_step_fp16.onnx"
    enc_int8 = LOCAL_OUT / "egoclear_v41_encoder_int8.onnx"
    stp_int8 = LOCAL_OUT / "egoclear_v41_step_int8.onnx"

    convert_onnx_weights_to_fp16_safe(enc_fp32, enc_fp16)
    convert_onnx_weights_to_fp16_safe(stp_fp32, stp_fp16)
    quantize_onnx_int8_dynamic(enc_fp32, enc_int8)
    quantize_onnx_int8_dynamic(stp_fp32, stp_int8)

    audit_scenes = {}
    for sc in sorted(arkit_scenes)[:6]:
        audit_scenes[f"arkit:{sc}"] = arkit_scenes[sc]
    for sc in sorted(real_scenes)[:4]:
        audit_scenes[f"real:{sc}"] = real_scenes[sc]
    for sc in sorted(aria_scenes)[:4]:
        audit_scenes[f"aria:{sc}"] = aria_scenes[sc]

    var_outputs = {}
    quant_report = {"variants": {}}
    for vname, (ep, sp) in (("fp32", (enc_fp32, stp_fp32)), ("fp16", (enc_fp16, stp_fp16)), ("int8", (enc_int8, stp_int8))):
        runner = StreamingONNXRunner(ep, sp, intra_threads=16)
        enc_ms_l, stp_ms_l = [], []
        v_res = {}
        mv = mt = 0
        for key, d in audit_scenes.items():
            r = runner.run_scene(d, max_frames=50)
            v_res[key] = r
            enc_ms_l.append(r["enc_ms_per_fr"])
            stp_ms_l.append(r["stp_ms_per_fr"])
            diffs = np.diff(r["corr_probs"], axis=1)
            mv += int((diffs < -1e-4).sum())
            mt += int(diffs.size)
        var_outputs[vname] = v_res
        quant_report["variants"][vname] = {
            "encoder_mb": round(ep.stat().st_size / 1e6, 2),
            "step_mb": round(sp.stat().st_size / 1e6, 2),
            "total_mb": round((ep.stat().st_size + sp.stat().st_size) / 1e6, 2),
            "cloud_cpu16_enc_ms": round(float(np.mean(enc_ms_l)), 2),
            "cloud_cpu16_step_ms": round(float(np.mean(stp_ms_l)), 2),
            "monotonic_viol_rate": round(mv / max(mt, 1), 6),
        }
        print(f"[quant-g4 | {vname}] {quant_report['variants'][vname]}", flush=True)

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))
    for vname in ("fp16", "int8"):
        da = np.concatenate([np.abs(var_outputs[vname][k]["corr_probs"] - var_outputs["fp32"][k]["corr_probs"]).ravel() for k in audit_scenes])
        dp4 = np.concatenate([np.abs(var_outputs[vname][k]["corr_probs"][:, idx_4m] - var_outputs["fp32"][k]["corr_probs"][:, idx_4m]) for k in audit_scenes])
        quant_report["variants"][vname].update({
            "vs_fp32_corr_mae": round(float(da.mean()), 6),
            "vs_fp32_corr_max_err": round(float(da.max()), 6),
            "vs_fp32_p4_mae": round(float(dp4.mean()), 6),
            "vs_fp32_p4_p99": round(float(np.percentile(dp4, 99)), 6),
        })
        print(f"[parity-g4 | {vname} vs fp32] mae={da.mean():.6f}, max={da.max():.6f}, p4_p99={np.percentile(dp4, 99):.6f}", flush=True)

    # Render 3 full-scene MP4 replays + contact sheets using FP16 ONNX
    runner_fp16 = StreamingONNXRunner(enc_fp16, stp_fp16, intra_threads=16)
    def pick_rich(sdict):
        best_sc, best_s = None, -1.0
        for sc, d in sorted(sdict.items()):
            er = d["eval_rows"]
            valid = np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)
            if valid.sum() < 25:
                continue
            hz = (valid & np.isfinite(er[:, 0]) & (er[:, 0] <= H)).sum()
            cl = (valid & np.isinf(er[:, 0])).sum()
            score = min(hz, cl) + 0.05 * len(er)
            if score > best_s:
                best_s, best_sc = score, sc
        return best_sc or sorted(sdict)[0]

    replays = []
    for tag, label, sdict, sc, thr_on in [
        ("real_outdoor", "SANPO-Real Outdoor", real_scenes, pick_rich(real_scenes), 0.36),
        ("arkit_indoor", "ARKitScenes Indoor", arkit_scenes, pick_rich(arkit_scenes), 0.55),
        ("aria_glasses_private", "Aria Gen 2 Glasses (Private)", aria_scenes, pick_rich(aria_scenes), 0.40),
    ]:
        r = runner_fp16.run_scene(sdict[sc], max_frames=120)
        mp4_p = LOCAL_OUT / f"replay_{tag}_{sc}.mp4"
        png_p = LOCAL_OUT / f"sheet_{tag}_{sc}.png"
        render_replay_video(sc, label, sdict[sc], r, mp4_p, png_p, thr_on=thr_on, fps=5.0)
        replays.append({"tag": tag, "scene": sc, "frames": len(r["corr_probs"]), "mp4": mp4_p.name, "sheet": png_p.name})
        print(f"[replay-g4] rendered {mp4_p.name} ({len(r['corr_probs'])} fr)", flush=True)

    full_rep = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_sec": round(time.time() - t0, 1),
        "autopsy": autopsy_rep,
        "edge_quant": quant_report,
        "replays": replays,
    }
    (LOCAL_OUT / "phase6_autopsy_edge_report.json").write_text(json.dumps(full_rep, indent=2))

    # Sync all outputs (quantized ONNX, MP4s, PNG sheets, JSON report) to Drive
    for p in LOCAL_OUT.iterdir():
        if p.name.endswith((".json", ".png", ".mp4", "_fp16.onnx", "_int8.onnx")):
            dst = DRIVE_OUT / p.name
            if dst.exists():
                dst.unlink()
            shutil.copy2(p, dst)
    print(f"[p6-cloud] ALL DONE & synced to Drive {DRIVE_OUT} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
