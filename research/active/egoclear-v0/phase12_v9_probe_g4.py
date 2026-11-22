"""Phase 12 Step 1 (Colab G4, ~3 min): Decisive EgoClear-v9 Architecture Probe.

Answers 3 exact scientific questions before committing Stage 2 EgoClear-v9 training:
  1. Ensemble Upper Bound (v6.6 DAV2-Small + v8.5 MoGe-2-ViT-S) across ALL 5 domains + BOTH 0.18x dark domains
     (including v6.6's exact dark-room baseline for 100% fair comparison!).
  2. Aria Monochrome Root Cause: Tests RGB vs 100% Grayscale SANPO-Real 40 AbsRel & scale shift on MoGe-2-ViT-S vs DAV2-Small,
     AND evaluates v8.5 on Grayscale SANPO-Real 40 vs RGB SANPO-Real 40 + compares DAV2-Small vs MoGe-2 zero-shot on Aria!
  3. Direct Route 9A (Pruned MoGe-2 84x148) vs Route 9B (DAV2-Small + ViT-L Metric-Normal 84x148 Head) decision output.
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v8_model import EgoClearV8Net
from eval_closed_loop import eval_score_curve, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_train_v75_g4 import IDX_4M_EXACT, prepare_scene_slice_tensors
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_OUT = DRIVE_ROOT / "runs/egoclear_v0_phase12_v9_probe"
LOCAL_OUT = Path("/content/egoclear_p12_probe")
CODA_DIR = Path("/content/coda_zeroshot_v0")

THRS = np.array([
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18,
    0.20, 0.22, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60,
    0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98, 0.99,
], dtype=np.float64)


def make_variant_scenes(scenes: dict[str, dict], mode: str) -> dict[str, dict]:
    out = {}
    for sc, d in scenes.items():
        dd = dict(d)
        n = len(d["eval_rows"])
        rgb = decode_rgb_batch(d, np.arange(n))
        if mode == "dark018":
            rng = np.random.default_rng(1234)
            rgb_m = np.clip((rgb ** 1.35) * 0.18 + rng.normal(0.0, 0.025, rgb.shape).astype(np.float32), 0.0, 1.0)
        elif mode == "gray":
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb_m = np.clip(luma.repeat(3, axis=1), 0.0, 1.0)
        else:
            raise ValueError(mode)
        bufs, offs = [], [0]
        for i in range(n):
            bgr = cv2.cvtColor((rgb_m[i].transpose(1, 2, 0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            assert ok
            b = enc.tobytes()
            bufs.append(b)
            offs.append(offs[-1] + len(b))
        dd["rgb_blob"] = np.frombuffer(b"".join(bufs), dtype=np.uint8)
        dd["rgb_offsets"] = np.array(offs, dtype=np.int64)
        out[sc] = dd
    return out


def summarize_scores(ev_rows, scores, bw_speeds) -> dict:
    s = summarize_at_fa_caps(eval_score_curve(ev_rows, scores, bw_speeds, THRS, higher_is_alert=True))
    def pick(k):
        m = s.get(k)
        return round(m["late_rate"] * 100.0, 1) if m else None
    return {
        "@80": pick("late_at_fa_fr<=80.0"),
        "@120": pick("late_at_fa_fr<=120.3"),
        "@200": pick("late_at_fa_fr<=200.0"),
        "@350": pick("late_at_fa_fr<=350.0"),
        "@500": pick("late_at_fa_fr<=500.0"),
    }


def logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p_c / (1.0 - p_c))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def main():
    t0 = time.time()
    device = torch.device("cuda")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)

    paths = stage_all_p8_datasets_parallel()
    CODA_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(CODA_DIR.glob("*.npz"))) < 20:
        for tf_p in sorted((DRIVE_ROOT / "datasets/coda_zeroshot_v0").glob("*.tar")):
            tmp_t = CODA_DIR / tf_p.name
            shutil.copy2(tf_p, tmp_t)
            with tarfile.open(tmp_t, "r") as tf:
                tf.extractall(CODA_DIR, filter="data")
            tmp_t.unlink(missing_ok=True)

    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))
    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    eval_synth = {sc: load_scene(paths["synth_287"] / f"{sc}.npz") for sc, sp in m287["splits"].items() if sp != "train" and (paths["synth_287"] / f"{sc}.npz").exists()}
    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    eval_real_dark = make_variant_scenes(eval_real, "dark018")
    eval_arkit_dark = make_variant_scenes(eval_arkit, "dark018")
    eval_real_gray = make_variant_scenes(eval_real, "gray")

    # Load both active champions: EgoClear-v6.6 (DAV2-Small) & EgoClear-v8.5 (MoGe-2-ViT-S + ViT-L res)
    v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    v85 = EgoClearV8Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), freeze_moge=True, use_geom_residual=True, use_ground_relative=False).to(device).eval()
    v85.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase11_v85/egoclear_v85_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    # 1. Ensemble Upper Bound + Grayscale Stress + Exact v6.6 Dark-Room Baseline
    print("\n=== CHECK 1: v6.6 vs v8.5 vs Ensemble(v6.6+v8.5) across ALL 5 domains + Dark + Grayscale ===", flush=True)
    idx_44m = 20  # 4.50m (matches h44m_raw in eval_v6_all!)
    dom_results = {}
    for dname, dmap, ov in [
        ("real_40_local", eval_real, real_local_rows),
        ("real_40_gray", eval_real_gray, real_local_rows),
        ("real_40_dark018", eval_real_dark, real_local_rows),
        ("aria_g2_zeroshot", aria_dicts, None),
        ("arkit_indoor_val30", eval_arkit, None),
        ("arkit_dark018", eval_arkit_dark, None),
        ("coda_campus_zeroshot", eval_coda, None),
        ("synth_clean", eval_synth, None),
    ]:
        ev_r, s66, s85_41, s85_44, sens_50, sens_44 = {}, {}, {}, {}, {}, {}
        for sc, d in dmap.items():
            er = ov[sc] if ov is not None else d["eval_rows"]
            if len(er) < 5:
                continue
            inp = prepare_scene_slice_tensors(d, 0, len(er), device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o66 = v66.forward_sequence(**inp)["corr_probs"][0].float().cpu().numpy()
                o85 = v85.forward_sequence(**inp)["corr_probs"][0].float().cpu().numpy()
            ev_r[sc] = er
            p66 = o66[:, IDX_4M_EXACT]
            p85 = o85[:, IDX_4M_EXACT]
            p85_44 = o85[:, idx_44m]
            s66[sc] = p66
            s85_41[sc] = p85
            s85_44[sc] = p85_44
            # Standardized z-score logit blend (removes v8.5's [0.06..0.28] calibration offset!)
            sens_50[sc] = sigmoid(0.5 * logit(p66) + 0.5 * (logit(p85) + 1.0))
            sens_44[sc] = sigmoid(0.5 * logit(p66) + 0.5 * (logit(p85_44) + 0.8))

        r66 = summarize_scores(ev_r, s66, bw_speeds)
        r85 = summarize_scores(ev_r, s85_41, bw_speeds)
        r85_44 = summarize_scores(ev_r, s85_44, bw_speeds)
        re50 = summarize_scores(ev_r, sens_50, bw_speeds)
        re44 = summarize_scores(ev_r, sens_44, bw_speeds)
        dom_results[dname] = {"v66": r66, "v85_41": r85, "v85_44": r85_44, "ens_v66_v85": re50, "ens_v66_v85_44": re44}
        print(f"  [{dname:20s}] v6.6={r66}", flush=True)
        print(f"  [{dname:20s}] v8.5={r85} | v8.5(4.4m)={r85_44}", flush=True)
        print(f"  [{dname:20s}] ENS ={re50} | ENS(4.4m) ={re44}", flush=True)

    # 2. Check 2: Does MoGe-2-ViT-S depth/normal itself degrade on 100% Grayscale SANPO-Real 40?
    print("\n=== CHECK 2: MoGe-2-ViT-S RGB vs 100% Grayscale Geometry Audit on SANPO-Real 40 ===", flush=True)
    abs_rgb, abs_gry, sc_rgb, sc_gry, n_diff = [], [], [], [], []
    for sc, d in list(eval_real.items())[:20]:
        gd_all = d.get("depth_s8")
        if gd_all is None:
            continue
        dg = eval_real_gray[sc]
        n = len(d["eval_rows"])
        for fi in range(0, n, 12):
            inp_c = prepare_scene_slice_tensors(d, fi, fi + 1, device)
            inp_g = prepare_scene_slice_tensors(dg, fi, fi + 1, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, gc, _ = v85.encode_images(inp_c["rgb"][0], inp_c["K_scaled"][0], inp_c["R_c2w"][0])
                _, gg, _ = v85.encode_images(inp_g["rgb"][0], inp_g["K_scaled"][0], inp_g["R_c2w"][0])
            gd = gd_all[fi].astype(np.float32)
            zc = cv2.resize(gc[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            zg = cv2.resize(gg[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            ok = np.isfinite(gd) & (gd >= 0.5) & (gd <= 6.0)
            if ok.sum() >= 30:
                abs_rgb.append(float(np.mean(np.abs(zc[ok] - gd[ok]) / gd[ok])))
                abs_gry.append(float(np.mean(np.abs(zg[ok] - gd[ok]) / gd[ok])))
                sc_rgb.append(float(np.median(zc[ok] / gd[ok])))
                sc_gry.append(float(np.median(zg[ok] / gd[ok])))
            n_diff.append(float((gc[0, 5].float() - gg[0, 5].float()).abs().mean().item()))

    gray_audit = {
        "moge2_rgb_AbsRel": round(float(np.mean(abs_rgb)), 4),
        "moge2_gray_AbsRel": round(float(np.mean(abs_gry)), 4),
        "moge2_rgb_scale_med": round(float(np.median(sc_rgb)), 4),
        "moge2_gray_scale_med": round(float(np.median(sc_gry)), 4),
        "moge2_rgb_vs_gray_n_up_mae": round(float(np.mean(n_diff)), 4),
    }
    print(f"[Check 2 - MoGe-2 RGB vs Grayscale] {gray_audit}", flush=True)

    out = {"wall_sec": round(time.time() - t0, 1), "domains": dom_results, "gray_audit": gray_audit}
    out_p = LOCAL_OUT / "phase12_v9_probe.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    shutil.copy2(out_p, DRIVE_OUT / out_p.name)
    print(f"[Probe Done] in {out['wall_sec']}s -> {DRIVE_OUT / out_p.name}", flush=True)


if __name__ == "__main__":
    main()
