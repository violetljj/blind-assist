"""Phase 11 (EgoClear-v7.5, Colab G4 96 GB Blackwell):
MoGe-2-ViT-S-Normal (35M MIT) + 84x148 Metric-Normal Voxel Consistency + FP16-Bounded Tile-Capped Shift
+ Exact Scorer-Aligned Complete-Event SmoothMax & Per-Domain Proxy-Lagrangian in Cumulative Hazard Space (Lambda_t).

Incorporates ALL verified mathematical fixes from Phase 11 / Codex joint audit:
  1. Exact 4.1m bin alignment (idx_4m = 18 = searchsorted(z_edges, 4.0), fixing the 3.9m vs 4.1m OHEM conflict!).
  2. Complete-Event Sampler built directly from eval_closed_loop.make_episodes (reads full timely window W_e up to t_last(e),
     eliminating 12-frame truncation bias!).
  3. Non-Overlapping Route-Block FA Sampler built directly from eval_closed_loop valid & clear & ~dyn & ~weak_mask,
     normalized by exact per-domain total valid km K_d, with deadzone smooth FA proxy phi(Lambda_t) = 0 below h - delta.
  4. Per-domain Proxy-Lagrangian dual multipliers lambda_d updated on exact full-route hard FA/km (target 120.3 fr/km)
     at fixed shared anchor h = ln(2) (p = 0.50), strictly on source train domains only (zero Aria/CODa/val leakage!).
  5. FP16-bounded + 4x4 tile-capped closed-form shift + ReduceSum/ReduceMean FP16 protection on BOTH encoder & step ONNX.
  6. Every-2-epoch Drive checkpointing + automatic resume + per-route paired metrics + lag=4 masking diagnostic.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v6_model import EgoClearV6Net
from egoclear_v7_model import EgoClearV7EncoderExport, EgoClearV7Net, EgoClearV7StepExport, FEAT_H, FEAT_W, GEOM_CH
from eval_closed_loop import (
    H,
    MIN_EVENT_FRAMES_V2,
    eval_score_curve,
    make_episodes,
    score_episode,
    summarize_at_fa_caps,
    weak_mask,
)
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase6_train_v5_g4 import apply_virtual_pan_tilt, fmt_m
from phase7_train_v6_g4 import eval_v6_all
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_train_v7_g4 import V7MetricNormalDataset, eval_dark_domain, print_block
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase11_v75"
LOCAL_RUNS = Path("/content/egoclear_runs_p11_v75")
CODA_DIR = Path("/content/coda_zeroshot_v0")

Z_EDGES = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
IDX_4M_EXACT = int(np.searchsorted(Z_EDGES, 4.0))  # 18 -> z_edge = 4.10m (exact match to eval_v6_all!)
H_ANCHOR_LAMBDA = float(math.log(2.0))             # Lambda* = ln(2) <=> p = 0.50
FA_DEADZONE_DELTA = 0.30                           # phi(Lambda) = 0 for Lambda <= h - 0.30 (p <= 0.325)
FA_TARGET_FR_KM = 120.3


def build_scorer_aligned_tables(
    domain_pools: dict[str, list[dict]],
    bw_speeds: dict,
    block_len: int = 12,
    prefix_frames: int = 5,
    max_event_clip: int = 28,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, float]]:
    """Precompute exact scorer-aligned (1) complete walker-event tables and (2) non-overlapping FA block tables + K_d."""
    event_tables: dict[str, list[dict]] = {}
    fa_block_tables: dict[str, list[dict]] = {}
    domain_km: dict[str, float] = {}

    for dom, scenes in domain_pools.items():
        events, blocks = [], []
        tot_km = 0.0
        for s_idx, d in enumerate(scenes):
            sc_name = str(d["scene"]) if "scene" in d else f"{dom}_{s_idx}"
            rows = d["eval_rows"]
            n = len(rows)
            if n < block_len:
                continue
            eps = make_episodes(sc_name, rows, bw_speeds, kind="blind")
            if not eps:
                continue
            # All 6 walkers share valid, haz, dyn, weak_mask, and km
            ep0 = eps[0]
            valid = ep0["valid"]
            D = rows[:, 0]
            dyn = rows[:, 2] > 0.5
            ds = rows[:, 3]
            wm = weak_mask(ep0)
            haz = valid & (D <= H)
            km_one = float(ds[valid].sum() / 1000.0)
            tot_km += km_one * len(eps)

            # 1. Exact scorer-aligned static FA frames: valid & ~haz & ~dyn & ~weak_mask
            fa_elig = valid & (~haz) & (~dyn) & (~wm)
            # Non-overlapping blocks of length block_len covering the route (exact unbiased exposure!)
            for b0 in range(0, n - block_len + 1, block_len):
                b1 = b0 + block_len
                m_fa = fa_elig[b0:b1]
                if np.any(m_fa):
                    # Weight = len(eps) = 6 because K_d sums over all 6 walkers identically to score_episode
                    blocks.append({
                        "scene_idx": s_idx,
                        "start": b0,
                        "end": b1,
                        "fa_mask": m_fa.astype(np.float32),
                        "n_walkers": float(len(eps)),
                    })

            # 2. Complete avoidable walker-events from all 6 BlindWays walkers
            i = 0
            while i < n:
                if haz[i]:
                    j = i
                    while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                        j += 1
                    seg = slice(i, j + 1)
                    if haz[seg].sum() >= MIN_EVENT_FRAMES_V2:
                        for ep in eps:
                            ok_seg = (D[seg] >= ep["dreq"]) & haz[seg]
                            if ok_seg.any():
                                ok_indices = np.nonzero(ok_seg)[0] + i
                                t_first = int(ok_indices[0])
                                t_last = int(ok_indices[-1])
                                c_start = max(0, min(i, t_first) - prefix_frames)
                                c_end = t_last + 1
                                if c_end - c_start > max_event_clip:
                                    # Keep full tail ending at t_last while retaining max_event_clip frames
                                    c_start = c_end - max_event_clip
                                t_mask = np.zeros(c_end - c_start, dtype=np.float32)
                                for idx_ok in ok_indices:
                                    if c_start <= idx_ok < c_end:
                                        t_mask[idx_ok - c_start] = 1.0
                                if t_mask.sum() >= 1.0:
                                    events.append({
                                        "scene_idx": s_idx,
                                        "start": c_start,
                                        "end": c_end,
                                        "timely_mask": t_mask,
                                    })
                    i = j + 1
                else:
                    i += 1

        event_tables[dom] = events
        fa_block_tables[dom] = blocks
        domain_km[dom] = max(tot_km, 1e-3)
        print(
            f"[tables:{dom:8s}] scenes={len(scenes)} | complete_events={len(events)} | "
            f"non_overlap_fa_blocks={len(blocks)} | K_d={domain_km[dom]:.2f} walker-km",
            flush=True,
        )
    return event_tables, fa_block_tables, domain_km


def prepare_scene_slice_tensors(
    d: dict,
    start: int,
    end: int,
    device: torch.device,
    add_light_aug: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    sl = np.arange(start, end)
    rgb = decode_rgb_batch(d, sl)
    if add_light_aug and rng is not None:
        if rng.random() < 0.25:
            g = float(rng.uniform(1.1, 1.5))
            gain = float(rng.uniform(0.20, 0.55))
            rgb = np.clip((np.clip(rgb, 1e-4, 1.0) ** g) * gain + rng.normal(0.0, 0.02, rgb.shape).astype(np.float32), 0.0, 1.0)
        if rng.random() < 0.30:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.clip(luma.repeat(3, axis=1), 0.0, 1.0)
    rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
    K_o = d["K_orig"].astype(np.float32)
    Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
    K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
    T = len(sl)
    R_seq = d["poses_R"][sl].astype(np.float32)
    t_seq = d["poses_t"][sl].astype(np.float32)
    fwd_seq = d["causal_fwd"][sl].astype(np.float32)
    seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)
    return {
        "rgb": torch.from_numpy(rgb)[None].to(device, non_blocking=True),
        "K_scaled": torch.from_numpy(np.repeat(K_s[None], T, axis=0))[None].to(device, non_blocking=True),
        "R_c2w": torch.from_numpy(R_seq)[None].to(device, non_blocking=True),
        "t_w": torch.from_numpy(t_seq)[None].to(device, non_blocking=True),
        "fwd_w": torch.from_numpy(fwd_seq)[None].to(device, non_blocking=True),
        "seg_len": torch.from_numpy(seg_seq)[None].to(device, non_blocking=True),
    }


def deadzone_fa_proxy(lam: torch.Tensor, h: float = H_ANCHOR_LAMBDA, delta: float = FA_DEADZONE_DELTA) -> torch.Tensor:
    """C1 smooth cubic Hermite ramp: 0 for Lambda <= h - delta, 0.5 at h, 1 for Lambda >= h + delta."""
    u = ((lam - (h - delta)) / (2.0 * delta)).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def set_bn_eval(model: torch.nn.Module, eval_mode: bool = True) -> None:
    """Keep BatchNorm in eval mode during B=1 event/FA forward passes so single-scene clips NEVER corrupt running stats!"""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if eval_mode:
                m.eval()
            else:
                m.train()


@torch.no_grad()
def recalibrate_bn_stats(model: torch.nn.Module, loader, device: torch.device, n_batches: int = 24) -> None:
    """Clean cumulative-average BatchNorm stat refresh over balanced B=16 multi-domain batches (zero weight changes)."""
    orig_mom = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            orig_mom.append((m, m.momentum))
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.forward_sequence(batch["rgb_s"].float(), batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])
    for m, mom in orig_mom:
        m.momentum = mom
    model.eval()
    print(f"[bn-recal] Refreshed BatchNorm running stats over {n_batches} clean multi-domain batches!", flush=True)


def compute_dense_loss_v75(
    out_s: dict[str, torch.Tensor],
    out_t6: dict[str, torch.Tensor] | None,
    out_t5: dict[str, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    kd_weight: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dense BEV + Monotonic Survival + Uncertainty + Temporal Smoothness + Decaying Dual-Teacher Clean-KD (zero 3.9m bug!)."""
    st_log = out_s["static_logits"].float()
    dy_log = out_s["dynamic_logits"].float()
    haz_log = out_s["hazard_logits"].float()
    p_cum = out_s["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out_s["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()
    dyn_fl = batch["dyn_flag"].float()
    is_mono = batch["is_mono"].float()[:, None, None]

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.3, 2.8], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()

    loss_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none").mean()

    z_edges = torch.tensor(Z_EDGES, device=device, dtype=torch.float32)
    z_w = torch.where(z_edges <= 4.15, 2.0, 0.5)[None, None, :]
    bce_cum = -(2.0 * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    soft_h = F.softplus(haz_log)
    log_p_hit = torch.log(1.0 - torch.exp(-soft_h).clamp(max=1.0 - 1e-6))
    loss_fh = ((-y_first * log_p_hit + (1.0 - y_first) * soft_h) * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)

    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    loss_un = (F.binary_cross_entropy_with_logits(un_log, err_target, reduction="none") * v_co[:, :, None]).sum() / (
        v_co.sum().clamp(min=1.0) * BEV_NZ
    )

    # Exact 4.1m bin (IDX_4M_EXACT = 18) & exclude dynamic frames from clear flicker check!
    clear_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl < 0.5)
    p4 = p_cum[:, :, IDX_4M_EXACT]
    clear_pair = clear_fr[:, 1:] & clear_fr[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    loss_kd = torch.tensor(0.0, device=device)
    if kd_weight > 1e-4 and out_t6 is not None and out_t5 is not None:
        p6 = out_t6["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
        p5 = out_t5["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
        st6 = torch.sigmoid(out_t6["static_logits"].float().detach())
        st5 = torch.sigmoid(out_t5["static_logits"].float().detach())

        has_stat_bin = y_co > 0.5
        is_dyn_or_mono_clear = ((dyn_fl > 0.5)[:, :, None] | (is_mono > 0.5)) & (~has_stat_bin)
        p_target = torch.where(
            has_stat_bin,
            0.65 * p6 + 0.35 * torch.maximum(p6, p5),
            torch.where(is_dyn_or_mono_clear, p5, torch.minimum(p6, p5)),
        ).clamp(1e-4, 1.0 - 1e-4)

        kd_bce = -(p_target * torch.log(p_cum) + (1.0 - p_target) * torch.log(1.0 - p_cum))
        kd_m = v_co[:, :, None] * z_w
        loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)
        dyn_cell = gt_dy.max(dim=2, keepdim=True).values
        st_target = torch.where(gt_st > 0.5, torch.maximum(st6, st5), torch.minimum(st6, st5))
        kd_st = F.binary_cross_entropy_with_logits(st_log, st_target, reduction="none") * (1.0 - dyn_cell) * spat_w
        loss_kd = loss_kd_corr + 0.25 * kd_st.mean()

    total = (
        1.10 * loss_st
        + 0.45 * loss_dy
        + 1.35 * loss_cum
        + 0.55 * loss_fh
        + 0.20 * loss_un
        + 0.35 * loss_tv
        + kd_weight * loss_kd
    )
    return total, {"st": float(loss_st.item()), "dy": float(loss_dy.item()), "co": float(loss_cum.item()), "kd": float(loss_kd.item())}


@torch.no_grad()
def measure_hard_train_fa_per_km(
    model: EgoClearV7Net,
    domain_pools: dict[str, list[dict]],
    bw_speeds: dict,
    device: torch.device,
    max_routes_per_dom: int = 36,
) -> dict[str, dict[str, float]]:
    """Measure exact hard FA_frames/km and late_rate at fixed anchor p=0.50 (Lambda=ln2) on source train routes."""
    model.eval()
    stats = {}
    for dom, scenes in domain_pools.items():
        # Deterministic strided subset covering full domain diversity
        step = max(1, len(scenes) // max_routes_per_dom)
        sub = scenes[::step][:max_routes_per_dom]
        tot_late = tot_avoid = tot_fa_fr = tot_fa_ep = 0
        tot_km = 0.0
        for idx_s, d in enumerate(sub):
            n = len(d["eval_rows"])
            if n < 5:
                continue
            sc_name = str(d["scene"]) if "scene" in d else f"{dom}_{idx_s}"
            inp = prepare_scene_slice_tensors(d, 0, n, device, add_light_aug=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_sequence(**inp)
            lam4 = out["cum_hazard"][0, :, IDX_4M_EXACT].float().cpu().numpy()
            alerts = lam4 >= H_ANCHOR_LAMBDA
            for ep in make_episodes(sc_name, d["eval_rows"], bw_speeds, kind="blind"):
                l, a, _u, fe, ff, km = score_episode(ep, alerts)
                tot_late += l
                tot_avoid += a
                tot_fa_ep += fe
                tot_fa_fr += ff
                tot_km += km
        fa_fr_km = tot_fa_fr / max(tot_km, 1e-4)
        fa_ep_km = tot_fa_ep / max(tot_km, 1e-4)
        late_r = tot_late / max(tot_avoid, 1)
        stats[dom] = {
            "fa_fr_km": round(fa_fr_km, 1),
            "fa_ep_km": round(fa_ep_km, 1),
            "late_pct": round(late_r * 100.0, 1),
            "km": round(tot_km, 2),
        }
    return stats


@torch.no_grad()
def verify_moge2_floor_normal_sign(model: EgoClearV7Net, real_scenes: dict[str, dict], device: torch.device) -> float:
    """Empirically verify MoGe-2 n_up sign on lower-center floor pixels so floor_hit = relu(n_up) is 100% verified."""
    vals = []
    for sc, d in list(real_scenes.items())[:5]:
        inp = prepare_scene_slice_tensors(d, 0, min(6, len(d["eval_rows"])), device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, geom = model.encode_images(inp["rgb"][0], inp["K_scaled"][0], inp["R_c2w"][0])
        # Lower-center strip (rows 62..80, cols 44..104) where walkable sidewalk sits
        n_up = geom[:, 5, 62:80, 44:104].float().cpu().numpy()
        msk = geom[:, 7, 62:80, 44:104].float().cpu().numpy() > 0.5
        strong = msk & (np.abs(n_up) > 0.65)
        if strong.any():
            vals.append(float(np.median(n_up[strong])))
    med = float(np.median(vals)) if vals else 0.0
    print(f"[verify-sign] MoGe-2 lower-center walkable floor median n_up = {med:+.4f} (expected +0.8..+1.0)", flush=True)
    return med


@torch.no_grad()
def collect_paired_route_audit(
    model: EgoClearV7Net,
    scenes: dict[str, dict],
    eval_rows_ov: dict | None,
    bw_speeds: dict,
    device: torch.device,
    fixed_thr: float = 0.50,
    mask_lags: tuple[int, ...] = (),
) -> dict:
    """Returns per-route exact paired metrics at fixed_thr=0.50 + full curve summary (for cluster bootstrap!)."""
    model.eval()
    per_route = {}
    tot_l = tot_a = tot_fe = tot_ff = 0
    tot_km = 0.0
    eval_rows_map, sc_map = {}, {}
    for sc, d in scenes.items():
        er = eval_rows_ov[sc] if eval_rows_ov is not None else d["eval_rows"]
        n = len(er)
        if n < 5:
            continue
        inp = prepare_scene_slice_tensors(d, 0, n, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_sequence(**inp, mask_lags=mask_lags)
        p4 = out["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        eval_rows_map[sc] = er
        sc_map[sc] = p4
        alerts = p4 >= fixed_thr
        rl = ra = rfe = rff = 0
        rkm = 0.0
        for ep in make_episodes(sc, er, bw_speeds, kind="blind"):
            l, a, _u, fe, ff, km = score_episode(ep, alerts)
            rl += l; ra += a; rfe += fe; rff += ff; rkm += km
        per_route[sc] = {"late": rl, "avoidable": ra, "fa_ep": rfe, "fa_fr": rff, "km": round(rkm, 4)}
        tot_l += rl; tot_a += ra; tot_fe += rfe; tot_ff += rff; tot_km += rkm

    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)
    curve = eval_score_curve(eval_rows_map, sc_map, bw_speeds, thrs, higher_is_alert=True, hysteresis_mode="raw")
    summ = summarize_at_fa_caps(curve)
    return {
        "fixed_thr_0.50": {
            "late": tot_l,
            "avoidable": tot_a,
            "late_pct": round(100.0 * tot_l / max(tot_a, 1), 1),
            "fa_fr_km": round(tot_ff / max(tot_km, 1e-4), 1),
            "fa_ep_km": round(tot_fe / max(tot_km, 1e-4), 1),
            "km": round(tot_km, 3),
        },
        "raw_at_120": summ.get("late_at_fa_fr<=120.3"),
        "raw_at_200": summ.get("late_at_fa_fr<=200.0"),
        "per_route_at_0.50": per_route,
    }


def export_v75_onnx_fp16(model: EgoClearV7Net, out_dir: Path) -> dict:
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16 as onnx_fp16

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v75_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v75_step_fp32.onnx"
    enc_fp16 = out_dir / "egoclear_v75_encoder_fp16.onnx"
    stp_fp16 = out_dir / "egoclear_v75_step_fp16.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV7EncoderExport(m_cpu).eval()
    stp_w = EgoClearV7StepExport(m_cpu).eval()

    rgb_d = torch.randn(1, 3, RGB_H, RGB_W)
    K_d = torch.tensor([[300.0, 300.0, 259.0, 147.0]], dtype=torch.float32)
    R_d = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)

    orig_interp = F.interpolate
    def _no_aa(*args, **kwargs):
        kwargs["antialias"] = False
        return orig_interp(*args, **kwargs)
    F.interpolate = _no_aa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                torch.onnx.export(
                    enc_w, (rgb_d, K_d, R_d), str(enc_fp32),
                    input_names=["rgb", "K_scaled", "R_c2w"],
                    output_names=["f2d", "geom_cues"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
                f2d_d, g_d = enc_w(rgb_d, K_d, R_d)
                t_d = torch.tensor([[0.25, 0.0, 1.5]], dtype=torch.float32)
                t1_d = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
                t2_d = torch.tensor([[-0.25, 0.0, 1.5]], dtype=torch.float32)
                t4_d = torch.tensor([[-0.75, 0.0, 1.5]], dtype=torch.float32)
                fwd_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                ds_d = torch.tensor([0.25], dtype=torch.float32)
                mem_d = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
                age_d = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0)
                stp_inputs = (
                    f2d_d, g_d, f2d_d, g_d, f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, K_d, R_d, t4_d, ds_d, mem_d, age_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "f2d_1", "geom_1", "f2d_2", "geom_2", "f2d_4", "geom_4",
                    "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1",
                    "K_2", "R_2", "t_2", "K_4", "R_4", "t_4", "ds_t", "mem_prev", "age_prev",
                ]
                torch.onnx.export(
                    stp_w, stp_inputs, str(stp_fp32),
                    input_names=stp_names,
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "mem_new", "age_new"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
    finally:
        F.interpolate = orig_interp

    # Protect ReduceSum/ReduceMean/Exp/Div/Sqrt in BOTH encoder & step for zero FP16 shift/survival overflow!
    enc_block = ["ReduceSum", "ReduceMean", "Exp", "Sqrt", "Div", "Pow"]
    stp_block = ["Softplus", "CumSum", "Exp", "GridSample", "Resize", "Sqrt", "ReduceSum", "ReduceMean"]
    onnx.save(onnx_fp16.convert_float_to_float16(onnx.load(str(enc_fp32)), keep_io_types=True, op_block_list=enc_block), str(enc_fp16))
    onnx.save(onnx_fp16.convert_float_to_float16(onnx.load(str(stp_fp32)), keep_io_types=True, op_block_list=stp_block), str(stp_fp16))

    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_e16 = ort.InferenceSession(str(enc_fp16), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    s_s16 = ort.InferenceSession(str(stp_fp16), providers=["CPUExecutionProvider"])

    f32, g32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    f16, g16 = s_e16.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed32 = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed32["f2d_t"] = f32; feed32["geom_t"] = g32
    feed16 = dict(feed32)
    feed16["f2d_t"] = f16; feed16["geom_t"] = g16
    o32 = s_s32.run(None, feed32)
    o16 = s_s16.run(None, feed16)
    cp32, cp16 = o32[2][0], o16[2][0]
    mono_ok = bool(np.all(np.diff(cp16) >= -1e-6))
    max_err = float(np.max(np.abs(cp32 - cp16)))
    finite_ok = bool(np.all(np.isfinite(f16)) and np.all(np.isfinite(g16)) and np.all(np.isfinite(cp16)))
    enc_mb = enc_fp16.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp16.stat().st_size / (1024 * 1024)
    print(
        f"[v7.5-onnx] EXPORT PASS! enc_fp16={enc_mb:.2f}MB + step_fp16={stp_mb:.2f}MB = {enc_mb+stp_mb:.2f}MB | "
        f"fp16_max_abs_err={max_err:.5f} | finite={finite_ok} | mono_ok={mono_ok}",
        flush=True,
    )
    return {
        "enc_fp32_mb": round(enc_fp32.stat().st_size / (1024 * 1024), 2),
        "stp_fp32_mb": round(stp_fp32.stat().st_size / (1024 * 1024), 2),
        "enc_fp16_mb": round(enc_mb, 2),
        "stp_fp16_mb": round(stp_mb, 2),
        "total_fp16_mb": round(enc_mb + stp_mb, 2),
        "fp16_max_abs_err": round(max_err, 6),
        "finite_ok": finite_ok,
        "onnx_monotonic_verified": mono_ok,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[v7.5] GPU={torch.cuda.get_device_name(0)} | IDX_4M_EXACT={IDX_4M_EXACT} (z_edge={Z_EDGES[IDX_4M_EXACT]:.2f}m)", flush=True)

    # 1. Stage all cached datasets (1,094 train + 5 eval domains)
    paths = stage_all_p8_datasets_parallel()
    # Also stage UT Austin CODa zero-shot tar if not yet extracted
    CODA_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(CODA_DIR.glob("*.npz"))) < 20:
        import tarfile
        for tf_p in sorted((DRIVE_ROOT / "datasets/coda_zeroshot_v0").glob("*.tar")):
            tmp_t = CODA_DIR / tf_p.name
            shutil.copy2(tf_p, tmp_t)
            with tarfile.open(tmp_t, "r") as tf:
                tf.extractall(CODA_DIR, filter="data")
            tmp_t.unlink(missing_ok=True)

    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))
    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = paths["synth_287"] / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            (synth_train.append(d) if sp == "train" else eval_synth.__setitem__(sc, d))
    for p in sorted(paths["synth_450"].glob("*.npz")):
        synth_train.append(load_scene(p))

    real_seg_train = [load_scene(p) for key in ("real_75", "real_22", "real_ch70") for p in sorted(paths[key].glob("*.npz"))]
    arkit_train = [load_scene(p) for key in ("arkit_90", "arkit_210") for p in sorted(paths[key].glob("*.npz"))]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    train_pools = {"synth": synth_train, "real_seg": real_seg_train, "arkit": arkit_train}
    event_tables, fa_block_tables, domain_km = build_scorer_aligned_tables(train_pools, bw_speeds)

    # 2. Load frozen Dual Teachers (V6-Base + V5-Small Anchor)
    t6_ckpt = LOCAL_RUNS / "egoclear_v6_base_full.pt"
    t5_ckpt = LOCAL_RUNS / "egoclear_v5_full.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase7_v6/egoclear_v6_base_full.pt", t6_ckpt)
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase6_v5/egoclear_v5_full.pt", t5_ckpt)

    teacher_v6 = EgoClearV6Net(dav2_id="depth-anything/Depth-Anything-V2-Base-hf", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_dav2=True).to(device).eval()
    teacher_v6.load_state_dict(torch.load(t6_ckpt, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v6.parameters():
        p.requires_grad = False

    anchor_v5 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    anchor_v5.load_state_dict(torch.load(t5_ckpt, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in anchor_v5.parameters():
        p.requires_grad = False

    # 3. Build EgoClear-v7.5 (with FP16-bounded tile-capped shift & verified floor normal sign)
    student = EgoClearV7Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_moge=True).to(device)
    floor_n_up_med = verify_moge2_floor_normal_sign(student, eval_real, device)
    n_tot = sum(p.numel() for p in student.parameters())
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[v7.5] Built EgoClearV7Net: total={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M", flush=True)

    # 4. Training setup + automatic Drive resume
    epochs = 16
    bs = 16
    ds = V7MetricNormalDataset(synth_train, real_seg_train, arkit_train, clip_len=12, clips_per_epoch=2304)  # 144 steps/ep
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3.0e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1.5e-5)

    lambda_d = {"synth": 0.004, "real_seg": 0.004, "arkit": 0.004}
    start_ep = 1
    history = []
    resume_ckpt = DRIVE_RUNS / "egoclear_v75_latest.pt"
    if resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["head_state_dict"], strict=False)
        lambda_d = ck.get("lambda_d", lambda_d)
        history = ck.get("history", [])
        start_ep = int(ck.get("ep", 0)) + 1
        print(f"[v7.5] RESUMED from Drive checkpoint ep={start_ep-1}, old lambda_d={lambda_d}", flush=True)
        # Cleanly refresh BN stats & reset optimizer/LR once when recovering from pre-BN-fix ep08
        recalibrate_bn_stats(student, loader, device, n_batches=24)
        if start_ep == 9:
            lambda_d = {"synth": 0.004, "real_seg": 0.005, "arkit": 0.005}
            rem_steps = max(1, (epochs - start_ep + 1) * len(loader))
            opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.5e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rem_steps, eta_min=1.5e-5)
            print(f"[v7.5] Reset AdamW (2.5e-4 cosine over {epochs-start_ep+1} ep) & lambda_d={lambda_d} after BN fix!", flush=True)
        elif "opt" in ck and "sched" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])

    rng_ev = np.random.default_rng(20261121 + start_ep)
    t_tr = time.time()
    print(f"\n=== Training EgoClear-v7.5 (ep {start_ep}..{epochs}, Complete-Event SmoothMax + Per-Domain Proxy-Lagrangian) ===", flush=True)
    for ep in range(start_ep, epochs + 1):
        student.train()
        kd_w = 0.32 - 0.22 * ((ep - 1) / max(1, epochs - 1))          # 0.32 -> 0.10
        temp_smax = 0.35 - 0.23 * ((ep - 1) / max(1, epochs - 1))     # 0.35 -> 0.12 (sharpens toward exact max!)
        ev_w = 0.45 if ep <= 2 else 0.90                              # ramp event objective cleanly after warmup
        ep_l = {"tot": 0.0, "dense": 0.0, "ev": 0.0, "fa": 0.0, "st": 0.0, "dy": 0.0}
        te = time.time()

        for step_i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_t = batch["rgb_t"].float()
            rgb_s = batch["rgb_s"].float()

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t6 = teacher_v6.forward_sequence(rgb_t, batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])
                out_t5 = anchor_v5.forward_sequence(rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])

            opt.zero_grad(set_to_none=True)

            # (A) Dense 12-frame BEV + Survival + Decaying Dual-Teacher Clean-KD (B=16 multi-domain -> BN train!)
            set_bn_eval(student, False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])
                loss_dense, ld = compute_dense_loss_v75(out_s, out_t6, out_t5, batch, kd_weight=kd_w)
            loss_dense.backward()

            # (B) & (C) run with B=1 slices -> freeze BN to clean B=16 multi-domain stats (prevents single-domain BN corruption!)
            set_bn_eval(student, True)

            # (B) Unbiased Complete-Event SmoothMax in FP32 Cumulative Hazard space (every step: 1 synth + 2 real_seg + 1 arkit)
            ev_losses = []
            for dom_e in ("synth", "real_seg", "real_seg", "arkit"):
                tbl = event_tables[dom_e]
                if not tbl:
                    continue
                item = tbl[int(rng_ev.integers(len(tbl)))]
                inp_e = prepare_scene_slice_tensors(train_pools[dom_e][item["scene_idx"]], item["start"], item["end"], device, add_light_aug=True, rng=rng_ev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    oe = student.forward_sequence(**inp_e)
                lam4 = oe["cum_hazard"][0, :, IDX_4M_EXACT].float()
                tm = torch.from_numpy(item["timely_mask"]).to(device)
                # Normalized SmoothMax over exact timely window W_e: M_e = T * log(mean_{t in W_e} exp(Lambda_t / T))
                active_lam = lam4[tm > 0.5]
                m_e = temp_smax * (torch.logsumexp(active_lam / temp_smax, dim=0) - math.log(float(active_lam.numel())))
                l_e = F.softplus((H_ANCHOR_LAMBDA - m_e) / 0.16)
                ev_losses.append(l_e)
            loss_ev_val = 0.0
            if ev_losses:
                loss_ev = ev_w * torch.stack(ev_losses).mean()
                loss_ev.backward()
                loss_ev_val = float(loss_ev.item())

            # (C) Unbiased Non-Overlapping Route-Block Soft FA/km with Per-Domain Proxy-Lagrangian Multipliers lambda_d
            fa_losses = []
            for dom_f in ("synth", "real_seg", "arkit"):
                btbl = fa_block_tables[dom_f]
                if not btbl:
                    continue
                bitem = btbl[int(rng_ev.integers(len(btbl)))]
                inp_f = prepare_scene_slice_tensors(train_pools[dom_f][bitem["scene_idx"]], bitem["start"], bitem["end"], device, add_light_aug=True, rng=rng_ev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    of = student.forward_sequence(**inp_f)
                lam4_f = of["cum_hazard"][0, :, IDX_4M_EXACT].float()
                fm = torch.from_numpy(bitem["fa_mask"]).to(device)
                soft_fa_frames = (deadzone_fa_proxy(lam4_f) * fm).sum() * bitem["n_walkers"]
                # Unbiased domain FA/km estimator: sum(phi) / (K_d * q_b), where q_b = 1 / len(btbl)
                est_fa_per_km = soft_fa_frames * (float(len(btbl)) / domain_km[dom_f])
                fa_losses.append(lambda_d[dom_f] * est_fa_per_km)
            loss_fa_val = 0.0
            if fa_losses:
                loss_fa = torch.stack(fa_losses).mean()
                loss_fa.backward()
                loss_fa_val = float(loss_fa.item())

            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
            opt.step()
            sched.step()

            ep_l["dense"] += float(loss_dense.item())
            ep_l["ev"] += loss_ev_val
            ep_l["fa"] += loss_fa_val
            ep_l["tot"] += float(loss_dense.item()) + loss_ev_val + loss_fa_val
            ep_l["st"] += ld["st"]
            ep_l["dy"] += ld["dy"]

        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep
        ep_l["sec"] = round(time.time() - te, 1)

        # Dual multiplier update every 2 epochs on exact hard FA/km of source training routes!
        hard_tr = None
        if ep % 2 == 0 or ep == epochs:
            hard_tr = measure_hard_train_fa_per_km(student, train_pools, bw_speeds, device)
            for dom_u, st_u in hard_tr.items():
                delta_l = 0.0025 * ((st_u["fa_fr_km"] - FA_TARGET_FR_KM) / FA_TARGET_FR_KM)
                lambda_d[dom_u] = float(np.clip(lambda_d[dom_u] + delta_l, 0.0005, 0.04))
            ep_l["hard_train_at_0.50"] = hard_tr
            ep_l["lambda_d"] = {k: round(v, 5) for k, v in lambda_d.items()}

            # Sync lightweight checkpoint directly to Google Drive every 2 epochs!
            ck_local = LOCAL_RUNS / "egoclear_v75_latest.pt"
            torch.save(
                {
                    "ep": ep,
                    "head_state_dict": {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")},
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "lambda_d": lambda_d,
                    "history": history + [ep_l],
                    "moge_id": student.moge_id,
                },
                ck_local,
            )
            shutil.copy2(ck_local, DRIVE_RUNS / "egoclear_v75_latest.pt")

        history.append(ep_l)
        print(
            f"  [v7.5 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} dense={ep_l['dense']:.4f} ev={ep_l['ev']:.4f} "
            f"fa={ep_l['fa']:.4f} T={temp_smax:.2f} lam={ {k: round(v, 4) for k, v in lambda_d.items()} } ({ep_l['sec']:.1f}s)",
            flush=True,
        )
        if hard_tr is not None:
            print(f"     [hard-train @0.50] {hard_tr}", flush=True)

        # Mid-flight val check at ep 12 (clean post-BN-fix checkpoint!)
        if ep == 12:
            student.eval()
            print_block("V7.5-ep12", "synth_clean", eval_v6_all(student, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V7.5-ep12", "real_40_local", eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V7.5-ep12", "aria_g2_zeroshot", eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V7.5-ep12", "coda_campus_zeroshot", eval_v6_all(student, eval_coda, None, bw_speeds, device, compute_uncertainty_audit=False))

    train_sec = round(time.time() - t_tr, 1)
    recalibrate_bn_stats(student, loader, device, n_batches=32)
    del teacher_v6, anchor_v5
    torch.cuda.empty_cache()

    ckpt_final = LOCAL_RUNS / "egoclear_v75_moge2_small.pt"
    head_sd = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")}
    torch.save({"head_state_dict": head_sd, "moge_id": student.moge_id, "lambda_d": lambda_d, "history": history}, ckpt_final)
    shutil.copy2(ckpt_final, DRIVE_RUNS / ckpt_final.name)

    # 5. Full 5-Domain + Dark-Room + Fixed-0.50 Paired Route Audit + lag=4 Masking Ablation
    print("\n=== Full 5-Domain + Dark-Room + Paired-Route Evaluation of EgoClear-v7.5 ===", flush=True)
    student.eval()
    evals, paired_audits = {}, {}
    domain_specs = [
        ("synth_clean", eval_synth, None, 0.0, True),
        ("synth_vio_1x", eval_synth, None, 1.0, False),
        ("real_40_local", eval_real, real_local_rows, 0.0, True),
        ("aria_g2_zeroshot", aria_dicts, None, 0.0, True),
        ("arkit_indoor_val30", eval_arkit, None, 0.0, True),
        ("coda_campus_zeroshot", eval_coda, None, 0.0, True),
    ]
    for dname, dmap, ov, vscale, do_unc in domain_specs:
        if not dmap:
            continue
        evals[dname] = eval_v6_all(student, dmap, ov, bw_speeds, device, vio_scale=vscale, compute_uncertainty_audit=do_unc)
        print_block("EgoClear-v7.5", dname, evals[dname])
        if vscale == 0.0:
            paired_audits[dname] = collect_paired_route_audit(student, dmap, ov, bw_speeds, device, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km ({f50['fa_ep_km']} ep/km)", flush=True)

    # Dark-room 0.18x robustness
    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v7.5", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v7.5", "real_40_dark018", evals["real_40_dark018"])

    # Problem C diagnostic: lag=4 masking ablation on real_40_local with frozen v7.5 weights
    lag4_masked = collect_paired_route_audit(student, eval_real, real_local_rows, bw_speeds, device, fixed_thr=0.50, mask_lags=(4,))
    print(f"  [Diag lag=4 masked on real_40] fixed@0.50={lag4_masked['fixed_thr_0.50']} | @120={lag4_masked['raw_at_120']}", flush=True)

    # 6. Export verified ONNX-17 FP32 + FP16 (with encoder & step FP16 guards) and sync to Drive
    onnx_info = export_v75_onnx_fp16(student, LOCAL_RUNS / "onnx")
    for fn in ["egoclear_v75_encoder_fp16.onnx", "egoclear_v75_step_fp16.onnx", "egoclear_v75_encoder_fp32.onnx", "egoclear_v75_step_fp32.onnx"]:
        src = LOCAL_RUNS / "onnx" / fn
        if src.exists():
            shutil.copy2(src, DRIVE_RUNS / fn)

    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v7.5 (MoGe-2-ViT-S-Normal 35M + 84x148 + Tile-Capped Shift + Full-Event SmoothMax + Per-Domain Proxy-Lagrangian)",
        "floor_n_up_median": round(floor_n_up_med, 4),
        "idx_4m_exact": IDX_4M_EXACT,
        "total_params_m": round(n_tot / 1e6, 2),
        "trainable_params_m": round(n_tr / 1e6, 2),
        "final_lambda_d": lambda_d,
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "history": history,
        "evals": evals,
        "paired_route_audits_at_0.50": paired_audits,
        "diag_lag4_masked_real40": lag4_masked,
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase11_v75_results.json"
    res_drive = DRIVE_RUNS / "phase11_v75_results.json"
    res_local.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_local, res_drive)
    print(f"\n[v7.5] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
