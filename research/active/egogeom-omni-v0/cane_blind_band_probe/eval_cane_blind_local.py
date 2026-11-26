"""Fast Bit-Exact 5-Domain Local Evaluator for Cane-Blind Mid/Upper-Body Height-Band Probe.

Reads:
  1. Local frozen `v41_stream/` for `aria_gen2` (19), `arkit_val30_day` (30),
     `arkit_val30_dark018` (30), `coda_campus` (24)
  2. Local voxel height-band GT `aria_height_band_gt.npz` (exact 5 cm voxel bands)
  3. Local frozen `zeroshot_eval/` packs for ARKit & CODa `eval_rows` + `gt_bev_static`
  4. Colab-generated `sanpo_real_zs40_1d_bundle.npz` + `sanpo_real_zs40_meta.json` (40 clips)

Uses precomputed episode structures + Numba-JITted `score_episode_fast` verified bit-exact
against `eval_closed_loop.score_episode`, finishing all 5 domains + 4-domain cross-fits in ~25s!
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
STREAM_DIR = ART / "math_theory/v41_stream"
ZS_DIR = ART / "zeroshot_eval"

sys.path[:0] = [str(HERE), str(ROUTE), str(ROUTE.parent / "egoclear-v0")]

from eval_closed_loop import H as EVAL_H, MIN_EVENT_FRAMES_V2, REFRACTORY_S, make_episodes, score_episode, weak_mask  # noqa: E402
from run_cane_blind_colab import (  # noqa: E402
    ALL_ARMS,
    AVOID,
    BW_SPEEDS_PROXY,
    CLEAR,
    CLEAR_WARN,
    FA_BUDGET,
    FALSE,
    FFRAME,
    FINE_Q_GRID,
    KM,
    LATE,
    ON_CLEAR,
    Q_GRID,
    UNAVOID,
    VALID,
    ALERT,
    analyze_clip_fa_anatomy,
    assign_source_folds,
    build_bev_band_eval_rows,
    build_low_only_bev_mask,
    frame_level_metrics,
    merge_anatomy,
    run_all_readouts_on_clip,
    summarize_counts,
)


# ---------------------------------------------------------------------------
# Bit-Exact Numba Episode Scorer
# ---------------------------------------------------------------------------

@njit
def _score_ep_numba(
    alerts_raw: np.ndarray,
    valid: np.ndarray,
    haz: np.ndarray,
    dyn: np.ndarray,
    weak: np.ndarray,
    ok_full: np.ndarray,
    ds: np.ndarray,
    refr: float,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_has_ok: np.ndarray,
    clear: np.ndarray,
    cum_m: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    n = len(alerts_raw)
    alerts = np.empty(n, dtype=np.bool_)
    for i in range(n):
        alerts[i] = bool(alerts_raw[i]) and valid[i]

    late = 0
    avoid = 0
    unavoid = 0
    for s_idx in range(len(seg_starts)):
        if not seg_has_ok[s_idx]:
            unavoid += 1
            continue
        avoid += 1
        s0 = seg_starts[s_idx]
        s1 = seg_ends[s_idx]
        hit = False
        for k in range(s0, s1):
            if alerts[k] and ok_full[k]:
                hit = True
                break
        if not hit:
            late += 1

    # FA episodes (exact parity with eval_closed_loop.score_episode)
    fa_ep = 0
    fa_fr = 0
    n_idx = 0
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        if alerts[i]:
            idx[n_idx] = i
            n_idx += 1
            if (not haz[i]) and (not dyn[i]) and (not weak[i]):
                fa_fr += 1

    start = 0
    while start < n_idx:
        end = start
        while end + 1 < n_idx:
            gap = 0.0
            for t in range(idx[end] + 1, idx[end + 1] + 1):
                gap += ds[t]
            if gap <= refr:
                end += 1
            else:
                break
        any_bad = False
        for u in range(start, end + 1):
            fr = idx[u]
            if haz[fr] or dyn[fr] or weak[fr]:
                any_bad = True
                break
        if not any_bad:
            fa_ep += 1
        start = end + 1

    # Repeated clear warnings
    inside = False
    charged = False
    last_m = 0.0
    clear_warn = 0
    for i in range(n):
        if not clear[i]:
            inside = False
            continue
        if not inside:
            inside = True
            charged = False
        if alerts[i] and ((not charged) or (cum_m[i] - last_m > refr)):
            clear_warn += 1
            last_m = cum_m[i]
            charged = True

    return late, avoid, unavoid, fa_ep, fa_fr, clear_warn


class FastClipEval:
    def __init__(self, domain: str, name: str, device: str, source: str, fold: int, rows: np.ndarray, arms: dict):
        self.domain = domain
        self.name = name
        self.device = device
        self.source = source
        self.fold = fold
        self.rows = rows
        self.arms = arms
        self.episodes = make_episodes(name, rows, BW_SPEEDS_PROXY, "blind")
        valid = np.asarray(self.episodes[0]["valid"], dtype=np.bool_)
        self.valid = valid
        D = rows[:, 0]
        self.haz = valid & (D <= EVAL_H)
        self.dyn = rows[:, 2] > 0.5
        self.ds = rows[:, 3].astype(np.float64)
        self.weak = np.asarray(weak_mask(self.episodes[0]), dtype=np.bool_)
        self.clear = valid & (D > 4.0) & (~self.dyn)
        self.cum_m = np.cumsum(np.where(valid, self.ds, 0.0), dtype=np.float64)
        self.km = float(self.ds[valid].sum() / 1000.0)
        self.n_valid_int = int(valid.sum())
        self.n_clear_int = int(self.clear.sum())

        # Precompute hazard segments once (exact parity with score_episode)
        starts, ends = [], []
        i, n = 0, len(D)
        while i < n:
            if self.haz[i]:
                j = i
                while j + 1 < n and (self.haz[j + 1] or (j + 2 < n and self.haz[j + 2] and valid[j + 1])):
                    j += 1
                if int(self.haz[i : j + 1].sum()) >= MIN_EVENT_FRAMES_V2:
                    starts.append(i)
                    ends.append(j + 1)
                i = j + 1
            else:
                i += 1
        self.seg_starts = np.asarray(starts, dtype=np.int64)
        self.seg_ends = np.asarray(ends, dtype=np.int64)
        self.ep_fast = []
        for ep in self.episodes:
            ok_full = (D >= ep["dreq"])
            seg_has_ok = np.array([bool(ok_full[s:e].any()) for s, e in zip(starts, ends)], dtype=np.bool_)
            self.ep_fast.append((ok_full, float(ep["v"] * REFRACTORY_S), seg_has_ok))
        self.cache: dict[bytes, np.ndarray] = {}

    def count(self, alerts: np.ndarray) -> np.ndarray:
        a = np.asarray(alerts, dtype=np.bool_)
        key = np.packbits(a).tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        tot_late = tot_avoid = tot_unavoid = tot_fa_ep = tot_fa_fr = tot_cw = 0
        for ok_full, refr, seg_has_ok in self.ep_fast:
            l, av, un, fe, ff, cw = _score_ep_numba(
                a, self.valid, self.haz, self.dyn, self.weak, ok_full,
                self.ds, refr, self.seg_starts, self.seg_ends, seg_has_ok,
                self.clear, self.cum_m,
            )
            tot_late += l
            tot_avoid += av
            tot_unavoid += un
            tot_fa_ep += fe
            tot_fa_fr += ff
            tot_cw += cw

        a_val = a & self.valid
        res = np.array([
            float(tot_late),
            float(tot_avoid),
            float(tot_unavoid),
            float(tot_fa_ep),
            float(tot_fa_fr),
            6.0 * self.km,
            float(tot_cw),
            float(a_val.sum()),
            float(self.n_valid_int),
            float((a_val & self.clear).sum()),
            float(self.n_clear_int),
        ], dtype=np.float64)
        self.cache[key] = res
        return res


def verify_fast_scorer_parity(clips: list[FastClipEval]) -> None:
    rng = np.random.default_rng(20261125)
    sample = clips[: min(10, len(clips))]
    checked = 0
    for c in sample:
        n = len(c.rows)
        for p in np.linspace(0.02, 0.80, 20):
            mask = rng.random(n) < p
            fast = c.count(mask)
            ref_rows = [score_episode(ep, mask) for ep in c.episodes]
            ref_frozen = np.sum(ref_rows, axis=0)
            np.testing.assert_allclose(fast[:6], ref_frozen, rtol=0, atol=1e-12, err_msg=f"Parity fail {c.domain}/{c.name}")
            checked += 1
    print(f"[parity] FastClipEval bit-exact against score_episode across {checked} random tests!", flush=True)


def exact_frontier(clip_objs: list[FastClipEval], arm: str, field: str = "confirmed") -> dict:
    events: dict[float, np.ndarray] = {}
    base = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in clip_objs), np.zeros(11, dtype=np.float64))
    for c in clip_objs:
        vals = c.arms[arm][field]
        prev = c.count(np.zeros(len(vals), dtype=bool))
        for cut in np.unique(vals)[::-1]:
            nxt = c.count(vals >= cut)
            diff = nxt - prev
            if np.any(diff != 0):
                events[float(cut)] = events.get(float(cut), np.zeros(11, dtype=np.float64)) + diff
            prev = nxt

    rows = []
    cur = base.copy()

    def _pt(thr: float | None, arr: np.ndarray) -> dict:
        s = summarize_counts(arr)
        s["threshold"] = thr
        return s

    rows.append(_pt(None, cur))
    for cut in sorted(events, reverse=True):
        cur = cur + events[cut]
        rows.append(_pt(cut, cur))

    km = float(base[KM])
    avoid = int(base[AVOID])
    budget = 5.0 * km
    ok_clear = [r for r in rows if r["repeated_clear_warnings"] <= budget + 1e-9]
    ok_fa = [r for r in rows if r["fa_episodes"] <= budget + 1e-9]
    best_clear = max(ok_clear, key=lambda r: (r["timely"], -r["repeated_clear_warnings"]))
    best_fa = max(ok_fa, key=lambda r: (r["timely"], -r["fa_episodes"]))
    return {
        "avoidable": avoid,
        "walker_km": round(km, 4),
        "known_clear_frames": int(base[CLEAR]),
        "best_at_5_clear_km": best_clear,
        "best_at_5_orig_fa_km": best_fa,
    }


def fit_threshold(
    clip_objs: list[FastClipEval],
    arm: str,
    field: str = "confirmed",
    robust: bool = True,
    device_only: bool = False,
    q_grid: tuple[float, ...] | np.ndarray = Q_GRID,
):
    thresholds = [-np.inf, np.inf]
    if not device_only and clip_objs:
        train_scores = np.concatenate([c.arms[arm][field] for c in clip_objs])
        thresholds += list(np.unique(np.quantile(train_scores, q_grid, method="higher")))
    candidates = []
    for thr in thresholds:
        counts = sum((c.count(c.arms[arm][field] >= thr) for c in clip_objs), np.zeros(11, dtype=np.float64))
        if counts[FALSE] > FA_BUDGET * counts[KM]:
            continue
        if robust and counts[CLEAR_WARN] > FA_BUDGET * counts[KM]:
            continue
        candidates.append(((counts[LATE], counts[CLEAR_WARN], counts[ON_CLEAR], counts[ALERT], -thr), thr, counts))
    _, threshold, best = min(candidates, key=lambda row: row[0])
    return threshold, summarize_counts(best)


def run_2fold_crossfit(clip_objs: list[FastClipEval], arm: str, field: str = "confirmed") -> dict:
    shared_rows, dev_rows, shared_fine_rows, dev_fine_rows = [], [], [], []
    for f in (0, 1):
        train = [c for c in clip_objs if c.fold != f]
        test = [c for c in clip_objs if c.fold == f]
        if not train or not test:
            continue
        assert {c.source for c in train}.isdisjoint({c.source for c in test})
        t_shared, _ = fit_threshold(train, arm, field=field, robust=True, q_grid=Q_GRID)
        t_shared_f, _ = fit_threshold(train, arm, field=field, robust=True, q_grid=FINE_Q_GRID)
        devs = sorted({c.device for c in clip_objs})
        t_dev, t_dev_f = {}, {}
        for d in devs:
            tr_d = [c for c in train if c.device == d]
            t_dev[d] = fit_threshold(tr_d, arm, field=field, robust=True, q_grid=Q_GRID)[0] if tr_d else np.inf
            t_dev_f[d] = fit_threshold(tr_d, arm, field=field, robust=True, q_grid=FINE_Q_GRID)[0] if tr_d else np.inf
        for c in test:
            vals = c.arms[arm][field]
            shared_rows.append((c.domain, c.count(vals >= t_shared)))
            dev_rows.append((c.domain, c.count(vals >= t_dev[c.device])))
            shared_fine_rows.append((c.domain, c.count(vals >= t_shared_f)))
            dev_fine_rows.append((c.domain, c.count(vals >= t_dev_f[c.device])))

    def _agg(items):
        pooled = sum((x for _, x in items), np.zeros(11, dtype=np.float64))
        doms = sorted({d for d, _ in items})
        return {
            "pooled": summarize_counts(pooled),
            "domains": {d: summarize_counts(sum((x for dd, x in items if dd == d), np.zeros(11, dtype=np.float64))) for d in doms},
        }

    return {
        "shared_crossfit": _agg(shared_rows),
        "device_crossfit": _agg(dev_rows),
        "shared_crossfit_fine": _agg(shared_fine_rows),
        "device_crossfit_fine": _agg(dev_fine_rows),
    }


def fixed_dist_diag(clip_objs: list[FastClipEval], arm: str) -> dict:
    c2 = sum((c.count(c.arms[arm]["dist"] <= 2.0) for c in clip_objs), np.zeros(11, dtype=np.float64))
    c4 = sum((c.count(c.arms[arm]["dist"] <= 4.0) for c in clip_objs), np.zeros(11, dtype=np.float64))
    return {"dist_le_2m": summarize_counts(c2), "dist_le_4m": summarize_counts(c4)}


# ---------------------------------------------------------------------------
# Main Local Loader + Evaluator
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    aria_gt_bundle = np.load(PROBE_DIR / "aria_height_band_gt.npz", allow_pickle=False)
    sanpo_bundle = np.load(PROBE_DIR / "sanpo_real_zs40_1d_bundle.npz", allow_pickle=False)
    sanpo_meta = json.loads((PROBE_DIR / "sanpo_real_zs40_meta.json").read_text(encoding="utf-8"))

    suite_clips: dict[str, list[dict]] = {}
    suite_meta: dict[str, dict] = {}
    anatomy_by_suite_band: dict[str, dict] = {}
    max_rhyb_diff = 0.0

    local_suites = [
        ("aria_gen2", "aria_g2_20clips", 4.0),
        ("arkit_val30_day", "arkit_val30_shard00", 2.0),
        ("arkit_val30_dark018", "arkit_val30_shard00", 2.0),
        ("coda_campus", "coda_zeroshot_shard000", 4.0),
    ]

    for sname, folder, h_thr in local_suites:
        ts = time.time()
        src_paths = sorted((ZS_DIR / folder).rglob("*.npz"))
        clips_list = []
        cam_heights = defaultdict(list)
        low_only_cnt = 0
        valid_tot = 0
        anatomy_recs = defaultdict(list)

        for p in src_paths:
            with np.load(p, allow_pickle=False) as d_src:
                eval_rows = d_src["eval_rows"].astype(np.float64)
                gt_bev_static = d_src["gt_bev_static"] if "gt_bev_static" in d_src else None
                cam_name = str(d_src["cam"]) if "cam" in d_src else ("rover_flir" if sname == "coda_campus" else "unknown")

            with np.load(STREAM_DIR / sname / p.name, allow_pickle=False) as d_obs:
                dep_s4 = d_obs["depth_s4"]
                dh_s4 = d_obs["dh_s4"]
                nrm_s4 = d_obs["nrm_s4"]
                ls_s4 = d_obs["ls_s4"]
                Ks_arr = d_obs["K"]
                poses_R = d_obs["poses_R"]
                poses_t = d_obs["poses_t"]
                strict_fwd = d_obs["strict_fwd"]
                archived_rhyb = d_obs["r_hyb_strict"]

            T = len(eval_rows)
            v_sub, u_sub = np.indices(dep_s4.shape[1:], dtype=np.float32)
            u_sub = 4.0 * u_sub + 2.0
            v_sub = 4.0 * v_sub + 2.0
            for idx_h in range(0, T, 10):
                z_i = dep_s4[idx_h]
                K_i = Ks_arr[idx_h]
                xyz = np.stack(((u_sub - K_i[0, 2]) * z_i / K_i[0, 0], (v_sub - K_i[1, 2]) * z_i / K_i[1, 1], z_i), axis=-1) @ poses_R[idx_h].T
                fl = (xyz[..., 2] < 0) & (np.abs(nrm_s4[idx_h, 1]) > 0.65) & (z_i > 0.3)
                if fl.sum() >= 20:
                    cam_heights[cam_name].append(float(-np.percentile(xyz[..., 2][fl], 15)))

            arms = run_all_readouts_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, strict_fwd)
            if sname == "aria_gen2":
                diff = float(np.max(np.abs(arms["bev_full_ground"]["raw"] - archived_rhyb)))
                max_rhyb_diff = max(max_rhyb_diff, diff)

            gt_bands: dict[str, np.ndarray] = {}
            if sname == "aria_gen2":
                for st in ("centered_full", "strict_past"):
                    for b in ("full_ground", "low_only_ground", "midup_ground", "head_ground", "head_anchored_85", "head_anchored_70"):
                        gt_bands[f"{st}__{b}"] = aria_gt_bundle[f"{p.stem}__{st}__{b}"].astype(np.float64)
                val_cf = (np.isfinite(eval_rows[:, 0]) | np.isinf(eval_rows[:, 0])) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
                valid_tot += int(val_cf.sum())
                cf_full = gt_bands["centered_full__full_ground"][:, 0]
                cf_midup = gt_bands["centered_full__midup_ground"][:, 0]
                low_only_cnt += int((val_cf & (cf_full <= 4.0) & (cf_midup > 4.0)).sum())
                for b_ana in ("centered_full__full_ground", "centered_full__midup_ground", "centered_full__head_anchored_85", "strict_past__midup_ground", "strict_past__head_anchored_85"):
                    r_b = gt_bands[b_ana]
                    v_b = (np.isfinite(r_b[:, 0]) | np.isinf(r_b[:, 0])) & ~np.isnan(r_b[:, 1]) & (r_b[:, 1] >= 0.6)
                    clr_b = v_b & (r_b[:, 0] > 4.0) & (r_b[:, 2] <= 0.5)
                    anatomy_recs[b_ana].append(
                        analyze_clip_fa_anatomy(
                            dep_s4, dh_s4, nrm_s4, Ks_arr, poses_R, strict_fwd,
                            arms["bev_head_anchored_85"]["dist"],
                            arms["bev_head_anchored_85_pg"]["dist"],
                            clr_b,
                        )
                    )
            else:
                gt_bands["orig_full"] = eval_rows
                if gt_bev_static is not None:
                    gt_bands["bev_full"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
                    gt_bands["bev_low_only"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0,))
                    gt_bands["bev_midup"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))
                    gt_bands["bev_head"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (2,))
                    val_b = (np.isfinite(eval_rows[:, 0]) | np.isinf(eval_rows[:, 0])) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
                    valid_tot += int(val_b.sum())
                    low_only_cnt += int((val_b & build_low_only_bev_mask(gt_bev_static)).sum())
                    for b_ana in ("bev_full", "bev_midup"):
                        r_b = gt_bands[b_ana]
                        clr_b = val_b & (r_b[:, 0] > 4.0) & (r_b[:, 2] <= 0.5)
                        anatomy_recs[b_ana].append(
                            analyze_clip_fa_anatomy(
                                dep_s4, dh_s4, nrm_s4, Ks_arr, poses_R, strict_fwd,
                                arms["bev_head_anchored_85"]["dist"],
                                arms["bev_head_anchored_85_pg"]["dist"],
                                clr_b,
                            )
                        )

            clips_list.append({
                "domain": sname,
                "name": p.stem,
                "cam": cam_name,
                "arms": arms,
                "gt_bands": gt_bands,
            })

        assign_source_folds(clips_list)
        suite_clips[sname] = clips_list
        suite_meta[sname] = {
            "n_clips": len(clips_list),
            "hazard_thr_m": h_thr,
            "valid_frames": valid_tot,
            "low_only_frames": low_only_cnt,
            "est_cam_height_median_m": {k: round(float(np.median(v)), 3) for k, v in cam_heights.items() if v},
            "elapsed_s": round(time.time() - ts, 2),
        }
        anatomy_by_suite_band[sname] = {b: merge_anatomy(recs) for b, recs in anatomy_recs.items()}
        print(f"[load] {sname}: {len(clips_list)} clips readouts in {suite_meta[sname]['elapsed_s']}s", flush=True)

    print(f"[parity] Aria bev_full_ground vs archived r_hyb_strict max_abs_diff={max_rhyb_diff:.2e}", flush=True)
    if max_rhyb_diff > 1e-3:
        raise RuntimeError(f"Aria r_hyb_strict parity failed: max_diff={max_rhyb_diff}")

    # Load SANPO-Real ZS40 from Colab bundle
    sanpo_clips = []
    for cmeta in sanpo_meta["clips"]:
        stem = cmeta["name"]
        arms = {
            arm: {
                field: sanpo_bundle[f"{stem}__arm__{arm}__{field}"]
                for field in ("dist", "raw", "confirmed")
            }
            for arm in ALL_ARMS
        }
        gt_bands = {}
        for gk in ("orig_full", "bev_full", "bev_low_only", "bev_midup", "bev_head"):
            key = f"{stem}__gt__{gk}"
            if key in sanpo_bundle:
                gt_bands[gk] = sanpo_bundle[key].astype(np.float64)
        sanpo_clips.append({
            "domain": "sanpo_real_zs40",
            "name": stem,
            "cam": cmeta["cam"],
            "arms": arms,
            "gt_bands": gt_bands,
        })
    assign_source_folds(sanpo_clips)
    suite_clips["sanpo_real_zs40"] = sanpo_clips
    suite_meta["sanpo_real_zs40"] = {
        "n_clips": sanpo_meta["n_clips"],
        "hazard_thr_m": 4.0,
        "valid_frames": sanpo_meta["valid_frames"],
        "low_only_frames": sanpo_meta["low_only_frames"],
        "cam_counts": sanpo_meta["cam_counts"],
        "est_cam_height_median_m": sanpo_meta["est_cam_height_median_m"],
        "colab_gpu": sanpo_meta["gpu"],
        "colab_elapsed_s": sanpo_meta["elapsed_s"],
    }
    anatomy_by_suite_band["sanpo_real_zs40"] = sanpo_meta["anatomy_by_band"]
    print(f"[load] sanpo_real_zs40: {len(sanpo_clips)} clips from Colab bundle", flush=True)

    def make_eval_objs(suite_list: list[str], band_map: dict[str, str], cam_filter: str | None = None) -> list[FastClipEval]:
        objs = []
        for s in suite_list:
            bkey = band_map[s]
            for c in suite_clips[s]:
                if cam_filter is not None and c["cam"] != cam_filter:
                    continue
                if bkey not in c["gt_bands"]:
                    continue
                objs.append(FastClipEval(c["domain"], c["name"], c["device"], c["source"], c["fold"], c["gt_bands"][bkey], c["arms"]))
        return objs

    # Verify Numba bit-exact parity across all 5 domains
    parity_sample = []
    for s, bk in [
        ("aria_gen2", "centered_full__full_ground"),
        ("arkit_val30_day", "orig_full"),
        ("coda_campus", "orig_full"),
        ("sanpo_real_zs40", "orig_full"),
    ]:
        parity_sample.extend(make_eval_objs([s], {s: bk})[:3])
    verify_fast_scorer_parity(parity_sample)

    # 1) Denominators
    print("[eval] Computing denominators...", flush=True)
    denominators = {}
    aria_bands_all = [f"{st}__{b}" for st in ("centered_full", "strict_past") for b in ("full_ground", "low_only_ground", "midup_ground", "head_ground", "head_anchored_85", "head_anchored_70")]
    denominators["aria_gen2"] = {}
    for b in aria_bands_all:
        objs = make_eval_objs(["aria_gen2"], {"aria_gen2": b})
        zero = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in objs), np.zeros(11, dtype=np.float64))
        fm = frame_level_metrics(objs, "bev_full_ground", 4.0)
        denominators["aria_gen2"][b] = {
            "n_valid": fm["n_valid"],
            "n_pos_4m": fm["n_pos_4m"],
            "n_clear": int(zero[CLEAR]),
            "avoidable_6w": int(zero[AVOID]),
            "unavoidable_6w": int(zero[UNAVOID]),
            "walker_km": round(float(zero[KM]), 4),
        }
    for st in ("centered_full", "strict_past"):
        excl = 0
        for c in suite_clips["aria_gen2"]:
            rf = c["gt_bands"][f"{st}__full_ground"]
            rm = c["gt_bands"][f"{st}__midup_ground"]
            val = (np.isfinite(rf[:, 0]) | np.isinf(rf[:, 0])) & ~np.isnan(rf[:, 1]) & (rf[:, 1] >= 0.6)
            excl += int((val & (rf[:, 0] <= 4.0) & (rm[:, 0] > 4.0)).sum())
        denominators["aria_gen2"][f"{st}__low_only_exclusive_frames"] = excl

    for sname, h_thr in [("arkit_val30_day", 2.0), ("coda_campus", 4.0), ("sanpo_real_zs40", 4.0)]:
        denominators[sname] = {}
        for b in ("orig_full", "bev_full", "bev_low_only", "bev_midup", "bev_head"):
            objs = make_eval_objs([sname], {sname: b})
            zero = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in objs), np.zeros(11, dtype=np.float64))
            fm = frame_level_metrics(objs, "bev_full_ground", h_thr)
            denominators[sname][b] = {
                "n_valid": fm["n_valid"],
                "n_pos_eval_thr": fm["n_pos_eval_thr"],
                "n_pos_4m": fm["n_pos_4m"],
                "n_clear": int(zero[CLEAR]),
                "avoidable_6w": int(zero[AVOID]),
                "unavoidable_6w": int(zero[UNAVOID]),
                "walker_km": round(float(zero[KM]), 4),
            }
        orig_pos = sum(int(((c.rows[:, 0] <= 4.0) & c.valid).sum()) for c in make_eval_objs([sname], {sname: "orig_full"}))
        bev_pos = sum(int(((c.rows[:, 0] <= 4.0) & c.valid).sum()) for c in make_eval_objs([sname], {sname: "bev_full"}))
        both_pos = 0
        for co, cb in zip(make_eval_objs([sname], {sname: "orig_full"}), make_eval_objs([sname], {sname: "bev_full"})):
            both_pos += int((co.valid & (co.rows[:, 0] <= 4.0) & (cb.rows[:, 0] <= 4.0)).sum())
        denominators[sname]["orig_vs_bev_concordance"] = {
            "orig_pos_4m": orig_pos,
            "bev_pos_4m": bev_pos,
            "both_pos_4m": both_pos,
            "low_only_exclusive_frames": suite_meta[sname]["low_only_frames"],
        }

    # 2) Per-domain evaluation
    def eval_cohort(objs: list[FastClipEval], h_thr: float) -> dict:
        res = {}
        for arm in ALL_ARMS:
            res[arm] = {
                "frame": frame_level_metrics(objs, arm, h_thr),
                "frontier_confirmed": exact_frontier(objs, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag(objs, arm),
            }
        return res

    per_domain_results = {}
    print("[eval] Scoring aria_gen2...", flush=True)
    per_domain_results["aria_gen2"] = {}
    for b in ("centered_full__full_ground", "centered_full__midup_ground", "centered_full__head_anchored_85",
              "strict_past__full_ground", "strict_past__midup_ground", "strict_past__head_anchored_85"):
        per_domain_results["aria_gen2"][b] = eval_cohort(make_eval_objs(["aria_gen2"], {"aria_gen2": b}), 4.0)

    for sname, h_thr in [("arkit_val30_day", 2.0), ("arkit_val30_dark018", 2.0), ("coda_campus", 4.0)]:
        print(f"[eval] Scoring {sname}...", flush=True)
        per_domain_results[sname] = {}
        for b in ("orig_full", "bev_full", "bev_midup"):
            per_domain_results[sname][b] = eval_cohort(make_eval_objs([sname], {sname: b}), h_thr)

    print("[eval] Scoring sanpo_real_zs40 (all / camera_head / camera_chest + 2-fold crossfit)...", flush=True)
    per_domain_results["sanpo_real_zs40"] = {}
    for cam_sub in ("all", "camera_head", "camera_chest"):
        cf_arg = None if cam_sub == "all" else cam_sub
        per_domain_results["sanpo_real_zs40"][cam_sub] = {}
        for b in ("orig_full", "bev_full", "bev_midup"):
            objs = make_eval_objs(["sanpo_real_zs40"], {"sanpo_real_zs40": b}, cam_filter=cf_arg)
            if not objs:
                continue
            cohort_res = eval_cohort(objs, 4.0)
            if cam_sub == "all":
                for arm in ALL_ARMS:
                    cohort_res[arm]["crossfit_2fold"] = run_2fold_crossfit(objs, arm, "confirmed")
            per_domain_results["sanpo_real_zs40"][cam_sub][b] = cohort_res

    # 3) 4-Domain 38-source pooled + crossfit
    pooled_4dom_configs = {
        "full_ground_centered": {
            "aria_gen2": "centered_full__full_ground",
            "arkit_val30_day": "orig_full",
            "arkit_val30_dark018": "orig_full",
            "coda_campus": "orig_full",
        },
        "full_matched_bev": {
            "aria_gen2": "centered_full__full_ground",
            "arkit_val30_day": "bev_full",
            "arkit_val30_dark018": "bev_full",
            "coda_campus": "bev_full",
        },
        "midup_ground_matched": {
            "aria_gen2": "centered_full__midup_ground",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
        "midup_strict_past_aria": {
            "aria_gen2": "strict_past__midup_ground",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
        "head_anchored85_aria_midup_others": {
            "aria_gen2": "centered_full__head_anchored_85",
            "arkit_val30_day": "bev_midup",
            "arkit_val30_dark018": "bev_midup",
            "coda_campus": "bev_midup",
        },
    }
    four_domain_results = {}
    doms4 = ["aria_gen2", "arkit_val30_day", "arkit_val30_dark018", "coda_campus"]
    for cfg_name, bmap in pooled_4dom_configs.items():
        print(f"[eval] Scoring 4-domain 38-source config: {cfg_name}...", flush=True)
        objs4 = make_eval_objs(doms4, bmap)
        four_domain_results[cfg_name] = {}
        for arm in ALL_ARMS:
            four_domain_results[cfg_name][arm] = {
                "frontier_confirmed": exact_frontier(objs4, arm, "confirmed"),
                "crossfit_2fold": run_2fold_crossfit(objs4, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag(objs4, arm),
            }

    # Baseline sanity checks
    aria_base = per_domain_results["aria_gen2"]["centered_full__full_ground"]["bev_full_ground"]
    aria_auroc = aria_base["frame"]["auroc_raw"]
    print(
        f"\n[sanity] Aria centered_full/full_ground/bev_full_ground: "
        f"n_valid={aria_base['frame']['n_valid']}, n_pos={aria_base['frame']['n_pos_4m']}, "
        f"AUROC_raw={aria_auroc}, frontier_5clear={aria_base['frontier_confirmed']['best_at_5_clear_km']['timely']}/{aria_base['frontier_confirmed']['avoidable']}",
        flush=True,
    )
    if aria_auroc is None or abs(aria_auroc - 0.7060) > 0.010:
        raise RuntimeError(f"Aria baseline AUROC {aria_auroc} diverged >0.010 from archived 0.7060!")

    m6_base = four_domain_results["full_ground_centered"]["m6_full_ground"]
    m6_cf = m6_base["crossfit_2fold"]["shared_crossfit"]["pooled"]["timely"]
    m6_fr = m6_base["frontier_confirmed"]["best_at_5_clear_km"]["timely"]
    print(f"[sanity] 4-dom full_ground_centered/m6_full_ground: crossfit={m6_cf}/365, frontier={m6_fr}/365 (V5 ref: 1/365, 47/365)", flush=True)

    report = {
        "meta": {
            "checkpoint": sanpo_meta["checkpoint"],
            "total_local_elapsed_s": round(time.time() - t0, 2),
            "aria_rhyb_max_diff": max_rhyb_diff,
            "suites": suite_meta,
        },
        "denominators": denominators,
        "per_domain": per_domain_results,
        "four_domain_38source": four_domain_results,
        "residual_clear_fa_anatomy": anatomy_by_suite_band,
    }

    out_json = PROBE_DIR / "cane_blind_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Saved {out_json} ({out_json.stat().st_size / 1e3:.1f} KB) in {report['meta']['total_local_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
