"""Replay the retained indoor ToF-DAV2 depth arm with an RGB-only ground mask on ADT development.

Run from the checkout root with the existing GPU Python:
  python -B research/active/hardware-bringup/host/adt_ground_alert_replay_20261118.py run
  python -B research/active/hardware-bringup/host/adt_ground_alert_replay_20261118.py summarize

Predictions stream in original time order per recording, including frames without alerts, to
preserve ToF fit fallback. Scoring uses the original evaluation coordinate system and standing
frame selection. Only aggregate results belong in version control.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from adt_wall_attribution_20261118 import BASE, OUT as WALL, ROOT, read, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalkit"))
import core
from methods import Methods


OUT = ROOT / "artifacts.local/work/ground-alert-20261118/adt"
BASELINE = "tof_dav2s"


class OriginalIndoorDepth(Methods):
    """Use the original 518-px disparity and ToF fitter without loading unused UniDepth arms."""
    def __init__(self):
        import sanpo_scale_without_tof_dev_20261117 as source
        from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
        self.P, self.S = source, source.V1.S
        self.fit_quality, self.robust_affine = fit_quality, robust_affine
        self.zone_rays = source.V1.X.zone_rays()
        self.S.UPSTREAM = source.DAV2_SRC
        self.rel, _ = self.S.load_dav2_small(
            ROOT / "artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth")
        self._cache = None
        self.last_fit = {}


def readout(pred, rays, projected_area, axes, height, mask=None):
    """Identical raw masks, opening, projected area and cutoff to core.band_alert."""
    h, w = pred.shape
    p = pred.reshape(-1)
    ok = np.isfinite(p) & (p > .05)
    z = np.where(ok, p, 0)
    above, _, planar = core._corridor(rays * z[:, None], ok, axes, height)
    area = z**2 / projected_area
    old, new = {}, {}
    low_support = None
    for band, (lo, hi) in core.BANDS.items():
        raw = planar & (above >= lo) & (above < hi)
        opened = core._open(raw, h, w)
        old[band] = bool(area[opened].sum() >= core.MIN_AREA)
        if band == "LOW" and old[band]:
            low_support = opened.reshape(h,w)
        new[band] = (old[band] and bool(area[core._open(raw & ~mask.reshape(-1), h, w)].sum() >= core.MIN_AREA)
                     if mask is not None else old[band])
    return old, new, low_support


def replay(seq, rows, model, masker):
    src = BASE / "prep"
    fx, fy, cx, cy = read(src / f"{seq}.json")["K"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    floor = read(src / f"{seq}.floor.json")["floor_y"]
    with np.load(src / f"{seq}.npz") as d:
        ts, down, cam_y = d["t_ns"], d["down"], d["cam_y"]
        rgb, depth = d["rgb"], d["depth"]
    times = (ts-ts[0])/1e9
    expected = [i for i in range(len(times)) if float(cam_y[i]) - floor >= 1.3]
    assert len(expected) == len(rows)
    rays = core.rays(k, *depth.shape[1:])
    projected_area = fx * fy * np.linalg.norm(rays, axis=1)
    records = []
    for ordinal, (i, original) in enumerate(zip(expected, rows)):
        t = float(times[i])
        assert abs(t-original["t"]) < 1e-8
        z = depth[i].astype(np.float32) / 1000.
        z[z <= 0] = np.nan
        dw = down[i].astype(float)
        height = float(cam_y[i] - floor)
        # This common frame supplies reference depth ONLY to the original ToF-zone emulator. The
        # segmentation model below is passed the original frame's uint8 RGB array alone.
        f = core.Frame("adt_apartment", seq, i, t, cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR), k, z,
                       "ground_truth", dw, None, "head", extra={"ground":(-dw, height)})
        pred = model.tof_dav2s(f)
        assert pred is not None
        fwd = np.array([0, 0, 1.]) - dw*dw[2]
        fwd /= np.linalg.norm(fwd)
        axes = (dw, fwd, np.cross(dw, fwd))
        # Removing pixels cannot create an alert. Validate each original output even when skipping
        # semantic inference for an originally quiet frame.
        mask = None
        if any(original["alert"][BASELINE].values()):
            mask = masker.predict(rgb[i])
            if mask.shape != pred.shape or mask.dtype != np.bool_:
                raise ValueError("RGB mask must match the prediction in pixel geometry and have boolean dtype")
        before, after, low_support = readout(pred, rays, projected_area, axes, height, mask)
        if before != original["alert"][BASELINE]:
            raise ValueError(f"Original baseline mismatch at {seq} frame {i}: {before} vs {original['alert'][BASELINE]}")
        if mask is not None and not any(r["entry_crosscheck"] for r in records):
            reg = {"axes":axes, "ground":(-dw, height)}
            direct = core.band_alert(np.where(mask, np.nan, pred), f, reg)
            assert direct == after
            entry_crosscheck = True
        else:
            entry_crosscheck = False
        low_dist = original["truth_dist"]["LOW"]
        gt = original["truth"]
        if gt["LOW"] == 1 and gt["BODY"] == gt["HEAD"] == 0 and any(before.values()) and not any(after.values()):
            # Post-scoring diagnostic, never used as mask input.
            review = OUT / "review"
            review.mkdir(exist_ok=True)
            truth = core.regions(f, None)["corridor_LOW"].reshape(z.shape)
            marked = f.image.copy()
            for layer, col in [(mask,(255,255,0)), (truth,(0,255,0)), (low_support,(0,0,255))]:
                if layer is None:
                    continue
                contours,_ = cv2.findContours(layer.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked,contours,-1,col,1)
            cv2.imwrite(str(review / f"{seq}_{i}.jpg"), np.concatenate([f.image, marked], axis=1))
            np.savez_compressed(review / f"{seq}_{i}.npz",rgb=rgb[i], mask=mask, gt_low=truth,
                                predicted_low_support=low_support if low_support is not None else np.zeros_like(mask))
        records.append(dict(seq=seq, i=i, t=t, truth=original["truth"], before=before, after=after,
                            low_dist=low_dist, entry_crosscheck=entry_crosscheck,
                            low_support_overlap_px=int((low_support & mask).sum()) if low_support is not None else 0,
                            mask_area_px=int(mask.sum()) if mask is not None else None))
        if ordinal and ordinal % 50 == 0:
            print(seq, ordinal + 1, "/", len(rows), flush=True)
    return records


def run():
    import psutil
    import torch
    from rgb_ground_mask_20261118 import GroundMask
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    if psutil.virtual_memory().available < 2 * (1024**3):
        raise RuntimeError("Less than 2 GiB of available host RAM")
    OUT.mkdir(parents=True, exist_ok=True)
    model, masker = OriginalIndoorDepth(), GroundMask(device="cuda")
    grouped = defaultdict(list)
    for r in read(BASE / "run/rows_adt_apartment.json"):
        grouped[r["seq"]].append(r)
    for seq, rows in grouped.items():
        path = OUT / f"{seq}.json"
        if path.exists():
            assert len(read(path)) == len(rows)
            continue
        if psutil.virtual_memory().available < 2 * (1024**3):
            raise RuntimeError("Less than 2 GiB of available host RAM")
        start = time.perf_counter()
        records = replay(seq, rows, model, masker)
        tmp = path.with_suffix(".tmp")
        save(tmp, records)
        tmp.replace(path)
        print(seq, "completed", len(records), "seconds", round(time.perf_counter()-start, 1),
              "GPU reserved MiB", round(torch.cuda.max_memory_reserved()/2**20),
              "available host MiB", round(psutil.virtual_memory().available/2**20), flush=True)
    summarize()


def contiguous(rows):
    groups = []
    for r in rows:
        if (not groups or groups[-1][-1]["seq"] != r["seq"] or groups[-1][-1]["i"]+1 != r["i"] or
                r["t"]-groups[-1][-1]["t"] > .75):
            groups.append([])
        groups[-1].append(r)
    return groups


def hit(r, arm, detector):
    return any(r[arm].values()) if detector == "ANY" else r[arm][detector]


def score_group(rows, detector):
    n = len(rows)
    before = [r for r in rows if hit(r,"before",detector)]
    after = [r for r in rows if hit(r,"after",detector)]
    assert all(hit(r,"before",detector) for r in after)
    truth_events = contiguous(rows)
    initial_runs = contiguous(before)
    outcome = Counter()
    for event in initial_runs:
        remaining = sum(hit(r,"after",detector) for r in event)
        outcome["kept" if remaining == len(event) else "lost" if not remaining else "partial"] += 1
    both = lambda arm: sum(any(hit(r,arm,detector) for r in event) for event in truth_events)
    return dict(frames=n, before=len(before), after=len(after), lost=len(before)-len(after),
                truth_events=len(truth_events), truth_events_hit_before=both("before"), truth_events_hit_after=both("after"),
                original_alert_runs=len(initial_runs), full_runs_lost=outcome["lost"], partial_runs=outcome["partial"],
                runs_after=len(contiguous(after)))


def collect(rows, floor_cases):
    any_truth = lambda r: core.obstacle_truth(r)
    low = lambda r: r["truth"]["LOW"]
    low_only = lambda r: low(r)==1 and r["truth"]["BODY"]==0
    low_alone = lambda r: low_only(r) and r["truth"]["HEAD"]==0
    result = {}
    for view, target, detector in [
            ("any", any_truth, "ANY"), ("low_band", low, "LOW"),
            ("body_band", lambda r:r["truth"]["BODY"], "BODY"), ("head_band",lambda r:r["truth"]["HEAD"],"HEAD")]:
        result[view] = {name:score_group([r for r in rows if target(r)==v],detector)
                        for v,name in [(1,"positive"),(0,"negative"),(-1,"ambiguous")]}
    for view, predicate in [("all_low_positive",lambda r:low(r)==1),("low_only",low_only),
                            ("low_strictly_alone",low_alone),
                            ("low_strictly_alone_near",lambda r:low_alone(r) and r["low_dist"]<=1.5),
                            ("previous_floor_fp",lambda r:(r["seq"],r["i"]) in floor_cases)]:
        subset = [r for r in rows if predicate(r)]
        result[view] = {d:score_group(subset,d) for d in ["LOW","ANY"]}
        result[view]["lost_low_with_other_band_still_alerting"] = sum(
            r["before"]["LOW"] and not r["after"]["LOW"] and hit(r,"after","ANY") for r in subset)
        result[view]["baseline_low_alerts_with_mask_overlap"] = sum(r["low_support_overlap_px"]>0 for r in subset)
    return result


def summarize():
    original = read(BASE / "run/rows_adt_apartment.json")
    sequences = sorted({r["seq"] for r in original})
    by_seq = {s:read(OUT / f"{s}.json") for s in sequences}
    rows = [r for rst in by_seq.values() for r in rst]
    assert len(rows) == len(original)
    floor_cases = {(r["seq"],r["i"]) for r in read(WALL / "details.json")
                   if r["audit"].get(BASELINE,{}).get("eliminated",{}).get("floor_plane_3cm")}
    assert len(floor_cases) == 7
    summary = dict(standing_frames=len(rows), original_baseline_band_matches=3*len(rows),
                   masked_frames=sum(r["mask_area_px"] is not None for r in rows),
                   direct_filtered_readout_crosschecks=sum(r["entry_crosscheck"] for r in rows),
                   metrics=collect(rows,floor_cases),
                   per_sequence={s:collect(rs,floor_cases) for s,rs in by_seq.items()},
                   event_definition="One sequence, adjacent 2 Hz prepared indices AND elapsed <=0.75 s; a truth episode is hit if any of its rows alerts. For original alert runs, require all rows to go quiet to count it lost. runs_after can split or shrink.",
                   low_only_definition="LOW occupied, BODY strictly empty; HEAD unrestricted. low_strictly_alone additionally requires HEAD strictly empty.",
                   input_boundary="Ground segmenter sees raw RGB only. Both arms retain the original simulated coaxial 8x8 ToF readings sampled from ADT depth, ADT gravity and reference floor for height-band scoring. This is a paired incremental test, not an end-to-end no-reference test.")
    save(OUT / "summary.json",summary)
    metrics=summary["metrics"]
    for name in ["any","low_band","low_only","low_strictly_alone","previous_floor_fp"]:
        print(name, json.dumps(metrics[name]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run","summarize"])
    args = parser.parse_args()
    (run if args.stage == "run" else summarize)()
