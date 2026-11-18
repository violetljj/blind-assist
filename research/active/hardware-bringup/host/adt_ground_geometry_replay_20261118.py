"""Fixed ADT comparison of geometry protection and stricter RGB-ground thresholds.

Run with the indoor evaluation's offline GPU Python:
  python -B research/active/hardware-bringup/host/adt_ground_geometry_replay_20261118.py

Only frames with a changed band between previous base and semantic-only 0.80 need new readouts.
Morphological opening, positive area weighting and thresholding all preserve set inclusion.
Every candidate deletes a subset of the 0.80 semantic deletion, bounding its three alert bits by
the original two arms. All 1,992 development observations are retained in the local ledger.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from adt_ground_alert_replay_20261118 import BASE, BASELINE, OUT as PRIOR, ROOT, WALL, OriginalIndoorDepth, core, read, save

OUT = ROOT / "artifacts.local/work/ground-geometry-20261118/adt"


def threshold_name(threshold):
    return f"{threshold:.3f}".rstrip("0") if threshold in (.995, .999) else f"{threshold:.2f}"


def exact_depth(model, f, earlier, make_frame):
    """Restore an absent fit from the closest earlier standing frame having its own fit."""
    model.last_fit.clear()
    pred = model.tof_dav2s(f)
    history = []
    if pred is None:
        for j in reversed(earlier):
            history.append(j)
            if model.tof_dav2s(make_frame(j)) is not None:
                break
        pred = model.tof_dav2s(f)
    if pred is None:
        raise ValueError(f"No original indoor depth available at {f.seq}, frame {f.index}")
    return pred, history


def replay(seq, previous, model, evidence, thresholds, protect):
    changed = {r["i"] for r in previous if r["before"] != r["after"]}
    if not changed:
        return {}, []
    src = BASE / "prep"
    fx, fy, cx, cy = read(src / f"{seq}.json")["K"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    floor = read(src / f"{seq}.floor.json")["floor_y"]
    with np.load(src / f"{seq}.npz") as d:
        ts, down, cam_y, rgb, depth = d["t_ns"], d["down"], d["cam_y"], d["rgb"], d["depth"]
    def make_frame(i):
        t = (int(ts[i]) - int(ts[0])) / 1e9
        z = depth[i].astype(np.float32) / 1000.
        z[z <= 0] = np.nan
        dw = down[i].astype(float)
        return core.Frame("adt_apartment", seq, i, t, cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR), k,
                          z, "ground_truth", dw, None, "head", extra={"ground":(-dw, float(cam_y[i]-floor))})
    output, history_inputs = {}, []
    indices = [r["i"] for r in previous]
    for j, r in enumerate(previous):
        i = r["i"]
        if i not in changed:
            continue
        f = make_frame(i)
        assert f.t == r["t"]
        pred, history = exact_depth(model, f, indices[:j], make_frame)
        history_inputs.extend(history)
        # Only the original ToF emulator and evaluation coordinate system use ADT inputs. The
        # extra protection receives the predicted depth, known K and RGB-only ground probability.
        probability = evidence.probability(rgb[i])
        assert probability.dtype == np.float32 and probability.shape == pred.shape
        assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1.+1e-6
        protected = protect(pred, k, probability)
        assert protected.dtype == np.bool_ and protected.shape == pred.shape
        dw = f.down
        fwd = np.array([0, 0, 1.]) - dw*dw[2]
        fwd /= np.linalg.norm(fwd)
        reg = {"axes":(dw, fwd, np.cross(dw, fwd)), "ground":f.extra["ground"]}
        def alert(remove):
            return core.band_alert(np.where(remove, np.nan, pred), f, reg)
        base = core.band_alert(pred, f, reg)
        sem = probability >= .80
        geom = alert(sem & ~protected)
        controls = {threshold_name(t): alert(probability >= t) for t in thresholds}
        if base != r["before"] or controls["0.80"] != r["after"] or controls["1.01"] != base:
            raise ValueError(f"Reference arm mismatch at {seq} frame {i}")
        for band in core.BANDS:
            chain = [controls[threshold_name(t)][band] for t in thresholds]
            if chain != sorted(chain) or not (chain[0] <= geom[band] <= chain[-1]):
                raise ValueError(f"Nonmonotone result at {seq} frame {i} band {band}")
        output[i] = dict(geom=geom, thresholds=controls, mask80_px=int(sem.sum()),
                         restored_px=int((sem & protected).sum()), history_fit_queries=history)
    return output, history_inputs


def run():
    import psutil
    import torch
    from ground_geometry_filter_20261118 import CONTROL_THRESHOLDS, GroundEvidence, protection
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if psutil.virtual_memory().available < 2*1024**3:
        raise RuntimeError("Less than 2 GiB of available host RAM")
    prior_files = sorted(PRIOR.glob("Apartment*.json"))
    original = read(BASE / "run/rows_adt_apartment.json")
    row_ids = {(r["seq"],r["t"]):j for j,r in enumerate(original)}
    floor_cases = {(r["seq"],r["i"]) for r in read(WALL/"details.json")
                   if r["audit"].get(BASELINE,{}).get("eliminated",{}).get("floor_plane_3cm")}
    assert len(row_ids) == len(original) and len(floor_cases) == 7
    model, evidence = OriginalIndoorDepth(), GroundEvidence(device="cuda")
    rows = []
    replayed, history_used = defaultdict(int), defaultdict(list)
    for path in prior_files:
        previous = read(path)
        seq = path.stem
        actual, history = replay(seq, previous, model, evidence, CONTROL_THRESHOLDS, protection)
        replayed[seq] = len(actual)
        history_used[seq] = history
        for r in previous:
            j = row_ids[(seq,r["t"])]
            source = original[j]
            assert source["truth"] == r["truth"] and source["alert"][BASELINE] == r["before"]
            assert source["truth_dist"]["LOW"] == r["low_dist"]
            result = actual.get(r["i"])
            method = "replayed" if result is not None else "monotone"
            if result is None:
                assert r["before"] == r["after"]
                result = dict(geom=r["before"], thresholds={threshold_name(t):r["before"] for t in CONTROL_THRESHOLDS},
                              mask80_px=None, restored_px=None, history_fit_queries=None)
            rows.append(dict(seq=seq, i=r["i"], t=r["t"], row_index=j, truth=r["truth"], low_dist=r["low_dist"],
                             floor_case=(seq,r["i"]) in floor_cases, baseline=r["before"], sem80=r["after"],
                             method=method, **result))
        print(seq, "replayed", len(actual), "history queries", len(history), flush=True)
    rows.sort(key=lambda r:r["row_index"])
    assert [r["row_index"] for r in rows] == list(range(len(original)))
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/"rows.json",rows)
    counts = dict(frames=len(rows), model_replayed=sum(replayed.values()),
                  monotonic_completion=len(rows)-sum(replayed.values()), per_sequence_replayed=replayed,
                  per_sequence_history_fit_queries=history_used, thresholds=list(CONTROL_THRESHOLDS),
                  old_band_checks=6*sum(replayed.values()), gpu_peak_reserved_MiB=round(torch.cuda.max_memory_reserved()/2**20,1))
    save(OUT/"replay_checks.json",counts)
    print(counts, flush=True)


if __name__ == "__main__":
    run()
