"""Audit strict system false alerts on the existing 11/18 ADT apartment development rows.

Run prepare and score with the BlindAssist torch environment. All pixel data and crops stay in
ignored local artifacts. Ground-truth interventions measure an attribution ceiling, not a detector.
For an optional completed review, place manual_review.json in the output directory. Keys are
frame names with alerts and values map each alerted method to its manually inspected description.

From the checkout root, use the same offline GPU Python as the source 11/18 model run:
  python -B research/active/hardware-bringup/host/adt_wall_attribution_20261118.py prepare
  python -B research/active/hardware-bringup/host/adt_wall_attribution_20261118.py score
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evalkit"))
import core

ROOT = HERE.parents[3]
BASE = ROOT / "artifacts.local/work/direction-a-adt-20261118"
OUT = ROOT / "artifacts.local/work/sidewall-attribution-20261118/adt"
METHODS = ("tof_dav2s", "hybrid_in")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, v):
    path.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rows():
    return read(BASE / "run/rows_adt_apartment.json")


def short(seq):
    return seq.removeprefix("Apartment_release_").removesuffix("_M1292")


def load_review(path, entries):
    if not path.exists():
        return None
    notes = read(path)
    expected = {e["name"]: {m for m in METHODS if any(e["alert"][m].values())} for e in entries}
    expected = {name: arms for name, arms in expected.items() if arms}
    if not isinstance(notes, dict) or set(notes) != set(expected):
        raise ValueError("The manual review must cover exactly the frames with alarms.")
    for name, arms in expected.items():
        pair = notes[name]
        if (not isinstance(pair, dict) or set(pair) != arms or
                any(not isinstance(v, str) or not v.strip() for v in pair.values())):
            raise ValueError(f"Missing, empty or unexpected manual review for {name}.")
    return notes


def prepare():
    from methods import Methods
    from run_confirm import PIXELS
    rows = load_rows()
    negative = [r for r in rows if core.obstacle_truth(r) == 0]
    eye = read(BASE / "run/report.json")["adt_apartment"]["eye_heights"]
    methods = Methods()
    methods.set_unidepth_pixels(PIXELS)
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    entries = []
    for seq in sorted({r["seq"] for r in negative}):
        src = BASE / "prep"
        fx, fy, cx, cy = read(src / f"{seq}.json")["K"]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        floor = read(src / f"{seq}.floor.json")["floor_y"]
        with np.load(src / f"{seq}.npz") as d:
            times, down, cam_y = d["t_ns"], d["down"], d["cam_y"]
            rgbs, depths = d["rgb"], d["depth"]
        for row in (r for r in negative if r["seq"] == seq):
            i = int(np.argmin(abs((times - times[0]) / 1e9 - row["t"])))
            assert abs((times[i] - times[0]) / 1e9 - row["t"]) < 1e-8
            def frame_at(j):
                z = depths[j].astype(np.float32) / 1000
                z[z <= 0] = np.nan
                t = float((times[j] - times[0]) / 1e9)
                return core.Frame("adt_apartment", seq, j, t, cv2.cvtColor(rgbs[j], cv2.COLOR_RGB2BGR),
                                  k, z, "ground_truth", down[j].astype(float), eye[seq], "head",
                                  block=f"{seq}:{int(t // 30)}",
                                  extra={"ground": (-down[j].astype(float), float(cam_y[j] - floor))})
            f = frame_at(i)
            reg = core.regions(f, None)
            assert all(row["truth"][b] == reg[f"truth_{b}"] == 0 for b in core.BANDS)
            # A missing current fit inherits the last fit on a standing frame of the same sequence.
            methods.last_fit.clear()
            maps = {}
            for m in METHODS:
                pred = getattr(methods, m)(f)
                if m == "tof_dav2s" and pred is None:
                    for j in range(i-1, -1, -1):
                        if cam_y[j] - floor >= 1.3 and methods.tof_dav2s(frame_at(j)) is not None:
                            break
                    pred = methods.tof_dav2s(f)
                assert pred is not None and core.band_alert(pred, f, reg) == row["alert"][m], (seq, i, m)
                maps[m] = pred
            assert methods.source["hybrid_in"] == row["source"]["hybrid_in"], (seq, i)
            name = f"{short(seq)}_{i:04d}"
            np.savez_compressed(OUT / "maps" / f"{name}.npz", rgb=rgbs[i], depth=f.ref_depth, down=f.down, K=k,
                                height=reg["ground"][1], **maps)
            entries.append(dict(name=name, seq=seq, i=i, t=row["t"], alert={m: row["alert"][m] for m in METHODS}))
            print(name, {m: core.obstacle_alert(row, m) for m in METHODS}, flush=True)
    save(OUT / "index.json", entries)


def normal_masks(xyz, axes):
    # A 7-pixel baseline on the GT surface rejects depth discontinuities and invalid neighbourhoods.
    # For 'possible', uncertain normals remain eligible; no missing-normal pixel is declared non-wall.
    safe = np.isfinite(xyz).all(axis=-1)
    vectors = []
    for axis in (0, 1):
        l, r = np.roll(xyz, -3, axis=axis), np.roll(xyz, 3, axis=axis)
        safe &= (np.linalg.norm(l - xyz, axis=-1) < .20) & (np.linalg.norm(r - xyz, axis=-1) < .20)
        vectors.append(l - r)
    safe[:3] = safe[-3:] = False
    safe[:, :3] = safe[:, -3:] = False
    normal = np.cross(*vectors)
    length = np.linalg.norm(normal, axis=-1)
    safe &= length > 1e-7
    n = normal / np.where(safe, length, 1)[..., None]
    down, fwd, right = axes
    side30 = safe & (np.abs(n @ right) >= np.cos(np.radians(30)))
    side45 = safe & (np.abs(n @ right) >= np.cos(np.radians(45)) - 1e-7)
    vertical = safe & (np.abs(n @ down) <= np.sin(np.radians(30)))
    return safe, side30, side45, vertical


def inspect(entry):
    with np.load(OUT / "maps" / (entry["name"] + ".npz")) as d:
        rgb, ref, k, down, height = d["rgb"], d["depth"], d["K"], d["down"], float(d["height"])
        predictions = {m: d[m] for m in METHODS}
    h, w = ref.shape
    f = core.Frame("adt_apartment", entry["seq"], entry["i"], entry["t"], rgb[..., ::-1], k, ref,
                   "ground_truth", down, None, "head", extra={"ground": (-down, height)})
    reg = core.regions(f, None)
    rays = core.rays(k, h, w).reshape(h, w, 3)
    axes = reg["axes"]
    xyz = rays * ref[..., None]
    valid = reg["valid"].reshape(h,w)
    along, lat, above = xyz @ axes[1], np.abs(xyz @ axes[2]), height - xyz @ axes[0]
    lateral = valid & (lat > core.X_HALF)
    along_inside = (along >= core.Z_NEAR) & (along <= core.Z_FAR)
    height_inside = (above >= core.BANDS["LOW"][0]) & (above < core.BANDS["HEAD"][1])
    reliable, s30, s45, vertical = normal_masks(xyz, axes)
    masks = {
        "all_lateral": lateral,
        "lateral_or_missing": lateral | ~valid,
        "lateral_only": lateral & along_inside & height_inside,
        "parallel30": lateral & s30,
        "possible_side45": ~valid | (lateral & (s45 | ~reliable)),
        "possible_vertical": ~valid | (lateral & (vertical | ~reliable)),
        "floor_plane_3cm": valid & (np.abs(above) <= .03),
        "ref_floor_or_below_low": valid & (above < core.BANDS["LOW"][0]),
        "ref_beyond_far": valid & (along > core.Z_FAR),
        "ref_invalid": ~valid,
    }
    output = {}
    panels = []
    for m, pred in predictions.items():
        if not any(entry["alert"][m].values()):
            continue
        pts = (rays * pred[..., None]).reshape(-1, 3)
        ok = np.isfinite(pred.reshape(-1)) & (pred.reshape(-1) > .05)
        ph, _, planar = core._corridor(pts, ok, axes, height)
        area = core.pixel_area(k, h, w, pred.reshape(-1)).reshape(h,w)
        alert_area, removed_area = {}, {a:{} for a in masks}
        support = np.zeros((h,w), bool)
        weight = np.zeros((h,w), float)
        for b,(lo, hi) in core.BANDS.items():
            raw = (planar & (ph >= lo) & (ph < hi)).reshape(h,w)
            def opened(v):
                return core._open(v,h,w).reshape(h,w)
            baseline = opened(raw)
            alert_area[b] = float(area[baseline].sum())
            assert (alert_area[b] >= core.MIN_AREA) == entry["alert"][m][b]
            if entry["alert"][m][b]:
                support |= baseline
                weight[baseline] = area[baseline]
            for a, mask in masks.items():
                removed_area[a][b] = float(area[opened(raw & ~mask)].sum())
        gone = {a: max(values.values()) < core.MIN_AREA for a, values in removed_area.items()}
        frac = {a:float(weight[mask].sum()/weight.sum()) for a, mask in masks.items()}
        output[m] = dict(area=alert_area, remaining_area=removed_area, eliminated=gone, support_area_fraction=frac)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for selected, color in ((support, (0,0,255)), (support & lateral, (0,170,255)), (support & lateral & s30, (255,230,0))):
            draw, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, draw, -1, color, 2)
        panel = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), canvas], axis=1)
        cv2.rectangle(panel, (0,0),(1023,25),(25,25,25),-1)
        cv2.putText(panel, f'{entry["name"]} {entry["t"]:.2f}s {m} (red=alert, orange=lateral, cyan=parallel30)',(4,17),
                    cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
        panels.append((m,panel))
    return output, panels


def score():
    all_rows = load_rows()
    entries = read(OUT/"index.json")
    reviews = load_review(OUT/"manual_review.json", entries)
    details, panels = [], []
    (OUT/"review").mkdir(exist_ok=True)
    for e in entries:
        out, images = inspect(e)
        reviewed = reviews.get(e["name"], {}) if reviews is not None else None
        details.append({**e, "audit":out, "rgb_review":reviewed})
        for m, im in images:
            cv2.imwrite(str(OUT/"review"/(e["name"]+"_"+m+".jpg")),im)
            panels.append(cv2.resize(im,(640,320)))
    for off in range(0,len(panels),4):
        cv2.imwrite(str(OUT/"review"/f"sheet_{off//4}.jpg"),np.concatenate(panels[off:off+4],axis=0))
    summary = {"standing_frames":len(all_rows), "sequences":len({r["seq"] for r in all_rows}),
               "truth":dict(Counter(core.obstacle_truth(r) for r in all_rows)), "methods":{}}
    summary["audit_scope"] = {
        "sampling": "Existing 2 Hz standing-frame evaluation; same apartment across 12 sequences. All 23 strict negatives replayed; matched two methods in all three bands on all 23.",
        "events": "Consecutive alerted strict negatives in the same recording, adjacent prepared-frame indices, gap <= 0.75 s. A full original event is removed only if every frame goes quiet. Remaining original events counts partial and untouched events; post-intervention fragments are not re-counted.",
        "geometry": "Known common pinhole K, emulated coaxial ToF, device gravity and per-sequence reference floor. Geometric truth uses camera heading and does not establish future human trajectories. Real device calibration not tested by ADT.",
        "rgb_review": {
            "status": "reviewed" if reviews is not None else "unreviewed",
            "frames_with_notes": len(reviews or {}),
            "method_frame_notes": sum(len(v) for v in (reviews or {}).values()),
            "scope": "Where available, local notes describe the appearance under alert contours; they are not pixelwise wall annotations."
        },
        "interventions": "Delete matching raw predicted band pixels, then recompute the original 3x3 opening and 0.005 m^2 area threshold for all three bands. Oracle ceiling for local deletion, not the gain or recall of a deployable filter. Fraction of support uses opened pre-intervention alarm bands weighted by predicted frontal area.",
        "floor_intervention": "The same pixel has valid reference height within +/-0.03 m of the independent sequence floor. Mixed support can leave an event active.",
        "lateral_intervention": "A reference point at the same pixel has absolute lateral offset >0.35 m. The liberal bound includes invalid reference pixels. This also includes furniture and floor off to the side. A sidewall detector affecting pixels beyond its physical surface is outside this direct attribution.",
        "normal_estimator": "Cross product of horizontal and vertical 6-pixel reference-depth differences, discard if an endpoint is >0.20 m from centre or missing. parallel30/possible_side45 are within 30/45 degrees of lateral axis; possible_vertical permits any vertical azimuth (within 30 degrees of horizontal normal). Possible masks include unknown normals and invalid reference pixels."
    }
    for m in METHODS:
        fp = [r for r in details if m in r["audit"]]
        events = []
        for r in fp:
            if (not events or events[-1][-1]["seq"] != r["seq"] or
                    r["i"] != events[-1][-1]["i"] + 1 or r["t"] - events[-1][-1]["t"] > .75):
                events.append([])
            events[-1].append(r)
        cases = {}
        for a in fp[0]["audit"][m]["eliminated"]:
            cases[a] = dict(frames=sum(r["audit"][m]["eliminated"][a] for r in fp),
                            full_events=sum(all(r["audit"][m]["eliminated"][a] for r in ev) for ev in events),
                            partial_events=sum(any(r["audit"][m]["eliminated"][a] for r in ev)
                                               and not all(r["audit"][m]["eliminated"][a] for r in ev) for ev in events))
            cases[a]["remaining_frames"] = len(fp) - cases[a]["frames"]
            cases[a]["remaining_original_events"] = len(events) - cases[a]["full_events"]
        summary["methods"][m] = dict(fp_frames=len(fp), fp_events=len(events), cases=cases,
                                     events=[[r["name"] for r in ev] for ev in events],
                                     band_fp={b:sum(r["truth"][b]==0 and r["alert"][m][b] is True for r in all_rows) for b in core.BANDS},
                                     any_positive_hits=sum(core.obstacle_truth(r)==1 and core.obstacle_alert(r,m) is True for r in all_rows))
    save(OUT/"details.json",details)
    save(OUT/"summary.json",summary)
    print(json.dumps(summary,indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare","score"])
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else score)()
