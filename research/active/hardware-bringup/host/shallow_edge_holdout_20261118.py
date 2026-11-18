"""Replay the locked 5 cm and edge methods on reference-selected, independent routes."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import cv2
import numpy as np

import independent_low_replay_20261118 as replay
from ground_geometry_filter_20261118 import GroundEvidence, protection
import shallow_edge_method_20261118 as edge

ROOT = replay.ROOT / "artifacts.local/work/shallow-edge-20261118"
METHOD = "3f2449b480cdcbb459c8d5756220daf69db026f06b0771ea36892085d0bc4a62"
AREA = .005
VARIANT = "rgb_relief"


def key(row):
    return row["seq"], row["i"]


def component_scores(z, k, probability, bgr, view, floor):
    ids, edge_area = edge.edge_components(bgr, z, k, probability, view, VARIANT)
    depth = np.where(view, z, np.nan)
    prob = np.where(view, probability, 0.)
    removed = (prob >= .80) & ~protection(depth, k, prob)
    h, w = z.shape
    down = -np.asarray(floor[0])
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    valid = np.isfinite(depth.ravel()) & (depth.ravel() > .05)
    safe = np.where(valid, depth.ravel(), 0.)
    rays = replay.core.rays(k, h, w)
    above, _, corridor = replay.core._corridor(rays * safe[:, None], valid,
                                               (down, forward, np.cross(down, forward)), floor[1])
    eligible = corridor & ~removed.ravel()
    def band(lo, hi):
        return replay.core._open(eligible & (above >= lo) & (above < hi), h, w).reshape(h, w)
    old = [band(lo, hi) for lo, hi in replay.core.BANDS.values()]
    lowered = band(.05, .50)
    area = (safe ** 2 / (k[0, 0] * k[1, 1] * np.linalg.norm(rays, axis=1))).reshape(h, w)

    def readout(roi=None):
        def metric(mask):
            return float(area[mask if roi is None else mask & roi].sum())
        original = max(metric(m) for m in old)
        direct = max(original, metric(lowered))
        extra = edge.largest_area(ids, edge_area, roi)
        return dict(original=original >= AREA, direct=direct >= AREA, edge=max(direct,extra) >= AREA,
                    original_area=original, direct_area=direct, edge_component_area=extra)
    return readout


def get_view(row, manifest, size):
    path = row.get("aperture")
    if not path:
        return np.ones(size, dtype=bool)
    mask = cv2.imread(str(manifest.parent / path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != size:
        raise ValueError(f"Expected calibrated aperture on the model grid at {path}")
    return mask != 0


def setup(manifest, label_path, reference_path):
    labels = replay.read(label_path)
    digest = replay.sha(manifest)
    if labels["manifest_sha256"] != digest:
        raise ValueError("Validation labels were locked against different source inputs")
    dataset, rows = replay.inputs(manifest)
    indexed = {key(r): r for r in rows}
    planes, heights = {}, {}
    if reference_path is not None:
        ref = replay.read(reference_path)
        assert ref["manifest_sha256"] == digest
        planes = {key(r): r["ground"] for r in ref["rows"]}
        heights = ref["eye_height"]
    else:
        for r in rows:
            if r.get("ground") is None:
                raise ValueError("Provide the source reference planes or --reference")
            planes[key(r)] = r["ground"]
        for seq in sorted({r["seq"] for r in rows}):
            known = [r["eye_height"] for r in rows if r["seq"]==seq and r.get("eye_height") is not None]
            if known:
                heights[seq] = float(np.median(known))
    objects = defaultdict(list)
    events = {}
    for c in labels["cases"]:
        event_key = (c["site_id"], c["encounter_id"])
        frames = sorted((key(f) for f in c["pass_frames"]), key=lambda k:indexed[k]["t"])
        if not frames or len(frames) != len(set(frames)) or len({s for s,i in frames}) != 1:
            raise ValueError(f"Missing/duplicate pass timestamps: {event_key}")
        lo, hi = indexed[frames[0]]["t"], indexed[frames[-1]]["t"]
        envelope = [key(r) for r in rows if r["seq"] == frames[0][0] and lo <= r["t"] <= hi]
        if frames != envelope:
            raise ValueError(f"Incomplete available timeline for {event_key}")
        e = events.setdefault(event_key, dict(site=c["site_id"], id=c["encounter_id"], frames=frames, objects=[],
                                             window_ids=[], category=c.get("category","wheel_stop")))
        if e["frames"] != frames or c["object_id"] in e["objects"]:
            raise ValueError("Conflicting pass membership")
        e["objects"].append(c["object_id"])
        for wid in c.get("window_ids",[]):
            if wid not in e["window_ids"]:
                e["window_ids"].append(wid)
        for obs in c["observations"]:
            if obs.get("near", True):
                k = key(obs)
                assert k in frames
                fp = label_path.parent / obs["reference_footprint"]
                assert replay.sha(fp) == obs["reference_mask_sha256"]
                objects[k].append((event_key, c["object_id"], fp))
    controls, owners = [], {}
    for w in labels["windows"]:
        frames = [key(q) for q in w["frames"] if q.get("eligible",True)]
        if not frames or len({s for s,i in frames}) != 1:
            raise ValueError(f"Invalid empty episode {w['id']}")
        ts = np.asarray([indexed[k]["t"] for k in frames])
        if len(ts)<3 or ts[-1]-ts[0]<1.-1e-5 or np.any(np.diff(ts)<=0) or np.diff(ts).max()>min(w.get("max_gap_s",.50001),.50001)+1e-5:
            raise ValueError(f"Insufficient/irregular control coverage {w['id']}")
        for k in frames:
            if k in owners or any(k in e["frames"] for e in events.values()):
                raise ValueError("Overlapping positive and negative samples or duplicate controls")
            owners[k] = w["id"]
        controls.append(dict(site=w["site_id"], id=w["id"], frames=frames))
    named = {w["id"]:w for w in controls}
    for e in events.values():
        for wid in e["window_ids"]:
            if named[wid]["site"] != e["site"] or named[wid]["frames"][0][0] != e["frames"][0][0]:
                raise ValueError("Paired empty segment must share the recording and physical site")
    return dataset, rows, planes, heights, objects, events, controls


def run(manifest, labels, reference):
    method_bytes = Path(edge.__file__).read_bytes()
    if METHOD not in {hashlib.sha256(data).hexdigest() for data in (method_bytes, method_bytes.replace(b"\r\n", b"\n"))}:
        raise ValueError("Edge algorithm differs from the pre-validation freeze")
    dataset, rows, planes, heights, objects, events, controls = setup(manifest, labels, reference)
    root = ROOT / "results" / (replay.sha(manifest)[:12]+"-"+replay.sha(labels)[:12])
    root.mkdir(parents=True, exist_ok=True)
    import torch
    from shallow_prediction_cache_20261118 import ensure_memory
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = replay.Methods()
    del model.rel
    torch.cuda.empty_cache()
    model.set_unidepth_pixels(315560)
    ground = GroundEvidence(device="cuda")
    frames = {}
    for n, row in enumerate(rows,1):
        k = key(row)
        f = replay.load_frame(dataset,row,root,True,model_grid=True)
        f.eye_height = heights.get(f.seq)
        # Advance temporal scale through the untrimmed ordered input even if a reference is missing.
        ensure_memory()
        z = model.hybrid_out(f)
        result = dict(seq=f.seq, i=f.index, t=f.t, scale=model.source["hybrid_out"], alert=None, targets=[])
        if z is not None and planes.get(k) is not None:
            p = ground.probability(cv2.cvtColor(f.image,cv2.COLOR_BGR2RGB))
            reader = component_scores(z, f.K, p, f.image, get_view(row,manifest,z.shape), planes[k])
            result["alert"] = reader()
            for ek, oid, fp in objects[k]:
                mask = cv2.imread(str(fp),cv2.IMREAD_GRAYSCALE)
                if mask is None or (row.get("size") and mask.shape != tuple(reversed(row["size"]))):
                    raise ValueError(f"Wrong reference footprint grid {fp}")
                if mask.shape != z.shape:
                    mask = cv2.resize(mask,(z.shape[1],z.shape[0]),interpolation=cv2.INTER_NEAREST)
                result["targets"].append(dict(event=ek[1],object_id=oid, **reader(mask>0)))
        frames[k] = result
        if n % 25 == 0:
            print(dataset,n,"/",len(rows),flush=True)
    for e in events.values():
        sub = [frames[k] for k in e["frames"]]
        ts = np.array([r["t"] for r in sub])
        covered = len(ts)>=2 and np.diff(ts).max()<=.50002 and all(r["alert"] is not None for r in sub)
        near = [t for r in sub for t in r["targets"] if t["event"]==e["id"]]
        missing = set(e["objects"])-{t["object_id"] for t in near}
        e["silent"] = (not any(r["alert"]["original"] for r in sub)) if covered and not missing else None
        e["hits"] = {arm: sorted({t["object_id"] for t in near if t[arm]}) for arm in ("original","direct","edge")}
        e["sample_count"] = len(ts)
        e["max_gap_s"] = float(np.diff(ts).max()) if len(ts)>1 else None
    for w in controls:
        sub = [frames[k] for k in w["frames"]]
        w["complete"] = all(r["alert"] is not None for r in sub)
        w["alarms"] = {arm:[r["i"] for r in sub if r["alert"] is not None and r["alert"][arm]]
                       for arm in ("original","direct","edge")}
    output = dict(dataset=dataset,manifest_sha256=replay.sha(manifest),labels_sha256=replay.sha(labels),
                  reference_sha256=replay.sha(reference) if reference else None,method_sha256=METHOD,
                  variant=VARIANT,threshold_m2=AREA,events=list(events.values()),windows=controls,
                  frames=list(frames.values()))
    replay.save(root/"comparison.json",output)
    print(root/"comparison.json",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("labels",type=Path)
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    run(args.manifest.resolve(),args.labels.resolve(),args.reference)
