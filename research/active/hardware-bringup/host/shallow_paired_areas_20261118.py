"""Measure the three registered shallow branches on previously locked objects and controls."""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import independent_low_replay_20261118 as replay
from shallow_surface_evidence_20261118 import shallow_masks
from ground_geometry_filter_20261118 import protection

PREVIOUS = replay.ROOT / "artifacts.local/work/paired-low-sites-20261118"
OUT = replay.ROOT / "artifacts.local/work/shallow-alerts-20261118"
CATALOG = Path(__file__).resolve().parents[1] / "PAIRED_LOW_SITES_20261118.json"
SOURCES = {
    "egowalk-union": "newsource/egowalk/locked_union",
    "lin-yellow": "lamar/locked/lin_yellow",
    "oxford": "oxford/locked",
}


def by_frame(rows):
    result = {(r["seq"], r["i"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate recording timestamp")
    return result


def branches(packet, plane, view=None):
    z, k = packet["z"], packet["K"]
    probability, remove = packet["probability"], packet["geom"]
    corrected = view is not None and not view.all()
    if corrected:
        z = np.where(view, z, np.nan)
        probability = np.where(view, probability, 0.)
        remove = (probability >= .80) & ~protection(z, k, probability)
    h, w = z.shape
    global_mask, local_mask, diagnostics = shallow_masks(z, k, probability)
    # Use the old reference coordinate system exclusively for the literal baseline
    # and its diagnostic; the two prospective readouts above cannot see this plane.
    down = -np.asarray(plane[0])
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    axes = (down, forward, np.cross(down, forward))
    flat = z.ravel()
    valid = np.isfinite(flat) & (flat > .05)
    safe = np.where(valid, flat, 0.)
    rays = replay.core.rays(k, h, w)
    above, along, corridor = replay.core._corridor(rays * safe[:, None], valid, axes, plane[1])
    raw = corridor & (above >= .05) & (above < .50)
    keep = ~remove.ravel()
    lit = replay.core._open(raw & keep, h, w).reshape(h, w)
    masks = {"literal": lit, "global": global_mask, "local": local_mask}
    area = (safe ** 2 / (k[0, 0] * k[1, 1] * np.linalg.norm(rays, axis=1))).reshape(h, w)
    totals = {name: float(area[mask].sum()) for name, mask in masks.items()}
    # Diagnostic only: can predicted geometry carry the true object's area before
    # ground filtering, and what height would the old reference ground assign it?
    auxiliary = {
        "unfiltered_lowered": replay.core._open(raw, h, w).reshape(h, w),
        "in_corridor": corridor.reshape(h, w),
        "near_corridor": (corridor & (along <= 1.50)).reshape(h, w),
        "predicted_height": above.reshape(h, w),
    }
    old_masks = None
    if corrected:
        old_masks = [replay.core._open(corridor & keep & (above >= lo) & (above < hi), h, w).reshape(h, w)
                     for lo, hi in replay.core.BANDS.values()]
    return masks, totals, area, auxiliary, diagnostics, old_masks


def target_overlap(entry, labels_path, native_size, masks, area, auxiliary, old_masks, original_target):
    path = labels_path.parent / entry["reference_footprint"]
    if replay.sha(path) != entry["reference_mask_sha256"]:
        raise ValueError(f"Changed footprint {path}")
    roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if roi is None or tuple(reversed(roi.shape)) != tuple(native_size):
        raise ValueError(f"Wrong footprint grid {path}")
    if roi.shape != area.shape:
        roi = cv2.resize(roi, tuple(reversed(area.shape)), interpolation=cv2.INTER_NEAREST)
    roi = roi > 0
    target = {name: float(area[roi & mask].sum()) for name, mask in masks.items()}
    geom = {name: float(area[roi & auxiliary[name]].sum())
            for name in ("unfiltered_lowered", "in_corridor", "near_corridor")}
    selected = roi & auxiliary["in_corridor"]
    geom["median_predicted_height_m"] = (
        float(np.median(auxiliary["predicted_height"][selected])) if selected.any() else None)
    old_target = original_target if old_masks is None else any(
        float(area[roi & mask].sum()) >= replay.core.MIN_AREA for mask in old_masks)
    return target, geom, old_target


def measure_source(name, camera_valid):
    manifest = PREVIOUS / SOURCES[name] / "manifest.json"
    labels_path = PREVIOUS / "scoring/labels" / (name + ".json")
    digest = replay.sha(manifest)
    prior = PREVIOUS / "replay" / digest[:16]
    cache = OUT / "cache" / digest[:16]
    score_path = PREVIOUS / "scoring" / (replay.sha(labels_path)[:16] + ".json")
    labels, old, cache_index = map(replay.read, (labels_path, score_path, cache / "index.json"))
    references = replay.read(prior / "reference.json")
    predictions = replay.read(prior / "predictions.json")
    assert all(d["manifest_sha256"] == digest for d in (labels, old, cache_index, references, predictions))
    assert old["predictions_sha256"] == cache_index["old_prediction_sha256"] == replay.sha(prior / "predictions.json")
    assert old["labels_sha256"] == replay.sha(labels_path)
    cache_rows, pred, ref = (by_frame(d["rows"]) for d in (cache_index, predictions, references))
    assert cache_rows.keys() == pred.keys() == ref.keys()

    source_events = [r for r in replay.read(CATALOG)["events"] if r["input_manifest_sha256"] == digest]
    assert {e["labels_sha256"] for e in source_events} == {replay.sha(labels_path)}
    old_cases = {(c["encounter_id"], c["object_id"]): c for c in old["cases"]}
    events, windows, frame_diagnostics = {}, {}, []
    window_sites = {}
    target_refs, control_refs, pass_refs = defaultdict(list), defaultdict(list), defaultdict(list)
    for c in labels["cases"]:
        event_id, object_id = c["encounter_id"], c["object_id"]
        old_case = old_cases[event_id, object_id]
        for window_id in c["window_ids"]:
            assert window_sites.setdefault(window_id, c["site_id"]) == c["site_id"]
        if event_id not in events:
            locked = next(item for item in source_events if item["event"] == event_id)
            ordered = [r for (seq, i), r in pred.items() if seq == locked["sequence"]
                       and min(locked["rgb_frame_keys"]) <= i <= max(locked["rgb_frame_keys"])]
            assert all(r["support_path"] is not None for r in ordered) and ordered
            events[event_id] = dict(id=event_id, site=c["site_id"], category=c["category"], band=c["height_band"],
                                    objects=[], old_hit=False, pass_samples=len(ordered), observations=[],
                                    pass_old_alert_count=sum(any(r["evidence"]["geom"].values()) for r in ordered))
            if camera_valid:
                event = events[event_id]
                event["registered_pass_old_alert_count"] = event["pass_old_alert_count"]
                event["pass_old_alert_count"] = 0
                event["registered_old_hit"] = False
                for row in ordered:
                    pass_refs[row["seq"], row["i"]].append(event)
        event = events[event_id]
        event["objects"].append(object_id)
        event["registered_old_hit" if camera_valid else "old_hit"] |= old_case["hits"]["geom"]["ANY"]
        old_obs = by_frame(old_case["observations"])
        for entry in c["observations"]:
            if entry.get("near", True):
                key = (entry["seq"], entry["i"])
                record = old_obs[key]
                assert record["evaluable"]
                target_refs[key].append((event, object_id, entry, record["hits"]["geom"]["ANY"]))
    for w in old["windows"]:
        assert w["complete"] and w["id"] in window_sites
        window = dict(id=w["id"], site=window_sites[w["id"]], rows=[])
        windows[w["id"]] = window
        for r in w["rows"]:
            if r["alarm"] is not None:
                control_refs[r["seq"], r["i"]].append((window, r["alarm"]["geom"]))

    records = {}
    if camera_valid:
        from shallow_valid_view_20261118 import valid_view
        records = by_frame(replay.inputs(manifest)[1])
    for key in sorted(target_refs.keys() | control_refs.keys() | pass_refs.keys()):
        c = cache_rows[key]
        assert c["old_support_verified"]
        with np.load(cache / c["file"], allow_pickle=False) as p:
            view = valid_view(name, records[key], tuple(p["z"].shape)) if camera_valid else None
            masks, total, area, aux, diag, old_masks = branches(p, ref[key]["ground"], view)
        original_any = any(pred[key]["evidence"]["geom"].values())
        old_any = original_any if old_masks is None else any(
            float(area[mask].sum()) >= replay.core.MIN_AREA for mask in old_masks)
        for event in pass_refs[key]:
            event["pass_old_alert_count"] += int(old_any)
        common = dict(i=c["i"], t=c["t"], total=total)
        for window, original in control_refs[key]:
            assert original == original_any
            ctrl = dict(**common, old=old_any)
            if camera_valid:
                ctrl["registered_old"] = original_any
            window["rows"].append(ctrl)
        for event, object_id, entry, old_target in target_refs[key]:
            size = tuple(reversed(ref[key]["shape"]))
            target, geom, old_target = target_overlap(entry, labels_path, size, masks, area, aux, old_masks, old_target)
            if camera_valid:
                event["old_hit"] |= old_target
            event["observations"].append(dict(**common, object_id=object_id, old_any=old_any,
                                              old_target=old_target, target=target, diagnostic=geom))
        frame_diagnostics.append(dict(seq=key[0], i=key[1], **diag))
    summary = dict(source=name, manifest_sha256=digest, labels_sha256=replay.sha(labels_path),
                   previous_scores_sha256=replay.sha(score_path), cache_index_sha256=replay.sha(cache / "index.json"),
                   verified_cache_frames=cache_index["old_pixel_masks_and_areas_verified"],
                   measured_frames=len(frame_diagnostics))
    data = dict(sources=[summary], events=list(events.values()), windows=list(windows.values()),
                frames=frame_diagnostics)
    replay.save(OUT / (name + ("-areas-camera-valid.json" if camera_valid else "-areas.json")), data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[*SOURCES, "combine"])
    parser.add_argument("--camera-valid", action="store_true", help="Restrict all readouts to source camera rays")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.name == "combine":
        filename = "areas-camera-valid.json" if args.camera_valid else "areas.json"
        docs = [replay.read(OUT / (s + "-" + filename)) for s in SOURCES]
        replay.save(OUT / filename, {k: [r for d in docs for r in d[k]] for k in docs[0]})
    else:
        measure_source(args.name, args.camera_valid)
