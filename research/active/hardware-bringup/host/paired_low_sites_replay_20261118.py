"""Frozen paired low-site replay. Source manifests and blind ROI/control locks stay local.

python -B paired_low_sites_replay_20261118.py reference source.json
python -B paired_low_sites_replay_20261118.py predict source.json
python -B paired_low_sites_replay_20261118.py score source.json labels.json
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import independent_low_replay_20261118 as frozen
import numpy as np

BASE = frozen.ROOT / "artifacts.local/work/paired-low-sites-20261118"
frozen.OUT = BASE / "replay"
ARMS = frozen.ARMS


def match_object(packet_path, roi, alerts):
    overlap = frozen.roi_areas(packet_path, roi)
    areas = {a: {b: overlap[a][b]["area_m2"] for b in frozen.BANDS} for a in ARMS}
    matched = {
        a: {
            b: alerts[a][b] and areas[a][b] >= frozen.core.MIN_AREA
            for b in frozen.BANDS
        }
        for a in ARMS
    }
    hits = {
        a: {"LOW": bands["LOW"], "ANY": any(bands.values())}
        for a, bands in matched.items()
    }
    return areas, hits


def score(manifest, labels_path):
    labels = frozen.read(labels_path)
    digest = frozen.sha(manifest)
    if labels["manifest_sha256"] != digest:
        raise ValueError("Blind labels and predicted frame manifest do not match")
    run = frozen.OUT / digest[:16]
    p = frozen.read(run / "predictions.json")
    reference = frozen.read(run / "reference.json")
    if p["manifest_sha256"] != digest or reference["manifest_sha256"] != digest:
        raise ValueError("Different frozen source used for predictions/reference")
    if labels.get("reference_sha256") and labels["reference_sha256"] != frozen.sha(
        run / "reference.json"
    ):
        raise ValueError("The frozen source reference changed after object selection")
    by_key = {(r["seq"], r["i"]): r for r in p["rows"]}
    ref = {(r["seq"], r["i"]): r for r in reference["rows"]}
    if (
        set(by_key) != set(ref)
        or len(by_key) != len(p["rows"])
        or len(ref) != len(reference["rows"])
    ):
        raise ValueError(
            "Expected one prediction and one reference row per source frame"
        )
    cases = []
    for case in labels["cases"]:
        obs = []
        for entry in case["observations"]:
            if not entry.get("near", True):
                continue
            key = (entry["seq"], entry["i"])
            r = by_key[key]
            row = {
                "seq": key[0],
                "i": key[1],
                "t": r["t"],
                "evaluable": r["support_path"] is not None,
            }
            if not row["evaluable"]:
                obs.append(row)
                continue
            mask_path = labels_path.parent / entry["reference_footprint"]
            if frozen.sha(mask_path) != entry["reference_mask_sha256"]:
                raise ValueError(f"Reference footprint changed for {key}")
            roi = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if roi is None:
                raise FileNotFoundError(mask_path)
            packet_path = run / r["support_path"]
            areas, hits = match_object(packet_path, roi > 0, r["evidence"])
            row.update(matched_area_m2=areas, alert=r["evidence"], hits=hits)
            if "full_object_polygon" in entry:
                polygon = entry["full_object_polygon"]
                size = tuple(polygon["size"])
                if size != (roi.shape[1], roi.shape[0]):
                    raise ValueError(
                        f"Full object outline does not share the reference grid at {key}"
                    )
                full_mask = np.zeros(roi.shape, dtype=np.uint8)
                cv2.fillPoly(full_mask, [np.asarray(polygon["xy"], dtype=np.int32)], 1)
                full_area, full_hits = match_object(
                    packet_path, full_mask.astype(bool), r["evidence"]
                )
                row.update(full_object_area_m2=full_area, full_object_hits=full_hits)
            obs.append(row)
        ev = [x for x in obs if x["evaluable"]]
        cases.append(
            {
                **{k: v for k, v in case.items() if k != "observations"},
                "observations": obs,
                "hits": {
                    a: {
                        m: any(x["hits"][a][m] for x in ev) if ev else None
                        for m in ["LOW", "ANY"]
                    }
                    for a in ARMS
                },
            }
        )
    windows = []
    for window in labels["windows"]:
        sub = []
        for q in window["frames"]:
            key = (q["seq"], q["i"])
            r = by_key[key]
            if not q.get("eligible", True):
                sub.append({"seq": key[0], "i": key[1], "t": r["t"], "alarm": None})
                continue
            if not ref[key]["strict_empty"]:
                raise ValueError(
                    f"Selected negative lacks strict empty/observed geometry at {key}"
                )
            if r["support_path"] is None:
                sub.append({"seq": key[0], "i": key[1], "t": r["t"], "alarm": None})
                continue
            sub.append(
                {
                    "seq": key[0],
                    "i": key[1],
                    "t": r["t"],
                    "alarm": {a: any(r["evidence"][a].values()) for a in ARMS},
                }
            )
        sub.sort(key=lambda r: r["t"])
        known = [r for r in sub if r["alarm"] is not None]
        # Positive window burden and alert run count are distinct; unknown samples break runs.
        episodes = {a: 0 for a in ARMS}
        for i, r in enumerate(sub):
            if r["alarm"] is None:
                continue
            prev = sub[i - 1] if i else None
            for a in ARMS:
                continuous = (
                    prev
                    and prev["seq"] == r["seq"]
                    and prev["alarm"] is not None
                    and 0 < r["t"] - prev["t"] <= window["max_gap_s"]
                    and prev["alarm"][a]
                )
                if r["alarm"][a] and not continuous:
                    episodes[a] += 1
        durations = []
        current = []
        for r in sub:
            if r["alarm"] is None or (
                current
                and (
                    r["seq"] != current[-1]["seq"]
                    or not 0 < r["t"] - current[-1]["t"] <= window["max_gap_s"]
                )
            ):
                if current:
                    durations.append((len(current), current[-1]["t"] - current[0]["t"]))
                current = []
            if r["alarm"] is not None:
                current.append(r)
        if current:
            durations.append((len(current), current[-1]["t"] - current[0]["t"]))
        duration = max((s for n, s in durations), default=0)
        complete = any(n >= 3 and s >= 1.0 - 1e-5 for n, s in durations)
        windows.append(
            {
                **{k: v for k, v in window.items() if k != "frames"},
                "rows": sub,
                "complete": complete,
                "longest_observed_s": duration,
                "eligible_rows": len(known),
                "episodes": episodes,
                "alarm_windows": {
                    a: any(x["alarm"][a] for x in known) if known else None
                    for a in ARMS
                },
            }
        )
    assert len(
        {
            (c["site_id"], c["route_id"], c["encounter_id"], c["object_id"])
            for c in cases
        }
    ) == len(cases)
    assert len({w["id"] for w in windows}) == len(windows)
    dest = BASE / "scoring" / (frozen.sha(labels_path)[:16] + ".json")
    frozen.save(
        dest,
        {
            "manifest_sha256": digest,
            "labels_sha256": frozen.sha(labels_path),
            "predictions_sha256": frozen.sha(run / "predictions.json"),
            "cases": cases,
            "windows": windows,
        },
    )
    print("scores", dest, flush=True)


def aggregate(paths):
    """Group only user-listed locks; missing controls or events never become negatives."""
    cases = []
    windows = {}
    for p in paths:
        doc = frozen.read(p)
        for c in doc["cases"]:
            cases.append({**c, "manifest_sha256": doc["manifest_sha256"]})
        for w in doc["windows"]:
            key = (doc["manifest_sha256"], w["id"])
            if key in windows and windows[key] != w:
                raise ValueError("One control window has conflicting definitions")
            windows[key] = w
    seen = set()
    for c in cases:
        k = (
            c["source"],
            c["site_id"],
            c["route_id"],
            c["encounter_id"],
            c["object_id"],
        )
        if k in seen:
            raise ValueError(
                "An object in the same walking encounter would be counted twice"
            )
        seen.add(k)

    def group_summary(fields):
        groups = defaultdict(list)
        for case in cases:
            groups[tuple(case[f] for f in fields)].append(case)
        return [
            {**dict(zip(fields, key, strict=True)), **summarize(group, windows)}
            for key, group in groups.items()
        ]

    frozen.save(
        BASE / "scoring" / "cross_source.json",
        {
            "scores": [{"file": str(p), "sha256": frozen.sha(p)} for p in paths],
            "groups": group_summary(
                ("source", "site_id", "history", "category", "height_band")
            ),
            "history_by_band": group_summary(("history", "height_band")),
            "history_by_category": group_summary(("history", "category")),
            "all": summarize(cases, windows),
        },
    )


def paired_count(items, value):
    return {
        "n": len(items),
        **{a: sum(bool(value(x, a)) for x in items) for a in ARMS},
        "geom_only": sum(
            bool(value(x, "geom")) and not value(x, "sem80") for x in items
        ),
        "sem_only": sum(
            bool(value(x, "sem80")) and not value(x, "geom") for x in items
        ),
    }


def grouped_hits(groups):
    evaluable = [
        rows
        for rows in groups.values()
        if all(c["hits"]["baseline"]["ANY"] is not None for c in rows)
    ]
    return {
        band: paired_count(
            evaluable, lambda group, arm, b=band: any(c["hits"][arm][b] for c in group)
        )
        for band in ("LOW", "ANY")
    }


def summarize(cs, windows):
    passes = defaultdict(list)
    objects = defaultdict(list)
    for case in cs:
        location = (case["source"], case["site_id"])
        passes[(*location, case["route_id"], case["encounter_id"])].append(case)
        objects[(*location, case["object_id"])].append(case)
    encounter_hits = grouped_hits(passes)
    unique_windows = {}
    for case in cs:
        for window_id in case.get("window_ids", []):
            window = windows[case["manifest_sha256"], window_id]
            recording_samples = tuple((r["seq"], r["i"]) for r in window["rows"])
            identity = case["source"], recording_samples
            previous = unique_windows.setdefault(identity, window)
            if (
                previous["rows"] != window["rows"]
                or previous["episodes"] != window["episodes"]
            ):
                raise ValueError(
                    "Shared empty samples disagree; combine the source manifests and replay chronologically"
                )
    complete = [w for w in unique_windows.values() if w["complete"]]

    control_rows = [r for w in complete for r in w["rows"] if r["alarm"] is not None]
    return {
        "objects": len(objects),
        "object_hits": grouped_hits(objects),
        "object_encounters": len(cs),
        "sites": len({(c["source"], c["site_id"]) for c in cs}),
        "control_alarm_frames": paired_count(control_rows, lambda r, a: r["alarm"][a]),
        "encounters": len(passes),
        "evaluable_encounters": encounter_hits["ANY"]["n"],
        "uncaptured_encounter_ids": [
            key[-1]
            for key, rows in passes.items()
            if any(c["hits"]["baseline"]["ANY"] is None for c in rows)
        ],
        "hits": encounter_hits,
        "unique_windows": len(unique_windows),
        "complete_windows": len(complete),
        "encounters_with_complete_windows": sum(
            any(
                windows[c["manifest_sha256"], q]["complete"]
                for c in rows
                for q in c.get("window_ids", [])
            )
            for rows in passes.values()
        ),
        "control_windows_with_alarm": paired_count(
            complete, lambda w, a: w["alarm_windows"][a]
        ),
        "control_alarm_runs": {
            a: sum(w["episodes"][a] for w in complete) for a in ARMS
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["reference", "predict", "score", "aggregate"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("paths", type=Path, nargs="*")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if args.stage == "aggregate":
        aggregate([manifest] + [p.resolve() for p in args.paths])
    elif args.stage == "score":
        if len(args.paths) != 1:
            parser.error("score needs a frozen label file")
        score(manifest, args.paths[0].resolve())
    else:
        import psutil
        import torch

        if psutil.virtual_memory().available < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB of available host RAM")
        torch.cuda.set_per_process_memory_fraction(0.75)
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        dataset, rows = frozen.inputs(manifest)
        out = frozen.OUT / frozen.sha(manifest)[:16]
        out.mkdir(parents=True, exist_ok=True)
        (frozen.reference if args.stage == "reference" else frozen.predict)(
            manifest, dataset, rows, out
        )
        print(args.stage, len(rows), out, flush=True)
