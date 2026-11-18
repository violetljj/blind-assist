"""Reference-side preparation of the previously identified SANPO circular plaza platform.

It is an old physical facility even though the approach is a revisit. Freeze a 2.5 Hz
sequence around the original reference nomination before computing any alerts. The local
manifests, polygons and visibility reviews are required inputs to reproduce the scoring.
The independent driver uses the old source dataset ID so ToF emulation seeds are stable.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import independent_low_sanpo_fetch_20261118 as acquire
import numpy as np
import paired_low_sites_replay_20261118 as paired

F = paired.frozen
OUT = paired.BASE / "sanpo-round"
OLD = F.ROOT / "artifacts.local/work/independent-low-20261118/sanpo-new"


def source(dense):
    path = OUT / ("dense_manifest.json" if dense else "manifest.json")
    sessions = F.read(path)["sessions"]
    if len(sessions) != 1:
        raise ValueError("The circular-platform review expects one source recording")
    seq, indices = next(iter(sessions.items()))
    return path, seq, indices


def fetch(dense=False):
    p, seq, indices = source(dense)
    acquire.OUT = OUT
    acquire.DATA = OUT / "data"
    paths = [f"{seq}/description.json"] + [
        acquire.filename(seq, i, rgb) for i in indices for rgb in (False, True)
    ]

    def one(rel):
        src = OLD / "data" / rel
        dst = OUT / "data" / rel
        if src.exists():
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
            assert src.stat().st_size == dst.stat().st_size
            return "existing", dst.stat().st_size
        if not acquire.fetch(rel):
            raise FileNotFoundError(rel)
        return "official_public", dst.stat().st_size

    with ThreadPoolExecutor(6) as pool:
        result = list(pool.map(one, paths))
    F.save(
        OUT / ("dense_receipt.json" if dense else "receipt.json"),
        {
            "manifest_sha256": F.sha(p),
            "requested_pairs": len(indices),
            "existing_bytes": sum(b for k, b in result if k == "existing"),
            "downloaded_bytes": sum(b for k, b in result if k == "official_public"),
        },
    )


def inspect():
    manifest, seq, indices = source(False)
    dataset, rows = F.inputs(manifest)
    rd = F.OUT / F.sha(manifest)[:16]
    table = {r["i"]: r for r in F.read(rd / "reference.json")["rows"]}
    view = OUT / "blind"
    view.mkdir(exist_ok=True)
    eligible = []
    for r in rows:
        f = F.load_frame(dataset, r, rd, True)
        base = cv2.resize(f.image, (400, 225))
        ref = table[f.index]
        if ref["ground"]:
            h, w = f.ref_depth.shape
            masks = F.blind_roi(
                manifest,
                seq,
                r["i"],
                [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
                return_masks=True,
            )
            for key, color in [("full", (0, 190, 255)), ("near", (25, 255, 50))]:
                m = masks["LOW"][key]["mask"]
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(f.image, contours, -1, color, 2)
            eligible.append(
                {
                    "i": r["i"],
                    "LOW": {
                        k: {q: v for q, v in m.items() if q != "mask"}
                        for k, m in masks["LOW"].items()
                    },
                    "strict_empty": ref["strict_empty"],
                }
            )
        panel = cv2.hconcat([base, cv2.resize(f.image, (400, 225))])
        cv2.putText(
            panel, f"{r['i']} / {r['t']:.1f}s", (5, 22), 0, 0.65, (0, 50, 255), 2
        )
        cv2.imwrite(str(view / f"{r['i']:03d}.jpg"), panel)
    F.save(view / "ref.json", eligible)
    for j in range(0, len(indices), 12):
        tiles = [
            cv2.resize(cv2.imread(str(view / f"{n:03d}.jpg")), (454, 128))
            for n in indices[j : j + 12]
        ]
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(
            str(view / f"sheet{j // 12}.jpg"),
            cv2.vconcat(
                [cv2.hconcat(tiles[k : k + 2]) for k in range(0, len(tiles), 2)]
            ),
        )


def lock():
    import independent_low_controls_20261118 as controls

    manifest, seq, _ = source(True)
    run = F.OUT / F.sha(manifest)[:16]
    refs = F.read(run / "reference.json")
    controls.PLAN = manifest
    controls.ROOT = paired.BASE
    review = F.read(OUT / "blind/source_review.json")
    if review["manifest_sha256"] != F.sha(manifest) or review["seq"] != seq:
        raise ValueError("Manual review and source recording do not match")
    clear_indices = set(review["visible_empty_candidate_rgb_indices"])
    if not clear_indices.issubset({r["i"] for r in refs["rows"]}):
        raise ValueError("Unrecognized source review indices")
    obs = []
    for key, polygon in sorted(
        review["polygons_800x450"].items(), key=lambda item: int(item[0])
    ):
        i = int(key)
        result = F.blind_roi(manifest, seq, i, polygon, (800, 450), return_masks=True)
        near, full = result["LOW"]["near"], result["LOW"]["full"]
        if full["area_m2"] < 0.005:
            raise ValueError(f"No visible platform reference surface at {i}")
        path = OUT / "blind" / f"platform_near_{i}.png"
        if not cv2.imwrite(str(path), near["mask"].astype(np.uint8) * 255):
            raise OSError(path)
        obs.append(
            {
                "seq": seq,
                "i": i,
                "t": i / 15,
                "near": near["area_m2"] >= 0.005,
                "polygon_800x450": polygon,
                "reference_footprint": str(path.relative_to(OUT)),
                "reference_mask_sha256": F.sha(path),
                "reference_area_m2": near["area_m2"],
                "approach_full_forward_p10_m": full["forward_p10_m"],
                "near_forward_p10_m": near["forward_p10_m"],
            }
        )
    assert (
        obs[0]["approach_full_forward_p10_m"] - obs[-1]["near_forward_p10_m"] > 0.15
        and obs[-1]["near"]
    )
    rows = []
    for ref in refs["rows"]:
        i = ref["i"]
        full, near, _ = controls.visibility(seq, i, ref, refs["dataset"])
        clear_rgb = i in clear_indices
        rows.append(
            {
                "seq": seq,
                "i": i,
                "t": ref["t"],
                "strict": ref["strict_empty"],
                "rgb_clear": clear_rgb,
                "full": full,
                "near": near,
                "eligible": bool(
                    clear_rgb and ref["strict_empty"] and min(full) >= 0.005
                ),
            }
        )
    good = []
    for r in rows:
        if r["eligible"]:
            if not good or not good[-1] or r["t"] - good[-1][-1]["t"] > 0.5 + 1e-8:
                good.append([])
            good[-1].append(r)
        elif good and good[-1]:
            good.append([])
    good = [g for g in good if g]
    qualifying = [
        g for g in good if len(g) >= 3 and g[-1]["t"] - g[0]["t"] >= 1.0 - 1e-8
    ]
    # Nearest visually and geometrically supported interval, up to two seconds. No such
    # interval: only retain the longest supported consecutive fragment and label it partial.
    chosen = next(iter(qualifying), None)
    if chosen is None:
        chosen = max(
            good, key=lambda g: (g[-1]["t"] - g[0]["t"], -g[0]["t"]), default=[]
        )
    if chosen:
        chosen = [r for r in chosen if r["t"] - chosen[0]["t"] <= 2.0 + 1e-8]
    F.save(OUT / "blind/negative_visibility.json", rows)
    wind_id = "plaza-approach-nearest-covered-empty"
    case = {
        "source": "SANPO",
        "site_id": "circular-plaza-old",
        "history": "previously_seen_facility",
        "category": "low_platform",
        "height_band": "LOW",
        "route_id": seq,
        "object_id": "old-concentric-stone-platform",
        "encounter_id": "round-platform-revisit",
        "window_ids": [wind_id] if chosen else [],
        "observations": obs,
    }
    windows = (
        [
            {
                "id": wind_id,
                "route_id": seq,
                "max_gap_s": 0.50001,
                "coverage_scope": "3 lateral strips to 2.5 m; see visibility for native near-1.5 m coverage",
                "frames": [{"seq": seq, "i": x["i"], "eligible": True} for x in chosen],
            }
        ]
        if chosen
        else []
    )
    doc = {
        "manifest_sha256": F.sha(manifest),
        "reference_sha256": F.sha(run / "reference.json"),
        "eligibility": "Recognizable historical circular platform; geometric fore-aft approach and same-object reference footprints; eligible near observations before unblinding.",
        "cases": [case],
        "windows": windows,
    }
    path = OUT / "labels_locked.json"
    if path.exists() and F.read(path) != doc:
        raise ValueError("Refusing to replace changed blind labels")
    F.save(path, doc)
    print(
        "lock",
        F.sha(path),
        [(r["i"], r["reference_area_m2"], r["near_forward_p10_m"]) for r in obs],
        "control",
        [r["i"] for r in chosen],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["fetch", "dense", "inspect", "lock"])
    a = p.parse_args()
    if a.action == "inspect":
        inspect()
    elif a.action == "lock":
        lock()
    else:
        fetch(a.action == "dense")
