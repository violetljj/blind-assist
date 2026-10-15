"""Acquire fresh AGD image groups without opening evaluation heatmap pixels.

Selection uses native index labels, image SHA and 64-bit perceptual hashes only.
The official Seen train split has no pixel heatmaps in this archive.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import threading
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

from acquire_data import Source, agd_index, dump, jsonl, sha

SEED = "20261015-fp-learned-fresh"


class RecordedSource(Source):
    def get(self, lo, hi, offline=False):
        if self.budget["bytes"] + hi-lo+1 > 500_000_000:
            raise RuntimeError("500 MB download bound reached")
        value = super().get(lo, hi, offline)
        with self.lock:
            with (self.out / "resumed_range_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"lo": lo, "hi": hi, "bytes": len(value), "sha256": sha(value)}) + "\n")
        return value


def rank(s):
    return hashlib.sha256((SEED + str(s)).encode()).hexdigest()


def phash(path, flip=False):
    with Image.open(path) as im:
        im = im.convert("L")
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(array, type=2, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low)
    return sum(int(v) << i for i, v in enumerate(bits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--phash-distance", type=int, default=6)
    p.add_argument("--wall-seconds", type=int, default=900)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    start, cpu = time.monotonic(), time.process_time()
    assert "artifacts.local" in a.root.as_posix()
    out = a.root.resolve() / "data"
    out.mkdir(parents=True, exist_ok=True)
    budget = {"bytes": 0, "lock": threading.Lock()}
    source = RecordedSource("AGD20K", a.audit.resolve(), out, start + a.wall_seconds, budget)
    groups, missing = agd_index(source)
    old = [json.loads(x) for x in (a.old / "data/samples.jsonl").read_text().splitlines()]
    old = [r for r in old if r["dataset"] == "AGD20K"]
    audit = json.loads((a.audit / "agd20k_audit/summary.json").read_text())["samples"]
    excluded = {r["source_image_id"] for r in old} | {r["key"][3] for r in audit}
    old_hashes = {r["source_image_id"]: (phash(r["image_path"]), phash(r["image_path"], True)) for r in old}
    refs = [(r["source_image_id"], r["image_sha256"], old_hashes[r["source_image_id"]]) for r in old]
    for path in sorted((a.audit / "agd20k_audit/samples").glob("*_image.jpg")):
        refs.append(("old_audit:" + path.name, sha(path.read_bytes()), (phash(path), phash(path, True))))
    # Calibration has priority; retain its original role and never refill from new images.
    retained, old_exclusions = [], []
    for role, old_role in (("calibration", "calibration"), ("fit", "evaluation")):
        for row in sorted([r for r in old if r["split"] == old_role], key=lambda r: rank(r["source_image_id"])):
            hs = old_hashes[row["source_image_id"]]
            matches = [(min((h ^ q).bit_count() for h in hs for q in old_hashes[r["source_image_id"]]), r)
                       for r in retained]
            nearest = min(matches, key=lambda x: x[0]) if matches else (65, None)
            if nearest[0] <= a.phash_distance:
                old_exclusions.append({"sample_id": row["sample_id"], "source_image_id": row["source_image_id"],
                    "intended_role": role, "image_sha256": row["image_sha256"],
                    "phash64_original_flip": [f"{h:016x}" for h in hs],
                    "duplicate_of": nearest[1]["sample_id"], "distance": nearest[0]})
                continue
            retained.append({**row, "original_split": old_role, "split": role,
                "phash64_original_flip": [f"{h:016x}" for h in hs]})
    for role in ("fit", "calibration"):
        rows = [r for r in retained if r["split"] == role]
        jsonl(out / (role + "_samples.jsonl"), rows)
        jsonl(out / (role + "_observations.jsonl"), [{k: r[k] for k in ("sample_id", "image_path", "action")} for r in rows])
    jsonl(out / "old_split_exclusions.jsonl", old_exclusions)
    available = {i: pairs for i, pairs in groups.items() if i not in excluded}
    # A full balanced ordering is fixed before downloading RGB or GT.
    remaining = dict(available)
    ac, oc, ordered = Counter(), Counter(), []
    while remaining:
        options = [(ac[q["action"]], oc[q["object"]], rank(i + q["action"]), i, q)
                   for i, pairs in remaining.items() for q in pairs]
        _, _, _, identifier, pair = min(options, key=lambda x: x[:3])
        ordered.append({"source_image_id": identifier, **pair})
        ac[pair["action"]] += 1
        oc[pair["object"]] += 1
        del remaining[identifier]
    dump(out / "selection_plan.json", {"seed": SEED, "target": a.count,
         "phash_hamming_exclusion_lte": a.phash_distance, "excluded_ids": sorted(excluded),
         "available_groups": len(available), "ordered_candidates": ordered,
         "GT_pixels_read": False, "selection": "Action then object balanced; SHA ties; RGB-only duplicate exclusions"})
    # Entire ordered inventory is much smaller than 500 MB; no archive download.
    assert sum(source.entries[r[k]]["compressed_size"] + 5000
               for r in ordered for k in ("image_member", "gt_member")) < 450_000_000
    source.connect()
    image_dir, gt_dir = out / "images", out / "GT"
    image_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    selected, rejected, inspected = [], [], []

    def fetch_image(row):
        path = image_dir / (row["source_image_id"] + ".jpg")
        if not path.exists():
            path.write_bytes(source.member(row["image_member"]))
        with Image.open(path) as im:
            im.load()
            width, height = im.size
        return path, sha(path.read_bytes()), (phash(path), phash(path, True)), width, height

    for offset in range(0, len(ordered), 24):
        batch = ordered[offset:offset + 24]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            values = list(pool.map(fetch_image, batch))
        for row, (path, digest, ph, width, height) in zip(batch, values):
            distances = [(min((h ^ q).bit_count() for h in ph for q in old_ph), identifier, digest == old_sha)
                         for identifier, old_sha, old_ph in refs]
            nearest = min(distances)
            duplicate = next((x for x in distances if x[2]), None)
            record = {**row, "image_path": str(path), "image_sha256": digest,
                      "phash64_original_flip": [f"{h:016x}" for h in ph], "width": width, "height": height,
                      "nearest_reference_id": nearest[1], "nearest_hamming": nearest[0]}
            inspected.append(record)
            if duplicate or nearest[0] <= a.phash_distance:
                rejected.append({**record, "reason": "exact_sha" if duplicate else "near_phash"})
                continue
            selected.append(record)
            refs.append((row["source_image_id"], digest, ph))
            if len(selected) == a.count:
                break
        if len(selected) == a.count:
            break
    assert len(selected) == a.count, f"Only {len(selected)} RGB-disjoint groups remain"
    jsonl(out / "rgb_inspected.jsonl", inspected)
    jsonl(out / "duplicate_exclusions.jsonl", rejected)
    dump(out / "rgb_selection_seal.json", {"selected_ids": [r["source_image_id"] for r in selected],
         "selected_sha256": [r["image_sha256"] for r in selected], "phash_threshold": a.phash_distance,
         "GT_pixels_read": False, "selected_before_GT_download": True})

    def fetch_gt(row):
        path = gt_dir / (row["source_image_id"] + "__" + row["action"] + ".png")
        if not path.exists():
            path.write_bytes(source.member(row["gt_member"]))
        body = path.read_bytes()
        # Inspect PNG header only, never decode/open native heatmap pixels here.
        assert body[:8] == b"\x89PNG\r\n\x1a\n" and body[12:16] == b"IHDR"
        assert struct.unpack(">II", body[16:24]) == (row["width"], row["height"])
        return path, sha(body)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        gt_values = list(pool.map(fetch_gt, selected))
    samples = []
    for row, (gt, digest) in zip(selected, gt_values):
        samples.append({**row, "sample_id": "agd_" + row["source_image_id"] + "__" + row["action"],
            "dataset": "AGD20K", "split": "evaluation", "evidence_role": "fresh_image_Development",
            "gt_path": str(gt), "gt_sha256": digest, "label_kind": "native_continuous_heatmap",
            "source_image_member": row["image_member"], "source_GT_member": row["gt_member"], "primary": True})
    jsonl(out / "samples.jsonl", samples)
    jsonl(out / "observations.jsonl", [{k: r[k] for k in ("sample_id", "image_path", "action")} for r in samples])
    dump(out / "range_receipts.json", {"archive_id": source.file_id, "archive_size": source.size, "requests": source.receipts})
    summary = {"status": "SUCCESS", "count": len(samples), "old_AGD_groups": len(old),
        "old_and_audit_excluded_ids": len(excluded), "available_groups": len(available),
        "fit_count": sum(r["split"] == "fit" for r in retained),
        "calibration_count": sum(r["split"] == "calibration" for r in retained),
        "old_split_exclusions": len(old_exclusions),
        "RGB_candidates_checked": len(inspected), "duplicate_exclusions": len(rejected),
        "selected_actions": dict(Counter(r["action"] for r in samples)),
        "selected_objects": dict(Counter(r["object"] for r in samples)),
        "GT_pixels_read": False, "GT_header_crc_sha_only": True,
        "observations_sha256": sha((out / "observations.jsonl").read_bytes()),
        "samples_sha256": sha((out / "samples.jsonl").read_bytes()),
        "new_network_body_bytes": budget["bytes"], "wall_s": time.monotonic()-start,
        "cpu_s": time.process_time()-cpu, "backend": "CPU", "placement_reason": "TASK_NOT_GPU_SUITABLE",
        "native_train_pixel_GT_available": False,
        "limit": "pHash exclusion is an image-level near-duplicate audit, not proof of scene/physical-instance independence."}
    assert budget["bytes"] <= 500_000_000 and summary["cpu_s"] <= 600
    dump(out / "data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
