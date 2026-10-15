"""Acquire public Development data and reproduce deterministic pilot selection.

CPU placement: TASK_NOT_GPU_SUITABLE (HTTP/archive/RLE metadata work).
MeViS is cross-domain evidence, not first-person assistive-use validation.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import time
import zipfile

import requests
from PIL import Image

BASE = "https://huggingface.co/datasets/FudanCVL/MeViSv2/resolve/main/valid_u/"
TAR_SIZE = 746655744
TAR_SHA256 = "bcb641213c9b6b09a73503e143f3aa57898320872bf1626207e68d6ed6464c21"
SEED = 20261015


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record_download(root, url, path, status, byte_range=None):
    row = {"url": url, "path": str(path.resolve()), "bytes": path.stat().st_size,
           "sha256": sha256(path), "http_status": status, "range": byte_range,
           "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (root / "download_receipts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def download(root, url, filename, byte_range=None):
    path = root / filename
    if path.exists():
        return path
    headers = {"Range": byte_range} if byte_range else {}
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                if byte_range:
                    assert r.status_code == 206, "Range response required; full body rejected"
                with path.open("wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
                record_download(root, url, path, r.status_code, byte_range)
                return path
        except requests.RequestException as e:
            print(json.dumps({"stage": "download_retry", "attempt": attempt,
                              "file": filename, "error": type(e).__name__}), flush=True)
    raise RuntimeError(f"Download failed: {filename}")


def acquire_refego_annotations(root):
    """Small HTTP ranges avoid a stalled single 19 MB S3 response.

    Annotation access is public. This function does not obtain Ego4D pixels,
    inspect credential values, accept a license, or bypass authorization.
    """
    url = "https://refegopublic.s3.amazonaws.com/RefEgoAnnotations2.zip"
    size, block = 19248412, 1048576
    archive = root / "RefEgoAnnotations2_ranged.zip"
    if not archive.exists():
        def fetch(i):
            lo, hi = i * block, min(size, (i + 1) * block) - 1
            path = download(root, url, f"refego_chunk_{i:02}.part", f"bytes={lo}-{hi}")
            assert path.stat().st_size == hi - lo + 1
            return i, path
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            parts = sorted(pool.map(fetch, range((size + block - 1) // block)))
        with archive.open("wb") as out:
            for _, part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, out)
        record_download(root, url, archive, 206, "19 contiguous HTTP ranges")
    assert archive.stat().st_size == size
    report = {"archive": str(archive.resolve()), "archive_sha256": sha256(archive),
              "version": "RefEgoAnnotations2 (official scaling-fix release)",
              "scope": "Public annotation metadata only; no test-model evaluation", "splits": {}}
    with zipfile.ZipFile(archive) as z:
        for split in ("train", "val", "test"):
            with z.open(f"RefEgoAnnotations/refego_{split}_RefCOCOstyle_fps2.json") as f:
                data = json.load(f)
            ann = data["annotations"]
            report["splits"][split] = {
                "images": len(data["images"]), "query_frame_annotations": len(ann),
                "clips": len({x["clip_id"] for x in ann}),
                "videos": len({x["video_id"] for x in ann}),
                "target_absent_query_frames": sum(not x["is_obj_in"] for x in ann),
                "same_type_multiple_query_frames": sum(x["is_multi"] == "multi" for x in ann),
                "is_multi_values": dict(collections.Counter(str(x["is_multi"]) for x in ann)),
                "annotation_keys": list(ann[0]), "first_annotation": ann[0],
                "first_image": data["images"][0],
                "image_dimensions": dict(collections.Counter(
                    f"{im['width']}x{im['height']}" for im in data["images"]))}
            del data, ann
    write_json(root / "refego_annotation_audit.json", report)
    return report


def rle_counts(rle):
    """Decode COCO's compressed counts without requiring a native extension."""
    s = rle["counts"]
    if isinstance(s, list):
        return s
    counts, p = [], 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 31) << (5 * k)
            more = bool(c & 32)
            k += 1
            if not more and c & 16:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_bbox(rle):
    if rle is None:
        return None
    height, width = rle["size"]
    counts = rle_counts(rle)
    assert sum(counts) == height * width and min(counts) >= 0
    pos, xmin, ymin, xmax, ymax = 0, width, height, -1, -1
    for i, length in enumerate(counts):
        if i % 2 and length:
            end = pos + length - 1
            x1, y1 = divmod(pos, height)
            x2, y2 = divmod(end, height)
            xmin, xmax = min(xmin, x1), max(xmax, x2)
            if x1 == x2:
                ymin, ymax = min(ymin, y1), max(ymax, y2)
            else:
                ymin, ymax = 0, height - 1
        pos += length
    return None if xmax < 0 else [xmin, ymin, xmax + 1, ymax + 1]


def selection(meta, masks, max_rows=576):
    """GT-stratified Development enrichment, fixed ordering and fixed seed.

    Pick six source-distinct expressions with visibility transitions, then six
    further source-distinct expressions with multiple annotated objects. Choose
    a contiguous 36-frame window with most present/absent transitions. Add up to
    four 24-frame explicit no-target expressions from the selected sources.
    """
    candidates = []
    for vid, v in sorted(meta.items()):
        seen = set()
        for eid, expr in sorted(v["expressions"].items(), key=lambda x: int(x[0])):
            ids = expr["anno_id"]
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
            aid = str(ids[0])
            boxes = [rle_bbox(rle) for rle in masks[aid]]
            assert len(boxes) == len(v["frames"])
            present = [b is not None for b in boxes]
            if not any(present):
                continue
            transitions = sum(a != b for a, b in zip(present, present[1:]))
            all_ids = {i for e in v["expressions"].values() for i in e["anno_id"]}
            candidates.append({"source_id": vid, "query_id": eid, "query": expr["exp"],
                               "annotation_id": aid, "length": len(boxes),
                               "transitions": transitions,
                               "present_frames": sum(present),
                               "annotated_objects": len(all_ids),
                               "presence": present})
    rng = random.Random(SEED)
    shuffled = sorted(candidates, key=lambda c: (c["source_id"], int(c["query_id"])))
    rng.shuffle(shuffled)
    rank = {(c["source_id"], c["query_id"]): i for i, c in enumerate(shuffled)}
    chosen, sources = [], set()
    for mode in ("visibility_transition", "multiple_annotated_objects"):
        pool = sorted(candidates, key=lambda c: (
            -c["transitions"] if mode == "visibility_transition" else -c["annotated_objects"],
            rank[(c["source_id"], c["query_id"])]))
        count = 0
        for c in pool:
            if c["source_id"] in sources:
                continue
            if mode == "visibility_transition" and not c["transitions"]:
                continue
            if mode == "multiple_annotated_objects" and c["annotated_objects"] < 2:
                continue
            c = dict(c)
            n = min(36, c["length"])
            def score(start):
                p = c["presence"][start:start + n]
                return (sum(a != b for a, b in zip(p, p[1:])),
                        min(sum(p), n - sum(p)), sum(p), -start)
            start = max(range(c["length"] - n + 1), key=score)
            c.update(start=start, stop=start+n, selection_reason=mode)
            chosen.append(c)
            sources.add(c["source_id"])
            count += 1
            if count == 6:
                break
    for vid in sorted(sources):
        negative = [(eid, e) for eid, e in meta[vid]["expressions"].items()
                    if not e["anno_id"]]
        if negative:
            eid, e = sorted(negative, key=lambda x: int(x[0]))[0]
            chosen.append({"source_id": vid, "query_id": eid, "query": e["exp"],
                           "annotation_id": None, "start": 0,
                           "stop": min(24, len(meta[vid]["frames"])),
                           "selection_reason": "explicit_no_target_expression"})
        if sum(c["selection_reason"] == "explicit_no_target_expression" for c in chosen) == 4:
            break
    assert len(chosen) <= 24 and sum(c["stop"] - c["start"] for c in chosen) <= max_rows <= 600
    return candidates, chosen


def extract_selected(tar_path, root, wanted):
    images = root / "images"
    n = 0
    with tarfile.open(tar_path, "r:") as t:
        for member in t:
            if member.name not in wanted:
                continue
            relative = Path(member.name)
            assert relative.parts[0] == "JPEGImages" and ".." not in relative.parts
            dest = images.joinpath(*relative.parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(member) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            n += 1
    assert n == len(wanted), (n, len(wanted))
    return images


def materialize(root):
    meta_path = root / "mevis_meta_expressions_v2.json"
    if not meta_path.exists() and (root / "mevis_meta_expressions_v2_fast.json").exists():
        shutil.copyfile(root / "mevis_meta_expressions_v2_fast.json", meta_path)
    mask_path = root / "mevis_mask_dict.json"
    if not mask_path.exists() and (root / "mevis_mask_dict_fast.json").exists():
        shutil.copyfile(root / "mevis_mask_dict_fast.json", mask_path)
    meta = json.loads(meta_path.read_bytes())["videos"]
    masks = json.loads(mask_path.read_bytes())
    candidates, chosen = selection(meta, masks)
    write_json(root / "candidate_selection.json", {"seed": SEED, "algorithm": selection.__doc__,
               "candidates": candidates, "selected": chosen})
    tar_path = root / "mevis_JPEGImages.tar"
    if not tar_path.exists():
        with tar_path.open("wb") as out:
            for part in ("mevis_JPEGImages_prefix64m.tar.part", "mevis_JPEGImages_tail.tar.part"):
                with (root / part).open("rb") as f:
                    shutil.copyfileobj(f, out)
    assert tar_path.stat().st_size == TAR_SIZE
    assert sha256(tar_path) == TAR_SHA256, "Official Hugging Face LFS object hash mismatch"
    wanted = {f"JPEGImages/{c['source_id']}/{meta[c['source_id']]['frames'][i]}.jpg"
              for c in chosen for i in range(c["start"], c["stop"])}
    image_root = extract_selected(tar_path, root, wanted)
    rows = []
    for c in chosen:
        vid = c["source_id"]
        for i in range(c["start"], c["stop"]):
            frame = meta[vid]["frames"][i]
            image = image_root / vid / f"{frame}.jpg"
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            aid = c["annotation_id"]
            rle = masks[aid][i] if aid is not None else None
            bbox = rle_bbox(rle)
            if rle is not None:
                assert rle["size"] == [height, width], (vid, frame, rle["size"], width, height)
            rows.append({"sequence_id": f"mevis_{vid}_q{c['query_id']}", "source_id": vid,
                         "frame_idx": int(frame), "timestamp_s": None,
                         "image_path": str(image.resolve()), "query": c["query"],
                         "gt_present": bbox is not None, "gt_bbox_xyxy": bbox,
                         "split": "development", "source_split": "valid_u", "width": width,
                         "height": height, "annotation_source": "official_MeViSv2_COCO_RLE",
                         "dataset": "MeViSv2", "annotation_id": aid,
                         "query_id": c["query_id"], "selection_reason": c["selection_reason"],
                         "image_sha256": sha256(image)})
    manifest = root / "samples.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"dataset": "MeViSv2", "source_split": "valid_u", "evidence_role": "Development",
               "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
               "source_image_archive_sha256": TAR_SHA256,
               "query_sequences": len(chosen), "source_videos": len({r["source_id"] for r in rows}),
               "labeled_query_frames": len(rows), "unique_images": len(wanted),
               "present": sum(r["gt_present"] for r in rows),
               "absent": sum(not r["gt_present"] for r in rows),
               "explicit_no_target_rows": sum(r["selection_reason"] == "explicit_no_target_expression" for r in rows),
               "timestamp_policy": "Unknown source FPS; timestamp_s=null; report annotated-frame delays only.",
               "bbox_convention": "pixel xyxy, lower inclusive and upper exclusive, derived from complete target masks",
               "selection": "GT-stratified enrichment, not natural-frequency population estimation",
               "backend": "CPU", "placement_reason": "TASK_NOT_GPU_SUITABLE"}
    write_json(root / "data_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--refego-annotations-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    assert "artifacts.local" in args.root.as_posix(), "Use canonical artifact path"
    root.mkdir(parents=True, exist_ok=True)
    if args.refego_annotations_only:
        report = acquire_refego_annotations(root)
        print(json.dumps({"archive": report["archive"],
                          "splits": {k: {n: v for n, v in s.items()
                                         if n not in ("first_annotation", "first_image")}
                                     for k, s in report["splits"].items()}}, indent=2), flush=True)
        return
    if not args.materialize_only:
        download(root, BASE + "meta_expressions_v2.json", "mevis_meta_expressions_v2.json")
        download(root, BASE + "mask_dict.json", "mevis_mask_dict.json")
        if not (root / "mevis_JPEGImages.tar").exists():
            download(root, BASE + "JPEGImages.tar", "mevis_JPEGImages_prefix64m.tar.part", "bytes=0-67108863")
            download(root, BASE + "JPEGImages.tar", "mevis_JPEGImages_tail.tar.part", "bytes=67108864-746655743")
    materialize(root)


if __name__ == "__main__":
    main()
