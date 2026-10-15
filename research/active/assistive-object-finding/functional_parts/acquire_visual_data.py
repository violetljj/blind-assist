"""Acquire RGB-selected fresh AGD groups; preserve previous fit/cal byte-for-byte."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import struct
import threading
import time
import requests
from PIL import Image
from acquire_data import Source, agd_index, dump, jsonl, sha
from acquire_rank_data import phash

SEED = "20261015-fp-visual-fresh"


def rank(value):
    return hashlib.sha256((SEED + str(value)).encode()).hexdigest()


class CountedSource(Source):
    """Count successful and failed streamed range bodies, including retries."""
    def get(self, lo, hi, offline=False):
        cached = self.cached(lo, hi)
        if cached is not None:
            return cached
        if offline:
            raise ValueError("Required metadata not cached")
        if self.form is None:
            raise RuntimeError("Connect first")
        url, params = self.form
        for attempt in range(3):
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Original acquisition deadline reached")
            with self.lock:
                # Reserve full response sizes before requests, including failed attempts.
                self.budget["reserved"] += hi - lo + 1
                if self.budget["reserved"] > 299_500_000:
                    raise RuntimeError("300 MB conservative request bound reached")
            record = {"lo": lo, "hi": hi, "attempt": attempt, "bytes": 0}
            try:
                with requests.get(url, params=params, headers={"Range": f"bytes={lo}-{hi}"},
                                  stream=True, timeout=(10, 25)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{self.size}":
                        raise ValueError("Exact range unsupported")
                    body = bytearray()
                    for block in response.iter_content(65536):
                        body.extend(block)
                        record["bytes"] += len(block)
                        with self.lock:
                            self.budget["bytes"] += len(block)
                        if len(body) > hi-lo+1 or time.monotonic() > self.deadline:
                            raise TimeoutError("Download bound reached")
                assert len(body) == hi-lo+1
                record["status"] = "SUCCESS"
                return bytes(body)
            except requests.RequestException as exc:
                record["status"] = type(exc).__name__
                if attempt == 2:
                    raise
            finally:
                with self.lock:
                    self.receipts.append(record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wall-seconds", type=int, default=900)
    p.add_argument("--inventory-only", action="store_true")
    a = p.parse_args()
    assert "artifacts.local" in a.root.as_posix() and 1 <= a.count <= 150
    out = a.root.resolve() / "data"
    out.mkdir(parents=True, exist_ok=True)
    start, cpu = time.monotonic(), time.process_time()
    budget = {"bytes": 0, "reserved": 0, "lock": threading.Lock()}
    source = CountedSource("AGD20K", a.audit.resolve(), out, start+a.wall_seconds, budget)
    status = "FAILED"
    try:
        groups, _ = agd_index(source)
        old = [json.loads(x) for x in (a.old / "data/samples.jsonl").read_text().splitlines()]
        old = [r for r in old if r["dataset"] == "AGD20K"]
        audit = json.loads((a.audit / "agd20k_audit/summary.json").read_text())["samples"]
        prior_paths = sorted((a.previous / "data/images").glob("*.jpg"))
        excluded = {r["source_image_id"] for r in old} | {r["key"][3] for r in audit} | {p.stem for p in prior_paths}
        remaining = {i: pairs for i, pairs in groups.items() if i not in excluded}
        ac, oc, ordered = Counter(), Counter(), []
        while remaining:
            options = [(ac[q["action"]], oc[q["object"]], rank(i+q["action"]), i, q)
                       for i, pairs in remaining.items() for q in pairs]
            _, _, _, identifier, pair = min(options, key=lambda x: x[:3])
            ordered.append({"source_image_id": identifier, **pair})
            ac[pair["action"]] += 1
            oc[pair["object"]] += 1
            del remaining[identifier]
        plan = {"seed": SEED, "target": a.count, "paired_groups": len(groups),
                "excluded_ids": sorted(excluded), "prior_cached_images": len(prior_paths),
                "available_groups": len(ordered), "ordered_candidates": ordered,
                "GT_pixels_read": False, "phash_hamming_exclusion_lte": 6}
        dump(out / "selection_plan.json", plan)
        if a.inventory_only:
            status = "INVENTORY_ONLY"
            print(json.dumps({k:v for k,v in plan.items() if k not in ("excluded_ids", "ordered_candidates")}))
            return
        for name in ("fit_samples.jsonl", "calibration_samples.jsonl", "fit_observations.jsonl", "calibration_observations.jsonl"):
            shutil.copyfile(a.previous / "data" / name, out / name)
        refpaths = [(r["source_image_id"], Path(r["image_path"])) for r in old]
        refpaths += [(p.stem, p) for p in prior_paths]
        refpaths += [("audit:"+p.name, p) for p in sorted((a.audit / "agd20k_audit/samples").glob("*_image.jpg"))]
        refs = [(i, sha(p.read_bytes()), (phash(p), phash(p, True))) for i,p in refpaths]
        jsonl(out / "excluded_rgb_fingerprints.jsonl", [{"id":i,"sha256":s,"phash64":[f"{h:016x}" for h in hs]} for i,s,hs in refs])
        network_start = time.monotonic()
        source.deadline = network_start + a.wall_seconds
        source.connect()
        images, gts = out / "images", out / "GT"
        images.mkdir(exist_ok=True)
        gts.mkdir(exist_ok=True)

        def rgb(row):
            path = images / (row["source_image_id"]+".jpg")
            if not path.exists():
                path.write_bytes(source.member(row["image_member"]))
            with Image.open(path) as im:
                im.load()
                width,height = im.size
            return path,sha(path.read_bytes()),(phash(path),phash(path,True)),width,height

        selected, inspected, rejected = [], [], []
        for offset in range(0, len(ordered), 8):
            batch = ordered[offset:offset+8]
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                values = list(pool.map(rgb, batch))
            for row,(path,digest,hs,width,height) in zip(batch,values):
                distances = [(min((h^q).bit_count() for h in hs for q in oldhs),i,digest==s) for i,s,oldhs in refs]
                nearest = min(distances)
                record = {**row,"image_path":str(path),"image_sha256":digest,"width":width,"height":height,
                          "phash64_original_flip":[f"{h:016x}" for h in hs],"nearest_reference_id":nearest[1],"nearest_hamming":nearest[0]}
                inspected.append(record)
                if nearest[0] <= 6 or any(d[2] for d in distances):
                    rejected.append(record)
                    continue
                selected.append(record)
                refs.append((row["source_image_id"],digest,hs))
                if len(selected) == a.count:
                    break
            if len(selected) == a.count:
                break
        assert selected, "No disjoint images remain"
        jsonl(out / "rgb_inspected.jsonl", inspected)
        jsonl(out / "duplicate_exclusions.jsonl", rejected)
        dump(out / "rgb_selection_seal.json", {"selected_ids":[r["source_image_id"] for r in selected],
             "selected_sha256":[r["image_sha256"] for r in selected],"GT_pixels_read":False,"selected_before_GT_download":True})
        jsonl(out / "observations_ready.jsonl", [{"sample_id":"agd_"+r["source_image_id"]+"__"+r["action"],"image_path":r["image_path"],"action":r["action"]} for r in selected])

        def gt(row):
            path = gts / (row["source_image_id"]+"__"+row["action"]+".png")
            if not path.exists():
                path.write_bytes(source.member(row["gt_member"]))
            body = path.read_bytes()
            assert body[:8] == b"\x89PNG\r\n\x1a\n" and body[12:16] == b"IHDR"
            assert struct.unpack(">II",body[16:24]) == (row["width"],row["height"])
            return path,sha(body)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            gt_values = list(pool.map(gt,selected))
        samples = [{**r,"sample_id":"agd_"+r["source_image_id"]+"__"+r["action"],"dataset":"AGD20K",
                    "split":"evaluation","evidence_role":"fresh_image_Development","gt_path":str(g),"gt_sha256":s,
                    "label_kind":"native_continuous_heatmap","primary":True,
                    "source_image_member":r["image_member"],"source_GT_member":r["gt_member"]} for r,(g,s) in zip(selected,gt_values)]
        jsonl(out / "samples.jsonl", samples)
        shutil.copyfile(out / "observations_ready.jsonl", out / "observations.jsonl")
        dump(out / "data_summary.json", {"count":len(samples),"fit_count":158,"calibration_count":79,
             "old_split_byte_identical":True,"excluded_ids":len(excluded),"prior_cached_RGB":len(prior_paths),
             "available_groups":len(ordered),"RGB_checked":len(inspected),"duplicates":len(rejected),
             "actions":dict(Counter(r["action"] for r in samples)),"objects":dict(Counter(r["object"] for r in samples)),
             "GT_pixels_read":False,"GT_header_crc_sha_only":True,
             "samples_sha256":sha((out/"samples.jsonl").read_bytes()),"observations_sha256":sha((out/"observations.jsonl").read_bytes())})
        status = "SUCCESS"
    finally:
        dump(out / ("inventory_receipt.json" if a.inventory_only else "acquisition_receipt.json"),
             {"status":status,"wall_s":time.monotonic()-start,"cpu_s":time.process_time()-cpu,
              "cpu_scope":"one Python process including all download threads; no child processes",
              "network_wall_s": time.monotonic()-network_start if "network_start" in locals() else 0,
              "network_received_body_bytes":budget["bytes"],"network_reserved_upper_bound_bytes":budget["reserved"]+500000,
              "network_counter_scope":"HTTP form and all iter_content chunks including retried ranges; protocol headers excluded",
              "wall_limit_s":a.wall_seconds,"download_limit_bytes":300000000,"backend":"CPU","placement_reason":"TASK_NOT_GPU_SUITABLE",
              "GT_pixels_read":False,"requests":source.receipts})
        print(status, flush=True)


if __name__ == "__main__":
    main()
