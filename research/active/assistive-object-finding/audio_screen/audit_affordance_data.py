"""Read-only public AGD20K ZIP range audit; no model/GPU execution.

Retains the central directory, six deterministic test image/GT pairs and
download receipts. Requires HTTP 206; never falls back to the 6.7 GB body.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import html
import io
import json
import re
import struct
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

PUBLIC_URL = "https://drive.google.com/file/d/1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk/view"
FILE_ID = "1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk"
SIZE = 6702534286
LIMIT = 150_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Ranges(io.RawIOBase):
    def __init__(self, root, deadline):
        self.root = root
        self.deadline = deadline
        self.started_cpu = time.process_time()
        self.pos = 0
        self.cache = []
        self.downloaded = 0
        self.receipts = []
        self.session = requests.Session()
        initial = root / "zip_tail.bin"
        if initial.exists():
            data = initial.read_bytes()
            self.cache.append((SIZE-len(data), SIZE, data))
            self.downloaded += len(data)
            self.receipts.append({"public_url": PUBLIC_URL, "range": f"bytes={SIZE-len(data)}-{SIZE-1}",
                                  "bytes": len(data), "sha256": digest(data), "path": str(initial.name),
                                  "status": 206, "kind": "same-task initial range probe"})
        for path in sorted(root.glob("range_*.bin")):
            lo, hi = map(int, path.stem.split("_")[1:])
            data = path.read_bytes()
            if len(data) != hi-lo+1:
                raise ValueError("existing range cache has wrong length")
            self.cache.append((lo, hi+1, data))
            self.downloaded += len(data)
            self.receipts.append({"public_url": PUBLIC_URL, "range": f"bytes={lo}-{hi}",
                                  "bytes": len(data), "sha256": digest(data), "path": path.name,
                                  "status": 206, "kind": "same-task range cache"})
        response = self.session.get("https://drive.google.com/uc", params={
            "id": FILE_ID, "export": "download"}, timeout=20)
        response.raise_for_status()
        self.downloaded += len(response.content)
        form = re.search(r'<form[^>]+action="([^"]+)"', response.text)
        if not form:
            raise ValueError("public download form not found; no fallback")
        self.action = html.unescape(form.group(1))
        self.params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', response.text))

    def check(self):
        if dt.datetime.now(dt.timezone.utc) >= self.deadline:
            raise TimeoutError("audit wall deadline reached")
        if time.process_time()-self.started_cpu >= 600:
            raise TimeoutError("audit 600 CPU-second budget reached")

    def get(self, start, end):
        self.check()
        end = min(end, SIZE-1)
        for lo, hi, data in self.cache:
            if lo <= start and end < hi:
                return data[start-lo:end-lo+1]
        length = end-start+1
        if length < 0 or self.downloaded+length > LIMIT:
            raise RuntimeError("range would exceed 150 MB task budget")
        response = self.session.get(self.action, params=self.params,
                                    headers={"Range": f"bytes={start}-{end}"},
                                    stream=True, timeout=(15, 40))
        try:
            expected = f"bytes {start}-{end}/{SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"exact range unsupported: HTTP {response.status_code}; no full-body fallback")
            parts = []
            actual = 0
            for part in response.iter_content(256*1024):
                actual += len(part)
                self.downloaded += len(part)
                self.check()
                if actual > length or self.downloaded > LIMIT:
                    raise RuntimeError("response exceeds exact range or total budget")
                parts.append(part)
            data = b"".join(parts)
            if len(data) != length:
                raise ValueError("range response length mismatch")
            path = self.root/f"range_{start}_{end}.bin"
            path.write_bytes(data)
            self.cache.append((start, end+1, data))
            self.receipts.append({"public_url": PUBLIC_URL, "range": f"bytes={start}-{end}",
                                  "bytes": len(data), "sha256": digest(data), "path": path.name,
                                  "status": response.status_code})
            (self.root/"range_receipts.json").write_text(json.dumps({
                "public_url": PUBLIC_URL, "archive_size": SIZE, "new_task_body_bytes": self.downloaded,
                "budget_bytes": LIMIT, "requests": self.receipts}, indent=2), encoding="utf-8")
            return data
        finally:
            response.close()

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else SIZE+offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = SIZE-self.pos if size < 0 else size
        data = self.get(self.pos, min(SIZE-1, self.pos+size-1)) if size else b""
        self.pos += len(data)
        return data

    def seekable(self):
        return True

    def readable(self):
        return True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", default="2026-10-15T06:15:00+00:00")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    remote = Ranges(out, dt.datetime.fromisoformat(args.deadline_utc))
    started = time.monotonic()
    try:
        with zipfile.ZipFile(remote) as archive:
            infos = archive.infolist()
            names = [x.filename for x in infos]
            write_json(out/"zip_entries.json", [{"name": x.filename, "compressed_size": x.compress_size,
                                                "size": x.file_size, "crc32": f"{x.CRC:08x}",
                                                "offset": x.header_offset} for x in infos])
            print(json.dumps({"entries": len(infos), "bytes_so_far": remote.downloaded,
                              "first_paths": names[:12]}, ensure_ascii=True), flush=True)
            if args.index_only:
                return
            image_map = {}
            gt_map = {}
            split_counts = Counter()
            for info in infos:
                parts = info.filename.split("/")
                if info.is_dir() or "__MACOSX" in parts:
                    continue
                for divide in ("Seen", "Unseen"):
                    if divide not in parts:
                        continue
                    di = parts.index(divide)
                    if len(parts) > di+1:
                        split_counts[f"{divide}/{parts[di+1]}"] += 1
                    if "testset" not in parts:
                        continue
                    for kind, mapping in (("egocentric", image_map), ("GT", gt_map)):
                        if kind in parts:
                            ki = parts.index(kind)
                            if len(parts) == ki+4:
                                mapping[(divide, parts[ki+1], parts[ki+2], Path(parts[-1]).stem)] = info
            pairs = sorted(set(image_map) & set(gt_map))
            counts = Counter((key[0], key[1], key[2]) for key in pairs)
            write_json(out/"split_and_pair_index.json", {"split_file_counts": dict(split_counts),
                       "test_pairs": len(pairs), "groups": [{"divide": k[0], "action": k[1], "object": k[2], "pairs": v}
                                                            for k, v in sorted(counts.items())]})
            requested = [("hold", "cup"), ("drink_with", "cup"), ("pour", "cup"),
                         ("open", "refrigerator"), ("open", "microwave"), ("hold", "knife"),
                         ("cut_with", "knife"), ("hold", "hammer"), ("hit", "hammer"),
                         ("hold", "scissors"), ("cut_with", "scissors"), ("hold", "bottle")]
            selected = []
            for action, obj in requested:
                group = [key for key in pairs if key[:3] == ("Seen", action, obj)]
                if group:
                    selected.append(group[0])
            for key in pairs:
                if len(selected) >= args.max_pairs:
                    break
                if key[0] == "Seen" and key[:3] not in [x[:3] for x in selected]:
                    selected.append(key)
            selected = selected[:args.max_pairs]
            if len(selected) != args.max_pairs:
                raise ValueError(f"expected {args.max_pairs} deterministic pairs, found {len(selected)}")
            write_json(out/"selection.json", {"rule": "first lexicographic Seen test pair per predeclared action/object group; fill with first other groups; no visual/GT-content selection",
                                               "requested": requested, "selected": selected})
            sample_dir = out/"samples"
            sample_dir.mkdir(exist_ok=True)
            tiles, records = [], []
            for index, key in enumerate(selected):
                remote.check()
                image_info, gt_info = image_map[key], gt_map[key]
                payloads = []
                for info in (image_info, gt_info):
                    # Prefetch a complete compressed member so small ZipFile reads hit cache.
                    prefetched_end = min(SIZE-1, info.header_offset+30+len(info.filename.encode())+
                                         len(info.extra)+info.compress_size+4096)
                    remote.get(info.header_offset, prefetched_end)
                    data = archive.read(info)
                    path = sample_dir/f"{index:02d}_{key[1]}_{key[2]}_{'GT.png' if info is gt_info else 'image'+Path(info.filename).suffix}"
                    path.write_bytes(data)
                    payloads.append((path, data))
                im = Image.open(io.BytesIO(payloads[0][1])); im.load()
                gt = Image.open(io.BytesIO(payloads[1][1])); gt.load()
                gray = gt.convert("L")
                histogram = gray.histogram()
                values = [i for i, n in enumerate(histogram) if n]
                record = {"key": key, "image_member": image_info.filename, "GT_member": gt_info.filename,
                          "image_size": im.size, "GT_size": gt.size, "GT_mode": gt.mode,
                          "GT_unique_values": values, "GT_nonzero_pixels": sum(histogram[1:]),
                          "GT_intermediate_1_to_254_pixels": sum(histogram[1:255]),
                          "GT_pixels": gray.width*gray.height,
                          "image_sha256": digest(payloads[0][1]), "GT_sha256": digest(payloads[1][1])}
                records.append(record)
                rgb = im.convert("RGB")
                resized_gt = gray.resize(rgb.size, Image.Resampling.BILINEAR)
                tint = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
                tint.putalpha(resized_gt.point(lambda x: int(x*0.65)))
                overlay = Image.alpha_composite(rgb.convert("RGBA"), tint).convert("RGB")
                tile = Image.new("RGB", (480, 260), "white")
                draw = ImageDraw.Draw(tile)
                draw.text((5, 4), f"{index:02d} {key[1]} / {key[2]}", fill="black")
                for x, pic in ((0, rgb), (240, overlay)):
                    thumb = ImageOps.contain(pic, (236, 205))
                    tile.paste(thumb, (x+(240-thumb.width)//2, 25+(205-thumb.height)//2))
                draw.text((5, 235), f"image {im.size} GT {gt.size}; {len(values)} gray values", fill="black")
                tiles.append(tile)
                print(json.dumps({"sample": index, "action": key[1], "object": key[2],
                                  "gray_values": len(values), "bytes_so_far": remote.downloaded}), flush=True)
            montage = Image.new("RGB", (1920, ((len(tiles)+3)//4)*260), "white")
            for index, tile in enumerate(tiles):
                montage.paste(tile, ((index%4)*480, (index//4)*260))
            montage.save(out/"image_GT_overlay_montage.jpg", quality=92)
            summary = {"status": "SUCCESS", "archive_size_bytes": SIZE, "archive_entries": len(infos),
                       "new_task_downloaded_body_bytes_including_initial_probe": remote.downloaded,
                       "cpu_s": time.process_time()-remote.started_cpu, "wall_s": time.monotonic()-started,
                       "test_pairs_indexed": len(pairs), "split_file_counts": dict(split_counts),
                       "samples": records, "limits": ["Data audit only; no training, model inference or GPU use.",
                         "Continuous GT heatmaps are not automatically physical contact/safe grasp labels.",
                         "Seen public test labels used only for availability/format inspection, not model selection."]}
            write_json(out/"summary.json", summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='samples'}, indent=2), flush=True)
    finally:
        remote.session.close()


if __name__ == "__main__":
    main()
