"""Count corridor cases using independent SceneFun3D FARO depth projections."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
from time import perf_counter
import zipfile
import zlib

import cv2
import numpy as np
import requests

from arkit_selective_heading_dev_20261116 import (
    HALF_WIDTH, MIN_SOLID_ANGLE, free_rays, in_corridor, make_cloud, samples,
)


SOURCE = "https://cvg-data.inf.ethz.ch/scenefun3d/v1/train"
MAX_BYTES = 410_000_000
SUBSAMPLE = 4
CATEGORIES = ("head_occupied_travel_empty", "travel_occupied", "other_clear", "unknown")


class Download:
    def __init__(self):
        self.bytes = 0
        self.lock = threading.Lock()
        self.deadline = perf_counter() + 480

    def request(self, session, url, start, end, total):
        if perf_counter() > self.deadline:
            raise TimeoutError("Task's network time limit reached")
        size = end - start + 1
        with self.lock:
            if self.bytes + size > MAX_BYTES:
                raise RuntimeError("Public data byte budget reached")
            self.bytes += size
        with session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                         stream=True, timeout=(8, 15)) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                raise ValueError(f"Invalid Range response {response.status_code} for {url}")
            data = response.raw.read(size + 1)
        if len(data) != size:
            raise ValueError(f"Truncated public ZIP range for {url}")
        return data


class Tail(io.BytesIO):
    def __init__(self, data, total):
        super().__init__(data)
        self.total = total
        self.offset = total - len(data)

    def seek(self, n, whence=0):
        if whence == 1:
            n = self.tell() + n
        elif whence == 2:
            n = self.total + n
        if n < self.offset:
            raise EOFError("ZIP directory is not in the requested tail")
        super().seek(n - self.offset)
        return self.tell()

    def tell(self):
        return self.offset + super().tell()


class Archive:
    def __init__(self, url, session, dl):
        self.url, self.session, self.dl = url, session, dl
        response = session.head(url, timeout=(8, 12))
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.tail_offset = max(0, self.size - (1 << 20))
        self.tail = self.dl.request(session, url, self.tail_offset, self.size - 1, self.size)
        with zipfile.ZipFile(Tail(self.tail, self.size)) as archive:
            self.index = {Path(member.filename).stem.split("_", 1)[-1]: member
                          for member in archive.infolist()
                          if member.filename.endswith((".png", ".pincam")) and not member.is_dir()}

    def read(self, key):
        member = self.index[key]
        start = member.header_offset
        end = min(self.size - 1, start + 255 + member.compress_size)
        if start >= self.tail_offset:
            data = self.tail[start - self.tail_offset:end + 1 - self.tail_offset]
        else:
            data = self.dl.request(self.session, self.url, start, end, self.size)
        if data[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        n_name, n_extra = struct.unpack_from("<HH", data, 26)
        offset = 30 + n_name + n_extra
        payload = data[offset:offset + member.compress_size]
        if len(payload) != member.compress_size:
            raise ValueError("ZIP member is longer than the chosen range")
        if member.compress_type == zipfile.ZIP_DEFLATED:
            payload = zlib.decompress(payload, -15)
        elif member.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Unsupported ZIP compression")
        if len(payload) != member.file_size or (zlib.crc32(payload) & 0xffffffff) != member.CRC:
            raise ValueError("ZIP member CRC/size verification failed")
        return payload


def angles(poses, times):
    forward = poses[:, :2, 2]
    norm = np.linalg.norm(forward, axis=1)
    heads = forward / np.maximum(norm[:, None], 1e-9)
    future = np.searchsorted(times, times + .8 - .025)
    eligible = np.zeros(len(times), dtype=bool)
    degrees = np.full(len(times), np.nan)
    travels = np.zeros_like(heads)
    for i, j in enumerate(future):
        if j >= len(times) or not .75 <= times[j] - times[i] <= .90 or norm[i] < .5:
            continue
        if np.max(np.diff(times[i:j + 1])) > .22:
            continue
        delta = poses[j, :2, 3] - poses[i, :2, 3]
        distance = np.linalg.norm(delta)
        if distance < .30 * (times[j] - times[i]):
            continue
        travels[i] = delta / distance
        degrees[i] = np.degrees(np.arccos(np.clip(heads[i] @ travels[i], -1., 1.)))
        eligible[i] = degrees[i] <= 35.
    return heads, travels, degrees, eligible


def select_anchors(times, windows, degrees):
    # A minimum spacing is applied before evenly sampling each pose stratum.
    buckets = [[], []]
    for i in sorted(windows):
        bucket = buckets[int(degrees[i] >= 7.5)]
        if not bucket or times[i] - times[bucket[-1]] >= .8 - 1e-7:
            bucket.append(i)
    chosen = []
    for bucket, count in zip(buckets, (6, 12)):
        if len(bucket) > count:
            bucket = [bucket[i] for i in np.linspace(0, len(bucket) - 1, count).round().astype(int)]
        chosen.extend(bucket)
    return sorted(chosen)


def make_view(depth, pose, intrinsics):
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[:depth.shape[0]:2, :depth.shape[1]:2]
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).astype(np.float32)
    weight = (4. / (fx * fy) / np.power((ray ** 2).sum(axis=-1), 1.5)).astype(np.float32)
    confidence = (depth > 0).astype(np.uint8) * 2
    return dict(d=depth, c=confidence, K=intrinsics, p=pose,
                cloud=make_cloud(depth, confidence, pose, ray, weight))


def decode_views(poses, refs, ks, keys, dim_indexes):
    points = {}
    for i in dim_indexes:
        key = keys[i]
        value, calib = refs.read(key), ks.read(key)
        k = np.fromstring(calib.decode("ascii"), sep=" ")
        img = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_UNCHANGED)
        if len(k) != 6 or img is None or img.dtype != np.uint16 or (int(k[1]), int(k[0])) != img.shape:
            raise ValueError(f"Geometry/intrinsics mismatch at {key}: {k}, {None if img is None else img.shape}")
        depth = img[::SUBSAMPLE, ::SUBSAMPLE].astype(np.float32) * .001
        points[i] = make_view(depth, poses[i], tuple(k[2:] / SUBSAMPLE))
        points[i].update(depth_hash=hashlib.sha256(value).hexdigest(),
                         intrinsics_hash=hashlib.sha256(calib).hexdigest())
    return points


def label(origin, direction, views):
    votes = np.array([in_corridor(v["cloud"], origin, direction, HALF_WIDTH)[0] for v in views])
    target = samples(origin, direction, HALF_WIDTH)
    free = np.logical_or.reduce([free_rays(target, v["d"], v["c"], v["p"], v["K"]) for v in views])
    label = np.full(2, -1, dtype=np.int8)
    label[(votes >= MIN_SOLID_ANGLE).sum(axis=0) >= 2] = 1
    label[(votes == 0).all(axis=0) & (free.mean(axis=1) >= .75)] = 0
    return label.tolist(), free.mean(axis=1).round(3).tolist()


def category_for(head, travel):
    if 1 in travel:
        return "travel_occupied"
    if travel == [0, 0]:
        return "head_occupied_travel_empty" if 1 in head else "other_clear"
    return "unknown"


def select_windows(times, eligible, candidates):
    ok = sorted(candidates)
    stamps = times[ok]
    out = {}
    for i in np.flatnonzero(eligible):
        if i not in candidates:
            continue
        pair = []
        for desired in (times[i] - .4, times[i] + .4):
            local = int(np.argmin(np.abs(stamps - desired)))
            if abs(stamps[local] - desired) > .075:
                break
            pair.append(ok[local])
        if len(pair) == 2 and pair[0] < i < pair[1]:
            out[i] = (pair[0], i, pair[1])
    return out


def capture(posefile, out_dir, dl):
    visit, video = posefile.stem.split("_", 1)
    task = dict(visit=visit, video=video)
    with np.load(posefile, allow_pickle=False) as data:
        times, poses = data["ts"], data["T_wc"]
        keys = [str(x) for x in data["timestamp_text"]]
    if len(keys) != len(times) or poses.shape != (len(keys), 4, 4) or not (np.diff(times) > 0).all():
        return dict(**task, status="INVALID_POSE")
    if np.median(poses[:, 2, 1]) > -.55:
        return dict(**task, status="UP_AXIS_NOT_SUPPORTED")
    heads, travels, degrees, eligible = angles(poses, times)
    try:
        with requests.Session() as session:
            url = f"{SOURCE}/{visit}/{video}"
            refs = Archive(url + "/hires_depth.zip", session, dl)
            intrinsics = Archive(url + "/hires_wide_intrinsics.zip", session, dl)
            matching = {i for i, key in enumerate(keys) if key in refs.index and key in intrinsics.index}
            windows = select_windows(times, eligible, matching)
            anchors = select_anchors(times, windows, degrees)
            needed = sorted({j for i in anchors for j in windows[i]})
            views = decode_views(poses, refs, intrinsics, keys, needed)
    except (requests.RequestException, TimeoutError, ValueError, OSError, RuntimeError) as exc:
        return dict(**task, status="SOURCE_FAILURE", error=str(exc))
    rows = []
    for i in anchors:
        observations = [views[j] for j in windows[i]]
        origin = poses[i, :3, 3]
        head, head_coverage = label(origin, heads[i], observations)
        travel, travel_coverage = label(origin, travels[i], observations)
        rows.append(dict(timestamp=keys[i], time=float(times[i]), direction_deg=round(float(degrees[i]), 3),
                         head=head, travel=travel, head_coverage=head_coverage,
                         travel_coverage=travel_coverage, category=category_for(head, travel),
                         reference_frames=[keys[j] for j in windows[i]]))
    if views:
        with (out_dir / f"{visit}_{video}_reference.npz").open("xb") as stream:
            np.savez_compressed(stream, ts=times[needed], T_wc=poses[needed],
                                depth_mm=np.rint(np.array([views[i]["d"] for i in needed]) * 1000).astype(np.uint16),
                                K=np.array([views[i]["K"] for i in needed]),
                                frame_keys=np.array([keys[i] for i in needed]))
    return dict(**task, status="COMPLETE", pose_frames=len(times), pose_eligible=int(eligible.sum()),
                eligible_ref_windows=len(windows), sampled=len(anchors), matched_frames=len(matching),
                references={keys[i]: {k: v for k, v in views[i].items() if k.endswith("_hash")} for i in needed},
                rows=rows)


def count_episodes(result, category):
    starts, last = [], -np.inf
    for r in result.get("rows", []):
        if r["category"] != category:
            last = -np.inf
        else:
            if r["time"] - last > 1.0:
                starts.append(r["timestamp"])
            last = r["time"]
    return starts


def summarize(output):
    categories = {}
    for category in CATEGORIES:
        per_visit = [dict(visit=r["visit"], starts=count_episodes(r, category)) for r in output]
        categories[category] = dict(frames=sum(sum(row["category"] == category for row in r.get("rows", []))
                                               for r in output),
                                    episodes=sum(len(r["starts"]) for r in per_visit),
                                    visits=sum(bool(r["starts"]) for r in per_visit), rows=per_visit)
    return categories


def recheck(pose_root, output_dir):
    """Reproduce the expanded reference check without fetching public files."""
    supplement_dir = output_dir / "supplement"
    results = json.loads((supplement_dir / "summary.json").read_text(encoding="utf-8"))["combined"]
    verified = 0
    for capture in results:
        source = f"{capture['visit']}_{capture['video']}"
        if not capture["rows"]:
            continue
        with np.load(pose_root / f"{source}.npz", allow_pickle=False) as raw_pose:
            poses, times = raw_pose["T_wc"], raw_pose["ts"]
            pose_key = {key: i for i, key in enumerate(raw_pose["timestamp_text"])}
        heads, travels, _, _ = angles(poses, times)
        filename = output_dir / f"{source}_reference.npz"
        if not filename.exists():
            filename = supplement_dir / filename.name
        with np.load(filename, allow_pickle=False) as reference:
            stamps, view_poses = reference["ts"], reference["T_wc"]
            view_keys = list(reference["frame_keys"])
            images = reference["depth_mm"].astype(np.float32) * .001
            views = {key: make_view(depth, pose, intrinsics) for key, depth, pose, intrinsics
                     in zip(view_keys, images, view_poses, reference["K"])}
        for row in capture["rows"]:
            i = pose_key[row["timestamp"]]
            origin = poses[i, :3, 3]
            base_keys = set(row["reference_frames"])
            base_views = [views[key] for key in row["reference_frames"]]
            original = [label(origin, d, base_views)[0] for d in (heads[i], travels[i])]
            if original != [row["head"], row["travel"]] or category_for(*original) != row["category"]:
                raise ValueError(f"Cached reference changed the original label for {source}/{row['timestamp']}")
            verified += 1
            nearby = (np.abs(stamps - times[i]) <= 2.) & (np.linalg.norm(view_poses[:, :3, 3] - origin, axis=1) <= 1.)
            broader = [views[key] for key, keep in zip(view_keys, nearby) if keep or key in base_keys]
            head, travel = (label(origin, direction, broader) for direction in (heads[i], travels[i]))
            row.update(category=category_for(head[0], travel[0]), head_broader=head[0],
                       travel_broader=travel[0], head_coverage_broader=head[1],
                       travel_coverage_broader=travel[1], views=len(broader))
    saved = json.loads((output_dir / "reference_coverage_check.json").read_text(encoding="utf-8"))
    keys = ("category", "head_broader", "travel_broader", "head_coverage_broader", "travel_coverage_broader", "views")
    for capture, expected in zip(results, saved["visits"], strict=True):
        if capture["visit"] != expected["visit"]:
            raise ValueError("Cached reference audit changed visit identity")
        for row, wanted in zip(capture["rows"], expected["rows"], strict=True):
            if row["timestamp"] != wanted["timestamp"] or any(row[k] != wanted[k] for k in keys):
                raise ValueError(f"Cached reference audit changed the expanded result at {row['timestamp']}")
    if verified != saved["original_labels_verified"]:
        raise ValueError("Cached reference audit changed the number of checked labels")
    print(f"Verified {verified} original and expanded labels")
    return summarize(results)


def run_screen(args):
    started = perf_counter()
    choices = []
    for name in sorted(args.pose_root.glob("*.npz")):
        with np.load(name, allow_pickle=False) as data:
            _, _, degree, eligible = angles(data["T_wc"], data["ts"])
        choices.append((int((eligible & (degree >= 7.5)).sum()), name))
    choices = sorted(choices, key=lambda item: (-item[0], item[1].stem))[:12]
    with (args.output / "source_selection.json").open("x", encoding="utf-8") as stream:
        json.dump([{"source": p.stem, "angled_pose_frames": n} for n, p in choices], stream, indent=2)
    download = Download()
    with ThreadPoolExecutor(max_workers=8) as pool:
        output = list(pool.map(lambda entry: capture(entry[1], args.output, download), choices))
    categories = summarize(output)
    report = dict(reserved_range_bytes=download.bytes, wall_seconds=round(perf_counter() - started, 3),
                  results=output, categories=categories)
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for r in output:
        print(r["visit"], r["video"], r["status"], r.get("sampled"), r.get("error", ""))
    print(download.bytes, report["wall_seconds"])
    return categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recheck", action="store_true", help="verify cached expanded reference labels offline")
    args = parser.parse_args()
    categories = recheck(args.pose_root, args.output) if args.recheck else run_screen(args)
    for key, result in categories.items():
        print(key, result["frames"], result["episodes"], result["visits"])


if __name__ == "__main__":
    main()
