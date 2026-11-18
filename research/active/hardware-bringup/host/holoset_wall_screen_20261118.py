"""HoloSet (Zenodo 7200131, CC BY 4.0): real indoor walking side-wall intake.

Run with the GPU venv's Python (CPU only): python holoset_wall_screen_20261118.py
<stage> <work_dir> [--sequences NAME ...]. Run fetch-depth, geometry and fetch-rgb,
then real_wall_holoset_visibility_20261118.py --root <work_dir> to create visibility.jsonl.
Finish with texture and sheet. Only geometry and sensor observations enter the texture cutoffs.
Source transforms follow Microsoft's HoloLens2ForCV StreamRecorderConverter.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import tarfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests

SEQUENCES = tuple(f"{site}-center-seq{i}" for site, n in
                  (("research", 4), ("student", 2), ("campus", 2)) for i in range(1, n + 1))
BASE = "https://zenodo.org/api/records/7200131/files"


def range_read(url, start, end):
    header = "bytes=-65536" if start is None else f"bytes={start}-{end}"
    with requests.get(url, headers={"Range": header}, stream=True, timeout=(15, 90)) as response:
        if response.status_code != 206:
            raise IOError(f"Expected partial response, got {response.status_code}")
        first = int(response.headers["Content-Range"].split()[1].split("-")[0])
        if start is not None and first != start:
            raise IOError(f"Wrong range from server, expected {start}, got {first}")
        return first, response.content


def read_index(seq):
    url = f"{BASE}/{seq}.zip/content"
    first, trailer = range_read(url, None, None)
    with zipfile.ZipFile(io.BytesIO(trailer)) as archive:
        return url, {
            Path(entry.filename).name: (first + entry.header_offset, entry)
            for entry in archive.infolist()
            if not entry.is_dir() and not entry.filename.startswith("__MACOSX")
        }


def get_member(folder, name, url, index):
    """Resume a compressed ZIP member and verify its uncompressed CRC."""
    start, entry = index[name]
    target = folder / name
    receipt = folder / f"{name}.crc32"
    if target.is_file() and target.stat().st_size == entry.file_size and receipt.is_file():
        if receipt.read_text() == f"{entry.CRC:08x}":
            return target
    _, header = range_read(url, start, start + 29)
    filename_size, extra_size = struct.unpack_from("<HH", header, 26)
    data_start = start + 30 + filename_size + extra_size
    packed = folder / (name + ".download")
    done = packed.stat().st_size if packed.exists() else 0
    if done < entry.compress_size:
        requested_start = data_start + done
        requested_end = data_start + entry.compress_size - 1
        with requests.get(url, headers={"Range": f"bytes={requested_start}-{requested_end}"},
                          stream=True, timeout=(15, 90)) as r:
            if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {requested_start}-"):
                raise IOError(f"ZIP member range mismatch: {name}")
            with packed.open("ab") as out:
                for block in r.iter_content(2 << 20):
                    out.write(block)
    if packed.stat().st_size != entry.compress_size:
        raise IOError(f"Incomplete ZIP member: {name}")
    decoder = zlib.decompressobj(-15) if entry.compress_type == zipfile.ZIP_DEFLATED else None
    if entry.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        raise ValueError(f"Unsupported ZIP method for {name}")
    tmp = folder / (name + ".part")
    crc = 0
    with packed.open("rb") as stream, tmp.open("wb") as out:
        for block in iter(lambda: stream.read(1 << 20), b""):
            data = decoder.decompress(block) if decoder is not None else block
            crc = zlib.crc32(data, crc)
            out.write(data)
    if crc != entry.CRC or tmp.stat().st_size != entry.file_size:
        raise IOError(f"Incorrect ZIP checksum/size: {name}")
    tmp.replace(target)
    receipt.write_text(f"{crc:08x}")
    return target


def read_pv(folder):
    path = next(folder.glob("*_pv.txt"))
    with path.open() as stream:
        header = np.fromstring(stream.readline(), sep=",")
    timestamps = np.loadtxt(path, delimiter=",", usecols=0, skiprows=1, dtype=np.int64, converters={0: int})
    frames = np.loadtxt(path, delimiter=",", skiprows=1)
    return header, timestamps, frames


def fetch(folder, rgb):
    folder.mkdir(parents=True, exist_ok=True)
    url, index = read_index(folder.name)
    pose_file = "Depth Long Throw_rig2world.txt"
    names = (pose_file, "Depth Long Throw_extrinsics.txt", "Depth Long Throw_lut.bin",
             next(k for k in index if k.endswith("_pv.txt")))
    for name in names:
        get_member(folder, name, url, index)
    bundle = next(k for k in index if k.startswith("PV.tar" if rgb else "Depth Long Throw.tar"))
    output = folder / ("rgb" if rgb else "depth")
    output.mkdir(exist_ok=True)
    if rgb:
        header, pvts, _ = read_pv(folder)
        w, h = header[2:].astype(int)
        depth_ts = [int(line.split(",")[0]) for line in (folder / pose_file).read_text().splitlines()]
        wanted = set()
        for ts in depth_ts:
            nearest = pvts[np.argmin(np.abs(pvts - ts))]
            if abs(nearest - ts) <= 750000:
                wanted.add(nearest)
    member = get_member(folder, bundle, url, index)
    with tarfile.open(member, "r|*") as archive:
        for info in archive:
            src = Path(info.name)
            if not info.isfile() or not src.stem.isdigit():
                continue
            if rgb and (src.suffix != ".bytes" or int(src.stem) not in wanted):
                continue
            if not rgb and src.suffix != ".pgm":
                continue
            data = archive.extractfile(info).read()
            if rgb:
                bgra = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
                if not cv2.imwrite(str(output / f"{src.stem}.png"), bgra[:, :, :3], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise IOError(f"Could not write RGB frame {src.stem}")
            else:
                (output / src.name).write_bytes(data)


def local_floor(points, ok, lateral, forward, relative_y):
    # Horizontal measured patches near the walker, stable across a nonzero ground area.
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normal = np.cross(dx, dy)
    norms = np.linalg.norm(normal, axis=2)
    flat = np.zeros(ok.shape, bool)
    flat[1:-1, 1:-1] = ((abs(normal[:, :, 1]) >= .9659 * norms) & (norms > 1e-7)
                         & (np.linalg.norm(dx, axis=2) < .17) & (np.linalg.norm(dy, axis=2) < .17))
    candidates = (flat & ok & (abs(lateral) <= 1.2) & (forward >= 0) & (forward <= 2.5)
                  & (relative_y >= -2.2) & (relative_y <= -.8))
    hist, edges = np.histogram(relative_y[candidates], np.arange(-2.2, -.77, .04))
    k = hist.argmax()
    center = edges[k] + .02
    mask = candidates & (abs(relative_y - center) <= .05)
    if mask.sum() < 50 or np.ptp(lateral[mask]) < .25 or np.ptp(forward[mask]) < .25:
        return None, "no_local_floor"
    if hist[np.abs(edges[:-1] + .02 - center) >= .09].max(initial=0) > .5 * hist[k]:
        return None, "multiple_floor_levels"
    return float(np.median(relative_y[mask])), None


def geometry(folder):
    dep = np.loadtxt(folder / "Depth Long Throw_rig2world.txt", delimiter=",")
    dep_ts = np.array([int(x.split(",")[0]) for x in (folder / "Depth Long Throw_rig2world.txt").read_text().splitlines()])
    header, pv_ts, pv = read_pv(folder)
    cx, cy, w, h_rgb = header
    cam_to_rig = np.linalg.inv(np.loadtxt(folder / "Depth Long Throw_extrinsics.txt", delimiter=",").reshape(4, 4))
    rays = np.fromfile(folder / "Depth Long Throw_lut.bin", np.float32).reshape(288, 320, 3)[::2, ::2]
    pixels = folder / "side_pixels"
    pixels.mkdir(exist_ok=True)
    tags = Counter()
    with (folder / "geometry.jsonl").open("w", encoding="utf8") as out:
        for path in sorted((folder / "depth").glob("*.pgm")):
            ts = int(path.stem)
            i, j = np.argmin(abs(dep_ts - ts)), np.argmin(abs(pv_ts - ts))
            lo, hi = max(i - 2, 0), min(i + 2, len(dep) - 1)
            dt = (dep[hi, 0] - dep[lo, 0]) / 1e7
            speed = float(np.linalg.norm(dep[hi, [4, 12]] - dep[lo, [4, 12]]) / max(dt, .001))
            delta = (int(pv_ts[j]) - ts) / 1e7
            row = dict(seq=folder.name, depth_ts=ts, rgb_ts=int(pv_ts[j]), rgb_dt=delta, speed=speed)
            if dep_ts[i] != ts:
                row["tag"] = "no_depth_pose"
            elif abs(delta) > .075:
                row["tag"] = "no_rgb_match"
            elif speed < .30:
                row["tag"] = "slow"
            else:
                T = dep[i, 1:].reshape(4, 4) @ cam_to_rig
                P = pv[j, 3:].reshape(4, 4)
                fx, fy = pv[j, 1:3]
                depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[::2, ::2]
                points = (depth[:, :, None] * rays * .001) @ T[:3, :3].T + T[:3, 3]
                delta_xyz = points - P[:3, 3]
                heading = -P[:3, 2].copy()
                heading[1] = 0
                heading /= np.linalg.norm(heading)
                lateral = delta_xyz @ np.cross(heading, [0., 1., 0.])
                forward = delta_xyz @ heading
                relative_y = delta_xyz[:, :, 1]
                camera = delta_xyz @ P[:3, :3]
                optical_z = np.where(camera[:, :, 2] < -.05, camera[:, :, 2], -1.)
                u, v = cx - fx * camera[:, :, 0] / optical_z, cy + fy * camera[:, :, 1] / optical_z
                ok = ((depth > 0) & (camera[:, :, 2] < -.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h_rgb))
                floor, tag = local_floor(points, ok, lateral, forward, relative_y)
                if tag is None:
                    row["floor_y"] = float(P[1, 3] + floor)
                    height = relative_y - floor
                    zone = ok & (forward >= .3) & (forward <= 2.) & (height >= .6) & (height <= 1.9)
                    if np.any(zone & (abs(lateral) <= .25)):
                        tag = "body_pts"
                    else:
                        masks = [zone & (sg * lateral >= .35) & (sg * lateral <= 1.2) for sg in (-1, 1)]
                        present = [m for m in masks if len(np.unique(((height[m] - .6) / .1).astype(int))) >= 10]
                        row["side_pts"] = sum(int(m.sum()) for m in masks)
                        if present:
                            mask = np.logical_or.reduce(present)
                            uv = (v[mask].astype(int) * int(w) + u[mask].astype(int)).astype(np.int32)
                            row["pixel_file"] = f"{ts}.npz"
                            np.savez_compressed(pixels / row["pixel_file"], uv=uv)
                            tag = "side"
                        else:
                            tag = "none" if row["side_pts"] < 50 else "other"
                row["tag"] = tag
            tags[row["tag"]] += 1
            out.write(json.dumps(row) + "\n")
    print(folder.name, dict(tags), flush=True)

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def texture(root, sequences):
    rows = []
    for seq in sequences:
        folder = root / seq
        geometry_rows = [r for r in read_rows(folder / "geometry.jsonl") if r["tag"] in ("side", "none")]
        visibility = {r["depth_ts"]: r for r in read_rows(folder / "visibility.jsonl")}
        if {r["depth_ts"] for r in geometry_rows} != visibility.keys():
            raise ValueError(f"Visibility and geometry have different timestamps: {seq}")
        for r in geometry_rows:
            vis = visibility[r["depth_ts"]]
            r["distributed_coverage"] = vis["distributed_coverage"]
            r["repeated_body_voxels"] = vis["repeated_body_voxels"]
            image = cv2.imread(str(folder / "rgb" / f"{r['rgb_ts']}.png"))
            if image is None:
                raise FileNotFoundError(f"Missing matched PV frame: {seq} {r['rgb_ts']}")
            if r["tag"] == "side":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                blurred = cv2.GaussianBlur(gray, (0, 0), 1.)
                mag = np.hypot(cv2.Sobel(blurred, cv2.CV_32F, 1, 0), cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
                with np.load(folder / "side_pixels" / r["pixel_file"]) as data:
                    r["tex"] = float(np.median(mag.ravel()[data["uv"]]))
            rows.append(r)
    included = [r for r in rows if r["repeated_body_voxels"] < 3]
    cuts = np.percentile([r["tex"] for r in included if r["tag"] == "side"], [100/3, 200/3])
    for r in rows:
        if r["tag"] == "none":
            r["stratum"] = "none"
        else:
            r["stratum"] = "low" if r["tex"] <= cuts[0] else ("mid" if r["tex"] <= cuts[1] else "high")
    for name, group in (("cohort.jsonl", included), ("suspect.jsonl", [r for r in rows if r["repeated_body_voxels"] >= 3])):
        tmp = root / (name + ".part")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in group), encoding="utf8")
        tmp.replace(root / name)
    counts = {s: dict(Counter(r["stratum"] for r in included if r["seq"] == s)) for s in sequences}
    summary = dict(cutoffs=cuts.tolist(), counts=counts, included=len(included), possible_occupancy=len(rows) - len(included))
    (root / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


def sheet(root, sequences):
    candidates = [r for r in read_rows(root / "cohort.jsonl") if r["stratum"] == "low" and r["seq"] in sequences]
    rng = np.random.default_rng(0)
    selected = [candidates[i] for i in sorted(rng.choice(len(candidates), min(16, len(candidates)), replace=False))]
    tiles = []
    for index, r in enumerate(selected, 1):
        folder = root / r["seq"]
        im = cv2.imread(str(folder / "rgb" / f"{r['rgb_ts']}.png"))
        with np.load(folder / "side_pixels" / r["pixel_file"]) as data:
            v, u = np.unravel_index(data["uv"], im.shape[:2])
        # A translucent overlay keeps the material visible between adjacent measured points.
        mask = np.zeros(im.shape[:2], np.uint8)
        mask[v, u] = 1
        marked = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
        im[marked] = (.65 * im[marked] + .35 * np.array([255, 0, 255])).astype(np.uint8)
        tile = cv2.resize(im, (380, 214), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 22, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, f"{index} {r['seq']} {r['depth_ts']}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(root / "low16.jpg"), np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0,len(tiles),4)]))
    (root / "low16.json").write_text(json.dumps(selected, indent=1), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch-depth", "geometry", "fetch-rgb", "texture", "sheet"])
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=SEQUENCES)
    args = parser.parse_args()
    if args.stage == "texture":
        texture(args.work_dir, args.sequences)
    elif args.stage == "sheet":
        sheet(args.work_dir, args.sequences)
    else:
        for seq in args.sequences:
            folder = args.work_dir / seq
            if args.stage == "geometry":
                geometry(folder)
            else:
                fetch(folder, args.stage == "fetch-rgb")


if __name__ == "__main__":
    main()
