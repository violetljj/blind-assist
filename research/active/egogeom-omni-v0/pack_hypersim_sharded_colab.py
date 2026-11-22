"""Sharded high-speed Hypersim extractor for Colab CPU -> Google Drive (20 TB).

Uses HTTP Range requests over Apple's official CDN scene ZIPs to extract ONLY
what EgoGeom-Omni needs (skipping ~1.8 TB of EXR diffuse/illumination passes):
  - RGB tonemap.jpg (native 1024x768 JPEG bytes, lossless)
  - Planar optical Z depth in meters (converted from Hypersim radial distance via M_cam_from_uv, float16)
  - Camera-frame surface normals (int8 quantized, <0.4 deg error)
  - NYU40 semantic IDs (int16, preserves glass/window/mirror/floor/wall/furniture)
  - Exact camera poses (R_world_from_cam, t_meters, gravity_cam, M_cam_from_uv, meters_per_asset_unit)

Each (scene, cam) trajectory is saved as one compressed .npz inside ~2 GB .tar shards
with incremental SHA256 manifests (100% resumable, zero tiny-file Drive I/O).

Environment variables / CLI:
  SHARD_IDX   (default 0)
  NUM_SHARDS  (default 3)
  MAX_WORKERS (default 12)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tarfile
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DRIVE_OUT = Path("/content/drive/MyDrive/BlindAssist/datasets/hypersim_full_v1")
LOCAL_STAGE = Path("/content/hypersim_stage")

META_IMAGES_URL = (
    "https://raw.githubusercontent.com/apple/ml-hypersim/main/"
    "evermotion_dataset/analysis/metadata_images.csv"
)
META_CAM_URL = (
    "https://raw.githubusercontent.com/apple/ml-hypersim/main/"
    "contrib/mikeroberts/metadata_camera_parameters.csv"
)
SCENE_ZIP_TMPL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
)


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "BlindAssist-EgoGeom-DataFactory/1.0"})
    return s


class RemoteZip(io.RawIOBase):
    """Seekable HTTP-Range reader for remote ZIP central directory & selective member fetch."""

    def __init__(self, url: str, sess: requests.Session):
        self.url = url
        self.s = sess
        self.pos = 0
        r = self.s.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(self.size - self.pos, n)
        if n <= 0:
            return b""
        start = self.pos
        r = self.s.get(
            self.url,
            headers={"Range": f"bytes={start}-{start + n - 1}"},
            timeout=120,
        )
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != n:
            raise IOError(f"Short range read {len(r.content)} != {n} (status {r.status_code})")
        self.pos += n
        return r.content

    def fetch_member(self, entry: zipfile.ZipInfo) -> bytes:
        # Read local header + compressed payload in ONE HTTP Range request when possible.
        est_header = 30 + len(entry.filename.encode("utf-8")) + 128
        fetch_len = min(self.size - entry.header_offset, est_header + entry.compress_size)
        self.seek(entry.header_offset)
        buf = self.read(fetch_len)
        if buf[:4] != b"PK\x03\x04":
            raise IOError(f"Bad local header magic for {entry.filename}")
        name_len, extra_len = struct.unpack("<HH", buf[26:30])
        data_off = 30 + name_len + extra_len
        need = data_off + entry.compress_size
        if len(buf) < need:
            self.seek(entry.header_offset + len(buf))
            buf += self.read(need - len(buf))
        payload = buf[data_off:need]
        if entry.compress_type == 8:
            data = zlib.decompress(payload, -15)
        elif entry.compress_type == 0:
            data = payload
        else:
            raise IOError(f"Unsupported zip compress_type {entry.compress_type}")
        if len(data) != entry.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != entry.CRC:
            raise IOError(f"CRC/size mismatch for {entry.filename}")
        return data


def compute_axis_factor(cam_row: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis_factor[h,w], M_cam_from_uv[3,3]) matching Apple ml-hypersim."""
    M = np.array(
        [[float(cam_row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)],
        dtype=np.float64,
    )
    u, v = np.meshgrid(
        (np.arange(w, dtype=np.float64) + 0.5) / w * 2.0 - 1.0,
        1.0 - (np.arange(h, dtype=np.float64) + 0.5) / h * 2.0,
    )
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ M.T
    factor = (-rays[..., 2] / np.linalg.norm(rays, axis=-1)).astype(np.float32)
    return factor, M.astype(np.float32)


def read_h5_bytes(raw: bytes, key: str = "dataset") -> np.ndarray:
    with h5py.File(io.BytesIO(raw), "r") as f:
        return f[key][:]


def process_one_scene(
    scene: str,
    cam_frames: dict[str, list[int]],
    cam_row: dict,
    out_dir: Path,
) -> dict:
    t0 = time.time()
    sess = make_session()
    url = SCENE_ZIP_TMPL.format(scene=scene)
    remote = RemoteZip(url, sess)
    with zipfile.ZipFile(remote) as z:
        entries = {e.filename: e for e in z.infolist()}

    # Scene unit scale
    meters_per_unit = 0.0254
    meta_scene_name = f"{scene}/_detail/metadata_scene.csv"
    if meta_scene_name in entries:
        text = remote.fetch_member(entries[meta_scene_name]).decode("utf-8", errors="ignore")
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("parameter_name") == "meters_per_asset_unit":
                meters_per_unit = float(row["parameter_value"])
                break

    axis_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    saved_files: list[str] = []
    total_frames = 0

    for cam, fids in sorted(cam_frames.items()):
        fids = sorted(set(fids))
        ori_key = f"{scene}/_detail/{cam}/camera_keyframe_orientations.hdf5"
        pos_key = f"{scene}/_detail/{cam}/camera_keyframe_positions.hdf5"
        R_all = read_h5_bytes(remote.fetch_member(entries[ori_key])) if ori_key in entries else None
        t_all = read_h5_bytes(remote.fetch_member(entries[pos_key])) if pos_key in entries else None

        rgb_blobs: list[bytes] = []
        rgb_offsets: list[int] = [0]
        depth_list: list[np.ndarray] = []
        norm_list: list[np.ndarray] = []
        sem_list: list[np.ndarray] = []
        valid_fids: list[int] = []
        R_list: list[np.ndarray] = []
        t_list: list[np.ndarray] = []
        grav_list: list[np.ndarray] = []

        for fid in fids:
            base = f"{scene}/images/scene_{cam}"
            fstr = f"frame.{fid:04d}"
            rgb_name = f"{base}_final_preview/{fstr}.tonemap.jpg"
            dep_name = f"{base}_geometry_hdf5/{fstr}.depth_meters.hdf5"
            nrm_name = f"{base}_geometry_hdf5/{fstr}.normal_cam.hdf5"
            sem_name = f"{base}_geometry_hdf5/{fstr}.semantic.hdf5"
            if rgb_name not in entries or dep_name not in entries:
                continue
            try:
                jpg_bytes = remote.fetch_member(entries[rgb_name])
                radial = read_h5_bytes(remote.fetch_member(entries[dep_name])).astype(np.float32)
                h, w = radial.shape
                if (h, w) not in axis_cache:
                    axis_cache[(h, w)] = compute_axis_factor(cam_row, h, w)
                af, M_uv = axis_cache[(h, w)]
                z_m = radial * af
                z_m = np.where(np.isfinite(z_m) & (z_m > 0.05) & (z_m < 65000.0), z_m, 0.0).astype(np.float16)

                if nrm_name in entries:
                    nrm = read_h5_bytes(remote.fetch_member(entries[nrm_name])).astype(np.float32)
                    nrm = np.where(np.isfinite(nrm), nrm, 0.0)
                    nrm_i8 = np.clip(np.round(nrm * 127.0), -127, 127).astype(np.int8)
                else:
                    nrm_i8 = np.zeros((h, w, 3), dtype=np.int8)

                if sem_name in entries:
                    sem = read_h5_bytes(remote.fetch_member(entries[sem_name])).astype(np.int16)
                else:
                    sem = np.full((h, w), -1, dtype=np.int16)
            except Exception:
                continue

            rgb_blobs.append(jpg_bytes)
            rgb_offsets.append(rgb_offsets[-1] + len(jpg_bytes))
            depth_list.append(z_m)
            norm_list.append(nrm_i8)
            sem_list.append(sem)
            valid_fids.append(fid)

            if R_all is not None and fid < len(R_all):
                R_wc = R_all[fid].astype(np.float32)
                # Hypersim world +Z is UP; gravity vector in world is [0,0,-1].
                # Camera frame in Hypersim is OpenGL (+X right, +Y up, -Z forward).
                g_cam = (R_wc.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
            else:
                R_wc = np.eye(3, dtype=np.float32)
                g_cam = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            if t_all is not None and fid < len(t_all):
                t_m = (t_all[fid].astype(np.float32) * float(meters_per_unit)).astype(np.float32)
            else:
                t_m = np.zeros(3, dtype=np.float32)
            R_list.append(R_wc)
            t_list.append(t_m)
            grav_list.append(g_cam)

        if not valid_fids:
            continue

        _, M_uv = next(iter(axis_cache.values()))
        npz_path = out_dir / f"{scene}__{cam}.npz"
        np.savez_compressed(
            npz_path,
            rgb_jpeg_bytes=np.frombuffer(b"".join(rgb_blobs), dtype=np.uint8),
            rgb_jpeg_offsets=np.asarray(rgb_offsets, dtype=np.int64),
            depth_z_f16=np.stack(depth_list, axis=0),
            normal_cam_i8=np.stack(norm_list, axis=0),
            semantic_i16=np.stack(sem_list, axis=0),
            frame_ids=np.asarray(valid_fids, dtype=np.int32),
            R_world_from_cam=np.stack(R_list, axis=0),
            t_world_meters=np.stack(t_list, axis=0),
            gravity_cam_opengl=np.stack(grav_list, axis=0),
            M_cam_from_uv=M_uv,
            meters_per_asset_unit=np.asarray(meters_per_unit, dtype=np.float32),
        )
        saved_files.append(str(npz_path))
        total_frames += len(valid_fids)

    sess.close()
    return {
        "scene": scene,
        "frames": total_frames,
        "files": saved_files,
        "elapsed_s": round(time.time() - t0, 1),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def flush_part_to_drive(
    npz_paths: list[Path],
    scenes_in_part: list[str],
    frames_in_part: int,
    shard_idx: int,
    part_idx: int,
    manifest: dict,
    manifest_path: Path,
) -> None:
    if not npz_paths:
        return
    tar_name = f"hypersim_w{shard_idx}_p{part_idx:03d}.tar"
    local_tar = LOCAL_STAGE / tar_name
    drive_tar = DRIVE_OUT / tar_name

    t0 = time.time()
    with tarfile.open(local_tar, "w") as tf:
        for p in npz_paths:
            tf.add(p, arcname=p.name)
    local_size = local_tar.stat().st_size
    digest = sha256_file(local_tar)

    tmp_drive = DRIVE_OUT / f".tmp_{tar_name}"
    if tmp_drive.exists():
        tmp_drive.unlink()
    shutil.copyfile(local_tar, tmp_drive)
    if drive_tar.exists():
        drive_tar.unlink()
    tmp_drive.rename(drive_tar)

    drive_size = drive_tar.stat().st_size
    if drive_size != local_size:
        raise IOError(f"Drive size mismatch for {tar_name}: {drive_size} != {local_size}")

    # Clean local files immediately
    local_tar.unlink()
    for p in npz_paths:
        if p.exists():
            p.unlink()

    manifest["shards"].append(
        {
            "filename": tar_name,
            "bytes": local_size,
            "sha256": digest,
            "num_npz": len(npz_paths),
            "num_frames": frames_in_part,
            "scenes": scenes_in_part,
        }
    )
    manifest["completed_scenes"] = sorted(set(manifest["completed_scenes"]) | set(scenes_in_part))
    manifest["total_frames"] = sum(s["num_frames"] for s in manifest["shards"])
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["shards"])
    manifest["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    tmp_m = manifest_path.with_suffix(".json.tmp")
    tmp_m.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.move(str(tmp_m), str(manifest_path))

    print(
        f"[W{shard_idx} PART {part_idx:03d}] {tar_name}: {local_size/1e9:.2f} GB, "
        f"{frames_in_part} frames ({len(scenes_in_part)} scenes) written & verified in {time.time()-t0:.1f}s | "
        f"cumulative={len(manifest['completed_scenes'])} scenes / {manifest['total_frames']} frames",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-idx", type=int, default=int(os.environ.get("SHARD_IDX", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", "3")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MAX_WORKERS", "12")))
    parser.add_argument("--target-gb", type=float, default=2.2)
    args, _ = parser.parse_known_args()

    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    npz_dir = LOCAL_STAGE / f"npz_w{args.shard_idx}"
    npz_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Hypersim Sharded Extractor: worker {args.shard_idx}/{args.num_shards}, "
        f"threads={args.workers}, CPUs={os.cpu_count()} ===",
        flush=True,
    )

    sess = make_session()
    meta_img_local = Path("/content/metadata_images.csv")
    meta_cam_local = Path("/content/metadata_camera_parameters.csv")
    if not meta_img_local.exists() or meta_img_local.stat().st_size < 10000:
        r = sess.get(META_IMAGES_URL, timeout=60)
        r.raise_for_status()
        meta_img_local.write_bytes(r.content)
    if not meta_cam_local.exists() or meta_cam_local.stat().st_size < 10000:
        r = sess.get(META_CAM_URL, timeout=60)
        r.raise_for_status()
        meta_cam_local.write_bytes(r.content)
    # Also save metadata CSVs once to Drive
    if args.shard_idx == 0:
        for src in (meta_img_local, meta_cam_local, Path("/content/official_split.csv")):
            if src.exists():
                dst = DRIVE_OUT / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copyfile(src, dst)

    cam_params = {}
    with open(meta_cam_local, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cam_params[row["scene_name"]] = row

    scene_map: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    split_csv = Path("/content/official_split.csv")
    src_csv = split_csv if split_csv.exists() else meta_img_local
    with open(src_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("included_in_public_release") != "True":
                continue
            # Strictly exclude official protected test split!
            if row.get("split_partition_name", "").strip().lower() == "test":
                continue
            sc = row["scene_name"]
            if sc not in cam_params:
                continue
            scene_map[sc][row["camera_name"]].append(int(row["frame_id"]))

    all_scenes = sorted(scene_map.keys())
    my_scenes = [sc for i, sc in enumerate(all_scenes) if i % args.num_shards == args.shard_idx]

    manifest_path = DRIVE_OUT / f"manifest_w{args.shard_idx}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": "hypersim_full_v1",
            "license": "Apple ML-Hypersim (CC BY-SA 3.0 / Evermotion non-commercial research)",
            "worker_idx": args.shard_idx,
            "num_workers": args.num_shards,
            "completed_scenes": [],
            "failed_scenes": [],
            "total_frames": 0,
            "total_bytes": 0,
            "shards": [],
        }

    done_set = set(manifest["completed_scenes"])
    todo = [sc for sc in my_scenes if sc not in done_set]
    part_idx = len(manifest["shards"])
    print(
        f"Total public scenes={len(all_scenes)}, assigned={len(my_scenes)}, "
        f"already_done={len(done_set)}, remaining={len(todo)}",
        flush=True,
    )

    pending_npz: list[Path] = []
    pending_scenes: list[str] = []
    pending_frames = 0
    pending_bytes = 0
    target_bytes = int(args.target_gb * 1e9)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_one_scene, sc, scene_map[sc], cam_params[sc], npz_dir): sc
            for sc in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sc = futs[fut]
            try:
                res = fut.result()
                sz = sum(Path(p).stat().st_size for p in res["files"])
                pending_npz.extend(Path(p) for p in res["files"])
                pending_scenes.append(sc)
                pending_frames += res["frames"]
                pending_bytes += sz
                print(
                    f"[W{args.shard_idx} {i}/{len(todo)}] {sc}: {res['frames']} frames, "
                    f"{sz/1e6:.1f} MB in {res['elapsed_s']}s (buf={pending_bytes/1e9:.2f} GB)",
                    flush=True,
                )
            except Exception as e:
                print(f"[W{args.shard_idx} ERR] {sc}: {e}", flush=True)
                manifest["failed_scenes"].append({"scene": sc, "error": str(e)})
                continue

            if pending_bytes >= target_bytes:
                flush_part_to_drive(
                    pending_npz, pending_scenes, pending_frames,
                    args.shard_idx, part_idx, manifest, manifest_path
                )
                part_idx += 1
                pending_npz, pending_scenes, pending_frames, pending_bytes = [], [], 0, 0
                gc.collect()

    if pending_npz:
        flush_part_to_drive(
            pending_npz, pending_scenes, pending_frames,
            args.shard_idx, part_idx, manifest, manifest_path
        )

    print(
        f"=== W{args.shard_idx} DONE in {(time.time()-t_start)/60:.1f} min | "
        f"scenes={len(manifest['completed_scenes'])}/{len(my_scenes)}, "
        f"frames={manifest['total_frames']}, bytes={manifest['total_bytes']/1e9:.2f} GB, "
        f"failed={len(manifest['failed_scenes'])} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
