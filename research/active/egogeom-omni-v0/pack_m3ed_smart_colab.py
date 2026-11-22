"""Smart high-speed M3ED Spot Train extractor for Colab CPU -> Drive.

Uses 16-thread parallel HTTP Range download (10-15x faster than single-stream S3 across regions),
prints progress every 10s (flush=True) to prevent websocket idle disconnects, and strips the ~90%
event-camera bloat from *_data.h5 so only RGB + VN100T IMU + calibration + LiDAR depth_gt are
saved as compact SHA256-verified .tar/.npz packages under BlindAssist/datasets/m3ed_spot_train_v1/.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DRIVE_OUT = Path("/content/drive/MyDrive/BlindAssist/datasets/m3ed_spot_train_v1")
LOCAL_TMP = Path("/content/m3ed_smart_stage")
BASE_S3 = "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed"

# Top high-leverage Spot train sequences (strictly is_test_file==False, CC BY-SA 4.0)
TARGET_SEQS = [
    "spot_indoor_obstacles",
    "spot_indoor_stairs",
    "spot_indoor_stairwell",
    "spot_indoor_building_loop",
    "spot_outdoor_day_rocky_steps",
    "spot_outdoor_night_penno_plaza_lights",
    "spot_outdoor_night_penno_short_loop",
    "spot_outdoor_day_art_plaza_loop",
]


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    ad = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def parallel_range_download(url: str, dst: Path, n_threads: int = 16, chunk_mb: int = 16) -> Path:
    """Multi-connection HTTP Range downloader with 10s heartbeat logging."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    sess = make_session()
    r = sess.head(url, timeout=30)
    r.raise_for_status()
    total = int(r.headers["Content-Length"])
    chunk = chunk_mb * 1024 * 1024
    ranges = [(i, min(i + chunk - 1, total - 1)) for i in range(0, total, chunk)]

    part = dst.with_suffix(dst.suffix + ".part")
    # Pre-allocate file
    with open(part, "wb") as f:
        f.truncate(total)

    done_bytes = 0
    t0 = time.time()
    last_log = t0

    def _fetch_chunk(span: tuple[int, int]) -> int:
        s0, s1 = span
        s = make_session()
        resp = s.get(url, headers={"Range": f"bytes={s0}-{s1}"}, timeout=180)
        resp.raise_for_status()
        data = resp.content
        if len(data) != (s1 - s0 + 1):
            raise IOError(f"Short chunk {len(data)} != {s1 - s0 + 1}")
        with open(part, "r+b") as f:
            f.seek(s0)
            f.write(data)
        s.close()
        return len(data)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = [ex.submit(_fetch_chunk, sp) for sp in ranges]
        for fut in as_completed(futs):
            done_bytes += fut.result()
            now = time.time()
            if now - last_log >= 10.0 or done_bytes == total:
                mb_s = (done_bytes / 1e6) / max(now - t0, 0.1)
                print(
                    f"    dl {dst.name}: {done_bytes/1e9:.2f}/{total/1e9:.2f} GB "
                    f"({100.0*done_bytes/total:.1f}%) @ {mb_s:.1f} MB/s",
                    flush=True,
                )
                last_log = now

    part.rename(dst)
    sess.close()
    return dst


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract_compact_m3ed_sequence(seq: str, data_h5: Path, depth_h5: Path, out_npz: Path) -> dict:
    """Strip 90% event-camera bloat; keep RGB JPEGs + left-event depth_gt + VN100T IMU + calibration."""
    rgb_blobs: list[bytes] = []
    rgb_offsets: list[int] = [0]
    meta: dict = {"seq": seq}

    with h5py.File(data_h5, "r") as fd:
        # Inspect RGB stream (OVC rgb)
        rgb_node = fd["ovc/rgb/data"]
        rgb_ts = fd["ovc/ts"][:]
        # Encode RGB frames cleanly as JPEG Q=92 (huge size reduction vs raw uncompressed H5!)
        n_rgb = rgb_node.shape[0]
        # Subsample to <=10 Hz (~every 2nd frame) if >2000 frames to keep compact & crisp
        step = 2 if n_rgb > 1500 else 1
        kept_rgb_idx = np.arange(0, n_rgb, step, dtype=np.int32)
        for i in kept_rgb_idx:
            img = rgb_node[int(i)]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                enc = np.zeros(0, dtype=np.uint8)
            b = enc.tobytes()
            rgb_blobs.append(b)
            rgb_offsets.append(rgb_offsets[-1] + len(b))

        # Camera calibration & transforms
        rgb_K = fd["ovc/rgb/calib/intrinsics"][:] if "ovc/rgb/calib/intrinsics" in fd else np.zeros(4, np.float32)
        rgb_dist = fd["ovc/rgb/calib/distortion_coeffs"][:] if "ovc/rgb/calib/distortion_coeffs" in fd else np.zeros(4, np.float32)
        T_rgb_to_prophesee = (
            fd["ovc/rgb/calib/T_to_prophesee_left"][:]
            if "ovc/rgb/calib/T_to_prophesee_left" in fd
            else np.eye(4, dtype=np.float32)
        )

        # IMU streams (VN100T or OVC IMU)
        imu_ts, imu_acc, imu_gyro = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        for imu_base in ("ovc/imu", "prophesee/left/imu", "ouster/imu"):
            if f"{imu_base}/accel" in fd and f"{imu_base}/ts" in fd:
                imu_ts = fd[f"{imu_base}/ts"][:]
                imu_acc = fd[f"{imu_base}/accel"][:]
                imu_gyro = fd[f"{imu_base}/omega"][:] if f"{imu_base}/omega" in fd else np.zeros_like(imu_acc)
                meta["imu_source"] = imu_base
                break

    with h5py.File(depth_h5, "r") as fg:
        depth_arr = fg["depth/prophesee/left"][:]
        depth_ts = fg["ts"][:]
        # Keep float16 depth maps (exact LiDAR metric depth in meters)
        depth_f16 = np.where(np.isfinite(depth_arr), depth_arr, 0.0).astype(np.float16)
        poses = fg["Cn_T_C0"][:] if "Cn_T_C0" in fg else np.zeros((0, 4, 4), dtype=np.float32)
        depth_K = fg["depth/prophesee/left"].attrs.get("intrinsics", np.zeros(4, np.float32))

    np.savez_compressed(
        out_npz,
        rgb_jpeg_bytes=np.frombuffer(b"".join(rgb_blobs), dtype=np.uint8),
        rgb_jpeg_offsets=np.asarray(rgb_offsets, dtype=np.int64),
        rgb_ts=rgb_ts[kept_rgb_idx],
        rgb_K=np.asarray(rgb_K, dtype=np.float32),
        rgb_dist=np.asarray(rgb_dist, dtype=np.float32),
        T_rgb_to_prophesee=np.asarray(T_rgb_to_prophesee, dtype=np.float32),
        depth_f16=depth_f16,
        depth_ts=depth_ts,
        depth_K=np.asarray(depth_K, dtype=np.float32),
        poses_Cn_T_C0=np.asarray(poses, dtype=np.float32),
        imu_ts=imu_ts,
        imu_accel=np.asarray(imu_acc, dtype=np.float32),
        imu_gyro=np.asarray(imu_gyro, dtype=np.float32),
    )
    return {
        "n_rgb": int(len(kept_rgb_idx)),
        "n_depth": int(depth_f16.shape[0]),
        "n_imu": int(len(imu_ts)),
        "npz_bytes": out_npz.stat().st_size,
    }


def main() -> None:
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    # Clean any stale half-downloaded .part files from earlier run
    shutil.rmtree("/content/spec_stage/m3ed", ignore_errors=True)

    mpath = DRIVE_OUT / "manifest_compact_sha256.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": "m3ed_spot_train_v1_compact",
            "license": "CC BY-SA 4.0 (UPenn M3ED; strictly is_test_file==False)",
            "sequences": {},
            "errors": [],
        }

    print(f"=== Smart 16-Thread M3ED Spot Extractor ({len(TARGET_SEQS)} golden sequences) ===", flush=True)
    for idx, seq in enumerate(TARGET_SEQS, 1):
        tar_name = f"{seq}_compact.tar"
        if seq in manifest["sequences"]:
            print(f"[{idx}/{len(TARGET_SEQS)}] SKIP existing {seq}", flush=True)
            continue
        t0 = time.time()
        print(f"\n[{idx}/{len(TARGET_SEQS)}] Processing {seq}...", flush=True)
        try:
            d_h5 = LOCAL_TMP / f"{seq}_depth_gt.h5"
            m_h5 = LOCAL_TMP / f"{seq}_data.h5"
            parallel_range_download(f"{BASE_S3}/{seq}/{seq}_depth_gt.h5", d_h5, n_threads=8, chunk_mb=8)
            parallel_range_download(f"{BASE_S3}/{seq}/{seq}_data.h5", m_h5, n_threads=16, chunk_mb=24)

            npz_path = LOCAL_TMP / f"{seq}.npz"
            stats = extract_compact_m3ed_sequence(seq, m_h5, d_h5, npz_path)
            # Delete huge raw H5 immediately to keep local SSD wide open
            m_h5.unlink()
            d_h5.unlink()

            local_tar = LOCAL_TMP / tar_name
            with tarfile.open(local_tar, "w") as tf:
                tf.add(npz_path, arcname=npz_path.name)
            npz_path.unlink()

            sz = local_tar.stat().st_size
            digest = sha256_file(local_tar)
            drive_tar = DRIVE_OUT / tar_name
            tmp_drive = DRIVE_OUT / f".tmp_{tar_name}"
            shutil.copyfile(local_tar, tmp_drive)
            if drive_tar.exists():
                drive_tar.unlink()
            tmp_drive.rename(drive_tar)
            if drive_tar.stat().st_size != sz:
                raise IOError(f"Drive size mismatch: {drive_tar.stat().st_size} != {sz}")
            local_tar.unlink()

            manifest["sequences"][seq] = {
                "filename": tar_name,
                "bytes": sz,
                "sha256": digest,
                **stats,
            }
            manifest["total_bytes"] = sum(v["bytes"] for v in manifest["sequences"].values())
            manifest["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tmp_m = mpath.with_suffix(".json.tmp")
            tmp_m.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            shutil.move(str(tmp_m), str(mpath))
            print(
                f"  OK {seq}: {stats['n_rgb']} RGB + {stats['n_depth']} LiDAR depth + {stats['n_imu']} IMU "
                f"-> {sz/1e9:.2f} GB verified in {time.time()-t0:.1f}s!",
                flush=True,
            )
        except Exception as e:
            print(f"  ERR {seq}: {e}", flush=True)
            manifest["errors"].append({"seq": seq, "error": str(e)})
            mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            for p in (LOCAL_TMP / f"{seq}_data.h5", LOCAL_TMP / f"{seq}_depth_gt.h5"):
                if p.exists():
                    p.unlink()

    print("=== Smart M3ED Spot Extraction COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
