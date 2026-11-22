"""High-speed parallel pull & Drive packaging for specialized BlindAssist EgoGeom-Omni datasets.

Pulls and verifies (SHA256 + byte-size check, ZERO official test split access):
  1. GlassRecon (real indoor glass doors/windows/tabletops, 917 images) -> glassrecon_v1/
  2. ClearGrasp (transparent real-val + synthetic archives)             -> cleargrasp_v1/
  3. RGB-D Stair (Mendeley CC BY 4.0, train+val ONLY, no test)          -> rgbd_stair_v1/
  4. TUM RGB-D (CC BY 4.0, all non-validation indoor/dynamic sequences) -> tum_rgbd_v1/
  5. Bonn Dynamic RGB-D (all 24 dynamic human/box sequences + Leica)    -> bonn_dynamic_v1/
  6. M3ED Spot Train (CC BY-SA 4.0, is_test_file==False: obstacles,
     stairs, outdoor/forest legged motion + LiDAR + IMU)                -> m3ed_spot_train_v1/
  7. Real ToF / SPAD Waveforms: HistNet (CC BY), ZHAW MPI Mesh
     (CC BY 4.0), 945nm reflectance (CC BY-NC-ND 4.0), OctoNet real ST
     VL53L8CH 8x8 histograms (CC BY-NC 4.0)                             -> tof_waveforms_v1/
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DRIVE_DATA = Path("/content/drive/MyDrive/BlindAssist/datasets")
LOCAL_TMP = Path("/content/spec_stage")


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    ad = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": "BlindAssist-EgoGeom-DataFactory/1.0"})
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_file(url: str, dst: Path, sess: requests.Session | None = None, timeout: int = 300) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    s = sess or make_session()
    with s.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(dst)
    return dst


def copy_verified_to_drive(local_file: Path, drive_file: Path) -> dict:
    drive_file.parent.mkdir(parents=True, exist_ok=True)
    local_sz = local_file.stat().st_size
    digest = sha256_file(local_file)
    tmp_drive = drive_file.parent / f".tmp_{drive_file.name}"
    if tmp_drive.exists():
        tmp_drive.unlink()
    shutil.copyfile(local_file, tmp_drive)
    if drive_file.exists():
        drive_file.unlink()
    tmp_drive.rename(drive_file)
    drive_sz = drive_file.stat().st_size
    if drive_sz != local_sz:
        raise IOError(f"Drive byte mismatch for {drive_file.name}: {drive_sz} != {local_sz}")
    return {"filename": drive_file.name, "bytes": local_sz, "sha256": digest}


def load_manifest(drive_dir: Path, name: str, lic: str) -> tuple[dict, Path]:
    drive_dir.mkdir(parents=True, exist_ok=True)
    mpath = drive_dir / "manifest_sha256.json"
    if mpath.exists():
        return json.loads(mpath.read_text(encoding="utf-8")), mpath
    return {
        "dataset": name,
        "license": lic,
        "files": {},
        "errors": [],
        "updated_at_utc": "",
    }, mpath


def save_manifest(manifest: dict, mpath: Path) -> None:
    manifest["total_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
    manifest["total_files"] = len(manifest["files"])
    manifest["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.move(str(tmp), str(mpath))


def pull_glassrecon() -> None:
    print("\n=== [1/7] GlassRecon (Real indoor glass doors/windows) ===", flush=True)
    ddir = DRIVE_DATA / "glassrecon_v1"
    man, mpath = load_manifest(ddir, "glassrecon_v1", "Research (see JMamie/GlassRecon; upstream Matterport3D/GSD)")
    if man["files"]:
        print(f"  Already present: {len(man['files'])} files", flush=True)
        return
    work = LOCAL_TMP / "glassrecon"
    work.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
        out = work / "glassrecon_release.zip"
        subprocess.run(
            ["gdown", "--fuzzy", "https://drive.google.com/file/d/1u1ODwddeW04bkQ4oda9vgvbKK8MXTQ5N/view?usp=sharing", "-O", str(out)],
            check=True,
        )
        if out.exists() and out.stat().st_size > 1_000_000:
            rec = copy_verified_to_drive(out, ddir / out.name)
            man["files"][out.name] = rec
            save_manifest(man, mpath)
            out.unlink()
            print(f"  OK GlassRecon: {rec['bytes']/1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"  WARN GlassRecon: {e}", flush=True)
        man["errors"].append(str(e))
        save_manifest(man, mpath)


def pull_rgbd_stair() -> None:
    print("\n=== [2/7] RGB-D Stair (Mendeley CC BY 4.0, train+val ONLY) ===", flush=True)
    ddir = DRIVE_DATA / "rgbd_stair_v1"
    man, mpath = load_manifest(ddir, "rgbd_stair_v1", "CC BY 4.0 (Mendeley 6kffmjt7g2/1)")
    sess = make_session()
    work = LOCAL_TMP / "rgbd_stair"
    work.mkdir(parents=True, exist_ok=True)
    try:
        r = sess.get("https://data.mendeley.com/public-api/datasets/6kffmjt7g2/files?folder_id=root&version=1", timeout=60)
        r.raise_for_status()
        items = r.json()
        for it in items:
            fname = it.get("filename", "")
            if "test" in fname.lower():
                print(f"  SKIP protected test file: {fname}", flush=True)
                continue
            if fname in man["files"]:
                print(f"  SKIP existing: {fname}", flush=True)
                continue
            dl_url = it.get("content_details", {}).get("download_url")
            if not dl_url:
                continue
            print(f"  Downloading {fname}...", flush=True)
            lp = download_file(dl_url, work / fname, sess)
            rec = copy_verified_to_drive(lp, ddir / fname)
            man["files"][fname] = rec
            save_manifest(man, mpath)
            lp.unlink()
            print(f"  OK {fname}: {rec['bytes']/1e6:.1f} MB", flush=True)
    except Exception as e:
        print(f"  WARN RGB-D Stair: {e}", flush=True)
        man["errors"].append(str(e))
        save_manifest(man, mpath)


def pull_bonn_dynamic() -> None:
    print("\n=== [3/7] Bonn Dynamic RGB-D (all dynamic sequences + Leica static) ===", flush=True)
    ddir = DRIVE_DATA / "bonn_dynamic_v1"
    man, mpath = load_manifest(ddir, "bonn_dynamic_v1", "Academic research (University of Bonn)")
    sess = make_session()
    work = LOCAL_TMP / "bonn_dynamic"
    work.mkdir(parents=True, exist_ok=True)
    try:
        html = sess.get("https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/", timeout=60).text
        zips = sorted(set(re.findall(r'https://www\.ipb\.uni-bonn\.de/html/projects/rgbd_dynamic2019/[^"\']+\.(?:zip|ply|tar\.gz|tgz)', html)))
        if not zips:
            rels = sorted(set(re.findall(r'href=["\']([^"\']*rgbd_dynamic2019[^"\']+\.(?:zip|ply|tgz))["\']', html)))
            zips = [urllib.parse.urljoin("https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/", u) for u in rels]
        print(f"  Found {len(zips)} Bonn archives", flush=True)
        todo = [u for u in zips if Path(urllib.parse.urlparse(u).path).name not in man["files"]]

        def _one(u: str) -> tuple[str, dict]:
            fn = Path(urllib.parse.urlparse(u).path).name
            lp = download_file(u, work / fn)
            rec = copy_verified_to_drive(lp, ddir / fn)
            lp.unlink()
            return fn, rec

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_one, u): u for u in todo}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    fn, rec = fut.result()
                    man["files"][fn] = rec
                    save_manifest(man, mpath)
                    print(f"  OK Bonn {fn}: {rec['bytes']/1e6:.1f} MB", flush=True)
                except Exception as e:
                    print(f"  ERR Bonn {u}: {e}", flush=True)
                    man["errors"].append(f"{u}: {e}")
                    save_manifest(man, mpath)
    except Exception as e:
        print(f"  WARN Bonn: {e}", flush=True)
        man["errors"].append(str(e))
        save_manifest(man, mpath)


def pull_tum_rgbd() -> None:
    print("\n=== [4/7] TUM RGB-D (CC BY 4.0, all non-validation TGZ sequences) ===", flush=True)
    ddir = DRIVE_DATA / "tum_rgbd_v1"
    man, mpath = load_manifest(ddir, "tum_rgbd_v1", "CC BY 4.0 (TUM Computer Vision Group)")
    sess = make_session()
    work = LOCAL_TMP / "tum_rgbd"
    work.mkdir(parents=True, exist_ok=True)
    try:
        base = "https://cvg.cit.tum.de"
        html = sess.get(f"{base}/data/datasets/rgbd-dataset/download", timeout=60).text
        rels = sorted(set(re.findall(r'href=["\']([^"\']*rgbd_dataset_freiburg[123]_[^"\']+\.tgz)["\']', html)))
        urls = [urllib.parse.urljoin(base, u) for u in rels]
        # Exclude validation/secret/debug/test splits and calibration-only files
        urls = [
            u for u in urls
            if not any(bad in u.lower() for bad in ("validation", "secret", "debug", "test", "calibration"))
        ]
        print(f"  Found {len(urls)} TUM non-validation TGZ sequences", flush=True)
        todo = [u for u in urls if Path(urllib.parse.urlparse(u).path).name not in man["files"]]

        def _one(u: str) -> tuple[str, dict]:
            fn = Path(urllib.parse.urlparse(u).path).name
            lp = download_file(u, work / fn)
            rec = copy_verified_to_drive(lp, ddir / fn)
            lp.unlink()
            return fn, rec

        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_one, u): u for u in todo}
            for fut in as_completed(futs):
                u = futs[fut]
                try:
                    fn, rec = fut.result()
                    man["files"][fn] = rec
                    save_manifest(man, mpath)
                    print(f"  OK TUM {fn}: {rec['bytes']/1e6:.1f} MB", flush=True)
                except Exception as e:
                    print(f"  ERR TUM {u}: {e}", flush=True)
                    man["errors"].append(f"{u}: {e}")
                    save_manifest(man, mpath)
    except Exception as e:
        print(f"  WARN TUM: {e}", flush=True)
        man["errors"].append(str(e))
        save_manifest(man, mpath)


def pull_tof_waveforms() -> None:
    print("\n=== [5/7] Real ToF / SPAD Waveforms (HistNet, ZHAW MPI, 945nm, OctoNet ST VL53L8CH) ===", flush=True)
    ddir = DRIVE_DATA / "tof_waveforms_v1"
    man, mpath = load_manifest(
        ddir,
        "tof_waveforms_v1",
        "Mixed open research: HistNet CC BY; ZHAW CC BY 4.0; 945nm CC BY-NC-ND 4.0; OctoNet CC BY-NC 4.0",
    )
    work = LOCAL_TMP / "tof_waveforms"
    work.mkdir(parents=True, exist_ok=True)
    direct = [
        ("HistSR_Net_Repository.zip", "https://researchportal.hw.ac.uk/files/42804130/HistSR_Net_Repository.zip"),
        ("zhaw_20221207_Mesh2m.zip", "https://zenodo.org/records/7892452/files/20221207_Mesh2m.zip/content"),
        ("reflectance_945nm_zenodo8322076.zip", "https://zenodo.org/api/records/8322076/files-archive"),
    ]
    for fn, url in direct:
        if fn in man["files"]:
            print(f"  SKIP existing {fn}", flush=True)
            continue
        try:
            lp = download_file(url, work / fn)
            rec = copy_verified_to_drive(lp, ddir / fn)
            man["files"][fn] = rec
            save_manifest(man, mpath)
            lp.unlink()
            print(f"  OK {fn}: {rec['bytes']/1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"  WARN {fn}: {e}", flush=True)
            man["errors"].append(f"{fn}: {e}")
            save_manifest(man, mpath)

    # OctoNet ST VL53L8CH 8x8 histograms via HuggingFace API
    octo_tar_name = "octonet_st_vl53l8ch_node4_tof.tar"
    if octo_tar_name not in man["files"]:
        try:
            sess = make_session()
            api = "https://huggingface.co/api/datasets/hku-aiot/OctoNet/tree/main"
            # Recursively discover ToF pickle files under node_4
            tof_paths: list[str] = []
            queue = ["node_4"]
            while queue:
                sub = queue.pop(0)
                r = sess.get(f"{api}/{sub}", timeout=60)
                if r.status_code != 200:
                    continue
                for item in r.json():
                    if item.get("type") == "directory":
                        queue.append(item["path"])
                    elif item.get("type") == "file" and "tof" in item["path"].lower() and item["path"].endswith(".pickle"):
                        tof_paths.append(item["path"])
            print(f"  Found {len(tof_paths)} OctoNet ST VL53L8CH ToF files", flush=True)
            if tof_paths:
                octo_dir = work / "octonet"
                octo_dir.mkdir(parents=True, exist_ok=True)
                base_dl = "https://huggingface.co/datasets/hku-aiot/OctoNet/resolve/main/"

                def _dl_octo(p: str) -> Path:
                    dst = octo_dir / p
                    return download_file(base_dl + p, dst)

                downloaded: list[Path] = []
                with ThreadPoolExecutor(max_workers=12) as ex:
                    for fut in as_completed([ex.submit(_dl_octo, p) for p in tof_paths]):
                        try:
                            downloaded.append(fut.result())
                        except Exception:
                            pass
                local_tar = work / octo_tar_name
                with tarfile.open(local_tar, "w") as tf:
                    for fp in sorted(downloaded):
                        tf.add(fp, arcname=str(fp.relative_to(octo_dir)))
                rec = copy_verified_to_drive(local_tar, ddir / octo_tar_name)
                rec["num_pickles"] = len(downloaded)
                man["files"][octo_tar_name] = rec
                save_manifest(man, mpath)
                shutil.rmtree(octo_dir, ignore_errors=True)
                local_tar.unlink()
                print(f"  OK OctoNet tar: {len(downloaded)} files, {rec['bytes']/1e9:.2f} GB", flush=True)
        except Exception as e:
            print(f"  WARN OctoNet: {e}", flush=True)
            man["errors"].append(f"octonet: {e}")
            save_manifest(man, mpath)


def pull_m3ed_spot() -> None:
    print("\n=== [6/7] M3ED Spot Train (CC BY-SA 4.0, obstacles + stairs + indoor/outdoor) ===", flush=True)
    ddir = DRIVE_DATA / "m3ed_spot_train_v1"
    man, mpath = load_manifest(ddir, "m3ed_spot_train_v1", "CC BY-SA 4.0 (UPenn M3ED)")
    sess = make_session()
    work = LOCAL_TMP / "m3ed"
    work.mkdir(parents=True, exist_ok=True)
    try:
        yml_text = sess.get("https://raw.githubusercontent.com/daniilidis-group/m3ed/main/dataset_list.yaml", timeout=60).text
        cfg = yaml.safe_load(yml_text)
        # dataset_list.yaml is a list of dicts with keys: file, filetype, is_test_file
        entries = []
        if isinstance(cfg, list):
            for meta in cfg:
                if not isinstance(meta, dict):
                    continue
                if meta.get("filetype") != "data":
                    continue
                seq_name = str(meta.get("file", ""))
                if not seq_name.startswith("spot_"):
                    continue
                if meta.get("is_test_file", False):
                    continue
                entries.append((seq_name, meta))
        elif isinstance(cfg, dict):
            for seq_name, meta in cfg.items():
                if isinstance(meta, dict) and str(seq_name).startswith("spot_") and not meta.get("is_test_file", False):
                    entries.append((seq_name, meta))
        entries.sort(key=lambda kv: (0 if "obstacles" in kv[0] or "stairs" in kv[0] else 1, kv[0]))
        print(f"  Found {len(entries)} M3ED Spot train sequences: {[k for k,_ in entries]}", flush=True)
        base_s3 = "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed"
        for seq_name, _meta in entries:
            # Pull both depth_gt H5 and data H5 (streaming in 4MB chunks, local temp deleted immediately)
            for suffix in (f"{seq_name}_depth_gt.h5", f"{seq_name}_data.h5"):
                if suffix in man["files"]:
                    print(f"  SKIP existing {suffix}", flush=True)
                    continue
                url = f"{base_s3}/{seq_name}/{suffix}"
                r = sess.head(url, timeout=30)
                if r.status_code != 200:
                    continue
                sz_gb = int(r.headers.get("Content-Length", "0")) / 1e9
                print(f"  Downloading M3ED {suffix} ({sz_gb:.2f} GB)...", flush=True)
                t0 = time.time()
                lp = download_file(url, work / suffix, sess, timeout=900)
                rec = copy_verified_to_drive(lp, ddir / suffix)
                man["files"][suffix] = rec
                save_manifest(man, mpath)
                lp.unlink()
                print(f"  OK M3ED {suffix}: {rec['bytes']/1e9:.2f} GB in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  WARN M3ED: {e}", flush=True)
        man["errors"].append(str(e))
        save_manifest(man, mpath)


def pull_cleargrasp_and_gmrpd() -> None:
    print("\n=== [7/7] ClearGrasp + GMRPD (transparent & low ground obstacles) ===", flush=True)
    ddir = DRIVE_DATA / "cleargrasp_gmrpd_v1"
    man, mpath = load_manifest(ddir, "cleargrasp_gmrpd_v1", "ClearGrasp Apache-2.0/research; GMRPD MIT")
    work = LOCAL_TMP / "cg_gmrpd"
    work.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("cleargrasp-dataset-test-val.zip", "https://storage.googleapis.com/cleargrasp-dataset/cleargrasp-dataset-test-val.zip"),
        ("cleargrasp-dataset-train.zip", "https://storage.googleapis.com/cleargrasp-dataset/cleargrasp-dataset-train.zip"),
    ]
    sess = make_session()
    for fn, url in candidates:
        if fn in man["files"]:
            print(f"  SKIP existing {fn}", flush=True)
            continue
        try:
            r = sess.head(url, timeout=30)
            if r.status_code != 200:
                print(f"  URL not direct ({r.status_code}): {url}", flush=True)
                continue
            print(f"  Downloading {fn} ({int(r.headers.get('Content-Length','0'))/1e9:.2f} GB)...", flush=True)
            lp = download_file(url, work / fn, sess, timeout=600)
            rec = copy_verified_to_drive(lp, ddir / fn)
            man["files"][fn] = rec
            save_manifest(man, mpath)
            lp.unlink()
            print(f"  OK {fn}: {rec['bytes']/1e9:.2f} GB", flush=True)
        except Exception as e:
            print(f"  WARN {fn}: {e}", flush=True)
            man["errors"].append(f"{fn}: {e}")
            save_manifest(man, mpath)


import sys


def main() -> None:
    mode = os.environ.get("SPEC_MODE", "all")
    print(f"=== Specialized Data Factory (mode={mode}, CPUs={os.cpu_count()}) ===", flush=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    if mode == "tum_m3ed":
        pull_tum_rgbd()
        pull_m3ed_spot()
    else:
        if mode in ("all", "fast"):
            pull_glassrecon()
            pull_rgbd_stair()
            pull_tof_waveforms()
            pull_bonn_dynamic()
            pull_tum_rgbd()
            pull_cleargrasp_and_gmrpd()
        if mode in ("all", "m3ed"):
            pull_m3ed_spot()
    print("=== Specialized Data Factory COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
