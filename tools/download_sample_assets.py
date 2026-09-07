"""Download bounded, hash-verified Poly Haven assets for Willow sample materials."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts.local/unreal/sample-materials-v2"
ASSETS = {"concrete_pavement": "4k", "concrete_wall_007": "4k", "modular_street_seating": "4k", "tree_small_02": "2k"}


def request(url):
    if urllib.parse.urlparse(url).hostname not in ("api.polyhaven.com", "dl.polyhaven.org"):
        raise ValueError("Only official Poly Haven endpoints are allowed")
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "BlindAssist-Development-AssetFetch/1.0"}), timeout=90)


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def metadata(asset):
    path = ROOT / f"{asset}-files.json"
    if not path.exists():
        with request(f"https://api.polyhaven.com/files/{asset}") as response:
            data = response.read()
        json.loads(data)
        path.write_bytes(data)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select(asset, resolution, data):
    if asset in ("concrete_pavement", "concrete_wall_007"):
        for channel in ("Diffuse", "nor_dx", "Rough", "AO"):
            choices = data[channel][resolution]
            item = min((choices[fmt] for fmt in ("jpg", "png") if fmt in choices), key=lambda value: value["size"])
            yield Path(urllib.parse.urlparse(item["url"]).path).name, item
    else:
        main = data["fbx"][resolution]["fbx"]
        yield Path(urllib.parse.urlparse(main["url"]).path).name, main
        for relative, item in main.get("include", {}).items():
            if any(marker in Path(relative).name for marker in ("_diff_", "_nor_gl_", "_rough_", "_metal_", "_alpha_")):
                yield relative, item


def download(job):
    asset, relative, item = job
    path = (ROOT / asset / relative).resolve()
    if not path.is_relative_to((ROOT / asset).resolve()):
        raise ValueError("Included path escapes asset folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    reused = False
    if path.exists():
        existing_md5, existing_sha = hashes(path)
        reused = existing_md5 == item["md5"] and path.stat().st_size == item["size"]
    if not reused:
        for attempt in range(3):
            try:
                partial = path.with_name(path.name + ".part")
                with request(item["url"]) as source, partial.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                existing_md5, existing_sha = hashes(partial)
                if existing_md5 != item["md5"] or partial.stat().st_size != item["size"]:
                    raise ValueError(f"Provider hash/size mismatch: {asset}/{relative}")
                partial.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return {"asset": asset, "relative_path": relative, "local_path": str(path), "bytes": path.stat().st_size,
            "url": item["url"], "provider_md5": item["md5"], "computed_md5": existing_md5,
            "sha256": existing_sha, "reused_matching_file": reused}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(asset, relative, item) for asset, resolution in ASSETS.items()
            for relative, item in select(asset, resolution, metadata(asset))]
    print(json.dumps({"files": len(jobs), "expected_bytes": sum(item["size"] for _, _, item in jobs)}), flush=True)
    files, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                files.append(result)
                print(json.dumps({"downloaded": result["asset"] + "/" + result["relative_path"], "bytes": result["bytes"]}), flush=True)
            except Exception as error:
                asset, relative, _ = futures[future]
                errors.append({"path": asset + "/" + relative, "error": str(error)})
    manifest = {"status": "PASS" if not errors else "FAIL", "provider": "Poly Haven", "license": "CC0",
                "license_source": "https://polyhaven.com/license", "assets": {
                    asset: {"resolution": resolution, "source": f"https://polyhaven.com/a/{asset}",
                            "metadata_source": f"https://api.polyhaven.com/files/{asset}",
                            "metadata_sha256": hashes(ROOT / f"{asset}-files.json")[1]}
                    for asset, resolution in ASSETS.items()},
                "files": sorted(files, key=lambda item: (item["asset"], item["relative_path"])), "errors": errors}
    path = ROOT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(path.resolve()), "files": len(files),
                      "bytes": sum(item["bytes"] for item in files), "errors": errors}), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
