#!/usr/bin/env python3
"""Fetch official SelectiveCL source and Seen checkpoint with byte/hash receipts."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import re
import tarfile
import time

import requests

COMMIT = "5422ed50ba34c2fb5f368fddf4b46cf5c9b597ed"
FILE_ID = "1cYC2PBEjhLntySyP51R46J7i8f1Cf1NT"
WEIGHT_BYTES = 712621477
BUDGET_BYTES = 2_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts, network_bytes = [], 0
    started = time.perf_counter()
    session = requests.Session()
    source_url = f"https://codeload.github.com/hynnsk/SelectiveCL/tar.gz/{COMMIT}"
    archive_path = root / "selectivecl_source.tar.gz"
    if not archive_path.exists():
        response = session.get(source_url, timeout=(15, 60))
        response.raise_for_status()
        if len(response.content) > 100_000_000:
            raise ValueError("Source archive unexpectedly exceeds 100MB")
        archive_path.write_bytes(response.content)
        network_bytes += len(response.content)
    vendor = root / "vendor" / "SelectiveCL"
    vendor.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts[1:]
            if not parts or ".." in parts:
                raise ValueError("Invalid source archive member")
            target = vendor.joinpath(*parts)
            if not target.resolve().is_relative_to(vendor.resolve()):
                raise ValueError("Source archive member outside vendor root")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.extractfile(member).read()
            target.write_bytes(data)
            files.append({"path": str(target.relative_to(vendor)), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    save(root / "source_receipt.json", {"repo": "https://github.com/hynnsk/SelectiveCL",
         "commit": COMMIT, "archive_sha256": digest(archive_path),
         "archive_bytes": archive_path.stat().st_size, "files": files})
    print(json.dumps({"stage": "SOURCE_READY", "vendor": str(vendor), "files": len(files)}), flush=True)
    model_root = root / "models"
    model_root.mkdir(exist_ok=True)
    weights = model_root / "AGD20K-Seen.pth"
    if not weights.exists() or weights.stat().st_size != WEIGHT_BYTES:
        landing = session.get("https://drive.google.com/uc", params={"id": FILE_ID, "export": "download"}, timeout=20)
        landing.raise_for_status()
        network_bytes += len(landing.content)
        form = re.search(r'<form[^>]+action="([^"]+)"', landing.text)
        if not form:
            raise ValueError("Official public Drive form unavailable")
        action = html.unescape(form.group(1))
        params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', landing.text))
        partial = weights.with_suffix(".pth.partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > WEIGHT_BYTES:
            raise ValueError("Preserved partial checkpoint exceeds expected size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = session.get(action, params=params, headers=headers, stream=True, timeout=(20, 60))
        try:
            if offset:
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Cannot resume exact public checkpoint range; preserved partial")
            elif response.status_code != 200:
                raise ValueError(f"Expected checkpoint HTTP200, got {response.status_code}")
            if "application/octet-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("Checkpoint response is not binary")
            received = offset
            with partial.open("ab" if offset else "wb") as stream:
                for part in response.iter_content(1 << 20):
                    received += len(part)
                    network_bytes += len(part)
                    if received > WEIGHT_BYTES or network_bytes > BUDGET_BYTES:
                        raise RuntimeError("Checkpoint or task download-byte limit exceeded")
                    stream.write(part)
            if received != WEIGHT_BYTES:
                raise ValueError(f"Incomplete checkpoint {received}/{WEIGHT_BYTES}")
            partial.replace(weights)
        finally:
            response.close()
    receipts.append({"official_source": "https://github.com/hynnsk/SelectiveCL",
                     "public_url": f"https://drive.google.com/file/d/{FILE_ID}/view",
                     "path": str(weights), "bytes": weights.stat().st_size, "sha256": digest(weights)})
    result = {"status": "SUCCESS", "source_commit": COMMIT, "models": receipts,
              "new_network_body_bytes": network_bytes, "limit_bytes": BUDGET_BYTES,
              "elapsed_s": time.perf_counter() - started}
    save(root / "model_receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
