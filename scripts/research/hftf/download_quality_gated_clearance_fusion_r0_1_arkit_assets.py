#!/usr/bin/env python3
"""Download the explicitly authorized ARKitScenes R0.1 source assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_media_preflight_protocol"
PREFLIGHT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_asset_header_preflight"
MEDIA_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_media_manifest"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def download(url: str, target: Path, expected_bytes: int) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists() or target.exists():
        raise FileExistsError(f"overwrite forbidden: {target}")
    request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-clearance-fusion-r0.1-acquisition"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
        copied = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            copied += len(block)
        stream.flush()
        os.fsync(stream.fileno())
    require(copied == expected_bytes, f"content length mismatch for {url}: {copied} != {expected_bytes}")
    digest = sha256_file(partial)
    partial.replace(target)
    return digest, copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    require(not args.output_root.exists(), f"overwrite forbidden: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(preflight.get("schema") == PREFLIGHT_SCHEMA, "preflight schema drift")
    require(preflight["protocol_sha256"] == sha256_file(args.protocol), "preflight protocol SHA mismatch")
    require(preflight["terminal"].endswith("AVAILABLE_MEDIA_UNOPENED"), "media headers are not all available")
    require(preflight["media_body_bytes_read"] is False and preflight["label_or_model_fields_read"] is False, "preflight boundary violated")
    require(preflight["available_asset_count"] == preflight["asset_count"] == 15, "asset count drift")
    free = shutil.disk_usage(args.output_root.parent).free
    require(free >= int(preflight["total_content_length_bytes"]) + 1_000_000_000, "insufficient bounded working space")
    by_key = {(row["video_id"], row["asset"]): row for row in preflight["assets"]}
    require(len(by_key) == 15, "duplicate preflight rows")
    args.output_root.mkdir(parents=True)
    files = []
    for (video_id, asset), row in sorted(by_key.items()):
        target = args.output_root / "raw" / "Validation" / str(video_id) / asset
        digest, size = download(row["url"], target, int(row["content_length_bytes"]))
        files.append({"visit_id": row["visit_id"], "video_id": video_id, "asset": asset, "url": row["url"], "bytes": size, "sha256": digest})
    manifest = {
        "schema": MEDIA_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "preflight_sha256": sha256_file(args.preflight),
        "files": files,
        "file_count": len(files),
        "labels_opened": False,
        "model_outputs_read": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "terminal": "QUALITY_GATED_CLEARANCE_FUSION_R0_1_ARKIT_MEDIA_DOWNLOADED_INTEGRITY_BOUND",
    }
    path = args.output_root / "manifest.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
