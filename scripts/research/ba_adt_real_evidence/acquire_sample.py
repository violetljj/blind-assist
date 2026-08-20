#!/usr/bin/env python3
"""Acquire the bounded ADT sample RGB preview and GT archive.

The RGB preview is the only future perception input. Ground truth is stored as
an evaluation/mining sidecar and must never be exposed to an RGB estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


MANIFEST_URL = "https://explorer.projectaria.com/data/adt/download_links"
SAMPLE_ID = "Apartment_release_golden_skeleton_seq100_10s_sample_M1292"
MEMBERS = ("video_main_rgb", "main_groundtruth")
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch_manifest() -> dict[str, Any]:
    request = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "BlindAssist-ADT0/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"ADT manifest request failed: HTTP {response.status}")
        return json.load(response)


def acquire(member: dict[str, Any], output_dir: Path) -> tuple[Path, str]:
    filename = str(member["filename"])
    if Path(filename).name != filename:
        raise ValueError(f"unsafe ADT filename: {filename}")
    expected_bytes = int(member["file_size_bytes"])
    expected_sha1 = str(member["sha1sum"])
    target = output_dir / filename
    if target.is_file():
        if target.stat().st_size != expected_bytes or digest(target, "sha1") != expected_sha1:
            raise RuntimeError(f"existing ADT member identity mismatch: {target}")
        return target, "reused_verified"

    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"partial download requires review: {partial}")
    request = urllib.request.Request(
        str(member["download_url"]), headers={"User-Agent": "BlindAssist-ADT0/1.0"}
    )
    written = 0
    sha1 = hashlib.sha1()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > expected_bytes:
                    raise RuntimeError(f"download exceeded manifest size: {filename}")
                stream.write(chunk)
                sha1.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if written != expected_bytes or sha1.hexdigest() != expected_sha1:
            raise RuntimeError(f"download identity mismatch: {filename}")
        partial.replace(target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target, "downloaded_verified"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    manifest = fetch_manifest()
    sequence = manifest["sequences"][SAMPLE_ID]
    selected = [sequence[name] for name in MEMBERS]
    total_bytes = sum(int(member["file_size_bytes"]) for member in selected)
    if total_bytes > MAX_TOTAL_BYTES:
        raise RuntimeError(f"bounded sample grew beyond {MAX_TOTAL_BYTES} bytes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, member in zip(MEMBERS, selected, strict=True):
        path, action = acquire(member, args.output_dir)
        rows.append(
            {
                "role": "RGB_SYSTEM_INPUT" if role == "video_main_rgb" else "GT_EVALUATOR_ONLY",
                "manifest_key": role,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha1": digest(path, "sha1"),
                "sha256": digest(path, "sha256"),
                "action": action,
            }
        )

    receipt = {
        "schema_version": "ba_adt_sample_acquisition_v1",
        "route": "BA-ADT-REAL-EVIDENCE",
        "stage": "ADT-0",
        "source": "Aria Digital Twin",
        "manifest_url": MANIFEST_URL,
        "sequence_id": SAMPLE_ID,
        "member_count": len(rows),
        "total_bytes": total_bytes,
        "members": rows,
        "firewall": {
            "rgb_system_input": [rows[0]["filename"]],
            "gt_evaluator_only": [rows[1]["filename"]],
            "gt_may_enter_rgb_estimator": False,
        },
        "claim_ceiling": "sample_acquisition_and_gt_only_episode_mining",
        "terminal": "ADT0_SAMPLE_ACQUIRED",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VALID", "terminal": receipt["terminal"], "total_bytes": total_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
