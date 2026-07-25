#!/usr/bin/env python3
"""Acquire only frozen ADT main-groundtruth archives with SHA-1 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


LINKS_URL = "https://explorer.projectaria.com/data/adt/download_links"
MAX_TOTAL_BYTES = 800 * 1024 * 1024


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"metadata request failed: {response.status}")
        return json.load(response)


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire(url: str, target: Path, expected_size: int, expected_sha1: str) -> str:
    if target.is_file():
        if target.stat().st_size != expected_size or _sha1(target) != expected_sha1:
            raise AssertionError(f"existing target identity mismatch: {target}")
        return "reused_verified"
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        raise AssertionError(f"partial file requires explicit review: {part}")

    digest = hashlib.sha1()
    written = 0
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-ADT-R0/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("xb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > expected_size:
                raise AssertionError(f"download exceeded manifest size: {target.name}")
            digest.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    if written != expected_size:
        raise AssertionError(
            f"download size mismatch for {target.name}: {written} != {expected_size}"
        )
    if digest.hexdigest() != expected_sha1:
        raise AssertionError(f"download SHA-1 mismatch: {target.name}")
    part.replace(target)
    return "downloaded_verified"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    freeze_bytes = args.freeze.read_bytes()
    freeze = json.loads(freeze_bytes)
    assert freeze["terminal"] == "ADT_GROUNDTRUTH_PRESCREEN_SELECTION_FROZEN"
    assert freeze["rgb_payload_requested"] is False
    assert freeze["groundtruth_payload_requested"] is False
    assert freeze["selected_total_groundtruth_bytes"] <= MAX_TOTAL_BYTES

    live = _get_json(LINKS_URL)["sequences"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    for selection in freeze["selections"]:
        sequence_id = selection["sequence_id"]
        expected = selection["main_groundtruth"]
        current = live[sequence_id]["main_groundtruth"]
        for field in ("filename", "sha1sum", "file_size_bytes"):
            if current[field] != expected[field]:
                raise AssertionError(
                    f"live ADT manifest drift: {sequence_id} {field}"
                )
        filename = current["filename"]
        if Path(filename).name != filename:
            raise AssertionError(f"unsafe manifest filename: {filename}")
        target = args.output_dir / filename
        action = _acquire(
            current["download_url"],
            target,
            current["file_size_bytes"],
            current["sha1sum"],
        )
        members.append(
            {
                "sequence_id": sequence_id,
                "proxy_stratum": selection["proxy_stratum"],
                "path": target.as_posix(),
                "filename": filename,
                "bytes": current["file_size_bytes"],
                "sha1": current["sha1sum"],
                "action": action,
            }
        )

    receipt = {
        "schema_version": "adt_groundtruth_prescreen_acquisition_r0",
        "source_id": "ARIA_DIGITAL_TWIN",
        "freeze_path": args.freeze.as_posix(),
        "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "member_count": len(members),
        "total_bytes": sum(member["bytes"] for member in members),
        "members": members,
        "rgb_or_vrs_member_count": 0,
        "candidate_signal_computed": False,
        "role_split_frozen": False,
        "cell_truth_frozen": False,
        "terminal": "ADT_GROUNDTRUTH_PRESCREEN_PAYLOAD_ACQUIRED",
        "authority": {
            "may_inventory_groundtruth_archive_members": True,
            "may_decode_rgb_or_run_signal": False,
            "may_freeze_r0_admission": False,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": receipt["terminal"],
                "member_count": receipt["member_count"],
                "total_bytes": receipt["total_bytes"],
                "rgb_or_vrs_member_count": receipt["rgb_or_vrs_member_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
